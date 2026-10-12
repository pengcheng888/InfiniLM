# 查看该文档/workspace_codex/dynamo仓库学习记录.md。

> 查看该文档/workspace_codex/dynamo仓库学习记录.md。

<details><summary>9 previous messages</summary>

> 我已能打开这份学习记录，内容是此前围绕 Dynamo 仓库的问答、翻译和架构分析记录。刚才输出被截断了，我先补齐完整行数和后半部分，再给你一份准确摘要。
>
> 这份记录最后停在一个未完成的请求：把已验证的 Frontend/3 个 vLLM worker 启动与测试命令整理成新的 Markdown 复现文档。我会按记录里的实际端口、代理绕过和验证步骤创建文档，不改现有学习记录。
>
> 我核对到实际验证使用的是 `18000/18081-18083`，且 Frontend 必须绕过本机代理。现在我会新建一份独立复现指南，包含预检查、后台启动、健康检查、拓扑验证、推理测试和按 PID 停止，避免把 unrelated 的 `8000` 服务误杀。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./Dynamo-Qwen3-三副本聚合式推理复现指南.md` (+354 -0)
>
> </details>
>
> 文档已创建。接下来我做两类校验：确认 Markdown 代码围栏和文件内容完整；同时查询当前 `18000` 实例的 readiness 响应，确保文档里的拓扑判断字段与实际 API 输出一致。
>
> 发现一个文档格式错误：我在写入时把所有代码块结束标记误写成了 ```` ```+ ````，会导致 Markdown 后半部分解析错乱。当前 `18000` 服务已经停止，所以无法在线核对接口响应；我先修正围栏，再把 readiness 判断写成对实际返回更宽容的说明。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./Dynamo-Qwen3-三副本聚合式推理复现指南.md` (+4 -4)
>
> </details>
>
> 抽查确认问题仍存在：刚才的正则把 `+` 当成了量词，没有替换字面量加号。我改用精确匹配行内容修正，然后重新检查。
>
> 字节检查显示实际异常行是两个反引号加一个 `+`，不是三个反引号；前两次替换模式都写错了。现在按这个精确字节形态修正。
</details>

已查看 [dynamo仓库学习记录.md](./dynamo仓库学习记录.md)。这份记录共 667 行，最后一条未完成请求是把已验证的 3 副本 Dynamo/vLLM 启动与测试步骤整理成 Markdown。

我已按记录中的实际参数创建复现文档：

[Dynamo-Qwen3-三副本聚合式推理复现指南.md](./Dynamo-Qwen3-三副本聚合式推理复现指南.md)

内容包括：

- GPU/端口/模型/依赖预检查
- Frontend 与 GPU 0-2 三个 worker 的后台启动命令
- `NO_PROXY` 代理绕过配置
- 健康检查、拓扑检查、GPU 进程检查
- 仓库自带冒烟测试与 OpenAI API 调用
- 6 请求验证 round-robin 的步骤
- 日志排查与按 PID 安全停止服务

已校验 Markdown 代码围栏数量为 50，结构完整。当前 `127.0.0.1:18000` 服务已不再运行，复现时可直接按文档重新启动。

> 分析下 Qwen3-0.6B 三副本聚合式推理服务过程中整体推理流程、函数调用过程、数据流向、模块关系等等。

<details><summary>8 previous messages</summary>

> 我会把这条链路拆成三层来核对源码：OpenAI 请求进入 Frontend、Dynamo runtime 的组件发现与路由、vLLM worker 的 prefill/decode 执行。先读入口函数和消息类型，避免只按配置推断流程。
>
> <details><summary>Explored 3 searches</summary>
>
> - Searched for `chat/completions|ChatCompletion|openai` in `frontend`
> - Searched for `class .*Router|route\(|round_robin|RoundRobin|worker` in `frontend`
> - Searched for `module.*vllm|class .*Vllm|generate\(|prefill|decode` in `vllm`
>
> </details>
>
> 初步看，这个部署的关键路径在 `frontend/main.py`、`frontend/vllm_processor.py` 和 `vllm/worker_factory.py`。我接下来只读这些文件的相关区间，确认 HTTP 回调如何进入 Python、`routed_engine.generate()` 如何选 worker，以及 aggregated worker 为什么在拓扑里显示为 decode。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./components/src/dynamo/frontend/main.py`
> - Read `./components/src/dynamo/frontend/vllm_processor.py`
> - Read `./components/src/dynamo/vllm/worker_factory.py`
>
> </details>
>
> 源码显示 worker 工厂里 `AGGREGATED` 与 `DECODE` 共用 `_run_decode_worker`，区别在于 `worker_type=WorkerType.Aggregated` 且没有对 decode peer 的依赖；这解释了拓扑命名。还需要补齐两个关键点：Python `RoutedEngine` 到 Rust transport 的边界，以及 vLLM handler 内部从请求对象到 `AsyncLLM.generate` 的调用。
>
> <details><summary>Explored 20 files, 16 searches</summary>
>
> - Searched for `^(async )?def run_input|^async def run_input|class RoutedEngine|def generate\(|async def generate\(` in `.`
> - Searched for `class .*Handler|def generate\(` in `vllm`
> - Searched for `class .*Router|struct .*Router|enum .*Router|round_robin|RoundRobin` in `lib`
> - Searched for `class RoutedEngine|RoutedEngine` in `python`
> - Read `./components/src/dynamo/vllm/handlers.py`
> - Searched for `def run_input|run_input\(` in `python`
> - Read `./components/src/dynamo/vllm/handlers.py`
> - Searched for `def _generate_token_mode|def _generate_text_mode|async def _generate` in `handlers.py`
> - Read `./lib/bindings/python/rust/llm/routed_engine.rs`
> - Read `./lib/llm/src/entrypoint/input.rs`
> - Read `./components/src/dynamo/vllm/handlers.py`
> - Read `./lib/bindings/python/rust/llm/entrypoint.rs`
> - Searched for `pub struct PrefillRoutedEngine|impl PrefillRoutedEngine|struct HttpFrontend|chat_engine_factory|ChatEngineFactory` in `entrypoint`
> - Searched for `PrefillRoutedEngine` in `llm`
> - Read `./lib/llm/src/entrypoint/input/http.rs`
> - Read `./lib/llm/src/discovery/watcher.rs`
> - Read `./lib/llm/src/entrypoint.rs`
> - Searched for `build_preprocessed_pipeline|build_preprocessed_routing_with_selector|PushRouter::from_client|RouterMode::RoundRobin` in `common.rs`
> - Searched for `register_model\(|ModelDeploymentCard|list_and_watch` in `vllm`
> - Read `./lib/llm/src/entrypoint/input/common.rs`
> - Read `./lib/runtime/src/pipeline/network/egress/push_router.rs`
> - Read `./components/src/dynamo/vllm/main.py`
> - Searched for `def register_model|async def register_model` in `dynamo`
> - Searched for `async def register_model\(|def register_model\(` in `.`
> - Searched for `async def chat_engine_factory|chat_engine_factory =|VllmProcessor\(` in `vllm_processor.py`
> - Searched for `round_robin_prepared|async fn round_robin\(` in `push_router.rs`
> - Read `./components/src/dynamo/frontend/vllm_processor.py`
> - Read `./lib/runtime/src/pipeline/network/egress/push_router.rs`
> - Searched for `register_model` in `python`
> - Searched for `def generator|async def generator` in `vllm_processor.py`
> - Read `./lib/llm/src/local_model.rs`
> - Read `./lib/bindings/python/rust/lib.rs`
> - Read `./components/src/dynamo/frontend/vllm_processor.py`
> - Read `./components/src/dynamo/frontend/vllm_processor.py`
> - Read `./components/src/dynamo/frontend/vllm_processor.py`
> - Read `./components/src/dynamo/frontend/vllm_processor.py`
>
> </details>
>
> 请求面主链路已经确定：HTTP 层是 Rust，chat 预处理和响应后处理是 Python，路由和 TCP 传输回到 Rust，worker 端再由 Python handler 调 vLLM `AsyncLLM`。最后我再核对 endpoint server 的 Python/Rust 调用桥和 file discovery 的实例维护，这样模块关系图能覆盖控制面和数据面。
>
> <details><summary>Explored 9 files, 12 searches</summary>
>
> - Searched for `serve_endpoint\(` in `rust`
> - Searched for `FileDiscovery|file discovery|DYN_FILE_KV|discovery-backend.*file|FileKv` in `runtime`
> - Read `./lib/runtime/src/component/client.rs`
> - Searched for `fn serve_endpoint|serve_endpoint` in `python`
> - Read `./lib/runtime/src/storage/kv.rs`
> - Searched for `File` in `discovery`
> - Searched for `AsyncEngineCallback|serve_endpoint` in `engine.rs`
> - Read `./lib/bindings/python/rust/lib.rs`
> - Read `./lib/bindings/python/rust/lib.rs`
> - Searched for `chat_completions|ChatCompletions` in `http`
> - Searched for `round_robin_prepared|select_untracked_worker|RoutePicker` in `push_router.rs`
> - Searched for `def worker\(|async def worker\(|WorkerFactory|AsyncLLM` in `main.py`
> - Read `./components/src/dynamo/vllm/main.py`
> - Read `./lib/runtime/src/pipeline/network/egress/push_router.rs`
> - Read `./lib/runtime/src/routing_policy/picker.rs`
> - Read `./lib/llm/src/http/service/openai.rs`
> - Searched for `disaggregation_mode|DisaggregationMode.AGGREGATED` in `config.py`
> - Searched for `AGGREGATED|DECODE` in `worker_factory.py`
> - Searched for `request_plane|RequestPlane|TCP|tcp` in `distributed.rs`
> - Searched for `def generate_tokens|async def generate_tokens` in `handlers.py`
> - Read `./components/src/dynamo/vllm/handlers.py`
>
> </details>
</details>

**总体结论**

这次部署是“3 个完整 vLLM 推理副本 + 1 个 Dynamo Frontend”的 replicated aggregated 架构。每个 GPU 上有一个独立 vLLM EngineCore、独立 KV cache、独立 scheduler；一个请求只会被 round-robin 路由到其中一个 worker，并在该 worker 内完成 prompt prefill 和后续 decode。Frontend 不执行模型 forward，只负责 OpenAI 协议、chat 模板、tokenize/detokenize、流式响应组装和路由分发。

**启动与控制面**

启动时先形成控制面：

```text
Frontend process
  main() -> async_async_main()
    -> DistributedRuntime(file discovery, tcp request plane)
    -> make_engine()
    -> run_input(http)
    -> HttpFrontend
    -> list_and_watch(AllModels)
    -> ModelWatcher
    -> build_preprocessed_routing()
    -> chat_engine_factory()
    -> VllmProcessor

3 个 worker process
  python -m dynamo.vllm
    -> worker()
    -> create_runtime()
    -> setup_vllm_engine()
    -> AsyncLLM.from_vllm_config()
    -> WorkerFactory.create()
    -> _create_decode_worker()
    -> Endpoint.serve_endpoint(DecodeWorkerHandler.generate)
    -> register_vllm_model()
    -> discovery.register(ModelDeploymentCard)
```

关键点如下：

- Frontend 入口在 [main.py](./dynamo/components/src/dynamo/frontend/main.py:394)，它创建 `DistributedRuntime`，指定 `file` discovery 和 `tcp` request plane。
- HTTP 入口由 Rust `run_input()` 启动，见 [input.rs](./dynamo/lib/llm/src/entrypoint/input.rs:99)。
- worker 默认 `DisaggregationMode.AGGREGATED`；`AGGREGATED` 和 `DECODE` 共用 `_create_decode_worker()` 路径，但 `AGGREGATED` 会注册为 `WorkerType.Aggregated`，见 [worker_factory.py](./dynamo/components/src/dynamo/vllm/worker_factory.py:1328)。
- 每个 worker 通过 `register_model()` 发布 Model Deployment Card，最终写入 discovery backend，见 [main.py](./dynamo/components/src/dynamo/vllm/main.py:937) 和 [local_model.rs](./dynamo/lib/llm/src/local_model.rs:465)。
- `file` backend 实际由 `DYN_FILE_KV` 指定目录，见 [kv.rs](./dynamo/lib/runtime/src/storage/kv.rs:160)。因此 Frontend 和 3 个 worker 必须使用同一个 `DYN_FILE_KV`。
- Frontend 通过 `list_and_watch()` 监听模型和 worker 变化，见 [http.rs](./dynamo/lib/llm/src/entrypoint/input/http.rs:369)。
- `--router-min-initial-workers 3` 在构建路由前等待至少 3 个可用 worker，见 [common.rs](./dynamo/lib/llm/src/entrypoint/input/common.rs:266)。

控制面只传注册信息、模型元数据、endpoint 地址和健康/负载元数据；prompt、token、生成结果都走数据面，不经过 file discovery。

**请求数据面**

一次 `/v1/chat/completions` 请求的完整路径是：

```text
curl / OpenAI client
  -> Frontend Rust Axum HTTP server
  -> ModelManager 选择 Qwen3-0.6B 对应 engine
  -> PythonAsyncEngine
  -> VllmProcessor.generator()
  -> chat template + tokenize + SamplingParams
  -> PreprocessedRequest(token_ids, sampling, stop, routing)
  -> Rust RoutedEngine
  -> preprocessed routing pipeline
  -> PushRouter(round-robin)
  -> TCP request plane
  -> GPU worker 的 Python ingress
  -> DecodeWorkerHandler.generate()
  -> BaseWorkerHandler.generate_tokens()
  -> vLLM AsyncLLM.generate()
  -> EngineCore scheduler + GPU forward
  -> token delta stream
  -> 原路返回 Frontend
  -> vLLM OutputProcessor + StreamingPostProcessor
  -> OpenAI SSE 或聚合后的非流式 JSON
  -> client
```

分层细节：

1. HTTP 解析与模型选择
   `/v1/chat/completions` 路由注册在 [openai.rs](./dynamo/lib/llm/src/http/service/openai.rs:4057)。Rust handler 解析请求、检查 readiness，然后从 `ModelManager` 取 chat engine，并在 [openai.rs](./dynamo/lib/llm/src/http/service/openai.rs:2968) 调用 `engine.generate(request)`。

2. Python 预处理
   Rust engine 实际是 Python `VllmProcessor` 的包装。`VllmProcessor.generator()` 在 [vllm_processor.py](./dynamo/components/src/dynamo/frontend/vllm_processor.py:567) 接收 OpenAI 请求。

   它完成：
   - 校验 messages、tools、reasoning 参数
   - 应用 Qwen chat template
   - tokenize 成 `prompt_token_ids`
   - 生成 vLLM `SamplingParams`
   - 构造 Dynamo `PreprocessedRequest`

   核心转换在 [vllm_processor.py](./dynamo/components/src/dynamo/frontend/vllm_processor.py:613) 和 [vllm_processor.py](./dynamo/components/src/dynamo/frontend/vllm_processor.py:713)。发给 worker 的主体不是原始文本，而是类似：

   ```json
   {
     "model": "Qwen3-0.6B",
     "token_ids": [...],
     "stop_conditions": {...},
     "sampling_options": {...},
     "output_options": {...},
     "eos_token_ids": [...],
     "routing": {...}
   }
   ```

3. Python/Rust 路由边界
   `VllmProcessor` 调用 `self.routed_engine.generate()`，见 [vllm_processor.py](./dynamo/components/src/dynamo/frontend/vllm_processor.py:917)。
   PyO3 导出的 `RoutedEngine.generate()` 在 [routed_engine.rs](./dynamo/lib/bindings/python/rust/llm/routed_engine.rs:31)：
   - 把 Python dict 反序列化为 Rust `PreprocessedRequest`
   - 继承 request id、trace、取消信号
   - 调用 Rust `PrefillRoutedEngine`
   - 把 Rust 输出重新 pythonize 回 Python async stream

4. Round-robin 选 worker
   Rust 先构建 `PushRouter`，见 [common.rs](./dynamo/lib/llm/src/entrypoint/input/common.rs:290)。
   round-robin 实现在 [push_router.rs](./dynamo/lib/runtime/src/pipeline/network/egress/push_router.rs:899)：
   - 从当前 routable/free worker 列表取候选
   - `RoutePicker` 用 cursor 递增并按候选数取模
   - 选定 `instance_id`
   - 通过 request plane 发送到该 instance

   cursor 逻辑在 [picker.rs](./dynamo/lib/runtime/src/routing_policy/picker.rs:120)。之前 6 个串行请求得到 `2、2、2` 的分布，就是这个机制的实际验证。候选列表顺序不必然等于 GPU 0/1/2 的数值顺序，但三个 worker 稳定时会均匀轮转。

5. TCP 数据面
   `--request-plane tcp` 对应 raw TCP + msgpack 的请求面，定义在 [distributed.rs](./dynamo/lib/runtime/src/distributed.rs:797)。
   worker 的 `DYN_SYSTEM_PORT=18081/18082/18083` 就是三个 worker 数据面监听端口的区分。请求不会经过 HTTP proxy；这也是当时必须设置 `NO_PROXY` 的原因，否则 Frontend 拉取 worker 元数据或连接本机 endpoint 时可能被代理拦截。

6. Worker 端接收
   worker 启动时调用 `Endpoint.serve_endpoint(handler.generate)`。Python handler 被包装成 Rust ingress，见 [lib.rs](./dynamo/lib/bindings/python/rust/lib.rs:1431)。

   对 aggregated/decode worker，实际执行类是 `DecodeWorkerHandler`，入口在 [handlers.py](./dynamo/components/src/dynamo/vllm/handlers.py:3378)。这里名字里的 `Decode` 不表示“只做 decode”；它只是复用 decode handler 实现，`AGGREGATED` 模式会在同一个 engine 里执行完整生成。

7. vLLM 执行
   `DecodeWorkerHandler.generate()` 选择 token-in/token-out 模式，进入 `_generate_token_mode()`，见 [handlers.py](./dynamo/components/src/dynamo/vllm/handlers.py:3510)。
   它把 `token_ids` 组装成 vLLM prompt，然后进入 `generate_tokens()`。
   `generate_tokens()` 最终调用 `self.engine_client.generate()`，也就是 vLLM `AsyncLLM.generate()`，见 [handlers.py](./dynamo/components/src/dynamo/vllm/handlers.py:3199)。

8. GPU 内部生命周期
   每个 worker 的 `AsyncLLM` 后面是独立 vLLM EngineCore：
   - prompt token ids 先进入 prefill batch
   - 计算 KV cache
   - 生成的 token 进入 decode batch
   - 每次迭代输出 delta token ids
   - stop/EOS/max tokens 后结束

   因为是 `AGGREGATED`，prefill 和 decode 都在同一个 worker 的同一个 vLLM 实例内完成，不存在跨 worker KV transfer。又因为启动时禁用了 KV events，Dynamo 也没有可用于 KV-aware routing 的 cache 索引，所以本次路由策略就是纯 round-robin。

**返回流**

worker 返回的不是最终 OpenAI 文本，而是 Dynamo 内部 `LLMEngineOutput` 形态：

```json
{
  "index": 0,
  "token_ids": [...],
  "finish_reason": null,
  "completion_usage": {...}
}
```

返回路径：

```text
vLLM RequestOutput
  -> BaseWorkerHandler.generate_tokens() 转 dict
  -> Rust ManyOut stream
  -> TCP response stream
  -> Frontend RoutedEngine pythonize
  -> VllmProcessor._generate_and_stream()
  -> EngineCoreOutput
  -> vLLM OutputProcessor
  -> StreamingPostProcessor
  -> OpenAI choice/delta
```

对应源码：

- worker 输出 token chunk：[handlers.py](./dynamo/components/src/dynamo/vllm/handlers.py:3271)
- Frontend 调用 routed engine 并消费 stream：[vllm_processor.py](./dynamo/components/src/dynamo/frontend/vllm_processor.py:914)
- token 转文本和 OpenAI choice：[vllm_processor.py](./dynamo/components/src/dynamo/frontend/vllm_processor.py:993)
- 输出 envelope：[vllm_processor.py](./dynamo/components/src/dynamo/frontend/vllm_processor.py:1029)

流式请求直接变成 SSE delta；非流式请求内部仍先走 stream，再由 Rust HTTP 层聚合成一个 JSON response。

**模块关系**

```text
Client
  |
  | HTTP JSON
  v
Dynamo Frontend process
  |-- Rust HttpService / Axum
  |-- Rust ModelManager
  |-- Python VllmProcessor
  |     |-- vLLM InputProcessor / Tokenizer
  |     |-- vLLM OutputProcessor
  |     |-- StreamingPostProcessor
  |-- Rust RoutedEngine / PushRouter
  |-- File discovery watcher
  |
  | TCP msgpack: one selected instance
  +---------------------> Worker GPU 0
  +---------------------> Worker GPU 1
  +---------------------> Worker GPU 2
                              |
                              v
                         DecodeWorkerHandler
                              |
                              v
                         AsyncLLM / EngineCore
                              |
                              v
                         Qwen3-0.6B forward
```

**关键参数的架构含义**

- `CUDA_VISIBLE_DEVICES=0/1/2`：每个 worker 只看见一张卡，进程级 GPU 隔离。
- `--tensor-parallel-size 1`：每个副本单卡推理，不是三卡张量并行。
- `--gpu-memory-utilization 0.3`：每个独立 vLLM 进程按本卡总显存预留约 30%，所以三张卡各占约 24-25 GiB。
- `--enforce-eager`：禁用 CUDA Graph，降低首次验证复杂度，不代表模型不用 GPU。
- `--kv-events-config {"enable_kv_cache_events":false}`：禁用 KV 事件发布，因此没有 prefix-cache-aware 路由。
- `--router-mode round-robin`：只看可用 worker 集合，不看 KV 命中率或负载。
- `--served-model-name Qwen3-0.6B`：客户端模型名与本地路径解耦。
- `--namespace qwen3-agg`：把控制面发现范围限制在该 namespace。
- 三个不同 `DYN_SYSTEM_PORT`：避免三个 worker 在同一主机上的数据面监听冲突。

当前分析基于源码和此前成功测试记录；`18000` 服务现在未运行，因此没有新的 live trace。

> Frontend 入口在 [main.py (line 394)](./dynamo/components/src/dynamo/frontend/main.py:394)，它创建 `DistributedRuntime`，指定 `file` discovery 和 `tcp` request plane。  &#x20;
>
> 上面的内容中`file` discovery 和 `tcp` request plane，这二者分别是什么意思，作用是什么？请分析

`file` discovery 和 `tcp` request plane 是两条不同平面，前者解决“谁可用、在哪里”，后者解决“请求和响应怎么传”。

| 配置 | 所属平面 | 本质 | 作用 |
| --- | --- | --- | --- |
| `--discovery-backend file` | 控制面 / 服务发现 | 用本地文件目录做服务注册表 | Frontend 发现 3 个 worker、模型卡、endpoint 地址、实例增减 |
| `--request-plane tcp` | 数据面 / 推理请求传输 | raw TCP 流式 RPC | 把 token 请求发给选中的 worker，并把生成 token 流返回 Frontend |

**`file` discovery**

`file` discovery 表示 Dynamo 使用本地文件系统作为 discovery/KV 存储。启动时通过：

```bash
DYN_FILE_KV=/tmp/dynamo-qwen3-agg/discovery
```

指定这个注册表目录。源码中，`discovery-backend` 解析为 `file` 时会读取 `DYN_FILE_KV`，否则落到默认临时目录，见 [kv.rs](./dynamo/lib/runtime/src/storage/kv.rs:160)。

它的作用是：

1. 每个 vLLM worker 启动后，把自己的 `ModelDeploymentCard` 和 endpoint 信息写入这个目录。
2. Frontend 监听同一个目录，知道 `qwen3-agg` 下出现了哪些模型和 worker。
3. `--router-min-initial-workers 3` 就是在这个 discovery 结果上等待 3 个 worker 就绪。
4. worker 退出或失效后，discovery 事件会更新可用实例集合，路由候选随之变化。

因此，`file` discovery 只传“元数据”：模型名、模型能力、worker 类型、endpoint 地址、实例 ID、健康信息等。它不传用户的 prompt，也不传生成的 token。

这套配置适合单机部署，因为 Frontend 和 3 个 worker 必须能访问同一个 `DYN_FILE_KV` 目录。多机部署通常改用 Kubernetes 原生 discovery 或 etcd。

**`tcp` request plane**

`tcp` request plane 是实际推理数据的传输方式。Frontend 通过 HTTP 收到 `/v1/chat/completions` 请求后，会完成 chat 模板、tokenize 和路由选择，然后把 `PreprocessedRequest` 发送给选中的 worker。这个请求和后续 token 输出就走 request plane。

在这个部署里：

```text
Frontend :18000
  -> worker GPU0 :18081
  -> worker GPU1 :18082
  -> worker GPU2 :18083
```

其中 `18000` 是 Frontend 对外 OpenAI HTTP 端口；`18081-18083` 是三个 worker 的 Dynamo 系统端口，也就是它们在 TCP request plane 上的监听端口。

`tcp` 模式表示 Dynamo 使用 raw TCP 流式传输，支持请求帧、响应流、取消和错误传播。源码中 `RequestPlaneMode::Tcp` 的说明是“raw TCP request plane with msgpack support”，见 [distributed.rs](./dynamo/lib/runtime/src/distributed.rs:797)。

它的作用是：

1. 承载 Frontend 到 worker 的 token 请求。
2. 承载 worker 到 Frontend 的 delta token 输出。
3. 支持流式生成、取消、finish reason、错误信息。
4. 在 round-robin 选中某个 worker 后，把请求精确发给该实例。

**二者关系**

```text
控制面：file discovery
  “当前有哪些 worker？地址是什么？模型是什么？是否 ready？”
        |
        v
Frontend 维护 3 个候选 worker
        |
        | round-robin 选择一个
        v
数据面：tcp request plane
  “把这个请求发给选中的 worker，并把 token stream 带回来”
```

一次推理的简化流程是：

```text
curl -> Frontend HTTP
          |
          | 使用 file discovery 维护的 worker 列表选实例
          v
      PushRouter round-robin
          |
          | 通过 tcp request plane 连接 18081/18082/18083
          v
      vLLM worker GPU 0/1/2
          |
          | token stream 通过 tcp 返回
          v
      Frontend 组装 OpenAI response/SSE
          |
          v
      curl client
```

两个配置可以独立替换：discovery 可以换成 `etcd`、Kubernetes 或 `mem`；request plane 也可以换成 NATS。你的部署选择 `file + tcp`，含义就是：单机文件注册表负责服务发现，本机 raw TCP 负责实际推理数据传输。

> InfiniLM可以启动单一的服务。查看/workspace_codex/InfiniLM/run_server.sh启动服务，
>
> 任务：参考dynamo框架, 完善Infinilm推理框架的能力，支持上三副本聚合式推理服务。 infinilm.frontend提供统一 OpenAI API 与轮询路由的完整步骤。  infinilm.infinilm 启动推理服务。部署选择 `file + tcp`，单机文件注册表负责服务发现，本机 raw TCP 负责实际推理数据传输。 &#x20;
>
> 注意：尽量新增代码，不要大幅度修改InfiniLM的已有逻辑。
>
> 分析上述需求，提供修改方案。

<details><summary>6 previous messages</summary>

> 我会先只读梳理 InfiniLM 当前的启动方式、包结构、OpenAI 入口和现有服务间通信能力，再对照 Dynamo 的 discovery/request-plane 分层给出增量改造方案，不先动代码。
>
> <details><summary>Explored a file, a search, a list</summary>
>
> - Read `./run_server.sh`
> - Searched for `AGENTS.md` in `.`
> - Listed files
>
> </details>
>
> 当前 `run_server.sh` 只启动一个 `inference_server.py`，没有 frontend、注册表或副本路由。接下来我重点看这个 server 的 HTTP 协议、推理 API、请求对象和入口点定义，判断是复用它的现有接口作为 worker API，还是需要新增内部协议。
>
> <details><summary>Explored 4 files, a search, ran 2 commands</summary>
>
> <details><summary>Explored 6 files, a search</summary>
>
> - Read `./python/infinilm/server/inference_server.py`
> - Read `./python/infinilm/server/inference_server.py`
> - Read `./pyproject.toml`
> - Read `./setup.py`
> - Searched for `entry_points|console_scripts|inference_server|FastAPI|Flask|aiohttp|uvicorn|openai|chat` in `python`
> - Read `./python/infinilm/server/inference_server.py`
> - Read `./python/infinilm/server/inference_server.py`
>
> </details>
>
> - Ran `git status --short; git log -5 --oneline; rg -n "frontend|router|registry|discovery|round.?robin|request_plane|DYN_" . -g '*.py' -g '*.sh' -g '*.toml' | head -250`
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./python/infinilm/base_config.py`
> - Read `./python/infinilm/base_config.py`
>
> </details>
>
> - Ran `find python/infinilm -maxdepth 2 -type d -print | sort; git branch --show-current; git remote -v`
>
> <details><summary>Explored a file</summary>
>
> - Read `./python/infinilm/base_config.py`
>
> </details>
>
> </details>
>
> 现有推理端已经是 OpenAI-compatible HTTP，并且 `AsyncLLMEngine` 本身提供异步请求队列和 token stream；这有利于把“推理 worker 数据面”做成旁路 TCP 服务，而不是重写 scheduler。还需要确认 engine 的取消和健康接口，方案里要保证客户端断开时能终止 GPU 请求。
>
> <details><summary>Explored 2 files, 3 searches, ran a command</summary>
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Searched for `class AsyncLLMEngine|def is_healthy|def add_aborted_req|def stream_request|def add_chat_request` in `llm.py`
> - Read `./python/infinilm/llm/llm.py`
> - Read `./test/service/request.py`
> - Read `./python/infinilm/llm/llm.py`
>
> </details>
>
> - Ran `cat python/infinilm/__init__.py; cat python/infinilm/server/__init__.py 2>/dev/null || true; cat python/infinilm/llm/__init__.py | sed -n '1,120p'`
>
> <details><summary>Explored 2 searches</summary>
>
> - Searched for `if __name__ == .__main__.:|def main\(` in `infinilm`
> - Searched for `__main__.py` in `infinilm`
>
> </details>
>
> </details>
</details>

下面方案按“新增旁路能力，不改推理核心”的原则设计：`AsyncLLMEngine`、scheduler、model runner、KV cache、已有单服务 HTTP 逻辑保持不动，只给现有 `InferenceServer` 增加很小的生命周期挂钩，新增 frontend、file discovery、raw TCP worker 数据面和集群启动脚本。

**现状判断**

当前 [run_server.sh](./InfiniLM/run_server.sh:1) 只是启动一个进程：

```text
CUDA_VISIBLE_DEVICES=3,4
  -> python/infinilm/server/inference_server.py
  -> 单个 AsyncLLMEngine
  -> 单个 FastAPI OpenAI HTTP 服务 :8100
```

已有能力：

- [inference_server.py](./InfiniLM/python/infinilm/server/inference_server.py:260) 已提供 `/v1/chat/completions`
- [inference_server.py](./InfiniLM/python/infinilm/server/inference_server.py:295) 已提供 `/health`
- [llm.py](./InfiniLM/python/infinilm/llm/llm.py:567) 的 `AsyncLLMEngine` 已有异步请求队列、token stream、取消接口
- [llm.py](./InfiniLM/python/infinilm/llm/llm.py:904) 已支持 `add_chat_request()`
- 没有服务发现、worker 注册、统一 frontend、round-robin 路由和内部 raw TCP 数据面

所以改造重点是补齐“多副本编排层”，不是改推理本身。

---

## 1. 目标架构

```text
Client
  |
  | OpenAI HTTP JSON / SSE
  v
infinilm.frontend :18000
  |-- /v1/chat/completions
  |-- /v1/models
  |-- /health
  |-- /v1/models/{model}/ready
  |-- /metrics
  |
  |-- RoundRobinRouter
  |-- WorkerRegistry
  |
  | file discovery: DYN_FILE_KV=/tmp/xxx/discovery
  |   worker 0/1/2 heartbeat + metadata
  |
  | raw TCP request plane
  +-----------------------> worker GPU0 :18181
  +-----------------------> worker GPU1 :18182
  +-----------------------> worker GPU2 :18183
                                  |
                                  v
                         AggregatedTcpWorker
                                  |
                                  v
                         AsyncLLMEngine
                                  |
                                  v
                         prefill + decode + token stream
```

控制面与数据面分离：

- **file discovery**：只负责注册、心跳、发现 worker、判断 ready。
- **tcp request plane**：只负责传输 chat 请求、token 输出、取消和错误。
- 每个 worker 是一个完整 aggregated 推理实例，独立完成 prefill 和 decode。
- 三个副本之间不做 tensor parallel，也不做 KV transfer；`tp=1`，每副本独占一张 GPU。

---

## 2. 新增模块设计

### 2.1 `infinilm.discovery.file_registry`

新增：

```text
python/infinilm/discovery/__init__.py
python/infinilm/discovery/file_registry.py
```

提供两个核心类：

```python
class FileServiceRegistry:
    async def register(record: WorkerRecord) -> None
    async def heartbeat(record: WorkerRecord) -> None
    async def unregister(instance_id: str) -> None

class FileServiceWatcher:
    async def watch(namespace: str, endpoint: str) -> AsyncIterator[list[WorkerRecord]]
```

文件注册表目录：

```bash
DYN_FILE_KV=/tmp/infinilm-cluster/discovery
```

每个 worker 一个 JSON 文件，例如：

```text
/tmp/infinilm-cluster/discovery/
  qwen3-agg.infinilm.generate.host-pid-uuid.json
  qwen3-agg.infinilm.generate.host-pid-uuid.json
  qwen3-agg.infinilm.generate.host-pid-uuid.json
```

注册内容建议：

```json
{
  "schema_version": 1,
  "namespace": "qwen3-agg",
  "component": "infinilm",
  "endpoint": "generate",
  "instance_id": "host-pid-uuid",
  "worker_type": "aggregated",
  "model": "/data-aisoft/mechdancer/models/Qwen3-8B",
  "served_model_name": "Qwen3-8B",
  "transport": "tcp",
  "host": "127.0.0.1",
  "port": 18181,
  "http_host": "127.0.0.1",
  "http_port": 18081,
  "registered_at": 1760000000.0,
  "heartbeat_at": 1760000002.0,
  "lease_expires_at": 1760000010.0,
  "engine_healthy": true,
  "metadata": {
    "gpu": 0,
    "tp": 1,
    "max_batch_size": 32
  }
}
```

实现要点：

- 写入使用 `tmp file + os.replace()`，避免 frontend 读到半截 JSON。
- worker 每 2 秒 heartbeat 一次。
- frontend 每 500ms 到 1ms 轮询目录；单机场景足够，不需要引入 etcd。
- `heartbeat_at` 或 `lease_expires_at` 过期后，watcher 将 worker 从候选集合移除。
- worker 正常退出时删除自己的注册文件；异常退出后靠 lease 过期。
- 注册时机必须在模型加载完成、`AsyncLLMEngine.start()` 成功、TCP server 已监听之后，避免 frontend 过早路由。

---

### 2.2 `infinilm.runtime.tcp_protocol`

新增：

```text
python/infinilm/runtime/__init__.py
python/infinilm/runtime/tcp_protocol.py
```

建议使用 UTF-8 newline-delimited JSON over raw TCP：

- 不引入新依赖。
- 支持流式返回。
- JSON 序列化会转义 token 文本中的换行，`readline()` 安全。
- 单机本机通信性能瓶颈主要在 token 生成，不在 JSON 编码。

协议格式：

**请求：**

```json
{"op": "hello", "protocol_version": 1}
{"op": "chat", "request_id": "cmpl-xxx", "payload": {...OpenAI请求...}}
```

**响应：**

```json
{"type": "accepted", "request_id": "cmpl-xxx"}
{"type": "delta", "request_id": "cmpl-xxx", "content": "Ready"}
{"type": "finish", "request_id": "cmpl-xxx", "finish_reason": "stop", "usage": {...}}
{"type": "error", "request_id": "cmpl-xxx", "message": "..."}
```

**健康检查：**

```json
{"op": "health"}
```

返回：

```json
{"type": "health", "status": "ready", "engine_healthy": true}
```

取消策略：

- frontend 检测到客户端断开时，直接关闭到 worker 的 TCP 连接。
- worker 端 `reader` 收到 EOF 后，取消对应生成 task。
- worker 捕获 `asyncio.CancelledError` 后调用现有：

```python
engine.add_aborted_req(req, FinishReason.CANCELED)
```

这可以复用 [llm.py](./InfiniLM/python/infinilm/llm/llm.py:705) 的取消机制，不需要修改 scheduler。

---

### 2.3 `infinilm.runtime.worker_service`

新增：

```text
python/infinilm/runtime/worker_service.py
```

核心类：

```python
class AggregatedTcpWorker:
    def __init__(self, engine, inference_server, metadata): ...

    async def start(self) -> None:
        # asyncio.start_server(...)

    async def stop(self) -> None:
        # close tcp server, cancel tasks, unregister

    async def _handle_client(self, reader, writer) -> None:
        # handshake / health / chat

    async def _generate(self, request_id, payload, writer) -> None:
        # add_chat_request -> stream_request -> delta frames
```

worker 内部生成流程：

```text
TCP chat frame
  -> normalize messages
  -> build SamplingParams
  -> AsyncLLMEngine.add_chat_request()
  -> AsyncLLMEngine.stream_request()
  -> token_output
  -> {"type":"delta"} TCP frame
  -> {"type":"finish"} TCP frame
```

这里直接复用现有 `InferenceServer` 的经验：

- `_normalize_messages()`
- `_build_sampling_params()`
- `engine.add_chat_request()`
- `engine.stream_request()`
- EOS 过滤逻辑
- finish reason 转换逻辑

为了避免 worker 调太多私有方法，可以把这几段小逻辑抽到新的共享 helper，例如：

```text
python/infinilm/server/chat_utils.py
```

但只做机械搬移，不改变现有行为。

---

### 2.4 `infinilm.frontend`

新增：

```text
python/infinilm/frontend/__init__.py
python/infinilm/frontend/__main__.py
python/infinilm/frontend/app.py
python/infinilm/frontend/router.py
python/infinilm/frontend/client.py
```

#### `client.py`

```python
class TcpWorkerClient:
    async def connect(host, port) -> None
    async def health() -> dict
    async def stream_chat(request_id, payload) -> AsyncIterator[dict]
    async def close() -> None
```

#### `router.py`

```python
class RoundRobinRouter:
    def select(self, workers: list[WorkerRecord]) -> WorkerRecord
```

实现：

```python
index = self.cursor % len(workers)
self.cursor += 1
return workers[index]
```

候选集合必须满足：

- `worker_type == "aggregated"`
- namespace 匹配
- model 或 `served_model_name` 匹配
- lease 未过期
- `engine_healthy == true`
- TCP health 可选通过

故障处理：

- 连接失败且尚未产生 token：尝试下一个 worker。
- 已经输出部分 token 后失败：返回错误，不重试，避免重复生成。
- worker 连续失败：临时 quarantine，例如 5 秒，同时继续接受 discovery heartbeat 恢复。

#### `app.py`

FastAPI 统一入口：

| Endpoint | 作用 |
| --- | --- |
| `GET /health` | frontend 进程健康 |
| `GET /v1/models` | 从 discovery 聚合模型列表 |
| `GET /v1/models/{model}/ready` | 检查副本数量和健康状态 |
| `POST /v1/chat/completions` | OpenAI API，支持 stream 和非 stream |
| `GET /metrics` | 路由和 worker 指标 |

`POST /v1/chat/completions` 流程：

```text
解析 JSON
  -> 校验 model/messages
  -> WorkerRegistry.get_ready_workers()
  -> RoundRobinRouter.select()
  -> TcpWorkerClient.stream_chat()
  -> stream=true: 直接转 OpenAI SSE
  -> stream=false: 聚合 delta 后返回 chat.completion JSON
```

非流式响应内部仍然走 token stream，最后聚合，这与 Dynamo/vLLM 常见行为一致。

---

### 2.5 `infinilm.infinilm` 启动命令

新增：

```text
python/infinilm/infinilm.py
```

并在 `pyproject.toml` 中添加：

```toml
[project.scripts]
infinilm.frontend = "infinilm.frontend:main"
infinilm.infinilm = "infinilm.infinilm:main"
```

同时支持：

```bash
python -m infinilm.frontend
python -m infinilm.infinilm
```

`infinilm.infinilm` 是聚合式推理 worker 启动器，它复用现有 `InferenceServer`：

```text
infinilm.infinilm
  -> BaseConfig 解析原推理参数
  -> InferenceServer(...)
  -> FastAPI lifespan 中启动 AsyncLLMEngine
  -> engine ready 后启动 AggregatedTcpWorker
  -> 注册到 file discovery
  -> heartbeat
  -> 退出时 unregister
```

为了尽量少改 `inference_server.py`，建议只加两个可选 callback：

```python
class InferenceServer:
    def __init__(
        ...,
        engine_ready_callback=None,
        engine_stopping_callback=None,
    ):
        ...
```

生命周期改为：

```python
self.engine.start()
if self.engine_ready_callback:
    await self.engine_ready_callback(self.engine)
yield
if self.engine_stopping_callback:
    await self.engine_stopping_callback(self.engine)
self.engine.stop()
```

这是对已有文件的最小侵入式修改。scheduler、model runner、KV cache、HTTP chat 处理逻辑均不动。

---

## 3. CLI 参数新增

现有 `BaseConfig` 保持兼容，新增参数建议：

```bash
--served-model-name Qwen3-8B
--namespace qwen3-agg
--discovery-backend file
--file-kv /tmp/infinilm-cluster/discovery
--request-plane tcp
--worker-host 127.0.0.1
--worker-port 18181
--advertise-host 127.0.0.1
--registry-ttl 10
```

其中：

- `--port` 保持原含义：worker 本地 HTTP 诊断端口。
- `--worker-port` 是 raw TCP request-plane 端口。
- 三个 worker 必须使用不同 `--port` 和 `--worker-port`。
- `--namespace` 用于隔离不同集群。
- `--file-kv` 等价于 Dynamo 的 `DYN_FILE_KV`。

frontend 参数：

```bash
--namespace qwen3-agg
--discovery-backend file
--file-kv /tmp/infinilm-cluster/discovery
--request-plane tcp
--router-mode round-robin
--router-min-initial-workers 3
--http-host 127.0.0.1
--http-port 18000
--served-model-name Qwen3-8B
```

初期只实现：

```text
--discovery-backend file
--request-plane tcp
--router-mode round-robin
```

后续再扩展 etcd、Kubernetes、least-loaded、KV-aware 路由。

---

## 4. 集群启动脚本

不改现有 [run_server.sh](./InfiniLM/run_server.sh:1)，新增：

```text
run_cluster.sh
```

示例结构：

```bash
MODEL=/data-aisoft/mechdancer/models/Qwen3-8B
NAMESPACE=infinilm-agg
RUN_DIR=/tmp/infinilm-agg
FILE_KV="$RUN_DIR/discovery"

mkdir -p "$FILE_KV"

# frontend
python -m infinilm.frontend \
  --namespace "$NAMESPACE" \
  --discovery-backend file \
  --file-kv "$FILE_KV" \
  --request-plane tcp \
  --router-mode round-robin \
  --router-min-initial-workers 3 \
  --http-host 127.0.0.1 \
  --http-port 18000 &
FRONTEND_PID=$!

# workers
for gpu in 0 1 2; do
  http_port=$((18081 + gpu))
  worker_port=$((18181 + gpu))

  CUDA_VISIBLE_DEVICES=$gpu \
  python -m infinilm.infinilm \
    --model "$MODEL" \
    --served-model-name Qwen3-8B \
    --namespace "$NAMESPACE" \
    --discovery-backend file \
    --file-kv "$FILE_KV" \
    --request-plane tcp \
    --worker-host 127.0.0.1 \
    --worker-port "$worker_port" \
    --host 127.0.0.1 \
    --port "$http_port" \
    --device nvidia \
    --tp 1 \
    --block-size 256 \
    --max-new-tokens 2644 \
    --num-blocks 500 \
    --max-batch-size 32 \
    --log-level INFO \
    --enable-paged-attn \
    --attn flash-attn \
    --enable-graph &
done
```

端口规划：

| GPU | Worker HTTP | Worker TCP |
| --- | --- | --- |
| 0 | 18081 | 18181 |
| 1 | 18082 | 18182 |
| 2 | 18083 | 18183 |

Frontend 对外 OpenAI API：

```text
http://127.0.0.1:18000/v1
```

---

## 5. 数据流细节

### 流式请求

```text
client
  -> POST /v1/chat/completions {"stream": true}
  -> frontend round-robin 选择 worker
  -> TCP {"op":"chat", ...}
  -> worker AsyncLLMEngine
  -> token_output
  -> TCP {"type":"delta","content":"..."}
  -> frontend OpenAI SSE chunk
  -> client
```

### 非流式请求

```text
client
  -> POST /v1/chat/completions {"stream": false}
  -> frontend round-robin 选择 worker
  -> worker 仍按 token stream 返回
  -> frontend 聚合 content / usage / finish_reason
  -> client 收到一个 chat.completion JSON
```

### 客户端取消

```text
client disconnect
  -> frontend generator cancelled
  -> frontend closes worker TCP connection
  -> worker reader EOF
  -> worker cancel generate task
  -> engine.add_aborted_req(request, CANCELED)
  -> scheduler abort request
```

---

## 6. 指标与验证

新增 `/metrics` 指标：

```text
infinilm_router_workers{namespace="..."} 3
infinilm_router_requests_total{worker_id="...",status="success"}
infinilm_router_errors_total{worker_id="...",type="connect|stream|engine"}
infinilm_router_requests_inflight{worker_id="..."}
infinilm_router_tokens_total{worker_id="..."}
```

验证步骤：

1. 检查 frontend readiness：

```bash
curl http://127.0.0.1:18000/v1/models/Qwen3-8B/ready | jq
```

期望：

```json
{
  "ready": true,
  "workers": 3
}
```

2. 单请求冒烟：

```bash
curl http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-8B",
    "messages": [{"role":"user","content":"只回复 READY"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

3. 连续 6 个请求验证轮询：

```bash
for i in $(seq 1 6); do
  curl -s http://127.0.0.1:18000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model":"Qwen3-8B",
      "messages":[{"role":"user","content":"Return READY"}],
      "temperature":0,
      "max_tokens":8
    }' >/dev/null || exit 1
done
```

期望三个 worker 各处理 2 个请求。

4. 验证故障剔除：

```bash
# 停掉一个 worker
# readiness 从 3 变成 2
# 后续请求仍可成功
```

5. 验证客户端取消：

- 发起长流式请求。
- 中断 curl。
- 确认 worker 日志出现 request canceled。
- 确认 GPU 请求进入 abort 队列。

---

## 7. 测试方案

新增测试建议放在：

```text
test/service/test_file_registry.py
test/service/test_tcp_worker_protocol.py
test/service/test_round_robin_router.py
test/service/test_frontend_api.py
test/service/test_cluster_e2e.py
```

单元测试使用 fake engine，不依赖 GPU：

- file registry 原子写、heartbeat、TTL 过期。
- round-robin 3 worker 下 6 请求分布为 `2/2/2`。
- TCP 协议 delta/finish/error 顺序。
- 客户端断开后调用 `add_aborted_req()`。
- frontend stream 与非 stream 响应格式。
- worker crash 后 registry 剔除。

集成测试：

- 3 个真实 GPU worker。
- 每个 worker 独占 GPU 0/1/2。
- `/v1/models` 返回一个逻辑模型。
- readiness 报告 3 副本。
- 6 个请求均匀路由。
- 停掉一个 worker 后服务降级可用。
- worker 恢复后自动重新加入路由。

---

## 8. 推荐实施顺序

1. 新增 `file_registry` 和单元测试。
2. 新增 TCP protocol、fake engine worker 测试。
3. 新增 `AggregatedTcpWorker`，复用 `AsyncLLMEngine`。
4. 给 `InferenceServer` 加最小 lifecycle callback。
5. 新增 `infinilm.frontend` 和 round-robin router。
6. 新增 `infinilm.infinilm` 启动入口。
7. 新增 `run_cluster.sh`。
8. 做 3 GPU 端到端验证。
9. 最后补 `/metrics`、故障转移和取消语义的强化测试。

这个方案不需要依赖 Dynamo 包，也不改 InfiniLM 的模型执行内核；Dynamo 主要作为控制面/数据面分层、aggregated worker、round-robin 和 readiness 语义的参考。

> 要求：新建的文件尽量在一个文件夹中，那么这个文件夹的命名推荐叫什么？请分析。

推荐命名为：

```text
python/infinilm/cluster/
```

这个名称最贴合需求边界：它不是单进程 inference server，也不是模型执行内核，而是“多副本推理集群编排层”，涵盖 frontend、worker 注册、file discovery、TCP 数据面和 round-robin 路由。

**推荐目录结构**

```text
python/infinilm/cluster/
  __init__.py
  config.py          # 集群参数、namespace、file KV、端口、路由配置
  registry.py        # file discovery 注册、heartbeat、watch、TTL
  tcp.py             # raw TCP 协议编解码、client/server transport
  protocol.py        # chat/health/delta/finish/error 消息定义
  router.py          # round-robin 路由与 worker 候选管理
  worker.py          # infinilm.infinilm 聚合式推理 worker 入口
  frontend.py        # 统一 OpenAI API frontend
  metrics.py         # router/worker 指标
```

对应的命令入口可以通过 `pyproject.toml` 映射，避免为了命令名在 `cluster/` 外新增很多文件：

```toml
[project.scripts]
infinilm.frontend = "infinilm.cluster.frontend:main"
infinilm.infinilm = "infinilm.cluster.worker:main"
```

这样实际模块路径是：

```bash
python -m infinilm.cluster.frontend
python -m infinilm.cluster.worker
```

同时也能提供用户侧命令：

```bash
infinilm.frontend
infinilm.infinilm
```

**为什么不选其他名字**

- `frontend/`：太窄，放不下 worker、registry、TCP 数据面。
- `router/`：也太窄，只能表达选 worker，不能涵盖服务发现和 worker 生命周期。
- `distributed/`：仓库已有 `infinilm.distributed`，当前语义偏向 TP/PP/模型并行，混入集群服务发现会造成概念冲突。
- `runtime/`：容易和 C++/Python 推理 runtime 混淆，而且 Dynamo 的 `runtime` 范围很大，InfiniLM 这里只是集群编排层。
- `serving/`：语义可以，但和现有 `server/` 太接近，容易分不清单进程 HTTP server 和多副本编排层。
- `replicated/` 或 `aggregated/`：太绑定当前三副本 round-robin 形态，后续支持 P/D 分离、least-loaded、etcd/K8s discovery 时不自然。

因此，`cluster` 是最清晰、扩展性最好的命名。它表达的是“多进程、多副本、多节点能力”，但不预设具体路由算法、部署 backend 或推理拆分方式。

> 按照上述推荐实施顺序，进行修改和测试。

<details><summary>38 previous messages</summary>

> 我会按已定顺序落地为一个独立的 `python/infinilm/cluster/` 包：先实现 file registry 和 TCP 协议，再接 worker/frontend，最后补测试与打包入口。现有 `AsyncLLMEngine` 和推理 server 逻辑只做组合复用，不做结构性改动。
>
> <details><summary>Ran a command</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/cluster/__init__.py
> +"""Multi-worker serving support for InfiniLM."""
> +
> +from .config import FrontendOptions, WorkerOptions
> +
> +__all__ = [
> +    "FrontendOptions",
> +    "WorkerOptions",
> +]
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/cluster/config.py
> +"""Configuration and command-line parsing for the serving cluster."""
> +
> +from __future__ import annotations
> +
> +import argparse
> +import os
> +from dataclasses import dataclass
> +from pathlib import Path
> +
> +
> +def _add_cluster_arguments(parser: argparse.ArgumentParser) -> None:
> +    parser.add_argument(
> +        "--namespace",
> +        default=os.getenv("INFINILM_NAMESPACE", "infinilm-agg"),
> +        help="discovery namespace shared by frontend and workers",
> +    )
> +    parser.add_argument(
> +        "--discovery-backend",
> +        choices=["file"],
> +        default="file",
> +        help="service discovery backend; only file is supported initially",
> +    )
> +    parser.add_argument(
> +        "--file-kv",
> +        default=os.getenv("DYN_FILE_KV", "/tmp/infinilm_store_kv"),
> +        help="directory used by the file discovery backend",
> +    )
> +    parser.add_argument(
> +        "--request-plane",
> +        choices=["tcp"],
> +        default="tcp",
> +        help="transport used between frontend and inference workers",
> +    )
> +    parser.add_argument(
> +        "--registry-ttl",
> +        type=float,
> +        default=10.0,
> +        help="seconds after which a worker without heartbeat is removed",
> +    )
> +    parser.add_argument(
> +        "--heartbeat-interval",
> +        type=float,
> +        default=2.0,
> +        help="worker discovery heartbeat interval in seconds",
> +    )
> +
> +
> +@dataclass(frozen=True)
> +class WorkerOptions:
> +    namespace: str
> +    file_kv: Path
> +    registry_ttl: float
> +    heartbeat_interval: float
> +    served_model_name: str | None
> +    worker_host: str
> +    worker_port: int
> +    advertise_host: str
> +    component: str = "infinilm"
> +    endpoint: str = "generate"
> +    worker_type: str = "aggregated"
> +
> +
> +@dataclass(frozen=True)
> +class FrontendOptions:
> +    namespace: str
> +    file_kv: Path
> +    router_mode: str
> +    router_min_initial_workers: int
> +    http_host: str
> +    http_port: int
> +    connect_timeout: float
> +    request_timeout: float
> +    health_interval: float
> +    registry_ttl: float
> +    heartbeat_interval: float
> +
> +
> +def parse_worker_cluster_args(argv: list[str] | None = None) -> WorkerOptions:
> +    parser = argparse.ArgumentParser(add_help=False)
> +    _add_cluster_arguments(parser)
> +    parser.add_argument("--served-model-name", default=None)
> +    parser.add_argument("--worker-host", default="127.0.0.1")
> +    parser.add_argument("--worker-port", type=int, default=18181)
> +    parser.add_argument("--advertise-host", default="127.0.0.1")
> +    args, _ = parser.parse_known_args(argv)
> +    return WorkerOptions(
> +        namespace=args.namespace,
> +        file_kv=Path(args.file_kv),
> +        registry_ttl=args.registry_ttl,
> +        heartbeat_interval=args.heartbeat_interval,
> +        served_model_name=args.served_model_name,
> +        worker_host=args.worker_host,
> +        worker_port=args.worker_port,
> +        advertise_host=args.advertise_host,
> +    )
> +
> +
> +def remove_cluster_worker_args(argv: list[str]) -> list[str]:
> +    """Remove cluster-only flags so BaseConfig can parse inference flags."""
> +    parser = argparse.ArgumentParser(add_help=False)
> +    _add_cluster_arguments(parser)
> +    parser.add_argument("--served-model-name")
> +    parser.add_argument("--worker-host")
> +    parser.add_argument("--worker-port", type=int)
> +    parser.add_argument("--advertise-host")
> +    _, remaining = parser.parse_known_args(argv)
> +    return remaining
> +
> +
> +def parse_frontend_args(argv: list[str] | None = None) -> FrontendOptions:
> +    parser = argparse.ArgumentParser(description="Start the InfiniLM cluster frontend")
> +    _add_cluster_arguments(parser)
> +    parser.add_argument("--router-mode", choices=["round-robin"], default="round-robin")
> +    parser.add_argument("--router-min-initial-workers", type=int, default=1)
> +    parser.add_argument("--http-host", default="127.0.0.1")
> +    parser.add_argument("--http-port", type=int, default=18000)
> +    parser.add_argument("--connect-timeout", type=float, default=3.0)
> +    parser.add_argument("--request-timeout", type=float, default=1000.0)
> +    parser.add_argument("--health-interval", type=float, default=1.0)
> +    args = parser.parse_args(argv)
> +    return FrontendOptions(
> +        namespace=args.namespace,
> +        file_kv=Path(args.file_kv),
> +        router_mode=args.router_mode,
> +        router_min_initial_workers=args.router_min_initial_workers,
> +        http_host=args.http_host,
> +        http_port=args.http_port,
> +        connect_timeout=args.connect_timeout,
> +        request_timeout=args.request_timeout,
> +        health_interval=args.health_interval,
> +        registry_ttl=args.registry_ttl,
> +        heartbeat_interval=args.heartbeat_interval,
> +    )
> *** End Patch
> PATCH`
>
> </details>
>
> 基础配置模块已加入。现在实现控制面：worker 注册文件会使用原子写和 TTL，frontend watcher 定期刷新候选集合，只保留健康且未过期的 aggregated worker。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./InfiniLM/python/infinilm/cluster/registry.py` (+242 -0)
>
> </details>
>
> 控制面实现完成。接下来新增 raw TCP 数据面：使用 newline-delimited JSON 帧，支持 handshake、health、chat token 流、finish/error 和取消；worker 端会复用现有 `AsyncLLMEngine.add_chat_request()` 与 `stream_request()`，客户端断开时走现有 abort 队列。
>
> <details><summary>Created 2 files, ran a command</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/cluster/protocol.py
> +"""Wire messages for the InfiniLM cluster TCP request plane."""
> +
> +from __future__ import annotations
> +
> +from dataclasses import dataclass
> +from typing import Any
> +
> +
> +PROTOCOL_VERSION = 1
> +MAX_FRAME_BYTES = 16 * 1024 * 1024
> +
> +
> +@dataclass(frozen=True)
> +class WorkerHello:
> +    protocol_version: int = PROTOCOL_VERSION
> +    worker_type: str = "aggregated"
> +    model: str | None = None
> +    served_model_name: str | None = None
> +
> +
> +def encode_frame(message: dict[str, Any]) -> bytes:
> +    import json
> +
> +    return (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
> +        "utf-8"
> +    )
> +
> +
> +def decode_frame(line: bytes | str) -> dict[str, Any]:
> +    import json
> +
> +    if isinstance(line, bytes):
> +        line = line.decode("utf-8")
> +    message = json.loads(line)
> +    if not isinstance(message, dict):
> +        raise ValueError("TCP frame must be a JSON object")
> +    return message
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/cluster/tcp.py
> +"""Async raw-TCP transport for frontend-to-worker inference traffic."""
> +
> +from __future__ import annotations
> +
> +import asyncio
> +import logging
> +from typing import Any, AsyncIterator
> +
> +from .protocol import MAX_FRAME_BYTES, WorkerHello, decode_frame, encode_frame
> +
> +logger = logging.getLogger(__name__)
> +
> +
> +async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
> +    line = await reader.readline()
> +    if not line:
> +        raise ConnectionError("worker closed the TCP connection")
> +    if len(line) > MAX_FRAME_BYTES:
> +        raise ValueError(f"TCP frame exceeds {MAX_FRAME_BYTES} bytes")
> +    return decode_frame(line)
> +
> +
> +async def write_frame(
> +    writer: asyncio.StreamWriter,
> +    message: dict[str, Any],
> +) -> None:
> +    writer.write(encode_frame(message))
> +    await writer.drain()
> +
> +
> +class TcpWorkerClient:
> +    """One-request-per-connection client for streamed chat generation."""
> +
> +    def __init__(
> +        self,
> +        host: str,
> +        port: int,
> +        *,
> +        connect_timeout: float = 3.0,
> +    ) -> None:
> +        self.host = host
> +        self.port = port
> +        self.connect_timeout = connect_timeout
> +        self._reader: asyncio.StreamReader | None = None
> +        self._writer: asyncio.StreamWriter | None = None
> +
> +    @property
> +    def connected(self) -> bool:
> +        return self._reader is not None and self._writer is not None
> +
> +    async def connect(self, hello: WorkerHello) -> None:
> +        if self.connected:
> +            return
> +        self._reader, self._writer = await asyncio.wait_for(
> +            asyncio.open_connection(self.host, self.port),
> +            timeout=self.connect_timeout,
> +        )
> +        await write_frame(
> +            self._writer,
> +            {
> +                "op": "hello",
> +                "protocol_version": hello.protocol_version,
> +                "worker_type": hello.worker_type,
> +            },
> +        )
> +        response = await read_frame(self._reader)
> +        if response.get("type") != "hello_ack":
> +            raise ConnectionError(f"invalid worker handshake response: {response}")
> +        if response.get("protocol_version") != hello.protocol_version:
> +            raise ConnectionError("worker protocol version mismatch")
> +
> +    async def close(self, request_id: str | None = None) -> None:
> +        writer = self._writer
> +        if writer is None:
> +            return
> +        if request_id is not None and not writer.is_closing():
> +            try:
> +                await write_frame(writer, {"op": "cancel", "request_id": request_id})
> +            except (ConnectionError, OSError):
> +                pass
> +        writer.close()
> +        try:
> +            await writer.wait_closed()
> +        except (ConnectionError, OSError):
> +            pass
> +        self._reader = None
> +        self._writer = None
> +
> +    async def stream_chat(
> +        self,
> +        request_id: str,
> +        payload: dict[str, Any],
> +    ) -> AsyncIterator[dict[str, Any]]:
> +        if not self.connected:
> +            raise ConnectionError("worker client is not connected")
> +        assert self._reader is not None
> +        assert self._writer is not None
> +        await write_frame(
> +            self._writer,
> +            {"op": "chat", "request_id": request_id, "payload": payload},
> +        )
> +        try:
> +            while True:
> +                frame = await read_frame(self._reader)
> +                frame_type = frame.get("type")
> +                if frame_type in {"finish", "error"}:
> +                    if frame_type == "error":
>                         raise RuntimeError(frame.get("message", "worker error"))
> +                    return
> +                if frame_type != "delta":
> +                    if frame_type == "accepted":
> +                        continue
> +                    raise RuntimeError(f"unexpected worker frame: {frame}")
> +                yield frame
> +        finally:
> +            await self.close(request_id)
> +
> +    async def health(self, hello: WorkerHello) -> dict[str, Any]:
> +        await self.connect(hello)
> +        assert self._writer is not None
> +        await write_frame(self._writer, {"op": "health"})
> +        response = await read_frame(self._reader)
> +        if response.get("type") != "health":
> +            raise RuntimeError(f"invalid health response: {response}")
> +        await self.close()
> +        return response
> *** End Patch
> PATCH`
> - Wrote `./InfiniLM/python/infinilm/cluster/tcp.py` (+126 -0)
> - Wrote `./InfiniLM/python/infinilm/cluster/worker_service.py` (+263 -0)
>
> </details>
>
> TCP worker 端已接入现有异步引擎。现在实现 frontend 的核心路由：稳定排序候选 worker、原子递增 cursor、失败未出 token 时换下一个 worker，已经输出后不重试，避免重复生成。
>
> <details><summary>Ran a command</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/tcp.py
> @@
> -                if frame_type == "finish":
> -                    return
> +                if frame_type == "finish":
> +                    yield frame
> +                    return
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/cluster/metrics.py
> +"""Small in-process metrics collector for the serving cluster."""
> +
> +from __future__ import annotations
> +
> +from collections import defaultdict
> +from dataclasses import dataclass, field
> +
> +
> +@dataclass
> +class RouterMetrics:
> +    requests_total: int = 0
> +    errors_total: int = 0
> +    tokens_total: int = 0
> +    inflight: int = 0
> +    worker_requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
> +    worker_errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
> +
> +    def render(self, worker_count: int) -> str:
> +        lines = [
> +            "# HELP infinilm_router_workers Number of discovered workers.",
> +            "# TYPE infinilm_router_workers gauge",
> +            f"infinilm_router_workers {worker_count}",
> +            "# HELP infinilm_router_requests_total Total chat requests routed.",
> +            "# TYPE infinilm_router_requests_total counter",
> +            f"infinilm_router_requests_total {self.requests_total}",
> +            "# HELP infinilm_router_errors_total Total routing or worker errors.",
> +            "# TYPE infinilm_router_errors_total counter",
> +            f"infinilm_router_errors_total {self.errors_total}",
> +            "# HELP infinilm_router_tokens_total Token chunks streamed to clients.",
> +            "# TYPE infinilm_router_tokens_total counter",
> +            f"infinilm_router_tokens_total {self.tokens_total}",
> +        ]
> +        for worker_id, value in sorted(self.worker_requests.items()):
> +            lines.append(f'infinilm_router_worker_requests_total{{worker_id="{worker_id}"}} {value}')
> +        for worker_id, value in sorted(self.worker_errors.items()):
> +            lines.append(f'infinilm_router_worker_errors_total{{worker_id="{worker_id}"}} {value}')
> +        return "\n".join(lines) + "\n"
> +
> +
> +@dataclass
> +class FrontendMetrics:
> +    router: RouterMetrics = field(default_factory=RouterMetrics)
> +
> +    def render(self, worker_count: int) -> str:
> +        return self.router.render(worker_count)
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/cluster/router.py
> +"""Worker selection and request routing for the InfiniLM frontend."""
> +
> +from __future__ import annotations
> +
> +import asyncio
> +import logging
> +import uuid
> +from typing import Any, AsyncIterator
> +
> +from .metrics import RouterMetrics
> +from .protocol import WorkerHello
> +from .registry import FileServiceWatcher, WorkerRecord
> +from .tcp import TcpWorkerClient
> +
> +logger = logging.getLogger(__name__)
> +
> +
> +class NoWorkersAvailable(RuntimeError):
> +    pass
> +
> +
> +class RoundRobinRouter:
> +    def __init__(
> +        self,
> +        watcher: FileServiceWatcher,
> +        *,
> +        connect_timeout: float,
> +        request_timeout: float,
> +    ) -> None:
> +        self.watcher = watcher
> +        self.connect_timeout = connect_timeout
> +        self.request_timeout = request_timeout
> +        self.cursor = 0
> +        self.metrics = RouterMetrics()
> +
> +    def available_workers(self, model_name: str | None = None) -> list[WorkerRecord]:
> +        workers = self.watcher.workers
> +        if model_name:
> +            workers = [worker for worker in workers if worker.served_model_name == model_name]
> +        return workers
> +
> +    def select(self, workers: list[WorkerRecord]) -> WorkerRecord:
> +        if not workers:
> +            raise NoWorkersAvailable("no healthy aggregated workers are available")
> +        index = self.cursor % len(workers)
> +        self.cursor += 1
> +        return workers[index]
> +
> +    async def stream_chat(
> +        self,
> +        payload: dict[str, Any],
> +        *,
> +        model_name: str | None = None,
> +    ) -> AsyncIterator[dict[str, Any]]:
> +        request_id = payload.get("request_id") or f"cmpl-{uuid.uuid4().hex}"
> +        payload = {**payload, "request_id": request_id}
> +        workers = list(self.available_workers(model_name))
> +        if not workers:
> +            raise NoWorkersAvailable(
> +                f"no healthy workers for model {model_name or '<default>'}"
> +            )
> +
> +        last_error: Exception | None = None
> +        for _ in range(len(workers)):
> +            worker = self.select(workers)
> +            client = TcpWorkerClient(
> +                worker.host,
> +                worker.port,
> +                connect_timeout=self.connect_timeout,
> +            )
> +            hello = WorkerHello(
> +                model=worker.model,
> +                served_model_name=worker.served_model_name,
> +            )
> +            produced_output = False
> +            try:
> +                await client.connect(hello)
> +                self.metrics.requests_total += 1
> +                self.metrics.inflight += 1
> +                self.metrics.worker_requests[worker.instance_id] += 1
> +
> +                async def timed_stream() -> AsyncIterator[dict[str, Any]]:
> +                    async for frame in client.stream_chat(request_id, payload):
> +                        yield frame
> +
> +                async with asyncio.timeout(self.request_timeout):
> +                    async for frame in timed_stream():
> +                        produced_output = True
> +                        self.metrics.tokens_total += 1
> +                        yield {**frame, "worker_id": worker.instance_id}
> +                return
> +            except (ConnectionError, OSError, TimeoutError, asyncio.TimeoutError) as exc:
> +                last_error = exc
> +                self.metrics.errors_total += 1
> +                self.metrics.worker_errors[worker.instance_id] += 1
> +                logger.warning(
> +                    "Worker %s failed for request %s before retry boundary: %s",
> +                    worker.instance_id,
> +                    request_id,
> +                    exc,
> +                )
> +                if produced_output:
> +                    raise
> +                await client.close()
> +                continue
> +            except Exception:
> +                self.metrics.errors_total += 1
> +                self.metrics.worker_errors[worker.instance_id] += 1
> +                raise
> +            finally:
> +                self.metrics.inflight = max(0, self.metrics.inflight - 1)
> +
> +        raise ConnectionError(f"all workers failed for request {request_id}: {last_error}")
> *** End Patch
> PATCH`
>
> </details>
>
> 路由层已完成。现在补统一 OpenAI frontend：生命周期内启动 file watcher，提供 `/v1/models`、readiness、chat stream/非 stream 和 `/metrics`；非流式请求内部仍消费 worker token 流后聚合成一个 OpenAI JSON。
>
> <details><summary>Ran a command</summary>
>
> - Ran ```apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/cluster/frontend.py
> +"""Unified OpenAI API frontend for a multi-worker InfiniLM cluster."""
> +
> +from __future__ import annotations
> +
> +import json
> +import logging
> +import time
> +import uuid
> +from contextlib import asynccontextmanager
> +from typing import Any, AsyncIterator
> +
> +import uvicorn
> +from fastapi import FastAPI, Request
> +from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
> +
> +from .config import FrontendOptions, parse_frontend_args
> +from .metrics import FrontendMetrics
> +from .registry import FileServiceRegistry, FileServiceWatcher
> +from .router import NoWorkersAvailable, RoundRobinRouter
> +
> +logger = logging.getLogger(__name__)
> +
> +
> +class ClusterFrontend:
> +    def __init__(self, options: FrontendOptions) -> None:
> +        self.options = options
> +        self.registry = FileServiceRegistry(
> +            options.file_kv,
> +            lease_seconds=options.registry_ttl,
> +        )
> +        self.watcher = FileServiceWatcher(self.registry)
> +        self.router = RoundRobinRouter(
> +            self.watcher,
> +            connect_timeout=options.connect_timeout,
> +            request_timeout=options.request_timeout,
> +        )
> +        self.metrics = FrontendMetrics(router=self.router.metrics)
> +
> +    def _resolve_model(self, requested: str | None) -> tuple[str, list] | None:
> +        workers = self.router.available_workers(requested)
> +        if workers:
> +            return requested or workers[0].served_model_name, workers
> +        if not requested and self.watcher.workers:
> +            return self.watcher.workers[0].served_model_name, self.watcher.workers
> +        return None
> +
> +    def create_app(self) -> FastAPI:
> +        @asynccontextmanager
> +        async def lifespan(app: FastAPI):
> +            await self.watcher.refresh(self.options.namespace)
> +            self.watcher.start(
> +                self.options.namespace,
> +                interval=max(0.1, self.options.health_interval),
> +            )
> +            try:
> +                yield
> +            finally:
> +                await self.watcher.stop()
> +
> +        app = FastAPI(lifespan=lifespan)
> +
> +        @app.get("/health")
> +        async def health():
> +            worker_count = len(self.watcher.workers)
> +            return {
> +                "status": "healthy",
> +                "namespace": self.options.namespace,
> +                "workers": worker_count,
> +                "ready": worker_count >= self.options.router_min_initial_workers,
> +            }
> +
> +        @app.get("/v1/models")
> +        async def models():
> +            names = sorted({worker.served_model_name for worker in self.watcher.workers})
> +            return {
> +                "object": "list",
> +                "data": [
> +                    {
> +                        "id": name,
> +                        "object": "model",
> +                        "created": int(time.time()),
> +                        "owned_by": "infinilm",
> +                    }
> +                    for name in names
> +                ],
> +            }
> +
> +        @app.get("/v1/models/{model_name}/ready")
> +        async def model_ready(model_name: str):
> +            workers = self.router.available_workers(model_name)
> +            ready = len(workers) >= self.options.router_min_initial_workers
> +            return {
> +                "ready": ready,
> +                "model": model_name,
> +                "namespace": self.options.namespace,
> +                "worker_type": "aggregated",
> +                "workers": len(workers),
> +                "worker_ids": [worker.instance_id for worker in workers],
> +            }
> +
> +        @app.get("/metrics")
> +        async def metrics():
> +            return PlainTextResponse(
> +                self.metrics.render(len(self.watcher.workers)),
> +                media_type="text/plain; version=0.0.4",
> +            )
> +
> +        @app.post("/v1/chat/completions")
> +        @app.post("/chat/completions")
> +        async def chat_completions(request: Request):
> +            try:
> +                payload = await request.json()
> +            except Exception:
> +                return JSONResponse(
> +                    status_code=400,
> +                    content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
> +                )
> +            if not isinstance(payload, dict) or not payload.get("messages"):
> +                return JSONResponse(
> +                    status_code=400,
> +                    content={
> +                        "error": {
> +                            "message": "messages is required",
> +                            "type": "invalid_request_error",
> +                        }
> +                    },
> +                )
> +
> +            requested_model = payload.get("model")
> +            resolved = self._resolve_model(requested_model)
> +            if resolved is None:
> +                return JSONResponse(
> +                    status_code=503 if not requested_model else 404,
> +                    content={
> +                        "error": {
> +                            "message": f"model is not ready: {requested_model or '<default>'}",
> +                            "type": "service_unavailable" if not requested_model else "invalid_request_error",
> +                        }
> +                    },
> +                )
> +            model_name, _ = resolved
> +            payload = {**payload, "model": model_name}
> +            request_id = f"cmpl-{uuid.uuid4().hex}"
> +
> +            if payload.get("stream", False):
> +                return StreamingResponse(
> +                    self._stream_response(request_id, model_name, payload),
> +                    media_type="text/event-stream",
> +                )
> +
> +            try:
> +                response = await self._non_stream_response(request_id, model_name, payload)
> +            except NoWorkersAvailable as exc:
> +                return JSONResponse(
> +                    status_code=503,
> +                    content={"error": {"message": str(exc), "type": "service_unavailable"}},
> +                )
> +            except Exception as exc:
> +                logger.exception("Chat request %s failed", request_id)
> +                return JSONResponse(
> +                    status_code=500,
> +                    content={"error": {"message": str(exc), "type": "server_error"}},
> +                )
> +            return JSONResponse(content=response)
> +
> +        return app
> +
> +    async def _stream_response(
> +        self,
> +        request_id: str,
> +        model_name: str,
> +        payload: dict[str, Any],
> +    ) -> AsyncIterator[str]:
> +        def sse(chunk: dict[str, Any]) -> str:
> +            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
> +
> +        first = {
> +            "id": request_id,
> +            "object": "chat.completion.chunk",
> +            "created": int(time.time()),
> +            "model": model_name,
> +            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
> +        }
> +        try:
> +            yield sse(first)
> +            async for frame in self.router.stream_chat(payload, model_name=model_name):
> +                if frame.get("type") != "delta":
> +                    continue
> +                chunk = {
> +                    "id": request_id,
> +                    "object": "chat.completion.chunk",
> +                    "created": int(time.time()),
> +                    "model": model_name,
> +                    "choices": [
> +                        {
> +                            "index": 0,
> +                            "delta": {"content": frame.get("content", "")},
> +                            "finish_reason": None,
> +                        }
> +                    ],
> +                }
> +                yield sse(chunk)
> +            yield sse(
> +                {
> +                    "id": request_id,
> +                    "object": "chat.completion.chunk",
> +                    "created": int(time.time()),
> +                    "model": model_name,
> +                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
> +                }
> +            )
> +        except Exception as exc:
> +            logger.exception("Streaming request %s failed", request_id)
> +            yield sse(
> +                {
> +                    "id": request_id,
> +                    "object": "chat.completion.chunk",
> +                    "model": model_name,
> +                    "choices": [
> +                        {
> +                            "index": 0,
> +                            "delta": {},
> +                            "finish_reason": "error",
> +                        }
> +                    ],
> +                    "error": {"message": str(exc)},
> +                }
> +            )
> +        yield "data: [DONE]\n\n"
> +
> +    async def _non_stream_response(
> +        self,
> +        request_id: str,
> +        model_name: str,
> +        payload: dict[str, Any],
> +    ) -> dict[str, Any]:
> +        content_parts: list[str] = []
> +        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
> +        finish_reason = "stop"
> +        async for frame in self.router.stream_chat(payload, model_name=model_name):
> +            if frame.get("type") == "delta":
> +                content_parts.append(frame.get("content", ""))
> +            elif frame.get("type") == "finish":
> +                if isinstance(frame.get("usage"), dict):
> +                    usage = frame["usage"]
> +                finish_reason = frame.get("finish_reason") or "stop"
> +
> +        return {
> +            "id": request_id,
> +            "object": "chat.completion",
> +            "created": int(time.time()),
> +            "model": model_name,
> +            "system_fingerprint": None,
> +            "choices": [
> +                {
> +                    "index": 0,
> +                    "message": {"role": "assistant", "content": "".join(content_parts)},
> +                    "logprobs": None,
> +                    "finish_reason": finish_reason,
> +                }
> +            ],
> +            "usage": usage,
> +        }
> +
> +
> +def main(argv: list[str] | None = None) -> None:
> +    options = parse_frontend_args(argv)
> +    logging.basicConfig(
> +        level=logging.INFO,
> +        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
> +    )
> +    frontend = ClusterFrontend(options)
> +    logger.info(
> +        "Starting InfiniLM cluster frontend at %s:%s namespace=%s file_kv=%s",
> +        options.http_host,
> +        options.http_port,
> +        options.namespace,
> +        options.file_kv,
> +    )
> +    uvicorn.run(
> +        frontend.create_app(),
> +        host=options.http_host,
> +        port=options.http_port,
> +    )
> +
> +
> +if __name__ == "__main__":
> +    main()
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/frontend/__init__.py
> +"""Compatibility export for the ``infinilm.frontend`` command."""
> +
> +from infinilm.cluster.frontend import main
> +
> +__all__ = ["main"]
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/frontend/__main__.py
> +from . import main
> +
> +if __name__ == "__main__":
> +    main()
> *** End Patch
> PATCH```
>
> </details>
>
> frontend 已完成。下一步是把 `infinilm.infinilm` 做成现有 `InferenceServer` 的组合入口：不复制推理初始化逻辑，只在 FastAPI lifespan 中包一层，让 engine ready 后启动 TCP worker 并注册，engine 停止前反注册。
>
> <details><summary>Created a file, ran a command</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/worker_service.py
> @@
>      def __init__(
>          self,
>          *,
>          engine: AsyncLLMEngine,
>          inference_server: Any,
>          registry: FileServiceRegistry,
>          record: WorkerRecord,
> +        bind_host: str | None = None,
>      ) -> None:
> @@
>          self.record = record
> +        self.bind_host = bind_host or record.host
> @@
>          self._server = await asyncio.start_server(
>                          self._handle_client,
> -            host=self.record.host,
> +            host=self.bind_host,
>              port=self.record.port,
>          )
> @@
> -            advertised_host = self.record.host if self.record.host != "0.0.0.0" else "127.0.0.1"
> -            self.record = replace(self.record, host=advertised_host, port=port)
> +            self.record = replace(self.record, port=port)
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/base_config.py
> @@
> -    def __init__(self):
> +    def __init__(self, argv: list[str] | None = None):
> @@
> -        self.args, self.extra = self.parser.parse_known_args()
> +        self.args, self.extra = self.parser.parse_known_args(argv)
> *** End Patch
> PATCH`
> - Wrote `./InfiniLM/python/infinilm/cluster/worker.py` (+179 -0)
>
> </details>
>
> worker 启动入口已完成，现有 `inference_server.py` 没被改动。接下来补打包入口和集群启动脚本，然后先做静态编译检查，再写 fake-engine 单元测试覆盖 registry、round-robin、TCP 流式与 frontend API。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/pyproject.toml
> @@
>  [project.urls]
>  Homepage = "https://github.com/InfiniTensor/InfiniLM"
> +
> +[project.scripts]
> +infinilm.frontend = "infinilm.cluster.frontend:main"
> +infinilm.infinilm = "infinilm.cluster.worker:main"
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/setup.py
> @@
>  from setuptools import setup
>  from setuptools.command.build import build
>  from setuptools.command.develop import develop
>  from setuptools.command.egg_info import egg_info
> +from setuptools import find_packages
> @@
>      name="InfiniLM",
> @@
> -    package_dir={"": "python"},
> -    packages=["infinilm", "infinilm.models", "infinilm.lib", "infinilm.distributed"],
> +    package_dir={"": "python"},
> +    packages=find_packages(where="python"),
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Delete File: /workspace_codex/InfiniLM/python/infinilm/frontend/__init__.py
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Delete File: /workspace_codex/InfiniLM/python/infinilm/frontend/__main__.py
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Add File: /workspace_codex/InfiniLM/python/infinilm/cluster/run_cluster.sh
> +#!/usr/bin/env bash
> +set -euo pipefail
> +
> +MODEL="${MODEL:-/data-aisoft/mechdancer/models/Qwen3-8B}"
> +NAMESPACE="${NAMESPACE:-infinilm-agg}"
> +RUN_DIR="${RUN_DIR:-/tmp/infinilm-agg}"
> +FILE_KV="${RUN_DIR}/discovery"
> +SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B}"
> +GPUS=(${GPUS:-0 1 2})
> +FRONTEND_PORT="${FRONTEND_PORT:-18000}"
> +WORKER_HTTP_BASE_PORT="${WORKER_HTTP_BASE_PORT:-18081}"
> +WORKER_TCP_BASE_PORT="${WORKER_TCP_BASE_PORT:-18181}"
> +
> +mkdir -p "$FILE_KV"
> +pids=()
> +
> +cleanup() {
> +  local status=$?
> +  for pid in "${pids[@]:-}"; do
> +    if kill -0 "$pid" 2>/dev/null; then
> +      kill "$pid" || true
> +    fi
> +  done
> +  exit "$status"
> +}
> +trap cleanup EXIT INT TERM
> +
> +python -m infinilm.cluster.frontend \
> +  --namespace "$NAMESPACE" \
> +  --discovery-backend file \
> +  --file-kv "$FILE_KV" \
> +  --request-plane tcp \
> +  --router-mode round-robin \
> +  --router-min-initial-workers "${#GPUS[@]}" \
> +  --http-host 127.0.0.1 \
> +  --http-port "$FRONTEND_PORT" &
> +pids+=("$!")
> +
> +for i in "${!GPUS[@]}"; do
> +  gpu="${GPUS[$i]}"
> +  http_port=$((WORKER_HTTP_BASE_PORT + i))
> +  tcp_port=$((WORKER_TCP_BASE_PORT + i))
> +
> +  CUDA_VISIBLE_DEVICES="$gpu" \
> +  python -m infinilm.cluster.worker \
> +    --model "$MODEL" \
> +    --served-model-name "$SERVED_MODEL_NAME" \
> +    --namespace "$NAMESPACE" \
> +    --discovery-backend file \
> +    --file-kv "$FILE_KV" \
> +    --request-plane tcp \
> +    --worker-host 127.0.0.1 \
> +    --worker-port "$tcp_port" \
> +    --advertise-host 127.0.0.1 \
> +    --host 127.0.0.1 \
> +    --port "$http_port" \
> +    --device nvidia \
> +    --tp 1 \
> +    --block-size 256 \
> +    --max-new-tokens 2644 \
> +    --num-blocks 500 \
> +    --max-batch-size 32 \
> +    --log-level INFO \
> +    --enable-paged-attn \
> +    --attn flash-attn \
> +    --enable-graph &
> +  pids+=("$!")
> +done
> +
> +echo "frontend: http://127.0.0.1:${FRONTEND_PORT}/v1"
> +echo "discovery: $FILE_KV"
> +wait -n "${pids[@]}"
> *** End Patch
> PATCH`
> - Ran `chmod +x python/infinilm/cluster/run_cluster.sh && python -m compileall -q python/infinilm/cluster python/infinilm/base_config.py`
> - Ran `git status --short`
>
> </details>
>
> 静态编译通过。现在添加不依赖 GPU 的测试：fake `AsyncLLMEngine` 直接产生 token 流，覆盖注册表 TTL、round-robin 2/2/2 分布、TCP worker 的 delta/finish/cancel，以及 FastAPI frontend 的 OpenAI 流式和非流式响应。
>
> <details><summary>Created a file, explored a file, a search, ran a command</summary>
>
> - Ran `PYTHONPATH=python python - <<'PY'
> from infinilm.cluster.config import parse_frontend_args
> from infinilm.cluster.registry import FileServiceRegistry
> from infinilm.cluster.frontend import ClusterFrontend
> print('cluster imports: OK')
> PY`
>
> <details><summary>Explored a file, a search</summary>
>
> - Searched for `class TokenOutput|@dataclass` in `request.py`
> - Read `./python/infinilm/llm/request.py`
>
> </details>
>
> - Wrote `./InfiniLM/test/service/test_cluster.py` (+256 -0)
>
> </details>
>
> 测试文件已加入。我现在运行新增测试和 Ruff；如果 fake 流式取消暴露异步生成器生命周期问题，会先修协议层再重新跑。
>
> <details><summary>Explored a file, ran 8 commands</summary>
>
> - Ran `PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
>
> - Ran `python -m ruff check python/infinilm/cluster test/service/test_cluster.py`
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/worker_service.py
> @@
> -from .protocol import PROTOCOL_VERSION, read_frame, write_frame
> +from .protocol import PROTOCOL_VERSION
>  from .registry import FileServiceRegistry, WorkerRecord
> +from .tcp import read_frame, write_frame
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>          with TemporaryDirectory() as temporary:
> -            registry = FileServiceRegistry(Path(temporary))
>              watcher = type("Watcher", (), {"workers": []})()
> *** End Patch
> PATCH
>
> python -m ruff check --fix python/infinilm/cluster test/service/test_cluster.py`
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>      def test_three_workers_receive_two_requests_each(self):
> -        with TemporaryDirectory() as temporary:
> -            watcher = type("Watcher", (), {"workers": []})()
> -            router = RoundRobinRouter(
> -                watcher,
> -                connect_timeout=0.2,
> -                request_timeout=1.0,
> -            )
> -            watcher.workers = [f"worker-{index}" for index in range(3)]
> -            counts = {worker: 0 for worker in watcher.workers}
> -            for _ in range(6):
> -                counts[router.select(watcher.workers)] += 1
> -            self.assertEqual(list(counts.values()), [2, 2, 2])
> +        watcher = type("Watcher", (), {"workers": []})()
> +        router = RoundRobinRouter(
> +            watcher,
> +            connect_timeout=0.2,
> +            request_timeout=1.0,
> +        )
> +        watcher.workers = [f"worker-{index}" for index in range(3)]
> +        counts = {worker: 0 for worker in watcher.workers}
> +        for _ in range(6):
> +            counts[router.select(watcher.workers)] += 1
> +        self.assertEqual(list(counts.values()), [2, 2, 2])
> *** End Patch
> PATCH
>
> PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>                  health = await client.health(hello)
>                  self.assertEqual(health["status"], "ready")
> +                await client.connect(hello)
>  
>                  stream = client.stream_chat(
> *** End Patch
> PATCH
>
> PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
>
> <details><summary>Explored a file</summary>
>
> - Read `./test/service/test_cluster.py`
>
> </details>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>                  health = await client.health(hello)
>                  self.assertEqual(health["status"], "ready")
> +                await client.connect(hello)
> *** End Patch
> PATCH
>
> PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>          finally:
> -            if not request.is_finished():
> -                request.finished = True
> +            pass
> *** End Patch
> PATCH
>
> PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>          try:
>              for text in ("Ready", "."):
> -                await asyncio.sleep(0)
> +                await asyncio.sleep(0.05)
> *** End Patch
> PATCH
>
> PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
>
> </details>
>
> 新增 4 个 fake-engine 测试全部通过，包括 TCP 取消传播。接下来做格式/静态检查、检查 CLI 参数剥离是否保留原推理参数，并审查关键实现里可能影响真实启动的端口通告和 lifespan 包装问题。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `python -m ruff check python/infinilm/cluster test/service/test_cluster.py && python -m compileall -q python/infinilm/cluster python/infinilm/base_config.py`
> - Ran `PYTHONPATH=python python -m infinilm.cluster.frontend --help`
> - Ran `PYTHONPATH=python python - <<'PY'
> from infinilm.cluster.config import parse_worker_cluster_args, remove_cluster_worker_args
> args = ['--model', '/m', '--served-model-name', 'Qwen', '--namespace', 'agg', '--worker-port', '18181', '--tp', '1', '--enable-graph']
> options = parse_worker_cluster_args(args)
> remaining = remove_cluster_worker_args(args)
> print(options)
> print(remaining)
> assert remaining == ['--model', '/m', '--tp', '1', '--enable-graph']
> PY`
>
> </details>
>
> 静态与单元测试通过。我发现两处需要先加固：`asyncio.timeout` 不兼容项目声明的 Python 3.10；worker heartbeat 当前按 TTL 自动推导，没有使用 CLI 的 heartbeat 参数。我会修掉，再做 `infinilm.infinilm` 构建级检查。
>
> <details><summary>Explored 2 files, ran 10 commands</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/worker_service.py
> @@
>          registry: FileServiceRegistry,
>          record: WorkerRecord,
>          bind_host: str | None = None,
> +        heartbeat_interval: float | None = None,
> @@
>          self.bind_host = bind_host or record.host
> +        self.heartbeat_interval = heartbeat_interval
> @@
>      async def _heartbeat(self) -> None:
> -        interval = max(0.1, self.registry.lease_seconds / 3.0)
> +        interval = self.heartbeat_interval or max(
> +            0.1, self.registry.lease_seconds / 3.0
> +        )
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/worker.py
> @@
>                      registry=registry,
>                      record=record,
>                      bind_host=cluster_options.worker_host,
> +                    heartbeat_interval=cluster_options.heartbeat_interval,
>                  )
> @@
> -    setup_logging("INFO")
> +    setup_logging("INFO")
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/router.py
> @@
> -                async def timed_stream() -> AsyncIterator[dict[str, Any]]:
> -                    async for frame in client.stream_chat(request_id, payload):
> -                        yield frame
> -
> -                async with asyncio.timeout(self.request_timeout):
> -                    async for frame in timed_stream():
> -                        produced_output = True
> -                        self.metrics.tokens_total += 1
> -                        yield {**frame, "worker_id": worker.instance_id}
> +                stream = client.stream_chat(request_id, payload)
> +                deadline = asyncio.get_running_loop().time() + self.request_timeout
> +                try:
> +                    while True:
> +                        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
> +                        try:
> +                            frame = await asyncio.wait_for(
> +                                stream.__anext__(),
> +                                timeout=remaining,
> +                            )
> +                        except StopAsyncIteration:
> +                            break
> +                        produced_output = True
> +                        self.metrics.tokens_total += 1
> +                        yield {**frame, "worker_id": worker.instance_id}
> +                finally:
> +                    await stream.aclose()
>                  return
> *** End Patch
> PATCH`
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/worker.py
> @@
>      server = ClusterInferenceServer(
> @@
>          pre_transpose=cfg.pre_transpose,
>      )
> +    server.log_level = cfg.log_level
>      return server
> @@
> -    setup_logging("INFO")
> +    setup_logging(server.log_level)
> *** End Patch
> PATCH
>
> PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
>
> - Ran `PYTHONPATH=python python - <<'PY'
> from infinilm.cluster.worker import _build_server
> server = _build_server([
>     '--model', '/data-aisoft/mechdancer/models/Qwen3-8B',
>     '--device', 'nvidia',
>     '--served-model-name', 'Qwen3-8B',
>     '--namespace', 'infinilm-build-check',
>     '--worker-port', '0',
>     '--host', '127.0.0.1',
>     '--port', '0',
>     '--tp', '1',
>     '--skip-load',
> ])
> app = server._create_app()
> print('worker model_id:', server.model_id)
> print('fastapi lifespan wrapped:', app.router.lifespan_context.__name__)
> PY`
>
> - Ran `python -m ruff check python/infinilm/cluster test/service/test_cluster.py setup.py && git diff --check`
>
> - Ran `git diff --stat && git status --short`
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./python/infinilm/cluster/worker.py`
> - Read `./setup.py`
>
> </details>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/worker.py
> @@
>  def main(argv: list[str] | None = None) -> None:
>      args = sys.argv[1:] if argv is None else argv
> -    setup_logging(server.log_level)
>      server = _build_server(args)
> +    setup_logging(server.log_level)
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/setup.py
> @@
>  import subprocess
> -from pathlib import Path
>  
> -from setuptools import setup
> +from setuptools import find_packages, setup
> @@
> -from setuptools import egg_info
> -from setuptools import find_packages
> +from setuptools import egg_info
> *** End Patch
> PATCH
>
> python -m ruff check --fix python/infinilm/cluster test/service/test_cluster.py setup.py && git diff --check`
>
> - Ran `sed -n '1,15p' setup.py; python -m ruff check python/infinilm/cluster test/service/test_cluster.py setup.py; git diff --check`
>
> - Ran `PYTHONPATH=python python - <<'PY'
> from infinilm.cluster.worker import _build_server
> server = _build_server([
>     '--model', '/data-aisoft/mechdancer/models/Qwen3-8B',
>     '--device', 'nvidia',
>     '--served-model-name', 'Qwen3-8B',
>     '--namespace', 'infinilm-build-check',
>     '--worker-port', '0',
>     '--host', '127.0.0.1',
>     '--port', '0',
>     '--tp', '1',
>     '--skip-load',
> ])
> app = server._create_app()
> assert server.model_id == 'Qwen3-8B'
> assert app.router.lifespan_context.__name__ == 'cluster_lifespan'
> print('worker build check: OK')
> PY`
>
> - Ran `PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
>
> - Ran `bash -n python/infinilm/cluster/run_cluster.sh && python -m compileall -q python/infinilm/cluster python/infinilm/base_config.py`
>
> </details>
>
> CPU/fake-engine 测试、CLI 构建检查、脚本语法和 Ruff 都通过。现在检查 GPU、模型和端口状态；如果 0-2 卡可用，我会按新增脚本启动真实三副本做 OpenAI API 和轮询验证。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `ss -ltnp | grep -E ':(18000|18081|18082|18083|18181|18182|18183)\b' || true`
> - Ran `test -f /data-aisoft/mechdancer/models/Qwen3-8B/config.json && du -sh /data-aisoft/mechdancer/models/Qwen3-8B`
> - Ran `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader`
> - Ran `PYTHONPATH=/workspace_codex/InfiniLM/python MODEL=/data-aisoft/mechdancer/models/Qwen3-8B GPUS='0 1 2' RUN_DIR=/tmp/infinilm-agg-test NAMESPACE=infinilm-agg-test SERVED_MODEL_NAME=Qwen3-8B python/infinilm/cluster/run_cluster.sh`
>
> </details>
>
> 集群脚本已启动，Frontend 监听 `18000`，三个 worker 正在加载 16GB 模型。我会轮询 readiness 和 worker HTTP health，等三个注册文件全部出现后再发推理请求。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `find /tmp/infinilm-agg-test/discovery -type f -maxdepth 1 -print -exec cat {} \;`
> - Ran `curl --silent --show-error --fail --retry 20 --retry-delay 3 --retry-connrefused http://127.0.0.1:18000/v1/models/Qwen3-8B/ready`
> - Ran `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader`
>
> </details>
>
> 三个 worker 已全部注册 ready，分别占用 GPU 0/1/2，各约 39GB；file discovery 记录中的 HTTP、TCP 端口和 GPU 映射正确。现在先做非流式和流式真实推理，再连续发 6 个请求看 `/metrics` 中三个 worker 是否各 2 次。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `curl --silent --show-error --fail --max-time 180 http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"只回复 READY"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'`
> - Ran `curl --silent --show-error --fail --max-time 180 -N http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Return the single word READY."}],"temperature":0,"max_tokens":32,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'`
> - Ran `for i in 1 2 3 4 5 6; do curl --silent --show-error --fail --max-time 180 http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Return READY."}],"temperature":0,"max_tokens":8}' >/dev/null || exit 1; done; echo '6 requests: OK'; curl --silent http://127.0.0.1:18000/metrics`
>
> </details>
>
> 真实推理和轮询验证通过：新增 2 次冒烟后再发 6 次，三个 worker 的请求计数为 `3/3/2`，等价于后 6 次的 `2/2/2`，错误数为 0。我再修正启动脚本的一个故障语义：单个 worker 退出不应触发整个集群清理，只有 frontend 退出时才回收全部进程，然后做一次 worker 退出/恢复验证。
>
> <details><summary>Ran a command</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/run_cluster.sh
> @@
>  echo "frontend: http://127.0.0.1:${FRONTEND_PORT}/v1"
>  echo "discovery: $FILE_KV"
> -wait -n "${pids[@]}"
> +
> +# A worker crash is a routing-domain failure; frontend remains available with
> +# the remaining replicas. When frontend exits, cleanup reclaims every worker.
> +wait "${pids[0]}"
> *** End Patch
> PATCH
>
> bash -n python/infinilm/cluster/run_cluster.sh`
>
> </details>
>
> 当前运行中的集群仍使用旧脚本逻辑，我会先停止这次自测集群，再用修正后的脚本重新拉起，然后验证“杀掉一个 worker 后 readiness 降为 2、恢复后回到 3”。
>
> <details><summary>Ran 6 commands</summary>
>
> - Ran `PYTHONPATH=/workspace_codex/InfiniLM/python MODEL=/data-aisoft/mechdancer/models/Qwen3-8B GPUS='0 1 2' RUN_DIR=/tmp/infinilm-agg-test NAMESPACE=infinilm-agg-test SERVED_MODEL_NAME=Qwen3-8B python/infinilm/cluster/run_cluster.sh`
> - Ran `curl --silent --show-error --fail --retry 20 --retry-delay 2 --retry-connrefused http://127.0.0.1:18000/v1/models/Qwen3-8B/ready`
> - Ran `for i in $(seq 1 30); do body=$(curl --silent http://127.0.0.1:18000/v1/models/Qwen3-8B/ready); workers=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("workers", 0))'); echo "poll $i workers=$workers"; test "$workers" = 3 && break; sleep 2; done; test "$workers" = 3`
> - Ran `python3 - <<'PY'
> import json
> from pathlib import Path
> for path in Path('/tmp/infinilm-agg-test/discovery').glob('*.json'):
>     record=json.loads(path.read_text())
>     print(record['metadata']['cuda_visible_devices'], record['instance_id'].split('-')[0], record['port'])
> PY`
> - Ran `sleep 12; curl --silent http://127.0.0.1:18000/v1/models/Qwen3-8B/ready; echo; find /tmp/infinilm-agg-test/discovery -type f -maxdepth 1 | wc -l`
> - Ran `kill 3872526 && sleep 3 && curl --silent http://127.0.0.1:18000/v1/models/Qwen3-8B/ready && echo && curl --silent http://127.0.0.1:18000/health`
>
> </details>
>
> 单 worker 故障语义符合预期：frontend 未退出，ready 从 3 降到 2。现在用 2 副本发一次请求确认降级可用，再重新启动 GPU1 worker 验证自动回到 3 副本。
>
> <details><summary>Ran 10 commands</summary>
>
> - Ran `curl --silent --show-error --fail --max-time 180 http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Return READY."}],"temperature":0,"max_tokens":8}'`
> - Ran `nohup env PYTHONPATH=/workspace_codex/InfiniLM/python CUDA_VISIBLE_DEVICES=1 python -m infinilm.cluster.worker --model /data-aisoft/mechdancer/models/Qwen3-8B --served-model-name Qwen3-8B --namespace infinilm-agg-test --discovery-backend file --file-kv /tmp/infinilm-agg-test/discovery --request-plane tcp --worker-host 127.0.0.1 --worker-port 18182 --advertise-host 127.0.0.1 --host 127.0.0.1 --port 18082 --device nvidia --tp 1 --block-size 256 --max-new-tokens 2644 --num-blocks 500 --max-batch-size 32 --log-level INFO --enable-paged-attn --attn flash-attn --enable-graph >/tmp/infinilm-gpu1-restart.log 2>&1 & echo $!`
> - Ran `sleep 3; tail -30 /tmp/infinilm-gpu1-restart.log`
> - Ran `for i in $(seq 1 30); do body=$(curl --silent http://127.0.0.1:18000/v1/models/Qwen3-8B/ready); workers=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("workers", 0))'); echo "poll $i workers=$workers"; test "$workers" = 3 && break; sleep 2; done; test "$workers" = 3; printf '%s\n' "$body"`
> - Ran `ps -p 3876117 -o pid,stat,cmd || true; ls -l /tmp/infinilm-gpu1-restart.log; cat /tmp/infinilm-gpu1-restart.log`
> - Ran `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader`
> - Ran `env PYTHONPATH=/workspace_codex/InfiniLM/python CUDA_VISIBLE_DEVICES=1 python -m infinilm.cluster.worker --model /data-aisoft/mechdancer/models/Qwen3-8B --served-model-name Qwen3-8B --namespace infinilm-agg-test --discovery-backend file --file-kv /tmp/infinilm-agg-test/discovery --request-plane tcp --worker-host 127.0.0.1 --worker-port 18182 --advertise-host 127.0.0.1 --host 127.0.0.1 --port 18082 --device nvidia --tp 1 --block-size 256 --max-new-tokens 2644 --num-blocks 500 --max-batch-size 32 --log-level INFO --enable-paged-attn --attn flash-attn --enable-graph`
> - Ran `for i in $(seq 1 20); do body=$(curl --silent http://127.0.0.1:18000/v1/models/Qwen3-8B/ready); workers=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("workers", 0))'); echo "poll $i workers=$workers"; test "$workers" = 3 && break; sleep 2; done; test "$workers" = 3; printf '%s\n' "$body"`
> - Ran `curl --silent --show-error --fail --max-time 180 http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Return READY."}],"temperature":0,"max_tokens":8}' | python3 -m json.tool`
> - Ran `curl --silent http://127.0.0.1:18000/metrics`
>
> </details>
>
> 恢复验证通过：readiness 回到 3，恢复后推理成功，错误计数为 0。我现在停止这轮自测的 frontend、GPU0/2 worker 和单独恢复的 GPU1 worker，然后做最终代码检查和变更汇总。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `curl --silent --show-error http://127.0.0.1:18000/health || test $? -eq 7`
> - Ran `python -m ruff check python/infinilm/cluster test/service/test_cluster.py setup.py && PYTHONPATH=python python -m pytest -q test/service/test_cluster.py && git diff --check && bash -n python/infinilm/cluster/run_cluster.sh`
> - Ran `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader`
>
> </details>
>
> 最后再做两处可靠性收紧：启动时真正等待 `router-min-initial-workers` 后再对外提供服务；过期的 discovery JSON 文件从目录中清理，避免长期自测/崩溃后残留文件误导人工排查。
>
> <details><summary>Explored 3 searches, ran 5 commands</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/registry.py
> @@
>              if record.worker_type != "aggregated":
>                  continue
> -            if not record.engine_healthy or record.lease_expires_at < now:
> +            if not record.engine_healthy:
>                  continue
> +            if record.lease_expires_at < now:
> +                path.unlink(missing_ok=True)
> +                continue
>              records.append(record)
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/frontend.py
> @@
>          @asynccontextmanager
>          async def lifespan(app: FastAPI):
>              await self.watcher.refresh(self.options.namespace)
> +            while (
> +                len(self.watcher.workers)
> +                < self.options.router_min_initial_workers
> +            ):
> +                logger.info(
> +                    "Waiting for %d workers; currently %d",
> +                    self.options.router_min_initial_workers,
> +                    len(self.watcher.workers),
> +                )
> +                await asyncio.sleep(
> +                    max(0.1, self.options.health_interval)
> +                )
> +                await self.watcher.refresh(self.options.namespace)
>              self.watcher.start(
> *** End Patch
> PATCH`
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/frontend.py
> @@
>  import json
>  import logging
>  import time
>  import uuid
> +import asyncio
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>              expired = replace(record, lease_expires_at=time.time() - 1)
>              asyncio.run(registry.register(expired))
>              self.assertEqual(asyncio.run(registry.snapshot("test")), [])
> +            self.assertEqual(list(Path(temporary).glob("*.json")), [])
> *** End Patch
> PATCH
>
> python -m ruff check --fix python/infinilm/cluster test/service/test_cluster.py && PYTHONPATH=python python -m pytest -q test/service/test_cluster.py && git diff --check`
>
> - Ran `find python/infinilm/cluster -maxdepth 1 -type f -printf '%f\n' | sort; wc -l python/infinilm/cluster/*.py python/infinilm/cluster/run_cluster.sh test/service/test_cluster.py`
>
> - Ran `git diff -- pyproject.toml setup.py python/infinilm/base_config.py && git status --short`
>
> - Ran `python -m ruff check python/infinilm/cluster test/service/test_cluster.py setup.py && python -m compileall -q python/infinilm/cluster python/infinilm/base_config.py && bash -n python/infinilm/cluster/run_cluster.sh`
>
> <details><summary>Explored 3 searches</summary>
>
> - Searched for `class FileServiceRegistry|class FileServiceWatcher|class TcpWorkerService|class RoundRobinRouter|class ClusterFrontend|def main|create_app|run_cluster` in `cluster`
> - Searched for `def test_|class .*Test` in `test_cluster.py`
> - Searched for `project.scripts|def __init__\(self, argv` in `pyproject.toml`
>
> </details>
>
> </details>
</details>

已按推荐顺序完成实现和测试。新增能力集中在 [python/infinilm/cluster](./InfiniLM/python/infinilm/cluster/__init__.py:1)，现有 `InferenceServer`、`AsyncLLMEngine`、scheduler、model runner 和 KV cache 逻辑没有结构性修改，原 [run_server.sh](./InfiniLM/run_server.sh:1) 也保持不变。

**实现内容**

- 新增 file discovery 注册表和 watcher：
  - [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:43)
  - worker 使用原子 JSON 文件注册、周期 heartbeat、TTL 失效和自动清理。
  - frontend 通过 watcher 维护当前可用 aggregated worker 集合。

- 新增 raw TCP request plane：
  - [protocol.py](./InfiniLM/python/infinilm/cluster/protocol.py:1)
  - [tcp.py](./InfiniLM/python/infinilm/cluster/tcp.py:1)
  - 使用 UTF-8 newline-delimited JSON 传输 `hello`、`health`、`chat`、`delta`、`finish`、`error` 和 `cancel` 帧。
  - frontend 关闭 TCP 连接时会发送 cancel，worker 复用现有 `AsyncLLMEngine.add_aborted_req()` 取消请求。

- 新增 aggregated worker 服务：
  - [worker_service.py](./InfiniLM/python/infinilm/cluster/worker_service.py:19)
  - 直接组合现有 `AsyncLLMEngine`，worker 内完成 prefill、decode 和 token stream。
  - [worker.py](./InfiniLM/python/infinilm/cluster/worker.py:47) 包装现有 FastAPI lifespan：engine ready 后启动 TCP server 并注册，退出时停止并反注册。

- 新增统一 OpenAI frontend 和 round-robin 路由：
  - [frontend.py](./InfiniLM/python/infinilm/cluster/frontend.py:25)
  - [router.py](./InfiniLM/python/infinilm/cluster/router.py:22)
  - 支持 `/v1/chat/completions`、流式 SSE、非流式聚合响应、`/v1/models`、`/health`、`/v1/models/{model}/ready` 和 `/metrics`。
  - frontend 启动时会等待 `--router-min-initial-workers` 个 worker ready。
  - 请求失败且尚未输出 token 时会尝试下一个 worker；已输出 token 后不重试，避免重复生成。

- 新增集群启动脚本：
  - [run_cluster.sh](./InfiniLM/python/infinilm/cluster/run_cluster.sh:1)
  - 默认启动 GPU 0/1/2 三个 worker。
  - Frontend: `18000`
  - Worker HTTP: `18081-18083`
  - Worker TCP: `18181-18183`
  - 单个 worker crash 时 frontend 继续服务，剩余副本降级可用；frontend 退出时脚本回收全部 worker。

- 新增命令入口：
  - [pyproject.toml](./InfiniLM/pyproject.toml:21)
  - `infinilm.frontend`
  - `infinilm.infinilm`

- 对既有代码只做最小修改：
  - [base_config.py](./InfiniLM/python/infinilm/base_config.py:49) 允许 worker 入口先剥离集群参数，再复用原推理参数解析。
  - [setup.py](./InfiniLM/setup.py:36) 改用 `find_packages`，确保 `infinilm.cluster` 等子包被安装。

**启动方式**

仓库内直接启动：

```bash
cd /workspace_codex/InfiniLM
PYTHONPATH=$PWD/python \
MODEL=/data-aisoft/mechdancer/models/Qwen3-8B \
GPUS="0 1 2" \
SERVED_MODEL_NAME=Qwen3-8B \
python/infinilm/cluster/run_cluster.sh
```

安装为 editable 包后也可以使用：

```bash
infinilm.frontend --help
infinilm.infinilm --help
```

或模块方式：

```bash
python -m infinilm.cluster.frontend
python -m infinilm.cluster.worker
```

**测试结果**

- 新增 fake-engine 测试：[test_cluster.py](./InfiniLM/test/service/test_cluster.py:86)

覆盖项包括：

- file registry 注册、TTL 过期和过期文件清理
- round-robin 三 worker 下 6 个请求分布为 `2/2/2`
- TCP health、chat stream、finish usage
- 客户端取消传播到 `AsyncLLMEngine.add_aborted_req()`
- frontend OpenAI 流式与非流式响应

结果：

```text
4 passed
```

静态检查也通过：

```text
ruff: passed
compileall: passed
git diff --check: passed
run_cluster.sh syntax: passed
worker build check: passed
```

**真实 GPU 验证**

已用 `Qwen3-8B` 在 GPU 0/1/2 上完成真实三副本测试：

- readiness 从 0 到 3
- 三个 worker 分别占用 GPU 0/1/2
- 每个模型进程约 39GB 显存
- file discovery 中 GPU、HTTP port、TCP port、namespace、model 信息正确
- 非流式 `/v1/chat/completions` 成功
- 流式 `/v1/chat/completions` 成功返回 OpenAI SSE 和 `[DONE]`
- 连续 6 个请求后，结合前 2 次冒烟，三个 worker 计数为 `3/3/2`，证明 6 个验证请求分布为 `2/2/2`
- `infinilm_router_errors_total` 为 `0`

故障验证也通过：

- 停止 GPU1 worker 后，frontend 未退出
- readiness 从 3 降为 2
- 剩余 2 副本继续成功推理
- 重新启动 GPU1 worker 后，readiness 自动恢复为 3
- 恢复后推理成功

自测集群和所有测试进程已停止，GPU 显存已释放。当前工作区里 `.gitignore` 和 `scripts/test_perf.py` 原有修改仍保留，我没有改动它们。

> 安装为 editable 包后也可以使用。那如何安装呢？bash build.sh --infinilm可以安装么。

<details><summary>2 previous messages</summary>

> 我先查看工作区根目录的 `build.sh` 和 InfiniLM 的构建脚本参数，确认 `--infinilm` 实际执行的是构建、安装还是拷贝，避免给出错误安装命令。
>
> <details><summary>Explored a file, a search</summary>
>
> - Read `./build.sh`
> - Searched for `infinilm|pip install|editable|develop|-e` in `build.sh`
>
> </details>
</details>

可以。`bash build.sh --infinilm` 会安装为 editable 包。

对应逻辑在 [build.sh](./build.sh:29)：

```bash
cd InfiniLM
xmake f -cv
xmake build _infinilm && xmake install _infinilm
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

所以执行方式是：

```bash
cd /workspace_codex
bash build.sh --infinilm
```

它会做三件事：

1. 重新配置 xmake
2. 编译并安装 `_infinilm` C++ 扩展
3. 在 `/workspace_codex/InfiniLM` 上执行 `pip install -e .`

安装完成后，新增的两个命令入口会可用：

```bash
infinilm.frontend --help
infinilm.infinilm --help
```

也可以验证导入路径：

```bash
python3 -c 'import infinilm.cluster; print(infinilm.cluster.__file__)'
```

应指向：

```text
/workspace_codex/InfiniLM/python/infinilm/cluster/__init__.py
```

如果 C++ 扩展已经编译好，只想更新 Python editable 安装和命令入口，也可以进入仓库执行：

```bash
cd /workspace_codex/InfiniLM
pip install -e . --no-build-isolation
```

但在当前机器上，推荐直接用：

```bash
cd /workspace_codex
bash build.sh --infinilm
```

因为 InfiniLM 依赖 `_infinilm` 原生扩展，完整脚本会一并处理。

> 请执行脚本，安装/

> 请执行脚本，安装。

<details><summary>7 previous messages</summary>

> 我现在执行 `/workspace_codex/build.sh --infinilm`。这个脚本会先配置和编译 `_infinilm` 原生扩展，再执行 `pip install -e .`，耗时可能比纯 Python 安装长。
>
> <details><summary>Ran a command</summary>
>
> - Ran `bash build.sh --infinilm`
>
> </details>
>
> 安装脚本已启动，当前在 xmake 平台和工具链检查阶段。我继续跟踪编译和 pip 安装输出。
>
> 脚本里的原生扩展编译和安装成功了，但 editable 安装失败。原因是 `pyproject.toml` 中带点号的命令名被 TOML 解析成了嵌套表：`project.scripts.infinilm.frontend`，而 setuptools 要求每个 entry point 都是字符串。我会把两个命令名加引号修正，然后重新执行安装脚本。
>
> <details><summary>Edited a file, ran a command</summary>
>
> - Wrote `./InfiniLM/pyproject.toml` (+2 -2)
> - Ran `bash build.sh --infinilm`
>
> </details>
>
> 安装已完成。现在验证 editable 包是否指向当前仓库、两个带点号的命令入口是否真正生成，并快速执行 help。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `python3 -c 'import infinilm, infinilm.cluster; print(infinilm.__file__); print(infinilm.cluster.__file__)'`
> - Ran `command -v infinilm.frontend; command -v infinilm.infinilm`
> - Ran `infinilm.frontend --help | head -20`
> - Ran `infinilm.infinilm --help | head -25`
>
> </details>
</details>

安装完成。执行的是：

```bash
cd /workspace_codex
bash build.sh --infinilm
```

结果：

- `_infinilm` 原生扩展编译并安装成功
- `InfiniLM 0.1.0` 已成功安装为 editable 包
- editable 包指向当前仓库：

```text
/workspace_codex/InfiniLM/python/infinilm/__init__.py
```

两个命令入口也已安装成功：

```text
/usr/local/bin/infinilm.frontend
/usr/local/bin/infinilm.infinilm
```

并且 `infinilm.frontend --help`、`infinilm.infinilm --help` 均可正常执行。

补充说明：第一次安装时 `pip` 失败，原因是 `pyproject.toml` 中带点号的脚本名没有加引号，被 TOML 解析成嵌套字段。我已将其修正为：

```toml
[project.scripts]
"infinilm.frontend" = "infinilm.cluster.frontend:main"
"infinilm.infinilm" = "infinilm.cluster.worker:main"
```

随后重新执行安装脚本，已成功完成。
