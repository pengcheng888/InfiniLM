# 查看该文档/workspace_codex/dynamo仓库学习记录.md。 (2)

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
> - Searched for `module.*vllm|class .*Vllm|generate\(|prefill|decode` in `vllm`
> - Searched for `chat/completions|ChatCompletion|openai` in `frontend`
> - Searched for `class .*Router|route\(|round_robin|RoundRobin|worker` in `frontend`
>
> </details>
>
> 初步看，这个部署的关键路径在 `frontend/main.py`、`frontend/vllm_processor.py` 和 `vllm/worker_factory.py`。我接下来只读这些文件的相关区间，确认 HTTP 回调如何进入 Python、`routed_engine.generate()` 如何选 worker，以及 aggregated worker 为什么在拓扑里显示为 decode。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./components/src/dynamo/frontend/vllm_processor.py`
> - Read `./components/src/dynamo/frontend/main.py`
> - Read `./components/src/dynamo/vllm/worker_factory.py`
>
> </details>
>
> 源码显示 worker 工厂里 `AGGREGATED` 与 `DECODE` 共用 `_run_decode_worker`，区别在于 `worker_type=WorkerType.Aggregated` 且没有对 decode peer 的依赖；这解释了拓扑命名。还需要补齐两个关键点：Python `RoutedEngine` 到 Rust transport 的边界，以及 vLLM handler 内部从请求对象到 `AsyncLLM.generate` 的调用。
>
> <details><summary>Explored 20 files, 16 searches</summary>
>
> - Searched for `class .*Router|struct .*Router|enum .*Router|round_robin|RoundRobin` in `lib`
> - Searched for `^(async )?def run_input|^async def run_input|class RoutedEngine|def generate\(|async def generate\(` in `.`
> - Searched for `class .*Handler|def generate\(` in `vllm`
> - Searched for `def run_input|run_input\(` in `python`
> - Read `./components/src/dynamo/vllm/handlers.py`
> - Searched for `class RoutedEngine|RoutedEngine` in `python`
> - Searched for `def _generate_token_mode|def _generate_text_mode|async def _generate` in `handlers.py`
> - Read `./components/src/dynamo/vllm/handlers.py`
> - Read `./lib/bindings/python/rust/llm/routed_engine.rs`
> - Read `./lib/llm/src/entrypoint/input.rs`
> - Read `./components/src/dynamo/vllm/handlers.py`
> - Searched for `pub struct PrefillRoutedEngine|impl PrefillRoutedEngine|struct HttpFrontend|chat_engine_factory|ChatEngineFactory` in `entrypoint`
> - Read `./lib/bindings/python/rust/llm/entrypoint.rs`
> - Read `./lib/llm/src/entrypoint/input/http.rs`
> - Searched for `PrefillRoutedEngine` in `llm`
> - Read `./lib/llm/src/discovery/watcher.rs`
> - Searched for `register_model\(|ModelDeploymentCard|list_and_watch` in `vllm`
> - Read `./lib/llm/src/entrypoint.rs`
> - Searched for `build_preprocessed_pipeline|build_preprocessed_routing_with_selector|PushRouter::from_client|RouterMode::RoundRobin` in `common.rs`
> - Read `./lib/runtime/src/pipeline/network/egress/push_router.rs`
> - Read `./components/src/dynamo/vllm/main.py`
> - Searched for `def register_model|async def register_model` in `dynamo`
> - Read `./lib/llm/src/entrypoint/input/common.rs`
> - Searched for `async def register_model\(|def register_model\(` in `.`
> - Searched for `async def chat_engine_factory|chat_engine_factory =|VllmProcessor\(` in `vllm_processor.py`
> - Searched for `round_robin_prepared|async fn round_robin\(` in `push_router.rs`
> - Searched for `register_model` in `python`
> - Read `./lib/runtime/src/pipeline/network/egress/push_router.rs`
> - Read `./components/src/dynamo/frontend/vllm_processor.py`
> - Read `./lib/llm/src/local_model.rs`
> - Searched for `def generator|async def generator` in `vllm_processor.py`
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
> - Searched for `AsyncEngineCallback|serve_endpoint` in `engine.rs`
> - Read `./lib/runtime/src/storage/kv.rs`
> - Searched for `File` in `discovery`
> - Searched for `fn serve_endpoint|serve_endpoint` in `python`
> - Read `./lib/bindings/python/rust/lib.rs`
> - Read `./lib/bindings/python/rust/lib.rs`
> - Searched for `round_robin_prepared|select_untracked_worker|RoutePicker` in `push_router.rs`
> - Searched for `def worker\(|async def worker\(|WorkerFactory|AsyncLLM` in `main.py`
> - Searched for `chat_completions|ChatCompletions` in `http`
> - Read `./lib/runtime/src/pipeline/network/egress/push_router.rs`
> - Read `./lib/runtime/src/routing_policy/picker.rs`
> - Read `./components/src/dynamo/vllm/main.py`
> - Read `./lib/llm/src/http/service/openai.rs`
> - Searched for `AGGREGATED|DECODE` in `worker_factory.py`
> - Searched for `disaggregation_mode|DisaggregationMode.AGGREGATED` in `config.py`
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
> - Listed files
> - Searched for `AGENTS.md` in `.`
> - Read `./run_server.sh`
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
> <details><summary>Explored 3 files, 3 searches</summary>
>
> - Read `./python/infinilm/llm/llm.py`
> - Read `./test/service/request.py`
> - Searched for `class AsyncLLMEngine|def is_healthy|def add_aborted_req|def stream_request|def add_chat_request` in `llm.py`
> - Read `./python/infinilm/llm/llm.py`
> - Searched for `__main__.py` in `infinilm`
> - Searched for `if __name__ == .__main__.:|def main\(` in `infinilm`
>
> </details>
>
> - Ran `cat python/infinilm/__init__.py; cat python/infinilm/server/__init__.py 2>/dev/null || true; cat python/infinilm/llm/__init__.py | sed -n '1,120p'`
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
> - Ran `git status --short`
> - Ran `chmod +x python/infinilm/cluster/run_cluster.sh && python -m compileall -q python/infinilm/cluster python/infinilm/base_config.py`
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
> - Ran `python -m ruff check python/infinilm/cluster test/service/test_cluster.py`
>
> - Ran `PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
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
> - Ran `PYTHONPATH=python python - <<'PY'
> from infinilm.cluster.config import parse_worker_cluster_args, remove_cluster_worker_args
> args = ['--model', '/m', '--served-model-name', 'Qwen', '--namespace', 'agg', '--worker-port', '18181', '--tp', '1', '--enable-graph']
> options = parse_worker_cluster_args(args)
> remaining = remove_cluster_worker_args(args)
> print(options)
> print(remaining)
> assert remaining == ['--model', '/m', '--tp', '1', '--enable-graph']
> PY`
> - Ran `PYTHONPATH=python python -m infinilm.cluster.frontend --help`
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
> - Ran `git diff --stat && git status --short`
>
> - Ran `python -m ruff check python/infinilm/cluster test/service/test_cluster.py setup.py && git diff --check`
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
> - Ran `bash -n python/infinilm/cluster/run_cluster.sh && python -m compileall -q python/infinilm/cluster python/infinilm/base_config.py`
>
> - Ran `PYTHONPATH=python python -m pytest -q test/service/test_cluster.py`
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
> </details>
>
> CPU/fake-engine 测试、CLI 构建检查、脚本语法和 Ruff 都通过。现在检查 GPU、模型和端口状态；如果 0-2 卡可用，我会按新增脚本启动真实三副本做 OpenAI API 和轮询验证。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader`
> - Ran `test -f /data-aisoft/mechdancer/models/Qwen3-8B/config.json && du -sh /data-aisoft/mechdancer/models/Qwen3-8B`
> - Ran `ss -ltnp | grep -E ':(18000|18081|18082|18083|18181|18182|18183)\b' || true`
>
> </details>
>
> 集群脚本已启动，Frontend 监听 `18000`，三个 worker 正在加载 16GB 模型。我会轮询 readiness 和 worker HTTP health，等三个注册文件全部出现后再发推理请求。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `curl --silent --show-error --fail --retry 20 --retry-delay 3 --retry-connrefused http://127.0.0.1:18000/v1/models/Qwen3-8B/ready`
> - Ran `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader`
> - Ran `find /tmp/infinilm-agg-test/discovery -type f -maxdepth 1 -print -exec cat {} \;`
>
> </details>
>
> 三个 worker 已全部注册 ready，分别占用 GPU 0/1/2，各约 39GB；file discovery 记录中的 HTTP、TCP 端口和 GPU 映射正确。现在先做非流式和流式真实推理，再连续发 6 个请求看 `/metrics` 中三个 worker 是否各 2 次。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `curl --silent --show-error --fail --max-time 180 -N http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Return the single word READY."}],"temperature":0,"max_tokens":32,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'`
> - Ran `curl --silent --show-error --fail --max-time 180 http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"只回复 READY"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'`
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
> <details><summary>Ran 9 commands</summary>
>
> - Ran `curl --silent --show-error --fail --max-time 180 http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Return READY."}],"temperature":0,"max_tokens":8}'`
> - Ran `nohup env PYTHONPATH=/workspace_codex/InfiniLM/python CUDA_VISIBLE_DEVICES=1 python -m infinilm.cluster.worker --model /data-aisoft/mechdancer/models/Qwen3-8B --served-model-name Qwen3-8B --namespace infinilm-agg-test --discovery-backend file --file-kv /tmp/infinilm-agg-test/discovery --request-plane tcp --worker-host 127.0.0.1 --worker-port 18182 --advertise-host 127.0.0.1 --host 127.0.0.1 --port 18082 --device nvidia --tp 1 --block-size 256 --max-new-tokens 2644 --num-blocks 500 --max-batch-size 32 --log-level INFO --enable-paged-attn --attn flash-attn --enable-graph >/tmp/infinilm-gpu1-restart.log 2>&1 & echo $!`
> - Ran `sleep 3; tail -30 /tmp/infinilm-gpu1-restart.log`
> - Ran `for i in $(seq 1 30); do body=$(curl --silent http://127.0.0.1:18000/v1/models/Qwen3-8B/ready); workers=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("workers", 0))'); echo "poll $i workers=$workers"; test "$workers" = 3 && break; sleep 2; done; test "$workers" = 3; printf '%s\n' "$body"`
> - Ran `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader`
> - Ran `ps -p 3876117 -o pid,stat,cmd || true; ls -l /tmp/infinilm-gpu1-restart.log; cat /tmp/infinilm-gpu1-restart.log`
> - Ran `for i in $(seq 1 20); do body=$(curl --silent http://127.0.0.1:18000/v1/models/Qwen3-8B/ready); workers=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("workers", 0))'); echo "poll $i workers=$workers"; test "$workers" = 3 && break; sleep 2; done; test "$workers" = 3; printf '%s\n' "$body"`
> - Ran `curl --silent http://127.0.0.1:18000/metrics`
> - Ran `curl --silent --show-error --fail --max-time 180 http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"Return READY."}],"temperature":0,"max_tokens":8}' | python3 -m json.tool`
>
> </details>
>
> 恢复验证通过：readiness 回到 3，恢复后推理成功，错误计数为 0。我现在停止这轮自测的 frontend、GPU0/2 worker 和单独恢复的 GPU1 worker，然后做最终代码检查和变更汇总。
>
> <details><summary>Ran 5 commands</summary>
>
> - Ran `env PYTHONPATH=/workspace_codex/InfiniLM/python CUDA_VISIBLE_DEVICES=1 python -m infinilm.cluster.worker --model /data-aisoft/mechdancer/models/Qwen3-8B --served-model-name Qwen3-8B --namespace infinilm-agg-test --discovery-backend file --file-kv /tmp/infinilm-agg-test/discovery --request-plane tcp --worker-host 127.0.0.1 --worker-port 18182 --advertise-host 127.0.0.1 --host 127.0.0.1 --port 18082 --device nvidia --tp 1 --block-size 256 --max-new-tokens 2644 --num-blocks 500 --max-batch-size 32 --log-level INFO --enable-paged-attn --attn flash-attn --enable-graph`
> - Ran `PYTHONPATH=/workspace_codex/InfiniLM/python MODEL=/data-aisoft/mechdancer/models/Qwen3-8B GPUS='0 1 2' RUN_DIR=/tmp/infinilm-agg-test NAMESPACE=infinilm-agg-test SERVED_MODEL_NAME=Qwen3-8B python/infinilm/cluster/run_cluster.sh`
> - Ran `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader`
> - Ran `curl --silent --show-error http://127.0.0.1:18000/health || test $? -eq 7`
> - Ran `python -m ruff check python/infinilm/cluster test/service/test_cluster.py setup.py && PYTHONPATH=python python -m pytest -q test/service/test_cluster.py && git diff --check && bash -n python/infinilm/cluster/run_cluster.sh`
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
> - Ran `python -m ruff check python/infinilm/cluster test/service/test_cluster.py setup.py && python -m compileall -q python/infinilm/cluster python/infinilm/base_config.py && bash -n python/infinilm/cluster/run_cluster.sh`
>
> - Ran `git diff -- pyproject.toml setup.py python/infinilm/base_config.py && git status --short`
>
> - Ran `find python/infinilm/cluster -maxdepth 1 -type f -printf '%f\n' | sort; wc -l python/infinilm/cluster/*.py python/infinilm/cluster/run_cluster.sh test/service/test_cluster.py`
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
> - Searched for `infinilm|pip install|editable|develop|-e` in `build.sh`
> - Read `./build.sh`
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

Copied conversation

Source conversation: 01a123ab-9167-7b42-b54f-496c36b308c6

> /workspace_codex/InfiniLM/test/service/test_cluster.py测试脚本该如何使用？

在 InfiniLM 仓库根目录执行即可：

```bash
cd /workspace_codex/InfiniLM
PYTHONPATH=python python -m pytest -q test/service/test_cluster.py
```

如果你已经执行过：

```bash
cd /workspace_codex
bash build.sh --infinilm
```

则可以直接运行：

```bash
cd /workspace_codex/InfiniLM
python -m pytest -q test/service/test_cluster.py
```

正常输出应为：

```text
4 passed
```

这个脚本测试的是集群编排层，不会加载 `Qwen3-8B`，也不会占用 GPU。它使用 fake `AsyncLLMEngine` 和临时目录，实际会启动本机随机端口的 TCP worker/frontend 测试服务。

可用测试项如下：

```bash
# 1. file discovery 注册、快照、TTL 过期和过期文件清理
PYTHONPATH=python python -m pytest -v \
  test/service/test_cluster.py::FileRegistryTest::test_register_snapshot_and_lease_expiry

# 2. round-robin 三副本下 6 个请求分布为 2/2/2
PYTHONPATH=python python -m pytest -v \
  test/service/test_cluster.py::RoundRobinRouterTest::test_three_workers_receive_two_requests_each

# 3. TCP health、chat token 流、finish usage、客户端取消
PYTHONPATH=python python -m pytest -v \
  test/service/test_cluster.py::TcpWorkerServiceTest::test_health_chat_stream_and_cancellation

# 4. frontend OpenAI 流式与非流式接口
PYTHONPATH=python python -m pytest -v \
  test/service/test_cluster.py::ClusterFrontendTest::test_openai_stream_and_non_stream
```

常用方式：

```bash
# 显示详细测试名和结果
PYTHONPATH=python python -m pytest -v test/service/test_cluster.py

# 遇到第一个失败立即停止
PYTHONPATH=python python -m pytest -x test/service/test_cluster.py

# 显示完整异常栈和 print/log 输出
PYTHONPATH=python python -m pytest -s -vv test/service/test_cluster.py
```

也可以不使用 pytest，直接用 `unittest`：

```bash
cd /workspace_codex/InfiniLM
PYTHONPATH=python python test/service/test_cluster.py
```

如果只想在无 GPU 环境里确认集群层逻辑，这个脚本就是推荐入口；真实三副本 GPU 验证则使用：

```bash
cd /workspace_codex/InfiniLM
PYTHONPATH=$PWD/python \
MODEL=/data-aisoft/mechdancer/models/Qwen3-8B \
GPUS="0 1 2" \
SERVED_MODEL_NAME=Qwen3-8B \
python/infinilm/cluster/run_cluster.sh
```

> FileRegistryTest::test_register_snapshot_and_lease_expiry是在测试什么？请分析

`FileRegistryTest::test_register_snapshot_and_lease_expiry` 位于 [test_cluster.py](./InfiniLM/test/service/test_cluster.py:86)，它是在验证 file discovery 注册表的核心生命周期行为：**worker 注册后能被发现；heartbeat lease 过期后会被剔除；过期的注册文件会被清理**。

测试逻辑分四步。

1. **创建临时 file registry**

```python
with TemporaryDirectory() as temporary:
    registry = FileServiceRegistry(Path(temporary), lease_seconds=10)
```

这相当于模拟一个独立的 discovery 目录，例如：

```text
/tmp/infinilm-xxx/
```

`lease_seconds=10` 表示 worker 注册记录有 10 秒租约。生产中 worker 会周期性 heartbeat，heartbeat 会刷新 `heartbeat_at` 和 `lease_expires_at`。

2. **创建并注册一个 worker 记录**

```python
record = registry.new_record(
    namespace="test",
    component="infinilm",
    endpoint="generate",
    worker_type="aggregated",
    model="/models/qwen",
    served_model_name="qwen",
    host="127.0.0.1",
    port=18181,
)

asyncio.run(registry.register(record))
```

这一步会把 worker 元数据写入一个 JSON 文件，内容大致包含：

- namespace
- component
- endpoint
- instance_id
- worker_type
- model
- served_model_name
- TCP host/port
- registered_at
- heartbeat_at
- lease_expires_at
- engine_healthy

然后测试执行：

```python
self.assertEqual(len(asyncio.run(registry.snapshot("test"))), 1)
```

验证点：

- JSON 文件能被正确读取
- 记录能被反序列化成 `WorkerRecord`
- namespace 为 `test` 的 worker 能被 frontend discovery 查询到
- 该 worker 是 `aggregated`、`engine_healthy=true`、lease 未过期，因此是可路由候选

3. **模拟 lease 过期**

```python
expired = replace(record, lease_expires_at=time.time() - 1)
asyncio.run(registry.register(expired))
```

这里不是模拟正常 heartbeat，而是直接把同一条记录的 `lease_expires_at` 改成过去时间，再写回注册表。

它模拟的是生产中的异常情况：

```text
worker 进程 crash
  -> 无法继续 heartbeat
  -> lease_expires_at 不再刷新
  -> frontend 下一次 snapshot 时发现记录过期
```

4. **验证过期 worker 被剔除，文件被删除**

```python
self.assertEqual(asyncio.run(registry.snapshot("test")), [])
self.assertEqual(list(Path(temporary).glob("*.json")), [])
```

这两个断言分别验证：

- `snapshot()` 不再返回过期 worker，因此 router/readiness 不会把它当作可用副本
- 过期 JSON 文件被物理删除，避免长期运行后 discovery 目录堆积死记录

**这个测试对应的生产行为**

如果三个 worker 正常运行：

```text
snapshot("infinilm-agg") -> 3 workers
ready -> true
round-robin -> 3 个候选
```

如果 GPU1 worker 崩溃且不再 heartbeat：

```text
10 秒 lease 过期
snapshot() -> 2 workers
ready -> false 或按策略降级
round-robin -> 只剩 2 个候选
```

如果 GPU1 worker 之后恢复并重新注册：

```text
snapshot() -> 3 workers
ready -> true
round-robin -> 恢复 3 个候选
```

**它没有覆盖的内容**

这个测试主要覆盖 file registry 本身，不覆盖：

- heartbeat 后台任务是否周期执行
- watcher 轮询是否及时刷新
- 多 namespace 隔离
- malformed JSON 文件处理
- frontend readiness 响应
- 真实 TCP worker 连接

这些行为由其他测试和真实 GPU 集成测试覆盖。总体来说，这个测试是 file discovery 控制面的最小闭环验证：**注册可见、租约有效可路由、租约过期不可路由且文件被清理**。

> asyncio.run(registry.register(record))函数是阻塞的么？它是如何运行的？请分析

<details><summary>2 previous messages</summary>

> 我会对照当前 `FileServiceRegistry.register()` 的实现说明 `asyncio.run()` 的执行顺序和阻塞边界，重点区分“调用线程被阻塞”和“事件循环内是否可并发”这两件事。
>
> <details><summary>Explored a file</summary>
>
> - Read `./python/infinilm/cluster/registry.py`
>
> </details>
</details>

`asyncio.run(registry.register(record))` 对当前调用线程来说是**同步等待直到注册完成**的；但 `register()` 内部的文件写入被放到了线程池里执行，因此在事件循环内部是**非阻塞式 async I/O 模型**。这两个说法不矛盾，关键在于看的边界不同。

具体实现是：

- `register()` 定义在 [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:98)
- `_write()` 定义在 [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:90)

核心代码：

```python
async def register(self, record: WorkerRecord) -> WorkerRecord:
    await asyncio.to_thread(self._write, record)
    logger.info(...)
    return record
```

`_write()` 做的是同步文件操作：

```python
def _write(self, record: WorkerRecord) -> None:
    self.root.mkdir(parents=True, exist_ok=True)
    destination = self._path(record)

    temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
    payload = json.dumps(asdict(record), separators=(",", ":"), sort_keys=True)

    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
```

也就是说，它写的是临时文件，然后用 `os.replace()` 原子替换成正式 JSON 文件，避免 frontend 读到半截 JSON。

**`asyncio.run()` 如何执行**

测试里的调用是：

```python
asyncio.run(registry.register(record))
```

执行过程如下：

```text
当前同步测试线程
  |
  | registry.register(record)
  |   只创建 coroutine 对象，此时函数体还没执行
  |
  | asyncio.run(...)
  |   1. 创建一个新的 event loop
  |   2. 把 coroutine 包装成 Task
  |   3. 运行 event loop，直到 coroutine 完成
  |   4. 关闭 event loop
  |
  v
返回 registry.register(record) 的结果
```

所以从测试线程看，这行代码会阻塞，直到：

1. `_write()` 完成
2. logger 输出完成
3. coroutine 返回

测试中的下一条语句才会执行：

```python
self.assertEqual(len(asyncio.run(registry.snapshot("test"))), 1)
```

**为什么还要用 `asyncio.to_thread()`**

如果 `register()` 直接写成：

```python
async def register(self, record):
    self._write(record)
```

那么在生产环境的 uvicorn event loop 中，磁盘 I/O 会直接占用事件循环线程。文件慢时，HTTP 请求、TCP worker、heartbeat、watcher 都可能被拖住。

当前实现：

```python
await asyncio.to_thread(self._write, record)
```

含义是：

```text
当前 event loop
  |
  | 把同步的 _write() 交给默认 ThreadPoolExecutor
  |
  | await 期间 event loop 可以运行其他 coroutine
  |
  v
线程完成文件写入后，恢复 register()
```

因此在生产 worker 中：

```python
await service.start()
```

内部会：

```python
await registry.register(record)
```

这个 `await` 期间，uvicorn event loop 仍可处理其他任务。真正阻塞的是线程池里的一个工作线程，不是 event loop 主线程。

**两种使用场景的区别**

在单元测试中：

```python
asyncio.run(registry.register(record))
```

每个 `asyncio.run()` 都会创建一个独立的新 event loop，适合同步测试代码。它不是长期运行的异步服务，只是等待这一个 coroutine 完成。

在生产代码中不会这样写，而是：

```python
await registry.register(record)
```

因为它已经在 uvicorn 或 `TcpWorkerService` 的 event loop 里运行。

不能在已有 async 函数里写：

```python
async def start():
    asyncio.run(registry.register(record))  # 错误用法
```

原因是当前线程已经有 event loop，`asyncio.run()` 会报类似：

```text
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**总结**

`asyncio.run(registry.register(record))` 的行为可以概括为：

- `registry.register(record)` 本身只是创建 coroutine，不执行函数体。
- `asyncio.run()` 创建临时 event loop 并同步等待该 coroutine 完成。
- `register()` 内部把磁盘写入交给 `asyncio.to_thread()`。
- 文件写入发生在线程池工作线程上，不阻塞生产 event loop。
- 对调用 `asyncio.run()` 的测试线程来说，整体调用是阻塞的。
- 这种写法只适合单元测试；生产路径中应使用 `await registry.register(...)`。

> 将上述解释整理添加到代码的中文注释中，包括：
> (1) `asyncio.run(registry.register(record))` 对当前调用线程来说是**同步等待直到注册完成**的  。
> (2) `_write`  函数写的是临时文件，然后用 `os.replace()` 原子替换成正式 JSON 文件，避免 frontend 读到半截 JSON。 &#x20;
> (3) asyncio.run(registry.register(record))的执行过程的流程图。
> (4) 为什么还要用 `asyncio.to_thread()`  。
> 请添加上述四点注释信息。

<details><summary>2 previous messages</summary>

> 我会把说明放在两处最贴近语义的位置：`registry.py` 的 `_write()`/`register()` 说明文件原子写和 `to_thread`，测试用例处说明 `asyncio.run()` 对同步测试线程的执行流程图。然后跑 Ruff 和相关测试。
>
> <details><summary>Explored a search, ran a command</summary>
>
> - Ran ```apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/registry.py
> @@
>      def _write(self, record: WorkerRecord) -> None:
> +        """Synchronously write one record through an atomic file replacement.
> +
> +        先写入同目录下的 ``.tmp`` 临时文件，再用 ``os.replace()`` 把临时文件
> +        原子替换成正式 JSON 文件。frontend watcher 只读取 ``*.json``，因此不会
> +        读到半截 JSON；写入失败时也只会留下临时文件，不会破坏旧注册记录。
> +        """
>          self.root.mkdir(parents=True, exist_ok=True)
>          destination = self._path(record)
>          temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
> @@
>      async def register(self, record: WorkerRecord) -> WorkerRecord:
> +        """Register a worker without blocking the serving event loop.
> +
> +        ``_write()`` contains synchronous disk operations. In the production
> +        worker it is called from the uvicorn event loop, so this method uses
> +        ``asyncio.to_thread()`` to move the disk write to a thread-pool worker.
> +        While that worker thread performs the file operation, the event loop can
> +        continue serving HTTP/TCP traffic and running heartbeat/watcher tasks.
> +
> +        Tests may call ``asyncio.run(registry.register(record))``. That call is
> +        synchronous from the test thread: it creates a temporary event loop,
> +        runs this coroutine to completion, and then closes the loop.
> +        """
>          await asyncio.to_thread(self._write, record)
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>              record = registry.new_record(
> @@
>                      port=18181,
>                  )
> +
> +            # ``registry.register(record)`` only creates a coroutine object; its
> +            # body has not started yet. ``asyncio.run()`` is synchronous from
> +            # this test thread and runs that coroutine to completion:
> +            #
> +            # test thread
> +            #   -> registry.register(record)
> +            #      create coroutine object, do not execute body
> +            #   -> asyncio.run(...)
> +            #      create temporary event loop
> +            #      run register() until _write() and logging finish
> +            #      close temporary event loop
> +            #   -> return to the next assertion
> +            #
> +            # Production async code instead uses ``await registry.register()``
> +            # because it is already running inside the frontend/worker loop.
>              asyncio.run(registry.register(record))
> *** End Patch
> PATCH
>
> python -m ruff check python/infinilm/cluster/registry.py test/service/test_cluster.py && PYTHONPATH=python python -m pytest -q test/service/test_cluster.py::FileRegistryTest::test_register_snapshot_and_lease_expiry```
>
> <details><summary>Explored a search</summary>
>
> - Searched for `Synchronously write|Register a worker|test thread|create temporary event loop` in `registry.py`
>
> </details>
>
> </details>
</details>

已将四点说明添加为中文代码注释：

1. `_write()` 的临时文件写入和 `os.replace()` 原子替换说明，避免 frontend 读到半截 JSON：
   [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:90)

2. `register()` 为什么使用 `asyncio.to_thread()`，以及生产 event loop 不被磁盘 I/O 阻塞的原因：
   [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:105)

3. `asyncio.run(registry.register(record))` 对当前测试线程是同步等待直到注册完成的说明：
   [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:113)

4. `asyncio.run()` 的执行流程图已添加到测试调用处，包括“创建 coroutine 对象但不执行函数体”“创建临时 event loop”“执行到完成”“关闭 loop”“返回下一条断言”：
   [test_cluster.py](./InfiniLM/test/service/test_cluster.py:101)

验证结果：

```text
ruff: passed
FileRegistryTest::test_register_snapshot_and_lease_expiry: 1 passed
```

> 需要添加中文注释。请修改。

<details><summary>2 previous messages</summary>

> 我会把刚添加的两处说明和测试流程图全部改为中文注释，保留代码行为不变，然后重新跑 Ruff 和对应测试。
>
> <details><summary>Ran a command</summary>
>
> - Ran ```apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/registry.py
> @@
>      def _write(self, record: WorkerRecord) -> None:
> -        """Synchronously write one record through an atomic file replacement.
> +        """通过原子文件替换同步写入一条 worker 注册记录。
>  
> -        先写入同目录下的 ``.tmp`` 临时文件，再用 ``os.replace()`` 把临时文件
> -        原子替换成正式 JSON 文件。frontend watcher 只读取 ``*.json``，因此不会
> -        读到半截 JSON；写入失败时也只会留下临时文件，不会破坏旧注册记录。
> +        先写入同目录下的 ``.tmp`` 临时文件，再用 ``os.replace()`` 把临时文件
> +        原子替换成正式 JSON 文件。frontend watcher 只读取 ``*.json``，因此不会
> +        读到半截 JSON；写入失败时也只会留下临时文件，不会破坏旧注册记录。
>          """
> @@
>      async def register(self, record: WorkerRecord) -> WorkerRecord:
> -        """Register a worker without blocking the serving event loop.
> +        """注册 worker，同时避免阻塞服务端 event loop。
>  
> -        ``_write()`` contains synchronous disk operations. In the production
> -        worker it is called from the uvicorn event loop, so this method uses
> -        ``asyncio.to_thread()`` to move the disk write to a thread-pool worker.
> -        While that worker thread performs the file operation, the event loop can
> -        continue serving HTTP/TCP traffic and running heartbeat/watcher tasks.
> +        ``_write()`` 包含同步磁盘操作。生产环境的 worker 会从 uvicorn
> +        event loop 调用注册逻辑，因此这里使用 ``asyncio.to_thread()``
> +        把文件写入交给线程池。工作线程写文件期间，event loop 仍可继续
> +        处理 HTTP/TCP 请求，以及 heartbeat/watcher 等后台任务。
>  
> -        Tests may call ``asyncio.run(registry.register(record))``. That call is
> -        synchronous from the test thread: it creates a temporary event loop,
> -        runs this coroutine to completion, and then closes the loop.
> +        测试代码可以调用 ``asyncio.run(registry.register(record))``。
> +        对当前测试线程来说，这个调用是同步等待：它会创建临时 event loop，
> +        执行当前 coroutine 直到注册完成，然后关闭该 event loop。
>          """
> *** End Patch
> PATCH
>
> apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
> -            # ``registry.register(record)`` only creates a coroutine object; its
> -            # body has not started yet. ``asyncio.run()`` is synchronous from
> -            # this test thread and runs that coroutine to completion:
> +            # ``registry.register(record)`` 只会创建 coroutine 对象，此时
> +            # 函数体还没有开始执行。对当前测试线程来说，
> +            # ``asyncio.run()`` 会同步等待 coroutine 执行完成：
> @@
> -            # test thread
> +            # 测试线程
>              #   -> registry.register(record)
> -            #      create coroutine object, do not execute body
> +            #      创建 coroutine 对象，但不执行函数体
>              #   -> asyncio.run(...)
> -            #      create temporary event loop
> -            #      run register() until _write() and logging finish
> -            #      close temporary event loop
> -            #   -> return to the next assertion
> +            #      创建临时 event loop
> +            #      执行 register()，直到 _write() 和日志输出完成
> +            #      关闭临时 event loop
> +            #   -> 返回测试函数，继续执行下一条断言
> @@
> -            # Production async code instead uses ``await registry.register()``
> -            # because it is already running inside the frontend/worker loop.
> +            # 生产环境中的 async 代码已经在 frontend/worker 的 event loop 内，
> +            # 因此应使用 ``await registry.register()``，而不是再次调用
> +            # ``asyncio.run()``。
>              asyncio.run(registry.register(record))
> *** End Patch
> PATCH
>
> python -m ruff check python/infinilm/cluster/registry.py test/service/test_cluster.py && PYTHONPATH=python python -m pytest -q test/service/test_cluster.py::FileRegistryTest::test_register_snapshot_and_lease_expiry```
>
> </details>
</details>

已改为全中文注释：

- [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:90)：说明 `_write()` 使用临时文件和 `os.replace()` 原子替换，避免 frontend 读到半截 JSON。
- [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:105)：说明为什么使用 `asyncio.to_thread()`，以及生产 event loop 如何保持响应。
- [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:114)：说明 `asyncio.run()` 对当前测试线程是同步等待。
- [test_cluster.py](./InfiniLM/test/service/test_cluster.py:101)：`asyncio.run()` 执行流程图已全部改为中文，并说明生产环境应使用 `await registry.register()`。

验证通过：

```text
ruff: passed
FileRegistryTest::test_register_snapshot_and_lease_expiry: 1 passed
```

> 在FileRegistryTest中，想要打印temporary的路径。请完善代码

<details><summary>6 previous messages</summary>

> 我会在 `FileRegistryTest` 中直接打印 `TemporaryDirectory()` 返回的路径，并在注册后打印实际生成的 JSON 文件路径，这样既能看到根目录，也能确认注册文件落盘位置。
>
> <details><summary>Ran a command</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>      def test_register_snapshot_and_lease_expiry(self):
>          with TemporaryDirectory() as temporary:
> +            print(f"temporary directory: {temporary}")
>              registry = FileServiceRegistry(Path(temporary), lease_seconds=10)
> @@
>              asyncio.run(registry.register(record))
> +            print(f"registered record file: {Path(temporary) / '*.json'}")
>              self.assertEqual(len(asyncio.run(registry.snapshot("test"))), 1)
> *** End Patch
> PATCH
>
> PYTHONPATH=python python -m pytest -s -q test/service/test_cluster.py::FileRegistryTest::test_register_snapshot_and_lease_expiry`
>
> </details>
>
> 测试文件里已经有人加入了 `logger.info(...)`，但测试模块没有定义 `logger`，导致当前测试失败。我会保留打印意图，改成直接 `print()` 并补充实际注册文件路径打印，然后重新验证。
>
> <details><summary>Explored a file, ran a command</summary>
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
>      def test_register_snapshot_and_lease_expiry(self):
>          with TemporaryDirectory() as temporary:
> -
> -            logger.info(
> -                "temporary path %s",
> -                 temporary
> -            )
> -            
> +            # 打印本次测试使用的 discovery 根目录；with 块结束后会自动删除。
> +            print(f"temporary directory: {temporary}")
>              registry = FileServiceRegistry(Path(temporary), lease_seconds=10)
> @@
>              asyncio.run(registry.register(record))
> +            record_files = list(Path(temporary).glob("*.json"))
> +            print(f"registered record files: {[str(path) for path in record_files]}")
>              self.assertEqual(len(asyncio.run(registry.snapshot("test"))), 1)
> *** End Patch
> PATCH
>
> python -m ruff check test/service/test_cluster.py && PYTHONPATH=python python -m pytest -s -q test/service/test_cluster.py::FileRegistryTest::test_register_snapshot_and_lease_expiry`
>
> </details>
>
> 当前文件里又新增了 `time.sleep(100)`，这会让单测固定卡 100 秒。我会保留“便于检查临时目录”的意图，改成通过环境变量 `INFINILM_TEST_PAUSE_SECONDS` 控制可选暂停，默认 0；同时打印注册文件路径，测试仍可快速结束。
>
> <details><summary>Explored a file, ran a command</summary>
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
>  import asyncio
>  import json
> +import os
>  import time
> @@
>              asyncio.run(registry.register(record))
>              record_files = list(Path(temporary).glob("*.json"))
>              print(f"registered record files: {[str(path) for path in record_files]}")
> +
> +            # 临时目录会在 with 块结束时自动删除。需要人工检查时设置
> +            # INFINILM_TEST_PAUSE_SECONDS，例如：
> +            # INFINILM_TEST_PAUSE_SECONDS=30 python -m pytest ... -s
> +            pause_seconds = float(os.getenv("INFINILM_TEST_PAUSE_SECONDS", "0"))
> +            if pause_seconds > 0:
> +                print(f"pausing for {pause_seconds} seconds to inspect files")
> +                time.sleep(pause_seconds)
> +
>              self.assertEqual(len(asyncio.run(registry.snapshot("test"))), 1)
> -            import time 
> -            time.sleep(100)
>  
>              expired = replace(record, lease_expires_at=time.time() - 1)
> *** End Patch
> PATCH
>
> python -m ruff check test/service/test_cluster.py && PYTHONPATH=python python -m pytest -s -q test/service/test_cluster.py::FileRegistryTest::test_register_snapshot_and_lease_expiry`
>
> </details>
</details>

已在 `FileRegistryTest` 中完善临时路径打印：

- 打印 discovery 根目录：[test_cluster.py](./InfiniLM/test/service/test_cluster.py:89)
- 注册成功后打印实际 JSON 注册文件路径：[test_cluster.py](./InfiniLM/test/service/test_cluster.py:121)
- 将你加入的固定 `time.sleep(100)` 改成可控暂停，通过环境变量设置，默认不暂停：[test_cluster.py](./InfiniLM/test/service/test_cluster.py:124)

运行命令：

```bash
cd /workspace_codex/InfiniLM
PYTHONPATH=python python -m pytest -s -q \
  test/service/test_cluster.py::FileRegistryTest::test_register_snapshot_and_lease_expiry
```

本次输出示例：

```text
temporary directory: /tmp/tmp4y4v9g9_
registered record files: ['/tmp/tmp4y4v9g9_/test.infinilm.generate.4055454-5422f3ddd770.json']
.
1 passed
```

如果需要人工检查临时目录，可以暂停 30 秒：

```bash
INFINILM_TEST_PAUSE_SECONDS=30 \
PYTHONPATH=python python -m pytest -s -q \
  test/service/test_cluster.py::FileRegistryTest::test_register_snapshot_and_lease_expiry
```

验证结果：`ruff passed`，该测试 `1 passed`。

> lease_expires_at 和 registered_at这两个时间分别是什么意思？过期文件再哪里被删除的？请分析。

<details><summary>2 previous messages</summary>

> 我会结合 `WorkerRecord` 的生成、heartbeat 刷新和 `snapshot()` 检查逻辑，说明两个时间字段的作用以及过期文件的清理位置。
>
> <details><summary>Explored a file</summary>
>
> - Read `./python/infinilm/cluster/registry.py`
>
> </details>
</details>

`registered_at` 和 `lease_expires_at` 是 worker 注册文件里的两个时间戳，单位都是 Unix epoch 秒，也就是 `time.time()` 返回的秒级浮点数。

**1. `registered_at`：首次注册时间**

定义在 [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:32)，赋值在 [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:64)：

```python
now = time.time()
...
registered_at=now,
heartbeat_at=now,
lease_expires_at=now + self.lease_seconds,
```

含义：

```text
registered_at = 这个 worker 实例第一次创建注册记录的时间
```

例如：

```json
"registered_at": 1791613868.53
```

表示这个 worker 是该时间点注册的。

它的主要作用是诊断和观测：

- 判断 worker 是什么时候加入集群的
- 排查 worker 是否重启过
- 对比多个 worker 的加入时间
- 作为注册记录的历史信息

当前代码中，`registered_at` 不参与过期判断，也不会在 heartbeat 时更新。worker 重启后会生成新的 `instance_id` 和新的 `registered_at`。

**2. `lease_expires_at`：注册租约过期时间**

定义在 [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:34)，初始值在 [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:77)：

```python
lease_expires_at = now + self.lease_seconds
```

如果 `lease_seconds=10`，则：

```text
lease_expires_at = 注册时间 + 10 秒
```

含义：

```text
在这个时间点之前，注册记录有效；
超过这个时间点，认为 worker 没有 heartbeat，注册记录过期。
```

它和 `heartbeat_at` 的关系在 [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:126)：

```python
now = time.time()
updated = replace(
    record,
    heartbeat_at=now,
    lease_expires_at=now + self.lease_seconds,
    engine_healthy=healthy,
)
```

也就是说，每次 heartbeat 会同时更新：

```text
heartbeat_at      = 这次心跳时间
lease_expires_at  = 这次心跳时间 + TTL
```

例如 `lease_seconds=10`：

```text
10:00:00 heartbeat
  heartbeat_at = 10:00:00
  lease_expires_at = 10:00:10

10:00:02 heartbeat
  heartbeat_at = 10:00:02
  lease_expires_at = 10:00:12

10:00:04 heartbeat
  heartbeat_at = 10:00:04
  lease_expires_at = 10:00:14
```

只要 worker 还活着并持续 heartbeat，`lease_expires_at` 就一直被向后推。

如果 worker 进程 crash，heartbeat 停止：

```text
最后一次 heartbeat: 10:00:04
TTL: 10 秒
lease_expires_at: 10:00:14

10:00:14 后 frontend snapshot 会认为它过期
```

**过期文件在哪里删除**

删除位置在 `FileServiceRegistry._snapshot()`，具体代码是 [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:180)：

```python
now = time.time()

for path in self.root.glob("*.json"):
    ...
    if record.lease_expires_at < now:
        path.unlink(missing_ok=True)
        continue

    records.append(record)
```

核心判断：

```python
record.lease_expires_at < now
```

即：

```text
租约过期时间 < 当前时间
```

如果成立，就执行：

```python
path.unlink(missing_ok=True)
```

这会删除当前过期的 JSON 注册文件。

例如：

```text
lease_expires_at = 10:00:14
当前时间 now     = 10:00:20

10:00:14 < 10:00:20
=> 记录过期
=> 删除该 worker 的 JSON 文件
=> 不加入可用 worker 列表
```

注意，`snapshot()` 是通过 `asyncio.to_thread()` 调用 `_snapshot()` 的，见 [registry.py](./InfiniLM/python/infinilm/cluster/registry.py:187)，所以实际删除发生在线程池工作线程里，避免阻塞 frontend event loop。

**frontend 什么时候触发删除**

frontend 的 `FileServiceWatcher` 会周期性调用：

```python
await self.registry.snapshot(namespace)
```

因此周期性 snapshot 时会顺带清理过期文件。

对应调用链是：

```text
FileServiceWatcher.run()
  -> self.refresh(namespace)
      -> self.registry.snapshot(namespace)
          -> asyncio.to_thread(self._snapshot, ...)
              -> 检查 lease_expires_at
              -> path.unlink(missing_ok=True)
```

所以它不是操作系统定时器自动删除，而是 frontend watcher 轮询 discovery 目录时由 `snapshot()` 主动清理。

**总结对比**

| 字段 | 含义 | 初始值 | heartbeat 是否更新 | 是否决定过期 |
| --- | --- | --- | --- | --- |
| `registered_at` | worker 首次注册时间 | `time.time()` | 否 | 否 |
| `heartbeat_at` | 最近一次心跳时间 | `time.time()` | 是 | 间接说明 |
| `lease_expires_at` | 注册租约截止时间 | `now + lease_seconds` | 是 | 是 |

测试里这句：

```python
expired = replace(record, lease_expires_at=time.time() - 1)
```

就是把租约截止时间手动改成过去，模拟 worker 停止 heartbeat 后租约过期的场景。随后：

```python
asyncio.run(registry.snapshot("test"))
```

会进入 `_snapshot()`，发现：

```text
lease_expires_at < now
```

然后删除该 JSON 文件并返回空列表。

> 添加中文注释，包括：`registered_at`  ，`heartbeat_at`  ，`lease_expires_at`  的含义；lease_expires_at的更新；文件的删除；frontend 的 `FileServiceWatcher` 会周期性  snapshot。请添加。

<details><summary>2 previous messages</summary>

> 我会把时间戳说明加在 `WorkerRecord` 字段和 heartbeat 逻辑处，把租约删除说明加在 `_snapshot()`，并把 frontend 周期刷新说明加在 `FileServiceWatcher.run()`。
>
> <details><summary>Ran a command</summary>
>
> - Ran ```apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/python/infinilm/cluster/registry.py
> @@
>  class WorkerRecord:
> @@
> -    registered_at: float
> -    heartbeat_at: float
> -    lease_expires_at: float
> +    # worker 实例首次创建注册记录的时间。该字段用于观测和排查，例如判断
> +    # worker 何时加入集群；heartbeat 时不会更新，也不参与过期判断。
> +    registered_at: float
> +
> +    # 最近一次 heartbeat 的时间。worker 正常运行时会周期性刷新该时间。
> +    # 它用于人工观察最近心跳，过期判断使用的是 lease_expires_at。
> +    heartbeat_at: float
> +
> +    # 当前注册租约的截止时间，由 heartbeat_at + lease_seconds 得到。
> +    # frontend snapshot 时如果发现 lease_expires_at < 当前时间，就认为
> +    # worker 已停止 heartbeat，该注册记录过期且不可路由。
> +    lease_expires_at: float
> @@
>      async def heartbeat(
> @@
>      ) -> WorkerRecord:
> +        """刷新 worker 心跳和注册租约。
> +
> +        每次 heartbeat 都会把 ``heartbeat_at`` 更新为当前时间，并把
> +        ``lease_expires_at`` 更新为 ``当前时间 + lease_seconds``。
> +        因此只要 worker 存活并持续 heartbeat，注册租约就会一直被顺延；
> +        worker crash 后 heartbeat 停止，lease_expires_at 不再更新，
> +        最终会被 frontend snapshot 判定为过期。
> +        """
>          now = time.time()
> @@
>      def _snapshot(
> @@
>          if not self.root.exists():
>              return []
>  
>          now = time.time()
>          records: list[WorkerRecord] = []
> +
>          for path in self.root.glob("*.json"):
> @@
>              if not record.engine_healthy:
>                  continue
> +
> +            # 文件删除发生在 frontend 周期性调用 snapshot() 的过程中：
> +            # 如果注册租约已过期，说明 worker 没有继续 heartbeat。
> +            # 这里直接删除对应 JSON 文件，避免长期运行后 discovery 目录
> +            # 堆积已失效的 worker 记录，同时不会把该 worker 返回给路由器。
>              if record.lease_expires_at < now:
>                  path.unlink(missing_ok=True)
>                  continue
> @@
>      async def run(self, namespace: str, interval: float) -> None:
> +        """周期性 snapshot file registry，维护 frontend 可见的 worker 集合。
> +
> +        frontend 的 FileServiceWatcher 不是被动等待文件事件，而是按 interval
> +        轮询 registry.snapshot()。每次 snapshot 都会读取 JSON 记录、过滤可用
> +        worker，并清理已经超过 lease_expires_at 的过期注册文件。
> +        """
>          while not self._stop.is_set():
> *** End Patch
> PATCH
>
> python -m ruff check python/infinilm/cluster/registry.py && PYTHONPATH=python python -m pytest -q test/service/test_cluster.py::FileRegistryTest::test_register_snapshot_and_lease_expiry```
>
> </details>
</details>

已添加中文注释，覆盖以下内容：

- `registered_at`、`heartbeat_at`、`lease_expires_at` 三个字段的含义和是否参与过期判断：[registry.py](./InfiniLM/python/infinilm/cluster/registry.py:32)
- heartbeat 如何更新 `heartbeat_at` 和 `lease_expires_at`，以及 worker crash 后租约停止顺延：[registry.py](./InfiniLM/python/infinilm/cluster/registry.py:145)
- `_snapshot()` 中在哪里删除过期 JSON 文件，为什么删除后不会返回给 router：[registry.py](./InfiniLM/python/infinilm/cluster/registry.py:210)
- frontend `FileServiceWatcher` 如何按 interval 周期性 snapshot，并在 snapshot 中过滤和清理过期注册：[registry.py](./InfiniLM/python/infinilm/cluster/registry.py:238)

验证通过：

```text
ruff: passed
FileRegistryTest::test_register_snapshot_and_lease_expiry: 1 passed
```

> RoundRobinRouterTest::test_three_workers_receive_two_requests_each测试是在做什么

<details><summary>2 previous messages</summary>

> 我会结合当前测试代码和 `RoundRobinRouter.select()` 的实现，说明它验证的是路由 cursor 的均匀分配，而不是真实网络请求。
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./test/service/test_cluster.py`
> - Read `./python/infinilm/cluster/router.py`
>
> </details>
</details>

`RoundRobinRouterTest::test_three_workers_receive_two_requests_each` 位于 [test_cluster.py](./InfiniLM/test/service/test_cluster.py:140)，它是在验证 **round-robin 路由器在 3 个可用 worker 之间连续选择 6 次时，每个 worker 会被均匀选中 2 次**。

这是一个纯单元测试，不启动 TCP、不启动 GPU、不调用真实 `AsyncLLMEngine`，只测试 `RoundRobinRouter.select()` 这个选择函数。

**测试构造**

```python
watcher = type("Watcher", (), {"workers": []})()
```

这里动态创建了一个非常简单的 fake watcher 对象，只提供：

```python
watcher.workers
```

真实 `FileServiceWatcher` 会从 file discovery 中读取可用 worker；但这个测试只关心路由算法，所以直接把 `workers` 设置为三个字符串：

```python
watcher.workers = [f"worker-{index}" for index in range(3)]
```

也就是：

```text
worker-0
worker-1
worker-2
```

这里用字符串代替真实的 `WorkerRecord`，因为 `select()` 本身只要求列表非空，并按下标取值。

然后创建 router：

```python
router = RoundRobinRouter(
    watcher,
    connect_timeout=0.2,
    request_timeout=1.0,
)
```

`connect_timeout` 和 `request_timeout` 在这个测试里不会真正使用，因为测试没有调用 `stream_chat()`。

**核心执行**

初始化计数表：

```python
counts = {worker: 0 for worker in watcher.workers}
```

即：

```python
{
    "worker-0": 0,
    "worker-1": 0,
    "worker-2": 0,
}
```

然后连续调用 6 次 `select()`：

```python
for _ in range(6):
    counts[router.select(watcher.workers)] += 1
```

`RoundRobinRouter.select()` 的实现在 [router.py](./InfiniLM/python/infinilm/cluster/router.py:42)：

```python
def select(self, workers: list[WorkerRecord]) -> WorkerRecord:
    if not workers:
        raise NoWorkersAvailable(...)

    index = self.cursor % len(workers)
    self.cursor += 1
    return workers[index]
```

初始状态：

```text
cursor = 0
workers = [worker-0, worker-1, worker-2]
```

6 次选择过程是：

| 次数 | cursor 取模前 | index | 选中的 worker | 选择后 cursor |
| --- | ---: | ---: | --- | ---: |
| 1 | 0 | 0 | `worker-0` | 1 |
| 2 | 1 | 1 | `worker-1` | 2 |
| 3 | 2 | 2 | `worker-2` | 3 |
| 4 | 3 | 0 | `worker-0` | 4 |
| 5 | 4 | 1 | `worker-1` | 5 |
| 6 | 5 | 2 | `worker-2` | 6 |

因此结果是：

```python
{
    "worker-0": 2,
    "worker-1": 2,
    "worker-2": 2,
}
```

**最终断言**

```python
self.assertEqual(list(counts.values()), [2, 2, 2])
```

注意 Python 3.7 之后 dict 保持插入顺序，因此 `counts.values()` 的顺序就是：

```text
worker-0 的次数
worker-1 的次数
worker-2 的次数
```

断言等价于：

```text
worker-0 处理 2 次
worker-1 处理 2 次
worker-2 处理 2 次
```

**它验证了什么**

1. **轮询顺序是循环的**
   请求不会被全部发给第一个 worker，而是依次分给不同 worker。

2. **3 个 worker 时分配是均匀的**
   6 次请求刚好是两轮 round-robin，每个 worker 2 次。

3. **cursor 是 router 状态并会递增**
   如果 cursor 每次都被重置为 0，所有请求都会选择 `worker-0`，测试会失败。

4. **候选列表为空时会抛错**
   这一点由 `select()` 的 `NoWorkersAvailable` 逻辑保证，但当前测试没有单独覆盖空列表场景。

**它没有验证什么**

这个测试不验证：

- file discovery 是否返回正确 worker
- worker 是否健康
- TCP 连接是否成功
- 真实推理是否成功
- worker 故障后是否切换
- worker 数量变化时分布如何变化
- 并发请求下的线程/协程安全性

这些属于 `TcpWorkerServiceTest`、`ClusterFrontendTest` 和真实 GPU 集成测试的职责。

**对应生产行为**

如果 frontend 发现 3 个可用 worker：

```text
GPU0 worker
GPU1 worker
GPU2 worker
```

连续来 6 个请求，期望路由是：

```text
请求 1 -> worker A
请求 2 -> worker B
请求 3 -> worker C
请求 4 -> worker A
请求 5 -> worker B
请求 6 -> worker C
```

因此该测试就是三副本聚合式推理服务中 round-robin 负载均衡算法的最小正确性验证。

> 为RoundRobinRouterTest::test_three_workers_receive_two_requests_each中添加简洁的中文注释。

<details><summary>2 previous messages</summary>

> 我会在该测试的 fake watcher、候选列表、6 次选择和最终断言处添加简洁中文注释，保持测试逻辑不变。
>
> <details><summary>Ran a command</summary>
>
> - Ran `apply_patch <<'PATCH'
> *** Begin Patch
> *** Update File: /workspace_codex/InfiniLM/test/service/test_cluster.py
> @@
>  class RoundRobinRouterTest(unittest.TestCase):
>      def test_three_workers_receive_two_requests_each(self):
> +        # 使用只带 workers 属性的 fake watcher，隔离 file discovery，
> +        # 只验证 RoundRobinRouter 自身的选择算法。
>          watcher = type("Watcher", (), {"workers": []})()
>          router = RoundRobinRouter(
>              watcher,
> @@
>              request_timeout=1.0,
>          )
> +
> +        # 模拟 3 个可用 aggregated worker。
>          watcher.workers = [f"worker-{index}" for index in range(3)]
>          counts = {worker: 0 for worker in watcher.workers}
> +
> +        # 连续路由 6 个请求，相当于每个 worker 应轮询到 2 次。
>          for _ in range(6):
>              counts[router.select(watcher.workers)] += 1
> +
> +        # 验证 3 个 worker 的选择次数依次为 2/2/2。
>          self.assertEqual(list(counts.values()), [2, 2, 2])
> *** End Patch
> PATCH
>
> python -m ruff check test/service/test_cluster.py && PYTHONPATH=python python -m pytest -q test/service/test_cluster.py::RoundRobinRouterTest::test_three_workers_receive_two_requests_each`
>
> </details>
</details>

已为 `RoundRobinRouterTest::test_three_workers_receive_two_requests_each` 添加简洁中文注释，位置在 [test_cluster.py](./InfiniLM/test/service/test_cluster.py:140)。

注释覆盖四点：

- fake watcher 用于隔离 file discovery，只测试路由算法
- 模拟 3 个可用 aggregated worker
- 连续 6 个请求相当于每 worker 轮询 2 次
- 断言选择次数应为 `2/2/2`

验证结果：

```text
ruff: passed
RoundRobinRouterTest::test_three_workers_receive_two_requests_each: 1 passed
```

> TcpWorkerServiceTest::test_health_chat_stream_and_cancellation是在测试什么？请分析。

<details><summary>2 previous messages</summary>

> 我会按测试的三个阶段分析：TCP health 握手、真实 TCP 连接上的 fake token 流、以及客户端取消是否传播到 engine 的 abort 队列。先读取当前测试代码和对应 worker 实现，确保解释与最新注释一致。
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./test/service/test_cluster.py`
> - Read `./python/infinilm/cluster/worker_service.py`
>
> </details>
</details>

`TcpWorkerServiceTest::test_health_chat_stream_and_cancellation` 位于 [test_cluster.py](./InfiniLM/test/service/test_cluster.py:163)，它是在测试 **InfiniLM 集群 worker 的 raw TCP 数据面闭环**，重点包括：

1. worker TCP 服务能否启动
2. health 协议是否正确
3. chat 请求能否通过 TCP 进入 worker 并调用 engine
4. token 输出能否按协议流式返回
5. finish 帧和 usage 是否正确
6. 客户端取消请求时，worker 是否调用现有 engine 的 abort 机制

这个测试不加载真实模型、不占用 GPU、不启动 frontend，而是使用 `FakeEngine` 模拟 `AsyncLLMEngine` 的行为，实际测试的是 `TcpWorkerService`、`TcpWorkerClient` 和 TCP 协议层。

**1. 构造一个临时 worker**

测试首先创建临时目录和 file registry：

```python
with TemporaryDirectory() as temporary:
    registry = FileServiceRegistry(Path(temporary))
```

然后创建一个 `WorkerRecord`：

```python
record = registry.new_record(
    namespace="test",
    component="infinilm",
    endpoint="generate",
    worker_type="aggregated",
    model="/models/qwen",
    served_model_name="qwen",
    host="127.0.0.1",
    port=0,
)
```

几个关键点：

- `namespace="test"`：模拟一个独立集群 namespace。
- `worker_type="aggregated"`：表示该 worker 是聚合式推理副本，prefill 和 decode 都在同一个 worker 内完成。
- `host="127.0.0.1"`：单机测试。
- `port=0`：让操作系统分配一个随机空闲端口，避免测试端口冲突。

接着创建 fake engine 和 fake inference server：

```python
engine = FakeEngine()
service = TcpWorkerService(
    engine=engine,
    inference_server=FakeInferenceServer(),
    registry=registry,
    record=record,
)
```

`FakeEngine` 模拟真实 `AsyncLLMEngine` 的这些接口：

```python
is_healthy()
add_chat_request()
stream_request()
add_aborted_req()
```

`FakeInferenceServer` 模拟现有 HTTP inference server 中的这些辅助逻辑：

```python
_normalize_messages()
_build_sampling_params()
_convert_finish_reason()
```

因此，这个测试验证的是集群编排层，而不是模型推理内核。

**2. 启动 worker TCP 服务**

```python
await service.start()
```

`TcpWorkerService.start()` 会做两件事：

1. 启动真实 `asyncio` TCP server
2. 把 worker 记录注册到 file discovery，并启动 heartbeat

因为 `record.port=0`，`service.start()` 会读取实际绑定的端口，并更新：

```python
service.record.port
```

所以后续客户端连接使用：

```python
client = TcpWorkerClient(service.record.host, service.record.port)
```

这说明测试可以并发运行，不会固定占用某个端口。

**3. 测试 health 协议**

第一段是：

```python
health = await client.health(hello)
self.assertEqual(health["status"], "ready")
```

`TcpWorkerClient.health()` 的内部流程是：

```text
connect
  -> 发送 hello 帧
  -> 等待 hello_ack
  -> 发送 health op
  -> 等待 health 响应
  -> close
```

worker 端在 [worker_service.py](./InfiniLM/python/infinilm/cluster/worker_service.py:125) 处理 health：

```python
if operation == "health":
    await send({
        "type": "health",
        "status": "ready" if self.engine.is_healthy() else "unhealthy",
        "engine_healthy": self.engine.is_healthy(),
        "worker_id": self.record.instance_id,
    })
```

因为 `FakeEngine.is_healthy()` 返回 `True`，所以测试断言：

```python
health["status"] == "ready"
```

这一步验证：

- TCP 握手协议版本正确
- worker 能处理 health op
- health 状态来自 engine 的 `is_healthy()`
- client 能正确解析 health 帧

由于 `health()` 结束后会关闭连接，所以测试随后重新连接：

```python
await client.connect(hello)
```

这是为下一段 chat 请求准备新的连接。

**4. 测试 chat token 流**

核心代码：

```python
stream = client.stream_chat(
    "test-request",
    {"messages": [{"role": "user", "content": "hi"}]},
)

frames = [frame async for frame in stream]
```

客户端会发送：

```json
{
  "op": "chat",
  "request_id": "test-request",
  "payload": {
    "messages": [
      {"role": "user", "content": "hi"}
    ]
  }
}
```

worker 收到 `chat` 后，在 [worker_service.py](./InfiniLM/python/infinilm/cluster/worker_service.py:134) 创建异步任务：

```python
requests[request_id] = asyncio.create_task(
    self._generate(request_id, payload, send)
)
```

`_generate()` 的流程是：

```text
发送 accepted
  -> normalize messages
  -> build SamplingParams
  -> engine.add_chat_request()
  -> engine.stream_request()
  -> 每个 token 输出发送 delta 帧
  -> finished 时发送 finish 帧
```

`FakeEngine.stream_request()` 依次产生：

```text
TokenOutput(token_text="Ready")
TokenOutput(token_text=".")
TokenOutput(token_text="", finished=True, finish_reason=EOS_TOKEN)
```

worker 会把前两个 token 发成：

```json
{"type": "delta", "request_id": "test-request", "content": "Ready"}
{"type": "delta", "request_id": "test-request", "content": "."}
```

最后一个 token 是 EOS token，`token_text=""`，所以不会再发送 delta，而是发送：

```json
{
  "type": "finish",
  "request_id": "test-request",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 2,
    "total_tokens": 5
  }
}
```

测试断言帧顺序：

```python
self.assertEqual(
    [frame["type"] for frame in frames],
    ["delta", "delta", "finish"],
)
```

这验证 TCP 协议的输出顺序没有错乱。

再断言内容：

```python
self.assertEqual(
    "".join(frame.get("content", "") for frame in frames),
    "Ready.",
)
```

即两个 delta 帧拼接后得到：

```text
Ready.
```

最后断言 usage：

```python
self.assertEqual(
    frames[-1]["usage"]["prompt_tokens"],
    3,
)
```

`FakeRequest.get_prompt_length()` 返回 3，`get_num_generated_tokens()` 返回 2，因此 finish 帧中的 usage 应该是：

```json
{
  "prompt_tokens": 3,
  "completion_tokens": 2,
  "total_tokens": 5
}
```

这一段完整验证了：

```text
TCP chat frame
  -> worker handler
  -> engine.add_chat_request()
  -> engine.stream_request()
  -> TCP delta frame
  -> TCP finish frame
```

**5. 测试客户端取消**

第二段使用新的 TCP client：

```python
cancel_client = TcpWorkerClient(
    service.record.host,
    service.record.port,
)
await cancel_client.connect(hello)
```

然后创建一个流式请求：

```python
cancelled_stream = cancel_client.stream_chat(
    "cancel-request",
    {"messages": [{"role": "user", "content": "hi"}]},
)
```

先读取第一个输出：

```python
await anext(cancelled_stream)
```

此时客户端已经收到第一个 `delta`，请求还没有完成。

然后关闭 async generator：

```python
await cancelled_stream.aclose()
```

这会触发 `TcpWorkerClient.stream_chat()` 的 `finally`：

```python
finally:
    await self.close(request_id)
```

`close(request_id)` 会向 worker 发送：

```json
{
  "op": "cancel",
  "request_id": "cancel-request"
}
```

worker 在 [worker_service.py](./InfiniLM/python/infinilm/cluster/worker_service.py:157) 处理 cancel：

```python
elif operation == "cancel" and isinstance(request_id, str):
    task = requests.get(request_id)
    if task is not None and not task.done():
        task.cancel()
```

任务被取消后，`_generate()` 进入 `finally`：

```python
finally:
    if req is not None and not req.is_finished():
        self.engine.add_aborted_req(req, abort_reason)
```

这里调用的是 fake engine 的：

```python
def add_aborted_req(self, request, reason=FinishReason.CANCELED):
    self.aborted.set()
```

测试最后等待这个事件：

```python
await asyncio.wait_for(engine.aborted.wait(), timeout=1.0)
```

如果 1 秒内没有触发 abort，测试失败。

这一段验证的取消链路是：

```text
client generator aclose
  -> client 发送 cancel 帧
  -> worker 找到对应 request task
  -> task.cancel()
  -> _generate finally
  -> engine.add_aborted_req()
```

这对应生产环境中的场景：

```text
OpenAI 客户端断开 SSE
  -> frontend 停止消费 worker stream
  -> frontend 关闭到 worker 的 TCP 连接
  -> worker 取消生成任务
  -> AsyncLLMEngine abort 请求
```

**6. 测试服务停止清理**

最后：

```python
await service.stop()
```

`stop()` 会：

- 取消 heartbeat task
- 关闭 TCP server
- 从 file registry 删除 worker 注册文件

因此测试也顺带覆盖了 worker 的正常退出路径。

**这个测试的整体数据流**

```text
TcpWorkerClient
  |
  | hello
  v
TcpWorkerService
  |
  | health op
  v
FakeEngine.is_healthy()
  |
  | ready
  v
TcpWorkerClient
  |
  | chat op
  v
TcpWorkerService._handle_client()
  |
  | create task
  v
TcpWorkerService._generate()
  |
  | add_chat_request / stream_request
  v
FakeEngine
  |
  | TokenOutput
  v
Tcp delta frames
  |
  | finish frame + usage
  v
TcpWorkerClient
```

取消流：

```text
TcpWorkerClient.stream_chat()
  |
  | aclose()
  v
close(request_id)
  |
  | cancel op
  v
TcpWorkerService._handle_client()
  |
  | task.cancel()
  v
TcpWorkerService._generate() finally
  |
  | add_aborted_req()
  v
FakeEngine.aborted.set()
```

**它没有覆盖的内容**

这个测试不验证：

- 真实 Qwen 模型 forward
- GPU 推理
- tokenizer 和 chat template
- frontend HTTP/OpenAI API
- round-robin 路由
- 多 worker 集群
- worker 故障切换
- malformed frame
- request timeout
- engine error
- 并发多个请求

这些分别由 `ClusterFrontendTest`、真实 `run_cluster.sh` 集成测试，或后续可补充的错误/并发测试覆盖。

**总结**

`test_health_chat_stream_and_cancellation` 是 `TcpWorkerService` 的核心数据面测试。它用一个 fake engine 验证真实 asyncio TCP 服务的三个关键能力：

1. **health 查询可用**
2. **chat token stream 正确**
3. **客户端取消能传播到 engine abort 队列**

它是三副本聚合式推理服务中 frontend 到 worker raw TCP request plane 的基础正确性验证。

> self._server = await asyncio.start_server这行函数是在做什么？请分析

`self._server = await asyncio.start_server(...)` 在 [worker_service.py](./InfiniLM/python/infinilm/cluster/worker_service.py:37)，作用是：**在当前 asyncio event loop 中创建并启动一个 raw TCP server，用来接收 frontend 发来的推理请求**。

核心代码是：

```python
self._server = await asyncio.start_server(
    self._handle_client,
    host=self.record.host,
    port=self.record.port,
)
```

这行代码做了几件事。

**1. 创建 TCP 监听 socket**

参数含义：

```python
host=self.record.host
```

表示监听地址。测试和单机部署中通常是：

```text
127.0.0.1
```

如果写成：

```text
0.0.0.0
```

则表示监听本机所有网卡，允许其他机器连接。

```python
port=self.record.port
```

表示监听端口。

在三副本部署中，三个 worker 分别使用：

```text
GPU0 worker: 18181
GPU1 worker: 18182
GPU2 worker: 18183
```

这样 frontend 可以根据 file discovery 中的地址和端口连接对应 worker。

**2. 注册客户端连接处理函数**

第一个参数：

```python
self._handle_client
```

是一个 async callback：

```python
async def _handle_client(
    self,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    ...
```

`asyncio.start_server()` 的行为是：

```text
TCP server 启动
  |
  | 有客户端连接进来
  v
为该连接创建任务
  |
  | 调用 self._handle_client(reader, writer)
  v
_handle_client 处理这个连接上的帧
```

每个 TCP 连接都会得到一对：

```python
reader  # 读取客户端发来的数据
writer  # 向客户端写响应数据
```

在 `_handle_client()` 中，worker 会依次处理：

```text
hello
  -> hello_ack
  -> health / chat / cancel
```

其中 `chat` 会创建生成任务，并把 token 输出通过 `writer` 发回 frontend。

**3. `await` 表示等待 server 完成启动，而不是等待所有客户端**

这行：

```python
self._server = await asyncio.start_server(...)
```

容易误解为“阻塞等待客户端连接”。实际上不是。

它的执行过程是：

```text
调用 asyncio.start_server()
  -> 创建 socket
  -> bind 到 host:port
  -> listen
  -> 在 event loop 中注册 accept 事件
  -> 返回 asyncio.AbstractServer
```

`await` 只等待 TCP server 完成启动并返回 server 对象。

它不会一直等到客户端连接结束。

也就是说：

```python
await asyncio.start_server(...)
```

完成后，worker 会继续执行后面的代码：

```python
sockets = [socket for socket in (self._server.sockets or []) if socket]
...
await self.registry.register(self.record)
...
```

server 会继续在 event loop 后台接受连接。

**4. 返回值保存在 `self._server`**

返回对象类型是：

```python
asyncio.AbstractServer
```

保存到：

```python
self._server
```

是为了后续可以停止服务：

```python
async def stop(self):
    ...
    if self._server is not None:
        self._server.close()
        await self._server.wait_closed()
        self._server = None
```

对应行为：

```text
self._server.close()
  -> 停止接受新连接

await self._server.wait_closed()
  -> 等待已有连接和 server 清理完成
```

因此 `self._server` 是 worker TCP 数据面的生命周期句柄。

**5. `port=0` 时会自动分配空闲端口**

测试里创建 worker record 时使用：

```python
port=0
```

`asyncio.start_server()` 遇到端口 `0` 时，会请求操作系统分配一个当前空闲的随机端口。

启动完成后，代码会读取实际绑定地址：

```python
sockets = [socket for socket in (self._server.sockets or []) if socket]
if sockets:
    host, port = sockets[0].getsockname()[:2]
    self.record = replace(self.record, port=port)
```

这一步很关键。

原因是：

- 配置里的 `record.port` 可能是 `0`
- 真正可用的端口要等 OS 分配后才知道
- worker 注册到 file discovery 时，必须写入真实端口
- frontend 后续会用这个真实端口连接 worker

所以流程是：

```text
record.port = 0
  |
  | asyncio.start_server()
  v
OS 分配实际端口，例如 52341
  |
  | socket.getsockname()
  v
self.record.port = 52341
  |
  | registry.register(self.record)
  v
frontend discovery 看到真实端口 52341
```

**6. 它启动的是 worker 的 raw TCP request plane，不是 HTTP server**

这个 InfiniLM worker 进程里实际有两个服务端：

```text
InferenceServer / uvicorn
  -> HTTP 诊断接口
  -> /health
  -> /v1/models
  -> /v1/chat/completions

TcpWorkerService / asyncio.start_server
  -> frontend 到 worker 的 raw TCP 数据面
  -> hello
  -> health
  -> chat
  -> delta
  -> finish
  -> cancel
```

所以这行代码不是启动 uvicorn，也不是启动 OpenAI HTTP 服务，而是启动集群内部的数据面：

```text
frontend
  |
  | raw TCP JSON lines
  v
asyncio.start_server()
  |
  v
_handle_client()
  |
  v
AsyncLLMEngine
```

**7. 每个连接如何并发处理**

`asyncio.start_server()` 会为每个客户端连接调用 `_handle_client()`。

由于 `_handle_client()` 是 async 函数，多个连接可以并发运行：

```text
frontend connection A -> _handle_client task A
frontend connection B -> _handle_client task B
frontend connection C -> _handle_client task C
```

在 `_handle_client()` 内部，收到 `chat` 后还会创建独立任务：

```python
requests[request_id] = asyncio.create_task(
    self._generate(request_id, payload, send)
)
```

因此支持同一个 TCP 连接上多个 request 并发生成，只要 `request_id` 不同。

**8. 对应三副本部署中的位置**

在真实三副本集群里，每个 GPU worker 进程都会执行一次：

```python
await asyncio.start_server(...)
```

结果类似：

```text
GPU0 worker
  host=127.0.0.1
  port=18181
  self._server listening

GPU1 worker
  host=127.0.0.1
  port=18182
  self._server listening

GPU2 worker
  host=127.0.0.1
  port=18183
  self._server listening
```

frontend 从 file discovery 中读到这些地址：

```json
{
  "host": "127.0.0.1",
  "port": 18181
}
```

然后 round-robin 选择一个 worker，并通过 TCP 连接发送 chat 请求。

**总结**

这行代码的完整含义是：

```python
self._server = await asyncio.start_server(
    self._handle_client,
    host=self.record.host,
    port=self.record.port,
)
```

等价于：

```text
在指定 host:port 上启动一个 raw TCP server；
server 启动成功后立即返回；
后台持续接受客户端连接；
每来一个连接，就创建一个 _handle_client(reader, writer) 任务；
_handle_client 负责处理 hello/health/chat/cancel 协议；
返回的 self._server 用于之后关闭 TCP 服务。
```

它是 InfiniLM 三副本集群中 worker 侧 request-plane 的启动点。

> TcpWorkerServiceTest::test_health_chat_stream_and_cancellation测试中，await service.start()函数是阻塞的么？请分析

<details><summary>1 previous message</summary>

> 我先核对该测试和 `TcpWorkerService.start()` 的当前实现，再区分“等待协程完成”和“阻塞 event loop/等待客户端连接”这两个容易混淆的点。
</details>

**结论**

`await service.start()` 不是“长期阻塞等待客户端连接”的调用；它是一个**异步初始化流程**。对当前 `run()` 协程来说，它会顺序等待初始化完成；对 event loop 来说，它不会一直占住事件循环，也不会等待 TCP 客户端连接进来。

更精确地说：

```text
await service.start()
```

的含义是：

```text
等待 worker 完成启动准备工作
  1. 启动 raw TCP server，完成 bind/listen
  2. 读取 OS 实际分配的端口
  3. 把 worker 注册到 file discovery
  4. 创建后台 heartbeat task
  5. 返回
```

它返回后，TCP server 仍在 event loop 后台继续接受连接。

**源码位置**

`TcpWorkerService.start()` 在 [worker_service.py](./python/infinilm/cluster/worker_service.py:41)：

```python
async def start(self) -> None:
    if self._server is not None:
        return

    self._server = await asyncio.start_server(
        self._handle_client,
        host=self.bind_host,
        port=self.record.port,
    )
    sockets = [socket for socket in (self._server.sockets or []) if socket]
    if sockets:
        host, port = sockets[0].getsockname()[:2]
        self.record = replace(self.record, port=port)

    await self.registry.register(self.record)
    self._heartbeat_task = asyncio.create_task(
        self._heartbeat(),
        name=f"infinilm-worker-heartbeat-{self.record.instance_id}",
    )
```

测试中的调用在 [test_cluster.py](./test/service/test_cluster.py:185)：

```python
await service.start()
```

**逐段分析**

1. `await asyncio.start_server(...)` 不是等待客户端

这一步做的是：

```text
创建 socket
  -> bind 到 host:port
  -> listen
  -> 注册 accept 事件
  -> 返回 asyncio.AbstractServer
```

它只等待 server 完成启动，不等待客户端连接。

如果后续有 frontend 或测试 client 连接进来，event loop 会调用：

```python
self._handle_client(reader, writer)
```

但这发生在后台，不需要 `start()` 一直等待。

2. `port=0` 时必须等待启动完成

测试里创建 record 时使用了：

```python
port=0
```

这表示让操作系统分配一个随机空闲端口。真正端口只有 `asyncio.start_server()` 完成后才能通过：

```python
socket.getsockname()
```

拿到。

所以 `await service.start()` 返回后，下面这行才能拿到正确端口：

```python
client = TcpWorkerClient(service.record.host, service.record.port)
```

如果不 `await`，`service.record.port` 可能仍然是 `0`，测试会连接错误地址。

3. `await self.registry.register(...)` 会等待注册文件写完

`registry.register()` 内部使用：

```python
await asyncio.to_thread(self._write, record)
```

见 [registry.py](./python/infinilm/cluster/registry.py:113)。

这意味着：

```text
当前协程等待注册完成
  |
  | _write() 放到线程池执行
  v
event loop 不被磁盘 I/O 阻塞
```

因此它是“当前协程等待完成”，但不是“阻塞 event loop”。

4. `asyncio.create_task(self._heartbeat())` 不等待 heartbeat

这一步只是创建后台任务：

```python
self._heartbeat_task = asyncio.create_task(self._heartbeat())
```

它不会等 heartbeat 循环结束。heartbeat task 会在后续 event loop 调度时开始运行。

所以 `start()` 不会等待第一次 heartbeat，更不会等待无限循环退出。

**调用时序**

```text
asyncio.run(run())
  |
  v
run()
  |
  | await service.start()
  v
TcpWorkerService.start()
  |
  | await asyncio.start_server()
  |   TCP bind/listen 完成
  |
  | 读取实际端口，更新 record.port
  |
  | await registry.register()
  |   文件写入线程完成
  |
  | create_task(_heartbeat)
  |   后台任务已调度，不等待
  |
  v
start() 返回
  |
  v
测试继续创建 TcpWorkerClient
  |
  v
client.health() 连接 worker
  |
  v
event loop 调用 _handle_client()
```

**三种“阻塞”的区别**

| 问题 | 答案 |
| --- | --- |
| `run()` 协程会不会等 `start()` 完成才执行下一行？ | 会。这是 `await` 的正常语义。 |
| `start()` 会不会阻塞 event loop？ | 不会长期阻塞。磁盘写入放到 `asyncio.to_thread()`，TCP server 启动是异步 API。 |
| `start()` 会不会等待客户端连接？ | 不会。它只启动监听，客户端连接由后台 `_handle_client()` 处理。 |
| `asyncio.run(run())` 会不会阻塞测试线程？ | 会。整个测试协程运行完成后才返回，这是 unittest 中使用 `asyncio.run()` 的正常行为。 |

**总结**

`await service.start()` 是一个**有限时间的异步初始化等待**：等 TCP server 绑定成功、实际端口确定、worker 注册完成、heartbeat task 创建完成。它不等待客户端连接，也不等待 heartbeat 循环结束，更不会长期阻塞 event loop。

在这个测试里，`await` 是必须的；否则 TCP server 可能还没启动，`service.record.port` 也可能还是 `0`，后续 `TcpWorkerClient` 就无法正确连接。

> 根据上述问题回答，为代码添加上中文注释。

<details><summary>7 previous messages</summary>

> 我会把 `TcpWorkerService.start()` 的等待语义、TCP 监听不等待客户端、`port=0` 时获取实际端口、注册完成和 heartbeat 后台任务这些点，补到实现和测试调用处；只加注释，不改行为。
>
> 当前工作区已有未提交改动：一个 Markdown 重命名和 `test_cluster.py` 的注释修改。我只在相关位置追加中文注释，不碰这些已有内容。
>
> 现在开始添加注释：`worker_service.py` 说明实现语义，`test_cluster.py` 说明测试里为什么必须 `await`，行为保持不变。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./python/infinilm/cluster/worker_service.py` (+18 -0)
> - Wrote `./test/service/test_cluster.py` (+5 -0)
>
> </details>
>
> 注释已添加。接下来运行 Ruff、编译检查和对应的 `TcpWorkerServiceTest`，确认注释没有引入格式或行为问题。
>
> Ruff 和编译检查通过；测试收集在导入原生 `infinicore` 时缺少 `libinfinicore_cpp_api.so`，这是当前 shell 的动态库路径问题，不是注释改动导致。我先尝试加载项目环境后再重跑。
>
> 加载环境后测试仍在 `infinicore` 的设备预加载阶段崩溃，说明当前执行环境无法初始化原生运行时；注释改动本身没有触发这个路径。我再做一次 diff 检查，确认只增加了注释。
</details>

已按前述分析添加中文注释，未修改任何执行逻辑。

主要位置：

- [worker_service.py](./python/infinilm/cluster/worker_service.py:41)：为 `TcpWorkerService.start()` 添加 docstring，说明 `await start()` 等待的是初始化完成，而不是等待客户端连接或 heartbeat 循环结束。
- [worker_service.py](./python/infinilm/cluster/worker_service.py:52)：说明 `asyncio.start_server()` 只等待 bind/listen 完成，后续 accept 由 event loop 调度。
- [worker_service.py](./python/infinilm/cluster/worker_service.py:60)：说明 `port=0` 时为什么必须在注册 discovery 前读取实际绑定端口。
- [worker_service.py](./python/infinilm/cluster/worker_service.py:67)：说明注册等待语义，以及 `asyncio.to_thread()` 如何避免阻塞 event loop。
- [worker_service.py](./python/infinilm/cluster/worker_service.py:72)：说明 heartbeat 是后台 task，不会被 `start()` 等待。
- [test_cluster.py](./test/service/test_cluster.py:186)：在 `await service.start()` 调用处说明测试为什么必须等待初始化完成。

验证结果：

```text
ruff: passed
compileall: passed
git diff --check: passed
```

对应的 `TcpWorkerServiceTest` 在当前环境中未能执行：测试模块导入 `infinicore` 时先是缺少 `libinfinicore_cpp_api.so`，加载环境后又在 InfiniCore 设备预加载阶段 abort。这是当前原生运行环境问题，不是本次注释改动导致。

> health = await client.health(hello)和await client.connect(hello)中，内部的流程是什么？请分析。

<details><summary>8 previous messages</summary>

> Stream error
>
> Reconnecting... waiting for network
> Connection failed: error sending request
>
> Stream error
>
> Reconnecting... waiting for network
> Connection failed: error sending request
>
> Stream error
>
> Reconnecting... waiting for network
> Connection failed: error sending request
>
> Stream error
>
> Reconnecting... waiting for network
> Connection failed: error sending request
>
> 我先查看 `TcpWorkerClient.connect()`、`health()` 和 worker 端 `_handle_client()` 的当前代码，把客户端握手、health 请求、连接关闭和后续重连的完整时序核对清楚。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./python/infinilm/cluster/worker_service.py`
> - Read `./python/infinilm/cluster/tcp.py`
> - Read `./python/infinilm/cluster/protocol.py`
> - Read `./test/service/test_cluster.py`
>
> </details>
>
> 代码已经核对完。关键点是：`health()` 内部会自己完成一次“连接 + 握手 + health 查询 + 关闭”，所以测试后面的 `await client.connect(hello)` 实际是在建立第二条新连接，用于后面的 chat 请求。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./python/infinilm/cluster/tcp.py`
> - Read `./python/infinilm/cluster/worker_service.py`
> - Read `./test/service/test_cluster.py`
> - Read `./python/infinilm/cluster/tcp.py`
>
> </details>
</details>

**结论**

这两行虽然都和连接有关，但作用不同：

```python
health = await client.health(hello)
await client.connect(hello)
```

第一行 `client.health(hello)` 是一次完整的临时 health 会话：

```text
建立 TCP 连接
  -> 发送 hello 握手
  -> 收到 hello_ack
  -> 发送 health
  -> 收到 health 响应
  -> 关闭 TCP 连接
  -> 返回 health dict
```

第二行 `client.connect(hello)` 是重新建立一条新的 TCP 连接，并完成同样的 hello 握手。因为 `health()` 最后已经调用 `close()` 把连接关掉了，所以这里必须重新连接，后面的 `stream_chat()` 才能用这条新连接发送 chat 请求。

---

## 1. `health = await client.health(hello)` 的内部流程

入口在 [tcp.py](./python/infinilm/cluster/tcp.py:119)：

```python
async def health(self, hello: WorkerHello) -> dict[str, Any]:
    await self.connect(hello)
    assert self._writer is not None
    await write_frame(self._writer, {"op": "health"})
    response = await read_frame(self._reader)
    if response.get("type") != "health":
        raise RuntimeError(f"invalid health response: {response}")
    await self.close()
    return response
```

它分为四步。

### 1.1 先调用 `connect(hello)`

```python
await self.connect(hello)
```

`health()` 自己不会直接假设连接已经存在，而是先建立连接并完成握手。

`connect()` 在 [tcp.py](./python/infinilm/cluster/tcp.py:51)：

```python
async def connect(self, hello: WorkerHello) -> None:
    if self.connected:
        return

    self._reader, self._writer = await asyncio.wait_for(
        asyncio.open_connection(self.host, self.port),
        timeout=self.connect_timeout,
    )

    await write_frame(
        self._writer,
        {
            "op": "hello",
            "protocol_version": hello.protocol_version,
            "worker_type": hello.worker_type,
        },
    )

    response = await read_frame(self._reader)

    if response.get("type") != "hello_ack":
        raise ConnectionError(...)

    if response.get("protocol_version") != hello.protocol_version:
        raise ConnectionError("worker protocol version mismatch")
```

具体过程：

```text
TcpWorkerClient.health()
  |
  | await self.connect(hello)
  v
asyncio.open_connection(host, port)
  |
  | 建立 TCP socket 连接
  v
得到 reader / writer
  |
  | write_frame({"op":"hello", ...})
  v
发送一行 JSON：
{"op":"hello","protocol_version":1,"worker_type":"aggregated"}\n
  |
  | await read_frame(reader)
  v
等待 worker 返回 hello_ack
  |
  | 校验 type == hello_ack
  | 校验 protocol_version == 1
  v
connect() 返回
```

注意两点：

- `connect_timeout` 只包住 `asyncio.open_connection()`，也就是 TCP socket 建立阶段。
- 后续发送 hello、等待 `hello_ack` 没有单独的 timeout。如果 worker 已接受连接但不回包，`connect()` 可能一直等待。

### 1.2 worker 端处理 hello

worker 端每个 TCP 连接都会进入 `_handle_client()`，见 [worker_service.py](./python/infinilm/cluster/worker_service.py:111)。

它首先读取第一帧：

```python
hello = await read_frame(reader)
```

然后检查：

```python
if hello.get("op") != "hello":
    raise ValueError("first TCP frame must be hello")

if hello.get("protocol_version") != PROTOCOL_VERSION:
    raise ValueError("unsupported cluster protocol version")
```

校验通过后返回：

```python
{
    "type": "hello_ack",
    "protocol_version": PROTOCOL_VERSION,
    "worker_type": self.record.worker_type,
    "served_model_name": self.record.served_model_name,
}
```

所以第一阶段的完整交互是：

```text
client                              worker
  |                                   |
  | {"op":"hello",...}\n              |
  | --------------------------------> |
  |                                   |
  |                                   | 校验 op 和 protocol_version
  |                                   |
  | {"type":"hello_ack",...}\n        |
  | <-------------------------------- |
  |                                   |
  | 校验 type 和 protocol_version      |
```

客户端只严格校验：

```python
type == "hello_ack"
protocol_version == hello.protocol_version
```

它不会校验 worker 返回的 `worker_type` 和 `served_model_name`。

---

### 1.3 客户端发送 `health` 帧

握手完成后，`health()` 继续执行：

```python
await write_frame(self._writer, {"op": "health"})
```

实际发送的 TCP 内容是一行 JSON：

```json
{"op":"health"}\n
```

这里没有 `request_id`，因为 health 是连接级操作，不是某个生成请求。

---

### 1.4 worker 端处理 `health`

worker 在握手后进入循环：

```python
while True:
    frame = await read_frame(reader)
    operation = frame.get("op")
```

看到 `operation == "health"` 后，构造响应：

```python
{
    "type": "health",
    "status": "ready" if self.engine.is_healthy() else "unhealthy",
    "engine_healthy": self.engine.is_healthy(),
    "worker_id": self.record.instance_id,
}
```

对应 [worker_service.py](./python/infinilm/cluster/worker_service.py:143)。

如果 fake engine 的 `is_healthy()` 返回 `True`，响应是：

```json
{
  "type": "health",
  "status": "ready",
  "engine_healthy": true,
  "worker_id": "..."
}
```

然后 worker 会继续留在 `while True`，等待这条连接上的下一帧。

---

### 1.5 客户端读取并校验 health 响应

客户端执行：

```python
response = await read_frame(self._reader)
```

`read_frame()` 会：

1. `await reader.readline()` 等待一整行 JSON。
2. 检查是否为空，空表示连接关闭。
3. 检查帧大小。
4. JSON 解析成 dict。

然后校验：

```python
if response.get("type") != "health":
    raise RuntimeError(...)
```

所以如果 worker 返回错误帧或其他类型，`health()` 会抛异常。

---

### 1.6 `health()` 最后关闭连接

这是最关键的一步：

```python
await self.close()
return response
```

`close()` 在 [tcp.py](./python/infinilm/cluster/tcp.py:72)：

```python
writer.close()
await writer.wait_closed()
self._reader = None
self._writer = None
```

因为 `close()` 没有传 `request_id`，所以不会发送 cancel，只是直接关闭 TCP 连接。

关闭后：

```python
client.connected == False
```

此时 `client` 对象还在，但它不再持有可用的 TCP 连接。

worker 端随后会在 `read_frame(reader)` 处读到 EOF，抛出 `ConnectionError`，然后被 `_handle_client()` 捕获并清理连接。

所以：

```python
health = await client.health(hello)
```

返回后，第一条 TCP 连接已经完全关闭。

---

## 2. `await client.connect(hello)` 的内部流程

测试下一行是：

```python
await client.connect(hello)
```

由于 `health()` 已经关闭连接，`client.connected` 是 `False`，所以 `connect()` 不会走这个提前返回分支：

```python
if self.connected:
    return
```

它会重新建立一条新的 TCP 连接。

流程如下：

```text
client.connect(hello)
  |
  | self.connected == False
  v
asyncio.open_connection(host, port)
  |
  | 建立第二条 TCP 连接
  v
发送 hello 帧
  |
  | {"op":"hello","protocol_version":1,"worker_type":"aggregated"}\n
  v
worker._handle_client()
  |
  | 校验 op / protocol_version
  | 返回 hello_ack
  v
client 读取 hello_ack
  |
  | 校验 type / protocol_version
  v
connect() 返回
```

这次连接不会关闭，会保留在 `client._reader` 和 `client._writer` 里，供下一行使用：

```python
stream = client.stream_chat(...)
```

也就是说，测试里实际上建立了两条 TCP 连接：

```text
第 1 条连接：
  health 查询
  -> health 响应
  -> client.close()

第 2 条连接：
  hello 握手
  -> chat 请求
  -> delta / finish / error / cancel
  -> stream_chat() finally 中 close()
```

---

## 3. 两条调用的时序图

完整时序可以表示为：

```text
TcpWorkerClient                     TcpWorkerService
      |                                   |
      | await client.health(hello)        |
      |                                   |
      | --- TCP connect ----------------> |
      |                                   |
      | --- hello frame ----------------> |
      |                                   |
      | <-- hello_ack frame ------------- |
      |                                   |
      | --- health frame ---------------> |
      |                                   |
      |                                   | engine.is_healthy()
      | <-- health frame ---------------- |
      |                                   |
      | --- TCP close ------------------> |
      |                                   |
      |                                   | read EOF, cleanup
      |                                   |
      | health() return                   |
      |                                   |
      | await client.connect(hello)       |
      |                                   |
      | --- TCP connect ----------------> |
      |                                   |
      | --- hello frame ----------------> |
      |                                   |
      | <-- hello_ack frame ------------- |
      |                                   |
      | connect() return                  |
      |                                   |
      | stream_chat(...)                  |
      | --- chat frame -----------------> |
      |                                   |
      | <-- accepted/delta/finish ------- |
```

---

## 4. `WorkerHello` 里的 model 和 served_model_name 有没有被发送？

测试里构造了：

```python
hello = WorkerHello(
    model=record.model,
    served_model_name="qwen",
)
```

但 `connect()` 实际发送的 hello 帧只包含：

```python
{
    "op": "hello",
    "protocol_version": hello.protocol_version,
    "worker_type": hello.worker_type,
}
```

见 [tcp.py](./python/infinilm/cluster/tcp.py:58)。

也就是说：

- `hello.model`
- `hello.served_model_name`

当前不会发送到 worker。

worker 返回的 `hello_ack` 中包含 worker 自己注册的：

- `worker_type`
- `served_model_name`

但客户端当前也不校验这两个字段。

这一点不影响这个测试，因为 worker 和 client 使用的都是同一个 `service.record` 信息；但在真实 frontend 路由中，如果要做更强的模型匹配校验，当前握手协议还有增强空间。

---

## 5. `await` 在这里阻塞吗？

和前面 `service.start()` 类似，需要区分三种含义。

### 对当前协程来说

会等待。

例如：

```python
health = await client.health(hello)
```

会等待 health 响应读取完成并关闭连接后才返回。

```python
await client.connect(hello)
```

会等待第二条 TCP 连接建立和 `hello_ack` 校验完成后才返回。

### 对 event loop 来说

不会阻塞 event loop。

这些等待点都是异步 I/O：

```python
await asyncio.open_connection(...)
await write_frame(...)
await read_frame(...)
await writer.wait_closed()
```

当前协程在等待网络数据时会让出控制权，event loop 可以继续运行 worker 的 `_handle_client()`、heartbeat task 或其他任务。

### 对测试外层线程来说

`asyncio.run(run())` 会同步等待整个 `run()` 协程完成。这是 unittest 中使用 `asyncio.run()` 的正常行为。

---

## 6. 值得注意的边界情况

当前实现有两个小细节：

1. `connect_timeout` 只覆盖 TCP 建连

```python
await asyncio.wait_for(
    asyncio.open_connection(self.host, self.port),
    timeout=self.connect_timeout,
)
```

握手阶段的 `write_frame()` 和 `read_frame()` 没有超时保护。如果 worker 接受连接后不返回 `hello_ack`，`connect()` 可能无限等待。

2. worker health 状态调用了两次 `engine.is_healthy()`

```python
"status": "ready" if self.engine.is_healthy() else "unhealthy",
"engine_healthy": self.engine.is_healthy(),
```

如果 `is_healthy()` 是动态状态，理论上两次调用之间状态可能变化，导致 `status` 和 `engine_healthy` 不一致。更稳妥的写法是只调用一次并复用结果。不过在该测试中 fake engine 的健康状态是固定的，所以不会出现问题。

---

**总结**

`client.health(hello)` 是一个自包含的临时 health 会话：内部建立连接、握手、发送 health、读取响应、关闭连接，然后返回 health dict。

`client.connect(hello)` 在测试中是重新建立第二条连接：因为 health 会话已经关闭了第一条连接，后续 chat 请求需要一条新的、保持打开的 TCP 连接。

> 将上述内容补充到代码注释中。

<details><summary>6 previous messages</summary>

> 我会把这两条连接的生命周期补充到 `TcpWorkerClient.connect()`、`health()`、`close()` 和 worker 端 `_handle_client()` 的注释中，并在测试调用处说明“health 会关闭第一条连接，随后 connect 建立第二条连接”。只改注释，不改行为。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./python/infinilm/cluster/tcp.py`
> - Read `./python/infinilm/cluster/worker_service.py`
> - Read `./test/service/test_cluster.py`
>
> </details>
>
> 准备添加三类注释：客户端连接/健康检查生命周期、worker 端握手与 health 处理、测试中两条连接的关系。
>
> <details><summary>Edited 3 files</summary>
>
> - Wrote `./python/infinilm/cluster/tcp.py` (+31 -0)
> - Wrote `./python/infinilm/cluster/worker_service.py` (+12 -0)
> - Wrote `./test/service/test_cluster.py` (+5 -0)
>
> </details>
>
> 注释已添加完成。现在运行 Ruff、编译检查和 diff 检查；对应测试如果仍因原生运行环境无法导入，我会明确说明。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `python -m ruff check python/infinilm/cluster/tcp.py python/infinilm/cluster/worker_service.py test/service/test_cluster.py && python -m compileall -q python/infinilm/cluster/tcp.py python/infinilm/cluster/worker_service.py test/service/test_cluster.py && git diff --check`
> - Ran `PYTHONPATH=python python -m pytest -q test/service/test_cluster.py::TcpWorkerServiceTest::test_health_chat_stream_and_cancellation`
>
> </details>
</details>

已将 `client.health(hello)` 和 `client.connect(hello)` 的完整内部流程补充为中文注释，未修改任何执行逻辑。

主要更新位置：

- [tcp.py](./python/infinilm/cluster/tcp.py:51)：说明 `connect()` 的流程，包括 TCP 建连、发送 `hello`、等待并校验 `hello_ack`；同时说明 `connect_timeout` 只覆盖 `open_connection()`，不覆盖握手读写。
- [tcp.py](./python/infinilm/cluster/tcp.py:94)：说明 `close()` 的行为；health 查询不传 `request_id`，chat 流结束时传 `request_id` 并先发送 `cancel`。
- [tcp.py](./python/infinilm/cluster/tcp.py:161)：说明 `health()` 是一次独立会话：连接、握手、发送 `health`、读取响应、关闭连接、返回结果。
- [worker_service.py](./python/infinilm/cluster/worker_service.py:126)：说明 worker 端每条连接的生命周期，第一帧必须是 `hello`，握手后进入 `health/chat/cancel` 命令循环。
- [worker_service.py](./python/infinilm/cluster/worker_service.py:168)：说明 health 响应来自 `engine.is_healthy()`，并注明当前连续调用两次 `is_healthy()` 的潜在动态状态问题。
- [test_cluster.py](./test/service/test_cluster.py:195)：说明 `health()` 使用并关闭第一条 TCP 连接。
- [test_cluster.py](./test/service/test_cluster.py:200)：说明后面的 `connect()` 建立第二条 TCP 连接，供 `stream_chat()` 使用。

验证结果：

```text
ruff: passed
compileall: passed
git diff --check: passed
```

目标测试仍无法在当前环境中收集执行：导入 `infinicore` 时缺少 `libinfinicore_cpp_api.so`。这是当前原生运行环境问题，和本次纯注释修改无关。
