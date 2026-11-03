
> 查看/workspace_codex/vllm-0.32.0开源仓库。vllm中model running和服务之间也是进程隔离么？二者通过什么通信？请分析vllm的代码。

<details><summary>4 previous messages</summary>

> 我先检查 `vllm-0.32.0` 的启动入口、V1 `EngineCore`/`AsyncLLM` 结构，以及 multiprocessing、ZeroMQ、shared memory 相关实现，确认“服务进程”和“model runner/执行进程”在 vLLM 中的实际边界。重点会区分 API server、AsyncLLMEngine、EngineCore、GPU worker 和 model runner，不把这些层级混为一谈。
>
> <details><summary>Explored 4 searches, 2 lists, ran 2 commands</summary>
>
> <details><summary>Explored a search</summary>
>
> - Searched for `EngineCore|ModelRunner|multiprocessing|Process|ZeroMQ|zmq|shared_memory|SharedMemory|EngineCoreClient|AsyncLLM|GPUWorker|worker`
>
> </details>
>
> - Ran `git status --short --branch && git log -1 --oneline`
>
> <details><summary>Explored 2 searches, a list</summary>
>
> - Listed files
> - Searched for `v1/(engine|core|worker|gpu_model_runner)|entrypoints|executor|multiproc|zmq|message_queue|client`
> - Searched for `VLLM_ENABLE_V1_MULTIPROCESSING|multiprocessing|EngineCoreClient|MPClient|EngineCoreProc|run_engine_core|Zmq|zmq|MessageQueue|SharedMemory` in `v1`
>
> </details>
>
> - Ran `nl -ba vllm/v1/outputs.py | sed -n '200,430p'; nl -ba vllm/v1/inputs.py 2>/dev/null | head -100`
>
> <details><summary>Explored a search, a list</summary>
>
> - Listed files in `v1`
> - Searched for `engine|core|worker|executor|utils.py|outputs.py`
>
> </details>
>
> </details>
>
> 关键证据已经找到：vLLM V1 明确把 `ModelRunnerOutput` 标注为“serialized and sent to the scheduler process”，说明它确实支持 scheduler 与执行侧进程隔离。接下来我会补齐两条链路：API server 到 `EngineCoreClient` 的通信，以及 EngineCore 到 GPU worker/model runner 的通信，确认 vLLM 是一层隔离还是多层隔离。
>
> <details><summary>Explored 17 files, a search</summary>
>
> - Read `./vllm/v1/engine/llm_engine.py`
> - Read `./vllm/v1/engine/core_client.py`
> - Read `./vllm/v1/engine/core.py`
> - Read `./vllm/v1/engine/utils.py`
> - Read `./vllm/v1/executor/multiproc_executor.py`
> - Read `./vllm/v1/engine/tensor_ipc.py`
> - Read `./vllm/v1/engine/core.py`
> - Read `./vllm/v1/engine/utils.py`
> - Read `./vllm/v1/engine/core.py`
> - Read `./vllm/v1/executor/multiproc_executor.py`
> - Searched for `execute_model\(|GPUModelRunner|model_runner` in `core.py`
> - Read `./vllm/entrypoints/launchers/api_server/entry.py`
> - Read `./vllm/entrypoints/cli/serve.py`
> - Read `./vllm/envs.py`
> - Read `./vllm/v1/engine/core.py`
> - Read `./vllm/v1/engine/core_client.py`
> - Read `./vllm/distributed/device_communicators/shm_broadcast.py`
> - Read `./vllm/v1/executor/multiproc_executor.py`
>
> </details>
</details>

是的，**vLLM 0.32.0 的 V1 架构中，服务层、EngineCore/scheduler、GPU worker/model runner 通常是进程隔离的**。而且不是简单的两进程，而是多层进程结构。

## 1. 进程结构

典型 V1 架构如下：

```text
API Server 进程
├── HTTP / OpenAI API
├── 请求解析、tokenizer、detokenizer
└── EngineCoreClient
        │ ZMQ IPC
        ▼
EngineCore 进程
├── Scheduler
├── KV cache block 管理
├── 请求状态管理
└── MultiprocExecutor
        │ shared-memory MessageQueue / IPC
        ▼
GPU Worker 进程
├── GPUWorker
├── GPUModelRunner
├── 模型权重
├── GPU KV cache
└── GPU forward / sampling
```

如果 tensor parallel、data parallel 或多节点配置开启，EngineCore 和 GPU Worker 进程数量还会增加。

vLLM 的代码明确说明：

```python
* InprocClient: In process EngineCore
* SyncMPClient: ZMQ + background proc EngineCore
* AsyncMPClient: ZMQ + background proc EngineCore
```

位置：[core_client.py:80-88](./vllm-0.32.0/vllm/v1/engine/core_client.py:80)

## 2. API Server 与 EngineCore 是否隔离？

是。

vLLM 通过 `EngineCoreClient` 访问 EngineCore。多进程模式下，`EngineCoreClient` 使用：

```python
SyncMPClient
AsyncMPClient
```

位置：[core_client.py:95-133](./vllm-0.32.0/vllm/v1/engine/core_client.py:95)

API server 侧不会直接执行 scheduler 或 GPU forward，而是发送 `EngineCoreRequest`：

```text
API Server
    -> EngineCoreRequest
    -> EngineCore
    -> EngineCoreOutputs
```

API server 启动时会先创建 ZMQ listener，再启动 EngineCore：

```python
zmq_listeners = bind_engine_zmq_listeners(...)
with launch_core_engines(...):
    ...
```

位置：[serve.py:331-339](./vllm-0.32.0/vllm/entrypoints/cli/serve.py:331)

API server 本身也由独立进程管理器启动：

```python
APIServerProcessManager(...)
```

位置：[serve.py:367-377](./vllm-0.32.0/vllm/entrypoints/cli/serve.py:367)

所以：

```text
HTTP server 的 CPU/GIL 负载
不会直接占用 EngineCore 进程的 Python GIL
```

但它们仍可能共享物理 CPU，除非配置 CPU affinity 或 cgroup。

## 3. API Server 与 EngineCore 使用什么通信？

主要是 **ZeroMQ + MessagePack**。

EngineCoreClient 的说明已经直接写明：

```python
SyncMPClient: ZMQ + background proc EngineCore
AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio
```

EngineCore 进程中建立 ZMQ `DEALER` socket：

```python
make_zmq_socket(
    ctx,
    input_address,
    zmq.DEALER,
    identity=identity,
    bind=False,
)
```

位置：[core.py:1845-1851](./vllm-0.32.0/vllm/v1/engine/core.py:1845)

EngineCore 进程收到请求后：

```python
type_frame, *data_frames = input_socket.recv_multipart(copy=False)
request = add_request_decoder.decode(data_frames)
self.input_queue.put_nowait((request_type, request))
```

位置：[core.py:1887-1931](./vllm-0.32.0/vllm/v1/engine/core.py:1887)

输出则通过 ZMQ `PUSH` socket 发回 API/Client：

```python
socket.send_multipart(encoder.encode(outputs))
```

位置：[core.py:1949-1995](./vllm-0.32.0/vllm/v1/engine/core.py:1949)

本机优先使用 Unix IPC 地址，而不是 TCP 网络地址。具体地址由：

```python
get_open_zmq_ipc_path()
```

生成。

因此 API Server 和 EngineCore 的主要数据路径是：

```text
API Server
  -> ZMQ DEALER/ROUTER
  -> MessagePack
  -> EngineCore

EngineCore
  -> ZMQ PUSH/PULL
  -> MessagePack
  -> API Server
```

## 4. Scheduler 与 ModelRunner 是否隔离？

是，V1 中 scheduler 位于 EngineCore 进程，GPUModelRunner 位于 GPU Worker 进程。

EngineCore 的 step 流程是：

```python
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(
    scheduler_output,
    non_block=True,
)
...
model_output = future.result()
...
self.scheduler.update_from_output(
    scheduler_output,
    model_output,
)
```

位置：[core.py:657-687](./vllm-0.32.0/vllm/v1/engine/core.py:657)

这里的 `self.scheduler` 在 EngineCore 进程中，而 `self.model_executor` 在多进程模式下将请求转发给 GPU Worker。

GPU Worker 的 `execute_model()` 最终调用：

```python
self.model_runner.execute_model(...)
```

代码搜索链路为：

```text
EngineCore.step()
  -> MultiprocExecutor.execute_model()
  -> Worker.execute_model()
  -> GPUWorker.execute_model()
  -> GPUModelRunner.execute_model()
```

`MultiprocExecutor.execute_model()` 明确使用 RPC：

```python
return self.collective_rpc(
    "execute_model",
    args=(scheduler_output,),
    ...
)
```

位置：[multiproc_executor.py:341-352](./vllm-0.32.0/vllm/v1/executor/multiproc_executor.py:341)

GPU worker 进程由 `WorkerProc` 单独创建：

```python
class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""
```

位置：[multiproc_executor.py:607-609](./vllm-0.32.0/vllm/v1/executor/multiproc_executor.py:607)

## 5. EngineCore 与 GPU Worker 使用什么通信？

这部分主要不是 ZMQ，而是 **共享内存 MessageQueue**。

初始化时，EngineCore 创建：

```python
self.rpc_broadcast_mq = MessageQueue(...)
scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
```

位置：[multiproc_executor.py:144-171](./vllm-0.32.0/vllm/v1/executor/multiproc_executor.py:144)

GPU Worker 进程从 handle 恢复队列：

```python
self.rpc_broadcast_mq = MessageQueue.create_from_handle(
    input_shm_handle,
    self.worker.rank,
)
```

位置：[multiproc_executor.py:614-620](./vllm-0.32.0/vllm/v1/executor/multiproc_executor.py:614)

同时每个 worker 创建 response MessageQueue：

```python
self.worker_response_mq = MessageQueue(1, 1)
```

位置：[multiproc_executor.py:621-624](./vllm-0.32.0/vllm/v1/executor/multiproc_executor.py:621)

因此本机的 EngineCore 到 GPU Worker 路径近似为：

```text
EngineCore scheduler
    -> shared-memory broadcast MessageQueue
    -> GPU Worker
    -> worker response MessageQueue
    -> EngineCore
```

这种设计避免了把每个 scheduler batch 通过普通 TCP 或 Python pickle 完整复制一遍。

## 6. vLLM 是否也会序列化 ModelRunner 输入输出？

是。

vLLM 的 `ModelRunnerOutput` 中明确写着：

```python
# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
```

位置：[outputs.py:264-267](./vllm-0.32.0/vllm/v1/outputs.py:264)

输出中包含：

```python
req_ids
sampled_token_ids
kv_connector_output
logprobs
pooler_output
```

位置：[outputs.py:267-310](./vllm-0.32.0/vllm/v1/outputs.py:267)

vLLM 对 tensor 输出会尽量采用：

- list；
- shared memory；
- tensor IPC；
- zero-copy buffer；
- 或其他专门的 tensor transport。

例如多模态 tensor 使用 `torch.multiprocessing.Queue`：

```python
# Send tensors via torch.multiprocessing.Queue for zero-copy IPC.
```

位置：[tensor_ipc.py:4-10](./vllm-0.32.0/vllm/v1/engine/tensor_ipc.py:4)

发送侧会将 tensor 放入共享内存：

```python
if not tensor.is_shared():
    tensor = tensor.share_memory_()
```

位置：[tensor_ipc.py:68-87](./vllm-0.32.0/vllm/v1/engine/tensor_ipc.py:68-87)

## 7. vLLM 如何解决 scheduler 与 ModelRunner 的状态边界？

vLLM 没有直接跨进程传递整个 Python 对象图，而是建立了可序列化的数据协议。

ModelRunner 输出使用：

```python
req_ids
req_id_to_index
sampled_token_ids
kv_connector_output
```

而不是把 worker 内部的 request 对象、GPU tensor、model 状态直接返回给 scheduler。

这和 InfiniLM 当前的差异很明显：

### InfiniLM 当前

```text
SchedulerOutput
├── InferenceRequest 对象
├── SpeculativeCacheOps
├── BlockManager 隐式状态
├── timing_steps 可变对象
└── kv_connector_metadata
```

ModelRunner 会直接修改 scheduler 中的 request：

```python
request.timing_steps[-1].engine_forward_started_at_ns = ...
```

### vLLM V1

```text
Scheduler process
    -> 序列化 batch descriptor
    -> worker process
    -> 序列化 ModelRunnerOutput
    -> scheduler process
```

因此，vLLM 的架构证明了：

> ModelRunner 与 scheduler 进程隔离是可行的，但前提是输入输出必须设计成显式的、可序列化的协议，而不是跨进程共享 Python 对象。

## 8. vLLM 中到底有几层隔离？

需要区分配置。

### 默认 V1 多进程模式

环境变量默认值为：

```python
VLLM_ENABLE_V1_MULTIPROCESSING=1
```

位置：[envs.py:1462-1465](./vllm-0.32.0/vllm/envs.py:1462)

通常是：

```text
API Server
  独立进程

EngineCore
  独立进程，负责 scheduler

GPU Worker
  一个或多个独立进程，负责 GPUModelRunner
```

### 关闭多进程

如果设置：

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

则可以使用 `InprocClient`，EngineCore 运行在调用进程内：

```python
if multiprocess_mode:
    return SyncMPClient(...)
return InprocClient(...)
```

位置：[core_client.py:117-133](./vllm-0.32.0/vllm/v1/engine/core_client.py:117)

但即使如此，使用 `MultiprocExecutor` 时，GPU Worker 仍可能是独立进程。

## 9. 与 InfiniLM 的对应关系

vLLM 的设计可以对应到 InfiniLM：

| vLLM | InfiniLM 当前实现 |
|---|---|
| API Server | `InferenceServer` / FastAPI |
| EngineCore | `AsyncLLMEngineStepThread` + `LLMEngine` |
| Scheduler | `python/infinilm/llm/scheduler.py` |
| GPU Worker | 当前没有独立于 server 的完整 worker 进程 |
| GPUModelRunner | `ModelRunner` |
| MessageQueue/ZMQ IPC | 当前主要是同进程线程、asyncio 和 ThreadPoolExecutor |
| ModelRunnerOutput | `ModelRunnerOutput`，但当前跨线程直接返回 |
| KV connector | ModelRunner 内部创建并持有 |

所以当前 InfiniLM 更接近：

```text
InferenceServer 进程
├── FastAPI event loop
└── AsyncLLMEngineStepThread
    ├── Scheduler
    ├── ModelRunner
    ├── GPU forward
    └── Mooncake connector
```

而 vLLM V1 更接近：

```text
API Server 进程
    │ ZMQ
    ▼
EngineCore 进程
    ├── Scheduler
    │   │ shared-memory MessageQueue
    │   ▼
    │ GPU Worker 进程
    │   └── GPUModelRunner
    └── outputs
```

## 最终结论

vLLM 0.32.0 V1 中：

1. **服务层和 EngineCore/scheduler 通常是进程隔离的**；
2. **EngineCore/scheduler 和 GPUModelRunner 也通常是进程隔离的**；
3. API Server 与 EngineCore 主要使用 **ZeroMQ IPC + MessagePack**；
4. EngineCore 与 GPU Worker 主要使用 **共享内存 MessageQueue**，并辅以 pipe/ZMQ 通知；
5. GPU/多模态 tensor 使用专门的 tensor IPC 或共享内存机制；
6. `ModelRunnerOutput` 被设计为可序列化的跨进程协议；
7. 该架构可以减少 HTTP server、scheduler 和 model execution 之间的 GIL 竞争，但 CPU affinity 仍然需要单独配置。

因此，InfiniLM 如果要参考 vLLM，最合理的方向不是只把 `model_runner.py` 文件移动到新进程，而是引入类似：

```text
Server/API process
    -> Scheduler/EngineCore process
    -> ModelRunner/GPU worker process
```

并为 `SchedulerOutput`、`ModelRunnerOutput` 和 KV connector completion 建立明确的 IPC 协议。

> **EngineCore/scheduler 和 GPUModelRunner 也通常是进程隔离的**； &#x20;
> ModelRunner 与 scheduler 进程隔离是可行的，但前提是输入输出必须设计成显式的、可序列化的协议。