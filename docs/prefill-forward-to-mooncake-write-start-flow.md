# PD P 侧 Prefill Forward 完成到 Mooncake Write Start 的执行流程

## 1. 结论摘要

本文分析 P 侧从：

```text
forward_completed_at_ns
```

到：

```text
batch_transfer_sync_write_started_at_ns
```

之间的代码执行路径。两个时间戳都在 P 进程内，但不在同一个线程上执行。

核心结论：

1. `forward_completed_at_ns` 在 StepThread 中记录，位于 `model_engine.forward()` 返回后、sampled token 转 Python list 之前。
2. P 侧 prefill 使用 `skip_sampling`，forward 输出是一组 `-1` 哨兵 token，不是真实生成 token。
3. `_update_requests()` 看到 `-1` 后把 P 侧 request 标记为 `FINISHED`。
4. `Scheduler.complete_requests()` 调用 Mooncake connector scheduler 的 `request_finished()`，把该 request 的 KV block table 放入 `_reqs_need_send`。
5. 该 forward step 返回 pending token 后，StepThread 会立即进入下一轮 `engine.step()`；如果 waiting/running 队列为空，会生成一个 connector-only `SchedulerOutput`。
6. connector-only step 调用 `start_load_kv()`，把 P 侧 request 的 block ids 投递到 Mooncake sender loop。
7. sender loop 的 `record_send_reqs()` 填充 `SendBlockMeta.local_block_ids`，记录 `kv_cache_send_ready_at_ns`，并设置 `ready` event。
8. D 侧请求会并发地通过 ZMQ 把 `MooncakeXferMetadata` 发给 P 侧 sender listener。该 metadata 包含 D 侧 request id、transfer id、D 侧 block ids 和 D 侧 KV cache 地址。
9. P 侧 `_sender_worker()` 同时等待：
   - D 侧 ZMQ metadata 到达；
   - P 侧 `SendBlockMeta.ready` 被设置。
10. 两边汇合后，`_build_transfer_params()` 构造 src pointer、dst pointer 和 transfer length。
11. `_send_blocks()` 被提交到 sender ThreadPool。进入该函数后，在调用 `engine.batch_transfer_sync_write()` 前记录 `batch_transfer_sync_write_started_at_ns`。

因此，这个间隔不是单一函数耗时，而是跨过：

```text
StepThread
API event loop submission path
sender asyncio loop thread
D -> P ZMQ control plane
sender ThreadPool executor queue
```

的一条异步链路。

## 2. 时间戳和线程边界

| 时间戳 | 记录线程 | 代码位置 | 含义 |
|---|---|---|---|
| `forward_completed_at_ns` | StepThread | `ModelRunner._model_forward()` | `model_engine.forward()` 刚返回 |
| `kv_connector_finalize_*_at_ns` | StepThread | `ModelRunner.maybe_get_kv_connector_output()` | forward 后 connector finalize 控制面 |
| `loop_submit_started_at_ns` | StepThread | `AsyncLLMEngine._step_loop()` | pending output 提交 API loop 前 |
| `kv_cache_send_ready_at_ns` | Mooncake sender loop thread | `record_send_reqs()` | P 侧 transfer id 和 local block ids 已准备好 |
| `batch_transfer_sync_write_started_at_ns` | sender ThreadPool worker | `_send_blocks()` | 即将调用 Mooncake TransferEngine write |

注意：

1. `kv_cache_send_ready_at_ns` 不是 `request_finished()` 的调用时间，而是在下一轮 connector-only step 将 block ids 投递到 sender loop 后记录。
2. `batch_transfer_sync_write_started_at_ns` 不是 `ready` event 的触发时间，还包含 ZMQ metadata 等待、transfer params 构造和 executor 排队。
3. P 侧 KV transfer 不等待 P 内部 token 返回给 proxy 后才开始；StepThread 下一轮 connector-only step 和 API loop 输出路径可以并发执行。

## 3. 总体时序图

```text
P StepThread
  |
  | -- model_engine.forward()
  |
  |记录 forward_completed_at_ns
  |
  | -- sampled_tokens.to_numpy().tolist()
  | -- connector finally: get_finished/get_errors/get_stats
  | -- Scheduler.update_from_output()
  | -- record timing batch/token event
  | -- _update_requests()
  |       token_id == -1
  |       request.mark_finished()
  |
  | -- Scheduler.complete_requests()
  |       connector.request_finished()
  |       _reqs_need_send[p_req_id] = (request, block_table)
  |       delay_free_blocks = true
  |
  | -- pending token 提交 API loop
  |       记录 loop_submit_started_at_ns
  |       call_soon_threadsafe(_batch_put)
  |
  | -- 立即进入下一轮 engine.step()
  |       waiting/running 为空
  |       build_connector_meta()
  |       connector-only SchedulerOutput
  |       start_load_kv()
  |       run_coroutine_threadsafe(record_send_reqs)
  v

Mooncake sender loop thread
  |
  | -- record_send_reqs()
  |       SendBlockMeta.local_block_ids = block_ids
  |       记录 kv_cache_send_ready_at_ns
  |       SendBlockMeta.ready.set()
  v

等待 / 汇合
  |
  | P ready event
  + D side ZMQ MooncakeXferMetadata
  |
  | -- _sender_worker()
  | -- send_kv_to_decode()
  | -- asyncio.wait(SendBlockMeta.ready)
  | -- _build_transfer_params()
  | -- run_in_executor(_send_blocks)
  v

sender ThreadPool worker
  |
  | -- _send_blocks()
  |记录 batch_transfer_sync_write_started_at_ns
  |
  | -- engine.batch_transfer_sync_write(...)
  v
```

D 侧 ZMQ metadata 是并发到达的，不严格位于上图下方。它通常在 P forward 执行期间就已经到达并进入 P 侧 sender listener：

```text
Proxy 收到请求
  -> 同时发起 P request 和 D request
  -> D scheduler 发现 do_remote_prefill=true
  -> D start_load_kv()
  -> D 通过 ZMQ DEALER 向 P sender ROUTER 发送 MooncakeXferMetadata
  -> P _mooncake_sender_listener 将 metadata 放入 sender_worker_queue
```

因此，P forward 结束时，D metadata 可能已经在 `sender_worker_queue` 或 `send_kv_to_decode()` 中等待 P ready event。

## 4. StepThread 内部详细流程

### 4.1 forward 返回并记录完成时间

代码位置：

```text
python/infinilm/llm/model_runner/model_runner.py
ModelRunner._model_forward()
```

核心代码：

```python
sampled_tokens = self.model_engine.forward(**model_input)
forward_completed_at_ns = time.monotonic_ns()
sampled_tokens_list = sampled_tokens.to_numpy().tolist()
```

PD P 侧 prefill 满足：

```text
kv_role = kv_producer
do_remote_decode = true
```

因此 `_should_skip_prefill_sampling()` 返回 true，model input 中有：

```python
model_input["skip_sampling"] = True
```

C++ `skip_sample_output_ids()` 会生成 CPU 上的 `-1` 哨兵 token，并在创建 CPU tensor 前同步 GPU stream。这样 KV cache 在被 Mooncake 读取前已经完成写入。

`forward_completed_at_ns` 之后的第一件事是：

```python
sampled_tokens.to_numpy().tolist()
```

这部分计入 P 侧请求级 `forward complete -> loop submit`，但不属于 Mooncake transfer。

### 4.2 connector context 的 finally 阶段

代码位置：

```text
python/infinilm/llm/model_runner/model_runner.py
ModelRunner.maybe_get_kv_connector_output()
```

`execute_model()` 使用：

```python
with self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output:
    sampled_tokens_list, forward_timing = self._model_forward(...)
```

`_model_forward()` 返回后进入 context `finally`：

```python
get_finished()
get_block_ids_with_load_errors()
get_kv_connector_stats()
```

当前 P 侧报告中的量级为：

| 阶段 | avg |
|---|---:|
| forward -> finalize start | 约 `0.310 ms` |
| kv_connector_finalize | 约 `0.475 ms` |

这个阶段主要处理历史 transfer 的完成状态和 timing stats，不执行当前 request 的 KV write。

### 4.3 Scheduler 更新和 timing 记录

代码位置：

```text
python/infinilm/llm/llm.py
AsyncLLMEngineStepThread.step()
```

`execute_model()` 返回后执行：

```python
self.scheduler.update_from_output(runner_output)
for timing_event in self.scheduler.pop_timing_events():
    self.timing_recorder.record_event(...)
...
self.timing_recorder.record_batch(...)
```

这一步会把 connector worker 产生的 `kv_send` timing event 移交 timing recorder，并记录当前 forward batch event。

### 4.4 `_update_requests()` 处理 -1 哨兵 token

代码位置：

```text
python/infinilm/llm/llm.py
AsyncLLMEngineStepThread._update_requests()
```

对每个 P 侧 prefill request，如果 sampled token 小于 0：

```python
if not isinstance(token_ids, list) and token_ids < 0:
    req.finish_reason = FinishReason.LENGTH
    req.mark_finished(req.finish_reason)
```

然后构造一个内部 `TokenOutput`：

```text
token_id = -1
token_text = ""
finished = true
finish_reason = LENGTH
```

这个 output 只给 PD proxy 内部消费，不代表客户端真实生成 token。P 侧 proxy 请求被改写为：

```python
max_tokens = 1
stream = true
```

所以 P 侧内部请求只做 prefill，不做真实 decode。

### 4.5 `complete_requests()` 注册待发送 KV blocks

代码位置：

```text
python/infinilm/llm/scheduler.py
Scheduler.complete_requests()
```

请求已经 `FINISHED` 后：

```python
delay_free_blocks, _ = self.connector.request_finished(
    req, req.block_table, self.block_size
)
```

Mooncake scheduler 侧：

```text
python/infinilm/kv_connector/mooncake/mooncake_connector_scheduler.py
MooncakeConnectorScheduler.request_finished()
```

核心逻辑：

```python
if request.status != RequestStatus.FINISHED:
    self._reqs_not_processed.add(transfer_id)
    return False, None

delay_free_blocks = len(block_ids) > 0
if delay_free_blocks:
    self._reqs_need_send[request.request_id] = (
        request,
        block_ids,
    )
return delay_free_blocks, None
```

含义：

1. P 侧 request 必须已经 `FINISHED`；
2. `block_ids` 是当前 request 的 P 侧 KV block table；
3. `_reqs_need_send` 是 connector scheduler 的待发送集合；
4. 返回 `delay_free_blocks=true` 后，这些 blocks 被放入 `pending_free_blocks`，直到 transfer 完成才释放。

这一步还没有把 request 交给 sender loop。

### 4.6 当前 pending token 提交 API loop

`_update_requests()` 返回 pending output 后，StepThread 在：

```text
python/infinilm/llm/llm.py
AsyncLLMEngine._step_loop()
```

执行：

```python
loop_submit_started_at_ns = time.monotonic_ns()
self._loop.call_soon_threadsafe(self._batch_put, pending)
```

随后 API event loop 会：

```text
_batch_put()
stream_request()
StreamingResponse
proxy send_request() 消费 P 内部响应
```

这条输出路径与下一轮 connector-only step 并发。P 侧 KV transfer 不需要等 proxy 完整消费 P 响应后才开始。

## 5. 下一轮 connector-only step：进入 sender loop

### 5.1 StepThread 立即进入下一轮 `engine.step()`

当前 prefill step 返回 `did_work=true` 后，`_step_loop()` 不进入 no-work wait，而是继续下一轮：

```python
did_work, pending = self.engine.step(...)
...
while True:
    ...
    did_work, pending = self.engine.step(...)
```

如果 P 侧 waiting 和 running 队列为空，`Scheduler.schedule()` 返回 connector-only output：

```python
scheduler_output = SchedulerOutput(
    scheduled_requests=[],
    speculative_cache_ops=...,
)
meta = self.connector.build_connector_meta()
scheduler_output.kv_connector_metadata = meta
```

`build_connector_meta()` 会把 `_reqs_need_send` 转成 `MooncakeConnectorMetadata`：

```python
for req_id, (req, block_ids) in self._reqs_need_send.items():
    meta.add_new_req(
        request_id=req_id,
        local_block_ids=block_ids,
        kv_transfer_params=req.kv_transfer_params,
        load_remote_cache=False,
    )
self._reqs_need_send.clear()
```

### 5.2 `start_load_kv()` 投递到 sender loop

connector-only step 没有 scheduled request，因此 `execute_model()` 不执行 GPU forward，但 context manager 仍会执行：

```python
self.kv_connector.bind_connector_metadata(...)
self.kv_connector.start_load_kv()
```

P 侧 `start_load_kv()`：

```text
python/infinilm/kv_connector/mooncake/mooncake_connector_worker.py
MooncakeConnectorWorker.start_load_kv()
```

执行：

```python
asyncio.run_coroutine_threadsafe(
    self.record_send_reqs(metadata), self.sender_loop
)
```

这跨越了第一个重要线程边界：

```text
StepThread -> Mooncake sender asyncio loop thread
```

### 5.3 `record_send_reqs()` 设置 ready 和 KV ready 时间戳

代码位置：

```text
python/infinilm/kv_connector/mooncake/mooncake_connector_worker.py
MooncakeConnectorWorker.record_send_reqs()
```

核心逻辑：

```python
send_meta = self.reqs_need_send[transfer_id]
send_meta.p_req_id = p_req_id
send_meta.local_block_ids = block_ids
send_meta.expire_time = time.perf_counter() + 480

if self._timing_enabled:
    send_meta.kv_cache_send_ready_at_ns = time.monotonic_ns()

send_meta.ready.set()
```

`kv_cache_send_ready_at_ns` 的准确含义是：

```text
P 侧 sender loop 已经拿到 transfer_id 对应的 local KV block ids，
并把这个 SendBlockMeta 标记为 ready。
```

它不是：

```text
forward 完成时间
request_finished() 调用时间
D metadata 到达时间
Mooncake write 开始时间
```

## 6. D 侧 ZMQ control plane 如何到达 P

### 6.1 Proxy 并发发起 P 和 D

Proxy 收到客户端请求后：

```python
asyncio.create_task(send_request(...))  # P request
return StreamingResponse(generate_stream())  # D request
```

P request 被改写为：

```text
do_remote_prefill=false
do_remote_decode=true
max_tokens=1
```

D request 被改写为：

```text
do_remote_prefill=true
do_remote_decode=false
remote_bootstrap_addr=P bootstrap
remote_engine_id=P engine id
transfer_id=xfer-<request-id>
```

### 6.2 D scheduler 生成 pull metadata

D 侧第一次调度该 request 时，connector scheduler 的 `get_num_new_matched_tokens()` 返回异步远端 KV 加载：

```python
return count, True
```

随后 D 侧 `start_load_kv()` 在 receiver loop 中执行：

```text
_start_load_kv()
receive_kv_from_single_worker()
```

构造 `MooncakeXferMetadata`：

```python
MooncakeXferMetadata(
    remote_hostname=D hostname,
    remote_port=D TransferEngine rpc port,
    remote_tp_size=...,
    remote_tp_rank=...,
    req_blocks={
        d_req_id: (transfer_id, d_local_block_ids)
    },
    kv_caches_base_addr=D KV cache base addresses,
)
```

并通过 ZMQ DEALER 发给 P 侧 sender ROUTER：

```python
await sock.send(encoded_data)
```

### 6.3 P sender listener 接收 metadata

P 侧 sender loop 中：

```text
_mooncake_sender_listener()
```

监听 ZMQ ROUTER socket：

```python
identity, metadata_bytes = await sock.recv_multipart()
await self.sender_worker_queue.put((identity, metadata_bytes))
```

然后 `_sender_worker()` 从 queue 取出 metadata：

```python
metadata = self._xfer_meta_decoder.decode(metadata_bytes)
await self.send_kv_to_decode(identity, sock, metadata)
```

## 7. P ready 与 D metadata 汇合

### 7.1 查找或创建 `SendBlockMeta`

`send_kv_to_decode()` 遍历 D metadata：

```python
for d_req_id, (transfer_id, _) in meta.req_blocks.items():
    if transfer_id not in self.reqs_need_send:
        self.reqs_need_send[transfer_id] = SendBlockMeta(
            p_req_id="",
            transfer_id=transfer_id,
            local_block_ids=[],
            ready=asyncio.Event(),
        )
    send_meta = self.reqs_need_send[transfer_id]
```

存在两种到达顺序：

1. D metadata 先到：P 先创建未 ready 的 `SendBlockMeta`，等待 P 侧 `record_send_reqs()` 设置 local block ids 和 ready event。
2. P ready 先到：`record_send_reqs()` 需要访问已存在的 `SendBlockMeta`。当前实现假设 D metadata 已先到达。

当前部署中 P prefill forward 通常明显长于 D 侧 API enqueue 和 ZMQ metadata 发送，所以常见顺序是：

```text
D metadata 先到
P forward 完成
P next connector-only step 设置 ready
```

### 7.2 等待 ready event

每个 request 创建一个等待 task：

```python
async def wait_and_ret(...):
    await send_meta.ready.wait()
    return d_req_id, send_meta
```

然后：

```python
done_tasks, pending_tasks = await asyncio.wait(
    wait_tasks,
    timeout=480,
    return_when=asyncio.FIRST_COMPLETED,
)
```

只要 ready event 被设置，sender task 就继续。

## 8. 构造 transfer 参数

ready 后执行：

```python
src_ptrs, dst_ptrs, lengths, ... = await self._build_transfer_params(
    ready_reqs, meta
)
```

主要步骤：

1. 比较 P local blocks 和 D remote blocks；
2. P block 数多于 D 请求 block 数时，只取最后 `num_remote_blocks` 个；
3. 对 local/remote block id 做连续段分组；
4. 对每个 KV layer / cache base address 生成：

```python
src_ptr = local_layer_addr + local_block_id * block_len
dst_ptr = remote_layer_addr + remote_block_id * block_len
length = block_len * len(contiguous_block_group)
```

如果 P local blocks 少于 D remote blocks，会记录 address mismatch，不执行 transfer。

这个阶段还没有开始 Mooncake write。

## 9. 提交 sender ThreadPool 并记录 write start

sender loop 执行：

```python
ret_value = await self.sender_loop.run_in_executor(
    self._sender_executor,
    self._send_blocks,
    remote_session,
    src_ptrs,
    dst_ptrs,
    lengths,
    ...
)
```

默认：

```text
num_sender_workers = 10
num_sender_tasks = 20
```

因此还存在：

```text
sender asyncio loop -> sender ThreadPool executor queue
```

这一排队时间没有单独 timestamp。

进入 sender ThreadPool worker 后，`_send_blocks()` 开头执行：

```python
started_at_ns = time.monotonic_ns()
```

如果 timing enabled，该值最终写入：

```text
batch_transfer_sync_write_started_at_ns
```

然后才调用：

```python
self.engine.batch_transfer_sync_write(
    remote_session,
    src_ptrs,
    dst_ptrs,
    lengths,
)
```

所以 `kv ready -> write start` 包含：

1. D metadata 等待或 P ready 等待；
2. `_build_transfer_params()`；
3. sender executor 提交；
4. sender ThreadPool 队列等待；
5. 进入 `_send_blocks()` 后的少量函数开销。

当前修正后 PD v2 报告中：

| 指标 | avg | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| KV ready -> write start | `5.203 ms` | `4.912 ms` | `8.569 ms` | `9.210 ms` | `9.228 ms` |

它不是纯 ZMQ 延迟，也不是纯 executor 排队时间。

## 10. 当前 timing 的观测盲区

当前 schema 可以直接观测：

```text
forward_completed_at_ns
kv_connector_finalize_started/completed_at_ns
loop_submit_started_at_ns
kv_cache_send_ready_at_ns
batch_transfer_sync_write_started/completed_at_ns
```

但 `forward_completed_at_ns -> batch_transfer_sync_write_started_at_ns` 中间还缺少：

| 缺失时间戳 | 价值 |
|---|---|
| `requests_updated_at_ns` | 确定 `_update_requests()` 完成时间 |
| `request_finished_completed_at_ns` | 确定 connector scheduler 注册 `_reqs_need_send` 的完成时间 |
| `connector_meta_built_at_ns` | 确定下一轮 connector-only metadata 构建时间 |
| `record_send_ready_scheduled_at_ns` | 确定 `run_coroutine_threadsafe()` 提交时间 |
| `zmq_metadata_received_at_ns` | 确定 D metadata 到达 P sender listener 的时间 |
| `sender_worker_dequeued_at_ns` | 确定 sender queue 等待时间 |
| `ready_wait_completed_at_ns` | 确定 P ready / D metadata 汇合时间 |
| `transfer_params_built_at_ns` | 确定 src/dst/length 构造耗时 |
| `executor_submitted_at_ns` | 确定 ThreadPool 提交时间 |

另外，P 侧 `kv_send` event 使用 D request id，而 P forward batch event 使用 P request id。当前 event 中没有直接暴露 transfer id 到 P forward request 的映射，因此无法严谨地按 request id 把：

```text
P forward_completed_at_ns
```

与：

```text
kv_cache_send_ready_at_ns
```

逐 request 相减。做整体阶段分析时必须避免伪造 request-level 关联。

## 11. 阶段划分总结

| 阶段 | 线程 / 执行体 | 是否阻塞 GPU forward | 是否阻塞 P 输出 |
|---|---|---|---|
| sampled token 转 Python list | StepThread | 否，forward 已返回 | 是，位于 loop submit 前 |
| connector finalize | StepThread | 否 | 是 |
| scheduler update / timing record | StepThread | 否 | 是 |
| `_update_requests()` 标记 FINISHED | StepThread | 否 | 是 |
| `request_finished()` 注册待发送 blocks | StepThread | 否 | 是 |
| pending token 提交 API loop | StepThread -> API loop | 否 | 是 |
| next connector-only step | StepThread | 否 | 否，与 API 输出并发 |
| `record_send_reqs()` 设置 ready | sender loop | 否 | 否 |
| D ZMQ metadata 发送和接收 | D receiver loop -> P sender loop | 否 | 否 |
| ready / metadata 汇合 | sender loop | 否 | 否 |
| transfer params 构造 | sender loop | 否 | 否 |
| executor 提交和排队 | sender loop -> ThreadPool | 否 | 否 |
| `batch_transfer_sync_write()` | sender ThreadPool | 否，可与后续 P forward 并发 | 否 |

最终可概括为：

```text
prefill forward 完成
    -> P request 产生 -1 内部哨兵 token
    -> request 标记 FINISHED
    -> block table 注册到 connector scheduler
    -> next connector-only step
    -> sender loop 填充 local blocks 并设置 ready
    -> 与 D ZMQ metadata 汇合
    -> 构造 src/dst/length
    -> 提交 sender ThreadPool
    -> batch_transfer_sync_write_started_at_ns
```
