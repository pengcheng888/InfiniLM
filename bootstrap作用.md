
> 每个prefill都要创建一个bootstrap，这个bootstrap的作用是什么意思？

这里的 **bootstrap 不是“再启动一个推理服务”**，而是 **prefill 侧为 Mooncake KV transfer 提供的控制面查询服务**。可以把它理解为：**告诉 decode “去哪里、找哪个 engine 拉取 prefill 发布的 KV”**。

**整体作用**

```text
prefill engine 启动
  -> Mooncake connector 创建 KV transfer engine
  -> prefill 暴露 bootstrap /query 服务
  -> frontend 查询 bootstrap，拿到 engine_id
  -> frontend 发起 P/D 请求
       prefill 计算 prompt 并发布 KV
       decode 根据 bootstrap_addr + engine_id 找到 prefill transfer engine
       decode 通过 Mooncake TCP/RDMA 拉取 KV
       decode 继续增量生成
```

它涉及三类地址，语义不同：

| 地址 | 作用 |
|---|---|
| prefill HTTP `host:port` | frontend 调用 OpenAI-compatible internal chat API，触发 prefill |
| prefill bootstrap `host:port` | 查询 Mooncake KV transfer 元数据，例如 engine ID |
| Mooncake TCP/RDMA 通道 | 真正传输 KV 数据，bootstrap 本身不传 KV |

**为什么每个 prefill 都有自己的 bootstrap**

1. **每个 prefill 都有独立的 KV transfer engine**

   多 prefill 时，每个 prefill 进程都有自己的模型副本、KV cache、Mooncake connector 和 `engine_id`。frontend 路由到某个 prefill 后，decode 必须知道“这个 prefill 对应哪个 transfer engine”。

2. **避免 KV 来源混淆**

   例如：

   ```text
   prefill-0 -> bootstrap 18528 -> engine-id prefill-0
   prefill-1 -> bootstrap 18529 -> engine-id prefill-1
   ```

   如果请求选中了 `prefill-1`，decode 就必须拿 `prefill-1` 的 bootstrap 地址和 engine ID。否则可能去找 `prefill-0` 的 KV，导致拉错 KV 或拉不到 KV。

3. **支持生命周期隔离**

   bootstrap 随 prefill 进程一起启动和退出。prefill 挂掉后，它的 bootstrap 也不可用，frontend 后台 reconciler 就不会把这个 prefill 加入 routable pool。这样避免把请求路由到已经不可发布的 KV 源。

4. **它是控制面，开销较小**

   bootstrap 不加载模型、不生成 token、不传输大块 KV，只提供 `/query` 这类元数据查询接口。真正耗带宽的是 Mooncake 数据面。

**代码中的位置**

- prefill 把 bootstrap 地址写入 registry metadata：
  [worker.py](./InfiniLM/python/infinilm/cluster/worker.py:277)

- frontend 初始化 prefill client 时查询 `/query`，保存 engine ID 映射：
  [worker_client_pool.py](./InfiniLM/python/infinilm/cluster/disaggregated/worker_client_pool.py:113)

- decode 请求会携带：

```python
{
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_bootstrap_addr": prefill_client_info["bootstrap_addr"],
    "remote_engine_id": prefill_client_info["dp_engine_id"][prefill_dp_rank],
    "transfer_id": f"xfer-{request_id}",
}
```

见 [pd_http_transport.py](./InfiniLM/python/infinilm/cluster/disaggregated/pd_http_transport.py:190)。

**一句话总结**

bootstrap 是 prefill Mooncake transfer engine 的“服务发现入口”。frontend 用它确认选中的 prefill 对应哪个 KV engine；decode 用它提供的地址和 engine ID 去 Mooncake 拉取该 prefill 发布的 KV。没有 bootstrap，decode 就不知道该从哪个 prefill engine 拉 KV，PD 分离推理链路无法成立。

> cluster中的etcd和file发现，可以替代bootstrap的功能么？请分析。

<details><summary>2 previous messages</summary>

> 我会先看一下 Mooncake connector 对 `remote_bootstrap_addr` 和 `/query` 的实际使用位置，再判断 file/etcd 能替代到什么程度。关键是区分“静态服务发现元数据”和“运行期 KV transfer 元数据”。
>
> <details><summary>Explored 3 files, 2 searches</summary>
>
> - Searched for `remote_bootstrap_addr|remote_engine_id|/query|bootstrap` in `kv_connector`
> - Read `./python/infinilm/kv_connector/mooncake/mooncake_connector_worker.py`
> - Searched for `class .*Bootstrap|start_bootstrap|bootstrap_server|query` in `mooncake_connector_worker.py`
> - Read `./python/infinilm/kv_connector/mooncake/mooncake_utils.py`
> - Read `./python/infinilm/kv_connector/mooncake/mooncake_connector_worker.py`
>
> </details>
</details>

不能直接替代。当前设计里 **file/etcd discovery 和 Mooncake bootstrap 是两层不同的发现系统**，前者发现“推理服务”，后者发现“KV transfer worker”。

**两层职责**

| 层 | file/etcd worker registry | Mooncake bootstrap |
|---|---|---|
| 发现对象 | prefill/decode HTTP 服务进程 | prefill 内部的 Mooncake transfer agent |
| 典型信息 | `host:port`、model、ready/draining、load、bootstrap 地址 | `dp_rank -> engine_id + worker_addr` |
| 消费者 | frontend router / client pool | decode 的 Mooncake connector |
| 请求面 | frontend -> worker HTTP OpenAI API | decode -> prefill Mooncake transfer worker |
| 生命周期 | worker heartbeat/lease | prefill engine 内部启动、注册、退出 |

当前流程是：

```text
file/etcd registry
  发现 prefill HTTP 服务
  并告诉 frontend prefill 的 bootstrap_host/bootstrap_port
          |
          v
frontend 查询 bootstrap /query
  得到 dp_rank -> engine_id
          |
          v
frontend 把 remote_bootstrap_addr + remote_engine_id 发给 decode
          |
          v
decode Mooncake connector 再查询 bootstrap /query
  得到真正 Mooncake worker_addr
          |
          v
decode 通过 Mooncake TCP/RDMA 拉取 KV
```

代码上，bootstrap 返回的不只是 `engine_id`，还包含每个 DP/TP/PP rank 的 transfer worker 地址：

```python
@dataclass
class EngineEntry:
    engine_id: EngineId
    # {tp_rank: {pp_rank: worker_addr}}
    worker_addr: dict[int, dict[int, WorkerAddr]]
```

见 [mooncake_utils.py](./InfiniLM/python/infinilm/kv_connector/mooncake/mooncake_utils.py:46)。

decode 侧真正依赖的是这个 `worker_addr`：

```python
async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr):
    url = remote_bootstrap_addr + "/query"
    ...
    self._remote_agents[remote_engine_id] = ...
```

见 [mooncake_connector_worker.py](./InfiniLM/python/infinilm/kv_connector/mooncake/mooncake_connector_worker.py:1017)。

**为什么 file/etcd 现在不能直接替代**

1. **当前 registry 没有 `worker_addr`**

   cluster registry 里的 PD metadata 主要是：

   ```text
   bootstrap_host
   bootstrap_port
   engine_id
   kv_role
   ```

   见 [worker.py](./InfiniLM/python/infinilm/cluster/worker.py:277)。

   但 decode 拉 KV 需要的是 Mooncake transfer worker 地址，也就是 bootstrap `/query` 返回的：

   ```text
   dp_rank -> engine_id + {tp_rank -> {pp_rank -> worker_addr}}
   ```

2. **Mooncake connector 当前只支持查询 HTTP bootstrap**

   decode 收到的参数是：

   ```text
   remote_bootstrap_addr
   remote_engine_id
   ```

   然后它内部访问：

   ```text
   {remote_bootstrap_addr}/query
   ```

   它不会去读 file JSON，也不会直接查 etcd。

3. **生命周期不同**

   worker registry 表示“这个 prefill HTTP 服务是否存活”。
   bootstrap 表示“这个 prefill 进程内的 Mooncake transfer agent 是否已经注册完成”。

   一个 prefill HTTP 服务可能已经注册到 file/etcd，但某个 TP/PP transfer worker 还没完成 Mooncake 注册。此时直接用 registry 元数据会有竞态。

4. **file backend 本身只适合同机**

   file registry 只是本机 JSON 文件，不能作为跨节点 Mooncake transfer 地址的一致性注册中心。跨节点如果统一，也应考虑 etcd，而不是 file。

**能否改造替代？**

可以，但不是 drop-in 替代，需要改 Mooncake connector 和 worker registry 协议。

可行的 etcd 统一方案是：

```text
prefill Mooncake worker
  -> 注册 engine_id、dp/tp/pp rank、worker_addr 到 etcd
  -> 每个 transfer worker 使用独立 lease
  -> decode connector 直接查/watch etcd
  -> 不再访问 per-prefill bootstrap /query
```

但这要求：

1. prefill 的每个 Mooncake worker 把 `worker_addr` 写入 etcd；
2. decode connector 支持从 etcd 读取映射；
3. etcd entry 与 worker 生命周期绑定，避免 stale `worker_addr`；
4. frontend 不再传 `remote_bootstrap_addr`，或传新的 registry locator；
5. 处理启动竞态：所有 TP/PP worker 注册完成前，prefill 不能被判定 ready；
6. 兼容现有 Mooncake HTTP bootstrap 模式。

**结论**

当前架构中：

```text
file/etcd discovery 不能替代 bootstrap；
它们负责告诉 frontend “prefill 服务在哪里”；
bootstrap 负责告诉 decode “Mooncake KV transfer agent 在哪里”。
```

如果未来要减少每 prefill 一个 bootstrap HTTP 服务，比较合理的方向是 **用 etcd 实现一个可选的 Mooncake transfer registry backend**，让 Mooncake connector 直接从 etcd 获取 `engine_id -> worker_addr` 映射。file 不适合承担这个职责。当前版本应继续保留 bootstrap。
