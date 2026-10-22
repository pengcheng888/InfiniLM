# `stream_chat` 的 routing key 分析

本文分析聚合式推理 frontend 中 `AggregatedRouter.stream_chat()` 的
`routing_key` 来源、生成优先级、使用方式，以及它与会话亲和路由的关系。

## 1. 结论

`stream_chat()` 中的 `routing_key` 不是客户端直接传入的明文 session id，而是在
router 内部根据请求 payload 生成的 SHA-256 哈希指纹。

核心代码位于 `python/infinilm/cluster/router.py`：

```python
routing_key = (
    affinity_key(payload)
    if self.router_mode == "session-affinity"
    else None
)
```

因此：

- 只有 `router_mode == "session-affinity"` 时才生成 `routing_key`。
- `round-robin`、`least-loaded`、`kv-aware` 模式下 `routing_key` 为 `None`。
- `AffinityStore` 中保存的是哈希指纹，不保存原始 session id、user 或对话内容。

## 2. 完整调用链

### 2.1 Frontend 读取 session header

客户端请求进入 `AggregatedFrontend.chat_completions()` 后，frontend 会依次读取：

```text
X-InfiniLM-Session-ID
X-Session-ID
```

对应逻辑：

```python
session_id = request.headers.get(
    "x-infinilm-session-id"
) or request.headers.get("x-session-id")
```

如果存在 session id，frontend 会把它写入 payload 的内部字段：

```python
ROUTING_KEY_FIELD = "__infinilm_routing_key"

payload = {
    **payload,
    ROUTING_KEY_FIELD: session_id,
}
```

这个字段只用于 frontend 内部路由，不属于 OpenAI API 的业务字段。

### 2.2 流式请求进入 router

流式请求链路为：

```text
chat_completions()
  -> _stream_response()
  -> AggregatedRouter.stream_chat()
  -> affinity_key(payload)
  -> routing_key
  -> AggregatedRouter.select()
  -> RoutingContext.routing_key
  -> SessionAffinityPolicy.select()
```

`stream_chat()` 在选择 worker 前先计算 `routing_key`：

```python
routing_key = (
    affinity_key(payload)
    if self.router_mode == "session-affinity"
    else None
)
```

随后每次尝试 worker 时都会传入同一个 key：

```python
worker = self.select(
    candidates,
    routing_key=routing_key,
    kv_prefix=kv_prefix,
    count_affinity=attempt_index == 0,
    attempt_index=attempt_index,
)
```

## 3. `affinity_key()` 的生成优先级

`affinity_key()` 定义在 `python/infinilm/cluster/affinity.py`。

`stream_chat()` 调用它时没有单独传入 HTTP headers：

```python
affinity_key(payload)
```

因此在这条链路中，key 的实际来源优先级如下。

### 优先级 1：内部 session 字段

frontend 如果读取到 session header，会先写入：

```python
payload["__infinilm_routing_key"] = session_id
```

`affinity_key()` 会优先使用该字段：

```python
header_value = payload.get(ROUTING_KEY_FIELD)
```

生成方式：

```python
sha256("identity:" + session_id)
```

也就是说，下面两个请求只要 session id 相同，就会得到同一个 routing key：

```http
X-InfiniLM-Session-ID: session-1
X-Session-ID: session-1
```

### 优先级 2：OpenAI `user` 字段

如果没有 session id，则使用请求体中的 `user` 字段：

```python
payload["user"]
```

生成方式：

```python
sha256("identity:" + str(payload["user"]))
```

这适合客户端没有显式 session header，但 OpenAI 请求中带有稳定用户标识的场景。

### 优先级 3：稳定对话前缀

如果既没有 session id，也没有 `user`，则会从对话内容中构造稳定前缀。

参与前缀计算的信息包括：

```text
model
第一条 system message
第一条 user message
```

示意：

```python
prefix = "\n---turn---\n".join(
    (
        model,
        first_system_message,
        first_user_message,
    )
)
```

生成方式：

```python
sha256("conversation:" + prefix)
```

后续对话轮次可能会追加：

```text
assistant 回复
新的 user message
```

但前缀仍使用最早的 system 和 user message，因此同一会话的后续请求可以得到同一个
routing key，进而有机会复用同一个 worker 的本地 prefix cache。

### 优先级 4：无法生成

如果以上信息都不可用，`affinity_key()` 返回 `None`。

此时 `SessionAffinityPolicy` 无法做会话亲和，会走无 key 的 least-loaded 回退路径。

## 4. routing key 的使用方式

`routing_key` 最终传入 `RoutingContext`：

```python
context = RoutingContext(
    candidates=workers,
    discovery_workers=self.watcher.workers,
    routing_key=routing_key,
    kv_prefix=kv_prefix,
    attempt_index=attempt_index,
    count_metrics=count_affinity,
)
```

`SessionAffinityPolicy` 会使用它查找现有绑定：

```python
lookup = self._store.get(
    context.routing_key,
    {worker.instance_id for worker in context.candidates},
)
```

### 4.1 可以复用的情况

如果满足以下条件，会复用原 worker：

```text
该 routing key 已有绑定
绑定的 worker 仍在当前候选集中
worker 处于 ready 状态
worker 负载没有明显超过最低负载 worker
```

负载判断使用：

```text
preferred_load <= minimum_load + load_slack
```

### 4.2 需要回退的情况

以下情况会回退到 least-loaded：

```text
没有 routing key
routing key 没有历史绑定
绑定 worker 已退出或处于 draining
绑定 worker 不在当前候选集中
绑定 worker 相对过忙
```

回退后会把该 routing key 重新绑定到新选择的 worker。

### 4.3 重试时的行为

`stream_chat()` 会在请求开始时计算一次 `routing_key`。

如果首个 worker 在尚未输出 token 前失败，router 会切换下一个候选 worker，
但仍然使用同一个 `routing_key`。

由于候选集中会排除已经尝试失败的 worker，session-affinity policy 会在剩余候选中
重新选择，并更新该 key 的绑定。

## 5. 与 `kv_prefix` 的区别

`stream_chat()` 中还会处理另一个内部字段：

```python
kv_prefix = payload.get("__infinilm_kv_prefix__")
```

二者职责不同：

| 字段 | 使用模式 | 作用 |
|---|---|---|
| `routing_key` | `session-affinity` | 表示稳定会话身份，用于复用绑定的 worker |
| `kv_prefix` | `kv-aware` | 表示请求的连续前缀哈希，用于查找 worker 本地 prefix cache |

两者都是 frontend 内部路由提示，不是 worker 必须理解的 OpenAI 字段。

## 6. 安全与隔离

### 6.1 保存的是哈希

`AffinityStore` 不保存原始 session id、user 或完整对话内容，而是保存：

```text
SHA-256 指纹
worker ID
最近更新时间
```

这可以降低敏感信息在路由状态中的暴露范围。

### 6.2 内部 routing key 不转发给 worker

HTTP worker client 构造请求时会过滤内部字段：

```python
_STANDARD_FIELDS = {
    "model",
    "messages",
    "stream",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "__infinilm_routing_key",
    "__infinilm_kv_prefix__",
}
```

`__infinilm_routing_key` 不会进入 `extra_body`，因此不会随请求发送给 worker。

worker 只负责推理，不需要知道 frontend 内部的会话亲和状态。

## 7. 相关测试

相关行为主要由 `test/service/test_cluster.py` 覆盖：

```text
AffinityTest
RouterModeTest.test_session_affinity_reuses_and_falls_back
HttpRequestPlaneTest.test_session_affinity_uses_frontend_header_and_hides_internal_key
```

其中 frontend 集成测试验证了两点：

```text
frontend 能从 X-Session-ID 提取会话标识
内部 __infinilm_routing_key 不会转发给 worker
```

## 8. 流程总结

```text
client
  -> 带 X-InfiniLM-Session-ID 或 X-Session-ID 请求 frontend
  -> frontend 把 session id 写入 payload["__infinilm_routing_key"]
  -> stream_chat() 调 affinity_key(payload)
  -> 依次尝试 session id / user / 稳定对话前缀
  -> 生成 SHA-256 routing_key
  -> SessionAffinityPolicy 查询 AffinityStore
  -> 命中且负载允许时复用原 worker
  -> miss 或原 worker 不可用时回退 least-loaded 并重新绑定
```
