# InfiniLM KV-Aware Routing 理解笔记

本文整理 InfiniLM 聚合式集群中 KV-aware routing 的原理、prefix cache index
的作用，以及一次两次请求后的 `prefix-cache.events.jsonl` 观测结果。

## 1. 核心结论

KV-aware routing 的目标是：在请求发往 worker 之前，找出哪个 worker 的本地
paged KV cache 中已经保存了当前 prompt 的最长连续前缀，并优先把请求路由到
这个 worker，从而减少重复 prefill。

它依赖两层信息：

```text
worker:
  本地 BlockManager 知道哪些 prefix block 已经有 KV cache

frontend:
  对当前请求 tokenize，计算同样的 prefix hash，
  再读取 worker 导出的 cache index，选择命中前缀最长的 worker
```

需要特别注意：

```text
cache index 不是 KV cache 本身。
```

实际 KV cache 仍然在 worker 的 GPU / paged cache 中；输出目录中的 index 只保存
可复用前缀块的哈希。

## 2. 基本概念

### 2.1 prefix block

InfiniLM 按固定 `block_size` 切分 token 序列。当前部署中使用：

```text
block_size = 256 tokens
```

只有完整 block 才进入 prefix index。例如：

```text
prompt token count = 391
```

则完整 prompt block 只有：

```text
block 0: token 0 ~ 255
```

末尾的 135 个 token 是 partial block，不能作为完整 prefix block 发布。

### 2.2 链式前缀哈希

每个 block 的哈希包含父前缀哈希：

```text
hash(block 0)                  = prefix 0 ~ 255
hash(block 1, parent=hash0)    = prefix 0 ~ 511
hash(block 2, parent=hash1)    = prefix 0 ~ 767
```

因此第 N 个哈希代表的是：

```text
从 prompt 开头到第 N 个 block 结束的完整前缀
```

这也是 KV-aware 只匹配连续前缀的原因。paged prefix cache 通常只能从开头连续
复用，不能把中间离散的几个 block 直接拼成一段可复用 KV cache。

### 2.3 namespace

不同模型、tokenizer 或 block size 可能产生不同 token 序列或不同 KV 空间。
为了避免误匹配，cache index 使用 namespace 隔离。

匹配时必须同时满足：

```text
namespace 相同
block_size 相同
连续 prefix hash 相同
```

## 3. Worker 侧：cache index 输出目录

### 3.1 为什么需要输出目录

frontend 是独立进程，不能直接读取 worker 内存中的 `BlockManager` 状态。
因此 worker 需要把本地 prefix cache 状态导出到文件，frontend 再读取这些文件。

输出目录规则：

```text
显式配置 cache-index-dir:
  {cache_index_dir}/{worker_instance_id}

file discovery 默认:
  {registry根目录的父目录}/cache-index/{worker_instance_id}
```

每个 worker 使用独立子目录，避免多个 worker 写同名文件相互覆盖。

### 3.2 输出文件

每个 worker 目录下有两个文件：

```text
prefix-cache.snapshot.json
prefix-cache.events.jsonl
```

#### snapshot

`prefix-cache.snapshot.json` 是权威全量快照，包含：

```text
schema_version
namespace
block_size
sequence
generated_at
hashes
```

写入时先写临时文件，再原子替换，避免 frontend 读到半写 JSON。

#### event log

`prefix-cache.events.jsonl` 是增量事件日志。事件类型包括：

```text
published: 新的完整 prefix block 被写入索引
touched:   已有 prefix block 被请求命中或复用
evicted:   prefix block 被淘汰
```

frontend 不需要每次重读完整 snapshot，可以先读 snapshot，再 tail 增量 event。
事件达到上限后，worker 会重写 snapshot 并压缩 event log；frontend 发现日志
变短后会重新加载 snapshot。

### 3.3 为什么不把完整索引放进 registry

registry 记录中只保存 cache index 元数据：

```text
namespace
block_size
snapshot path
events path
sequence
entries
```

不把完整 hash 集合放进 registry，原因是：

1. prefix index 可能很大，放在 heartbeat JSON 中会让 registry 记录过大；
2. cache 变化频繁，直接写 etcd 会造成很大压力；
3. snapshot/event 文件更适合原子替换和增量 tail；
4. registry 负责服务发现，不应承载高频 cache 状态同步。

## 4. Frontend 侧：KV-aware 请求链路

一次 chat 请求的 KV-aware 处理流程如下：

```text
client
  -> AggregatedFrontend.chat_completions()
  -> 解析模型和候选 worker
  -> PrefixHashCalculator tokenize 当前 prompt
  -> 计算 namespace 和连续 prefix hashes
  -> 把内部 __infinilm_kv_prefix__ 传给 AggregatedRouter
  -> KvIndexStore 读取 worker snapshot/event
  -> 查询每个 worker 的最长连续前缀命中
  -> KvAwarePolicy 过滤 min tokens 和负载 slack
  -> 选择命中 token 数最多的 worker
  -> 移除内部字段后请求 worker
  -> worker 本地实际复用 KV cache
```

### 4.1 frontend 计算 prompt hash

frontend 使用与 worker 相同的模型 processor：

```text
apply chat template
tokenize
按 block_size 生成链式哈希
```

多模态请求当前不参与 KV-aware 计算，会退回普通 fallback 路由。

### 4.2 frontend 读取 worker index

`KvIndexStore` 维护：

```text
worker_id -> prefix hash set
```

它会：

1. 读取每个 ready worker 的 snapshot；
2. tail event log；
3. `published/touched` 时加入 hash；
4. `evicted` 时删除 hash；
5. worker 退出、draining 或索引读取失败时删除对应状态。

### 4.3 最长连续前缀匹配

对一个请求的前缀哈希序列：

```text
[hash0, hash1, hash2]
```

frontend 会检查每个 worker：

```text
hash0 是否存在？
hash0、hash1 是否连续存在？
hash0、hash1、hash2 是否连续存在？
```

直到遇到第一个不存在的 hash 为止。命中结果包含：

```text
matched_blocks
matched_tokens = matched_blocks * block_size
```

### 4.4 选择 worker

KV-aware 并不是无条件选择命中最长的 worker。它会同时考虑负载：

```text
minimum_load = 所有候选 worker 的最低负载分
eligible worker 的负载分 <= minimum_load + load_slack
```

在 eligible worker 中，选择：

```text
matched_tokens 最大的 worker
```

如果没有有效命中，回退：

```text
kv-aware miss -> least-loaded fallback
```

因此 KV-aware 的语义是：

```text
在不明显加重某个 worker 的前提下，
优先选择可能复用最多 prefix KV 的 worker。
```

## 5. 两次请求后的 event 示例

以下来自一次实际运行：

```text
run_cluster.sh
run_req.sh
run_req.sh
```

观察文件：

```text
/workspace_codex/InfiniLM/infinilm-agg/cache-index/
3658317-500ff4980cec/prefix-cache.events.jsonl
```

其中：

```text
/workspace_codex/InfiniLM/infinilm-agg
```

是指向：

```text
/tmp/infinilm-agg
```

的软链接。

文件中有四条记录：

```json
{"hash":"e96ef7ce99ddd6ca1af2cc3a08e0b0f1","sequence":1,"timestamp":1792148938.1805139,"type":"published"}
{"hash":"e41e80591d539cf0d6f456ad8a7b70a7","sequence":2,"timestamp":1792148938.5858943,"type":"published"}
{"hash":"e96ef7ce99ddd6ca1af2cc3a08e0b0f1","sequence":3,"timestamp":1792149033.1250277,"type":"touched"}
{"hash":"eb614f20a9b61814adce05dbbb73e074","sequence":4,"timestamp":1792149033.534818,"type":"published"}
```

时间上分为两组：

```text
第一次请求: 2026-10-16 19:08:58 左右，Asia/Shanghai
第二次请求: 2026-10-16 19:10:33 左右，Asia/Shanghai
间隔约 94.94 秒
```

### 5.1 当前 prompt 的 token 结果

对 `run_req.sh` 中的实际请求执行同样的 prefix 计算，得到：

```json
{
  "namespace": "5c0f0c699e1ff6c0a45b2f3b4917870d084d267c9538e4c8e2a41c271b49228c",
  "hashes": [
    "e96ef7ce99ddd6ca1af2cc3a08e0b0f1"
  ],
  "token_count": 391,
  "block_size": 256
}
```

说明该 prompt 只有第一个 256-token block 是完整 prompt block：

```text
hash e96e... = prompt token 0 ~ 255
```

末尾约 135 个 token 是 partial block，不能单独发布为 prompt prefix block。

### 5.2 第一条：published e96e...

```json
{"hash":"e96e...","sequence":1,"type":"published"}
```

含义：

```text
第一次请求时，worker 完成了 prompt 的第一个完整 256-token block，
并把它发布到 prefix cache index。
```

这个 hash 是两次请求共享的 prompt 前缀。

### 5.3 第二条：published e41e...

```json
{"hash":"e41e...","sequence":2,"type":"published"}
```

第一次请求的 prompt 只有 391 个 token，因此 `e41e...` 不是第二个完整 prompt
block。

它来自：

```text
第一次请求的 prompt 尾部 + 生成的 token
```

当 prompt tail 和生成内容一起凑满一个新的 256-token block 后，worker 发布了
这个 continuation block：

```text
e41e...
```

它不是下一次相同 prompt 一定能命中的 prompt-only prefix。

### 5.4 第三条：touched e96e...

```json
{"hash":"e96e...","sequence":3,"type":"touched"}
```

第二次请求使用相同 prompt，因此第一个 256-token block 的 hash 仍然是：

```text
e96e...
```

worker 发现该 hash 对应的 KV cache 已存在，于是复用它，并记录：

```text
touched
```

这与 worker registry 中的统计一致：

```json
{
  "local_cache_hit_requests_total": 1,
  "local_cache_hit_tokens_total": 256,
  "local_cache_miss_requests_total": 1,
  "local_prefill_tokens_total": 526
}
```

解释：

```text
第一次请求:
  prompt tokens = 391
  cache hit = 0
  prefill tokens = 391

第二次请求:
  prompt tokens = 391
  cache hit = 256
  prefill tokens = 391 - 256 = 135

总计:
  391 + 135 = 526
```

### 5.5 第四条：published eb61...

```json
{"hash":"eb61...","sequence":4,"type":"published"}
```

第二次请求的 prompt tail 与新生成内容又形成了一个完整 continuation block，
因此发布了新的 hash：

```text
eb61...
```

它和 `e41e...` 不同，说明两次生成 continuation block 的 token 内容不同。因为
哈希是链式的，只要 block 内 token 序列不同，最终 hash 就不同。

这条不是替换 `e41e...`，也没有发生 eviction。两个 continuation hash 都留在
index 中。

### 5.6 为什么第二次没有 touched e41e...

第二次请求的 prompt 只有 391 个 token，只有：

```text
e96e...
```

这一个完整 prompt block。

`e41e...` 是第一次请求的：

```text
prompt tail + generated tokens
```

形成的 continuation block，不是第二次请求的第二个 prompt block。

因此第二次请求最多只能命中：

```text
256 tokens
```

不会命中 `e41e...`。

## 6. snapshot 为什么还是空的

当时 snapshot 内容为：

```json
{
  "block_size": 256,
  "generated_at": 1792148849.409138,
  "hashes": [],
  "namespace": "5c0f...",
  "schema_version": 1,
  "sequence": 0
}
```

这是 worker 启动时写出的初始空快照。之后只产生了 4 条 event，尚未达到默认的
event 压缩阈值，因此 snapshot 没有重写。

frontend 读取时会在内存中应用 event：

```text
snapshot: []
+ published e96e... -> {e96e...}
+ published e41e... -> {e96e..., e41e...}
+ touched e96e...   -> 集合不变，只表示复用
+ published eb61... -> {e96e..., e41e..., eb61...}
```

所以 registry metadata 中显示：

```json
{
  "entries": 3,
  "sequence": 4
}
```

这与 event log 完全一致。

## 7. 部署边界

### 7.1 file discovery

file discovery 仅适合同机部署，frontend 可以直接读取默认 cache-index
目录；跨节点 worker registry 必须使用 etcd。

### 7.2 etcd discovery

etcd 只保存 registry 记录，不传输 cache index 文件内容。跨节点部署时，
frontend 必须能访问 worker 导出的 snapshot/event 路径，因此需要配置共享目录：

```text
--cache-index-dir /shared/infinilm/cache-index
```

如果 etcd 模式没有配置共享 cache index 目录，worker 不会创建 cache index
exporter，frontend 也无法使用 KV-aware hint。

## 8. 设计边界

KV-aware routing 是 advisory routing，不是硬保证：

1. index 可能略微滞后于 worker 内存状态；
2. snapshot/event 读取失败时对应 worker 会被跳过；
3. 多模态请求当前回退普通路由；
4. 实际是否能复用 KV，最终仍由 worker 本地 engine 判断；
5. 索引创建或导出失败不应影响推理服务启动。

因此当 KV-aware 信息不可用时，系统会回退：

```text
kv-aware miss -> least-loaded
```

## 9. 总结

KV-aware 的完整闭环可以概括为：

```text
第一次请求:
  worker 计算并缓存 prompt prefix block
  worker 导出 published event

第二次相同前缀请求:
  frontend 计算相同 prefix hash
  从 cache index 中找到该 worker
  将请求路由到该 worker
  worker 复用本地 KV cache
  记录 touched event
```

在本文示例中：

```text
e96e...
```

是两次请求共享的第一个 256-token prompt 前缀：

```text
第一次 published
第二次 touched
```

而：

```text
e41e...
eb61...
```

分别是两次请求的 prompt tail 加生成内容形成的 continuation block，不代表
错误，也不代表淘汰。
