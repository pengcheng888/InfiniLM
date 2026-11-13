# Mooncake `get_finished` Mode

Decode 侧 Mooncake connector 保留两种 `get_finished()` 实现，便于做重复 A/B 测试。

## 配置

```bash
INFINILM_MOONCAKE_GET_FINISHED_MODE=legacy|lock_snapshot
```

- `lock_snapshot`：默认值。Receiver loop 用锁发布状态，StepThread 直接交换本地
  snapshot，不向 receiver event loop 提交 future。
- `legacy`：旧实现。StepThread 向 receiver event loop 提交两个
  `run_coroutine_threadsafe()` future，并阻塞等待结果。

未设置该环境变量或设置为空字符串时，使用 `lock_snapshot`。非法值会在服务启动时
报错。

## 启动示例

新版路径：

```bash
INFINILM_TIMING=true \
INFINILM_MOONCAKE_GET_FINISHED_MODE=lock_snapshot \
bash run_decode.sh
```

旧版路径：

```bash
INFINILM_TIMING=true \
INFINILM_MOONCAKE_GET_FINISHED_MODE=legacy \
bash run_decode.sh
```

该开关只影响 D 侧 receiver 的 `get_finished()` 路径。P 侧 `get_finished()` 仍走
sender loop，不参与该对比。为了保持变量单一，A/B 测试中除该环境变量外，P 侧和
D 侧其他启动参数应保持一致。

## 判读

Timing schema v5 中：

```text
kv_connector_get_finished_ms
```

表示 `get_finished()` 调用耗时。旧实现中该值通常在 `0.34 ms` 量级；锁快照实现中
通常在 `0.01 ms` 量级。

A/B 测试不能只比较单次运行。P 侧 prefill batching、Mooncake KV readiness 和 D 侧
partial batch 分布都有运行间波动，建议至少交替重复多轮，并同时记录：

```text
kv_connector_get_finished_ms
kv_connector_finalize_ms
decode batch size 分布
decode forward gap
Mean ITL
output throughput
```
