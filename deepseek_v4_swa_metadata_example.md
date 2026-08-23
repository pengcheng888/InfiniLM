# DeepSeek V4 SWA Metadata 示例

本文用一个单请求示例说明 `swa_indices` 和 `swa_topk_lengths` 在 prefill 阶段和 decode 阶段每次推理时的数值。

## 示例假设

- 单个请求。
- `block_size = 256`。
- `block_table = [10]`。
- 不启用 `dsv4_full_to_swa_block_ids` 重映射。
- prompt 长度为 5，对应 position `0..4`。
- 后续 decode 生成 3 个 token，对应 position `5..7`。
- SWA 窗口大小固定为 128。

在这些假设下，逻辑 position 到 cache slot 的映射为：

```text
slot(position) = block_table[position / 256] * 256 + position % 256
               = 10 * 256 + position
               = 2560 + position
```

`swa_indices` 每一行表示当前 token 做 SWA attention 时读取的最近 128 个 cache slot，顺序是从当前 token 往前看：

```text
[slot(position), slot(position - 1), slot(position - 2), ...]
```

如果窗口里某个历史 position 不存在，就填 `-1`。下面表格里的 `-1 x N` 表示连续 N 个 `-1`。

`swa_topk_lengths` 表示这一行里实际有效的 SWA slot 数量：

```text
swa_topk_lengths = min(position + 1, 128)
```

## Prefill 阶段

prefill 一次 forward 会同时处理 prompt 中尚未缓存的多个 token。这里假设 prompt 全部未缓存，所以本次推理处理 position `0..4`。

因此：

```text
swa_indices shape      = [5, 128]
swa_topk_lengths shape = [5]
```

| row | position | swa_indices | swa_topk_lengths |
| --- | --- | --- | --- |
| 0 | 0 | `[2560, -1 x 127]` | `1` |
| 1 | 1 | `[2561, 2560, -1 x 126]` | `2` |
| 2 | 2 | `[2562, 2561, 2560, -1 x 125]` | `3` |
| 3 | 3 | `[2563, 2562, 2561, 2560, -1 x 124]` | `4` |
| 4 | 4 | `[2564, 2563, 2562, 2561, 2560, -1 x 123]` | `5` |

完整张量可理解为：

```text
swa_indices =
[
  [2560, -1 x 127],
  [2561, 2560, -1 x 126],
  [2562, 2561, 2560, -1 x 125],
  [2563, 2562, 2561, 2560, -1 x 124],
  [2564, 2563, 2562, 2561, 2560, -1 x 123],
]

swa_topk_lengths = [1, 2, 3, 4, 5]
```

## Decode 阶段

decode 阶段每次 forward 通常只处理一个新 token。prefill 后 prompt 已覆盖 position `0..4`，第一次 decode 输入新生成 token 的 position 为 `5`。

### Decode 第 1 次推理

```text
position = 5
swa_indices shape      = [1, 128]
swa_topk_lengths shape = [1]

swa_indices =
[
  [2565, 2564, 2563, 2562, 2561, 2560, -1 x 122],
]

swa_topk_lengths = [6]
```

### Decode 第 2 次推理

```text
position = 6
swa_indices shape      = [1, 128]
swa_topk_lengths shape = [1]

swa_indices =
[
  [2566, 2565, 2564, 2563, 2562, 2561, 2560, -1 x 121],
]

swa_topk_lengths = [7]
```

### Decode 第 3 次推理

```text
position = 7
swa_indices shape      = [1, 128]
swa_topk_lengths shape = [1]

swa_indices =
[
  [2567, 2566, 2565, 2564, 2563, 2562, 2561, 2560, -1 x 120],
]

swa_topk_lengths = [8]
```

## 超过 128 个 token 后

当 position 达到 127 时，SWA 窗口刚好填满：

```text
position = 127
swa_indices = [2687, 2686, ..., 2560]
swa_topk_lengths = [128]
```

当 position 继续增长，例如 position 为 130 时，窗口仍然只保留最近 128 个位置：

```text
position = 130
swa_indices = [2690, 2689, ..., 2563]
swa_topk_lengths = [128]
```

也就是说，`swa_topk_lengths` 最多为 128；`swa_indices` 始终按当前 token 向前取最近 128 个可见 cache slot。
