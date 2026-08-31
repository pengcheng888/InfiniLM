# GLM-4.7-Flash FlashMLA Graph Sched Metadata Stability Analysis

Date: 2026-08-27

## Background

本轮排查目标是定位 GLM-4.7-Flash 服务在 Hygon 平台上开启 graph 后的稳定性问题。重点变量包括：

- `ENABLE_GRAPH`
- `reuse_sched_metadata`
- `infinicore::op::mha_varlen_` 是否调用

测试服务参数基本保持一致：

- Model: `/data/shared/GLM-4.7-Flash`
- TP: 4
- `MAX_NEW_TOKENS=2048`
- `MAX_BATCH_SIZE=8`
- `BLOCK_SIZE=64`
- `NUM_BLOCKS=512`
- 每轮 `scripts/test_perf.py` 发起 32 个请求
- 主要测试规模为 5 轮，共 160 个请求

源码关注点：

- `csrc/layers/mla_attention/backends/flashmla.cpp`
  - `flash_mla_with_kvcache` 中新增 `reuse_sched_metadata`
  - 控制是否将 `sched_meta.tile_scheduler_metadata` 和 `sched_meta.num_splits` 传入 `infinicore::op::flash_mla::dense_decode_fwd`
- `csrc/models/glm4_moe_lite/glm4_moe_lite_attention.cpp`
  - `infinicore::op::mha_varlen_` 曾被怀疑可能相关
  - 后续通过注释该调用进行排除

当前用于稳定性确认的源码状态：

```cpp
// csrc/layers/mla_attention/backends/flashmla.cpp
bool reuse_sched_metadata = {false};
if (!reuse_sched_metadata) {
    decode_tile_scheduler_metadata = std::nullopt;
    decode_num_splits = std::nullopt;
}
```

```cpp
// csrc/models/glm4_moe_lite/glm4_moe_lite_attention.cpp
// infinicore::op::mha_varlen_(
//     attn_output,
//     q,
//     key,
//     value,
//     ...
// );
```

## Test Matrix

| Case | Graph | `reuse_sched_metadata` | `mha_varlen_` | Result | Log |
|---|---:|---:|---:|---|---|
| A | off | true | commented | Pass, 160/160 | `glm47_nograph_reuse_sched_true_no_mha_varlen_5runs_2048_8bs_tp4_bs64_20260826_231126` |
| B | on | true | commented | Pass once, 160/160 | `glm47_graph_reuse_sched_true_no_mha_varlen_confirm_5runs_2048_8bs_tp4_bs64_20260827_110132` |
| C | on | true | commented | Fail reproduced | `glm47_graph_reuse_sched_true_no_mha_varlen_again_5runs_2048_8bs_tp4_bs64_20260827_113321` |
| D | on | false | commented | Pass, 160/160 | `glm47_graph_reuse_sched_false_no_mha_varlen_reconfirm_5runs_2048_8bs_tp4_bs64_20260827_122536` |
| E | on | false | commented | Pass, 160/160 | `glm47_graph_reuse_sched_false_no_mha_varlen_safety_5runs_2048_8bs_tp4_bs64_20260827_131519` |

此前还观察到：

- `graph + reuse_sched_metadata=true + mha_varlen_ commented`
  - `glm47_graph_reuse_sched_true_no_mha_varlen_5runs_2048_8bs_tp4_bs64_20260826_225319`
  - run1 约 17/32 成功后服务崩溃
  - 生成 core: `core.3117293`
- `graph + reuse_sched_metadata=false + mha_varlen_ commented`
  - `glm47_graph_reuse_sched_false_no_mha_varlen_5runs_2048_8bs_tp4_bs64_20260827_103026`
  - 5 轮 160/160 成功

## Detailed Results

### Case A: non-graph + reuse=true + mha_varlen_ commented

Log directory:

`/workspace_codex/InfiniLM/glm47_nograph_reuse_sched_true_no_mha_varlen_5runs_2048_8bs_tp4_bs64_20260826_231126`

Result:

| Run | Success | Output Tokens | Avg Time / Token |
|---|---:|---:|---:|
| run1 | 32/32 | 63496 | 67.21 ms |
| run2 | 32/32 | 64042 | 65.05 ms |
| run3 | 32/32 | 65504 | 64.75 ms |
| run4 | 32/32 | 65504 | 64.69 ms |
| run5 | 32/32 | 64881 | 65.19 ms |

Conclusion:

非 graph 模式下，即使 `reuse_sched_metadata=true`，服务没有崩溃。这说明 `reuse_sched_metadata=true` 本身不是充分触发条件。

### Case B: graph + reuse=true + mha_varlen_ commented, pass once

Log directory:

`/workspace_codex/InfiniLM/glm47_graph_reuse_sched_true_no_mha_varlen_confirm_5runs_2048_8bs_tp4_bs64_20260827_110132`

Result:

| Run | Success | Output Tokens | Avg Time / Token |
|---|---:|---:|---:|
| run1 | 32/32 | 65474 | 32.11 ms |
| run2 | 32/32 | 65433 | 32.18 ms |
| run3 | 32/32 | 65468 | 32.19 ms |
| run4 | 32/32 | 64170 | 32.04 ms |
| run5 | 32/32 | 65437 | 34.75 ms |

Conclusion:

该组合可以完整通过，说明问题不是 deterministic crash，而是概率性或依赖运行时状态、地址布局、调度时序。

### Case C: graph + reuse=true + mha_varlen_ commented, failure reproduced

Log directory:

`/workspace_codex/InfiniLM/glm47_graph_reuse_sched_true_no_mha_varlen_again_5runs_2048_8bs_tp4_bs64_20260827_113321`

Process result:

```text
server_pid=3767193
ready=1
ready_wait_seconds=76
run1_rc=0
run2_rc=1
server_rc=139
```

Request result:

| Run | Success | Output Tokens | Avg Time / Token | Notes |
|---|---:|---:|---:|---|
| run1 | 17/32 | 33850 | 31.96 ms | 后续请求出现 `peer closed connection` 和 `Connection error` |
| run2 | 0/32 | 0 | N/A | 全部 `Connection error`; stats formatting 触发 `TypeError` |

Failure evidence from `server.err`:

```text
Invalid address access: 0x7f7826fb6000, Error code: 1.
>>>>>>>> KERNEL VMFault !!!! <<<<<<
>>>>>>>> PID: 3767193, SIGNAL: 0 !!!! <<<<<<
```

Matched kernel:

```text
flash_fwd_mla_combine_kernel...CombineParams
```

Core:

```text
/workspace_codex/InfiniLM/core.3767193
```

Conclusion:

在 `mha_varlen_` 已注释的情况下仍然复现 VMFault 和 segfault，故 `mha_varlen_` 不是该 crash 的必要条件。故障落点在 FlashMLA dense decode 的 combine kernel。

### Case D: graph + reuse=false + mha_varlen_ commented, reconfirm

Log directory:

`/workspace_codex/InfiniLM/glm47_graph_reuse_sched_false_no_mha_varlen_reconfirm_5runs_2048_8bs_tp4_bs64_20260827_122536`

Result:

| Run | Success | Output Tokens | Avg Time / Token |
|---|---:|---:|---:|
| run1 | 32/32 | 65504 | 35.35 ms |
| run2 | 32/32 | 64605 | 35.21 ms |
| run3 | 32/32 | 65504 | 35.36 ms |
| run4 | 32/32 | 65504 | 35.29 ms |
| run5 | 32/32 | 63965 | 35.03 ms |

Error scan:

- no `KERNEL VMFault`
- no `Segmentation fault`
- no `Connection error`
- no `peer closed`
- no new core

### Case E: graph + reuse=false + mha_varlen_ commented, safety confirmation

Log directory:

`/workspace_codex/InfiniLM/glm47_graph_reuse_sched_false_no_mha_varlen_safety_5runs_2048_8bs_tp4_bs64_20260827_131519`

Result:

| Run | Success | Output Tokens | Avg Time / Token |
|---|---:|---:|---:|
| run1 | 32/32 | 65504 | 35.26 ms |
| run2 | 32/32 | 65504 | 35.39 ms |
| run3 | 32/32 | 65504 | 35.64 ms |
| run4 | 32/32 | 65504 | 35.45 ms |
| run5 | 32/32 | 65504 | 35.54 ms |

Error scan:

- no `KERNEL VMFault`
- no `Segmentation fault`
- no `Connection error`
- no `peer closed`
- no new core

Conclusion:

`graph + reuse_sched_metadata=false + mha_varlen_ commented` 连续多轮 160/160 通过，稳定性明显优于 `reuse_sched_metadata=true`。

## Analysis

### 1. `mha_varlen_` is not the root cause of this crash

`infinicore::op::mha_varlen_` 被注释后，`graph + reuse_sched_metadata=true` 仍然复现：

- VMFault
- `server_rc=139`
- core dump
- client side `peer closed connection` / `Connection error`

因此该 crash 不依赖 `mha_varlen_` 的执行。`mha_varlen_` 相关问题可以单独排查，但它不是这次 FlashMLA graph crash 的必要条件。

### 2. Graph itself is not sufficient to trigger the crash

`graph + reuse_sched_metadata=false + mha_varlen_ commented` 至少三轮 5x32 级别测试通过：

- `glm47_graph_reuse_sched_false_no_mha_varlen_5runs_2048_8bs_tp4_bs64_20260827_103026`
- `glm47_graph_reuse_sched_false_no_mha_varlen_reconfirm_5runs_2048_8bs_tp4_bs64_20260827_122536`
- `glm47_graph_reuse_sched_false_no_mha_varlen_safety_5runs_2048_8bs_tp4_bs64_20260827_131519`

这说明开启 graph 后，服务并非必然崩溃。

### 3. `reuse_sched_metadata=true` is not sufficient outside graph

`non-graph + reuse_sched_metadata=true + mha_varlen_ commented` 160/160 通过，说明非 graph 路径可以承受 metadata 复用。

### 4. Failure concentrates on graph + sched metadata reuse

最关键对比：

| Comparison | Result |
|---|---|
| graph + reuse=true + mha commented | 概率性 VMFault |
| graph + reuse=false + mha commented | 多轮稳定通过 |
| non-graph + reuse=true + mha commented | 稳定通过 |

这把问题收敛到 graph 场景下复用 FlashMLA sched metadata 的路径。

`reuse_sched_metadata=true` 时，代码会把历史保存的：

```cpp
sched_meta.tile_scheduler_metadata
sched_meta.num_splits
```

重新传入：

```cpp
infinicore::op::flash_mla::dense_decode_fwd(...)
```

而 `reuse_sched_metadata=false` 时，这两个 optional 被置空：

```cpp
decode_tile_scheduler_metadata = std::nullopt;
decode_num_splits = std::nullopt;
```

该路径强制 `dense_decode_fwd` 不消费旧 metadata。稳定测试表明，这能避开 VMFault。

### 5. Failure mode points to invalid device memory access in FlashMLA combine kernel

失败日志中 VMFault 匹配到：

```text
flash_fwd_mla_combine_kernel...CombineParams
```

这说明最终非法访问发生在 FlashMLA dense decode 的 combine 阶段。结合 graph+reuse 的触发条件，可能原因包括：

- graph capture/replay 时复用了已经不再匹配当前 batch/cache 状态的 metadata 内容
- metadata tensor 的 device pointer 生命周期不满足 graph replay 要求
- allocator 复用了 metadata buffer，但 graph replay 或 kernel 参数仍持有旧地址
- `sched_meta` 中缓存的 `tile_scheduler_metadata` / `num_splits` 与当前请求的 shape、split 数、cache seqlens 或 block table 不一致
- `dense_decode_fwd` 返回的新 metadata 没有在所有需要更新的场景里安全覆盖旧 metadata

上述判断是从测试结果和 VMFault kernel 名推断出的方向，尚未证明具体是哪一个内部字段越界。

## Root-Cause Hypothesis

当前最强假设：

在 `ENABLE_GRAPH=1` 时，`flash_mla_with_kvcache` 对 FlashMLA dense decode sched metadata 的复用不安全。复用的 `tile_scheduler_metadata` / `num_splits` 在某些请求序列或 graph replay 状态下与当前 kernel 参数不匹配，导致 `flash_fwd_mla_combine_kernel` 访问非法地址。

该问题表现为概率性：

- 同一组合 `graph + reuse=true + mha_varlen_ commented` 曾 160/160 通过
- 再次测试同一组合时复现 VMFault 和 core dump

因此它可能受以下因素影响：

- 请求长度分布
- batch 动态组合
- graph capture bucket / replay bucket
- allocator 地址布局
- metadata 初始化和更新时序

## Recommended Fix

短期稳定方案：

```cpp
bool reuse_sched_metadata = {false};
```

或在能可靠获取 graph 状态时使用更精确策略：

```cpp
bool reuse_sched_metadata = !enable_graph;
```

也就是：

- graph 模式下禁用 FlashMLA sched metadata 复用
- 非 graph 模式下可以继续保留复用优化

从现有测试看，禁用复用的主要代价是性能：

| Mode | Typical Avg Time / Token |
|---|---:|
| graph + reuse=true | about 32 ms |
| graph + reuse=false | about 35 ms |
| non-graph + reuse=true | about 65 ms |

因此 `graph + reuse=false` 仍明显快于 non-graph，同时稳定性显著更好。

## Final Conclusion

基于多轮测试，当前结论如下：

1. `mha_varlen_` 不是此次 VMFault/segfault 的根因。
2. graph 模式本身不是根因，因为 `reuse_sched_metadata=false` 时 graph 稳定通过。
3. `reuse_sched_metadata=true` 在非 graph 下可以正常工作。
4. 问题集中在 `graph + reuse_sched_metadata=true` 的 FlashMLA dense decode sched metadata 复用路径。
5. 最终崩溃落点是 `flash_fwd_mla_combine_kernel...CombineParams`，表现为 invalid address access。
6. 当前推荐保守修复是 graph 模式禁用 sched metadata 复用，保持 `reuse_sched_metadata=false`。

