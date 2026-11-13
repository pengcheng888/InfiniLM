# Direct prefill / decode 性能测试

[`examples/bench.py`](../../examples/bench.py) 提供两个 direct benchmark 模式。它们会绕过：

```text
tokenization
自回归 generate() 循环
request scheduler
detokenization
HTTP
```

可用模式为：

```text
prefill-direct
decode-direct
```

两个 direct 模式都必须使用 paged attention。它们直接调用
`InferEngine.forward()`，并且在每次被测 forward 开始前和完成后同步 device。

## 用法

### 前置条件

- 已安装并能导入 `infinilm`、`infinicore`。
- 模型路径存在，默认使用 `/data-aisoft/mechdancer/models/Qwen3-8B`。
- direct 模式必须使用 paged attention。
- direct benchmark 必须使用 `flash-attn`；未显式指定时会自动使用 `flash-attn`。
- 正式性能测试不要设置 `INFINILM_FORWARD_BYPASS_DELAY_MS` 或 Mooncake bypass 延迟变量。
- `run_bench.sh` 默认使用 NVIDIA 设备；可用 `CUDA_VISIBLE_DEVICES` 指定 GPU。

### 使用 run_bench.sh

默认配置为：

```text
batch size: 1,2,3,4,5,6,7,8
input length: 1024
warmup: 3 次
measured repeats: 30 次
random seed: 2026（固定）
mode: decode-direct
output: benchmark-decode-direct.jsonl
attention backend: flash-attn
graph compiling: enabled
```

直接测试 decode：

```bash
./run_bench.sh
```

直接测试 prefill：

```bash
BENCHMARK_MODE=prefill-direct ./run_bench.sh
```

指定模型和输出文件：

```bash
MODEL=/path/to/model \
BENCHMARK_MODE=prefill-direct \
BENCHMARK_OUTPUT=prefill-results.jsonl \
./run_bench.sh
```

快速验证链路，不作为正式性能数据：

```bash
BATCH_SIZES=1 \
INPUT_LENS=16 \
BENCHMARK_REPEATS=1 \
BENCHMARK_OUTPUT=smoke.jsonl \
./run_bench.sh
```

### run_bench.sh 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL` | `/data-aisoft/mechdancer/models/Qwen3-8B` | 模型目录 |
| `BATCH_SIZES` | `1,2,3,4,5,6,7,8` | physical batch size 列表 |
| `INPUT_LENS` | `1024` | 输入长度；decode 模式中同时表示 past KV length |
| `OUTPUT_LEN` | `1` | direct 模式下不参与 forward 构造 |
| `BENCHMARK_MODE` | `decode-direct` | 可选 `prefill-direct`、`decode-direct` |
| `BENCHMARK_REPEATS` | `30` | 每个形状的正式测量次数 |
| `BENCHMARK_OUTPUT` | `benchmark-${BENCHMARK_MODE}.jsonl` | JSONL 输出文件 |
| `CUDA_VISIBLE_DEVICES` | `3,4` | 可见 GPU |

`run_bench.sh` 当前固定传 `--enable-graph`。如需 eager 对照，请直接调用
`examples/bench.py` 并省略 `--enable-graph`。

### 直接调用 examples/bench.py

decode：

```bash
python examples/bench.py \
  --device nvidia \
  --model /path/to/model \
  --batch-size 1,2,3,4,5,6,7,8 \
  --input-len 1024 \
  --benchmark-mode decode-direct \
  --benchmark-repeats 30 \
  --benchmark-output decode-results.jsonl \
  --enable-paged-attn \
  --attn flash-attn \
  --warmup
```

prefill：

```bash
python examples/bench.py \
  --device nvidia \
  --model /path/to/model \
  --batch-size 1,2,3,4,5,6,7,8 \
  --input-len 1024 \
  --benchmark-mode prefill-direct \
  --benchmark-repeats 30 \
  --benchmark-output prefill-results.jsonl \
  --enable-paged-attn \
  --attn flash-attn \
  --warmup
```

`--output-len` 在 direct 模式下不参与 forward 构造；为了兼容现有参数解析，
`run_bench.sh` 仍传 `1`。

## 语义

对于 physical batch size `b` 和输入长度 `1024`：

```text
prefill token 数 = b * 1024
decode token 数 = b * 4
decode step 数 = 4
decode step 0 past KV length = 1024
decode step 3 past KV length = 1027
decode step 3 total KV length = 1028
```

`decode-direct` 有意不执行语义完整的 prefill。paged KV cache 由 engine
初始化为全 0，attention 路径会观察到指定长度的历史 token，但生成的 token ID
没有语义意义。因此这是性能 benchmark，不是正确性 benchmark。

`decode-direct` 固定执行 4 个 decode step，不暴露 step 数参数。step 0 输入
dummy token；step 1 到 step 3 使用上一步 sampled token。`BENCHMARK_REPEATS=30`
时，每个 batch size 会记录 `30 * 4 = 120` 个 step sample。warmup 固定 3 次
repeat，因此同样执行 `3 * 4 = 12` 个 step。

对于 physical batch size `b`，benchmark 会分配 `b` 个固定且互不重叠的
block table。例如 block size 为 256、decode 总长度为 1028 时，每个 request
占用 5 个 block。

direct 模式的每个样本只测量一次 physical forward。它们不会执行 `3+3+2`
这类多次 forward 的 request 分组，因此结果不包含 scheduler 分组开销和
batch 与 batch 之间的间隔。

## 计时口径

direct 模式的计时边界包含：

```text
Python InferEngine.forward wrapper
C++ InferEngine forward
RankWorker forward
graph replay 或 eager model forward
sampling
```

它不包含：

```text
HTTP
JSON parsing
tokenization
scheduler queueing
build_model_inputs
detokenization
Python generate() 循环
```

输出字段 `timing_scope` 固定为 `engine_forward_including_sampling`。
如果需要 model-only 时间，应在 C++ 中围绕 `graph->run()` 或
`model_->forward()` 增加打点。

## 输出

设置 `--benchmark-output` 后，benchmark 会写出 JSON Lines。每行是一个 JSON
对象，`event` 字段有四种：

| event | 含义 |
| --- | --- |
| `config` | 本次运行配置和计时口径 |
| `sample` | 一次被测 forward |
| `summary` | decode step summary 或一个 batch size 的 4-step overall 统计 |
| `scaling` | 相对 batch size 1 的吞吐倍数 |

`decode-direct` 会为每个 `batch_size + decode_step` 输出 step summary，也会为
每个 batch size 输出 4-step overall summary。两种 summary 都包含 count、
mean/avg、population standard deviation、minimum、p50、p95、p99、maximum 和
throughput。

分位数使用线性插值计算。p99 属于尾部分位数，如果需要比较稳定的 p99，
建议把 `BENCHMARK_REPEATS` 提高到 100 或以上；默认 30 次可以得到 p99，
但统计稳定性较弱。

查看 summary：

```bash
rg '"event": "summary"' decode-results.jsonl
```

查看原始样本：

```bash
rg '"event": "sample"' decode-results.jsonl
```

## Graph 模式

当前 paged graph compiler 会编译 decode batch size `1..min(max_batch_size, 255)`；
当 `max_batch_size` 大于等于 256 时，会额外追加 256。因此在本次
`max_batch_size=256` 的 decode-direct 测试中，batch `1..256` 全部有对应 graph。
packed prefill 形状不匹配它的 decode graph key，因此 `prefill-direct` 会走
eager path。

结果会同时记录 `enable_graph` 和 `expected_execution_mode`。其中
`expected_execution_mode` 是根据当前 compiler 行为推断出来的，不是 C++
runtime 回传的实际执行模式。

为了得到稳定对比，建议先直接调用 `examples/bench.py` 并省略
`--enable-graph` 来跑 eager mode，再单独跑 graph mode，并且只比较 expected
execution mode 相同的结果。

## 报告规范

不同平台和参数组合下的正式报告应按
[performance report convention](README.md) 生成。可参考：

- [prefill-direct input 1024 示例报告](nvidia-a100/prefill-direct-nvidia-a100-qwen3-8b-input1024-batch1-8-eager-20261111.md)
- [decode-direct input 1024 graph 1..256 示例报告](nvidia-a100/decode-direct-nvidia-a100-qwen3-8b-input1024-batch1-256-graph-20261111.md)
