# Direct benchmark 性能报告规范

本目录定义 `prefill-direct` 和 `decode-direct` 在不同平台、不同参数组合下
的性能报告规范。

## 当前示例

- 当前平台示例：[NVIDIA A100 性能报告索引](nvidia-a100/README.md)

以上示例是已填充完成的参考报告。新测试请使用以下模板：

- [Prefill-direct 模板](templates/prefill-direct-template.md)
- [Decode-direct 模板](templates/decode-direct-template.md)

历史 input 1024 报告已按该命名规则重命名，并移动到平台目录中。

## 文件命名

新生成的报告存放在对应平台目录中，例如 `nvidia-a100/`。趋势图存放在
对应平台目录下的 `assets/` 中。

文件名使用以下格式：

```text
<mode>-<platform>-<model>-<input_len>-<batch_range>-<execution_mode>-<date>.md
```

示例：

```text
decode-direct-nvidia-a100-qwen3-8b-input1024-batch1-256-graph-20261111.md
prefill-direct-metax-c500-qwen3-8b-input1024-batch1-8-eager-20261111.md
```

文件名中的模型名和平台名应保持简短。完整模型路径、driver/runtime 版本和
构建信息应写入报告开头的运行环境表。

## 必需章节

完整报告应包含以下章节：

1. 运行环境
2. 构建与测试命令
3. 执行模式与 graph 覆盖
4. 资源开销
5. 指标定义与计时口径
6. 核心结果
7. 趋势图
8. 关键批次或区间
9. 完整结果表
10. 分析、结论与建议
11. 数据完整性
12. 原始产物策略

## 指标规则

### Prefill-direct

```text
token_count = batch_size * input_len
throughput = batch_size * input_len / avg_forward_duration_s
```

Duration 表示一次物理 prefill forward。

### Decode-direct

当前 benchmark 在一次 repeat 中执行 4 个 decode step：

```text
avg_step_latency = four_step_repeat_duration / 4
single_step_throughput = batch_size / avg_step_latency
scaling = current_batch_throughput / batch1_throughput
```

每个 request 在一个 decode step 中生成 1 个 token，因此 single-step request
throughput 和 token throughput 的数值相同。

## 趋势图

每份标准报告都应包含：

1. Batch size 与 latency 的关系
2. Batch size 与 throughput 的关系
3. Batch size 与相对 batch 1 scaling 的关系

趋势图必须由 JSONL summary 记录生成，并存放在
`docs/performance-reports/<platform>/assets/` 中。
图片文件名应使用稳定的 run ID。

## 原始产物

原始 JSONL 和 log 文件是本机运行产物，通常会被 Git 忽略。报告可以为了
可追溯性保留它们的本地相对路径，但必须说明这些文件不随仓库提交。

如果结果需要长期审计，应将原始产物保存到外部 artifact 系统并链接到对应
位置，而不是直接提交大型 JSONL 或 log 文件。
