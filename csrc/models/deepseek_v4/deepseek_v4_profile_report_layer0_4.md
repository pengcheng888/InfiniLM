# DeepSeek V4 InfiniLM 性能分析报告

生成时间：2026-07-20 21:38:14

## 范围

- 模型：`/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8-layer0-4`
- 平台：Hygon BW1000，使用 InfiniLM/InfiniCore
- Profile 开关：`INFINILM_DSV4_PROFILE=1`
- 阶段划分：`prefill` 表示 `token_count > 1`；`decode` 表示 `token_count == 1`。
- 当前 DeepSeek V4 InfiniLM 路径中 attention forward 仍然是跳过状态；报告中 attention 侧的统计项是 MHC/norm/post 等结构耗时，不是完整 attention 计算耗时。
- 计时方式是在每个作用域内对 GPU 做同步后的 wall time。`total_ms` 是跨调用、跨 rank 累计时间；做单次调用对比时应主要参考 `avg_ms`。

## 运行命令

```bash
INFINILM_DSV4_PROFILE=1 python examples/test_infer.py --device hygon --model=/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8-layer0-4 --temperature 1.0 --top-p 0.8 --top-k 1 --max-new-tokens 16 --enable-paged-attn --attn paged-attn --tp 1
INFINILM_DSV4_PROFILE=1 python examples/test_infer.py --device hygon --model=/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8-layer0-4 --temperature 1.0 --top-p 0.8 --top-k 1 --max-new-tokens 16 --enable-paged-attn --attn paged-attn --tp 8
```

## 运行摘要

| 运行配置 | 权重加载耗时 ms | 生成总耗时 ms | prompt token ids | generated token ids | 原始日志 |
|---|---:|---:|---|---|---|
| TP=1 | 11115.649 | 3135.27 | `[0, 128803, 4117, 477, 440, 128804, 128822]` | `[57329, 1486, 41381, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780]` | `csrc/models/deepseek_v4/profile_logs/deepseek_v4_layer0_4_tp1_profile.log` |
| TP=8 | 7511.994 | 7736.95 | `[0, 128803, 4117, 477, 440, 128804, 128822]` | `[57329, 1486, 41381, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780]` | `csrc/models/deepseek_v4/profile_logs/deepseek_v4_layer0_4_tp8_profile.log` |

## TP=1 Prefill 阶段

| 模块 | 调用次数 | 总耗时 ms | 平均耗时 ms |
|---|---:|---:|---:|
| `decoder.layer` | 5 | 1085.361 | 217.072200 |
| `decoder.attn_hc_pre` | 5 | 696.288 | 139.257600 |
| `decoder.attn_norm` | 5 | 1.808 | 0.361600 |
| `decoder.attn_hc_post` | 5 | 0.756 | 0.151200 |
| `decoder.ffn_hc_pre` | 5 | 7.845 | 1.569000 |
| `decoder.ffn_norm` | 5 | 0.248 | 0.049600 |
| `decoder.moe` | 5 | 377.040 | 75.408000 |
| `decoder.ffn_hc_post` | 5 | 0.648 | 0.129600 |
| `moe.forward` | 5 | 376.998 | 75.399600 |
| `moe.topk` | 5 | 63.356 | 12.671200 |
| `moe.experts` | 5 | 269.193 | 53.838600 |
| `moe.shared_experts` | 5 | 42.698 | 8.539600 |
| `moe.add_shared` | 5 | 1.566 | 0.313200 |

## TP=1 Decode 阶段

| 模块 | 调用次数 | 总耗时 ms | 平均耗时 ms |
|---|---:|---:|---:|
| `decoder.layer` | 75 | 1892.466 | 25.232880 |
| `decoder.attn_hc_pre` | 75 | 113.646 | 1.515280 |
| `decoder.attn_norm` | 75 | 3.312 | 0.044160 |
| `decoder.attn_hc_post` | 75 | 7.682 | 0.102427 |
| `decoder.ffn_hc_pre` | 75 | 110.147 | 1.468627 |
| `decoder.ffn_norm` | 75 | 3.167 | 0.042227 |
| `decoder.moe` | 75 | 1641.297 | 21.883960 |
| `decoder.ffn_hc_post` | 75 | 8.280 | 0.110400 |
| `moe.forward` | 75 | 1640.814 | 21.877520 |
| `moe.topk` | 75 | 19.198 | 0.255973 |
| `moe.experts` | 75 | 1584.649 | 21.128653 |
| `moe.shared_experts` | 75 | 30.750 | 0.410000 |
| `moe.add_shared` | 75 | 4.028 | 0.053707 |

## TP=8 Prefill 阶段

| 模块 | 调用次数 | 总耗时 ms | 平均耗时 ms |
|---|---:|---:|---:|
| `decoder.layer` | 40 | 28728.082 | 718.202050 |
| `decoder.attn_hc_pre` | 40 | 16058.364 | 401.459100 |
| `decoder.attn_norm` | 40 | 21.255 | 0.531375 |
| `decoder.attn_hc_post` | 40 | 15.016 | 0.375400 |
| `decoder.ffn_hc_pre` | 40 | 430.733 | 10.768325 |
| `decoder.ffn_norm` | 40 | 28.920 | 0.723000 |
| `decoder.moe` | 40 | 12156.113 | 303.902825 |
| `decoder.ffn_hc_post` | 40 | 9.145 | 0.228625 |
| `moe.forward` | 40 | 12155.593 | 303.889825 |
| `moe.topk` | 40 | 2846.797 | 71.169925 |
| `moe.experts` | 40 | 7410.537 | 185.263425 |
| `moe.shared_experts` | 40 | 944.759 | 23.618975 |
| `moe.add_shared` | 40 | 81.831 | 2.045775 |
| `moe.allreduce` | 40 | 869.306 | 21.732650 |

## TP=8 Decode 阶段

| 模块 | 调用次数 | 总耗时 ms | 平均耗时 ms |
|---|---:|---:|---:|
| `decoder.layer` | 600 | 24807.897 | 41.346495 |
| `decoder.attn_hc_pre` | 600 | 1339.879 | 2.233132 |
| `decoder.attn_norm` | 600 | 30.346 | 0.050577 |
| `decoder.attn_hc_post` | 600 | 77.984 | 0.129973 |
| `decoder.ffn_hc_pre` | 600 | 1310.425 | 2.184042 |
| `decoder.ffn_norm` | 600 | 31.251 | 0.052085 |
| `decoder.moe` | 600 | 21866.411 | 36.444018 |
| `decoder.ffn_hc_post` | 600 | 96.854 | 0.161423 |
| `moe.forward` | 600 | 21858.444 | 36.430740 |
| `moe.topk` | 600 | 207.256 | 0.345427 |
| `moe.experts` | 600 | 19968.136 | 33.280227 |
| `moe.shared_experts` | 600 | 316.237 | 0.527062 |
| `moe.add_shared` | 600 | 42.717 | 0.071195 |
| `moe.allreduce` | 600 | 1295.414 | 2.159023 |

## 观察结论

- TP=1 prefill 阶段主要耗时集中在 `decoder.attn_hc_pre` 和 `decoder.moe`；MoE 内部最大耗时项是 `moe.experts`。
- TP=1 decode 阶段主要耗时集中在 `decoder.moe`，其中 `moe.experts` 占比最高。
- TP=8 的 prefill 和 decode 阶段都统计到了 `moe.allreduce`，说明 TP 通信路径已经生效并被 profile 捕获。
- TP=8 decode 阶段仍然主要耗时在 `moe.experts`；`moe.allreduce` 有可见开销，但小于专家计算耗时。
- 由于 profile 会在每个计时作用域前后同步当前 InfiniCore stream，这些数字用于诊断分析，不应直接视为生产环境延迟。

## 2026-07-21 最新补充：MHC post 系数修正后

### 8 token 精度对齐记录

- 输入 input_ids：[104937]
- 模型：/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8-layer0-4
- 对齐方式：InfiniLM attention forward 当前仍跳过；sglang 标准答案使用 SGLANG_DSV4_USE_ATTN=0 屏蔽 attention。
- 采样参数：temperature=1.0，top_p=0.8，top_k=1，max_new_tokens=8
- TP：8

| 实现 | generated token ids | 输出文本 |
|---|---|---|
| sglang no-attn | [117160, 60574, 106018, 11977, 65804, 97768, 117465, 101261] |  _“_ nalukop kasagaranKaginharian cia尷 enpresak |
| InfiniLM | [117160, 60574, 106018, 11977, 65804, 97768, 117465, 101261] |  _“_ nalukop kasagaranKaginharian cia尷 enpresak |

- sglang 原始输出：/workspace_codex/sglang_dsv4_8tok_noattn.jsonl，e2e_generate_ms=44437.091
- InfiniLM 原始输出：csrc/models/deepseek_v4/profile_logs/deepseek_v4_layer0_4_tp8_8tok_infinilm_latest.log，权重加载耗时 14301.439 ms，total_time=6268.54 ms

结论：前 8 个生成 token 完全一致。

### 最新 profile 运行命令

```bash
# TP=1
source ~/.bashrc && source /.myenv.sh
cd /workspace_codex/InfiniLM
INFINILM_DSV4_PROFILE=1 INFINILM_DSV4_MOE_MARLIN=1 \
python examples/test_infer.py --device hygon \
  --model=/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8-layer0-4 \
  --temperature 1.0 --top-p 0.8 --top-k 1 --max-new-tokens 16 \
  --enable-paged-attn --attn paged-attn --tp 1

# TP=8
source ~/.bashrc && source /.myenv.sh
cd /workspace_codex/InfiniLM
INFINILM_DSV4_PROFILE=1 INFINILM_DSV4_MOE_MARLIN=1 \
python examples/test_infer.py --device hygon \
  --model=/data/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8-layer0-4 \
  --temperature 1.0 --top-p 0.8 --top-k 1 --max-new-tokens 16 \
  --enable-paged-attn --attn paged-attn --tp 8
```

### 最新运行摘要

| 运行配置 | 权重加载耗时 ms | 生成总耗时 ms | prompt token ids | generated token ids | 原始日志 |
|---|---:|---:|---|---|---|
| TP=1 | 11467.205 | 3325.31 | [0, 128803, 4117, 477, 440, 128804, 128822] | [57329, 1486, 41381, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780] | csrc/models/deepseek_v4/profile_logs/deepseek_v4_layer0_4_tp1_profile_latest.log |
| TP=8 | 13703.918 | 7024.85 | [0, 128803, 4117, 477, 440, 128804, 128822] | [57329, 1486, 41381, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780, 113780] | csrc/models/deepseek_v4/profile_logs/deepseek_v4_layer0_4_tp8_profile_latest.log |

### TP=1 Prefill 阶段（最新）

| 模块 | 调用次数 | 总耗时 ms | 平均耗时 ms |
|---|---:|---:|---:|
| decoder.layer | 5 | 1248.105 | 249.621000 |
| decoder.attn_hc_pre | 5 | 678.074 | 135.614800 |
| decoder.attn_norm | 5 | 1.961 | 0.392200 |
| decoder.attn_hc_post | 5 | 187.023 | 37.404600 |
| decoder.ffn_hc_pre | 5 | 7.722 | 1.544400 |
| decoder.ffn_norm | 5 | 0.275 | 0.055000 |
| decoder.moe | 5 | 370.904 | 74.180800 |
| decoder.ffn_hc_post | 5 | 1.277 | 0.255400 |
| moe.forward | 5 | 370.857 | 74.171400 |
| moe.topk | 5 | 59.460 | 11.892000 |
| moe.experts | 5 | 274.012 | 54.802400 |
| moe.shared_experts | 5 | 35.709 | 7.141800 |
| moe.add_shared | 5 | 1.427 | 0.285400 |

### TP=1 Decode 阶段（最新）

| 模块 | 调用次数 | 总耗时 ms | 平均耗时 ms |
|---|---:|---:|---:|
| decoder.layer | 75 | 1900.435 | 25.339133 |
| decoder.attn_hc_pre | 75 | 109.798 | 1.463973 |
| decoder.attn_norm | 75 | 3.325 | 0.044333 |
| decoder.attn_hc_post | 75 | 13.001 | 0.173347 |
| decoder.ffn_hc_pre | 75 | 107.823 | 1.437640 |
| decoder.ffn_norm | 75 | 3.209 | 0.042787 |
| decoder.moe | 75 | 1643.652 | 21.915360 |
| decoder.ffn_hc_post | 75 | 13.700 | 0.182667 |
| moe.forward | 75 | 1643.066 | 21.907547 |
| moe.topk | 75 | 19.216 | 0.256213 |
| moe.experts | 75 | 1599.421 | 21.325613 |
| moe.shared_experts | 75 | 17.680 | 0.235733 |
| moe.add_shared | 75 | 4.136 | 0.055147 |

### TP=8 Prefill 阶段（最新）

| 模块 | 调用次数 | 总耗时 ms | 平均耗时 ms |
|---|---:|---:|---:|
| decoder.layer | 40 | 38930.663 | 973.266575 |
| decoder.attn_hc_pre | 40 | 18772.937 | 469.323425 |
| decoder.attn_norm | 40 | 1207.100 | 30.177500 |
| decoder.attn_hc_post | 40 | 5533.282 | 138.332050 |
| decoder.ffn_hc_pre | 40 | 3686.949 | 92.173725 |
| decoder.ffn_norm | 40 | 27.678 | 0.691950 |
| decoder.moe | 40 | 9678.397 | 241.959925 |
| decoder.ffn_hc_post | 40 | 13.863 | 0.346575 |
| moe.forward | 40 | 9677.895 | 241.947375 |
| moe.topk | 40 | 2970.503 | 74.262575 |
| moe.experts | 40 | 4289.694 | 107.242350 |
| moe.shared_experts | 40 | 1091.157 | 27.278925 |
| moe.add_shared | 40 | 128.861 | 3.221525 |
| moe.allreduce | 40 | 1195.075 | 29.876875 |

### TP=8 Decode 阶段（最新）

| 模块 | 调用次数 | 总耗时 ms | 平均耗时 ms |
|---|---:|---:|---:|
| decoder.layer | 600 | 8552.961 | 14.254935 |
| decoder.attn_hc_pre | 600 | 1335.045 | 2.225075 |
| decoder.attn_norm | 600 | 30.736 | 0.051227 |
| decoder.attn_hc_post | 600 | 135.022 | 0.225037 |
| decoder.ffn_hc_pre | 600 | 1315.748 | 2.192913 |
| decoder.ffn_norm | 600 | 32.512 | 0.054187 |
| decoder.moe | 600 | 5500.907 | 9.168178 |
| decoder.ffn_hc_post | 600 | 144.343 | 0.240572 |
| moe.forward | 600 | 5494.180 | 9.156967 |
| moe.topk | 600 | 207.597 | 0.345995 |
| moe.experts | 600 | 4559.744 | 7.599573 |
| moe.shared_experts | 600 | 166.156 | 0.276927 |
| moe.add_shared | 600 | 39.617 | 0.066028 |
| moe.allreduce | 600 | 491.746 | 0.819577 |

### 最新观察

- 8 token 生成已经与 sglang no-attn 标准完全一致，说明 MHC post 系数修正后，跨 token decode 精度路径已恢复。
- TP=8 decode 阶段 moe.forward 平均耗时为 9.156967 ms，相比旧报告中的 36.430740 ms 明显下降；主要来自 moe.experts 从旧报告 33.280227 ms 降到最新 7.599573 ms。
- TP=8 decode 阶段 moe.allreduce 平均耗时为 0.819577 ms，通信开销存在但已经不是主要瓶颈。
- TP=8 prefill 的若干 MHC/Norm 项偏大，仍需结合重复运行或去除首次 ATen/stream 初始化影响后再做生产性能判断。
