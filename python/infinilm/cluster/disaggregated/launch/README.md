# Disaggregated YAML Launcher

该 launcher 用于 PD 分离部署。一次调用只启动一个角色：

```text
frontend -> 统一 OpenAI API、发现 P/D worker、编排 prefill/decode
prefill  -> 计算 prompt、发布 Mooncake KV
decode   -> 拉取 KV、执行增量生成
```

`workers.replicas` 不允许出现在 YAML 中。需要多个 prefill 或 decode 副本时，
重复执行对应的 `--role prefill` 或 `--role decode` 命令。
`CUDA_VISIBLE_DEVICES` 只表示当前 worker 进程可见的 GPU，不表示副本数。

## 启动

```bash
# frontend
python -m infinilm.cluster.disaggregated.launch \
  --config python/infinilm/cluster/disaggregated/launch/configs/Qwen3-0.6B.yaml \
  --role frontend

# prefill
CUDA_VISIBLE_DEVICES=4 python -m infinilm.cluster.disaggregated.launch \
  --config python/infinilm/cluster/disaggregated/launch/configs/Qwen3-0.6B.yaml \
  --role prefill

# decode
CUDA_VISIBLE_DEVICES=5 python -m infinilm.cluster.disaggregated.launch \
  --config python/infinilm/cluster/disaggregated/launch/configs/Qwen3-0.6B.yaml \
  --role decode
```

安装 editable 包后可以使用：

```bash
infinilm.pd-cluster-launch \
  --config python/infinilm/cluster/disaggregated/launch/configs/Qwen3-0.6B.yaml \
  --role frontend
```

## 端口与 engine ID

示例 YAML 中 worker HTTP 端口和 Mooncake bootstrap 端口均为 `0`。每次启动时
Uvicorn/Mooncake 先绑定 `port=0`，然后把操作系统分配的真实端口写入 worker
registry，frontend 通过服务发现读取地址。

自动端口模式下，launcher 使用 `namespace-role-auto-PID` 生成 engine ID，避免
同一机器上重复启动的 prefill 冲突。也可以用环境变量 `ENGINE_ID` 或 YAML 的
`workers.engine_id` 显式指定；显式指定时运维系统必须保证全局唯一。

## Discovery

file backend 仅适合同机 frontend 与 worker。跨节点部署必须使用 etcd，并保证
所有角色使用相同 namespace、endpoints、prefix 和认证配置。

## 自动 KV 显存

KV cache 容量统一由 `engine.gpu_memory_utilization` 推导，默认 `0.85`。
prefill/decode worker 会先用 probe KV cache 和全量 graph profiling 估算
显存，再重新分配最终 KV cache 并重新 compile。Mooncake 注册的是最终 KV
cache 地址。旧 `engine.num_blocks` 字段不再允许。

## Dry Run

```bash
CUDA_VISIBLE_DEVICES=4 \
python -m infinilm.cluster.disaggregated.launch \
  --config python/infinilm/cluster/disaggregated/launch/configs/Qwen3-0.6B.yaml \
  --role prefill \
  --dry-run
```

Dry-run 不绑定端口、不创建模型 engine，也不依赖 Mooncake/GPU；它只验证 YAML
并打印最终命令。
