# Aggregated YAML Launcher

该 launcher 是“单角色进程启动器”：一次调用只启动一个 frontend 或一个
worker。`workers.replicas` 不允许出现在 YAML 中，worker 副本数由外部执行
多少次 `--role worker` 决定。

## 启动 frontend

```bash
python -m infinilm.cluster.aggregated.launch \
  --config python/infinilm/cluster/aggregated/launch/configs/Qwen3-0.6B.yaml \
  --role frontend
```

安装 editable 包后也可以使用：

```bash
infinilm.cluster-launch \
  --config python/infinilm/cluster/aggregated/launch/configs/Qwen3-0.6B.yaml \
  --role frontend
```

## 启动三个 worker 副本

每个 shell 调用启动一个 worker。`CUDA_VISIBLE_DEVICES` 只描述当前进程可见
的 GPU，不再用逗号分隔的 GPU 数量推导副本数：

```bash
CUDA_VISIBLE_DEVICES=0 python -m infinilm.cluster.aggregated.launch \
  --config python/infinilm/cluster/aggregated/launch/configs/Qwen3-0.6B.yaml \
  --role worker &

CUDA_VISIBLE_DEVICES=1 python -m infinilm.cluster.aggregated.launch \
  --config python/infinilm/cluster/aggregated/launch/configs/Qwen3-0.6B.yaml \
  --role worker &

CUDA_VISIBLE_DEVICES=2 python -m infinilm.cluster.aggregated.launch \
  --config python/infinilm/cluster/aggregated/launch/configs/Qwen3-0.6B.yaml \
  --role worker &
```

也可以把多张卡交给同一个 worker：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m infinilm.cluster.aggregated.launch \
  --config python/infinilm/cluster/aggregated/launch/configs/Qwen3-0.6B.yaml \
  --role worker
```

## 端口

示例配置使用 `workers.base_http_port: 0`。每次 worker 调用都会传
`--port 0`，由操作系统分配互不冲突的端口；worker 在 HTTP socket 绑定完成后
把真实端口写入 registry，frontend 不需要预知端口。

如果配置固定端口，launcher 使用：

```text
port = workers.base_http_port + workers.port_offset
```

因此同一台机器上重复执行 fixed-port worker 会发生端口冲突。多副本部署应
保持 `base_http_port: 0`。

## 模型路径覆盖

YAML 中的 `model.path` 可被环境变量覆盖，优先级为：

```text
INFINILM_MODEL_PATH > MODEL_PATH > MODEL > model.path
```

示例：

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/models/Qwen3-0.6B \
python -m infinilm.cluster.aggregated.launch \
  --config python/infinilm/cluster/aggregated/launch/configs/Qwen3-0.6B.yaml \
  --role worker
```

## Dry Run

`--dry-run` 只打印本次角色的实际命令，不创建子进程：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m infinilm.cluster.aggregated.launch \
  --config python/infinilm/cluster/aggregated/launch/configs/Qwen3-0.6B.yaml \
  --role worker \
  --dry-run
```

file discovery 仅适合同机部署；跨节点 frontend 与 worker 必须使用 etcd。

## 自动 KV 显存

KV cache 容量统一由 `engine.gpu_memory_utilization` 推导，默认 `0.85`：

```text
1. 先用 2 个 probe blocks 创建 engine
2. 加载权重并做全量 graph profiling
3. 销毁 profiling graph 并释放显存
4. 根据目标显存比例重新分配 KV cache
5. 重新完整 compile graph
6. 把最终 num_blocks 回填给 scheduler 和 Mooncake
```

因此 graph exec、graph inputs、block table、workspace 和模型权重都会进入
显存预算。`engine.num_blocks` 已移除，出现旧字段时配置解析会直接报错。
