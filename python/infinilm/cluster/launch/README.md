# YAML Cluster Launch

第一阶段仅支持聚合式 InfiniLM cluster。启动配置由一个 YAML 文件描述，
launcher 负责把它展开成现有的 frontend 和 worker CLI 命令。

## 启动

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m infinilm.cluster.launch \
  --config python/infinilm/cluster/launch/configs/Qwen3-0.6B.yaml
```

安装 editable 包后也可以使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
infinilm.cluster-launch \
  --config python/infinilm/cluster/launch/configs/Qwen3-0.6B.yaml
```

## 副本数

`workers.replicas` 不允许出现在 YAML 中。worker 数量只由
`CUDA_VISIBLE_DEVICES` 中逗号分隔的 GPU 数量决定：

```text
CUDA_VISIBLE_DEVICES=0,1,2 -> 3 个 worker
CUDA_VISIBLE_DEVICES=0     -> 1 个 worker
```

第 `i` 个 worker 会拿到第 `i` 个 GPU，并使用：

```text
port = workers.base_http_port + i + workers.port_offset
```

## 模型路径覆盖

YAML 中的 `model.path` 可被环境变量覆盖，优先级为：

```text
INFINILM_MODEL_PATH > MODEL_PATH > MODEL > model.path
```

示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
MODEL_PATH=/models/Qwen3-0.6B \
python -m infinilm.cluster.launch \
  --config python/infinilm/cluster/launch/configs/Qwen3-0.6B.yaml
```

## Dry Run

`--dry-run` 只打印实际会执行的命令，不创建子进程：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m infinilm.cluster.launch \
  --config python/infinilm/cluster/launch/configs/Qwen3-0.6B.yaml \
  --dry-run
```

当前 YAML schema 只包含聚合式部署需要的 frontend、discovery、worker 端口和
常用 engine 参数。PD 分离部署后续使用单独的 schema，避免把两种部署形态混在
同一个配置文件里。
