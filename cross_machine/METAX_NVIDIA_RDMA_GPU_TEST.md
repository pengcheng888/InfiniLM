# MetaX <-> NVIDIA RDMA GPU 通信测试记录

初版日期：2026-11-25
最近更新：2026-11-26

更新内容：

1. 补充 Mooncake RDMA 修复后的跨机 PD 验证结果。
2. 增加正向 `NVIDIA P -> MetaX D` 与反向 `MetaX P -> NVIDIA D` 两轮 v2 timing 数据摘要。
3. 固化跨机数据面与控制面地址，以及 `MC_TE_FILTERS` / `MC_GID_INDEX` 的推荐配置。
4. 记录一次历史排查：在 `batch_transfer_sync_write` 后加 5 秒 sleep 未改变语义输出，说明当时问题不是简单的 ready-flag 时序。

## 1. 测试目标

验证 Mooncake Transfer Engine v0.3.14 在 MetaX C550 与 NVIDIA A100 之间是否可以通过 RDMA 完成 GPU 缓冲区双向传输。

测试包含以下方向：

1. MetaX GPU -> NVIDIA GPU
2. NVIDIA GPU -> MetaX GPU

同时记录 MetaX 侧不设置 `MC_MACA_HOST_TRANSPORT=1` 时的失败现象，用于区分默认 MACA transport 与通用 RDMA host transport 的行为。

## 2. 测试环境

### 2.1 NVIDIA 节点

```text
Hostname: g3030
GPU: NVIDIA A100-SXM4-80GB
Mooncake: mooncake-transfer-engine-cuda13 0.3.14
Python: 3.12
PyTorch: 2.13.0+cu130
RDMA 地址: 192.168.100.10
HCA: mlx5_0, mlx5_1
协议: RoCE
```

NVIDIA 侧 Mooncake 安装包：

```text
mooncake_transfer_engine_cuda13-0.3.14-cp312-cp312-manylinux_2_28_x86_64.whl
```

### 2.2 MetaX 节点

MetaX 测试运行在远程服务器 `172.22.162.49` 的 `wangpengcheng_metax` 容器内。

```text
Host: 172.22.162.49
Container: wangpengcheng_metax
GPU: MetaX C550
Mooncake: v0.3.14
Git commit: e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c
Python: 3.10
PyTorch: 2.4.0+metax3.2.1.3
RDMA 地址: 192.168.100.30
HCA: mlx5_0, mlx5_1, mlx5_4, mlx5_5
协议: RoCE
```

MetaX 侧源码与测试脚本：

```text
/workspace_mooncake_v0.3.14
/workspace_mooncake_v0.3.14/test_mooncake.py
```

构建与安装记录见：

```text
/workspace_mooncake_v0.3.14/METAX_BUILD_AND_TEST_v0.3.14.md
```

## 3. 测试脚本与成功判据

测试脚本使用 `test_mooncake.py`。脚本流程：

1. server 创建随机 GPU buffer 并调用 `register_memory`。
2. client 创建全零 GPU buffer 并调用 `register_memory`。
3. client 通过 ZMQ 将 session id、远端地址和长度发送给 server。
4. server 调用 `transfer_sync_write`，把 server buffer 写入 client buffer。
5. client 打印传输后的 buffer，并与 server buffer 对比。

成功判据：

```text
server register_memory == 0
client register_memory == 0
transfer_sync_write == 0
client buffer 与 server buffer 数值一致
```

## 4. 关键配置

MetaX 侧跨平台 GPU RDMA 测试必须设置：

```bash
MC_MACA_HOST_TRANSPORT=1
```

设置后 Mooncake 日志显示：

```text
Using RDMA host transport for MACA
```

该配置使 MetaX 构建使用通用 RDMA transport，而不是默认的 MACA transport。跨 NVIDIA/MetaX 平台时，默认 MACA transport 无法与 NVIDIA 侧 RDMA transport 完成 GPU buffer 传输。

NVIDIA 侧无需该变量，直接使用 CUDA13 wheel 中的 RDMA transport。

## 5. MetaX GPU -> NVIDIA GPU

### 5.1 MetaX server

在 `172.22.162.49` 的 `wangpengcheng_metax` 容器内执行：

```bash
cd /workspace_mooncake_v0.3.14

CUDA_VISIBLE_DEVICES=1 \
MOONCAKE_HOSTNAME=192.168.100.30 \
MOONCAKE_PROTOCOL=rdma \
MC_MACA_HOST_TRANSPORT=1 \
MC_TRANSFER_TIMEOUT=20 \
python3 -u test_mooncake.py server \
  --device metax \
  --zmq-port 5583
```

关键输出：

```text
Using RDMA host transport for MACA
Mooncake 内存注册成功。ret_value: 0
Server TE 初始化完成，RPC: 15729
Server ZMQ ROUTER 监听 tcp://*:5583
```

### 5.2 NVIDIA client

在 NVIDIA 节点执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
MOONCAKE_HOSTNAME=192.168.100.10 \
MOONCAKE_PROTOCOL=rdma \
MC_TRANSFER_TIMEOUT=20 \
python -u /tmp/test_mooncake_v0314.py client \
  --device nvidia \
  --zmq-server 192.168.162.49 \
  --zmq-port 5583
```

关键输出：

```text
Using RDMA transport (RoCE/iWARP)
Mooncake 内存注册成功。ret_value: 0
Received RDMA ready ACK.
服务端回报 transfer 成功，ret=0
```

client 传输后的 64 个 `bfloat16` 数值与 MetaX server 的源 buffer 完全一致。

结论：

```text
MetaX GPU -> NVIDIA GPU RDMA 通信成功
```

## 6. NVIDIA GPU -> MetaX GPU

### 6.1 NVIDIA server

在 NVIDIA 节点执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
MOONCAKE_HOSTNAME=192.168.100.10 \
MOONCAKE_PROTOCOL=rdma \
MC_TRANSFER_TIMEOUT=20 \
python -u /tmp/test_mooncake_v0314.py server \
  --device nvidia \
  --zmq-port 5584
```

关键输出：

```text
Using RDMA transport (RoCE/iWARP)
Mooncake 内存注册成功。ret_value: 0
Server TE 初始化完成，RPC: 15812
Server ZMQ ROUTER 监听 tcp://*:5584
```

### 6.2 MetaX client

在 `172.22.162.49` 的 `wangpengcheng_metax` 容器内执行：

```bash
cd /workspace_mooncake_v0.3.14

CUDA_VISIBLE_DEVICES=2 \
MOONCAKE_HOSTNAME=192.168.100.30 \
MOONCAKE_PROTOCOL=rdma \
MC_MACA_HOST_TRANSPORT=1 \
MC_TRANSFER_TIMEOUT=20 \
python3 -u test_mooncake.py client \
  --device metax \
  --zmq-server 192.168.100.10 \
  --zmq-port 5584
```

关键输出：

```text
Using RDMA host transport for MACA
Mooncake 内存注册成功。ret_value: 0
Received RDMA ready ACK.
服务端回报 transfer 成功，ret=0
```

MetaX client 传输后的 64 个 `bfloat16` 数值与 NVIDIA server 的源 buffer 完全一致。

结论：

```text
NVIDIA GPU -> MetaX GPU RDMA 通信成功
```

## 7. 默认 MACA transport 失败对照

在 MetaX server 不设置 `MC_MACA_HOST_TRANSPORT=1` 时，MetaX 侧日志显示：

```text
Using MACA transport
Mooncake 内存注册成功。ret_value: 0
```

NVIDIA client GPU buffer 注册也成功，但 server 调用 `transfer_sync_write` 后返回：

```text
transfer_sync_write 失败，返回值: -1
```

该失败发生在 MetaX 默认 MACA transport 与 NVIDIA RDMA transport 的跨平台组合下。GPU buffer 注册成功并不代表两侧 transport 可以互通。

因此，跨 MetaX/NVIDIA 平台进行 GPU RDMA 传输时应固定使用：

```bash
MC_MACA_HOST_TRANSPORT=1
```

## 8. 测试中的非致命日志

测试过程中出现过以下 warning/error：

```text
Failed to modify QP to RTR
Re-establish connection
Retiring endpoint instead of resetting it
transport retry counter exceeded
Failed to restore fork state ... deregister already succeeded
```

其中 QP 建链失败后 Mooncake 会重试并切换可用 peer RNIC。最终 `transfer_sync_write` 返回 0，且数据完整一致，因此这些日志不影响本次通信结论。

`Failed to restore fork state ... deregister already succeeded` 出现在内存注销阶段，传输本身已经完成。

## 9. 结果摘要

```text
MetaX GPU -> NVIDIA GPU: 成功
NVIDIA GPU -> MetaX GPU: 成功

两侧 GPU 内存注册: 成功，返回 0
双向 transfer_sync_write: 成功，返回 0
数据校验: 64 个 bfloat16 数值一致
协议: RoCE/RDMA
必要配置: MetaX 侧 MC_MACA_HOST_TRANSPORT=1
```

最终结论：MetaX 与 NVIDIA 之间的 Mooncake RDMA GPU 缓冲区双向通信可用。

## 10. 2026-11-26 RDMA 修复后的跨机 PD 验证

### 10.1 结论

Mooncake 跨机 RDMA 修复后，以下链路均已验证可用：

1. `MetaX -> NVIDIA` GPU buffer 单测。
2. `NVIDIA -> MetaX` GPU buffer 单测。
3. `NVIDIA P -> MetaX D` 跨机 PD 分离服务。
4. `MetaX P -> NVIDIA D` 反向跨机 PD 分离服务。

两轮跨机 PD 服务均可正常输出中文语义内容，不再出现乱码。

### 10.2 推荐网络与 HCA 配置

跨机部署时建议区分控制面和数据面：

```text
NVIDIA 控制面 / API: 192.168.162.61
NVIDIA RDMA 数据面: 192.168.100.10

MetaX 控制面 / API: 192.168.162.49
MetaX RDMA 数据面: 192.168.100.30
```

跨机 Mooncake 建议固定以下配置：

```bash
MC_TE_FILTERS=mlx5_0
MC_GID_INDEX=3
```

MetaX 侧还需要：

```bash
MC_MACA_HOST_TRANSPORT=1
```

这样可以避免 Mooncake 自动选择到不兼容的 HCA/GID，导致 `Failed to modify QP to RTR`。

### 10.3 正向跨机 PD：NVIDIA P -> MetaX D

配置摘要：

```text
P 节点: NVIDIA A100, GPU 3
D 节点: MetaX C550, GPU 6
Proxy: NVIDIA
请求: 256
并发: 80
输入长度: 2048
输出长度: 512
warmup: 80
```

关键结果：

```text
Successful requests: 256
Failed requests: 0
Benchmark duration: 82.96 s
Output token throughput: 1579.94 tok/s
Mean TTFT: 3297.03 ms
Mean E2EL: 23905.82 ms
Mean ITL: 40.32 ms
D KV readiness avg: 3076.33 ms
```

详细报告：

```text
analysis_reports_nvidia/timing-pd-request-id-v2-256-cross-machine-rdma-fixed-20261126/timing-pd-service-v2-256-request-id-cross-machine-rdma-fixed-20261126.md
```

### 10.4 反向跨机 PD：MetaX P -> NVIDIA D

配置摘要：

```text
P 节点: MetaX C550, GPU 4
D 节点: NVIDIA A100, GPU 4
Proxy: NVIDIA
请求: 256
并发: 80
输入长度: 2048
输出长度: 512
warmup: 80
```

关键结果：

```text
Successful requests: 256
Failed requests: 0
Benchmark duration: 71.60 s
Output token throughput: 1830.49 tok/s
Mean TTFT: 5822.69 ms
Mean E2EL: 20679.79 ms
Mean ITL: 29.06 ms
D KV readiness avg: 5611.39 ms
```

详细报告：

```text
analysis_reports_nvidia/timing-pd-request-id-v2-256-cross-machine-reverse-rdma-fixed-20261126/timing-pd-service-v2-256-request-id-cross-machine-reverse-rdma-fixed-20261126.md
```

### 10.5 跨机脚本

正向链路：

```text
cross_machine/run_prefill_nvidia.sh
cross_machine/run_decode_metax.sh
cross_machine/run_proxy_nvidia.sh
```

反向链路：

```text
cross_machine/run_prefill_metax.sh
cross_machine/run_decode_nvidia.sh
cross_machine/run_proxy_nvidia.sh
```

反向链路中，Proxy 需要将 `PREFILL_API_HOST` 指向 MetaX 控制面，将 `DECODE_API_HOST` 指向 NVIDIA 控制面：

```bash
PREFILL_API_HOST=192.168.162.49 \
DECODE_API_HOST=192.168.162.61 \
bash cross_machine/run_proxy_nvidia.sh
```

### 10.6 历史排查补充

在 RDMA 修复前，曾在 P 侧 `batch_transfer_sync_write` 后加入 5 秒 sleep 做诊断。D 侧从收到 KV 到开始 decode 的间隔确实变为约 5 秒，但输出仍为乱码。

该测试说明当时的问题不是简单的 ready-flag 时序，而是跨平台 KV 数据面兼容性问题。Mooncake 修复后，无需额外 sleep，跨机 PD 语义输出恢复正常。
