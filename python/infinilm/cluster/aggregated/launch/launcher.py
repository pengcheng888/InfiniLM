"""将聚合式集群 YAML 配置展开成子进程命令并托管其生命周期。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import ClusterLaunchConfig


@dataclass(frozen=True)
class LaunchCommand:
    """一个待启动的 frontend/worker 子进程描述。"""

    name: str
    argv: list[str]
    # 只记录额外环境变量；真实启动时仍会继承父进程环境。
    environment: dict[str, str]


def parse_cuda_visible_devices(value: str | None) -> list[str]:
    """解析 CUDA_VISIBLE_DEVICES，每个逗号分隔项对应一个 worker。"""
    # 空值表示启动器无法确定副本数，必须在构建命令前失败。
    if value is None or not value.strip():
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must list at least one GPU; worker replicas "
            "are derived from it"
        )
    # 保留 GPU 的原始文本编号，仅去除逗号两侧空白。
    devices = [device.strip() for device in value.split(",")]
    if any(not device for device in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must not contain empty GPU entries")
    if not devices or any(device == "-1" for device in devices):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must list at least one visible GPU; "
            "worker replicas are derived from it"
        )
    return devices


def _append_discovery_args(config: ClusterLaunchConfig, argv: list[str]) -> None:
    """把服务发现配置转换为 frontend/worker 的公共 CLI 参数。"""
    discovery = config.discovery
    # file 与 etcd 使用同一个 backend 开关，但后续参数互斥。
    argv.extend(["--discovery-backend", discovery.backend])
    if discovery.backend == "file":
        assert discovery.file_kv is not None
        argv.extend(["--file-kv", str(discovery.file_kv)])
        return

    # etcd 连接和安全配置必须同时传给 frontend/worker，保证注册与查询一致。
    argv.extend(["--etcd-endpoints", discovery.etcd_endpoints])
    argv.extend(["--etcd-prefix", discovery.etcd_prefix])
    if discovery.etcd_username is not None:
        argv.extend(["--etcd-username", discovery.etcd_username])
    argv.extend(["--etcd-password-env", discovery.etcd_password_env])
    if discovery.etcd_ca_cert is not None:
        argv.extend(["--etcd-ca-cert", str(discovery.etcd_ca_cert)])
    if discovery.etcd_cert_file is not None:
        argv.extend(["--etcd-cert-file", str(discovery.etcd_cert_file)])
    if discovery.etcd_key_file is not None:
        argv.extend(["--etcd-key-file", str(discovery.etcd_key_file)])
    argv.extend(["--etcd-timeout", str(discovery.etcd_timeout)])


def build_launch_commands(
    config: ClusterLaunchConfig,
    *,
    environ: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> list[LaunchCommand]:
    """将 YAML 配置展开成 frontend 和每个 GPU worker 的启动命令。"""
    # 测试可注入独立环境；真实启动使用当前进程环境。
    source = os.environ if environ is None else environ
    python = sys.executable if python_executable is None else python_executable
    # 第一阶段不读取 workers.replicas，副本数完全由可见 GPU 数量决定。
    devices = parse_cuda_visible_devices(source.get("CUDA_VISIBLE_DEVICES"))
    min_workers = config.frontend.router_min_initial_workers or len(devices)

    # frontend 是统一 OpenAI 入口，只通过服务发现获取 worker 地址。
    frontend_argv = [
        python,
        "-m",
        "infinilm.cluster.aggregated.frontend",
        "--namespace",
        config.namespace,
    ]
    _append_discovery_args(config, frontend_argv)
    if config.discovery.cache_index_dir is not None:
        frontend_argv.extend(
            ["--cache-index-dir", str(config.discovery.cache_index_dir)]
        )
    frontend_argv.extend(
        [
            "--router-mode",
            config.frontend.router_mode,
            "--router-min-initial-workers",
            str(min_workers),
            "--http-host",
            config.frontend.host,
            "--http-port",
            str(config.frontend.port),
        ]
    )
    commands = [
        LaunchCommand("frontend", frontend_argv, environment={}),
    ]

    # 每个 worker 绑定一项 GPU；端口从 base_http_port + port_offset 连续分配。
    for index, device in enumerate(devices):
        port = config.workers.base_http_port + index + config.workers.port_offset
        worker_argv = [
            python,
            "-m",
            "infinilm.cluster.worker",
            "--model",
            str(config.model_path),
            "--served-model-name",
            config.served_model_name,
            "--namespace",
            config.namespace,
        ]
        _append_discovery_args(config, worker_argv)
        if config.discovery.cache_index_dir is not None:
            # kv-aware 路由依赖 frontend 和 worker 使用同一个前缀索引目录。
            worker_argv.extend(
                ["--cache-index-dir", str(config.discovery.cache_index_dir)]
            )
        worker_argv.extend(
            [
                "--advertise-host",
                config.workers.advertise_host,
                "--host",
                config.workers.host,
                "--port",
                str(port),
                "--device",
                config.engine.device,
                "--tp",
                str(config.engine.tp),
                "--block-size",
                str(config.engine.block_size),
                "--max-new-tokens",
                str(config.engine.max_new_tokens),
                "--num-blocks",
                str(config.engine.num_blocks),
                "--max-batch-size",
                str(config.engine.max_batch_size),
                "--log-level",
                config.engine.log_level,
            ]
        )
        if config.engine.enable_paged_attn:
            worker_argv.append("--enable-paged-attn")
        if config.engine.enable_graph:
            worker_argv.append("--enable-graph")
        worker_argv.extend(["--attn", config.engine.attn])
        commands.append(
            LaunchCommand(
                f"worker-{index}",
                worker_argv,
                # 子进程只看到自己的 GPU，避免 engine 自动占用全部设备。
                environment={"CUDA_VISIBLE_DEVICES": device},
            )
        )
    return commands


def render_command(command: LaunchCommand) -> str:
    """渲染 dry-run 输出；真实启动不经过 shell。"""
    # 环境变量前缀便于把 dry-run 命令直接复制到 shell 中验证。
    prefix = "".join(
        f"{key}={value} " for key, value in sorted(command.environment.items())
    )
    return prefix + " ".join(command.argv)


def _terminate(processes: Sequence[subprocess.Popen[object]]) -> None:
    """先优雅终止子进程，超时后强制退出。"""
    # 先统一发送 SIGTERM，避免级联退出时仍有成员继续提供服务。
    for process in processes:
        if process.poll() is None:
            process.terminate()
    # 每个进程最多等待 10 秒，超时后升级为 SIGKILL。
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_launch_commands(commands: Sequence[LaunchCommand]) -> None:
    """启动全部 frontend/worker；任一子进程退出则回收其余子进程。"""
    processes: list[subprocess.Popen[object]] = []
    process_names: list[str] = []
    stopping = False

    def request_stop(signum, frame) -> None:
        # 信号处理函数只切换状态，清理动作统一放到主循环和 finally 中。
        nonlocal stopping
        stopping = True

    # Ctrl+C 或 systemd 停止请求表示停止整个集群，而不是只停 launcher。
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)

    try:
        for command in commands:
            # 继承部署环境，再覆盖该 worker 独有的 CUDA_VISIBLE_DEVICES。
            environment = os.environ.copy()
            environment.update(command.environment)
            processes.append(subprocess.Popen(command.argv, env=environment))
            process_names.append(command.name)

        while not stopping:
            # 聚合集群要求成员全部存活；任一异常退出都触发整体清理。
            for name, process in zip(process_names, processes):
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"{name} exited before cluster shutdown with code {returncode}"
                    )
            time.sleep(0.5)
    finally:
        # 正常退出、子进程异常和外部信号复用同一条清理路径。
        _terminate(processes)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> None:
    """解析 launcher CLI，负责加载配置、dry-run 或启动集群。"""
    from argparse import ArgumentParser

    from .config import load_cluster_launch_config

    parser = ArgumentParser(
        description="Start an aggregated InfiniLM cluster from a YAML file"
    )
    parser.add_argument("--config", required=True, help="path to launch YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without starting processes",
    )
    args = parser.parse_args(argv)
    config = load_cluster_launch_config(args.config)
    commands = build_launch_commands(config)
    if args.dry_run:
        # dry-run 不创建目录、不启动进程，只展示展开后的 argv。
        for command in commands:
            print(render_command(command))
        return

    if config.discovery.backend == "file":
        # file 服务注册依赖目录存在，先创建可避免 frontend/worker 竞争。
        assert config.discovery.file_kv is not None
        config.discovery.file_kv.mkdir(parents=True, exist_ok=True)
    run_launch_commands(commands)


if __name__ == "__main__":
    main()
