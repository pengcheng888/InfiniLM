"""按角色将聚合式集群 YAML 转换为单个子进程命令并托管其生命周期。"""

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


def validate_cuda_visible_devices(value: str | None) -> str:
    """校验当前 worker 可见的 GPU 列表，并返回原始配置文本。

    逗号分隔项的数量不再表示 worker 副本数。例如
    ``CUDA_VISIBLE_DEVICES=0,1,2`` 表示“这一个 worker 可见 GPU 0、1、2”，
    适合把多张卡交给同一个推理进程使用。
    """
    if value is None or not value.strip():
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must list at least one GPU for --role worker"
        )

    # 只检查空项和明确隐藏的 -1，不把列表长度解释成副本数。
    devices = [device.strip() for device in value.split(",")]
    if any(not device for device in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must not contain empty GPU entries")
    if any(device == "-1" for device in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must list at least one visible GPU")
    return value


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
    role: str,
    environ: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> list[LaunchCommand]:
    """根据显式角色构建恰好一个 frontend 或 worker 启动命令。"""
    # 测试可注入独立环境；真实启动使用当前进程环境。
    source = os.environ if environ is None else environ
    python = sys.executable if python_executable is None else python_executable
    # 副本数由外部执行多少次 worker 命令决定，launcher 不再从 GPU 数量推导。
    min_workers = config.frontend.router_min_initial_workers or 1

    if role == "frontend":
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
        return [LaunchCommand("frontend", frontend_argv, environment={})]

    if role != "worker":
        raise ValueError(f"unknown launch role: {role}")

    # base_http_port=0 时传 --port 0，由 OS 分配端口；固定端口只叠加固定偏移。
    port = (
        0
        if config.workers.base_http_port == 0
        else config.workers.base_http_port + config.workers.port_offset
    )
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
        worker_argv.extend(["--cache-index-dir", str(config.discovery.cache_index_dir)])
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

    cuda_visible_devices = validate_cuda_visible_devices(
        source.get("CUDA_VISIBLE_DEVICES")
    )
    return [
        LaunchCommand(
            "worker",
            worker_argv,
            # 原样传递整段 GPU 列表；它描述当前进程可见性，不描述副本数。
            environment={"CUDA_VISIBLE_DEVICES": cuda_visible_devices},
        )
    ]


def render_command(command: LaunchCommand) -> str:
    """渲染 dry-run 输出；真实启动不经过 shell。"""
    # 环境变量前缀便于把 dry-run 命令直接复制到 shell 中验证。
    prefix = "".join(
        f"{key}={value} " for key, value in sorted(command.environment.items())
    )
    return prefix + " ".join(command.argv)


def _terminate(processes: Sequence[subprocess.Popen[object]]) -> None:
    """先优雅终止子进程，超时后强制退出。"""
    # 先统一发送 SIGTERM，避免退出时仍有成员继续提供服务。
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
    """启动当前角色的子进程，并转发退出信号。"""
    processes: list[subprocess.Popen[object]] = []
    process_names: list[str] = []
    stopping = False

    def request_stop(signum, frame) -> None:
        # 信号处理函数只切换状态，清理动作统一放到主循环和 finally 中。
        nonlocal stopping
        stopping = True

    # Ctrl+C 或 systemd 停止请求表示停止 launcher 启动的进程。
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)

    try:
        for command in commands:
            # 继承部署环境，再覆盖该角色明确声明的环境变量。
            environment = os.environ.copy()
            environment.update(command.environment)
            processes.append(subprocess.Popen(command.argv, env=environment))
            process_names.append(command.name)

        while not stopping:
            # 子进程异常退出时，launcher 立即返回非零状态。
            for name, process in zip(process_names, processes):
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"{name} exited before launcher shutdown with code {returncode}"
                    )
            time.sleep(0.5)
    finally:
        # 正常退出、子进程异常和外部信号复用同一条清理路径。
        _terminate(processes)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> None:
    """解析 launcher CLI，负责加载配置、dry-run 或启动单个角色进程。"""
    from argparse import ArgumentParser

    from .config import load_cluster_launch_config

    parser = ArgumentParser(
        description="Start one aggregated InfiniLM role from a YAML file"
    )
    parser.add_argument("--config", required=True, help="path to launch YAML")
    parser.add_argument(
        "--role",
        choices=("frontend", "worker"),
        required=True,
        help="process role; each invocation starts exactly one process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without starting processes",
    )
    args = parser.parse_args(argv)
    config = load_cluster_launch_config(args.config)
    commands = build_launch_commands(config, role=args.role)
    if args.dry_run:
        # dry-run 不创建目录、不启动进程，只展示实际 argv。
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
