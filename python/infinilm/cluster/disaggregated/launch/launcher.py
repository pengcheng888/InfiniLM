"""按角色将 PD 分离集群 YAML 转换为单个子进程命令。"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Mapping

from .config import PDLaunchConfig


@dataclass(frozen=True)
class PDLaunchCommand:
    """一个待启动的 PD frontend/prefill/decode 子进程描述。"""

    name: str
    argv: list[str]
    # 只记录本角色额外覆盖的环境变量；真实启动仍继承父进程环境。
    environment: dict[str, str]


def validate_cuda_visible_devices(value: str | None) -> str:
    """校验当前 PD worker 可见的 GPU 列表，不把数量解释成副本数。"""
    if value is None or not value.strip():
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must list at least one GPU for worker roles"
        )
    devices = [device.strip() for device in value.split(",")]
    if any(not device for device in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must not contain empty GPU entries")
    if any(device == "-1" for device in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must list at least one visible GPU")
    return value


def _append_discovery_args(config: PDLaunchConfig, argv: list[str]) -> None:
    """把 file/etcd 服务发现配置展开为现有 CLI 参数。"""
    discovery = config.discovery
    argv.extend(["--discovery-backend", discovery.backend])
    if discovery.backend == "file":
        assert discovery.file_kv is not None
        argv.extend(["--file-kv", str(discovery.file_kv)])
        return

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


def _worker_ports(config: PDLaunchConfig) -> tuple[int, int]:
    """计算当前 worker 的 HTTP 与 bootstrap 端口。"""
    # 0 交给 OS 分配；固定端口只叠加固定偏移，没有实例序号。
    http_port = (
        0
        if config.workers.base_http_port == 0
        else config.workers.base_http_port + config.workers.port_offset
    )
    bootstrap_port = (
        0
        if config.workers.base_bootstrap_port == 0
        else config.workers.base_bootstrap_port + config.workers.bootstrap_port_offset
    )
    return http_port, bootstrap_port


def _engine_id(
    config: PDLaunchConfig,
    role: str,
    port: int,
    environ: Mapping[str, str],
) -> str:
    """为 Mooncake producer 生成稳定的 engine ID。

    优先级为 ENGINE_ID 环境变量、YAML workers.engine_id、自动生成。
    自动端口模式下使用 launcher PID，保证同一机器上重复启动不重名。
    """
    environment_engine_id = environ.get("ENGINE_ID", "").strip()
    if environment_engine_id:
        return environment_engine_id
    if config.workers.engine_id is not None:
        return config.workers.engine_id
    if port == 0:
        return f"{config.namespace}-{role}-auto-{os.getpid()}"
    return f"{config.namespace}-{role}-{port}"


def _kv_transfer_config(
    config: PDLaunchConfig,
    engine_id: str,
    role: str,
) -> str:
    """构造 MooncakeConnector 的 KV transfer JSON。"""
    # worker.py 会按 prefill/decode 补充 kv_role；这里只传公共字段。
    raw = {
        "kv_connector": "MooncakeConnector",
        "engine_id": engine_id,
        "kv_connector_extra_config": {
            "mooncake_protocol": config.mooncake.protocol,
            "num_workers": config.mooncake.workers,
        },
    }
    if role == "prefill":
        raw["kv_role"] = "kv_producer"
    elif role == "decode":
        raw["kv_role"] = "kv_consumer"
    return json.dumps(raw, separators=(",", ":"))


def build_pd_launch_commands(
    config: PDLaunchConfig,
    *,
    role: str,
    environ: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> list[PDLaunchCommand]:
    """根据显式角色构建恰好一个 PD 服务进程命令。"""
    if role not in {"frontend", "prefill", "decode"}:
        raise ValueError(f"unknown PD launch role: {role}")

    source = os.environ if environ is None else environ
    python = sys.executable if python_executable is None else python_executable
    if role == "frontend":
        # PD frontend 不加载模型，也不依赖 CUDA；它只做发现、路由和编排。
        argv = [
            python,
            "-m",
            "infinilm.cluster.disaggregated.frontend",
            "--namespace",
            config.namespace,
        ]
        _append_discovery_args(config, argv)
        argv.extend(
            [
                "--router-mode",
                config.frontend.router_mode,
                "--min-prefill-workers",
                str(config.frontend.min_prefill_workers),
                "--min-decode-workers",
                str(config.frontend.min_decode_workers),
                "--http-host",
                config.frontend.host,
                "--http-port",
                str(config.frontend.port),
            ]
        )
        return [PDLaunchCommand("frontend", argv, environment={})]

    cuda_visible_devices = validate_cuda_visible_devices(
        source.get("CUDA_VISIBLE_DEVICES")
    )
    http_port, bootstrap_port = _worker_ports(config)
    engine_id = _engine_id(config, role, http_port, source)
    argv = [
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
    _append_discovery_args(config, argv)
    argv.extend(
        [
            "--advertise-host",
            config.workers.advertise_host,
            "--disaggregation-mode",
            role,
            "--mooncake-bootstrap-host",
            config.workers.bootstrap_host,
            "--mooncake-bootstrap-port",
            str(bootstrap_port),
            "--host",
            config.workers.host,
            "--port",
            str(http_port),
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
        argv.append("--enable-paged-attn")
    if config.engine.enable_graph:
        argv.append("--enable-graph")
    argv.extend(["--attn", config.engine.attn])
    argv.extend(["--kv-transfer-config", _kv_transfer_config(config, engine_id, role)])

    # Mooncake 底层通过这些环境变量发现本机传输地址；与显式 CLI 参数保持一致。
    environment = {
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "INFINILM_HOST_IP": config.workers.advertise_host,
        "INFINILM_MOONCAKE_BOOTSTRAP_HOST": config.workers.bootstrap_host,
        "INFINILM_MOONCAKE_BOOTSTRAP_PORT": str(bootstrap_port),
    }
    return [PDLaunchCommand(role, argv, environment=environment)]


def render_command(command: PDLaunchCommand) -> str:
    """渲染可复制的 dry-run 命令；真实启动不经过 shell。"""
    prefix = "".join(
        f"{key}={value} " for key, value in sorted(command.environment.items())
    )
    return prefix + " ".join(command.argv)


def main(argv: list[str] | None = None) -> None:
    """解析 PD launcher CLI，dry-run 或启动一个显式角色。"""
    from argparse import ArgumentParser

    from .config import load_pd_launch_config

    parser = ArgumentParser(
        description="Start one InfiniLM PD role from a YAML file"
    )
    parser.add_argument("--config", required=True, help="path to PD launch YAML")
    parser.add_argument(
        "--role",
        choices=("frontend", "prefill", "decode"),
        required=True,
        help="process role; each invocation starts exactly one process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command without starting a process",
    )
    args = parser.parse_args(argv)
    config = load_pd_launch_config(args.config)
    if config.engine.tp != 1:
        raise ValueError("PD cluster workers currently require engine.tp=1")

    commands = build_pd_launch_commands(config, role=args.role)
    if args.dry_run:
        print(render_command(commands[0]))
        return

    if config.discovery.backend == "file":
        assert config.discovery.file_kv is not None
        config.discovery.file_kv.mkdir(parents=True, exist_ok=True)

    command = commands[0]
    environment = os.environ.copy()
    environment.update(command.environment)
    # 与 run_pd_worker.sh 的 exec 语义一致：launcher 自身被 worker 替换，
    # SIGTERM/SIGINT 由 worker 的 DRAINING 逻辑直接处理。
    os.execvpe(command.argv[0], command.argv, environment)


if __name__ == "__main__":
    main()
