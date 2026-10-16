"""Configuration and command-line parsing for the serving cluster."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


def _add_cluster_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--namespace",
        default=os.getenv("INFINILM_NAMESPACE", "infinilm-agg"),
        help="discovery namespace shared by frontend and workers",
    )
    parser.add_argument(
        "--discovery-backend",
        choices=["file", "etcd"],
        default=os.getenv("DYN_DISCOVERY_BACKEND", "file"),
        help="service discovery backend; file is local-only and etcd is distributed",
    )
    parser.add_argument(
        "--file-kv",
        default=os.getenv("DYN_FILE_KV", "/tmp/infinilm_store_kv"),
        help="directory used by the file discovery backend",
    )
    parser.add_argument(
        "--etcd-endpoints",
        default=os.getenv("ETCD_ENDPOINTS", "127.0.0.1:2379"),
        help="comma-separated etcd endpoints, for example host:2379",
    )
    parser.add_argument(
        "--etcd-prefix",
        default=os.getenv("ETCD_PREFIX", "/infinilm"),
        help="etcd key prefix used by discovery",
    )
    parser.add_argument("--etcd-username", default=os.getenv("ETCD_USERNAME"))
    parser.add_argument(
        "--etcd-password-env",
        default=os.getenv("ETCD_PASSWORD_ENV", "ETCD_PASSWORD"),
        help="environment variable containing the optional etcd password",
    )
    parser.add_argument("--etcd-ca-cert", default=os.getenv("ETCD_CA_CERT"))
    parser.add_argument("--etcd-cert-file", default=os.getenv("ETCD_CERT_FILE"))
    parser.add_argument("--etcd-key-file", default=os.getenv("ETCD_KEY_FILE"))
    parser.add_argument(
        "--etcd-timeout",
        type=float,
        default=float(os.getenv("ETCD_TIMEOUT", "3")),
    )
    parser.add_argument(
        "--cache-index-dir",
        default=None,
        help="shared cache-index directory; required for cross-node kv-aware routing",
    )
    parser.add_argument(
        "--request-plane",
        choices=["http"],
        default="http",
        help="transport used between frontend and inference workers",
    )
    parser.add_argument(
        "--registry-ttl",
        type=float,
        default=10.0,
        help="seconds after which a worker without heartbeat is removed",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=2.0,
        help="worker discovery heartbeat interval in seconds",
    )


@dataclass(frozen=True)
class WorkerOptions:
    namespace: str
    file_kv: Path
    registry_ttl: float
    heartbeat_interval: float
    served_model_name: str | None
    advertise_host: str
    component: str = "infinilm"
    endpoint: str = "generate"
    disaggregation_mode: str = "aggregated"
    worker_type: str = "aggregated"
    request_plane: str = "http"
    discovery_backend: str = "file"
    etcd_endpoints: str = "127.0.0.1:2379"
    etcd_prefix: str = "/infinilm"
    etcd_username: str | None = None
    etcd_password_env: str = "ETCD_PASSWORD"
    etcd_ca_cert: str | None = None
    etcd_cert_file: str | None = None
    etcd_key_file: str | None = None
    etcd_timeout: float = 3.0
    cache_index_dir: Path | None = None
    drain_grace_seconds: float = 1.0
    mooncake_bootstrap_host: str = "127.0.0.1"
    mooncake_bootstrap_port: int = 8998


@dataclass(frozen=True)
class FrontendOptions:
    """frontend 启动和运行所需的不可变配置。"""

    # file discovery 的命名空间；frontend 只查找相同 namespace 下的 worker。
    namespace: str
    # file discovery 根目录，frontend 从这里周期性读取 worker JSON 记录。
    file_kv: Path
    # 路由策略名；支持 round-robin、least-loaded、session-affinity、kv-aware。
    router_mode: str
    # frontend 完成启动前需要发现的最小健康 worker 数量。
    router_min_initial_workers: int
    # frontend 对外 OpenAI API 的监听地址。
    http_host: str
    # frontend 对外 OpenAI API 的监听端口。
    http_port: int
    # frontend 访问 worker HTTP endpoint 的建连超时时间，单位秒。
    connect_timeout: float
    # 单次推理请求的整体超时时间，单位秒。
    request_timeout: float
    # file discovery watcher 的轮询间隔，单位秒；实际运行时下限为 0.1 秒。
    health_interval: float
    # discovery 记录的租约时长；超过该时长没有心跳的 worker 会被移除。
    registry_ttl: float
    # worker 心跳间隔配置；frontend 也会用它约束 watcher 轮询频率。
    heartbeat_interval: float
    # frontend 到 worker 的请求面类型；当前仅支持 http。
    request_plane: str = "http"
    discovery_backend: str = "file"
    etcd_endpoints: str = "127.0.0.1:2379"
    etcd_prefix: str = "/infinilm"
    etcd_username: str | None = None
    etcd_password_env: str = "ETCD_PASSWORD"
    etcd_ca_cert: str | None = None
    etcd_cert_file: str | None = None
    etcd_key_file: str | None = None
    etcd_timeout: float = 3.0
    cache_index_dir: Path | None = None


def parse_worker_cluster_args(argv: list[str] | None = None) -> WorkerOptions:
    parser = argparse.ArgumentParser(add_help=False)
    _add_cluster_arguments(parser)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument(
        "--disaggregation-mode",
        choices=["aggregated", "prefill", "decode"],
        default=None,
        help="authoritative worker role; derives worker type and KV role",
    )
    parser.add_argument(
        "--worker-type",
        choices=["aggregated", "prefill", "decode"],
        default=None,
        help="deprecated compatibility alias for --disaggregation-mode",
    )
    parser.add_argument(
        "--mooncake-bootstrap-host",
        default=os.getenv("INFINILM_MOONCAKE_BOOTSTRAP_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--mooncake-bootstrap-port",
        type=int,
        default=int(os.getenv("INFINILM_MOONCAKE_BOOTSTRAP_PORT", "8998")),
    )
    parser.add_argument(
        "--drain-grace-seconds",
        type=float,
        default=1.0,
        help="seconds to advertise DRAINING before worker shutdown",
    )
    args, _ = parser.parse_known_args(argv)
    explicit_mode = _has_option(argv or [], "--disaggregation-mode")
    explicit_worker_type = _has_option(argv or [], "--worker-type")
    if (
        explicit_mode
        and explicit_worker_type
        and args.disaggregation_mode != args.worker_type
    ):
        raise ValueError(
            "--disaggregation-mode and --worker-type must match when both are set"
        )
    mode = args.disaggregation_mode or args.worker_type or "aggregated"

    return WorkerOptions(
        namespace=args.namespace,
        file_kv=Path(args.file_kv),
        registry_ttl=args.registry_ttl,
        heartbeat_interval=args.heartbeat_interval,
        served_model_name=args.served_model_name,
        advertise_host=args.advertise_host,
        component={
            "aggregated": "infinilm",
            "prefill": "prefill",
            "decode": "backend",
        }[mode],
        disaggregation_mode=mode,
        worker_type=mode,
        request_plane=args.request_plane,
        discovery_backend=args.discovery_backend,
        etcd_endpoints=args.etcd_endpoints,
        etcd_prefix=args.etcd_prefix,
        etcd_username=args.etcd_username,
        etcd_password_env=args.etcd_password_env,
        etcd_ca_cert=args.etcd_ca_cert,
        etcd_cert_file=args.etcd_cert_file,
        etcd_key_file=args.etcd_key_file,
        etcd_timeout=args.etcd_timeout,
        cache_index_dir=Path(args.cache_index_dir) if args.cache_index_dir else None,
        drain_grace_seconds=args.drain_grace_seconds,
        mooncake_bootstrap_host=args.mooncake_bootstrap_host,
        mooncake_bootstrap_port=args.mooncake_bootstrap_port,
    )


def remove_cluster_worker_args(argv: list[str]) -> list[str]:
    """Remove cluster-only flags so BaseConfig can parse inference flags."""
    parser = argparse.ArgumentParser(add_help=False)
    _add_cluster_arguments(parser)
    parser.add_argument("--served-model-name")
    parser.add_argument("--advertise-host")
    parser.add_argument("--disaggregation-mode")
    parser.add_argument("--worker-type")
    parser.add_argument("--mooncake-bootstrap-host")
    parser.add_argument("--mooncake-bootstrap-port", type=int)
    parser.add_argument("--drain-grace-seconds", type=float)
    _, remaining = parser.parse_known_args(argv)
    return remaining


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def parse_frontend_args(argv: list[str] | None = None) -> FrontendOptions:
    parser = argparse.ArgumentParser(description="Start the InfiniLM cluster frontend")
    _add_cluster_arguments(parser)
    parser.add_argument(
        "--router-mode",
        choices=[
            "kv-aware",
            "least-loaded",
            "round-robin",
            "session-affinity",
        ],
        default="round-robin",
    )
    parser.add_argument("--router-min-initial-workers", type=int, default=1)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=18000)
    parser.add_argument("--connect-timeout", type=float, default=3.0)
    parser.add_argument("--request-timeout", type=float, default=1000.0)
    parser.add_argument("--health-interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    return FrontendOptions(
        namespace=args.namespace,
        file_kv=Path(args.file_kv),
        router_mode=args.router_mode,
        router_min_initial_workers=args.router_min_initial_workers,
        http_host=args.http_host,
        http_port=args.http_port,
        connect_timeout=args.connect_timeout,
        request_timeout=args.request_timeout,
        health_interval=args.health_interval,
        registry_ttl=args.registry_ttl,
        heartbeat_interval=args.heartbeat_interval,
        request_plane=args.request_plane,
        discovery_backend=args.discovery_backend,
        etcd_endpoints=args.etcd_endpoints,
        etcd_prefix=args.etcd_prefix,
        etcd_username=args.etcd_username,
        etcd_password_env=args.etcd_password_env,
        etcd_ca_cert=args.etcd_ca_cert,
        etcd_cert_file=args.etcd_cert_file,
        etcd_key_file=args.etcd_key_file,
        etcd_timeout=args.etcd_timeout,
        cache_index_dir=Path(args.cache_index_dir) if args.cache_index_dir else None,
    )
