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
        choices=["file"],
        default="file",
        help="service discovery backend; only file is supported initially",
    )
    parser.add_argument(
        "--file-kv",
        default=os.getenv("DYN_FILE_KV", "/tmp/infinilm_store_kv"),
        help="directory used by the file discovery backend",
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
    worker_type: str = "aggregated"
    request_plane: str = "http"


@dataclass(frozen=True)
class FrontendOptions:
    """frontend 启动和运行所需的不可变配置。"""

    # file discovery 的命名空间；frontend 只查找相同 namespace 下的 worker。
    namespace: str
    # file discovery 根目录，frontend 从这里周期性读取 worker JSON 记录。
    file_kv: Path
    # 路由策略名；默认 round-robin，也支持 least-loaded。
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


def parse_worker_cluster_args(argv: list[str] | None = None) -> WorkerOptions:
    parser = argparse.ArgumentParser(add_help=False)
    _add_cluster_arguments(parser)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--advertise-host", default="127.0.0.1")
    args, _ = parser.parse_known_args(argv)
    return WorkerOptions(
        namespace=args.namespace,
        file_kv=Path(args.file_kv),
        registry_ttl=args.registry_ttl,
        heartbeat_interval=args.heartbeat_interval,
        served_model_name=args.served_model_name,
        advertise_host=args.advertise_host,
        request_plane=args.request_plane,
    )


def remove_cluster_worker_args(argv: list[str]) -> list[str]:
    """Remove cluster-only flags so BaseConfig can parse inference flags."""
    parser = argparse.ArgumentParser(add_help=False)
    _add_cluster_arguments(parser)
    parser.add_argument("--served-model-name")
    parser.add_argument("--advertise-host")
    _, remaining = parser.parse_known_args(argv)
    return remaining


def parse_frontend_args(argv: list[str] | None = None) -> FrontendOptions:
    parser = argparse.ArgumentParser(description="Start the InfiniLM cluster frontend")
    _add_cluster_arguments(parser)
    parser.add_argument(
        "--router-mode",
        choices=["least-loaded", "round-robin"],
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
    )
