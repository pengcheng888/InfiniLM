"""加载并严格校验聚合式集群的第一阶段 YAML 启动配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

MODEL_PATH_ENV_VARS = ("INFINILM_MODEL_PATH", "MODEL_PATH", "MODEL")


@dataclass(frozen=True)
class DiscoveryConfig:
    """服务发现配置；file 用于单机，etcd 用于共享注册中心。"""

    backend: str = "file"
    file_kv: Path | None = None
    etcd_endpoints: str = "127.0.0.1:2379"
    etcd_prefix: str = "/infinilm"
    etcd_username: str | None = None
    etcd_password_env: str = "ETCD_PASSWORD"
    etcd_ca_cert: Path | None = None
    etcd_cert_file: Path | None = None
    etcd_key_file: Path | None = None
    etcd_timeout: float = 3.0
    cache_index_dir: Path | None = None


@dataclass(frozen=True)
class FrontendConfig:
    """统一 OpenAI 入口的监听与路由配置。"""

    host: str = "127.0.0.1"
    port: int = 18000
    router_mode: str = "round-robin"
    router_min_initial_workers: int | None = None


@dataclass(frozen=True)
class WorkerConfig:
    """聚合式 worker 的监听地址和端口分配基准。"""

    host: str = "127.0.0.1"
    advertise_host: str = "127.0.0.1"
    base_http_port: int = 18081
    port_offset: int = 10


@dataclass(frozen=True)
class EngineConfig:
    """传给 InfiniLM worker 的推理引擎参数。"""

    device: str = "nvidia"
    tp: int = 1
    block_size: int = 256
    max_new_tokens: int = 2644
    num_blocks: int = 500
    max_batch_size: int = 32
    log_level: str = "INFO"
    enable_paged_attn: bool = True
    enable_graph: bool = True
    attn: str = "flash-attn"


@dataclass(frozen=True)
class ClusterLaunchConfig:
    """YAML 解析后的聚合式集群完整配置。"""

    model_path: Path
    served_model_name: str
    namespace: str
    run_dir: Path
    discovery: DiscoveryConfig
    frontend: FrontendConfig
    workers: WorkerConfig
    engine: EngineConfig
    source_path: Path


def _mapping(parent: str, value: object) -> dict[str, object]:
    """校验 YAML 节点必须是字符串键的 mapping。"""
    if not isinstance(value, dict):
        raise ValueError(f"{parent} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{parent} keys must be strings")
    return value


def _reject_unknown(parent: str, value: dict[str, object], allowed: set[str]) -> None:
    """拒绝本阶段未支持的配置字段，避免配置被静默忽略。"""
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {parent} fields: {', '.join(sorted(unknown))}")


def _required_string(parent: str, value: dict[str, object], key: str) -> str:
    """读取必填且非空的字符串字段。"""
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{parent}.{key} must be a non-empty string")
    return raw


def _optional_string(
    parent: str,
    value: dict[str, object],
    key: str,
) -> str | None:
    """读取可选但设置后必须非空的字符串字段。"""
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{parent}.{key} must be a non-empty string when set")
    return raw


def _integer(
    parent: str,
    value: dict[str, object],
    key: str,
    default: int,
    *,
    minimum: int,
) -> int:
    """读取不小于 minimum 的整数字段，不接受 bool 冒充 int。"""
    raw = value.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise ValueError(f"{parent}.{key} must be an integer >= {minimum}")
    return raw


def _boolean(
    parent: str,
    value: dict[str, object],
    key: str,
    default: bool,
) -> bool:
    """读取布尔字段；YAML 的 true/false 会被 safe_load 转成 bool。"""
    raw = value.get(key, default)
    if not isinstance(raw, bool):
        raise ValueError(f"{parent}.{key} must be a boolean")
    return raw


def _number(
    parent: str,
    value: dict[str, object],
    key: str,
    default: float,
    *,
    minimum: float,
) -> float:
    """读取不小于 minimum 的数字字段，统一返回 float。"""
    raw = value.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < minimum:
        raise ValueError(f"{parent}.{key} must be a number >= {minimum}")
    return float(raw)


def _path(parent: str, value: dict[str, object], key: str) -> Path | None:
    """读取可选路径并展开 ~ 前缀。"""
    raw = _optional_string(parent, value, key)
    return Path(raw).expanduser() if raw is not None else None


def _resolved_path(path: Path | None, config_dir: Path) -> Path | None:
    """把相对路径锚定到 YAML 所在目录，并返回绝对路径。"""
    if path is None:
        return None
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _load_model_path(
    raw_model: dict[str, object],
    environ: Mapping[str, str],
) -> Path | None:
    """读取模型路径；环境变量覆盖优先于 YAML 默认值。"""
    raw = raw_model.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("model.path must be a non-empty string")

    # 环境变量用于部署环境覆盖；MODEL 保持与既有 run_cluster.sh 兼容。
    for name in MODEL_PATH_ENV_VARS:
        override = environ.get(name)
        if override is not None and override.strip():
            return Path(override).expanduser()
    return Path(raw).expanduser()


def load_cluster_launch_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ClusterLaunchConfig:
    """读取聚合式集群 YAML 配置，并校验第一阶段支持的字段。"""
    # 环境默认来自当前进程；测试可显式传入，便于隔离模型路径和 GPU 配置。
    source = os.environ if environ is None else environ
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load launch config {config_path}: {exc}") from exc

    root = _mapping("launch config", raw)
    # 严格字段检查能让配置拼错或版本不匹配时尽早失败。
    _reject_unknown(
        "launch config",
        root,
        {"version", "model", "cluster", "discovery", "frontend", "workers", "engine"},
    )
    if root.get("version") != 1:
        raise ValueError("launch config version must be 1")

    model = _mapping("model", root.get("model", {}))
    _reject_unknown("model", model, {"path", "served_model_name"})
    model_path = _load_model_path(model, source)
    assert model_path is not None
    served_model_name = _required_string("model", model, "served_model_name")

    cluster = _mapping("cluster", root.get("cluster", {}))
    _reject_unknown("cluster", cluster, {"namespace", "run_dir"})
    namespace = cluster.get("namespace", "infinilm-agg")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("cluster.namespace must be a non-empty string")
    run_dir_raw = _path("cluster", cluster, "run_dir") or Path("/tmp/infinilm-agg")

    discovery_raw = _mapping("discovery", root.get("discovery", {}))
    _reject_unknown(
        "discovery",
        discovery_raw,
        {
            "backend",
            "file_kv",
            "cache_index_dir",
            "etcd",
        },
    )
    backend = discovery_raw.get("backend", "file")
    if backend not in {"file", "etcd"}:
        raise ValueError("discovery.backend must be file or etcd")
    file_kv = _path("discovery", discovery_raw, "file_kv")
    cache_index_dir = _path("discovery", discovery_raw, "cache_index_dir")

    etcd_raw = _mapping("etcd", discovery_raw.get("etcd", {}))
    _reject_unknown(
        "discovery.etcd",
        etcd_raw,
        {
            "endpoints",
            "prefix",
            "username",
            "password_env",
            "ca_cert",
            "cert_file",
            "key_file",
            "timeout",
        },
    )

    frontend_raw = _mapping("frontend", root.get("frontend", {}))
    _reject_unknown(
        "frontend",
        frontend_raw,
        {"host", "port", "router_mode", "router_min_initial_workers"},
    )
    frontend_host = frontend_raw.get("host", "127.0.0.1")
    if not isinstance(frontend_host, str) or not frontend_host.strip():
        raise ValueError("frontend.host must be a non-empty string")
    router_mode = frontend_raw.get("router_mode", "round-robin")
    # 默认保持 round-robin，与现有 cluster frontend 行为一致。
    if router_mode not in {
        "kv-aware",
        "least-loaded",
        "round-robin",
        "session-affinity",
    }:
        raise ValueError(f"unsupported frontend.router_mode: {router_mode}")
    min_workers_raw = frontend_raw.get("router_min_initial_workers")
    if min_workers_raw is not None and (
        isinstance(min_workers_raw, bool)
        or not isinstance(min_workers_raw, int)
        or min_workers_raw < 1
    ):
        raise ValueError("frontend.router_min_initial_workers must be an integer >= 1")

    workers_raw = _mapping("workers", root.get("workers", {}))
    _reject_unknown(
        "workers",
        workers_raw,
        {"host", "advertise_host", "base_http_port", "port_offset", "replicas"},
    )
    if "replicas" in workers_raw:
        # 副本数唯一来源是 CUDA_VISIBLE_DEVICES，避免 YAML 与环境冲突。
        raise ValueError(
            "workers.replicas is not supported; worker count comes from "
            "CUDA_VISIBLE_DEVICES"
        )
    worker_host = workers_raw.get("host", "127.0.0.1")
    advertise_host = workers_raw.get("advertise_host", "127.0.0.1")
    for field, value in (("host", worker_host), ("advertise_host", advertise_host)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"workers.{field} must be a non-empty string")

    engine_raw = _mapping("engine", root.get("engine", {}))
    _reject_unknown(
        "engine",
        engine_raw,
        {
            "device",
            "tp",
            "block_size",
            "max_new_tokens",
            "num_blocks",
            "max_batch_size",
            "log_level",
            "enable_paged_attn",
            "enable_graph",
            "attn",
        },
    )
    device = engine_raw.get("device", "nvidia")
    attn = engine_raw.get("attn", "flash-attn")
    log_level = engine_raw.get("log_level", "INFO")
    for field, value in (("device", device), ("attn", attn), ("log_level", log_level)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"engine.{field} must be a non-empty string")

    config_dir = config_path.parent
    # run_dir 是本次集群运行时文件根目录；file 注册目录默认放在其中。
    run_dir = _resolved_path(run_dir_raw, config_dir)
    assert run_dir is not None
    if file_kv is None:
        file_kv = run_dir / "discovery"
    if backend == "etcd" and router_mode == "kv-aware" and cache_index_dir is None:
        # 多机 etcd 部署下，各 worker 的 cache index 必须落在共享存储。
        raise ValueError(
            "etcd + kv-aware routing requires discovery.cache_index_dir "
            "on shared storage"
        )

    discovery = DiscoveryConfig(
        # 相对路径统一在配置解析阶段解析，后续 launcher 只使用绝对路径。
        backend=backend,
        file_kv=_resolved_path(file_kv, config_dir),
        etcd_endpoints=etcd_raw.get("endpoints", "127.0.0.1:2379"),
        etcd_prefix=etcd_raw.get("prefix", "/infinilm"),
        etcd_username=_optional_string("discovery.etcd", etcd_raw, "username"),
        etcd_password_env=etcd_raw.get("password_env", "ETCD_PASSWORD"),
        etcd_ca_cert=_resolved_path(
            _path("discovery.etcd", etcd_raw, "ca_cert"), config_dir
        ),
        etcd_cert_file=_resolved_path(
            _path("discovery.etcd", etcd_raw, "cert_file"), config_dir
        ),
        etcd_key_file=_resolved_path(
            _path("discovery.etcd", etcd_raw, "key_file"), config_dir
        ),
        etcd_timeout=_number("discovery.etcd", etcd_raw, "timeout", 3.0, minimum=0.001),
        cache_index_dir=_resolved_path(cache_index_dir, config_dir),
    )
    if not isinstance(discovery.etcd_endpoints, str) or not discovery.etcd_endpoints:
        raise ValueError("discovery.etcd.endpoints must be a non-empty string")
    if not isinstance(discovery.etcd_prefix, str) or not discovery.etcd_prefix:
        raise ValueError("discovery.etcd.prefix must be a non-empty string")
    if (
        not isinstance(discovery.etcd_password_env, str)
        or not discovery.etcd_password_env
    ):
        raise ValueError("discovery.etcd.password_env must be a non-empty string")

    return ClusterLaunchConfig(
        model_path=_resolved_path(model_path, config_dir) or model_path,
        served_model_name=served_model_name,
        namespace=namespace,
        run_dir=run_dir,
        discovery=discovery,
        frontend=FrontendConfig(
            host=frontend_host,
            port=_integer("frontend", frontend_raw, "port", 18000, minimum=1),
            router_mode=router_mode,
            router_min_initial_workers=min_workers_raw,
        ),
        workers=WorkerConfig(
            host=worker_host,
            advertise_host=advertise_host,
            base_http_port=_integer(
                "workers", workers_raw, "base_http_port", 18081, minimum=1
            ),
            port_offset=_integer("workers", workers_raw, "port_offset", 10, minimum=0),
        ),
        engine=EngineConfig(
            device=device,
            tp=_integer("engine", engine_raw, "tp", 1, minimum=1),
            block_size=_integer("engine", engine_raw, "block_size", 256, minimum=1),
            max_new_tokens=_integer(
                "engine", engine_raw, "max_new_tokens", 2644, minimum=1
            ),
            num_blocks=_integer("engine", engine_raw, "num_blocks", 500, minimum=1),
            max_batch_size=_integer(
                "engine", engine_raw, "max_batch_size", 32, minimum=1
            ),
            log_level=log_level,
            enable_paged_attn=_boolean("engine", engine_raw, "enable_paged_attn", True),
            enable_graph=_boolean("engine", engine_raw, "enable_graph", True),
            attn=attn,
        ),
        source_path=config_path,
    )
