"""加载并严格校验 PD 分离集群 YAML 启动配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

MODEL_PATH_ENV_VARS = ("INFINILM_MODEL_PATH", "MODEL_PATH", "MODEL")


@dataclass(frozen=True)
class DiscoveryConfig:
    """PD frontend 与 worker 共享的服务发现配置。"""

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


@dataclass(frozen=True)
class FrontendConfig:
    """PD 统一 OpenAI 入口配置。"""

    host: str = "127.0.0.1"
    port: int = 18520
    router_mode: str = "round-robin"
    min_prefill_workers: int = 1
    min_decode_workers: int = 1


@dataclass(frozen=True)
class WorkerConfig:
    """一个 PD worker 进程的网络与 Mooncake bootstrap 配置。"""

    host: str = "127.0.0.1"
    advertise_host: str = "127.0.0.1"
    bootstrap_host: str = "127.0.0.1"
    base_http_port: int = 0
    base_bootstrap_port: int = 0
    port_offset: int = 0
    bootstrap_port_offset: int = 0
    engine_id: str | None = None


@dataclass(frozen=True)
class EngineConfig:
    """传给 prefill/decode worker 的推理引擎参数。"""

    device: str = "nvidia"
    tp: int = 1
    block_size: int = 256
    max_new_tokens: int = 32
    gpu_memory_utilization: float = 0.85
    max_batch_size: int = 32
    log_level: str = "INFO"
    enable_paged_attn: bool = True
    enable_graph: bool = True
    attn: str = "flash-attn"


@dataclass(frozen=True)
class MooncakeConfig:
    """Mooncake KV 传输参数。"""

    protocol: str = "tcp"
    workers: int = 4


@dataclass(frozen=True)
class PDLaunchConfig:
    """YAML 解析后的 PD 分离集群完整配置。"""

    model_path: Path
    served_model_name: str
    namespace: str
    run_dir: Path
    discovery: DiscoveryConfig
    frontend: FrontendConfig
    workers: WorkerConfig
    engine: EngineConfig
    mooncake: MooncakeConfig
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
    """拒绝未支持字段，避免拼错或不适用的配置被静默忽略。"""
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {parent} fields: {', '.join(sorted(unknown))}")


def _required_string(parent: str, value: dict[str, object], key: str) -> str:
    """读取必填且非空的字符串。"""
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{parent}.{key} must be a non-empty string")
    return raw


def _optional_string(
    parent: str,
    value: dict[str, object],
    key: str,
) -> str | None:
    """读取可选字符串；设置后必须非空。"""
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
    """读取不小于 minimum 的整数，并拒绝 bool。"""
    raw = value.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise ValueError(f"{parent}.{key} must be an integer >= {minimum}")
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


def _boolean(
    parent: str,
    value: dict[str, object],
    key: str,
    default: bool,
) -> bool:
    """读取 YAML 布尔字段。"""
    raw = value.get(key, default)
    if not isinstance(raw, bool):
        raise ValueError(f"{parent}.{key} must be a boolean")
    return raw


def _path(parent: str, value: dict[str, object], key: str) -> Path | None:
    """读取可选路径并展开 ~ 前缀。"""
    raw = _optional_string(parent, value, key)
    return Path(raw).expanduser() if raw is not None else None


def _resolved_path(path: Path | None, config_dir: Path) -> Path | None:
    """把相对路径锚定到 YAML 目录并转换为绝对路径。"""
    if path is None:
        return None
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _load_model_path(
    raw_model: dict[str, object],
    environ: Mapping[str, str],
) -> Path:
    """解析模型路径；环境变量优先于 YAML。"""
    raw = raw_model.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("model.path must be a non-empty string")
    for name in MODEL_PATH_ENV_VARS:
        override = environ.get(name)
        if override is not None and override.strip():
            return Path(override).expanduser()
    return Path(raw).expanduser()


def load_pd_launch_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> PDLaunchConfig:
    """读取并校验一个 PD YAML 配置文件。"""
    source = os.environ if environ is None else environ
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load PD launch config {config_path}: {exc}") from exc

    root = _mapping("PD launch config", raw)
    _reject_unknown(
        "PD launch config",
        root,
        {
            "version",
            "model",
            "cluster",
            "discovery",
            "frontend",
            "workers",
            "engine",
            "mooncake",
        },
    )
    if root.get("version") != 1:
        raise ValueError("PD launch config version must be 1")

    model = _mapping("model", root.get("model", {}))
    _reject_unknown("model", model, {"path", "served_model_name"})
    model_path = _load_model_path(model, source)
    served_model_name = _required_string("model", model, "served_model_name")

    cluster = _mapping("cluster", root.get("cluster", {}))
    _reject_unknown("cluster", cluster, {"namespace", "run_dir"})
    namespace = cluster.get("namespace", "infinilm-pd")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("cluster.namespace must be a non-empty string")
    run_dir_raw = _path("cluster", cluster, "run_dir") or Path("/tmp/infinilm-pd")
    config_dir = config_path.parent
    run_dir = _resolved_path(run_dir_raw, config_dir)
    assert run_dir is not None

    discovery_raw = _mapping("discovery", root.get("discovery", {}))
    _reject_unknown(
        "discovery", discovery_raw, {"backend", "file_kv", "etcd"}
    )
    backend = discovery_raw.get("backend", "file")
    if backend not in {"file", "etcd"}:
        raise ValueError("discovery.backend must be file or etcd")
    file_kv = _path("discovery", discovery_raw, "file_kv")
    if file_kv is None:
        file_kv = run_dir / "discovery"

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
        {
            "host",
            "port",
            "router_mode",
            "min_prefill_workers",
            "min_decode_workers",
        },
    )
    frontend_host = frontend_raw.get("host", "127.0.0.1")
    if not isinstance(frontend_host, str) or not frontend_host.strip():
        raise ValueError("frontend.host must be a non-empty string")
    router_mode = frontend_raw.get("router_mode", "round-robin")
    if router_mode not in {"round-robin", "least-loaded"}:
        raise ValueError(f"unsupported frontend.router_mode: {router_mode}")

    workers_raw = _mapping("workers", root.get("workers", {}))
    _reject_unknown(
        "workers",
        workers_raw,
        {
            "host",
            "advertise_host",
            "bootstrap_host",
            "base_http_port",
            "base_bootstrap_port",
            "port_offset",
            "bootstrap_port_offset",
            "engine_id",
            "replicas",
        },
    )
    if "replicas" in workers_raw:
        raise ValueError(
            "workers.replicas is not supported; run each worker command once per replica"
        )
    worker_host = workers_raw.get("host", "127.0.0.1")
    advertise_host = workers_raw.get("advertise_host", worker_host)
    bootstrap_host = workers_raw.get("bootstrap_host", advertise_host)
    for field, value in (
        ("host", worker_host),
        ("advertise_host", advertise_host),
        ("bootstrap_host", bootstrap_host),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"workers.{field} must be a non-empty string")
    if worker_host == "0.0.0.0" or advertise_host == "0.0.0.0":
        raise ValueError("workers 0.0.0.0 cannot be advertised; use a reachable host")

    engine_raw = _mapping("engine", root.get("engine", {}))
    _reject_unknown(
        "engine",
        engine_raw,
        {
            "device",
            "tp",
            "block_size",
            "max_new_tokens",
            "gpu_memory_utilization",
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
    gpu_memory_utilization = _number(
        "engine",
        engine_raw,
        "gpu_memory_utilization",
        0.85,
        minimum=0.001,
    )
    if gpu_memory_utilization > 1.0:
        raise ValueError("engine.gpu_memory_utilization must be <= 1.0")

    mooncake_raw = _mapping("mooncake", root.get("mooncake", {}))
    _reject_unknown("mooncake", mooncake_raw, {"protocol", "workers"})
    protocol = mooncake_raw.get("protocol", "tcp")
    if protocol not in {"tcp", "rdma"}:
        raise ValueError("mooncake.protocol must be tcp or rdma")

    discovery = DiscoveryConfig(
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
        etcd_timeout=float(etcd_raw.get("timeout", 3.0)),
    )
    for field, value in (
        ("etcd_endpoints", discovery.etcd_endpoints),
        ("etcd_prefix", discovery.etcd_prefix),
        ("etcd_password_env", discovery.etcd_password_env),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"discovery.etcd {field} must be a non-empty string")
    if discovery.etcd_timeout <= 0:
        raise ValueError("discovery.etcd.timeout must be > 0")

    return PDLaunchConfig(
        model_path=_resolved_path(model_path, config_dir) or model_path,
        served_model_name=served_model_name,
        namespace=namespace,
        run_dir=run_dir,
        discovery=discovery,
        frontend=FrontendConfig(
            host=frontend_host,
            port=_integer("frontend", frontend_raw, "port", 18520, minimum=1),
            router_mode=router_mode,
            min_prefill_workers=_integer(
                "frontend", frontend_raw, "min_prefill_workers", 1, minimum=1
            ),
            min_decode_workers=_integer(
                "frontend", frontend_raw, "min_decode_workers", 1, minimum=1
            ),
        ),
        workers=WorkerConfig(
            host=worker_host,
            advertise_host=advertise_host,
            bootstrap_host=bootstrap_host,
            # 0 表示交给 OS 自动分配，是多次启动 worker 的默认安全选择。
            base_http_port=_integer(
                "workers", workers_raw, "base_http_port", 0, minimum=0
            ),
            base_bootstrap_port=_integer(
                "workers", workers_raw, "base_bootstrap_port", 0, minimum=0
            ),
            port_offset=_integer("workers", workers_raw, "port_offset", 0, minimum=0),
            bootstrap_port_offset=_integer(
                "workers", workers_raw, "bootstrap_port_offset", 0, minimum=0
            ),
            engine_id=_optional_string("workers", workers_raw, "engine_id"),
        ),
        engine=EngineConfig(
            device=device,
            tp=_integer("engine", engine_raw, "tp", 1, minimum=1),
            block_size=_integer("engine", engine_raw, "block_size", 256, minimum=1),
            max_new_tokens=_integer(
                "engine", engine_raw, "max_new_tokens", 32, minimum=1
            ),
            gpu_memory_utilization=gpu_memory_utilization,
            max_batch_size=_integer(
                "engine", engine_raw, "max_batch_size", 32, minimum=1
            ),
            log_level=log_level,
            enable_paged_attn=_boolean(
                "engine", engine_raw, "enable_paged_attn", True
            ),
            enable_graph=_boolean("engine", engine_raw, "enable_graph", True),
            attn=attn,
        ),
        mooncake=MooncakeConfig(
            protocol=protocol,
            workers=_integer("mooncake", mooncake_raw, "workers", 4, minimum=1),
        ),
        source_path=config_path,
    )
