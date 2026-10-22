"""Worker registry backends and shared discovery protocols."""

from typing import Any

from ..environment import registry_ttl_seconds
from .base import WorkerRegistry
from .etcd_registry import EtcdWorkerRegistry
from .file_registry import FileWorkerRegistry
from .types import WorkerRecord, WorkerType
from .watcher import WorkerRegistryWatcher


def create_worker_registry(options: Any) -> WorkerRegistry:
    backend = options.discovery_backend
    # registry lease 在真正创建 backend 前读取环境，避免配置层层透传。
    ttl_seconds = registry_ttl_seconds()
    if backend == "file":
        # PD frontend 使用 discovery_root；旧配置对象仍使用 file_kv。
        discovery_root = getattr(options, "discovery_root", None)
        if discovery_root is None:
            discovery_root = options.file_kv
        return FileWorkerRegistry(
            discovery_root,
            lease_seconds=ttl_seconds,
        )
    if backend == "etcd":
        return EtcdWorkerRegistry(
            endpoints=options.etcd_endpoints,
            prefix=options.etcd_prefix,
            lease_seconds=ttl_seconds,
            username=options.etcd_username,
            password_env=options.etcd_password_env,
            ca_cert=options.etcd_ca_cert,
            cert_file=options.etcd_cert_file,
            key_file=options.etcd_key_file,
            timeout=options.etcd_timeout,
        )
    raise ValueError(f"unsupported discovery backend: {backend!r}")


__all__ = [
    "EtcdWorkerRegistry",
    "FileWorkerRegistry",
    "WorkerRegistry",
    "WorkerRegistryWatcher",
    "WorkerRecord",
    "WorkerType",
    "create_worker_registry",
]
