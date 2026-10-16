"""Worker registry backends and shared discovery protocols."""

from typing import Any

from .base import WorkerRegistry
from .etcd_registry import EtcdWorkerRegistry
from .file_registry import FileWorkerRegistry
from .types import WorkerRecord
from .watcher import WorkerRegistryWatcher


def create_worker_registry(options: Any) -> WorkerRegistry:
    backend = options.discovery_backend
    if backend == "file":
        return FileWorkerRegistry(
            options.file_kv,
            lease_seconds=options.registry_ttl,
        )
    if backend == "etcd":
        return EtcdWorkerRegistry(
            endpoints=options.etcd_endpoints,
            prefix=options.etcd_prefix,
            lease_seconds=options.registry_ttl,
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
    "create_worker_registry",
]
