"""Construction helpers for discovery registry backends."""

from __future__ import annotations

from typing import Any

from .etcd_registry import EtcdServiceRegistry
from .registry import FileServiceRegistry, ServiceRegistry


def create_service_registry(options: Any) -> ServiceRegistry:
    backend = options.discovery_backend
    if backend == "file":
        return FileServiceRegistry(
            options.file_kv,
            lease_seconds=options.registry_ttl,
        )
    if backend == "etcd":
        return EtcdServiceRegistry(
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
