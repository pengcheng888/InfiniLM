"""etcd-backed service discovery for distributed InfiniLM clusters."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict, replace
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from .registry import WorkerRecord

logger = logging.getLogger(__name__)


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    value = endpoint.strip()
    if not value:
        raise ValueError("etcd endpoint must not be empty")

    if "://" not in value:
        value = f"//{value}"
    parsed = urlsplit(value)
    if parsed.hostname is None:
        raise ValueError(f"invalid etcd endpoint: {endpoint!r}")
    return parsed.hostname, parsed.port or 2379


class EtcdServiceRegistry:
    """Stores one lease-backed WorkerRecord JSON per worker in etcd.

    The existence of an etcd key is authoritative. Worker timestamps remain in
    the record for observability, but snapshot() does not use process clocks to
    decide whether a remote worker is alive.
    """

    def __init__(
        self,
        *,
        endpoints: str,
        prefix: str = "/infinilm",
        lease_seconds: float = 10.0,
        username: str | None = None,
        password_env: str = "ETCD_PASSWORD",
        ca_cert: str | None = None,
        cert_file: str | None = None,
        key_file: str | None = None,
        timeout: float = 3.0,
    ) -> None:
        try:
            import etcd3
        except ImportError as exc:
            raise RuntimeError(
                "etcd discovery requires the optional dependency 'etcd3'; "
                "install with: pip install -e '.[etcd]'"
            ) from exc

        self.lease_seconds = max(0.1, float(lease_seconds))
        self.etcd_lease_seconds = max(1, math.ceil(self.lease_seconds))
        self.prefix = prefix.rstrip("/") or "/"
        if not self.prefix.startswith("/"):
            raise ValueError("etcd prefix must start with '/'")

        username = username or None
        password = (os.getenv(password_env) or None) if password_env else None
        ca_cert = ca_cert or None
        cert_file = cert_file or None
        key_file = key_file or None
        if username and not password:
            raise ValueError(
                f"etcd password environment variable {password_env!r} is not set"
            )
        if bool(cert_file) != bool(key_file):
            raise ValueError("etcd client certificate and key must be set together")

        self._clients: list[Any] = []
        for endpoint in endpoints.split(","):
            if not endpoint.strip():
                continue
            if endpoint.strip().startswith("https://") and ca_cert is None:
                raise ValueError("https etcd endpoints require --etcd-ca-cert")
            host, port = _parse_endpoint(endpoint)
            self._clients.append(
                etcd3.client(
                    host=host,
                    port=port,
                    ca_cert=ca_cert,
                    cert_key=key_file,
                    cert_cert=cert_file,
                    timeout=timeout,
                    user=username,
                    password=password,
                )
            )
        if not self._clients:
            raise ValueError("at least one etcd endpoint is required")
        self._client_index = 0
        self._leases: dict[str, Any] = {}

    def _key(self, record: WorkerRecord) -> str:
        from urllib.parse import quote

        parts = (
            record.namespace,
            record.component,
            record.endpoint,
            record.instance_id,
        )
        encoded = "/".join(quote(str(part), safe="") for part in parts)
        return f"{self.prefix}/discovery/{encoded}"

    @property
    def _discovery_prefix(self) -> str:
        return f"{self.prefix}/discovery"

    def _next_client(self) -> Any:
        client = self._clients[self._client_index % len(self._clients)]
        self._client_index += 1
        return client

    async def _call(self, operation: Callable[[Any], Any]) -> Any:
        errors: list[Exception] = []
        for _ in range(len(self._clients)):
            client = self._next_client()
            try:
                return await asyncio.to_thread(operation, client)
            except Exception as exc:
                errors.append(exc)
                logger.warning(
                    "etcd operation failed against %s:%s: %s",
                    getattr(client, "host", "unknown"),
                    getattr(client, "port", "unknown"),
                    exc,
                )
        raise errors[-1]

    def new_record(
        self,
        *,
        namespace: str,
        component: str,
        endpoint: str,
        worker_type: str,
        model: str,
        served_model_name: str,
        host: str,
        port: int,
        engine_healthy: bool = True,
        transport: str = "http",
        state: str = "ready",
        metadata: dict[str, Any] | None = None,
    ) -> WorkerRecord:
        now = time.time()
        return WorkerRecord(
            namespace=namespace,
            component=component,
            endpoint=endpoint,
            instance_id=f"{os.getpid()}-{uuid.uuid4().hex[:12]}",
            worker_type=worker_type,
            model=model,
            served_model_name=served_model_name,
            host=host,
            port=port,
            registered_at=now,
            heartbeat_at=now,
            lease_expires_at=now + self.lease_seconds,
            engine_healthy=engine_healthy,
            transport=transport,
            state=state,
            metadata=metadata or {},
        )

    async def _register(self, record: WorkerRecord) -> WorkerRecord:
        now = time.time()
        updated = replace(
            record,
            heartbeat_at=now,
            lease_expires_at=now + self.lease_seconds,
        )
        payload = json.dumps(asdict(updated), separators=(",", ":"), sort_keys=True)
        key = self._key(updated)

        errors: list[Exception] = []
        for _ in range(len(self._clients)):
            client = self._next_client()
            lease = None
            try:
                lease = await asyncio.to_thread(client.lease, self.etcd_lease_seconds)
                await asyncio.to_thread(
                    client.put,
                    key,
                    payload,
                    lease=lease,
                )
            except Exception as exc:
                errors.append(exc)
                if lease is not None:
                    try:
                        await asyncio.to_thread(lease.revoke)
                    except Exception:
                        pass
                continue
            self._leases[updated.instance_id] = lease
            return updated

        raise errors[-1]

    async def register(self, record: WorkerRecord) -> WorkerRecord:
        updated = await self._register(record)
        logger.info(
            "Registered worker %s at %s:%s in namespace %s through etcd",
            updated.instance_id,
            updated.host,
            updated.port,
            updated.namespace,
        )
        return updated

    async def heartbeat(
        self,
        record: WorkerRecord,
        *,
        engine_healthy: bool | None = None,
    ) -> WorkerRecord:
        now = time.time()
        updated = replace(
            record,
            heartbeat_at=now,
            lease_expires_at=now + self.lease_seconds,
            engine_healthy=(
                record.engine_healthy if engine_healthy is None else engine_healthy
            ),
        )
        lease = self._leases.get(updated.instance_id)
        payload = json.dumps(asdict(updated), separators=(",", ":"), sort_keys=True)
        key = self._key(updated)

        try:
            if lease is None:
                raise RuntimeError("lease is not tracked")
            await asyncio.to_thread(lease.refresh)
            client = getattr(lease, "etcd_client", None) or self._next_client()
            await asyncio.to_thread(client.put, key, payload, lease=lease)
            return updated
        except Exception as exc:
            logger.warning(
                "etcd heartbeat failed for %s; re-registering: %s",
                updated.instance_id,
                exc,
            )
            return await self._register(updated)

    async def unregister(self, record: WorkerRecord) -> None:
        lease = self._leases.pop(record.instance_id, None)
        try:
            await self._call(lambda client: client.delete(self._key(record)))
        except Exception:
            logger.exception(
                "Failed to delete etcd discovery record %s", record.instance_id
            )
        if lease is not None:
            try:
                await asyncio.to_thread(lease.revoke)
            except Exception:
                logger.debug(
                    "Could not revoke etcd lease for %s; it will expire",
                    record.instance_id,
                )
        logger.info("Unregistered etcd worker %s", record.instance_id)

    async def aclose(self) -> None:
        for client in self._clients:
            try:
                await asyncio.to_thread(client.close)
            except Exception:
                logger.debug("Could not close an etcd client", exc_info=True)

    async def snapshot(
        self,
        namespace: str,
        endpoint: str | None = None,
        served_model_name: str | None = None,
        worker_types: Iterable[str] | None = None,
    ) -> list[WorkerRecord]:
        raw_items = await self._call(
            lambda client: list(client.get_prefix(self._discovery_prefix))
        )
        allowed_worker_types = (
            {"aggregated"} if worker_types is None else set(worker_types)
        )
        records: list[WorkerRecord] = []

        for raw_value, _metadata in raw_items:
            try:
                payload = json.loads(raw_value.decode("utf-8"))
                record = WorkerRecord(**payload)
            except (AttributeError, UnicodeDecodeError, ValueError, TypeError):
                logger.warning("Ignoring malformed etcd discovery record")
                continue
            if record.namespace != namespace:
                continue
            if endpoint is not None and record.endpoint != endpoint:
                continue
            if served_model_name and record.served_model_name != served_model_name:
                continue
            if record.worker_type not in allowed_worker_types:
                continue
            if not record.engine_healthy:
                continue
            records.append(record)

        return sorted(records, key=lambda item: item.instance_id)
