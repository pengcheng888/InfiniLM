"""File-backed worker registry for a single-node InfiniLM cluster."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .types import (
    WorkerRecord,
    WorkerType,
    normalize_worker_types,
    worker_record_from_payload,
    worker_record_to_payload,
)

logger = logging.getLogger(__name__)

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")


class FileWorkerRegistry:
    """Writes one atomic JSON record per worker instance."""

    def __init__(self, root: Path, lease_seconds: float = 10.0) -> None:
        self.root = Path(root)
        self.lease_seconds = lease_seconds

    def new_record(
        self,
        *,
        namespace: str,
        component: str,
        endpoint: str,
        worker_type: WorkerType | str,
        model: str,
        served_model_name: str,
        host: str,
        port: int,
        engine_healthy: bool = True,
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
            state=state,
            metadata=metadata or {},
        )

    def _path(self, record: WorkerRecord) -> Path:
        safe_parts = [
            _UNSAFE_FILENAME.sub("-", part)
            for part in (record.namespace, record.component, record.endpoint)
        ]
        filename = f"{'.'.join(safe_parts)}.{record.instance_id}.json"
        return self.root / filename

    def _write(self, record: WorkerRecord) -> None:
        """通过原子文件替换同步写入一条 worker 注册记录。

        先写入同目录下的 ``.tmp`` 临时文件，再用 ``os.replace()`` 把临时文件
        原子替换成正式 JSON 文件。frontend watcher 只读取 ``*.json``，因此不会
        读到半截 JSON；写入失败时也只会留下临时文件，不会破坏旧注册记录。
        """
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path(record)
        temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
        payload = json.dumps(
            worker_record_to_payload(record),
            separators=(",", ":"),
            sort_keys=True,
        )
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)

    async def register(self, record: WorkerRecord) -> WorkerRecord:
        """注册 worker，同时避免阻塞服务端 event loop。

        ``_write()`` 包含同步磁盘操作。生产环境的 worker 会从 uvicorn
        event loop 调用注册逻辑，因此这里使用 ``asyncio.to_thread()``
        把文件写入交给线程池。工作线程写文件期间，event loop 仍可继续
        处理 HTTP/TCP 请求，以及 heartbeat/watcher 等后台任务。

        测试代码可以调用 ``asyncio.run(registry.register(record))``。
        对当前测试线程来说，这个调用是同步等待：它会创建临时 event loop，
        执行当前 coroutine 直到注册完成，然后关闭该 event loop。
        """
        await asyncio.to_thread(self._write, record)
        logger.info(
            "Registered worker %s at %s:%s in namespace %s",
            record.instance_id,
            record.host,
            record.port,
            record.namespace,
        )
        return record

    async def heartbeat(
        self,
        record: WorkerRecord,
        *,
        engine_healthy: bool | None = None,
    ) -> WorkerRecord:
        """刷新 worker 心跳和注册租约。

        每次 heartbeat 都会把 ``heartbeat_at`` 更新为当前时间，并把
        ``lease_expires_at`` 更新为 ``当前时间 + lease_seconds``。
        因此只要 worker 存活并持续 heartbeat，注册租约就会一直被顺延；
        worker crash 后 heartbeat 停止，lease_expires_at 不再更新，
        最终会被 frontend snapshot 判定为过期。
        """
        now = time.time()
        healthy = record.engine_healthy if engine_healthy is None else engine_healthy
        updated = replace(
            record,
            heartbeat_at=now,
            lease_expires_at=now + self.lease_seconds,
            engine_healthy=healthy,
        )
        await asyncio.to_thread(self._write, updated)
        return updated

    async def unregister(self, record: WorkerRecord) -> None:
        path = self._path(record)
        try:
            await asyncio.to_thread(path.unlink)
        except FileNotFoundError:
            return
        logger.info("Unregistered worker %s", record.instance_id)

    def _snapshot(
        self,
        namespace: str,
        endpoint: str | None = None,
        served_model_name: str | None = None,
        worker_types: Iterable[WorkerType | str] | None = None,
    ) -> list[WorkerRecord]:
        if not self.root.exists():
            return []

        now = time.time()
        records: list[WorkerRecord] = []

        for path in self.root.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                record = worker_record_from_payload(raw)
            except (OSError, ValueError, TypeError):
                logger.warning("Ignoring malformed discovery record %s", path)
                continue

            if record.namespace != namespace:
                continue
            if endpoint is not None and record.endpoint != endpoint:
                continue
            if served_model_name and record.served_model_name != served_model_name:
                continue
            allowed_worker_types = normalize_worker_types(worker_types)
            if record.worker_type not in allowed_worker_types:
                continue
            if not record.engine_healthy:
                continue

            # 文件删除发生在 frontend 周期性调用 snapshot() 的过程中：
            # 如果注册租约已过期，说明 worker 没有继续 heartbeat。
            # 这里直接删除对应 JSON 文件，避免长期运行后 discovery 目录
            # 堆积已失效的 worker 记录，同时不会把该 worker 返回给路由器。
            if record.lease_expires_at < now:
                path.unlink(missing_ok=True)
                continue
            records.append(record)

        return sorted(records, key=lambda item: item.instance_id)

    async def snapshot(
        self,
        namespace: str,
        endpoint: str | None = None,
        served_model_name: str | None = None,
        worker_types: Iterable[WorkerType | str] | None = None,
    ) -> list[WorkerRecord]:
        return await asyncio.to_thread(
            self._snapshot,
            namespace,
            endpoint,
            served_model_name,
            worker_types,
        )

    async def aclose(self) -> None:
        """File discovery owns no long-lived connection."""
