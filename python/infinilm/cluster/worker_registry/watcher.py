"""Periodic watcher for worker registry backends."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from .base import WorkerRegistry
from .types import WorkerRecord

logger = logging.getLogger(__name__)


class WorkerRegistryWatcher:
    """Polls a worker registry and publishes the current worker set."""

    def __init__(
        self,
        registry: WorkerRegistry,
        *,
        worker_types: Iterable[str] | None = None,
    ) -> None:
        self.registry = registry
        self.worker_types = (
            tuple(worker_types) if worker_types is not None else ("aggregated",)
        )
        self._task: asyncio.Task[None] | None = None
        self._workers: list[WorkerRecord] = []
        self._changed = asyncio.Event()
        self._stop = asyncio.Event()

    @property
    def workers(self) -> list[WorkerRecord]:
        return list(self._workers)

    @property
    def changed(self) -> asyncio.Event:
        return self._changed

    async def refresh(self, namespace: str) -> list[WorkerRecord]:
        workers = await self.registry.snapshot(
            namespace, worker_types=self.worker_types
        )
        old_ids = {worker.instance_id for worker in self._workers}
        new_ids = {worker.instance_id for worker in workers}
        self._workers = workers
        if old_ids != new_ids:
            self._changed.set()
            logger.info(
                "Discovery workers changed: added=%s removed=%s current=%d",
                sorted(new_ids - old_ids),
                sorted(old_ids - new_ids),
                len(workers),
            )
        return workers

    async def run(self, namespace: str, interval: float) -> None:
        """Periodically snapshot the configured registry backend."""
        while not self._stop.is_set():
            try:
                await self.refresh(namespace)
            except Exception:
                logger.exception("Failed to refresh worker registry")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def start(self, namespace: str, interval: float) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(
            self.run(namespace, interval), name="infinilm-worker-registry-watcher"
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
