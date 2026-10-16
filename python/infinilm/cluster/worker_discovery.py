"""worker 的 file discovery 注册与心跳服务。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from infinilm.llm import AsyncLLMEngine

from .cache_index import WorkerCacheIndexExporter
from .worker_registry import WorkerRecord, WorkerRegistry
from .worker_traffic import WorkerTrafficMetrics

logger = logging.getLogger(__name__)


class WorkerDiscoveryService:
    """把 worker 的实际服务地址注册到 file discovery，并维护心跳租约。"""

    def __init__(
        self,
        *,
        engine: AsyncLLMEngine,
        registry: WorkerRegistry,
        record: WorkerRecord,
        heartbeat_interval: float | None = None,
        traffic_metrics: WorkerTrafficMetrics | None = None,
        cache_index_exporter: WorkerCacheIndexExporter | None = None,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.record = record
        self.heartbeat_interval = heartbeat_interval
        self.traffic_metrics = traffic_metrics
        self.cache_index_exporter = cache_index_exporter
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._heartbeat_task is not None:
            return

        self.record = self._with_engine_load()
        if self.cache_index_exporter is not None:
            await asyncio.to_thread(self.cache_index_exporter.start)
        await self.registry.register(self.record)

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat(),
            name=f"infinilm-worker-discovery-heartbeat-{self.record.instance_id}",
        )

        logger.info(
            "Registered HTTP request-plane worker at http://%s:%s/v1",
            self.record.host,
            self.record.port,
        )

    async def start_draining(self) -> None:
        """Advertise DRAINING immediately before graceful shutdown."""
        if self.record.state == "draining":
            return

        self.record = replace(self.record, state="draining")
        self.record = await self.registry.heartbeat(
            self.record,
            engine_healthy=self.engine.is_healthy(),
        )
        logger.info(
            "Worker %s entered DRAINING state",
            self.record.instance_id,
        )

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        await self.registry.unregister(self.record)
        if self.cache_index_exporter is not None:
            await asyncio.to_thread(self.cache_index_exporter.stop)

    async def _heartbeat(self) -> None:
        interval = self.heartbeat_interval or max(
            0.1, self.registry.lease_seconds / 3.0
        )
        while True:
            await asyncio.sleep(interval)
            try:
                self.record = self._with_engine_load()
                self.record = await self.registry.heartbeat(
                    self.record,
                    engine_healthy=self.engine.is_healthy(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Worker %s discovery heartbeat failed; retrying",
                    self.record.instance_id,
                )
                await asyncio.sleep(
                    min(interval, max(0.1, self.registry.lease_seconds / 3.0))
                )

    def _with_engine_load(self) -> WorkerRecord:
        metadata = dict(self.record.metadata)

        get_load_stats = getattr(self.engine, "get_load_stats", None)
        if callable(get_load_stats):
            try:
                load = get_load_stats()
            except Exception as exc:
                logger.warning("Could not read worker load stats: %s", exc)
            else:
                if isinstance(load, dict):
                    metadata["load"] = load

        if self.traffic_metrics is not None:
            metadata["traffic"] = self.traffic_metrics.snapshot()

        if self.cache_index_exporter is not None:
            metadata["kv_index"] = self.cache_index_exporter.metadata()

        return replace(self.record, metadata=metadata)
