"""worker 的 file discovery 注册与心跳服务。"""

from __future__ import annotations

import asyncio
import logging

from infinilm.llm import AsyncLLMEngine

from .registry import FileServiceRegistry, WorkerRecord

logger = logging.getLogger(__name__)


class WorkerDiscoveryService:
    """把 worker 的实际服务地址注册到 file discovery，并维护心跳租约。"""

    def __init__(
        self,
        *,
        engine: AsyncLLMEngine,
        registry: FileServiceRegistry,
        record: WorkerRecord,
        heartbeat_interval: float | None = None,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.record = record
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._heartbeat_task is not None:
            return
        
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

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        await self.registry.unregister(self.record)

    async def _heartbeat(self) -> None:
        interval = self.heartbeat_interval or max(
            0.1, self.registry.lease_seconds / 3.0
        )
        while True:
            await asyncio.sleep(interval)
            self.record = await self.registry.heartbeat(
                self.record,
                engine_healthy=self.engine.is_healthy(),
            )
