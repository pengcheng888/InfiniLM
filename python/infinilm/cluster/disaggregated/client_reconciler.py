"""Background reconciliation for PD frontend worker clients."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from ..worker_registry import WorkerRecord, WorkerType
from .worker_client_pool import WorkerClientPoolManager

logger = logging.getLogger(__name__)


class PdClientReconciler:
    """在 frontend lifespan 内后台同步 discovery record 和 HTTP client。

    watcher 负责“发现 worker”，manager 负责“初始化/清理 client”。这个类
    把两者连接起来，并确保 bootstrap 不发生在 /health、/v1/models 或推理
    请求路径中。这里使用周期 snapshot，而不是变更事件：heartbeat、负载、
    DRAINING 状态和 metadata 都会随 snapshot 变化，不能只依赖 instance ID
    集合判断。
    """

    def __init__(
        self,
        *,
        watcher_records,
        manager: WorkerClientPoolManager,
        interval: float,
    ) -> None:
        self._watcher_records = watcher_records
        self._manager = manager
        self._interval = interval
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @staticmethod
    def _eligible_records(
        records: Iterable[WorkerRecord],
    ) -> list[WorkerRecord]:
        """只把可作为 PD 转发目标的 record 交给 client manager。"""
        return [
            record
            for record in records
            if record.worker_type in (WorkerType.PREFILL, WorkerType.DECODE)
            and record.state == "ready"
            and record.engine_healthy
        ]

    async def run(self) -> None:
        """周期执行全量 reconcile；单轮异常不能终止后台任务。"""
        while not self._stop.is_set():
            try:
                records = self._eligible_records(self._watcher_records())
                await self._manager.reconcile_records(records)
                # reconcile_records() 只调度初始化 task，不等待网络 I/O。
                # 这样某个慢 bootstrap 不会延迟下一轮 discovery 变化处理；
                # 同一 worker 的 pending task 由 manager 去重。
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to reconcile PD worker clients")

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        """启动后台 reconcile task；重复调用不会创建多个任务。"""
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(
            self.run(),
            name="infinilm-pd-client-reconciler",
        )

    async def stop(self) -> None:
        """停止后台任务。pending client 的资源由 manager.aclose() 统一清理。"""
        self._stop.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
