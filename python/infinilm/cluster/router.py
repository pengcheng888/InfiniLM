"""Worker selection and request routing for the InfiniLM frontend."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator

from .http_worker_client_factory import (
    HttpWorkerClientFactory,
    is_retryable_http_error,
)
from .metrics import RouterMetrics
from .registry import FileServiceWatcher, WorkerRecord

logger = logging.getLogger(__name__)


class NoWorkersAvailable(RuntimeError):
    pass


class RoundRobinRouter:
    def __init__(
        self,
        watcher: FileServiceWatcher,
        *,
        connect_timeout: float,
        request_timeout: float,
        request_plane: str = "http",
    ) -> None:
        self.watcher = watcher
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.request_plane = request_plane
        self.cursor = 0
        self.metrics = RouterMetrics()
        self.worker_client_factory = HttpWorkerClientFactory(
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )

    async def aclose(self) -> None:
        await self.worker_client_factory.aclose()

    def available_workers(self, model_name: str | None = None) -> list[WorkerRecord]:
        workers = self.watcher.workers
        workers = [worker for worker in workers if worker.transport == self.request_plane]
        if model_name:
            workers = [worker for worker in workers if worker.served_model_name == model_name]
        return workers

    def select(self, workers: list[WorkerRecord]) -> WorkerRecord:
        if not workers:
            raise NoWorkersAvailable("no healthy aggregated workers are available")
        index = self.cursor % len(workers)
        self.cursor += 1
        return workers[index]

    async def stream_chat(
        self,
        payload: dict[str, Any],
        *,
        model_name: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        request_id = payload.get("request_id") or f"cmpl-{uuid.uuid4().hex}"
        payload = {**payload, "request_id": request_id}
        workers = list(self.available_workers(model_name))
        if not workers:
            raise NoWorkersAvailable(
                f"no healthy workers for model {model_name or '<default>'}"
            )

        last_error: Exception | None = None
        for _ in range(len(workers)):
            worker = self.select(workers)
            client = self.worker_client_factory.create(worker)
            produced_output = False
            metrics_started = False
            try:
                self.metrics.requests_total += 1
                self.metrics.inflight += 1
                self.metrics.worker_requests[worker.instance_id] += 1
                metrics_started = True
                stream = client.stream_chat(payload)
                deadline = asyncio.get_running_loop().time() + self.request_timeout
                try:
                    while True:
                        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                        try:
                            frame = await asyncio.wait_for(
                                stream.__anext__(),
                                timeout=remaining,
                            )
                        except StopAsyncIteration:
                            break
                        produced_output = True
                        self.metrics.tokens_total += 1
                        yield {**frame, "worker_id": worker.instance_id}
                finally:
                    await stream.aclose()
                return
            except (
                ConnectionError,
                OSError,
                TimeoutError,
                asyncio.TimeoutError,
            ) as exc:
                last_error = exc
                self.metrics.errors_total += 1
                self.metrics.worker_errors[worker.instance_id] += 1
                logger.warning(
                        "Worker %s failed for request %s before retry boundary: %s (cause=%r)",
                    worker.instance_id,
                    request_id,
                    exc,
                        exc.__cause__,
                )
                if produced_output:
                    raise
                continue
            except Exception as exc:
                self.metrics.errors_total += 1
                self.metrics.worker_errors[worker.instance_id] += 1
                if not produced_output and is_retryable_http_error(exc):
                    last_error = exc
                    logger.warning(
                        "Worker %s failed for request %s before retry boundary: %s (cause=%r)",
                        worker.instance_id,
                        request_id,
                        exc,
                        exc.__cause__,
                    )
                    continue
                raise
            finally:
                if metrics_started:
                    self.metrics.inflight = max(0, self.metrics.inflight - 1)

        raise ConnectionError(f"all workers failed for request {request_id}: {last_error}")
