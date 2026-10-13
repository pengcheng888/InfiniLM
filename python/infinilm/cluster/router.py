"""Worker selection and request routing for the InfiniLM frontend."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
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


@dataclass(frozen=True)
class WorkerLoad:
    waiting_requests: int = 0
    running_requests: int = 0
    max_batch_size: int = 1
    kv_usage: float = 0.0


def _non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        denominator_value = float(denominator)
        if denominator_value <= 0:
            return 0.0
        return min(1.0, max(0.0, float(numerator) / denominator_value))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def worker_load(worker: WorkerRecord) -> WorkerLoad:
    raw = worker.metadata.get("load")
    if not isinstance(raw, dict):
        return WorkerLoad()

    max_batch_size = max(1, _non_negative_int(raw.get("max_batch_size"), 1))
    return WorkerLoad(
        waiting_requests=_non_negative_int(raw.get("waiting_requests")),
        running_requests=_non_negative_int(raw.get("running_requests")),
        max_batch_size=max_batch_size,
        kv_usage=_ratio(
            raw.get("kv_used", 0),
            raw.get("kv_capacity", raw.get("kv_used", 0) + 1),
        ),
    )


class ClusterRouter:
    """Routes chat requests by composite load score or round-robin."""

    def __init__(
        self,
        watcher: FileServiceWatcher,
        *,
        connect_timeout: float,
        request_timeout: float,
        request_plane: str = "http",
        router_mode: str = "round-robin",
    ) -> None:
        if router_mode not in {"least-loaded", "round-robin"}:
            raise ValueError(f"unsupported router mode: {router_mode}")

        self.watcher = watcher
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.request_plane = request_plane
        self.router_mode = router_mode
        self.cursor = 0
        self.metrics = RouterMetrics(router_mode=router_mode)
        self.worker_inflight: dict[str, int] = {}
        self.worker_client_factory = HttpWorkerClientFactory(
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )

    async def aclose(self) -> None:
        await self.worker_client_factory.aclose()

    def available_workers(self, model_name: str | None = None) -> list[WorkerRecord]:
        workers = self.watcher.workers
        workers = [worker for worker in workers if worker.transport == self.request_plane]
        workers = [worker for worker in workers if worker.state == "ready"]
        if model_name:
            workers = [worker for worker in workers if worker.served_model_name == model_name]
        return workers

    def load_score(self, worker: WorkerRecord) -> float:
        inflight = self.worker_inflight.get(worker.instance_id, 0)
        load = worker_load(worker)
        running_ratio = load.running_requests / load.max_batch_size
        return (
            0.90 * inflight
            + 0.05 * load.waiting_requests
            + 0.03 * running_ratio
            + 0.02 * load.kv_usage
        )

    def update_worker_metrics(self, workers: list[WorkerRecord]) -> None:
        for worker in workers:
            load = worker_load(worker)
            self.metrics.worker_inflight[worker.instance_id] = (
                self.worker_inflight.get(worker.instance_id, 0)
            )
            self.metrics.set_worker_load(
                worker.instance_id,
                score=self.load_score(worker),
                waiting_requests=load.waiting_requests,
                running_requests=load.running_requests,
                kv_usage=load.kv_usage,
            )

    def select(self, workers: list[WorkerRecord]) -> WorkerRecord:
        workers = [worker for worker in workers if worker.state == "ready"]
        if not workers:
            raise NoWorkersAvailable("no healthy aggregated workers are available")

        if self.router_mode == "round-robin":
            candidates = workers
        else:
            scored = [(self.load_score(worker), worker) for worker in workers]
            minimum = min(score for score, _ in scored)
            candidates = [
                worker for score, worker in scored if score == minimum
            ]

        index = self.cursor % len(candidates)
        self.cursor += 1
        selected = candidates[index]

        self.update_worker_metrics(workers)
        return selected

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

        loop = asyncio.get_running_loop()
        request_started_at = loop.time()
        attempts = 0
        last_error: Exception | None = None
        final_worker_id: str | None = None
        first_token_seen = False
        last_token_at: float | None = None
        duration_recorded = False

        self.metrics.requests_total += 1
        self.metrics.mark_request_mode(bool(payload.get("stream", False)))
        self.metrics.inflight += 1
        try:
            for _ in range(len(workers)):
                worker = self.select(workers)
                worker_id = worker.instance_id
                final_worker_id = worker_id
                attempts += 1
                self.metrics.attempts_total += 1
                if attempts > 1:
                    self.metrics.retries_total += 1
                    self.metrics.worker_retries[worker_id] += 1

                client = self.worker_client_factory.create(worker)
                produced_output = False
                worker_metrics_started = False
                try:
                    self.metrics.worker_requests[worker_id] += 1
                    self.worker_inflight[worker_id] = (
                        self.worker_inflight.get(worker_id, 0) + 1
                    )
                    self.metrics.worker_inflight[worker_id] = self.worker_inflight[worker_id]
                    worker_metrics_started = True

                    stream = client.stream_chat(payload)
                    deadline = loop.time() + self.request_timeout
                    try:
                        while True:
                            remaining = max(0.0, deadline - loop.time())
                            try:
                                frame = await asyncio.wait_for(
                                    stream.__anext__(),
                                    timeout=remaining,
                                )
                            except StopAsyncIteration:
                                break

                            produced_output = True
                            frame_type = frame.get("type")
                            if frame_type == "delta":
                                self.metrics.tokens_total += 1
                                now = loop.time()
                                if not first_token_seen:
                                    first_token_seen = True
                                    self.metrics.observe_ttft(
                                        worker_id, now - request_started_at
                                    )
                                elif last_token_at is not None:
                                    self.metrics.observe_inter_token(
                                        now - last_token_at
                                    )
                                last_token_at = now
                            elif frame_type == "finish":
                                usage = frame.get("usage")
                                if isinstance(usage, dict):
                                    self.metrics.record_usage(worker_id, usage)
                            yield {**frame, "worker_id": worker_id}
                    finally:
                        await stream.aclose()

                    self.metrics.mark_worker_success(worker_id)
                    return
                except (
                    ConnectionError,
                    OSError,
                    TimeoutError,
                    asyncio.TimeoutError,
                ) as exc:
                    last_error = exc
                    self.metrics.errors_total += 1
                    self.metrics.worker_errors[worker_id] += 1
                    logger.warning(
                        "Worker %s failed for request %s before retry boundary: %s (cause=%r)",
                        worker_id,
                        request_id,
                        exc,
                        exc.__cause__,
                    )
                    if produced_output:
                        raise
                    continue
                except Exception as exc:
                    self.metrics.errors_total += 1
                    self.metrics.worker_errors[worker_id] += 1
                    if not produced_output and is_retryable_http_error(exc):
                        last_error = exc
                        logger.warning(
                            "Worker %s failed for request %s before retry boundary: %s (cause=%r)",
                            worker_id,
                            request_id,
                            exc,
                            exc.__cause__,
                        )
                        continue
                    raise
                finally:
                    if worker_metrics_started:
                        self.worker_inflight[worker_id] = max(
                            0, self.worker_inflight.get(worker_id, 1) - 1
                        )
                        self.metrics.worker_inflight[worker_id] = self.worker_inflight[
                            worker_id
                        ]

            raise ConnectionError(
                f"all workers failed for request {request_id}: {last_error}"
            )
        finally:
            self.metrics.inflight = max(0, self.metrics.inflight - 1)
            if not duration_recorded:
                duration_recorded = True
                self.metrics.observe_duration(
                    final_worker_id, loop.time() - request_started_at
                )
