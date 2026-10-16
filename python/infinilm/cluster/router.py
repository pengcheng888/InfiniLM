"""Worker selection and request routing for the InfiniLM frontend."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator

from .affinity import affinity_key
from .http_worker_client_factory import (
    HttpWorkerClientFactory,
    is_retryable_http_error,
)
from .kv_index import KvIndexStore
from .metrics import RouterMetrics
from .routing import (
    RoutingContext,
    RoutingCursor,
    SelectionResult,
    WorkerCandidateFilter,
    WorkerLoadScorer,
    create_routing_policy,
    worker_load,
)
from .worker_registry import WorkerRecord, WorkerRegistryWatcher

logger = logging.getLogger(__name__)


class NoWorkersAvailable(RuntimeError):
    pass


class ClusterRouter:
    """Selects a routing policy and orchestrates request retries."""

    def __init__(
        self,
        watcher: WorkerRegistryWatcher,
        *,
        connect_timeout: float,
        request_timeout: float,
        request_plane: str = "http",
        router_mode: str = "round-robin",
    ) -> None:
        self.watcher = watcher
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.request_plane = request_plane
        self.router_mode = router_mode
        self.metrics = RouterMetrics(router_mode=router_mode)
        self.worker_inflight: dict[str, int] = {}
        self.load_scorer = WorkerLoadScorer()
        self.candidate_filter = WorkerCandidateFilter(request_plane)
        self.routing_cursor = RoutingCursor()
        self.kv_index = KvIndexStore()
        self.policy = create_routing_policy(
            router_mode,
            scorer=self.load_scorer,
            worker_inflight=self.worker_inflight,
            kv_index=self.kv_index,
            cursor=self.routing_cursor,
        )
        self.worker_client_factory = HttpWorkerClientFactory(
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )

    async def aclose(self) -> None:
        await self.worker_client_factory.aclose()

    @property
    def cursor(self) -> int:
        return self.routing_cursor.value

    @property
    def affinity_entries(self) -> int:
        """返回会话亲和条目数；非该模式下恒为 0。"""
        return self.policy.state_entries

    def routable_workers(self, model_name: str | None = None) -> list[WorkerRecord]:
        return self.candidate_filter.filter(self.watcher.workers, model_name)

    def load_score(self, worker: WorkerRecord) -> float:
        return self.load_scorer.score(
            worker,
            self.worker_inflight.get(worker.instance_id, 0),
        )

    def update_worker_metrics(self, workers: list[WorkerRecord]) -> None:
        self.refresh_kv_index()
        self.policy.retain_workers(
            {
                worker.instance_id
                for worker in self.watcher.workers
                if worker.state == "ready"
            }
        )

        for worker in workers:
            load = worker_load(worker)
            self.metrics.worker_inflight[worker.instance_id] = self.worker_inflight.get(
                worker.instance_id, 0
            )
            self.metrics.set_worker_load(
                worker.instance_id,
                score=self.load_score(worker),
                waiting_requests=load.waiting_requests,
                running_requests=load.running_requests,
                kv_usage=load.kv_usage,
            )

    def refresh_kv_index(self) -> None:
        self.kv_index.refresh(self.watcher.workers)
        self.metrics.kv_index_entries = self.kv_index.entries()

    def _record_selection_metrics(self, result: SelectionResult) -> None:
        if self.router_mode == "session-affinity":
            if result.affinity_keyed is False:
                self.metrics.affinity_unkeyed_requests_total += 1
                return
            if result.affinity_expired:
                self.metrics.affinity_expired_total += 1
            if result.affinity_hit:
                self.metrics.affinity_hits_total += 1
            else:
                self.metrics.affinity_misses_total += 1
            if result.affinity_evicted:
                self.metrics.affinity_evictions_total += 1
        elif self.router_mode == "kv-aware":
            if result.matched_tokens is None:
                self.metrics.kv_miss_requests_total += 1
            else:
                self.metrics.kv_hit_requests_total += 1
                self.metrics.kv_predicted_saved_tokens_total += result.matched_tokens

    def select(
        self,
        workers: list[WorkerRecord],
        *,
        routing_key: str | None = None,
        kv_prefix: dict[str, Any] | None = None,
        count_affinity: bool = True,
        attempt_index: int = 0,
    ) -> WorkerRecord:
        workers = self.candidate_filter.filter(workers)
        if not workers:
            raise NoWorkersAvailable("no healthy aggregated workers are available")

        context = RoutingContext(
            candidates=workers,
            discovery_workers=self.watcher.workers,
            routing_key=routing_key,
            kv_prefix=kv_prefix,
            attempt_index=attempt_index,
            count_metrics=count_affinity,
        )
        result = self.policy.select(context)
        if count_affinity:
            if self.router_mode == "session-affinity":
                self.metrics.affinity_requests_total += bool(result.affinity_keyed)
            elif self.router_mode == "kv-aware":
                self.metrics.kv_aware_requests_total += 1
            self._record_selection_metrics(result)

        self.update_worker_metrics(workers)
        return result.worker

    async def stream_chat(
        self,
        payload: dict[str, Any],
        *,
        model_name: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        request_id = payload.get("request_id") or f"cmpl-{uuid.uuid4().hex}"
        payload = {**payload, "request_id": request_id}
        workers = list(self.routable_workers(model_name))
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
        kv_prefix = payload.get("__infinilm_kv_prefix__")
        payload = {
            key: value
            for key, value in payload.items()
            if key != "__infinilm_kv_prefix__"
        }
        routing_key = (
            affinity_key(payload) if self.router_mode == "session-affinity" else None
        )
        attempted_worker_ids: set[str] = set()

        self.metrics.requests_total += 1
        self.metrics.mark_request_mode(bool(payload.get("stream", False)))
        self.metrics.inflight += 1
        try:
            for attempt_index in range(len(workers)):
                candidates = [
                    worker
                    for worker in workers
                    if worker.instance_id not in attempted_worker_ids
                ]
                if not candidates:
                    break
                worker = self.select(
                    candidates,
                    routing_key=routing_key,
                    kv_prefix=kv_prefix,
                    count_affinity=attempt_index == 0,
                    attempt_index=attempt_index,
                )
                worker_id = worker.instance_id
                attempted_worker_ids.add(worker_id)
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
                    self.metrics.worker_inflight[worker_id] = self.worker_inflight[
                        worker_id
                    ]
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
                                cache = frame.get("cache")
                                if isinstance(cache, dict):
                                    self.metrics.kv_actual_saved_tokens_total += max(
                                        0, int(cache.get("matched_tokens", 0))
                                    )
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
