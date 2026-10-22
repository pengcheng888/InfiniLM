from __future__ import annotations

from ....environment import kv_router_min_tokens
from ....routing import (
    RoutingContext,
    RoutingCursor,
    RoutingPolicy,
    SelectionResult,
    WorkerInflightTracker,
    WorkerLoadScorer,
)
from ...kv_index import KvIndexStore
from .least_loaded import LeastLoadedPolicy


class KvAwarePolicy(RoutingPolicy):
    name = "kv-aware"

    def __init__(
        self,
        index: KvIndexStore,
        fallback: LeastLoadedPolicy,
        scorer: WorkerLoadScorer,
        inflight: WorkerInflightTracker,
        cursor: RoutingCursor,
        load_slack: float = 1.0,
    ) -> None:
        self.index = index
        self.fallback = fallback
        self.scorer = scorer
        self.inflight = inflight
        self.cursor = cursor
        self.min_tokens = kv_router_min_tokens()
        self.load_slack = load_slack

    def select(self, context: RoutingContext) -> SelectionResult:
        self.index.refresh(context.discovery_workers)
        matches = self.index.lookup(context.kv_prefix, context.candidates)
        matches = [
            match for match in matches if match.matched_tokens >= self.min_tokens
        ]
        minimum_load = min(
            self.scorer.score(worker, self.inflight.count(worker.instance_id))
            for worker in context.candidates
        )
        eligible = [
            match
            for match in matches
            if self.scorer.score(
                next(
                    worker
                    for worker in context.candidates
                    if worker.instance_id == match.worker_id
                ),
                self.inflight.count(match.worker_id),
            )
            <= minimum_load + self.load_slack
        ]

        if eligible:
            best = max(eligible, key=lambda match: match.matched_tokens)
            worker = next(
                worker
                for worker in context.candidates
                if worker.instance_id == best.worker_id
            )
            self.cursor.increment()
            return SelectionResult(
                worker=worker,
                reason="kv-prefix-hit",
                matched_tokens=best.matched_tokens,
            )

        fallback_result = self.fallback.select(context)
        return SelectionResult(
            worker=fallback_result.worker,
            reason="kv-prefix-miss-fallback",
            matched_tokens=None,
        )
