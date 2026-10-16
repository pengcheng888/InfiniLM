from __future__ import annotations

import time

from ...affinity import AffinityStore
from ...environment import affinity_capacity, affinity_ttl_seconds
from ..context import RoutingContext, SelectionResult
from ..load import WorkerLoadScorer
from .base import RoutingCursor, RoutingPolicy
from .least_loaded import LeastLoadedPolicy


class SessionAffinityPolicy(RoutingPolicy):
    """优先把相同会话路由到已绑定的 worker，以提高本地缓存命中。"""

    name = "session-affinity"

    def __init__(
        self,
        fallback: LeastLoadedPolicy,
        scorer: WorkerLoadScorer,
        inflight: dict[str, int],
        cursor: RoutingCursor,
        *,
        load_slack: float = 1.0,
        clock=time.monotonic,
    ) -> None:
        self._store = AffinityStore(
            ttl_seconds=affinity_ttl_seconds(),
            capacity=affinity_capacity(),
            clock=clock,
        )
        self.fallback = fallback
        self.scorer = scorer
        self.inflight = inflight
        self.cursor = cursor
        self.load_slack = load_slack

    @property
    def state_entries(self) -> int:
        """返回当前会话绑定数量，供 router 指标读取。"""
        return len(self._store)

    def retain_workers(self, worker_ids: set[str] | frozenset[str]) -> None:
        """worker 集合变化时清理不可路由的会话绑定。"""
        self._store.retain_workers(worker_ids)

    def select(self, context: RoutingContext) -> SelectionResult:
        # 无法构造会话 key 时没有可复用绑定，直接走轻载回退。
        if context.routing_key is None:
            fallback_result = self.fallback.select(context)
            return SelectionResult(
                worker=fallback_result.worker,
                reason="unkeyed-fallback",
                affinity_keyed=False,
            )

        lookup = self._store.get(
            context.routing_key,
            {worker.instance_id for worker in context.candidates},
        )
        # 只复用仍在当前候选集里的绑定，worker 退出或 draining 时自动失效。
        preferred = next(
            (
                worker
                for worker in context.candidates
                if worker.instance_id == lookup.worker_id
            ),
            None,
        )
        minimum_load = min(
            self.scorer.score(worker, self.inflight.get(worker.instance_id, 0))
            for worker in context.candidates
        )

        # preferred 仍可服务且负载不超过最低分加 slack 时保持亲和。
        if (
            preferred is not None
            and self.scorer.score(
                preferred, self.inflight.get(preferred.instance_id, 0)
            )
            <= minimum_load + self.load_slack
        ):
            self._store.assign(context.routing_key, preferred.instance_id)
            self.cursor.increment()
            return SelectionResult(
                worker=preferred,
                reason="affinity-hit",
                affinity_keyed=True,
                affinity_hit=True,
                affinity_expired=lookup.expired,
            )

        # 绑定失效或原 worker 相对过忙时，回退轻载策略并重新绑定会话。
        fallback_result = self.fallback.select(context)
        evicted = self._store.assign(
            context.routing_key, fallback_result.worker.instance_id
        )
        return SelectionResult(
            worker=fallback_result.worker,
            reason="affinity-miss-fallback",
            affinity_keyed=True,
            affinity_hit=False,
            affinity_expired=lookup.expired,
            affinity_evicted=evicted,
        )
