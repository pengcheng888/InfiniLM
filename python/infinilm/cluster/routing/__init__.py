"""Routing policies and shared routing utilities."""

from ..kv_index import KvIndexStore
from .context import RoutingContext, SelectionResult
from .filter import WorkerCandidateFilter
from .load import LoadScoreWeights, WorkerLoad, WorkerLoadScorer, worker_load
from .policies import (
    KvAwarePolicy,
    LeastLoadedPolicy,
    RoundRobinPolicy,
    RoutingCursor,
    RoutingPolicy,
    SessionAffinityPolicy,
)


def create_routing_policy(
    router_mode: str,
    *,
    scorer: WorkerLoadScorer,
    worker_inflight: dict[str, int],
    kv_index: KvIndexStore,
    cursor: RoutingCursor,
) -> RoutingPolicy:
    if router_mode == "round-robin":
        return RoundRobinPolicy(cursor)

    least_loaded = LeastLoadedPolicy(scorer, worker_inflight, cursor)

    if router_mode == "least-loaded":
        return least_loaded

    if router_mode == "session-affinity":
        return SessionAffinityPolicy(
            least_loaded,
            scorer,
            worker_inflight,
            cursor,
        )

    if router_mode == "kv-aware":
        return KvAwarePolicy(
            kv_index,
            least_loaded,
            scorer,
            worker_inflight,
            cursor,
        )
    raise ValueError(f"unsupported router mode: {router_mode}")


__all__ = [
    "KvAwarePolicy",
    "LeastLoadedPolicy",
    "LoadScoreWeights",
    "RoundRobinPolicy",
    "RoutingContext",
    "RoutingCursor",
    "RoutingPolicy",
    "SelectionResult",
    "SessionAffinityPolicy",
    "WorkerCandidateFilter",
    "WorkerLoad",
    "WorkerLoadScorer",
    "create_routing_policy",
    "worker_load",
]
