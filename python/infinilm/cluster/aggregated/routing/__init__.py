"""Routing policies used only by the aggregated cluster frontend."""

from ...routing import (
    RoutingContext,
    RoutingCursor,
    SelectionResult,
    WorkerCandidateFilter,
    WorkerInflightTracker,
    WorkerLoadScorer,
    worker_load,
)
from ..kv_index import KvIndexStore
from .policies import (
    KvAwarePolicy,
    LeastLoadedPolicy,
    RoundRobinPolicy,
    SessionAffinityPolicy,
)


def create_routing_policy(
    router_mode: str,
    *,
    scorer: WorkerLoadScorer,
    worker_inflight: WorkerInflightTracker,
    kv_index: KvIndexStore,
    cursor: RoutingCursor,
):
    """Create one aggregated routing policy from the configured mode."""
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
    "RoutingContext",
    "RoundRobinPolicy",
    "SelectionResult",
    "SessionAffinityPolicy",
    "WorkerCandidateFilter",
    "WorkerLoadScorer",
    "create_routing_policy",
    "worker_load",
]
