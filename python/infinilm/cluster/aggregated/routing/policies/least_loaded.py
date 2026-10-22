from __future__ import annotations

from ....routing import (
    RoutingContext,
    RoutingCursor,
    RoutingPolicy,
    SelectionResult,
    WorkerInflightTracker,
    WorkerLoadScorer,
)


class LeastLoadedPolicy(RoutingPolicy):
    """选择综合负载评分最低的 worker。"""

    name = "least-loaded"

    def __init__(
        self,
        scorer: WorkerLoadScorer,
        inflight: WorkerInflightTracker,
        cursor: RoutingCursor,
    ) -> None:
        self.scorer = scorer
        self.inflight = inflight
        self.cursor = cursor

    def select(self, context: RoutingContext) -> SelectionResult:
        # inflight 来自 frontend 本地视图，其余负载来自 worker 心跳快照。
        scored = [
            (
                self.scorer.score(worker, self.inflight.count(worker.instance_id)),
                worker,
            )
            for worker in context.candidates
        ]
        minimum = min(score for score, _ in scored)
        # 平局时在最低分候选间轮询，避免固定打向同一个 worker。
        candidates = [worker for score, worker in scored if score == minimum]
        index = self.cursor.increment() % len(candidates)
        return SelectionResult(worker=candidates[index], reason=self.name)
