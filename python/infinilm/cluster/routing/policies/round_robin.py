from __future__ import annotations

from ..context import RoutingContext, SelectionResult
from .base import RoutingCursor, RoutingPolicy


class RoundRobinPolicy(RoutingPolicy):
    """在当前候选 worker 中按共享游标轮流选择。"""

    name = "round-robin"

    def __init__(self, cursor: RoutingCursor) -> None:
        self.cursor = cursor

    def select(self, context: RoutingContext) -> SelectionResult:
        # increment() 返回递增前的值，配合取模形成稳定轮询。
        index = self.cursor.increment() % len(context.candidates)
        return SelectionResult(worker=context.candidates[index], reason=self.name)
