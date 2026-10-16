"""Common routing policy interfaces and state."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import RoutingContext, SelectionResult


class RoutingCursor:
    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> int:
        current = self.value
        self.value += 1
        return current


class RoutingPolicy(ABC):
    name: str

    @property
    def state_entries(self) -> int:
        """返回策略持有的可观测状态条目数；无状态策略默认为 0。"""
        return 0

    def retain_workers(self, worker_ids: set[str] | frozenset[str]) -> None:
        """同步可路由 worker 集合；无状态策略默认无需处理。"""
        return None

    @abstractmethod
    def select(self, context: RoutingContext) -> SelectionResult:
        """Select exactly one worker from ``context.candidates``."""
