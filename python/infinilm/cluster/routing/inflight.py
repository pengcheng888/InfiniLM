"""Shared worker inflight tracking for routing and request orchestration."""

from __future__ import annotations


class WorkerInflightTracker:
    """维护每个 worker 的 frontend 本地活跃请求视图。"""

    def __init__(self) -> None:
        # 归零后删除 key，避免退出的 worker 长期保留在路由状态里。
        self._counts: dict[str, int] = {}

    def count(self, worker_id: str) -> int:
        """读取 worker 当前 inflight 数量，未知 worker 返回 0。"""
        return self._counts.get(worker_id, 0)

    def begin_request(self, worker_id: str) -> int:
        """记录一个请求开始，并返回更新后的 inflight 数量。"""
        value = self.count(worker_id) + 1
        self._counts[worker_id] = value
        return value

    def finish_request(self, worker_id: str) -> int:
        """记录一个请求结束；重复 finish 不会把计数变成负数。"""
        value = max(0, self.count(worker_id) - 1)
        if value == 0:
            self._counts.pop(worker_id, None)
        else:
            self._counts[worker_id] = value
        return value

    def has_inflight_requests(self, worker_id: str) -> bool:
        """判断 worker 是否仍有 frontend 本地视角的存量请求。"""
        return self.count(worker_id) > 0

    def snapshot(self) -> dict[str, int]:
        """返回当前活跃计数的浅拷贝，供测试和观测使用。"""
        return dict(self._counts)
