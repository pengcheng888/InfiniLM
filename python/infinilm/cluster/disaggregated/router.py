"""Routing state for the prefill/decode separated frontend."""

from __future__ import annotations

from ..routing import (
    RoutingCursor,
    WorkerInflightTracker,
    WorkerLoadScorer,
)
from ..worker_registry import WorkerRecord, WorkerType


class DisaggregatedRouter:
    """为 PD frontend 执行按角色的 worker 选择和请求计数。

    frontend 负责模型/API route 过滤和 P/D 请求编排；这个类只维护
    round-robin 游标、least-loaded 评分、worker inflight 和角色总量。
    """

    def __init__(self, router_mode: str) -> None:
        self.router_mode = router_mode
        self.worker_inflight = WorkerInflightTracker()
        self.load_scorer = WorkerLoadScorer()
        # P/D 游标独立，避免 decode 选择推进 prefill 的轮询位置。
        self.cursors: dict[WorkerType, RoutingCursor] = {
            WorkerType.PREFILL: RoutingCursor(),
            WorkerType.DECODE: RoutingCursor(),
        }
        # 统计“已开始转发”的角色请求总量；失败请求也不会回退该计数。
        self.role_request_totals: dict[str, int] = {
            WorkerType.PREFILL.value: 0,
            WorkerType.DECODE.value: 0,
        }

    def select_record(
        self,
        role: WorkerType,
        records: list[WorkerRecord],
    ) -> WorkerRecord:
        """在单个角色的候选集中执行 round-robin 或 least-loaded 选择。"""
        if not records:
            raise ValueError(f"No {role.value} worker records to route")

        if self.router_mode == "least-loaded":

            def score(record: WorkerRecord) -> tuple[float, str]:
                # 综合负载分越小越空闲；instance_id 用于稳定平局结果。
                return (
                    self.load_scorer.score(
                        record,
                        self.worker_inflight.count(record.instance_id),
                    ),
                    record.instance_id,
                )

            return min(records, key=score)

        if self.router_mode != "round-robin":
            raise ValueError(f"Unsupported PD router mode: {self.router_mode}")

        # 单调计数取模；worker 扩容后新增副本可自然进入轮询序列。
        cursor = self.cursors[role]
        return records[cursor.increment() % len(records)]

    def begin_request(
        self,
        record: WorkerRecord,
        *,
        count_inflight: bool = True,
    ) -> None:
        """记录角色请求开始，并默认增加对应 worker 的 inflight。

        PD frontend 会在 manager.acquire_workers() 中原子登记 P/D 两个
        worker 的 inflight，因此请求任务内再调用本方法时传
        ``count_inflight=False``，只推进角色请求总量，避免重复计数。
        """
        if count_inflight:
            self.worker_inflight.begin_request(record.instance_id)
        self.role_request_totals[record.worker_type.value] += 1

    def finish_request(self, instance_id: str) -> None:
        """记录请求结束；重复 finish 也不会把 inflight 变成负数。"""
        self.worker_inflight.finish_request(instance_id)
