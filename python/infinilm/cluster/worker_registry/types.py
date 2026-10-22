"""Shared data model for service discovery records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class WorkerType(Enum):
    """InfiniLM cluster worker 的封闭角色集合。

    内部逻辑只使用枚举成员；JSON、CLI 和监控输出在边界处转换为 `.value`。
    """

    AGGREGATED = "aggregated"
    PREFILL = "prefill"
    DECODE = "decode"


def coerce_worker_type(value: WorkerType | str) -> WorkerType:
    """把外部字符串角色转换为枚举，供 registry/测试入口复用。"""
    return value if isinstance(value, WorkerType) else WorkerType(value)


def normalize_worker_types(
    values: Iterable[WorkerType | str] | None,
) -> frozenset[WorkerType]:
    """归一化 snapshot/watcher 的角色过滤参数。"""
    if values is None:
        return frozenset({WorkerType.AGGREGATED})
    return frozenset(coerce_worker_type(value) for value in values)


@dataclass(frozen=True)
class WorkerRecord:
    namespace: str
    component: str
    endpoint: str
    instance_id: str
    worker_type: WorkerType
    model: str  # worker 本地加载的模型路径
    served_model_name: str  # 对外暴露的 model name
    host: str
    port: int
    # worker 实例首次创建注册记录的时间。该字段用于观测和排查，例如判断
    # worker 何时加入集群；heartbeat 时不会更新，也不参与过期判断。
    registered_at: float

    # 最近一次 heartbeat 的时间。worker 正常运行时会周期性刷新该时间。
    # 它用于人工观察最近心跳，过期判断使用的是 lease_expires_at。
    heartbeat_at: float

    # 当前注册租约的截止时间，由 heartbeat_at + lease_seconds 得到。
    # frontend snapshot 时如果发现 lease_expires_at < 当前时间，就认为
    # worker 已停止 heartbeat，该注册记录过期且不可路由。
    lease_expires_at: float
    engine_healthy: bool = True
    # ready 表示可接收新请求；draining 表示存量请求可继续，但不再接收新请求。
    state: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """确保记录进入内部逻辑后 worker_type 一定是枚举。"""
        object.__setattr__(self, "worker_type", coerce_worker_type(self.worker_type))

    def discovery_key(self) -> str:
        return ".".join((self.namespace, self.component, self.endpoint))


def worker_record_from_payload(payload: dict[str, Any]) -> WorkerRecord:
    """从 registry 存储 payload 解析 worker 记录，并兼容历史 transport 字段。"""
    data = dict(payload)
    if data.pop("transport", "http") != "http":
        raise ValueError("only HTTP worker records are supported")
    data["worker_type"] = coerce_worker_type(data.get("worker_type"))
    return WorkerRecord(**data)


def worker_record_to_payload(record: WorkerRecord) -> dict[str, Any]:
    """把内部记录转换为可 JSON 序列化的 registry payload。"""
    payload = asdict(record)
    payload["worker_type"] = record.worker_type.value
    return payload
