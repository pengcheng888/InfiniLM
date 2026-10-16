"""Shared data model for service discovery records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkerRecord:
    namespace: str
    component: str
    endpoint: str
    instance_id: str
    worker_type: str
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
    transport: str = "http"
    # ready 表示可接收新请求；draining 表示存量请求可继续，但不再接收新请求。
    state: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)

    def discovery_key(self) -> str:
        return ".".join((self.namespace, self.component, self.endpoint))
