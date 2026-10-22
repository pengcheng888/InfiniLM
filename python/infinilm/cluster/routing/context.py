"""Inputs and outputs for routing policy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..worker_registry import WorkerRecord


@dataclass(frozen=True)
class RoutingContext:
    candidates: list[WorkerRecord]
    discovery_workers: list[WorkerRecord]
    routing_key: str | None = None
    kv_prefix: dict[str, Any] | None = None
    attempt_index: int = 0
    count_metrics: bool = True


@dataclass(frozen=True)
class SelectionResult:
    worker: WorkerRecord
    reason: str
    affinity_keyed: bool | None = None
    affinity_hit: bool | None = None
    affinity_expired: bool = False
    affinity_evicted: bool = False
    matched_tokens: int | None = None
