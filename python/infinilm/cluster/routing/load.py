"""Worker load parsing and scoring used by routing policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..worker_registry import WorkerRecord


@dataclass(frozen=True)
class WorkerLoad:
    waiting_requests: int = 0
    running_requests: int = 0
    max_batch_size: int = 1
    kv_usage: float = 0.0


@dataclass(frozen=True)
class LoadScoreWeights:
    inflight: float = 0.90
    waiting_requests: float = 0.05
    running_ratio: float = 0.03
    kv_usage: float = 0.02


def _non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        denominator_value = float(denominator)
        if denominator_value <= 0:
            return 0.0
        return min(1.0, max(0.0, float(numerator) / denominator_value))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def worker_load(worker: WorkerRecord) -> WorkerLoad:
    """Parse the best-effort load snapshot published by a worker."""
    raw = worker.metadata.get("load")
    if not isinstance(raw, dict):
        return WorkerLoad()

    max_batch_size = max(1, _non_negative_int(raw.get("max_batch_size"), 1))
    return WorkerLoad(
        waiting_requests=_non_negative_int(raw.get("waiting_requests")),
        running_requests=_non_negative_int(raw.get("running_requests")),
        max_batch_size=max_batch_size,
        kv_usage=_ratio(
            raw.get("kv_used", 0),
            raw.get("kv_capacity", raw.get("kv_used", 0) + 1),
        ),
    )


class WorkerLoadScorer:
    """Calculates the composite score used by load-aware routing."""

    def __init__(self, weights: LoadScoreWeights = LoadScoreWeights()) -> None:
        self.weights = weights

    def score(self, worker: WorkerRecord, inflight: int = 0) -> float:
        load = worker_load(worker)
        running_ratio = load.running_requests / load.max_batch_size
        return (
            self.weights.inflight * max(0, inflight)
            + self.weights.waiting_requests * load.waiting_requests
            + self.weights.running_ratio * running_ratio
            + self.weights.kv_usage * load.kv_usage
        )
