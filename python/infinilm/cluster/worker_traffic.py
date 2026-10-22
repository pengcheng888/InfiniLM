"""Worker-side request counters independent of frontend routing metrics."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class WorkerTrafficMetrics:
    """Counts every request observed by this worker HTTP process.

    These counters include traffic routed by a frontend and traffic sent directly
    to the worker endpoint, unlike AggregatedRouter metrics.
    """

    received_requests_total: int = 0
    streaming_requests_total: int = 0
    non_streaming_requests_total: int = 0
    successful_requests_total: int = 0
    failed_requests_total: int = 0
    canceled_requests_total: int = 0
    active_requests: int = 0
    local_cache_hit_requests_total: int = 0
    local_cache_miss_requests_total: int = 0
    local_cache_hit_tokens_total: int = 0
    local_prefill_tokens_total: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def begin(self, *, streaming: bool) -> None:
        with self._lock:
            self.received_requests_total += 1
            self.active_requests += 1
            if streaming:
                self.streaming_requests_total += 1
            else:
                self.non_streaming_requests_total += 1

    def finish(self, *, canceled: bool = False, failed: bool = False) -> None:
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            if canceled:
                self.canceled_requests_total += 1
            elif failed:
                self.failed_requests_total += 1
            else:
                self.successful_requests_total += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "received_requests_total": self.received_requests_total,
                "streaming_requests_total": self.streaming_requests_total,
                "non_streaming_requests_total": self.non_streaming_requests_total,
                "successful_requests_total": self.successful_requests_total,
                "failed_requests_total": self.failed_requests_total,
                "canceled_requests_total": self.canceled_requests_total,
                "active_requests": self.active_requests,
                "local_cache_hit_requests_total": self.local_cache_hit_requests_total,
                "local_cache_miss_requests_total": self.local_cache_miss_requests_total,
                "local_cache_hit_tokens_total": self.local_cache_hit_tokens_total,
                "local_prefill_tokens_total": self.local_prefill_tokens_total,
            }

    def record_cache(self, cache: dict | None) -> None:
        if not isinstance(cache, dict):
            return
        matched = max(0, int(cache.get("matched_tokens", 0)))
        prefill = max(0, int(cache.get("prefill_tokens", 0)))
        with self._lock:
            if matched:
                self.local_cache_hit_requests_total += 1
            else:
                self.local_cache_miss_requests_total += 1
            self.local_cache_hit_tokens_total += matched
            self.local_prefill_tokens_total += prefill
