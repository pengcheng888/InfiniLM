"""Low-overhead timing for the streaming output path.

Enable collection with ``INFINILM_OUTPUT_TIMING=true``. Token traces are
sampled with ``INFINILM_OUTPUT_TRACE_SAMPLE_RATE`` and event-loop wake latency
with ``INFINILM_EVENT_LOOP_LAG_INTERVAL_MS``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Optional

DEFAULT_OUTPUT_TIMING_DIR = Path("/tmp/infinilm-output-timing")
DEFAULT_EVENT_LOOP_LAG_INTERVAL_MS = 10.0
DEFAULT_OUTPUT_TRACE_SAMPLE_RATE = 0.01
MAX_OUTPUT_TRACE_SAMPLES = 32
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def _truthy_environment(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OutputTimingConfig:
    enabled: bool = False
    root: Path = DEFAULT_OUTPUT_TIMING_DIR
    trace_sample_rate: float = DEFAULT_OUTPUT_TRACE_SAMPLE_RATE
    event_loop_lag_interval_ms: float = DEFAULT_EVENT_LOOP_LAG_INTERVAL_MS

    @classmethod
    def from_environment(cls) -> "OutputTimingConfig":
        raw_rate = os.environ.get(
            "INFINILM_OUTPUT_TRACE_SAMPLE_RATE",
            str(DEFAULT_OUTPUT_TRACE_SAMPLE_RATE),
        )
        raw_interval = os.environ.get(
            "INFINILM_EVENT_LOOP_LAG_INTERVAL_MS",
            str(DEFAULT_EVENT_LOOP_LAG_INTERVAL_MS),
        )
        try:
            trace_sample_rate = float(raw_rate)
        except ValueError as exc:
            raise RuntimeError(
                "INFINILM_OUTPUT_TRACE_SAMPLE_RATE must be a float"
            ) from exc
        if not 0.0 <= trace_sample_rate <= 1.0:
            raise RuntimeError(
                "INFINILM_OUTPUT_TRACE_SAMPLE_RATE must be between 0 and 1"
            )

        try:
            event_loop_lag_interval_ms = float(raw_interval)
        except ValueError as exc:
            raise RuntimeError(
                "INFINILM_EVENT_LOOP_LAG_INTERVAL_MS must be a float"
            ) from exc
        if not 0.5 <= event_loop_lag_interval_ms <= 1000.0:
            raise RuntimeError(
                "INFINILM_EVENT_LOOP_LAG_INTERVAL_MS must be between 0.5 and 1000"
            )

        return cls(
            enabled=_truthy_environment(os.environ.get("INFINILM_OUTPUT_TIMING")),
            root=Path(
                os.environ.get("INFINILM_OUTPUT_TIMING_DIR", DEFAULT_OUTPUT_TIMING_DIR)
            ),
            trace_sample_rate=trace_sample_rate,
            event_loop_lag_interval_ms=event_loop_lag_interval_ms,
        )


@dataclass
class OutputTokenTiming:
    """Timestamps carried by one token from the engine thread to HTTP."""

    engine_sampled_ready_at_ns: Optional[int] = None
    engine_decode_started_at_ns: Optional[int] = None
    engine_decode_completed_at_ns: Optional[int] = None
    engine_token_ready_at_ns: Optional[int] = None
    loop_callback_scheduled_at_ns: Optional[int] = None
    loop_callback_started_at_ns: Optional[int] = None
    queue_put_started_at_ns: Optional[int] = None
    queue_put_completed_at_ns: Optional[int] = None
    queue_get_returned_at_ns: Optional[int] = None
    timer_cancel_started_at_ns: Optional[int] = None
    timer_cancel_completed_at_ns: Optional[int] = None
    queue_task_done_started_at_ns: Optional[int] = None
    queue_task_done_completed_at_ns: Optional[int] = None
    route_received_at_ns: Optional[int] = None


@dataclass
class DurationAccumulator:
    count: int = 0
    total_ns: int = 0
    min_ns: Optional[int] = None
    max_ns: Optional[int] = None

    def add(self, duration_ns: int) -> None:
        if duration_ns < 0:
            return
        self.count += 1
        self.total_ns += duration_ns
        self.min_ns = (
            duration_ns if self.min_ns is None else min(self.min_ns, duration_ns)
        )
        self.max_ns = (
            duration_ns if self.max_ns is None else max(self.max_ns, duration_ns)
        )

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "total_ms": self.total_ns / 1_000_000,
            "avg_ms": self.total_ns / self.count / 1_000_000 if self.count else None,
            "min_ms": self.min_ns / 1_000_000 if self.min_ns is not None else None,
            "max_ms": self.max_ns / 1_000_000 if self.max_ns is not None else None,
        }


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


@dataclass
class EventLoopLagMonitor:
    interval_ms: float = DEFAULT_EVENT_LOOP_LAG_INTERVAL_MS
    enabled: bool = True
    _task: Optional[asyncio.Task] = None
    _stop_event: Optional[asyncio.Event] = None
    _values: Deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=100_000)
    )

    async def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="infinilm-loop-lag-monitor")

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            try:
                await self._task
            finally:
                self._task = None
                self._stop_event = None

    async def _run(self) -> None:
        assert self._stop_event is not None
        loop = asyncio.get_running_loop()
        interval = self.interval_ms / 1000.0
        while not self._stop_event.is_set():
            target = loop.time() + interval
            await asyncio.sleep(interval)
            self._values.append((loop.time(), max(0.0, loop.time() - target)))

    def stats(self, since_monotonic: float | None = None) -> dict[str, Any]:
        values = [
            value
            for timestamp, value in self._values
            if since_monotonic is None or timestamp >= since_monotonic
        ]
        return {
            "count": len(values),
            "avg_ms": sum(values) * 1000.0 / len(values) if values else None,
            "p50_ms": _quantile(values, 0.50) * 1000.0 if values else None,
            "p95_ms": _quantile(values, 0.95) * 1000.0 if values else None,
            "p99_ms": _quantile(values, 0.99) * 1000.0 if values else None,
            "max_ms": max(values) * 1000.0 if values else None,
        }


class OutputTimingCollector:
    """Aggregate one request without writing in the per-token hot path."""

    def __init__(
        self,
        request_id: str,
        config: OutputTimingConfig,
        monitor: EventLoopLagMonitor,
    ) -> None:
        self.request_id = request_id
        self.config = config
        self.monitor = monitor
        self.started_at_ns = time.monotonic_ns()
        self.started_monotonic = time.monotonic()
        self.finished_at_ns: Optional[int] = None
        self.token_count = 0
        self.sse_frame_count = 0
        self.sse_body_bytes = 0
        self.asgi_body_message_count = 0
        self.asgi_body_bytes = 0
        self.disconnect_check_count = 0
        self._durations: dict[str, DurationAccumulator] = {}
        self._trace: list[dict[str, Any]] = []
        self._random = random.Random(request_id)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _add(self, name: str, start: int, end: int) -> None:
        if start >= 0 and end >= start:
            accumulator = self._durations.get(name)
            if accumulator is None:
                accumulator = DurationAccumulator()
                self._durations[name] = accumulator
            accumulator.add(end - start)

    def record_token(self, timing: OutputTokenTiming, token_id: int) -> None:
        index = self.token_count
        self.token_count += 1
        self._add(
            "engine_sampled_to_token_ready",
            timing.engine_sampled_ready_at_ns or -1,
            timing.engine_token_ready_at_ns or -1,
        )
        self._add(
            "engine_decode",
            timing.engine_decode_started_at_ns or -1,
            timing.engine_decode_completed_at_ns or -1,
        )
        self._add(
            "engine_token_ready_to_loop_callback_scheduled",
            timing.engine_token_ready_at_ns or -1,
            timing.loop_callback_scheduled_at_ns or -1,
        )
        self._add(
            "loop_callback_wait",
            timing.loop_callback_scheduled_at_ns or -1,
            timing.loop_callback_started_at_ns or -1,
        )
        self._add(
            "loop_callback_to_queue_put",
            timing.loop_callback_started_at_ns or -1,
            timing.queue_put_started_at_ns or -1,
        )
        self._add(
            "async_queue_put",
            timing.queue_put_started_at_ns or -1,
            timing.queue_put_completed_at_ns or -1,
        )
        self._add(
            "queue_put_to_queue_get",
            timing.queue_put_completed_at_ns or -1,
            timing.queue_get_returned_at_ns or -1,
        )
        self._add(
            "timer_cancel",
            timing.timer_cancel_started_at_ns or -1,
            timing.timer_cancel_completed_at_ns or -1,
        )
        self._add(
            "queue_task_done",
            timing.queue_task_done_started_at_ns or -1,
            timing.queue_task_done_completed_at_ns or -1,
        )
        self._add(
            "queue_get_to_route_received",
            timing.queue_get_returned_at_ns or -1,
            timing.route_received_at_ns or -1,
        )

        should_trace = (
            index == 0
            or len(self._trace) < MAX_OUTPUT_TRACE_SAMPLES
            and self._random.random() < self.config.trace_sample_rate
        )
        if should_trace:
            self._trace.append(
                {
                    "token_index": index,
                    "token_id": token_id,
                    "timestamps_ns": {
                        key: getattr(timing, key) for key in timing.__dataclass_fields__
                    },
                }
            )

    def record_disconnect(self, start: int, end: int) -> None:
        self.disconnect_check_count += 1
        self._add("disconnect_check", start, end)

    def record_sse_json(self, start: int, end: int, body_bytes: int) -> None:
        self._add("sse_json", start, end)
        self.sse_frame_count += 1
        self.sse_body_bytes += body_bytes

    def record_route_yield(self, start: int, end: int, body_bytes: int) -> None:
        self._add("route_yield_to_resumed", start, end)

    def record_asgi_body_send(
        self, start: int, end: int, body_bytes: int, message_type: str
    ) -> None:
        if message_type != "http.response.body":
            return
        self.asgi_body_message_count += 1
        self.asgi_body_bytes += body_bytes
        self._add("asgi_body_send", start, end)

    def finish(self) -> dict[str, Any]:
        if self.finished_at_ns is not None:
            return {}
        self.finished_at_ns = time.monotonic_ns()
        payload = {
            "schema_version": 1,
            "event": "output_timing_finish",
            "request_id": self.request_id,
            "duration_ms": (self.finished_at_ns - self.started_at_ns) / 1_000_000,
            "token_count": self.token_count,
            "sse_frame_count": self.sse_frame_count,
            "sse_body_bytes": self.sse_body_bytes,
            "asgi_body_message_count": self.asgi_body_message_count,
            "asgi_body_bytes": self.asgi_body_bytes,
            "route_yield_body_bytes": self.sse_body_bytes,
            "disconnect_check_count": self.disconnect_check_count,
            "event_loop_lag": self.monitor.stats(self.started_monotonic),
            "durations_ms": {
                name: accumulator.as_dict()
                for name, accumulator in self._durations.items()
            },
            "token_trace": self._trace,
        }
        path = self.config.root / str(os.getpid()) / f"{self._safe_request_id()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        return payload

    def _safe_request_id(self) -> str:
        safe = _UNSAFE_FILENAME_CHARS.sub("_", self.request_id)[:80]
        if safe != self.request_id or not safe:
            digest = hashlib.sha256(self.request_id.encode()).hexdigest()[:12]
            safe = f"{safe}-{digest}"
        return safe


class DisabledOutputTimingCollector:
    """No-op collector used when output timing is disabled."""

    request_id: str
    enabled = False

    def record_token(self, timing: OutputTokenTiming, token_id: int) -> None:
        return

    def record_disconnect(self, start: int, end: int) -> None:
        return

    def record_sse_json(self, start: int, end: int, body_bytes: int) -> None:
        return

    def record_route_yield(self, start: int, end: int, body_bytes: int) -> None:
        return

    def record_asgi_body_send(
        self, start: int, end: int, body_bytes: int, message_type: str
    ) -> None:
        return

    def finish(self) -> dict[str, Any]:
        return {}


class OutputTimingService:
    def __init__(self, config: OutputTimingConfig | None = None) -> None:
        self.config = config or OutputTimingConfig.from_environment()
        self.monitor = EventLoopLagMonitor(
            interval_ms=self.config.event_loop_lag_interval_ms,
            enabled=self.config.enabled,
        )
        self._request_summaries: Deque[dict[str, Any]] = deque(maxlen=10_000)

    async def start(self) -> None:
        await self.monitor.start()

    async def stop(self) -> None:
        await self.monitor.stop()
        if not self.config.enabled:
            return
        self._write_process_summary()

    def create_collector(self, request_id: str) -> OutputTimingCollector:
        if not self.config.enabled:
            return DisabledOutputTimingCollector()  # type: ignore[return-value]
        return OutputTimingCollector(request_id, self.config, self.monitor)

    def record_finished_collector(self, payload: dict[str, Any]) -> None:
        if payload:
            self._request_summaries.append(payload)

    def _write_process_summary(self) -> None:
        token_counts = [item.get("token_count", 0) for item in self._request_summaries]
        duration_names = {
            name
            for item in self._request_summaries
            for name in item.get("durations_ms", {})
        }
        process_durations: dict[str, dict[str, Any]] = {}
        for name in duration_names:
            values = [
                item["durations_ms"][name]
                for item in self._request_summaries
                if name in item.get("durations_ms", {})
            ]
            total_count = sum(value.get("count", 0) for value in values)
            total_ns = sum(value.get("total_ms", 0.0) * 1_000_000 for value in values)
            avg_ms = total_ns / total_count / 1_000_000 if total_count else None
            max_ms = max(
                (
                    value.get("max_ms")
                    for value in values
                    if value.get("max_ms") is not None
                ),
                default=None,
            )
            process_durations[name] = {
                "request_count": len(values),
                "count": total_count,
                "avg_ms": avg_ms,
                "max_ms": max_ms,
            }
        payload = {
            "schema_version": 1,
            "event": "output_timing_process_summary",
            "pid": os.getpid(),
            "request_count": len(self._request_summaries),
            "token_count": sum(token_counts),
            "event_loop_lag": self.monitor.stats(),
            "durations_ms": process_durations,
        }
        path = self.config.root / str(os.getpid()) / "process_summary.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
