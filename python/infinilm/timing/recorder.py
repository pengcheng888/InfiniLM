"""Bounded, asynchronous JSONL recorder for timing summary events."""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

from infinilm.timing.config import (
    TIMING_DIR,
    TIMING_MODE,
    TIMING_SAMPLE_INTERVAL,
    timing_enabled_for,
)

SCHEMA_VERSION = 6
MAX_PENDING_RECORDS = 1024
MAX_WRITE_BATCH_RECORDS = 128


class DisabledTimingRecorder:
    enabled = False

    def record_batch(self, payload: dict[str, Any]) -> None:
        return

    def record_token(self, payload: dict[str, Any]) -> None:
        return

    def record_finish(self, payload: dict[str, Any]) -> None:
        return

    def record_event(self, event: str, payload: dict[str, Any]) -> None:
        return

    def should_record_token(self, token_index: int, finished: bool) -> bool:
        return False

    def stats(self) -> dict[str, int]:
        return {"dropped_records": 0, "pending_records": 0}

    def close(self) -> None:
        return


_DISABLED_TIMING_RECORDER = DisabledTimingRecorder()


class TimingRecorder:
    """Write timing events from a single background thread."""

    def __init__(
        self,
        *,
        root: Path | str = TIMING_DIR,
        sample_interval: int = TIMING_SAMPLE_INTERVAL,
    ) -> None:
        if TIMING_MODE != "summary":
            raise ValueError("TIMING_MODE must be 'summary'")
        if sample_interval < 1:
            raise ValueError("TIMING_SAMPLE_INTERVAL must be at least 1")

        self.enabled = True
        self._root = Path(root)
        self._sample_interval = sample_interval
        self._path = self._root / str(os.getpid()) / "timing.jsonl"
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=MAX_PENDING_RECORDS
        )
        self._lock = threading.Lock()
        self._closed = False
        self._stop_event = threading.Event()
        self._dropped_records = 0
        self._thread: threading.Thread | None = None

    @property
    def sample_interval(self) -> int:
        return self._sample_interval

    def should_record_token(self, token_index: int, finished: bool) -> bool:
        return (
            token_index == 0
            or finished
            or (token_index + 1) % self._sample_interval == 0
        )

    def record_batch(self, payload: dict[str, Any]) -> None:
        self._enqueue("batch", payload)

    def record_token(self, payload: dict[str, Any]) -> None:
        self._enqueue("token", payload)

    def record_finish(self, payload: dict[str, Any]) -> None:
        self._enqueue("finish", payload)

    def record_event(self, event: str, payload: dict[str, Any]) -> None:
        self._enqueue(event, payload)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "dropped_records": self._dropped_records,
                "pending_records": self._queue.qsize(),
            }

    def _enqueue(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._write_loop,
                    name="infinilm-timing-recorder",
                    daemon=True,
                )
                self._thread.start()

            record = {
                "schema_version": SCHEMA_VERSION,
                "event": event,
                "pid": os.getpid(),
                **payload,
            }
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped_records += 1

    def _write_loop(self) -> None:
        while True:
            try:
                record = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            try:
                records = [record]
                while len(records) < MAX_WRITE_BATCH_RECORDS:
                    try:
                        records.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                self._write_records(records)
            except (OSError, TypeError, ValueError) as exc:
                # Timing is diagnostic-only; I/O failures must not affect serving.
                with self._lock:
                    self._dropped_records += len(records)
                print(f"InfiniLM timing write failed: {exc}", flush=True)

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ]
        with self._path.open("a", encoding="utf-8") as output_file:
            output_file.write("".join(lines))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            self._stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)


def create_timing_recorder(
    cache_type: str = "paged",
) -> TimingRecorder | DisabledTimingRecorder:
    if not timing_enabled_for(cache_type):
        return _DISABLED_TIMING_RECORDER

    return TimingRecorder(
        root=TIMING_DIR,
        sample_interval=TIMING_SAMPLE_INTERVAL,
    )
