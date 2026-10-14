"""Snapshot and event-log export for a worker's local prefix cache."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any


class WorkerCacheIndexExporter:
    """Exports a BlockManager hash index without blocking the scheduler."""

    def __init__(
        self,
        *,
        block_manager: Any,
        output_dir: Path,
        namespace: str,
        block_size: int,
        max_events: int = 10000,
    ) -> None:
        self.block_manager = block_manager
        self.output_dir = Path(output_dir)
        self.namespace = namespace
        self.block_size = block_size
        self.max_events = max_events
        self.snapshot_path = self.output_dir / "prefix-cache.snapshot.json"
        self.events_path = self.output_dir / "prefix-cache.events.jsonl"
        self._events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._lock = threading.RLock()
        self._sequence = 0
        self._event_count = 0
        self._writer: threading.Thread | None = None
        self._running = False
        self._observer = self._enqueue

    def start(self) -> None:
        if self._running:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self.block_manager.add_index_observer(self._observer)
        self._writer = threading.Thread(
            target=self._write_loop,
            name=f"infinilm-cache-index-{self.namespace[:8]}",
            daemon=True,
        )
        self._writer.start()
        self.write_snapshot()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.block_manager.remove_index_observer(self._observer)
        self._events.put(None)
        if self._writer is not None:
            self._writer.join(timeout=2)
            self._writer = None
        self.write_snapshot()

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "namespace": self.namespace,
                "block_size": self.block_size,
                "snapshot": str(self.snapshot_path),
                "events": str(self.events_path),
                "sequence": self._sequence,
                "entries": len(self.block_manager.snapshot_prefix_index()),
            }

    def write_snapshot(self) -> dict[str, Any]:
        hashes = self.block_manager.snapshot_prefix_index()
        with self._lock:
            sequence = self._sequence
            payload = {
                "schema_version": 1,
                "namespace": self.namespace,
                "block_size": self.block_size,
                "sequence": sequence,
                "generated_at": time.time(),
                "hashes": hashes,
            }
            temporary = self.snapshot_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self.snapshot_path)
            return payload

    def _enqueue(self, event: dict[str, Any]) -> None:
        self._events.put({**event, "timestamp": time.time()})

    def _write_loop(self) -> None:
        while True:
            event = self._events.get()
            if event is None:
                return
            try:
                self._append_event(event)
            except Exception:
                # Cache routing remains advisory; worker inference continues if
                # index export encounters filesystem pressure.
                continue

    def _append_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            self._event_count += 1
            sequence = self._sequence
            line = json.dumps(
                {"sequence": sequence, **event},
                separators=(",", ":"),
                sort_keys=True,
            )
            with self.events_path.open("a", encoding="utf-8") as output:
                output.write(line + "\n")
                output.flush()
                os.fsync(output.fileno())

            if self._event_count < self.max_events:
                return
            # Compact after the bounded event budget. The next snapshot is the
            # authoritative state; readers reset when the log becomes shorter.
            self.write_snapshot()
            temporary = self.events_path.with_suffix(".tmp")
            temporary.write_text("", encoding="utf-8")
            os.replace(temporary, self.events_path)
            self._event_count = 0
