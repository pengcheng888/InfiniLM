"""Frontend-side index of worker-local prefix caches."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import WorkerRecord


@dataclass
class WorkerIndexState:
    namespace: str
    block_size: int
    hashes: set[str] = field(default_factory=set)
    sequence: int = 0
    snapshot_signature: tuple[int, int] | None = None
    events_offset: int = 0
    events_signature: tuple[int, int] | None = None
    updated_at: float = 0.0


@dataclass(frozen=True)
class WorkerPrefixMatch:
    worker_id: str
    matched_blocks: int
    matched_tokens: int


class KvIndexStore:
    """Reads per-worker snapshots and bounded event logs for routing hints."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerIndexState] = {}

    def __len__(self) -> int:
        return len(self._workers)

    def entries(self) -> int:
        return sum(len(state.hashes) for state in self._workers.values())

    def refresh(self, workers: list[WorkerRecord]) -> None:
        ready_ids = {
            worker.instance_id
            for worker in workers
            if worker.state == "ready" and worker.engine_healthy
        }
        for worker_id in list(self._workers):
            if worker_id not in ready_ids:
                del self._workers[worker_id]

        for worker in workers:
            if worker.instance_id not in ready_ids:
                continue
            metadata = (worker.metadata or {}).get("kv_index")
            if not isinstance(metadata, dict):
                self._workers.pop(worker.instance_id, None)
                continue
            try:
                self._load_worker(worker.instance_id, metadata)
            except (OSError, ValueError, TypeError):
                self._workers.pop(worker.instance_id, None)

    def _load_worker(self, worker_id: str, metadata: dict[str, Any]) -> None:
        namespace = str(metadata["namespace"])
        block_size = int(metadata["block_size"])
        snapshot_path = Path(metadata["snapshot"])
        events_path = Path(metadata["events"])
        snapshot_signature = self._signature(snapshot_path)
        state = self._workers.get(worker_id)

        if state is None or state.snapshot_signature != snapshot_signature:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if payload.get("namespace") != namespace:
                self._workers.pop(worker_id, None)
                return
            hashes = payload.get("hashes", [])
            state = WorkerIndexState(
                namespace=namespace,
                block_size=block_size,
                hashes={str(value) for value in hashes},
                sequence=int(payload.get("sequence", 0)),
                snapshot_signature=snapshot_signature,
                events_offset=0,
                events_signature=None,
            )
            self._workers[worker_id] = state

        state.block_size = block_size
        state.namespace = namespace
        state.updated_at = time.time()

        events_signature = self._signature(events_path)
        if events_signature is None:
            return
        if (
            state.events_signature is not None
            and events_signature[1] < state.events_offset
        ):
            # The worker compacted its log. Reload the authoritative snapshot
            # and reset the tail position.
            self._workers.pop(worker_id, None)
            return

        with events_path.open("rb") as events:
            events.seek(state.events_offset)
            for raw_line in events:
                line = raw_line.decode("utf-8")
                try:
                    event = json.loads(line)
                except ValueError:
                    state.events_offset += len(raw_line)
                    continue
                sequence = int(event.get("sequence", 0))
                if sequence <= state.sequence:
                    state.events_offset += len(raw_line)
                    continue
                event_type = event.get("type")
                block_hash = str(event.get("hash", ""))
                if event_type in {"published", "touched"}:
                    state.hashes.add(block_hash)
                elif event_type == "evicted":
                    state.hashes.discard(block_hash)
                state.sequence = sequence
                state.events_offset += len(raw_line)
        state.events_signature = self._signature(events_path)

    @staticmethod
    def _signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def lookup(
        self,
        prefix: dict[str, Any] | None,
        workers: list[WorkerRecord],
    ) -> list[WorkerPrefixMatch]:
        if not prefix:
            return []

        hashes = list(prefix.get("hashes") or [])
        namespace = str(prefix.get("namespace") or "")
        block_size = int(prefix.get("block_size") or 0)
        available = {worker.instance_id for worker in workers}
        result: list[WorkerPrefixMatch] = []

        for worker_id, state in self._workers.items():
            if worker_id not in available or state.namespace != namespace:
                continue
            if state.block_size != block_size:
                continue
            matched = 0
            for block_hash in hashes:
                if block_hash not in state.hashes:
                    break
                matched += 1
            if matched:
                result.append(
                    WorkerPrefixMatch(
                        worker_id=worker_id,
                        matched_blocks=matched,
                        matched_tokens=matched * block_size,
                    )
                )

        return sorted(
            result,
            key=lambda match: (-match.matched_tokens, match.worker_id),
        )
