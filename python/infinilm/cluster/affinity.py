"""Session and conversation affinity support for ClusterRouter."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

ROUTING_KEY_FIELD = "__infinilm_routing_key"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                value = part.get("text", "")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return ""


def _first_message(messages: list[Any], role: str) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            return _content_text(message.get("content"))
    return ""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def affinity_key(
    payload: Mapping[str, Any],
    headers: Mapping[str, Any] | None = None,
) -> str | None:
    """Build a privacy-preserving stable routing key for a chat request."""
    header_value = None
    if headers is not None:
        header_value = (
            headers.get("x-infinilm-session-id")
            or headers.get("x-session-id")
        )
    if not header_value:
        header_value = payload.get(ROUTING_KEY_FIELD)
    if not header_value and payload.get("user") is not None:
        header_value = str(payload["user"])

    if header_value:
        return _digest(f"identity:{header_value}")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    # Multi-turn conversations keep their first user turn. Hashing the stable
    # prefix lets later turns reuse the same worker and its local prefix cache.
    prefix = "\n---turn---\n".join(
        (
            str(payload.get("model") or ""),
            _first_message(messages, "system"),
            _first_message(messages, "user"),
        )
    )
    if not prefix.strip("\n---turn---\n"):
        return None
    return _digest(f"conversation:{prefix}")


@dataclass(frozen=True)
class AffinityLookup:
    worker_id: str | None
    hit: bool
    expired: bool = False


class AffinityStore:
    """Bounded LRU mapping from hashed affinity keys to worker IDs."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        capacity: int,
        clock=time.monotonic,
    ) -> None:
        self.ttl_seconds = max(0.0, ttl_seconds)
        self.capacity = max(1, capacity)
        self._clock = clock
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self,
        key: str,
        available_worker_ids: set[str] | frozenset[str],
    ) -> AffinityLookup:
        entry = self._entries.get(key)
        if entry is None:
            return AffinityLookup(None, hit=False)

        worker_id, updated_at = entry
        if self.ttl_seconds and self._clock() - updated_at > self.ttl_seconds:
            del self._entries[key]
            return AffinityLookup(None, hit=False, expired=True)

        if worker_id not in available_worker_ids:
            del self._entries[key]
            return AffinityLookup(None, hit=False)

        self._entries.move_to_end(key)
        return AffinityLookup(worker_id, hit=True)

    def assign(self, key: str, worker_id: str) -> bool:
        """Assign a key and report whether an older entry was evicted."""
        self._entries[key] = (worker_id, self._clock())
        self._entries.move_to_end(key)
        evicted = len(self._entries) > self.capacity
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return evicted

    def remove_worker(self, worker_id: str) -> None:
        stale = [
            key
            for key, (owner, _) in self._entries.items()
            if owner == worker_id
        ]
        for key in stale:
            del self._entries[key]

    def retain_workers(self, worker_ids: set[str] | frozenset[str]) -> None:
        stale = [
            key
            for key, (owner, _) in self._entries.items()
            if owner not in worker_ids
        ]
        for key in stale:
            del self._entries[key]
