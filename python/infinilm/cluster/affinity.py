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
        header_value = headers.get("x-infinilm-session-id") or headers.get(
            "x-session-id"
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
    """带 TTL 和容量上限的会话亲和绑定表。

    key 只保存哈希后的会话指纹，value 是绑定的 worker ID。LRU 负责
    限制内存占用，TTL 负责限制绑定有效期。
    """

    def __init__(
        self,
        *,
        ttl_seconds: float,
        capacity: int,
        clock=time.monotonic,
    ) -> None:
        # 0 表示禁用 TTL；不因为时间而过期，只可能因为容量上限被 LRU 淘汰。
        # 容量至少为 1，避免构造出永远无法存活的 store。
        self.ttl_seconds = max(0.0, ttl_seconds)
        self.capacity = max(1, capacity)

        # 注入时钟便于测试 TTL；entries 的 value 记录 worker 和最近更新时间。
        self._clock = clock
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self,
        key: str,
        available_worker_ids: set[str] | frozenset[str],
    ) -> AffinityLookup:
        """查找绑定；过期或 worker 不可路由时删除并视为 miss。"""
        # 没有绑定记录，直接返回 miss。
        entry = self._entries.get(key)
        if entry is None:
            return AffinityLookup(None, hit=False)

        worker_id, updated_at = entry
        # 超过 TTL 的绑定已失效，删除后返回 expired miss。
        if self.ttl_seconds and self._clock() - updated_at > self.ttl_seconds:
            del self._entries[key]
            return AffinityLookup(None, hit=False, expired=True)

        # 原 worker 已退出或不可路由时，绑定不能再复用。
        if worker_id not in available_worker_ids:
            del self._entries[key]
            return AffinityLookup(None, hit=False)

        # move_to_end 把 key 移到 OrderedDict 末尾，表示最近使用。
        # LRU 容量满时从头部删除，因此头部就是最久未用的会话。
        self._entries.move_to_end(key)
        return AffinityLookup(worker_id, hit=True)

    def assign(self, key: str, worker_id: str) -> bool:
        """绑定会话到 worker，并返回是否触发 LRU 淘汰。"""
        # 新绑定或重新绑定都刷新 worker 和时间。
        self._entries[key] = (worker_id, self._clock())
        # 移到末尾，表示这是最近更新的会话。
        self._entries.move_to_end(key)

        # popitem(last=False) 从头部移除最久未使用的会话。
        evicted = len(self._entries) > self.capacity
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return evicted

    def remove_worker(self, worker_id: str) -> None:
        """删除指定 worker 的全部会话绑定。"""
        # 不能在遍历 OrderedDict 时直接删除，先收集再清理。
        stale = [key for key, (owner, _) in self._entries.items() if owner == worker_id]
        for key in stale:
            del self._entries[key]

    def retain_workers(self, worker_ids: set[str] | frozenset[str]) -> None:
        """只保留仍可路由 worker 的绑定，用于 worker 变化后的清理。"""
        # worker 退出、draining 或被移出候选集后，对应绑定全部失效。
        stale = [
            key for key, (owner, _) in self._entries.items() if owner not in worker_ids
        ]
        for key in stale:
            del self._entries[key]
