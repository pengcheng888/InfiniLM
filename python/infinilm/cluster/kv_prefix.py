"""Frontend-side prefix hashing for KV-aware routing."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infinilm.llm.prefix_cache import hash_block_tokens
from infinilm.multimodal.multimodal import resolve_multimodal_inputs
from infinilm.processors import AutoInfinilmProcessor

_FINGERPRINT_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def model_cache_namespace(model_path: str, block_size: int) -> str:
    """Build a stable cache namespace from model/tokenizer metadata."""
    root = Path(model_path)
    digest = hashlib.sha256()
    digest.update(b"InfiniLM-kv-cache-namespace-v1\0")
    digest.update(str(block_size).encode("utf-8"))
    for name in _FINGERPRINT_FILES:
        digest.update(name.encode("utf-8"))
        path = root / name
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def prefix_hashes(token_ids: list[int], block_size: int) -> list[str]:
    """Return chained hashes for all full token blocks in a prompt."""
    parent = b""
    result: list[str] = []
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        if len(block) != block_size:
            break
        parent = hash_block_tokens(block, parent)
        result.append(parent.hex())
    return result


class PrefixHashCalculator:
    """Caches model processors and hashes text chat prompts."""

    def __init__(self) -> None:
        self._processors: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _processor(self, model_path: str):
        with self._lock:
            processor = self._processors.get(model_path)
            if processor is None:
                processor = AutoInfinilmProcessor.from_pretrained(model_path)
                self._processors[model_path] = processor
            return processor

    @staticmethod
    def _input_ids(processed: Any) -> list[int]:
        if isinstance(processed, Mapping):
            value = processed.get("input_ids")
        else:
            try:
                value = processed["input_ids"]
            except (KeyError, TypeError):
                return []

        flatten = getattr(value, "flatten", None)
        if callable(flatten):
            value = flatten()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        if not isinstance(value, list):
            return []
        if value and isinstance(value[0], list):
            value = value[0]
        return [int(token_id) for token_id in value]

    def calculate(
        self,
        payload: dict[str, Any],
        *,
        model_path: str,
        block_size: int,
    ) -> dict[str, Any] | None:
        """Hash a text-only chat request; return None when unsupported."""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None

        try:
            multimodal = resolve_multimodal_inputs(messages)
            if any(multimodal[name] for name in ("images", "videos", "audios")):
                return None

            processor = self._processor(model_path)
            prompt = processor.apply_chat_template(
                conversation=messages,
                add_generation_prompt=bool(
                    payload.get("add_generation_prompt", True)
                ),
                tokenize=False,
                **(payload.get("chat_template_kwargs") or {}),
            )
            token_ids = self._input_ids(
                processor(prompt, return_tensors="pt")
            )
        except Exception:
            # Unsupported processor/template combinations still route through
            # the existing least-loaded or affinity policies.
            return None

        return {
            "namespace": model_cache_namespace(model_path, block_size),
            "hashes": prefix_hashes(token_ids, block_size),
            "token_count": len(token_ids),
            "block_size": block_size,
        }
