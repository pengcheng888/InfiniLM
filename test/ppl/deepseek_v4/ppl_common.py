#!/usr/bin/env python3
"""Deterministic token-manifest and sliding-window helpers for PPL."""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


CORPUS_SCHEMA = "deepseek_v4_ppl_tokens/v1"
SCORING_METHOD = "sliding_window_shifted_cross_entropy_fp32_fp64_sum"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def sequence_sha256(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"token_ids[{index}] must be a non-negative integer")
        try:
            parsed = operator.index(value)
        except TypeError as error:
            raise ValueError(
                f"token_ids[{index}] must be a non-negative integer"
            ) from error
        if parsed < 0:
            raise ValueError(f"token_ids[{index}] must be non-negative")
        if index:
            digest.update(b",")
        digest.update(str(parsed).encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


@dataclass(frozen=True)
class Corpus:
    path: Path
    payload: dict[str, Any]
    token_ids: tuple[int, ...]
    manifest_sha256: str
    token_ids_sha256: str


@dataclass(frozen=True)
class Window:
    index: int
    token_start: int
    token_end: int
    score_start: int
    score_end: int
    token_ids: tuple[int, ...]

    @property
    def prediction_start(self) -> int:
        return self.score_start - self.token_start - 1

    @property
    def scored_token_count(self) -> int:
        return self.score_end - self.score_start


def build_manifest(
    token_ids: Sequence[int],
    *,
    source: str,
    source_sha256: str,
    tokenizer: str,
) -> dict[str, Any]:
    values = [int(value) for value in token_ids]
    if len(values) < 2:
        raise ValueError("at least two tokens are required for PPL")
    payload = {
        "schema": CORPUS_SCHEMA,
        "source": source,
        "source_sha256": source_sha256,
        "tokenizer": tokenizer,
        "token_count": len(values),
        "token_ids_sha256": sequence_sha256(values),
        "token_ids": values,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def load_manifest(path: str | Path) -> Corpus:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"invalid PPL manifest schema: {manifest_path}")
    expected_manifest_hash = str(payload.get("manifest_sha256", ""))
    semantic_payload = dict(payload)
    semantic_payload.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(
        canonical_json_bytes(semantic_payload)
    ).hexdigest()
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(f"manifest SHA256 mismatch: {manifest_path}")
    raw_ids = payload.get("token_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("manifest token_ids must be a JSON array")
    token_ids = tuple(int(value) for value in raw_ids)
    token_hash = sequence_sha256(token_ids)
    if token_hash != payload.get("token_ids_sha256"):
        raise ValueError("token_ids SHA256 mismatch")
    if len(token_ids) != payload.get("token_count") or len(token_ids) < 2:
        raise ValueError("manifest token_count is invalid")
    return Corpus(
        path=manifest_path,
        payload=payload,
        token_ids=token_ids,
        manifest_sha256=expected_manifest_hash,
        token_ids_sha256=token_hash,
    )


def iter_windows(
    token_ids: Sequence[int],
    window_size: int,
    stride: int,
    max_scored_tokens: int | None,
) -> Iterator[Window]:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if stride < 1 or stride >= window_size:
        raise ValueError("stride must satisfy 1 <= stride < window_size")
    if len(token_ids) < 2:
        raise ValueError("at least two tokens are required")
    if max_scored_tokens is not None and max_scored_tokens < 1:
        raise ValueError("max_scored_tokens must be positive or None")

    score_limit = len(token_ids)
    if max_scored_tokens is not None:
        score_limit = min(score_limit, 1 + max_scored_tokens)
    previous_end = 1
    index = 0
    while previous_end < score_limit:
        if index == 0:
            token_start = 0
            token_end = min(score_limit, window_size)
            score_start = 1
        else:
            token_end = min(score_limit, previous_end + stride)
            token_start = max(0, token_end - window_size)
            score_start = previous_end
        window = Window(
            index=index,
            token_start=token_start,
            token_end=token_end,
            score_start=score_start,
            score_end=token_end,
            token_ids=tuple(token_ids[token_start:token_end]),
        )
        if window.prediction_start < 0:
            raise AssertionError("window lacks a causal predecessor")
        yield window
        previous_end = token_end
        index += 1
