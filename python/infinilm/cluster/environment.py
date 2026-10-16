"""Cluster routing configuration read from process environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from math import isfinite

AFFINITY_TTL_SECONDS_ENV = "INFINILM_CLUSTER_AFFINITY_TTL_SECONDS"
DEFAULT_AFFINITY_TTL_SECONDS = 300.0
AFFINITY_CAPACITY_ENV = "INFINILM_CLUSTER_AFFINITY_CAPACITY"
DEFAULT_AFFINITY_CAPACITY = 4096
KV_ROUTER_MIN_TOKENS_ENV = "INFINILM_CLUSTER_KV_ROUTER_MIN_TOKENS"
DEFAULT_KV_ROUTER_MIN_TOKENS = 1


def affinity_ttl_seconds(
    environ: Mapping[str, str] | None = None,
) -> float:
    """读取 session-affinity 绑定 TTL；未配置或为空时使用默认值。"""
    source = os.environ if environ is None else environ
    raw_value = source.get(AFFINITY_TTL_SECONDS_ENV)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_AFFINITY_TTL_SECONDS

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{AFFINITY_TTL_SECONDS_ENV} must be a number: {raw_value!r}"
        ) from exc

    if value < 0 or not isfinite(value):
        raise ValueError(
            f"{AFFINITY_TTL_SECONDS_ENV} must be a finite non-negative number"
        )
    return value


def affinity_capacity(environ: Mapping[str, str] | None = None) -> int:
    """读取 session-affinity LRU 容量；未配置或为空时使用默认值。"""
    source = os.environ if environ is None else environ
    raw_value = source.get(AFFINITY_CAPACITY_ENV)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_AFFINITY_CAPACITY

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{AFFINITY_CAPACITY_ENV} must be an integer: {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(f"{AFFINITY_CAPACITY_ENV} must be a positive integer")
    return value


def kv_router_min_tokens(environ: Mapping[str, str] | None = None) -> int:
    """读取 kv-aware 最小可复用 token 数；未配置或为空时使用默认值。"""
    source = os.environ if environ is None else environ
    raw_value = source.get(KV_ROUTER_MIN_TOKENS_ENV)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_KV_ROUTER_MIN_TOKENS

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{KV_ROUTER_MIN_TOKENS_ENV} must be an integer: {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(f"{KV_ROUTER_MIN_TOKENS_ENV} must be a positive integer")
    return value
