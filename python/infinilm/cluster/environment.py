"""Cluster runtime and test configuration read from environment variables."""

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
CONNECT_TIMEOUT_ENV = "INFINILM_CLUSTER_CONNECT_TIMEOUT"
DEFAULT_CONNECT_TIMEOUT = 3.0
REQUEST_TIMEOUT_ENV = "INFINILM_CLUSTER_REQUEST_TIMEOUT"
DEFAULT_REQUEST_TIMEOUT = 1000.0
BOOTSTRAP_TIMEOUT_ENV = "INFINILM_CLUSTER_BOOTSTRAP_TIMEOUT_SECONDS"
DEFAULT_BOOTSTRAP_TIMEOUT = 3.0
HEALTH_INTERVAL_SECONDS_ENV = "INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS"
DEFAULT_HEALTH_INTERVAL_SECONDS = 1.0
REGISTRY_TTL_SECONDS_ENV = "INFINILM_CLUSTER_REGISTRY_TTL_SECONDS"
DEFAULT_REGISTRY_TTL_SECONDS = 10.0
HEARTBEAT_INTERVAL_SECONDS_ENV = "INFINILM_CLUSTER_HEARTBEAT_INTERVAL_SECONDS"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 2.0
DRAIN_GRACE_SECONDS_ENV = "INFINILM_CLUSTER_DRAIN_GRACE_SECONDS"
DEFAULT_DRAIN_GRACE_SECONDS = 1.0
CLUSTER_COMMUNICATION_TEST_ENV = "INFINILM_CLUSTER_COMMUNICATION_TEST"
CLUSTER_COMMUNICATION_TEST_TOKEN_ID_ENV = (
    "INFINILM_CLUSTER_COMMUNICATION_TEST_TOKEN_ID"
)
DEFAULT_CLUSTER_COMMUNICATION_TEST_TOKEN_ID = 1
CLUSTER_COMMUNICATION_TEST_KV_BLOCKS_ENV = (
    "INFINILM_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS"
)
DEFAULT_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS = 2048


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


def cluster_communication_test(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """读取是否启用轻量集群通信测试模式。"""
    source = os.environ if environ is None else environ
    raw_value = source.get(CLUSTER_COMMUNICATION_TEST_ENV)
    if raw_value is None or not raw_value.strip():
        return False

    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{CLUSTER_COMMUNICATION_TEST_ENV} must be a boolean: {raw_value!r}"
    )


def cluster_communication_test_token_id(
    environ: Mapping[str, str] | None = None,
) -> int:
    """读取轻量通信测试模式返回的常数 sampled token id。"""
    source = os.environ if environ is None else environ
    raw_value = source.get(
        CLUSTER_COMMUNICATION_TEST_TOKEN_ID_ENV,
        str(DEFAULT_CLUSTER_COMMUNICATION_TEST_TOKEN_ID),
    )
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CLUSTER_COMMUNICATION_TEST_TOKEN_ID

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{CLUSTER_COMMUNICATION_TEST_TOKEN_ID_ENV} "
            f"must be an integer: {raw_value!r}"
        ) from exc

    if value < 0:
        raise ValueError(
            f"{CLUSTER_COMMUNICATION_TEST_TOKEN_ID_ENV} must be >= 0"
        )
    return value


def cluster_communication_test_kv_blocks(
    environ: Mapping[str, str] | None = None,
) -> int:
    """读取轻量通信测试模式分配并注册的 KV block 数。

    该值只影响通信测试专用的 KV tensor，不影响正常推理的显存自动规划；
    未配置时默认分配 2048 个 block，保证 P/D 拉取测试有足够容量。
    """
    source = os.environ if environ is None else environ
    raw_value = source.get(
        CLUSTER_COMMUNICATION_TEST_KV_BLOCKS_ENV,
        str(DEFAULT_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS),
    )
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{CLUSTER_COMMUNICATION_TEST_KV_BLOCKS_ENV} "
            f"must be an integer: {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(
            f"{CLUSTER_COMMUNICATION_TEST_KV_BLOCKS_ENV} must be a positive integer"
        )
    return value


def _positive_timeout(
    env_name: str,
    default: float,
    environ: Mapping[str, str] | None,
) -> float:
    source = os.environ if environ is None else environ
    raw_value = source.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a number: {raw_value!r}") from exc

    if value <= 0 or not isfinite(value):
        raise ValueError(f"{env_name} must be a finite positive number")
    return value


def _non_negative_timeout(
    env_name: str,
    default: float,
    environ: Mapping[str, str] | None,
) -> float:
    source = os.environ if environ is None else environ
    raw_value = source.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a number: {raw_value!r}") from exc

    if value < 0 or not isfinite(value):
        raise ValueError(f"{env_name} must be a finite non-negative number")
    return value


def connect_timeout(environ: Mapping[str, str] | None = None) -> float:
    """读取 frontend 到 worker 的建连超时；未配置时使用默认值。"""
    return _positive_timeout(
        CONNECT_TIMEOUT_ENV,
        DEFAULT_CONNECT_TIMEOUT,
        environ,
    )


def request_timeout(environ: Mapping[str, str] | None = None) -> float:
    """读取 frontend 到 worker 的单次请求超时；未配置时使用默认值。"""
    return _positive_timeout(
        REQUEST_TIMEOUT_ENV,
        DEFAULT_REQUEST_TIMEOUT,
        environ,
    )


def bootstrap_timeout_seconds(environ: Mapping[str, str] | None = None) -> float:
    """读取 PD prefill bootstrap 查询的专用总超时；未配置时使用 3 秒。

    bootstrap 只发生在后台 client 初始化阶段，不能复用推理请求的
    ``REQUEST_TIMEOUT``；否则一个尚未就绪的 prefill 可能让 health 或新请求
    等待接近 1000 秒。
    """
    return _positive_timeout(
        BOOTSTRAP_TIMEOUT_ENV,
        DEFAULT_BOOTSTRAP_TIMEOUT,
        environ,
    )


def health_interval_seconds(environ: Mapping[str, str] | None = None) -> float:
    """读取 registry watcher 轮询间隔；未配置时使用默认值。"""
    return _positive_timeout(
        HEALTH_INTERVAL_SECONDS_ENV,
        DEFAULT_HEALTH_INTERVAL_SECONDS,
        environ,
    )


def registry_ttl_seconds(environ: Mapping[str, str] | None = None) -> float:
    """读取 worker 注册租约时长；未配置时使用默认值。"""
    return _positive_timeout(
        REGISTRY_TTL_SECONDS_ENV,
        DEFAULT_REGISTRY_TTL_SECONDS,
        environ,
    )


def heartbeat_interval_seconds(environ: Mapping[str, str] | None = None) -> float:
    """读取 worker discovery 心跳间隔；未配置时使用默认值。"""
    return _positive_timeout(
        HEARTBEAT_INTERVAL_SECONDS_ENV,
        DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        environ,
    )


def drain_grace_seconds(environ: Mapping[str, str] | None = None) -> float:
    """读取 worker DRAINING 宽限时间；未配置时使用默认值。"""
    return _non_negative_timeout(
        DRAIN_GRACE_SECONDS_ENV,
        DEFAULT_DRAIN_GRACE_SECONDS,
        environ,
    )
