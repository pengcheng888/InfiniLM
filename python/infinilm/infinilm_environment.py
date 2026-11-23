"""Environment-variable helpers shared by InfiniLM components."""

import os

MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV = (
    "INFINILM_MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS"
)

FORWARD_BYPASS_DELAY_MS_ENV = "INFINILM_FORWARD_BYPASS_DELAY_MS"

MOONCAKE_GET_FINISHED_MODE_ENV = "INFINILM_MOONCAKE_GET_FINISHED_MODE"
MOONCAKE_GET_FINISHED_LEGACY_MODE = "legacy"
MOONCAKE_GET_FINISHED_LOCK_SNAPSHOT_MODE = "lock_snapshot"


def get_mooncake_kv_transfer_bypass_delay_ms() -> int | None:
    """Return the simulated KV-transfer delay, or None to use real transfer."""
    raw_value = os.environ.get(MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV)
    if raw_value is None:
        return None

    normalized = raw_value.strip()
    if not normalized or normalized.lower() == "none":
        return None

    try:
        delay_ms = int(normalized)
    except ValueError as exc:
        raise RuntimeError(
            f"{MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV} must be none or "
            f"a positive integer, got {raw_value!r}"
        ) from exc

    if delay_ms < 1:
        raise RuntimeError(
            f"{MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV} must be at least "
            f"1 ms, got {raw_value!r}"
        )

    return delay_ms


def get_forward_bypass_delay_ms() -> int | None:
    """Return the simulated forward delay, or None to run real inference."""
    raw_value = os.environ.get(FORWARD_BYPASS_DELAY_MS_ENV)
    if raw_value is None:
        return None

    normalized = raw_value.strip()
    if not normalized or normalized.lower() == "none":
        return None

    try:
        delay_ms = int(normalized)
    except ValueError as exc:
        raise RuntimeError(
            f"{FORWARD_BYPASS_DELAY_MS_ENV} must be none or a positive integer, "
            f"got {raw_value!r}"
        ) from exc

    if delay_ms < 1:
        raise RuntimeError(
            f"{FORWARD_BYPASS_DELAY_MS_ENV} must be at least 1 ms, "
            f"got {raw_value!r}"
        )

    return delay_ms


def get_mooncake_get_finished_mode() -> str:
    """Select the D-side Mooncake get_finished implementation."""
    raw_value = os.environ.get(MOONCAKE_GET_FINISHED_MODE_ENV)
    if raw_value is None or not raw_value.strip():
        return MOONCAKE_GET_FINISHED_LOCK_SNAPSHOT_MODE

    mode = raw_value.strip().lower()
    supported_modes = {
        MOONCAKE_GET_FINISHED_LEGACY_MODE,
        MOONCAKE_GET_FINISHED_LOCK_SNAPSHOT_MODE,
    }
    if mode not in supported_modes:
        raise RuntimeError(
            f"{MOONCAKE_GET_FINISHED_MODE_ENV} must be one of "
            f"{sorted(supported_modes)}, got {raw_value!r}"
        )

    return mode
