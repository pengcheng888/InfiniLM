"""Environment-variable helpers shared by InfiniLM components."""

import os


MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV = (
    "INFINILM_MONONCAKE_KV_TRANSFER_BYPASS_DELAY_MS"
)


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
