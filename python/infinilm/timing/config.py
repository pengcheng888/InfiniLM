"""Configuration for the lightweight InfiniLM timing path."""

import os
from pathlib import Path


# Only INFINILM_TIMING is controlled through the environment. The remaining
# settings are intentionally module globals so tests and experimental scripts
# can override them without adding another startup variable.
TIMING_MODE = "summary"
TIMING_SAMPLE_INTERVAL = 32
TIMING_SUPPORTED_CACHE_TYPES = frozenset({"paged"})
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIMING_DIR = _PROJECT_ROOT / "timing"


def timing_enabled_from_environment() -> bool:
    raw_value = os.environ.get("INFINILM_TIMING")
    if raw_value is None:
        return False

    normalized = raw_value.strip().lower()
    if normalized not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise RuntimeError(f"INFINILM_TIMING must be a boolean: {raw_value!r}")
    return normalized in {"1", "true", "yes", "on"}


def timing_supported(cache_type: str) -> bool:
    return cache_type in TIMING_SUPPORTED_CACHE_TYPES


def timing_enabled_for(cache_type: str = "paged") -> bool:
    return timing_supported(cache_type) and timing_enabled_from_environment()
