"""Shared helpers for worker HTTP endpoint URLs."""

from __future__ import annotations

import ipaddress


def maybe_wrap_ipv6_address(hostname: str) -> str:
    """Wrap a bare IPv6 hostname in brackets and leave other hosts unchanged."""
    try:
        ipaddress.IPv6Address(hostname)
    except ValueError:
        return hostname
    return f"[{hostname}]"


def http_url(hostname: str, port: int) -> str:
    """Build an HTTP URL without breaking bare IPv6 hostnames."""
    wrapped_host = maybe_wrap_ipv6_address(hostname)
    return f"http://{wrapped_host}:{port}"
