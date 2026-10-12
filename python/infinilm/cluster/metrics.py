"""Small in-process metrics collector for the serving cluster."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RouterMetrics:
    requests_total: int = 0
    errors_total: int = 0
    tokens_total: int = 0
    inflight: int = 0
    worker_requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    worker_errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def render(self, worker_count: int) -> str:
        lines = [
            "# HELP infinilm_router_workers Number of discovered workers.",
            "# TYPE infinilm_router_workers gauge",
            f"infinilm_router_workers {worker_count}",
            "# HELP infinilm_router_requests_total Total chat requests routed.",
            "# TYPE infinilm_router_requests_total counter",
            f"infinilm_router_requests_total {self.requests_total}",
            "# HELP infinilm_router_errors_total Total routing or worker errors.",
            "# TYPE infinilm_router_errors_total counter",
            f"infinilm_router_errors_total {self.errors_total}",
            "# HELP infinilm_router_tokens_total Token chunks streamed to clients.",
            "# TYPE infinilm_router_tokens_total counter",
            f"infinilm_router_tokens_total {self.tokens_total}",
        ]
        for worker_id, value in sorted(self.worker_requests.items()):
            lines.append(f'infinilm_router_worker_requests_total{{worker_id="{worker_id}"}} {value}')
        for worker_id, value in sorted(self.worker_errors.items()):
            lines.append(f'infinilm_router_worker_errors_total{{worker_id="{worker_id}"}} {value}')
        return "\n".join(lines) + "\n"


@dataclass
class FrontendMetrics:
    router: RouterMetrics = field(default_factory=RouterMetrics)

    def render(self, worker_count: int) -> str:
        return self.router.render(worker_count)
