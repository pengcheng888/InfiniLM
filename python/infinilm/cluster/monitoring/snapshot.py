"""Built-in monitoring snapshot for ClusterFrontend."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import FrontendOptions
    from ..metrics import RouterMetrics
    from ..registry import FileServiceWatcher
    from ..router import ClusterRouter

from ..metrics import (
    INTER_TOKEN_BUCKETS,
    REQUEST_DURATION_BUCKETS,
    TTFT_BUCKETS,
)


def _histogram_snapshot(
    histogram: Any,
    bucket_bounds: tuple[float, ...],
) -> dict[str, float | int]:
    count = int(getattr(histogram, "count", 0))
    total = float(getattr(histogram, "sum", 0.0))
    get_quantile = getattr(histogram, "quantile", None)
    if callable(get_quantile):
        p50 = float(get_quantile(0.50, bucket_bounds))
        p95 = float(get_quantile(0.95, bucket_bounds))
        p99 = float(get_quantile(0.99, bucket_bounds))
    else:
        p50 = p95 = p99 = 0.0

    return {
        "count": count,
        "sum": total,
        "avg": total / count if count else 0.0,
        "p50": p50,
        "p95": p95,
        "p99": p99,
    }


def _worker_snapshot(worker: Any, metrics: RouterMetrics) -> dict[str, Any]:
    worker_id = worker.instance_id
    metadata = worker.metadata or {}
    raw_load = metadata.get("load")
    load = raw_load if isinstance(raw_load, dict) else {}
    raw_traffic = metadata.get("traffic")
    traffic = raw_traffic if isinstance(raw_traffic, dict) else {}
    gpu = metadata.get("cuda_visible_devices")
    heartbeat_age_seconds = max(0.0, time.time() - worker.heartbeat_at)
    max_batch_size = int(
        load.get("max_batch_size", metadata.get("max_batch_size", 0))
    )

    return {
        "instance_id": worker_id,
        "served_model_name": worker.served_model_name,
        "model": worker.model,
        "endpoint": f"{worker.host}:{worker.port}",
        "gpu": str(gpu) if gpu is not None else "-",
        "tp": metadata.get("tp"),
        "worker_type": worker.worker_type,
        "transport": worker.transport,
        "state": worker.state,
        "engine_healthy": bool(worker.engine_healthy),
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "load_snapshot_age_seconds": heartbeat_age_seconds,
        "lease_remaining_seconds": max(
            0.0, worker.lease_expires_at - time.time()
        ),
        "load_score": metrics.worker_load_scores.get(worker_id, 0.0),
        "inflight": metrics.worker_inflight.get(worker_id, 0),
        "waiting_requests": metrics.worker_waiting_requests.get(
            worker_id, int(load.get("waiting_requests", 0))
        ),
        "running_requests": metrics.worker_running_requests.get(
            worker_id, int(load.get("running_requests", 0))
        ),
        "max_batch_size": max_batch_size,
        "kv_usage_ratio": metrics.worker_kv_usage.get(
            worker_id, float(load.get("kv_usage_ratio", 0.0))
        ),
        "requests_total": metrics.worker_requests.get(worker_id, 0),
        "worker_received_requests_total": int(
            traffic.get("received_requests_total", 0)
        ),
        "untracked_requests_total": max(
            0,
            int(traffic.get("received_requests_total", 0))
            - int(metrics.worker_requests.get(worker_id, 0)),
        ),
        "worker_active_requests": int(traffic.get("active_requests", 0)),
        "worker_successful_requests_total": int(
            traffic.get("successful_requests_total", 0)
        ),
        "worker_failed_requests_total": int(
            traffic.get("failed_requests_total", 0)
        ),
        "worker_canceled_requests_total": int(
            traffic.get("canceled_requests_total", 0)
        ),
        "usage_prompt_tokens_total": metrics.worker_usage_prompt_tokens.get(
            worker_id, 0
        ),
        "usage_completion_tokens_total": (
            metrics.worker_usage_completion_tokens.get(worker_id, 0)
        ),
        "usage_total_tokens_total": metrics.worker_usage_total_tokens.get(
            worker_id, 0
        ),
        "errors_total": metrics.worker_errors.get(worker_id, 0),
        "retries_total": metrics.worker_retries.get(worker_id, 0),
        "last_success_timestamp": metrics.worker_last_success_timestamp.get(
            worker_id, 0.0
        ),
        "request_duration": _histogram_snapshot(
            metrics.worker_duration.get(worker_id),
            REQUEST_DURATION_BUCKETS,
        ),
        "ttft": _histogram_snapshot(
            metrics.worker_ttft.get(worker_id),
            TTFT_BUCKETS,
        ),
    }


def monitoring_snapshot(
    *,
    options: FrontendOptions,
    watcher: FileServiceWatcher,
    router: ClusterRouter,
) -> dict[str, Any]:
    """Build a JSON snapshot consumed by the built-in dashboard."""
    router.update_worker_metrics(watcher.workers)
    metrics = router.metrics
    workers = watcher.workers
    worker_states = {
        "ready": sum(worker.state == "ready" for worker in workers),
        "draining": sum(worker.state == "draining" for worker in workers),
    }

    return {
        "generated_at": time.time(),
        "namespace": options.namespace,
        "router_mode": router.router_mode,
        "request_plane": options.request_plane,
        "metrics_source": "frontend-router",
        "ready": worker_states["ready"] >= options.router_min_initial_workers,
        "worker_count": len(workers),
        "worker_states": worker_states,
        "minimum_workers": options.router_min_initial_workers,
        "global": {
            "requests_total": metrics.requests_total,
            "attempts_total": metrics.attempts_total,
            "successful_requests_total": metrics.successful_requests_total,
            "streaming_requests_total": metrics.streaming_requests_total,
            "non_streaming_requests_total": metrics.non_streaming_requests_total,
            "errors_total": metrics.errors_total,
            "retries_total": metrics.retries_total,
            "tokens_total": metrics.tokens_total,
            "usage_prompt_tokens_total": metrics.usage_prompt_tokens_total,
            "usage_completion_tokens_total": metrics.usage_completion_tokens_total,
            "usage_total_tokens_total": metrics.usage_total_tokens_total,
            "inflight": metrics.inflight,
            "affinity_requests_total": metrics.affinity_requests_total,
            "affinity_hits_total": metrics.affinity_hits_total,
            "affinity_misses_total": metrics.affinity_misses_total,
            "affinity_entries": len(router.affinity),
            "worker_received_requests_total": sum(
                int(
                    (worker.metadata or {}).get("traffic", {}).get(
                        "received_requests_total", 0
                    )
                )
                if isinstance((worker.metadata or {}).get("traffic"), dict)
                else 0
                for worker in workers
            ),
            "worker_active_requests": sum(
                int((worker.metadata or {}).get("traffic", {}).get("active_requests", 0))
                if isinstance((worker.metadata or {}).get("traffic"), dict)
                else 0
                for worker in workers
            ),
            "untracked_requests_total": sum(
                max(
                    0,
                    int(
                        (worker.metadata or {}).get("traffic", {}).get(
                            "received_requests_total", 0
                        )
                    )
                    - int(metrics.worker_requests.get(worker.instance_id, 0)),
                )
                if isinstance((worker.metadata or {}).get("traffic"), dict)
                else 0
                for worker in workers
            ),
        },
        "latency": {
            "request_duration": _histogram_snapshot(
                metrics.request_duration,
                REQUEST_DURATION_BUCKETS,
            ),
            "ttft": _histogram_snapshot(metrics.ttft, TTFT_BUCKETS),
            "inter_token_latency": _histogram_snapshot(
                metrics.inter_token_latency,
                INTER_TOKEN_BUCKETS,
            ),
        },
        "workers": [_worker_snapshot(worker, metrics) for worker in workers],
    }
