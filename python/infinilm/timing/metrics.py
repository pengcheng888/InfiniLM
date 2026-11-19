"""Compute summary metrics from InfiniLM timing JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _elapsed_ms(record: dict[str, Any], start: str, end: str) -> float | None:
    started_at_ns = record.get(start)
    completed_at_ns = record.get(end)
    if not isinstance(started_at_ns, int) or not isinstance(completed_at_ns, int):
        return None
    return (completed_at_ns - started_at_ns) / 1_000_000


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "avg_ms": None,
            "min_ms": None,
            "max_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "avg_ms": mean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p50_ms": median(ordered),
        "p95_ms": _percentile(ordered, 0.95),
        "p99_ms": _percentile(ordered, 0.99),
    }


def _count_distribution(
    values: list[int],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "avg": None,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "avg": mean(ordered),
        "min": ordered[0],
        "p50": median(ordered),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _forward_window_metrics(
    batches: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ordered = sorted(
        batches,
        key=lambda record: record["forward_started_at_ns"],
    )
    if not ordered:
        return None

    first_batch = ordered[0]
    last_batch = ordered[-1]
    pure_forward_ms = sum(
        value
        for batch in ordered
        if (
            value := _elapsed_ms(
                batch,
                "forward_started_at_ns",
                "forward_completed_at_ns",
            )
        )
        is not None
    )
    forward_window_ms = _elapsed_ms(
        {
            "forward_started_at_ns": first_batch.get("forward_started_at_ns"),
            "forward_completed_at_ns": last_batch.get("forward_completed_at_ns"),
        },
        "forward_started_at_ns",
        "forward_completed_at_ns",
    )

    first_idle_count = first_batch.get(
        "step_idle_wait_count_cumulative_at_forward_start"
    )
    last_idle_count = last_batch.get(
        "step_idle_wait_count_cumulative_at_forward_start"
    )
    idle_count_in_window = (
        last_idle_count - first_idle_count
        if isinstance(first_idle_count, int) and isinstance(last_idle_count, int)
        else None
    )
    interval_idle_counts = [
        current["step_idle_wait_count_cumulative_at_forward_start"]
        - previous["step_idle_wait_count_cumulative_at_forward_start"]
        for previous, current in zip(ordered, ordered[1:])
        if isinstance(
            previous.get("step_idle_wait_count_cumulative_at_forward_start"),
            int,
        )
        and isinstance(
            current.get("step_idle_wait_count_cumulative_at_forward_start"),
            int,
        )
    ]

    return {
        "first_batch_id": first_batch.get("batch_id"),
        "last_batch_id": last_batch.get("batch_id"),
        "forward_window_ms": forward_window_ms,
        "pure_forward_ms": pure_forward_ms,
        "non_forward_gap_ms": (
            forward_window_ms - pure_forward_ms
            if forward_window_ms is not None
            else None
        ),
        "step_idle_wait_count_in_window": idle_count_in_window,
        "entered_step_idle_wait": (
            idle_count_in_window > 0 if idle_count_in_window is not None else None
        ),
        "step_idle_wait_count_since_previous_batch": _count_distribution(
            interval_idle_counts
        ),
    }


def _forward_windows_by_pid(
    batches: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    batches_by_pid: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for batch in batches:
        batches_by_pid[batch.get("pid", "unknown")].append(batch)
    return {
        str(pid): _forward_window_metrics(pid_batches)
        for pid, pid_batches in batches_by_pid.items()
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _kv_connector_batch_metrics(
    phase_batches: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    finalize_values = [
        value
        for batch in phase_batches
        if (
            value := _elapsed_ms(
                batch,
                "kv_connector_finalize_started_at_ns",
                "kv_connector_finalize_completed_at_ns",
            )
        )
        is not None
    ]
    get_finished_values = [
        value
        for batch in phase_batches
        if (
            value := _elapsed_ms(
                batch,
                "kv_connector_get_finished_started_at_ns",
                "kv_connector_get_finished_completed_at_ns",
            )
        )
        is not None
    ]
    remaining_values = [
        finalize_value - get_finished_value
        for batch in phase_batches
        if (
            finalize_value := _elapsed_ms(
                batch,
                "kv_connector_finalize_started_at_ns",
                "kv_connector_finalize_completed_at_ns",
            )
        )
        is not None
        and (
            get_finished_value := _elapsed_ms(
                batch,
                "kv_connector_get_finished_started_at_ns",
                "kv_connector_get_finished_completed_at_ns",
            )
        )
        is not None
    ]
    return {
        "kv_connector_finalize_ms": _distribution(finalize_values),
        "kv_connector_get_finished_ms": _distribution(get_finished_values),
        "kv_connector_finalize_remaining_ms": _distribution(remaining_values),
    }


def _load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in root.glob("*/timing.jsonl"):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


def _merged_interval_union_ms(
    records: list[dict[str, Any]],
    start_field: str,
    end_field: str,
) -> float:
    by_pid: dict[Any, list[list[int]]] = defaultdict(list)
    for record in records:
        started = record.get(start_field)
        completed = record.get(end_field)
        if isinstance(started, int) and isinstance(completed, int):
            by_pid[record.get("pid", "unknown")].append([started, completed])

    total_ms = 0.0
    for intervals in by_pid.values():
        intervals.sort(key=lambda item: item[0])
        merged: list[list[int]] = []
        for started, completed in intervals:
            if not merged or started > merged[-1][1]:
                merged.append([started, completed])
            else:
                merged[-1][1] = max(merged[-1][1], completed)
        total_ms += (
            sum(completed - started for started, completed in merged) / 1_000_000
        )
    return total_ms


def _mooncake_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    sends = [record for record in records if record.get("event") == "kv_send"]
    receives = [record for record in records if record.get("event") == "kv_receive"]
    unique_sends: dict[tuple[Any, str], dict[str, Any]] = {}
    for record in sends:
        batch_id = record.get("batch_id")
        if isinstance(batch_id, str) and batch_id:
            unique_sends[(record.get("pid"), batch_id)] = record
    unique_send_records = list(unique_sends.values())

    send_bytes = sum(
        record.get("batch_bytes", 0)
        for record in unique_send_records
        if isinstance(record.get("batch_bytes", 0), int)
    )
    receive_bytes = sum(
        record.get("bytes", 0)
        for record in receives
        if isinstance(record.get("bytes", 0), int)
    )
    send_busy_ms = _merged_interval_union_ms(
        unique_send_records,
        "batch_transfer_sync_write_started_at_ns",
        "batch_transfer_sync_write_completed_at_ns",
    )
    receive_busy_ms = _merged_interval_union_ms(
        receives,
        "kv_cache_pull_started_at_ns",
        "kv_cache_pull_confirmed_at_ns",
    )

    decode_batches_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in records:
        if batch.get("event") != "batch" or batch.get("phase") != "decode":
            continue
        for request_id in batch.get("request_ids", []):
            if isinstance(request_id, str):
                decode_batches_by_request[request_id].append(batch)

    decode_start_delays: list[float] = []
    for receive in receives:
        confirmed = receive.get("kv_cache_pull_confirmed_at_ns")
        request_id = receive.get("request_id")
        if not isinstance(confirmed, int) or not isinstance(request_id, str):
            continue
        starts = [
            batch.get("forward_started_at_ns")
            for batch in decode_batches_by_request.get(request_id, [])
            if isinstance(batch.get("forward_started_at_ns"), int)
            and batch["forward_started_at_ns"] >= confirmed
        ]
        if starts:
            decode_start_delays.append((min(starts) - confirmed) / 1_000_000)

    return {
        "kv_send": {
            "request_event_count": len(sends),
            "unique_batch_count": len(unique_send_records),
            "kv_cache_send_ready_to_batch_transfer_sync_write_started_ms": (
                _distribution(
                    [
                        value
                        for record in sends
                        if (
                            value := _elapsed_ms(
                                record,
                                "kv_cache_send_ready_at_ns",
                                "batch_transfer_sync_write_started_at_ns",
                            )
                        )
                        is not None
                    ]
                )
            ),
            "batch_transfer_sync_write_request_weighted_ms": _distribution(
                [
                    value
                    for record in sends
                    if (
                        value := _elapsed_ms(
                            record,
                            "batch_transfer_sync_write_started_at_ns",
                            "batch_transfer_sync_write_completed_at_ns",
                        )
                    )
                    is not None
                ]
            ),
            "batch_transfer_sync_write_batch_weighted_ms": _distribution(
                [
                    value
                    for record in unique_send_records
                    if (
                        value := _elapsed_ms(
                            record,
                            "batch_transfer_sync_write_started_at_ns",
                            "batch_transfer_sync_write_completed_at_ns",
                        )
                    )
                    is not None
                ]
            ),
            "batch_bytes_total": send_bytes,
            "busy_union_ms": send_busy_ms,
            "throughput_during_busy_gbps": (
                send_bytes / (send_busy_ms / 1000) / (1000**3)
                if send_busy_ms > 0
                else None
            ),
        },
        "kv_receive": {
            "request_event_count": len(receives),
            "kv_cache_pull_readiness_ms": _distribution(
                [
                    value
                    for record in receives
                    if (
                        value := _elapsed_ms(
                            record,
                            "kv_cache_pull_started_at_ns",
                            "kv_cache_pull_confirmed_at_ns",
                        )
                    )
                    is not None
                ]
            ),
            "decode_start_after_kv_cache_pull_confirmed_ms": _distribution(
                decode_start_delays
            ),
            "request_bytes_total": receive_bytes,
            "busy_union_ms": receive_busy_ms,
            "throughput_during_busy_gbps": (
                receive_bytes / (receive_busy_ms / 1000) / (1000**3)
                if receive_busy_ms > 0
                else None
            ),
        },
    }


def compute_timing_metrics(root: Path | str) -> dict[str, Any]:
    records = _load_records(Path(root))
    tokens = [record for record in records if record.get("event") == "token"]
    batches = [record for record in records if record.get("event") == "batch"]
    finishes = [record for record in records if record.get("event") == "finish"]
    batches_by_id = {
        batch.get("batch_id"): batch
        for batch in batches
        if isinstance(batch.get("batch_id"), str)
    }

    token_metric_edges = (
        (
            "service_to_enqueue_ms",
            "service_entered_at_ns",
            "scheduler_enqueue_before_at_ns",
        ),
        (
            "enqueue_to_forward_start_ms",
            "scheduler_enqueue_before_at_ns",
            "forward_started_at_ns",
        ),
        ("forward_ms", "forward_started_at_ns", "forward_completed_at_ns"),
        (
            "forward_to_loop_submit_started_ms",
            "forward_completed_at_ns",
            "loop_submit_started_at_ns",
        ),
        ("batch_put_ms", "batch_put_started_at_ns", "batch_put_completed_at_ns"),
        (
            "batch_put_to_route_received_ms",
            "batch_put_completed_at_ns",
            "route_token_received_at_ns",
        ),
        ("route_token_ms", "route_token_received_at_ns", "route_token_end_at_ns"),
        ("first_token_e2e_ms", "service_entered_at_ns", "route_token_received_at_ns"),
    )

    token_metrics = {
        name: _distribution(
            [
                value
                for record in tokens
                if (value := _elapsed_ms(record, start, end)) is not None
            ]
        )
        for name, start, end in token_metric_edges
    }

    first_tokens = [record for record in tokens if record.get("token_index") == 0]
    finished_tokens = [record for record in tokens if record.get("finished") is True]
    request_metric_edges = (
        (
            "request_first_token_e2e_ms",
            first_tokens,
            "service_entered_at_ns",
            "route_token_received_at_ns",
        ),
        (
            "request_e2e_to_final_route_ms",
            finished_tokens,
            "service_entered_at_ns",
            "route_token_end_at_ns",
        ),
        (
            "request_prefill_forward_ms",
            first_tokens,
            "forward_started_at_ns",
            "forward_completed_at_ns",
        ),
        (
            "request_enqueue_to_prefill_start_ms",
            first_tokens,
            "scheduler_enqueue_before_at_ns",
            "forward_started_at_ns",
        ),
    )
    request_metrics = {
        name: _distribution(
            [
                value
                for record in selected_records
                if (value := _elapsed_ms(record, start, end)) is not None
            ]
        )
        for name, selected_records, start, end in request_metric_edges
    }

    tokens_with_finalize = []
    finalize_timing_fields = (
        "kv_connector_finalize_started_at_ns",
        "kv_connector_get_finished_started_at_ns",
        "kv_connector_get_finished_completed_at_ns",
        "kv_connector_finalize_completed_at_ns",
    )
    for token in tokens:
        batch = batches_by_id.get(token.get("batch_id"))
        if batch is None:
            continue
        if (
            batch.get("kv_connector_finalize_started_at_ns") is None
            or batch.get("kv_connector_finalize_completed_at_ns") is None
        ):
            continue
        tokens_with_finalize.append(
            {
                **token,
                **{
                    field: batch[field]
                    for field in finalize_timing_fields
                    if batch.get(field) is not None
                },
            }
        )
        if all(
            isinstance(tokens_with_finalize[-1].get(field), int)
            for field in finalize_timing_fields
        ):
            token_with_finalize = tokens_with_finalize[-1]
            token_with_finalize["kv_connector_finalize_remaining_ns"] = (
                token_with_finalize["kv_connector_finalize_started_at_ns"]
                + (
                    token_with_finalize["kv_connector_finalize_completed_at_ns"]
                    - token_with_finalize["kv_connector_get_finished_completed_at_ns"]
                )
                + (
                    token_with_finalize["kv_connector_get_finished_started_at_ns"]
                    - token_with_finalize["kv_connector_finalize_started_at_ns"]
                )
            )

    post_forward_metric_edges = (
        (
            "forward_to_kv_connector_finalize_start_ms",
            "forward_completed_at_ns",
            "kv_connector_finalize_started_at_ns",
        ),
        (
            "kv_connector_finalize_ms",
            "kv_connector_finalize_started_at_ns",
            "kv_connector_finalize_completed_at_ns",
        ),
        (
            "kv_connector_get_finished_ms",
            "kv_connector_get_finished_started_at_ns",
            "kv_connector_get_finished_completed_at_ns",
        ),
        (
            "kv_connector_finalize_remaining_ms",
            "kv_connector_finalize_started_at_ns",
            "kv_connector_finalize_remaining_ns",
        ),
        (
            "kv_connector_finalize_to_loop_submit_ms",
            "kv_connector_finalize_completed_at_ns",
            "loop_submit_started_at_ns",
        ),
    )
    post_forward_metrics = {
        name: _distribution(
            [
                value
                for record in tokens_with_finalize
                if (value := _elapsed_ms(record, start, end)) is not None
            ]
        )
        for name, start, end in post_forward_metric_edges
    }

    batch_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batches:
        batch_by_phase[str(batch.get("phase"))].append(batch)

    return {
        "record_count": len(records),
        "token_record_count": len(tokens),
        "batch_record_count": len(batches),
        "finish_record_count": len(finishes),
        "forward_window_by_pid": _forward_windows_by_pid(batches),
        "token_metrics_ms": token_metrics,
        "request_metrics_ms": request_metrics,
        "batch": {
            phase: {
                "batch_size_counts": {
                    str(key): value
                    for key, value in sorted(
                        Counter(
                            batch.get("batch_size")
                            for batch in phase_batches
                            if isinstance(batch.get("batch_size"), int)
                        ).items(),
                        key=lambda item: int(item[0]),
                    )
                },
                "forward_duration_ms": _distribution(
                    [
                        value
                        for batch in phase_batches
                        if (
                            value := _elapsed_ms(
                                batch,
                                "forward_started_at_ns",
                                "forward_completed_at_ns",
                            )
                        )
                        is not None
                    ]
                ),
                "forward_window_by_pid": _forward_windows_by_pid(phase_batches),
                **_kv_connector_batch_metrics(phase_batches),
            }
            for phase, phase_batches in batch_by_phase.items()
        },
        "post_forward_metrics_ms": post_forward_metrics,
        "mooncake": _mooncake_metrics(records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)
    report = compute_timing_metrics(args.timing_root)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report["token_metrics_ms"], ensure_ascii=False, indent=2))
        print(json.dumps(report["request_metrics_ms"], ensure_ascii=False, indent=2))
        print(json.dumps(report["batch"], ensure_ascii=False, indent=2))
        print(json.dumps(report["mooncake"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
