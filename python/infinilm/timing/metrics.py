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


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in root.glob("*/timing.jsonl"):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


def compute_timing_metrics(root: Path | str) -> dict[str, Any]:
    records = _load_records(Path(root))
    tokens = [record for record in records if record.get("event") == "token"]
    batches = [record for record in records if record.get("event") == "batch"]
    finishes = [record for record in records if record.get("event") == "finish"]

    token_metric_edges = (
        ("service_to_enqueue_ms", "service_entered_at_ns", "scheduler_enqueue_before_at_ns"),
        ("enqueue_to_forward_start_ms", "scheduler_enqueue_before_at_ns", "forward_started_at_ns"),
        ("forward_ms", "forward_started_at_ns", "forward_completed_at_ns"),
        (
            "forward_to_loop_submit_started_ms",
            "forward_completed_at_ns",
            "loop_submit_started_at_ns",
        ),
        ("batch_put_ms", "batch_put_started_at_ns", "batch_put_completed_at_ns"),
        ("batch_put_to_route_received_ms", "batch_put_completed_at_ns", "route_token_received_at_ns"),
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

    first_tokens = [
        record for record in tokens if record.get("token_index") == 0
    ]
    finished_tokens = [
        record for record in tokens if record.get("finished") is True
    ]
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

    batch_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batches:
        batch_by_phase[str(batch.get("phase"))].append(batch)

    return {
        "record_count": len(records),
        "token_record_count": len(tokens),
        "batch_record_count": len(batches),
        "finish_record_count": len(finishes),
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
            }
            for phase, phase_batches in batch_by_phase.items()
        },
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
