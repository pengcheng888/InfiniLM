"""Validate InfiniLM summary timing JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TIMESTAMP_FIELDS = (
    "service_entered_at_ns",
    "scheduler_enqueue_before_at_ns",
    "forward_started_at_ns",
    "forward_completed_at_ns",
    "loop_submit_started_at_ns",
    "batch_put_started_at_ns",
    "batch_put_completed_at_ns",
    "route_token_received_at_ns",
    "route_token_end_at_ns",
)


def _valid_timestamp(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ordered(values: list[Any]) -> bool:
    present = [value for value in values if _valid_timestamp(value)]
    return all(left <= right for left, right in zip(present, present[1:]))


def validate_timing_root(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    files = sorted(root.glob("*/timing.jsonl"))
    parse_errors: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    finish_counts: dict[str, int] = {}
    record_count = 0

    for path in files:
        try:
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{path}: {exc}")
            continue

        for line_number, record in enumerate(records, 1):
            record_count += 1
            location = f"{path}:{line_number}"
            event = record.get("event")
            request_id = record.get("request_id")
            if event == "finish":
                if not isinstance(request_id, str) or not request_id:
                    errors.append(f"{location}: missing request_id")
                    continue
                finish_counts[request_id] = finish_counts.get(request_id, 0) + 1
            elif event not in {"batch", "token"}:
                errors.append(f"{location}: unknown event {event!r}")
                continue

            if record.get("schema_version") != 2:
                errors.append(f"{location}: invalid schema_version")

            if event == "batch":
                request_ids = record.get("request_ids")
                if not isinstance(request_ids, list) or not all(
                    isinstance(request_id, str) and request_id
                    for request_id in request_ids
                ):
                    errors.append(f"{location}: invalid request_ids")
                if record.get("phase") not in {"prefill", "decode"}:
                    errors.append(f"{location}: invalid batch phase")
                if not isinstance(record.get("batch_size"), int):
                    errors.append(f"{location}: invalid batch_size")
                if not _ordered(
                    [
                        record.get("forward_started_at_ns"),
                        record.get("forward_completed_at_ns"),
                    ]
                ):
                    errors.append(f"{location}: invalid forward timestamps")
                continue

            if not isinstance(request_id, str) or not request_id:
                errors.append(f"{location}: missing request_id")
                continue

            missing = [
                field_name
                for field_name in TIMESTAMP_FIELDS
                if not _valid_timestamp(record.get(field_name))
            ]
            if missing:
                if event == "token":
                    errors.append(f"{location}: missing timestamps {missing}")
                else:
                    warnings.append(f"{location}: incomplete finish {missing}")
                continue

            if not _ordered(
                [
                    record["service_entered_at_ns"],
                    record["scheduler_enqueue_before_at_ns"],
                    record["forward_started_at_ns"],
                    record["forward_completed_at_ns"],
                    record["loop_submit_started_at_ns"],
                ]
            ):
                errors.append(f"{location}: invalid StepThread timestamps")

            # The submit timestamp is captured before call_soon_threadsafe(), so
            # all thread-boundary timestamps have a strict order.
            if not _ordered(
                [
                    record["forward_completed_at_ns"],
                    record["loop_submit_started_at_ns"],
                    record["batch_put_started_at_ns"],
                    record["batch_put_completed_at_ns"],
                    record["route_token_received_at_ns"],
                    record["route_token_end_at_ns"],
                ]
            ):
                errors.append(f"{location}: invalid API-loop timestamps")

    duplicated = {
        request_id: count for request_id, count in finish_counts.items() if count > 1
    }
    for request_id, count in duplicated.items():
        errors.append(f"{request_id}: has {count} finish records")

    return {
        "root": str(root),
        "file_count": len(files),
        "record_count": record_count,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
        "request_count": len(finish_counts),
        "valid": not parse_errors and not errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_timing_root(args.timing_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
