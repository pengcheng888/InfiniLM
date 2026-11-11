"""Validate InfiniLM summary timing JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
SUPPORTED_SCHEMA_VERSIONS = {3, 4}


def _valid_timestamp(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_count(value: Any) -> bool:
    return _valid_timestamp(value) and value >= 0


def _ordered(values: list[Any]) -> bool:
    present = [value for value in values if _valid_timestamp(value)]
    return all(left <= right for left, right in zip(present, present[1:]))


def _validate_common_kv_fields(
    record: dict[str, Any],
    location: str,
    *,
    event: str,
) -> list[str]:
    errors: list[str] = []
    expected_role = "prefill" if event == "kv_send" else "decode"
    expected_scope = (
        "producer_batch_transfer"
        if event == "kv_send"
        else "decode_local_kv_readiness"
    )
    if record.get("role") != expected_role:
        errors.append(f"{location}: invalid {event} role")
    if record.get("scope") != expected_scope:
        errors.append(f"{location}: invalid {event} scope")
    if record.get("clock_domain") != "local-monotonic":
        errors.append(f"{location}: invalid clock_domain")
    if record.get("cross_node_comparable") is not False:
        errors.append(f"{location}: kv timing must not be cross-node comparable")
    if record.get("outcome") not in {"success", "failed"}:
        errors.append(f"{location}: invalid {event} outcome")
    if not _valid_count(record.get("bytes")):
        errors.append(f"{location}: invalid {event} bytes")
    if not _valid_count(record.get("block_count")):
        errors.append(f"{location}: invalid {event} block_count")
    if not isinstance(record.get("mooncake_kv_transfer_bypassed"), bool):
        errors.append(f"{location}: invalid mooncake_kv_transfer_bypassed")
    return errors


def _validate_kv_timestamps(
    record: dict[str, Any],
    location: str,
    *,
    event: str,
) -> list[str]:
    if event == "kv_send":
        fields = (
            "kv_cache_send_ready_at_ns",
            "batch_transfer_sync_write_started_at_ns",
            "batch_transfer_sync_write_completed_at_ns",
        )
    else:
        fields = (
            "kv_cache_pull_started_at_ns",
            "kv_cache_pull_confirmed_at_ns",
        )

    if not all(_valid_timestamp(record.get(field)) for field in fields):
        return [f"{location}: missing or invalid {event} timestamps"]
    if not _ordered([record.get(field) for field in fields]):
        return [f"{location}: invalid {event} timestamp order"]
    return []


def _validate_kv_send_batch_fields(
    record: dict[str, Any],
    location: str,
) -> list[str]:
    errors: list[str] = []
    count_fields = (
        "batch_size",
        "batch_request_count",
        "batch_block_count",
        "batch_bytes",
        "request_block_count",
    )
    for field in count_fields:
        if not _valid_count(record.get(field)):
            errors.append(f"{location}: invalid {field}")
    if not isinstance(record.get("batch_id"), str) or not record.get("batch_id"):
        errors.append(f"{location}: invalid batch_id")
    return errors


def _validate_kv_send_batches(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    batches: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("event") != "kv_send":
            continue
        batch_id = record.get("batch_id")
        if isinstance(batch_id, str) and batch_id:
            batches[(record.get("pid"), batch_id)].append(record)

    for (_, batch_id), batch_records in batches.items():
        common_fields = (
            "batch_transfer_sync_write_started_at_ns",
            "batch_transfer_sync_write_completed_at_ns",
            "batch_request_count",
            "batch_block_count",
            "batch_bytes",
            "block_count",
            "bytes",
        )
        for field in common_fields:
            if len({record.get(field) for record in batch_records}) != 1:
                errors.append(f"kv_send batch {batch_id}: inconsistent {field}")

        expected_count = batch_records[0].get("batch_request_count")
        if expected_count != len(batch_records):
            errors.append(
                f"kv_send batch {batch_id}: expected {expected_count} requests, "
                f"got {len(batch_records)}"
            )

        request_blocks = [
            record.get("request_block_count")
            for record in batch_records
            if _valid_count(record.get("request_block_count"))
        ]
        if len(request_blocks) != len(batch_records) or sum(request_blocks) != (
            batch_records[0].get("batch_block_count")
        ):
            errors.append(
                f"kv_send batch {batch_id}: request_block_count does not sum "
                "to batch_block_count"
            )
    return errors


def validate_timing_root(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    files = sorted(root.glob("*/timing.jsonl"))
    parse_errors: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    finish_counts: dict[str, int] = {}
    record_count = 0
    records: list[dict[str, Any]] = []

    for path in files:
        try:
            file_records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{path}: {exc}")
            continue
        records.extend(file_records)

        for line_number, record in enumerate(file_records, 1):
            record_count += 1
            location = f"{path}:{line_number}"
            event = record.get("event")
            request_id = record.get("request_id")
            if event == "finish":
                if not isinstance(request_id, str) or not request_id:
                    errors.append(f"{location}: missing request_id")
                    continue
                finish_counts[request_id] = finish_counts.get(request_id, 0) + 1
            elif event not in {"batch", "token", "kv_send", "kv_receive"}:
                errors.append(f"{location}: unknown event {event!r}")
                continue

            if record.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
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
                finalize_fields = (
                    "kv_connector_finalize_started_at_ns",
                    "kv_connector_finalize_completed_at_ns",
                )
                if any(record.get(field) is not None for field in finalize_fields):
                    if not all(
                        _valid_timestamp(record.get(field))
                        for field in finalize_fields
                    ):
                        errors.append(
                            f"{location}: invalid kv_connector_finalize timestamps"
                        )
                    elif not _ordered(
                        [
                            record.get("forward_completed_at_ns"),
                            *(
                                record.get(field)
                                for field in finalize_fields
                            ),
                        ]
                    ):
                        errors.append(
                            f"{location}: invalid kv_connector_finalize order"
                        )
                continue

            if event in {"kv_send", "kv_receive"}:
                if not isinstance(request_id, str) or not request_id:
                    errors.append(f"{location}: missing request_id")
                errors.extend(_validate_common_kv_fields(record, location, event=event))
                errors.extend(_validate_kv_timestamps(record, location, event=event))
                if event == "kv_send":
                    errors.extend(_validate_kv_send_batch_fields(record, location))
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

    errors.extend(_validate_kv_send_batches(records))

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
