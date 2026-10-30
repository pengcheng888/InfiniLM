"""校验 request timing JSONL 的完整性和一致性。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TIMING_ROOT = Path("/tmp/infinilm-request-timing")
# 合法事件分为请求执行路径（step/finish）和 KV 传输路径两类。
VALID_EVENTS = {"step", "finish", "kv_send", "kv_receive"}
# role 对应 worker 类型；PD 的同一个 request 会分布在两个 role 文件中。
VALID_ROLES = {"aggregated", "prefill", "decode"}
# v3 起不再记录 ModelRunner 入口时间；scheduler batch ready 到 engine forward
# start 作为 一个合并区间 记录。
TIMESTAMP_FIELDS = (
    "server_received_at_ns",
    "scheduler_enqueued_at_ns",
    "scheduler_polled_at_ns",
    "scheduler_batch_ready_at_ns",
    "engine_forward_started_at_ns",
    "engine_forward_completed_at_ns",
)
TIMESTAMP_FIELDS_V5 = (
    "server_received_at_ns",
    "request_created_at_ns",
    "scheduler_enqueued_at_ns",
    "scheduler_polled_at_ns",
    "scheduler_batch_ready_at_ns",
    "engine_forward_started_at_ns",
    "engine_forward_completed_at_ns",
)
TIMESTAMP_FIELDS_V6 = (
    "server_received_at_ns",
    "body_read_completed_at_ns",
    "json_parsed_at_ns",
    "route_prepared_at_ns",
    "handler_started_at_ns",
    "preprocessing_started_at_ns",
    "chat_template_completed_at_ns",
    "multimodal_resolved_at_ns",
    "tokenization_completed_at_ns",
    "prompt_token_ids_ready_at_ns",
    "request_created_at_ns",
    "scheduler_enqueued_at_ns",
    "scheduler_polled_at_ns",
    "scheduler_batch_ready_at_ns",
    "engine_forward_started_at_ns",
    "engine_forward_completed_at_ns",
)
ENTRANCE_TIMESTAMP_FIELDS_V6 = TIMESTAMP_FIELDS_V6[1:10]
ENTRANCE_DURATION_FIELDS_V6 = (
    (
        "server_to_body_read",
        "server_received_at_ns",
        "body_read_completed_at_ns",
    ),
    (
        "body_read_to_json_parse",
        "body_read_completed_at_ns",
        "json_parsed_at_ns",
    ),
    (
        "json_parse_to_route_prepare",
        "json_parsed_at_ns",
        "route_prepared_at_ns",
    ),
    (
        "route_prepare_to_handler_start",
        "route_prepared_at_ns",
        "handler_started_at_ns",
    ),
    (
        "handler_start_to_preprocessing_start",
        "handler_started_at_ns",
        "preprocessing_started_at_ns",
    ),
    (
        "preprocessing_to_chat_template",
        "preprocessing_started_at_ns",
        "chat_template_completed_at_ns",
    ),
    (
        "chat_template_to_multimodal_resolve",
        "chat_template_completed_at_ns",
        "multimodal_resolved_at_ns",
    ),
    (
        "multimodal_to_tokenization",
        "multimodal_resolved_at_ns",
        "tokenization_completed_at_ns",
    ),
    (
        "tokenization_to_prompt_token_ids",
        "tokenization_completed_at_ns",
        "prompt_token_ids_ready_at_ns",
    ),
    (
        "prompt_token_ids_to_request_created",
        "prompt_token_ids_ready_at_ns",
        "request_created_at_ns",
    ),
)
# 兼容 schema v1/v2 的历史日志。
LEGACY_TIMESTAMP_FIELDS = (
    "server_received_at_ns",
    "scheduler_enqueued_at_ns",
    "scheduler_polled_at_ns",
    "scheduler_batch_ready_at_ns",
    "model_runner_entered_at_ns",
    "engine_forward_started_at_ns",
    "engine_forward_completed_at_ns",
)
SCHEDULER_SNAPSHOT_FIELDS = (
    "scheduler_waiting_queue_depth_before",
    "scheduler_running_queue_depth_before",
    "scheduler_remote_kv_requests_before",
    "scheduler_waiting_queue_depth_after",
    "scheduler_running_queue_depth_after",
    "scheduler_remote_kv_requests_after",
)


@dataclass
class FileValidationResult:
    """一个“进程 + 请求”JSONL 文件的校验结果。"""

    path: Path
    request_id: str | None = None
    role: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    complete: bool = False

    @property
    def status(self) -> str:
        """返回文件状态：结构错误优先于未完成。"""
        if self.errors:
            return "invalid"
        return "complete" if self.complete else "incomplete"

    def to_dict(self) -> dict[str, Any]:
        """转换为 CLI/JSON 输出使用的普通字典。"""
        return {
            "path": str(self.path),
            "request_id": self.request_id,
            "role": self.role,
            "status": self.status,
            "record_count": len(self.records),
            "errors": self.errors,
        }


def iter_timing_files(root: Path) -> list[Path]:
    """返回 timing root 下所有 <pid>/<request>.jsonl 文件。"""
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*/*.jsonl") if path.is_file())


def expected_timing_filename(request_id: str) -> str:
    """按 recorder 的安全规则反推 request 对应的 JSONL 文件名。"""
    safe_characters = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    )
    safe_prefix = "".join(
        character if character in safe_characters else "_" for character in request_id
    )[:80]
    if safe_prefix != request_id or not safe_prefix:
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:12]
        safe_prefix = f"{safe_prefix}-{digest}"
    return f"{safe_prefix}.jsonl"


def _append_error(result: FileValidationResult, line_number: int, message: str):
    """把错误挂到具体 JSONL 行号，便于定位半截或损坏记录。"""
    result.errors.append(f"line {line_number}: {message}")


def _validate_step(result: FileValidationResult, record: dict[str, Any]):
    """校验一条 step 的 phase、时间戳和 durations 自洽性。"""
    step_index = record.get("step_index")
    phase = record.get("phase")
    if not isinstance(step_index, int) or step_index < 0:
        result.errors.append("step has invalid step_index")
    if phase not in {"prefill", "decode"}:
        result.errors.append("step has invalid phase")
    forward_batch_id = record.get("forward_batch_id")
    if forward_batch_id is not None and (
        not isinstance(forward_batch_id, str) or not forward_batch_id
    ):
        result.errors.append("step has invalid forward_batch_id")

    schema_version = record.get("schema_version")
    legacy_schema = schema_version is None or schema_version <= 2
    entrance_values = (
        [record.get(field) for field in ENTRANCE_TIMESTAMP_FIELDS_V6]
        if schema_version == 6
        else []
    )
    complete_entrance_timing = bool(entrance_values) and all(
        isinstance(value, int) and value >= 0 for value in entrance_values
    )
    missing_entrance_timing = (
        any(value is not None for value in entrance_values)
        and not complete_entrance_timing
    )
    timestamp_fields = (
        LEGACY_TIMESTAMP_FIELDS
        if legacy_schema
        else (
            TIMESTAMP_FIELDS_V6
            if schema_version == 6
            and (complete_entrance_timing or missing_entrance_timing)
            else TIMESTAMP_FIELDS_V5
            if schema_version == 5
            else TIMESTAMP_FIELDS
        )
    )
    if not legacy_schema and "model_runner_entered_at_ns" in record:
        result.errors.append("schema v3 must not record model_runner_entered_at_ns")
        return

    timestamps = [record.get(field) for field in timestamp_fields]
    if any(not isinstance(value, int) or value < 0 for value in timestamps):
        result.errors.append("step has missing or non-integer timestamps")
        return
    if any(left > right for left, right in zip(timestamps, timestamps[1:])):
        result.errors.append("step timestamps are not monotonic")

    if schema_version is not None and schema_version >= 4:
        for field_name in SCHEDULER_SNAPSHOT_FIELDS:
            value = record.get(field_name)
            if not isinstance(value, int) or value < 0:
                result.errors.append(f"step has invalid {field_name}")

    # durations 必须能由时间戳重算出来，防止日志字段被单独写错。
    if legacy_schema:
        expected_durations = {
            "queue": timestamps[2] - timestamps[0],
            "scheduler": timestamps[3] - timestamps[2],
            "dispatch": timestamps[4] - timestamps[3],
            "model_input_build": timestamps[5] - timestamps[4],
            "engine_forward": timestamps[6] - timestamps[5],
        }
    else:
        timestamp_by_name = {field: record.get(field) for field in timestamp_fields}
        expected_durations = {
            "queue": (
                timestamp_by_name["scheduler_polled_at_ns"]
                - timestamp_by_name["server_received_at_ns"]
            ),
            "scheduler": (
                timestamp_by_name["scheduler_batch_ready_at_ns"]
                - timestamp_by_name["scheduler_polled_at_ns"]
            ),
            "scheduler_batch_ready_to_engine_forward_start": (
                timestamp_by_name["engine_forward_started_at_ns"]
                - timestamp_by_name["scheduler_batch_ready_at_ns"]
            ),
            "engine_forward": (
                timestamp_by_name["engine_forward_completed_at_ns"]
                - timestamp_by_name["engine_forward_started_at_ns"]
            ),
        }
        if schema_version == 5:
            expected_durations.update(
                {
                    "server_to_request_created": (
                        timestamp_by_name["request_created_at_ns"]
                        - timestamp_by_name["server_received_at_ns"]
                    ),
                    "request_created_to_scheduler_enqueue": (
                        timestamp_by_name["scheduler_enqueued_at_ns"]
                        - timestamp_by_name["request_created_at_ns"]
                    ),
                }
            )
        if schema_version == 6 and complete_entrance_timing:
            expected_durations.update(
                {
                    duration_name: (
                        timestamp_by_name[end_field] - timestamp_by_name[start_field]
                    )
                    for duration_name, start_field, end_field in (
                        ENTRANCE_DURATION_FIELDS_V6
                    )
                }
            )
    actual_durations = record.get("durations_ns")
    if not isinstance(actual_durations, dict):
        result.errors.append("step has no durations_ns object")
        return
    for name, expected in expected_durations.items():
        actual = actual_durations.get(name)
        if actual != expected:
            result.errors.append(
                f"step {name} duration mismatch: expected {expected}, got {actual}"
            )


def _validate_finish(
    result: FileValidationResult,
    record: dict[str, Any],
    steps: list[dict[str, Any]],
):
    """校验 finish 汇总是否与同文件 step 明细一致。"""
    if record.get("timing_mode") == "summary":
        expected_order = (
            record.get("server_received_at_ns"),
            record.get("request_created_at_ns"),
            record.get("scheduler_enqueued_at_ns"),
            record.get("first_scheduler_polled_at_ns"),
            record.get("scheduler_batch_ready_at_ns"),
            record.get("first_engine_forward_started_at_ns"),
            record.get("first_engine_forward_completed_at_ns"),
            record.get("last_engine_forward_completed_at_ns"),
            record.get("finish_local_monotonic_ns"),
        )
        timestamps = [value for value in expected_order if value is not None]
        if any(not isinstance(value, int) or value < 0 for value in timestamps):
            result.errors.append("summary finish has invalid timestamps")
        elif any(left > right for left, right in zip(timestamps, timestamps[1:])):
            result.errors.append("summary finish timestamps are not monotonic")

        steps = record.get("steps")
        forward_total = record.get("engine_forward_total_ns")
        estimated_total = record.get("engine_forward_total_estimate_ns")
        sample_interval = record.get("sample_interval")
        decode_steps = record.get("decode_step_count")
        sampled_steps = record.get("sampled_decode_step_count")
        sampled_forward = record.get("sampled_decode_forward_ns")
        if not isinstance(steps, int) or steps < 0:
            result.errors.append("summary finish has invalid steps")
        if forward_total is not None and (
            not isinstance(forward_total, int) or forward_total < 0
        ):
            result.errors.append("summary finish has invalid engine_forward_total_ns")
        if estimated_total is not None and (
            not isinstance(estimated_total, int) or estimated_total < 0
        ):
            result.errors.append(
                "summary finish has invalid engine_forward_total_estimate_ns"
            )
        if not isinstance(sample_interval, int) or sample_interval < 1:
            result.errors.append("summary finish has invalid sample_interval")
        if not isinstance(decode_steps, int) or decode_steps < 0:
            result.errors.append("summary finish has invalid decode_step_count")
        if not isinstance(sampled_steps, int) or not 0 <= sampled_steps <= decode_steps:
            result.errors.append("summary finish has invalid sampled_decode_step_count")
        if not isinstance(sampled_forward, int) or sampled_forward < 0:
            result.errors.append("summary finish has invalid sampled_decode_forward_ns")
        if record.get("schema_version") == 8:
            if not isinstance(estimated_total, int) or estimated_total < 0:
                result.errors.append("sampled summary finish has invalid estimate")
            if decode_steps > 0 and sampled_steps == 0:
                result.errors.append("sampled summary finish has no decode samples")

        first_forward_duration = (
            record.get("first_engine_forward_completed_at_ns")
            - record.get("first_engine_forward_started_at_ns")
            if isinstance(record.get("first_engine_forward_completed_at_ns"), int)
            and isinstance(record.get("first_engine_forward_started_at_ns"), int)
            else None
        )
        if (
            first_forward_duration is not None
            and isinstance(forward_total, int)
            and forward_total < first_forward_duration
        ):
            result.errors.append(
                "summary finish engine_forward_total_ns is less than first forward"
            )
        if estimated_total is not None and first_forward_duration is not None:
            first_forward_is_decode = (
                decode_steps > 0 and record.get("role") == "decode"
            )
            if (
                isinstance(estimated_total, int)
                and estimated_total < first_forward_duration
                and not first_forward_is_decode
            ):
                result.errors.append(
                    "summary finish estimate is less than first prefill forward"
                )
        return

    expected_steps = len(steps)
    if record.get("steps") != expected_steps:
        result.errors.append(
            f"finish steps mismatch: expected {expected_steps}, "
            f"got {record.get('steps')}"
        )

    expected_forward_total = sum(
        (
            step["durations_ns"].get("engine_forward", 0)
            if isinstance(step.get("durations_ns"), dict)
            else 0
        )
        for step in steps
    )
    if record.get("engine_forward_total_ns") != expected_forward_total:
        result.errors.append(
            "finish engine_forward_total_ns mismatch: "
            f"expected {expected_forward_total}, got {record.get('engine_forward_total_ns')}"
        )

    if steps:
        # finish 的首尾字段必须指向 step 序列的第一条和最后一条。
        first = steps[0]
        last = steps[-1]
        expected_fields = {
            "server_received_at_ns": first.get("server_received_at_ns"),
            "first_scheduler_polled_at_ns": first.get("scheduler_polled_at_ns"),
            "last_engine_forward_completed_at_ns": last.get(
                "engine_forward_completed_at_ns"
            ),
        }
        if first.get("schema_version") is not None and first.get("schema_version") <= 2:
            expected_fields["first_model_runner_entered_at_ns"] = first.get(
                "model_runner_entered_at_ns"
            )
        else:
            expected_fields["first_engine_forward_started_at_ns"] = first.get(
                "engine_forward_started_at_ns"
            )
        if first.get("schema_version") == 5:
            expected_fields["request_created_at_ns"] = first.get(
                "request_created_at_ns"
            )
        if first.get("schema_version") == 6:
            expected_fields["request_created_at_ns"] = first.get(
                "request_created_at_ns"
            )
        if first.get("schema_version") == 6 and all(
            first.get(field) is not None for field in ENTRANCE_TIMESTAMP_FIELDS_V6
        ):
            for field_name in (
                "body_read_completed_at_ns",
                "json_parsed_at_ns",
                "route_prepared_at_ns",
                "handler_started_at_ns",
                "preprocessing_started_at_ns",
                "chat_template_completed_at_ns",
                "multimodal_resolved_at_ns",
                "tokenization_completed_at_ns",
                "prompt_token_ids_ready_at_ns",
            ):
                expected_fields[field_name] = first.get(field_name)
        for field_name, expected in expected_fields.items():
            if record.get(field_name) != expected:
                result.errors.append(f"finish {field_name} does not match step records")


def _validate_transfer_event(
    result: FileValidationResult,
    record: dict[str, Any],
):
    """校验 KV 传输事件的本地时钟和 producer batch 字段。"""
    event = record["event"]
    # kv_send 在 P 侧，kv_receive 在 D 侧；scope 描述统计口径。
    expected_role = "prefill" if event == "kv_send" else "decode"
    expected_scope = (
        "producer_batch_transfer" if event == "kv_send" else "decode_local_kv_readiness"
    )
    if record.get("role") != expected_role:
        result.errors.append(f"{event} has unexpected role")
    if "scope" in record and record.get("scope") != expected_scope:
        result.errors.append(f"{event} has unexpected scope")
    if "clock_domain" in record and record.get("clock_domain") != "local-monotonic":
        result.errors.append(f"{event} has invalid clock_domain")
    if (
        "cross_node_comparable" in record
        and record.get("cross_node_comparable") is not False
    ):
        result.errors.append(f"{event} must set cross_node_comparable=false")

    started = record.get("started_at_ns")
    completed = record.get("completed_at_ns")
    duration = record.get("duration_ns")
    # 只允许同节点 monotonic 时间相减，不比较 P/D 的绝对时间。
    if (
        not isinstance(started, int)
        or not isinstance(completed, int)
        or not isinstance(duration, int)
        or started > completed
        or duration != completed - started
    ):
        result.errors.append(f"{event} has invalid timestamps or duration")
    if not isinstance(record.get("bytes"), int) or record["bytes"] < 0:
        result.errors.append(f"{event} has invalid bytes")
    if "schema_version" in record and not isinstance(record["schema_version"], int):
        result.errors.append(f"{event} has invalid schema_version")
    if event == "kv_send" and record.get("batch_id") is not None:
        # 新 schema 显式描述一次物理 producer batch，避免按 request 重复统计。
        batch_fields = {
            "batch_id": str,
            "batch_request_count": int,
            "batch_block_count": int,
            "batch_bytes": int,
        }
        for field_name, expected_type in batch_fields.items():
            if not isinstance(record.get(field_name), expected_type):
                result.errors.append(f"{event} has invalid {field_name}")
                return
        if record["batch_request_count"] <= 0 or record["batch_block_count"] < 0:
            result.errors.append(f"{event} has invalid batch counts")
        if record["batch_block_count"] != record.get("block_count"):
            result.errors.append(
                f"{event} batch_block_count does not match block_count"
            )
        if record["batch_bytes"] != record.get("bytes"):
            result.errors.append(f"{event} batch_bytes does not match bytes")
        if record["batch_request_count"] != record.get("batch_size"):
            result.errors.append(
                f"{event} batch_request_count does not match batch_size"
            )
        if (
            not isinstance(record.get("request_block_count"), int)
            or record["request_block_count"] < 0
        ):
            result.errors.append(f"{event} has invalid request_block_count")
    if record.get("outcome") not in {"success", "failed"}:
        result.errors.append(f"{event} has invalid outcome")


def validate_timing_file(path: Path) -> FileValidationResult:
    """校验一个 `<pid>/<request_id>.jsonl` 文件。"""
    result = FileValidationResult(path=path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        result.errors.append(f"cannot read file: {exc}")
        return result

    steps: list[dict[str, Any]] = []
    finish_records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            _append_error(result, line_number, "blank line")
            continue
        try:
            record = json.loads(line)
        except (TypeError, json.JSONDecodeError) as exc:
            _append_error(result, line_number, f"invalid JSON: {exc}")
            continue
        if not isinstance(record, dict):
            _append_error(result, line_number, "record is not an object")
            continue

        request_id = record.get("request_id")
        role = record.get("role")
        event = record.get("event")
        if not isinstance(request_id, str) or not request_id:
            _append_error(result, line_number, "missing request_id")
            continue
        schema_version = record.get("schema_version")
        if schema_version is not None and (
            not isinstance(schema_version, int)
            or schema_version <= 0
            or schema_version > 8
        ):
            _append_error(result, line_number, "invalid schema_version")
        if result.request_id is None:
            result.request_id = request_id
        elif result.request_id != request_id:
            _append_error(result, line_number, "request_id changed inside file")
        if path.name != expected_timing_filename(request_id):
            _append_error(
                result,
                line_number,
                "filename does not match request_id",
            )
        if role is not None and role not in VALID_ROLES:
            _append_error(result, line_number, "invalid role")
        if result.role is None and role is not None:
            result.role = role
        if role is not None and result.role is not None and result.role != role:
            _append_error(result, line_number, "role changed inside file")

        if event == "step":
            steps.append(record)
            _validate_step(result, record)
        elif event == "finish":
            finish_records.append(record)
        elif event in {"kv_send", "kv_receive"}:
            _validate_transfer_event(result, record)
        else:
            _append_error(result, line_number, f"unknown event {event!r}")
        result.records.append(record)

    # request_id/role 是跨文件关联 P/D 的主键，文件内不允许变化。
    if result.request_id is None:
        result.errors.append("file contains no valid request_id")
    # 兼容早期日志：role 字段加入前没有显式 role。带 KV 事件的旧 PD
    # 日志按事件推断；其余单角色日志按 aggregated 处理。
    if result.role is None and result.request_id is not None:
        event_names = {record.get("event") for record in result.records}
        if "kv_send" in event_names:
            result.role = "prefill"
        elif "kv_receive" in event_names:
            result.role = "decode"
        else:
            result.role = "aggregated"
    summary_finish = bool(
        finish_records and finish_records[0].get("timing_mode") == "summary"
    )
    if not steps and not summary_finish:
        result.errors.append("file contains no step records")

    if steps:
        step_indexes = [record.get("step_index") for record in steps]
        if step_indexes != list(range(len(steps))):
            result.errors.append("step_index sequence is not contiguous from zero")

    phases = [record.get("phase") for record in steps]
    # 三种角色的合法 phase 序列不同：普通服务包含 prefill+decode，
    # PD 的 P 只有一次 prefill，D 只有 decode。
    if result.role == "aggregated":
        expected_phases = ["prefill"] + ["decode"] * (len(steps) - 1)
    elif result.role == "prefill":
        expected_phases = ["prefill"]
    else:
        expected_phases = ["decode"] * len(steps)
    if steps and phases != expected_phases:
        result.errors.append(
            f"phase sequence mismatch: expected {expected_phases}, got {phases}"
        )

    if len(finish_records) > 1:
        result.errors.append("file contains multiple finish records")
    if finish_records:
        # 有 finish 只代表终态记录存在，仍要等所有错误检查完再判定 complete。
        _validate_finish(result, finish_records[0], steps)
        result.complete = not result.errors
    return result


def _batch_consistency_errors(
    file_results: list[FileValidationResult],
) -> dict[str, list[str]]:
    """校验跨 request 文件复制的 forward/Mooncake batch 事件。"""
    errors: dict[str, list[str]] = defaultdict(list)
    forward_batches: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    send_batches: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    # 先把复制到多个 request 文件中的物理 batch 事件还原到同一组。
    for file_result in file_results:
        pid = file_result.path.parent.name
        for record in file_result.records:
            if record.get("event") == "step":
                batch_id = record.get("forward_batch_id")
                if batch_id is not None and not (
                    isinstance(batch_id, str) and batch_id
                ):
                    continue
                # 新日志用显式 ID；旧日志在同一 PID 内用相同 forward 时间戳退化识别。
                key = (
                    (pid, batch_id)
                    if batch_id is not None
                    else (
                        pid,
                        record.get("engine_forward_started_at_ns"),
                        record.get("engine_forward_completed_at_ns"),
                    )
                )
                forward_batches[key].append(record)
            elif record.get("event") == "kv_send":
                batch_id = record.get("batch_id")
                if isinstance(batch_id, str) and batch_id:
                    send_batches[(pid, batch_id)].append(record)

    for key, records in forward_batches.items():
        request_ids = {record.get("request_id") for record in records}
        declared_sizes = {record.get("batch_size") for record in records}
        started_values = {
            record.get("engine_forward_started_at_ns") for record in records
        }
        completed_values = {
            record.get("engine_forward_completed_at_ns") for record in records
        }
        if len(declared_sizes) != 1 or records[0].get("batch_size") != len(records):
            message = (
                f"forward batch {key[-1]} has inconsistent batch_size: "
                f"expected {len(records)}, got {sorted(declared_sizes, key=str)}"
            )
            for request_id in request_ids:
                errors[request_id].append(message)
        # 一次物理 forward 只有一组 start/complete，同 batch 内必须完全一致。
        if len(started_values) != 1 or len(completed_values) != 1:
            message = f"forward batch {key[-1]} has inconsistent timestamps"
            for request_id in request_ids:
                errors[request_id].append(message)
        for field_name in SCHEDULER_SNAPSHOT_FIELDS:
            values = {record.get(field_name) for record in records}
            if len(values) != 1:
                message = f"forward batch {key[-1]} has inconsistent {field_name}"
                for request_id in request_ids:
                    errors[request_id].append(message)

    for (pid, batch_id), records in send_batches.items():
        request_ids = {record.get("request_id") for record in records}
        copied_fields = (
            "duration_ns",
            "batch_request_count",
            "batch_block_count",
            "batch_bytes",
            "bytes",
            "block_count",
        )
        for field_name in copied_fields:
            values = {record.get(field_name) for record in records}
            if len(values) != 1:
                message = f"kv_send batch {batch_id} has inconsistent {field_name}"
                for request_id in request_ids:
                    errors[request_id].append(message)

        # producer batch 会在每个 request 文件中复制一份，事件数必须等于请求数。
        expected_request_count = records[0].get("batch_request_count")
        if expected_request_count != len(records):
            message = (
                f"kv_send batch {batch_id} has inconsistent request count: "
                f"expected {expected_request_count}, got {len(records)}"
            )
            for request_id in request_ids:
                errors[request_id].append(message)

        request_block_counts = [
            record.get("request_block_count")
            for record in records
            if isinstance(record.get("request_block_count"), int)
        ]
        expected_blocks = records[0].get("batch_block_count")
        # 各 request 的 block 数求和后才等于整个 Mooncake batch 的 block 数。
        if (
            len(request_block_counts) == len(records)
            and sum(request_block_counts) != expected_blocks
        ):
            message = (
                f"kv_send batch {batch_id} request blocks do not sum to "
                f"batch_block_count"
            )
            for request_id in request_ids:
                errors[request_id].append(message)

    # pid only用于分组，不进入错误文本，避免暴露部署细节。
    return errors


def validate_timing_root(
    root: Path | str = DEFAULT_TIMING_ROOT,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """校验整个 timing root，并按 request id 聚合 P/D 完整性。"""
    root = Path(root)
    files = iter_timing_files(root)
    if request_id is not None:
        expected_name = expected_timing_filename(request_id)
        files = [path for path in files if path.name == expected_name]

    file_results = [validate_timing_file(path) for path in files]
    grouped: dict[str, list[FileValidationResult]] = defaultdict(list)
    for file_result in file_results:
        if file_result.request_id is not None:
            grouped[file_result.request_id].append(file_result)

    batch_errors = _batch_consistency_errors(file_results)
    batch_consistency_error_count = sum(
        len(messages) for messages in batch_errors.values()
    )

    request_groups: list[dict[str, Any]] = []
    for request, results in sorted(grouped.items()):
        # 同一个 request 在普通服务只有一个角色文件，在 PD 分离为 P/D 两个文件。
        roles = Counter(result.role for result in results)
        errors: list[str] = []
        for role, count in roles.items():
            if role in {"aggregated", "prefill", "decode"} and count > 1:
                errors.append(f"duplicate role {role}: {count} files")

        is_pd = {"prefill", "decode"}.issubset(roles)
        expected_roles = {"prefill", "decode"} if is_pd else {"aggregated"}
        if not expected_roles.issubset(roles):
            errors.append(
                f"missing expected roles: {sorted(expected_roles - set(roles))}"
            )

        events_by_role: dict[str, set[str]] = defaultdict(set)
        for result in results:
            events_by_role[result.role or "unknown"].update(
                record.get("event") for record in result.records
            )
        if is_pd:
            # P/D 请求必须同时具备发送侧和接收侧 KV 观测。
            if "kv_send" not in events_by_role.get("prefill", set()):
                errors.append("PD request has no prefill kv_send event")
            if "kv_receive" not in events_by_role.get("decode", set()):
                errors.append("PD request has no decode kv_receive event")
            decode_steps = [
                record
                for result in results
                if result.role == "decode"
                for record in result.records
                if record.get("event") == "step"
            ]
            receive_events = [
                record
                for result in results
                if result.role == "decode"
                for record in result.records
                if record.get("event") == "kv_receive"
            ]
            if decode_steps and receive_events:
                first_forward_started = decode_steps[0].get(
                    "engine_forward_started_at_ns"
                )
                receive_completed = receive_events[-1].get("completed_at_ns")
                if (
                    isinstance(first_forward_started, int)
                    and isinstance(receive_completed, int)
                    and receive_completed > first_forward_started
                ):
                    errors.append(
                        "PD decode starts before its latest kv_receive completes"
                    )
        errors.extend(batch_errors.pop(request, []))

        complete = all(result.complete for result in results) and not errors
        request_groups.append(
            {
                "request_id": request,
                "status": "complete" if complete else "incomplete",
                "roles": dict(roles),
                "files": [result.to_dict() for result in results],
                "errors": errors,
            }
        )

    status_counts = Counter(result.status for result in file_results)
    invalid_count = status_counts["invalid"]
    invalid_count += batch_consistency_error_count
    if not file_results:
        # 空目录不是 valid，按 incomplete 处理，便于上游脚本区分退出码。
        incomplete_count = 1
    else:
        incomplete_count = sum(
            group["status"] != "complete" for group in request_groups
        )
    return {
        "root": str(root),
        "root_exists": root.exists(),
        "file_count": len(file_results),
        "file_status_counts": dict(sorted(status_counts.items())),
        "request_count": len(request_groups),
        "complete_request_count": sum(
            group["status"] == "complete" for group in request_groups
        ),
        "incomplete_or_invalid_request_count": incomplete_count,
        "invalid_file_count": invalid_count,
        "batch_consistency_error_count": batch_consistency_error_count,
        "requests": request_groups,
        "files": [result.to_dict() for result in file_results],
    }


def _render_text(report: dict[str, Any]) -> str:
    """渲染人可读的校验报告。"""
    lines = [
        f"Timing root: {report['root']} exists={report['root_exists']}",
        (f"Files: {report['file_count']} status={report['file_status_counts']}"),
        (
            f"Requests: {report['request_count']} "
            f"complete={report['complete_request_count']} "
            f"incomplete_or_invalid={report['incomplete_or_invalid_request_count']}"
        ),
    ]
    for group in report["requests"]:
        lines.append(
            f"  {group['request_id']}: {group['status']} roles={group['roles']}"
        )
        for file_report in group["files"]:
            lines.append(
                f"    {file_report['status']}: {file_report['path']} "
                f"records={file_report['record_count']}"
            )
            lines.extend(f"      error: {error}" for error in file_report["errors"])
        lines.extend(f"    error: {error}" for error in group["errors"])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 validate_logs 命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timing-root",
        type=Path,
        default=DEFAULT_TIMING_ROOT,
        help="timing root containing <pid>/<request>.jsonl files",
    )
    parser.add_argument("--request-id", help="only validate one request id")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：invalid 返回 2，incomplete/空目录返回 1。"""
    args = build_arg_parser().parse_args(argv)
    report = validate_timing_root(args.timing_root, request_id=args.request_id)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render_text(report))

    if report["invalid_file_count"]:
        return 2
    if (
        report["incomplete_or_invalid_request_count"] or not report["file_count"]
    ) and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
