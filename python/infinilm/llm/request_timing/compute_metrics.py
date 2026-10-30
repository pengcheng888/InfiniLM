"""从 InfiniLM timing JSONL 计算请求级和节点级指标。"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from infinilm.llm.request_timing.validate_logs import (
    DEFAULT_TIMING_ROOT,
    expected_timing_filename,
    iter_timing_files,
    validate_timing_root,
)

ENTRANCE_METRIC_FIELDS = (
    ("server_to_body_read_ms", "server_received_at_ns", "body_read_completed_at_ns"),
    ("body_read_to_json_parse_ms", "body_read_completed_at_ns", "json_parsed_at_ns"),
    ("json_parse_to_route_prepare_ms", "json_parsed_at_ns", "route_prepared_at_ns"),
    (
        "route_prepare_to_handler_start_ms",
        "route_prepared_at_ns",
        "handler_started_at_ns",
    ),
    (
        "handler_start_to_preprocessing_start_ms",
        "handler_started_at_ns",
        "preprocessing_started_at_ns",
    ),
    (
        "preprocessing_to_chat_template_ms",
        "preprocessing_started_at_ns",
        "chat_template_completed_at_ns",
    ),
    (
        "chat_template_to_multimodal_resolve_ms",
        "chat_template_completed_at_ns",
        "multimodal_resolved_at_ns",
    ),
    (
        "multimodal_to_tokenization_ms",
        "multimodal_resolved_at_ns",
        "tokenization_completed_at_ns",
    ),
    (
        "tokenization_to_prompt_token_ids_ms",
        "tokenization_completed_at_ns",
        "prompt_token_ids_ready_at_ns",
    ),
    (
        "prompt_token_ids_to_request_created_ms",
        "prompt_token_ids_ready_at_ns",
        "request_created_at_ns",
    ),
)


def _milliseconds(value_ns: Any) -> float | None:
    """把纳秒转毫秒；无效观测值保持 None，不伪装成 0。"""
    return None if not isinstance(value_ns, (int, float)) else value_ns / 1_000_000


def _elapsedMilliseconds(started_at_ns: Any, completed_at_ns: Any) -> float | None:
    """在同一个本地时钟域内计算 completed-started 毫秒数。"""
    if not isinstance(started_at_ns, (int, float)) or not isinstance(
        completed_at_ns, (int, float)
    ):
        return None
    return _milliseconds(completed_at_ns - started_at_ns)


def _percentile(values: list[float], percentile: float) -> float | None:
    """线性插值分位数，兼容小样本 p95。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _distribution(values) -> dict[str, float | None]:
    """输出 count/均值/最小值/最大值/分位数分布。"""
    values = list(values)
    return {
        "count": float(len(values)),
        "avg_ms": mean(values) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "p50_ms": median(values) if values else None,
        "p95_ms": _percentile(values, 0.95),
    }


def _throughput(bytes_count: int | None, duration_ns: int | None) -> float | None:
    """按传输字节数和本地持续时长计算 GB/s。"""
    if not bytes_count or not duration_ns or duration_ns <= 0:
        return None
    return bytes_count / (duration_ns / 1_000_000_000) / (1000**3)


def _format_ms(value: float | None) -> str:
    """文本报告中的毫秒格式；缺失值显示为短横线。"""
    return "-" if value is None else f"{value:.3f}ms"


def _integer_distribution(values) -> dict[str, int | float | None]:
    """输出整数队列深度的 count/均值/最小/最大/分位数分布。"""
    values = list(values)
    return {
        "count": len(values),
        "avg": mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _sum_numeric(records: list[dict[str, Any]], key: str) -> int | float:
    """安全求和数值字段；malformed 记录交给 validator 报错。"""
    return sum(
        value
        for record in records
        if isinstance((value := record.get(key)), (int, float))
    )


def _record_duration_ms(record: dict[str, Any]) -> float:
    """安全读取事件 duration；invalid 值按 0 进入分布，由 validator 报错。"""
    duration_ns = record.get("duration_ns")
    return _milliseconds(duration_ns) if isinstance(duration_ns, (int, float)) else 0.0


def _load_request_groups(
    root: Path,
    request_id: str | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """读取所有日志并按 request_id 聚合，同时补齐旧日志角色。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    for path in iter_timing_files(root):
        # recorder 会清洗 unsafe/过长的 request id；过滤时必须使用同样的规则。
        if request_id and path.name != expected_timing_filename(request_id):
            continue
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                errors.append(f"{path}:{line_number}: invalid JSON")
                continue
            if isinstance(record, dict) and isinstance(record.get("request_id"), str):
                # 兼容 schema v1：没有 role 时按 KV 事件推断 P/D 角色。
                if record.get("role") is None:
                    record = dict(record)
                    record["role"] = (
                        "prefill"
                        if record.get("event") == "kv_send"
                        else "decode"
                        if record.get("event") == "kv_receive"
                        else "aggregated"
                    )
                record["_timing_pid"] = path.parent.name
                records.append(record)
        if records:
            groups[records[0]["request_id"]].extend(records)
    return groups, errors


def _role_metrics(records: list[dict[str, Any]], role: str) -> dict[str, Any]:
    """计算一个 request 在指定 worker 角色上的本地执行指标。"""
    role_records = [record for record in records if record.get("role") == role]
    steps = [record for record in role_records if record.get("event") == "step"]
    finish = next(
        (record for record in role_records if record.get("event") == "finish"),
        None,
    )
    summary_finish = bool(
        finish
        and finish.get("timing_mode") == "summary"
        and finish.get("schema_version") in {7, 8}
    )
    if not steps:
        if not summary_finish:
            return {"role": role, "status": "missing", "step_count": 0}

        first = dict(finish)
        first["scheduler_polled_at_ns"] = finish.get("first_scheduler_polled_at_ns")
        first["engine_forward_started_at_ns"] = finish.get(
            "first_engine_forward_started_at_ns"
        )
        first["engine_forward_completed_at_ns"] = finish.get(
            "first_engine_forward_completed_at_ns"
        )
        first_forward_ns = (
            finish.get("first_engine_forward_completed_at_ns")
            - finish.get("first_engine_forward_started_at_ns")
            if isinstance(finish.get("first_engine_forward_completed_at_ns"), int)
            and isinstance(finish.get("first_engine_forward_started_at_ns"), int)
            else None
        )
        engine_total_ns = finish.get("engine_forward_total_ns")
        if engine_total_ns is None:
            engine_total_ns = finish.get("engine_forward_total_estimate_ns")
        decode_total_ns = (
            engine_total_ns
            if role == "decode" and isinstance(engine_total_ns, int)
            else engine_total_ns - first_forward_ns
            if isinstance(engine_total_ns, int) and isinstance(first_forward_ns, int)
            else None
        )
        step_count = finish.get("steps") if isinstance(finish.get("steps"), int) else 0
        decode_step_count = (
            finish.get("decode_step_count")
            if isinstance(finish.get("decode_step_count"), int)
            else max(step_count - 1, 0)
        )
        first_durations = {
            "queue": finish.get("queue_ns"),
            "scheduler": finish.get("scheduler_ns"),
            "engine_forward": first_forward_ns,
        }
        entrance_metrics = {
            metric_name: _elapsedMilliseconds(
                first.get(start_field), first.get(end_field)
            )
            for metric_name, start_field, end_field in ENTRANCE_METRIC_FIELDS
        }
        return {
            "role": role,
            "status": "complete",
            "timing_mode": "summary",
            "sample_interval": finish.get("sample_interval"),
            "engine_forward_estimated": finish.get("schema_version") == 8,
            "sampled_decode_steps": finish.get("sampled_decode_step_count"),
            "step_count": step_count,
            "batch_size_counts": {},
            "server_to_request_created_ms": _elapsedMilliseconds(
                first.get("server_received_at_ns"),
                first.get("request_created_at_ns"),
            ),
            **entrance_metrics,
            "request_created_to_scheduler_enqueue_ms": _elapsedMilliseconds(
                first.get("request_created_at_ns"),
                first.get("scheduler_enqueued_at_ns"),
            ),
            "server_to_scheduler_enqueue_ms": _elapsedMilliseconds(
                first.get("server_received_at_ns"),
                first.get("scheduler_enqueued_at_ns"),
            ),
            "scheduler_enqueue_to_polled_ms": _elapsedMilliseconds(
                first.get("scheduler_enqueued_at_ns"),
                first.get("scheduler_polled_at_ns"),
            ),
            "scheduler_batch_build_ms": _elapsedMilliseconds(
                first.get("scheduler_polled_at_ns"),
                first.get("scheduler_batch_ready_at_ns"),
            ),
            "scheduler_batch_ready_to_engine_forward_start_ms": (
                _elapsedMilliseconds(
                    first.get("scheduler_batch_ready_at_ns"),
                    first.get("engine_forward_started_at_ns"),
                )
            ),
            "server_to_engine_forward_start_ms": _elapsedMilliseconds(
                first.get("server_received_at_ns"),
                first.get("engine_forward_started_at_ns"),
            ),
            "first_engine_forward_ms": _milliseconds(first_forward_ns),
            "scheduler_ms": _milliseconds(first_durations.get("scheduler")),
            "first_token_local_ms": _elapsedMilliseconds(
                first.get("server_received_at_ns"),
                first.get("engine_forward_completed_at_ns"),
            ),
            "local_span_ms": _elapsedMilliseconds(
                first.get("server_received_at_ns"),
                finish.get("last_engine_forward_completed_at_ns"),
            ),
            "engine_forward_total_ms": _milliseconds(engine_total_ns),
            "prefill_forward_total_ms": _milliseconds(first_forward_ns),
            "decode_forward_total_ms": _milliseconds(decode_total_ns),
            "decode_forward_avg_ms": _milliseconds(
                decode_total_ns / decode_step_count
                if isinstance(decode_total_ns, int) and decode_step_count > 0
                else None
            ),
            "finish_reason": finish.get("finish_reason"),
        }

    # 文件按 PID/request 读取；同一角色内保持 step 顺序。
    first = steps[0]
    last = steps[-1]
    forwards = [
        record["durations_ns"]["engine_forward"]
        for record in steps
        if isinstance(record.get("durations_ns"), dict)
        and isinstance(record["durations_ns"].get("engine_forward"), (int, float))
    ]
    decode_forwards = [
        record["durations_ns"]["engine_forward"]
        for record in steps
        if record.get("phase") == "decode"
        and isinstance(record.get("durations_ns"), dict)
        and isinstance(record["durations_ns"].get("engine_forward"), (int, float))
    ]
    prefill_forwards = [
        record["durations_ns"]["engine_forward"]
        for record in steps
        if record.get("phase") == "prefill"
        and isinstance(record.get("durations_ns"), dict)
        and isinstance(record["durations_ns"].get("engine_forward"), (int, float))
    ]
    batch_sizes = Counter(record.get("batch_size") for record in steps)
    first_durations = first.get("durations_ns")
    first_durations = first_durations if isinstance(first_durations, dict) else {}
    entrance_metrics = {
        metric_name: _elapsedMilliseconds(first.get(start_field), first.get(end_field))
        for metric_name, start_field, end_field in ENTRANCE_METRIC_FIELDS
    }
    return {
        "role": role,
        "status": "complete" if finish else "incomplete",
        "step_count": len(steps),
        "batch_size_counts": {
            str(key): value
            for key, value in sorted(batch_sizes.items(), key=lambda item: str(item[0]))
        },
        # 首次调度路径使用首条 step 的原始时间戳，避免 queue 混合入口
        # 预处理和 scheduler 队列等待两个不同阶段。
        "server_to_request_created_ms": _elapsedMilliseconds(
            first.get("server_received_at_ns"),
            first.get("request_created_at_ns"),
        ),
        **entrance_metrics,
        "request_created_to_scheduler_enqueue_ms": _elapsedMilliseconds(
            first.get("request_created_at_ns"),
            first.get("scheduler_enqueued_at_ns"),
        ),
        "server_to_scheduler_enqueue_ms": _elapsedMilliseconds(
            first.get("server_received_at_ns"),
            first.get("scheduler_enqueued_at_ns"),
        ),
        "scheduler_enqueue_to_polled_ms": _elapsedMilliseconds(
            first.get("scheduler_enqueued_at_ns"),
            first.get("scheduler_polled_at_ns"),
        ),
        "scheduler_batch_build_ms": _elapsedMilliseconds(
            first.get("scheduler_polled_at_ns"),
            first.get("scheduler_batch_ready_at_ns"),
        ),
        "scheduler_batch_ready_to_engine_forward_start_ms": _elapsedMilliseconds(
            first.get("scheduler_batch_ready_at_ns"),
            first.get("engine_forward_started_at_ns"),
        ),
        "server_to_engine_forward_start_ms": _elapsedMilliseconds(
            first.get("server_received_at_ns"),
            first.get("engine_forward_started_at_ns"),
        ),
        "first_engine_forward_ms": _elapsedMilliseconds(
            first.get("engine_forward_started_at_ns"),
            first.get("engine_forward_completed_at_ns"),
        ),
        "scheduler_ms": _milliseconds(first_durations.get("scheduler")),
        "scheduler_waiting_queue_depth_before": first.get(
            "scheduler_waiting_queue_depth_before"
        ),
        "scheduler_running_queue_depth_before": first.get(
            "scheduler_running_queue_depth_before"
        ),
        "scheduler_remote_kv_requests_before": first.get(
            "scheduler_remote_kv_requests_before"
        ),
        "scheduler_waiting_queue_depth_after": first.get(
            "scheduler_waiting_queue_depth_after"
        ),
        "scheduler_running_queue_depth_after": first.get(
            "scheduler_running_queue_depth_after"
        ),
        "scheduler_remote_kv_requests_after": first.get(
            "scheduler_remote_kv_requests_after"
        ),
        "first_token_local_ms": _elapsedMilliseconds(
            first.get("server_received_at_ns"),
            first.get("engine_forward_completed_at_ns"),
        ),
        "local_span_ms": _elapsedMilliseconds(
            first.get("server_received_at_ns"),
            last.get("engine_forward_completed_at_ns"),
        ),
        "engine_forward_total_ms": _milliseconds(sum(forwards)),
        # request 视角指标：同一物理 batch 内的多个 request 会共享同一段
        # forward duration，不能跨 request 相加当作 GPU busy 时间。
        "prefill_forward_total_ms": _milliseconds(sum(prefill_forwards)),
        "decode_forward_total_ms": _milliseconds(sum(decode_forwards)),
        "decode_forward_avg_ms": _milliseconds(
            mean(decode_forwards) if decode_forwards else None
        ),
        "finish_reason": finish.get("finish_reason") if finish else None,
    }


def _scheduler_snapshot_summary(
    records: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    """按物理 scheduler batch 去重统计完整调度状态快照。"""
    steps = [
        record
        for record in records
        if record.get("event") == "step" and record.get("role") == role
    ]
    batches: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in steps:
        pid = record.get("_timing_pid", "unknown")
        batch_id = record.get("forward_batch_id")
        key = (
            (pid, batch_id)
            if batch_id is not None
            else (
                pid,
                record.get("engine_forward_started_at_ns"),
                record.get("engine_forward_completed_at_ns"),
            )
        )
        batches[key].append(record)

    snapshots = [records_group[0] for records_group in batches.values()]
    snapshot_fields = (
        "scheduler_waiting_queue_depth_before",
        "scheduler_running_queue_depth_before",
        "scheduler_remote_kv_requests_before",
        "scheduler_waiting_queue_depth_after",
        "scheduler_running_queue_depth_after",
        "scheduler_remote_kv_requests_after",
    )
    batch_sizes = Counter(len(records_group) for records_group in batches.values())

    def field_values(field_name: str) -> list[int]:
        return [
            value
            for record in snapshots
            if isinstance((value := record.get(field_name)), int)
        ]

    active_before = [
        waiting + running + remote
        for record in snapshots
        if all(
            isinstance(record.get(field_name), int)
            for field_name in snapshot_fields[:3]
        )
        for waiting, running, remote in (
            (
                record["scheduler_waiting_queue_depth_before"],
                record["scheduler_running_queue_depth_before"],
                record["scheduler_remote_kv_requests_before"],
            ),
        )
    ]
    active_after = [
        waiting + running + remote + len(records_group)
        for records_group in batches.values()
        if records_group
        and all(
            isinstance(records_group[0].get(field_name), int)
            for field_name in snapshot_fields[3:]
        )
        for waiting, running, remote in (
            (
                records_group[0]["scheduler_waiting_queue_depth_after"],
                records_group[0]["scheduler_running_queue_depth_after"],
                records_group[0]["scheduler_remote_kv_requests_after"],
            ),
        )
    ]
    return {
        "snapshot_count": len(snapshots),
        "batch_size_counts": {
            str(key): value
            for key, value in sorted(batch_sizes.items(), key=lambda item: str(item[0]))
        },
        **{
            field_name: _integer_distribution(field_values(field_name))
            for field_name in snapshot_fields
        },
        "active_backlog_before": _integer_distribution(active_before),
        "active_backlog_after_including_scheduled": _integer_distribution(active_after),
    }


def _event_metrics(
    records: list[dict[str, Any]],
    event: str,
) -> dict[str, Any]:
    """提取请求级 KV 事件；同一事件多条记录时暂按最后一条处理。"""
    matching = [record for record in records if record.get("event") == event]
    # 当前不考虑传输异常重试；若历史日志存在多条记录，取最后一条作为终态。
    selected = matching[-1] if matching else {}
    duration_ns = selected.get("duration_ns")
    batch_bytes = selected.get("batch_bytes", selected.get("bytes"))
    is_batch = event == "kv_send" and selected.get("batch_id") is not None
    throughput_bytes = batch_bytes if is_batch else selected.get("bytes")
    return {
        "present": bool(matching),
        "outcome": selected.get("outcome"),
        "duration_ms": _milliseconds(duration_ns),
        "bytes": selected.get("bytes"),
        "block_count": selected.get("block_count"),
        "batch_id": selected.get("batch_id"),
        "batch_request_count": selected.get("batch_request_count"),
        "batch_block_count": selected.get("batch_block_count"),
        "batch_bytes": batch_bytes,
        "request_block_count": selected.get(
            "request_block_count",
            selected.get("block_count"),
        ),
        "throughput_gbps": _throughput(throughput_bytes, duration_ns),
        "scope": selected.get("scope"),
        "cross_node_comparable": selected.get("cross_node_comparable"),
    }


def _request_metrics(request_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """把普通服务和 PD 服务的 request 记录转换成统一指标结构。"""
    roles = {record.get("role") for record in records}
    # 同一 request 同时出现 prefill/decode 角色时，判定为 PD 分离请求。
    is_pd = {"prefill", "decode"}.issubset(roles)
    result: dict[str, Any] = {
        "request_id": request_id,
        "mode": "pd" if is_pd else "aggregated",
        "roles": sorted(str(role) for role in roles if role),
    }
    if is_pd:
        prefill = _role_metrics(records, "prefill")
        decode = _role_metrics(records, "decode")
        kv_send = _event_metrics(records, "kv_send")
        kv_receive = _event_metrics(records, "kv_receive")
        decode_steps = [
            record
            for record in records
            if record.get("role") == "decode" and record.get("event") == "step"
        ]
        first_decode = decode_steps[0] if decode_steps else {}
        if not first_decode:
            summary_decode_finish = next(
                (
                    record
                    for record in records
                    if record.get("role") == "decode"
                    and record.get("event") == "finish"
                    and record.get("timing_mode") == "summary"
                ),
                None,
            )
            if summary_decode_finish is not None:
                first_decode = {
                    "engine_forward_started_at_ns": (
                        summary_decode_finish.get("first_engine_forward_started_at_ns")
                    )
                }
        receive_records = [
            record for record in records if record.get("event") == "kv_receive"
        ]
        receive_completed = (
            receive_records[-1].get("completed_at_ns") if receive_records else None
        )
        # KV receive 和 decode step 都在 D 节点，二者可以使用 D 本地 monotonic 差值。
        decode_wait_after_kv_ns = (
            first_decode.get("engine_forward_started_at_ns") - receive_completed
            if first_decode
            and isinstance(first_decode.get("engine_forward_started_at_ns"), int)
            and isinstance(receive_completed, int)
            else None
        )
        result.update(
            {
                "prefill": prefill,
                "decode": decode,
                "kv_send": kv_send,
                "kv_receive": kv_receive,
                "p_compute_ms": prefill.get("engine_forward_total_ms"),
                "d_compute_ms": decode.get("engine_forward_total_ms"),
                "mooncake_transfer_ms": kv_send.get("duration_ms"),
                "d_kv_readiness_ms": kv_receive.get("duration_ms"),
                "decode_start_after_kv_ready_ms": _milliseconds(
                    decode_wait_after_kv_ns
                ),
                "cross_node_comparable": False,
            }
        )
    else:
        aggregated = _role_metrics(records, "aggregated")
        result.update(
            {
                "aggregated": aggregated,
                "ttft_local_ms": aggregated.get("first_token_local_ms"),
                "prefill_compute_ms": aggregated.get("first_engine_forward_ms"),
                "decode_compute_ms": aggregated.get("decode_forward_total_ms"),
            }
        )
    return result


def _physical_forward_summary(
    records: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    """按 forward_batch_id 去重 request step，统计物理 forward。"""
    steps = [
        record
        for record in records
        if record.get("event") == "step" and record.get("role") == role
    ]
    batches: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in steps:
        pid = record.get("_timing_pid", "unknown")
        batch_id = record.get("forward_batch_id")
        key = (
            (pid, batch_id)
            if batch_id is not None
            else (
                pid,
                record.get("engine_forward_started_at_ns"),
                record.get("engine_forward_completed_at_ns"),
            )
        )
        batches[key].append(record)

    # scheduler 会给 batch 中每个 request 复制同一组 forward 时间；这里还原为一次物理调用。
    physical_records: list[dict[str, Any]] = []
    for batch_steps in batches.values():
        first = batch_steps[0]
        batch_pid = first.get("_timing_pid", "unknown")
        first_durations = first.get("durations_ns")
        forward_duration_ns = (
            first_durations.get("engine_forward")
            if isinstance(first_durations, dict)
            else None
        )
        physical_records.append(
            {
                "batch_size": len(batch_steps),
                "declared_batch_size": first.get("batch_size"),
                "duration_ms": _milliseconds(forward_duration_ns),
                "pid": batch_pid,
                "server_received_at_ns": first.get("server_received_at_ns"),
                "engine_forward_completed_at_ns": first.get(
                    "engine_forward_completed_at_ns"
                ),
            }
        )

    batch_sizes = Counter(item["batch_size"] for item in physical_records)
    durations = [
        item["duration_ms"]
        for item in physical_records
        if isinstance(item["duration_ms"], (int, float))
    ]
    busy_ms = sum(durations)
    # busy 时长是本地 duration，可以跨进程求和；wall 时间不能跨 monotonic 域求差。
    pids = sorted({record["_timing_pid"] for record in steps}, key=str)
    process_summaries: list[dict[str, Any]] = []
    for pid in pids:
        pid_steps = [record for record in steps if record["_timing_pid"] == pid]
        pid_physical = [record for record in physical_records if record["pid"] == pid]
        pid_busy_ms = sum(
            record["duration_ms"]
            for record in pid_physical
            if isinstance(record["duration_ms"], (int, float))
        )
        timestamps = [
            record
            for record in pid_steps
            if isinstance(record.get("server_received_at_ns"), (int, float))
            and isinstance(record.get("engine_forward_completed_at_ns"), (int, float))
        ]
        local_wall_ms = (
            (
                max(record["engine_forward_completed_at_ns"] for record in timestamps)
                - min(record["server_received_at_ns"] for record in timestamps)
            )
            / 1_000_000
            if timestamps
            else 0.0
        )
        process_summaries.append(
            {
                "pid": pid,
                "request_step_records": len(pid_steps),
                "physical_forward_count": len(pid_physical),
                "forward_busy_ms": pid_busy_ms,
                "local_wall_ms": local_wall_ms,
                "forward_busy_ratio": (
                    pid_busy_ms / local_wall_ms if local_wall_ms > 0 else None
                ),
            }
        )

    # 不同 PID 可能位于不同机器；monotonic 时间差只能在同一 PID 内计算。
    single_process = process_summaries[0] if len(process_summaries) == 1 else None
    return {
        "request_step_records": len(steps),
        "physical_forward_count": len(physical_records),
        "batch_size_counts": {
            str(key): value
            for key, value in sorted(batch_sizes.items(), key=lambda item: str(item[0]))
        },
        "forward_duration_ms": _distribution(durations),
        "forward_busy_ms": busy_ms,
        "local_wall_ms": (single_process["local_wall_ms"] if single_process else None),
        "process_count": len(process_summaries),
        "processes": process_summaries,
        "forward_busy_ratio": (
            single_process["forward_busy_ratio"] if single_process else None
        ),
        "scheduler_snapshots": _scheduler_snapshot_summary(records, role),
    }


def _merged_interval_union(records: list[dict[str, Any]]) -> float:
    """返回并发 transfer interval 的并集时长，避免重叠时间重复计算。"""
    total_ms = 0.0
    pids = sorted({record.get("_timing_pid", "unknown") for record in records}, key=str)
    for pid in pids:
        intervals = sorted(
            (
                record["started_at_ns"],
                record["completed_at_ns"],
            )
            for record in records
            if record.get("_timing_pid", "unknown") == pid
            and isinstance(record.get("started_at_ns"), (int, float))
            and isinstance(record.get("completed_at_ns"), (int, float))
        )
        merged: list[list[Any]] = []
        # interval 已按开始时间排序；只合并本 PID 内重叠的传输区间。
        for started, completed in intervals:
            if not merged or started > merged[-1][1]:
                merged.append([started, completed])
            else:
                merged[-1][1] = max(merged[-1][1], completed)
        # 每个 PID 的 monotonic epoch 独立；先做进程内 union，再累加安全时长。
        total_ms += (
            sum(completed - started for started, completed in merged) / 1_000_000
        )
    return total_ms


def _transfer_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """分别输出 request-weighted 和去重 batch-weighted 传输指标。"""
    send_events = [record for record in records if record.get("event") == "kv_send"]
    receive_events = [
        record for record in records if record.get("event") == "kv_receive"
    ]
    unique_send: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in send_events:
        pid = record.get("_timing_pid", "unknown")
        batch_id = record.get("batch_id")
        key = (
            (pid, batch_id)
            if batch_id is not None
            else (
                pid,
                record.get("started_at_ns"),
                record.get("completed_at_ns"),
            )
        )
        unique_send[key] = record

    # P 侧 batch 字段会被复制到每个 request 文件，总量统计必须先去重。
    unique_send_records = list(unique_send.values())
    send_bytes = sum(
        record.get("batch_bytes", record.get("bytes"))
        if isinstance(record.get("batch_bytes", record.get("bytes")), (int, float))
        else 0
        for record in unique_send_records
    )
    send_busy_ms = _merged_interval_union(unique_send_records)
    receive_bytes = _sum_numeric(receive_events, "bytes")
    receive_busy_ms = _merged_interval_union(receive_events)
    return {
        "kv_send": {
            "request_event_count": len(send_events),
            "unique_batch_count": len(unique_send_records),
            "request_weighted_duration_ms": _distribution(
                _record_duration_ms(record) for record in send_events
            ),
            "batch_weighted_duration_ms": _distribution(
                _record_duration_ms(record) for record in unique_send_records
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
            "request_event_count": len(receive_events),
            "request_weighted_duration_ms": _distribution(
                _record_duration_ms(record) for record in receive_events
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


def _summary(
    request_metrics: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总普通服务、PD 服务、物理 forward 和 Mooncake 传输指标。"""
    aggregated = [item for item in request_metrics if item["mode"] == "aggregated"]
    pd = [item for item in request_metrics if item["mode"] == "pd"]

    def values(items: list[dict[str, Any]], key: str) -> list[float]:
        """只把有效数值送入分布统计，缺失项不影响其它请求。"""
        return [item[key] for item in items if isinstance(item.get(key), (int, float))]

    def role_values(items: list[dict[str, Any]], role: str, key: str) -> list[float]:
        """读取 request 内嵌角色指标，用于入口路径的分布统计。"""
        return [
            item[role][key]
            for item in items
            if isinstance(item.get(role, {}).get(key), (int, float))
        ]

    summary = {
        "request_count": len(request_metrics),
        "aggregated_request_count": len(aggregated),
        "pd_request_count": len(pd),
        "aggregated": {
            "server_to_request_created_ms": _distribution(
                role_values(aggregated, "aggregated", "server_to_request_created_ms")
            ),
            **{
                metric_name: _distribution(
                    role_values(aggregated, "aggregated", metric_name)
                )
                for metric_name, _, _ in ENTRANCE_METRIC_FIELDS
            },
            "request_created_to_scheduler_enqueue_ms": _distribution(
                role_values(
                    aggregated,
                    "aggregated",
                    "request_created_to_scheduler_enqueue_ms",
                )
            ),
            "ttft_local_ms": _distribution(values(aggregated, "ttft_local_ms")),
            "prefill_compute_ms": _distribution(
                values(aggregated, "prefill_compute_ms")
            ),
            "decode_compute_ms": _distribution(values(aggregated, "decode_compute_ms")),
        },
        "pd": {
            **{
                f"p_{metric_name}": _distribution(
                    role_values(pd, "prefill", metric_name)
                )
                for metric_name, _, _ in ENTRANCE_METRIC_FIELDS
            },
            **{
                f"d_{metric_name}": _distribution(
                    role_values(pd, "decode", metric_name)
                )
                for metric_name, _, _ in ENTRANCE_METRIC_FIELDS
            },
            "p_server_to_request_created_ms": _distribution(
                role_values(pd, "prefill", "server_to_request_created_ms")
            ),
            "d_server_to_request_created_ms": _distribution(
                role_values(pd, "decode", "server_to_request_created_ms")
            ),
            "p_compute_ms": _distribution(values(pd, "p_compute_ms")),
            "d_compute_ms": _distribution(values(pd, "d_compute_ms")),
            "mooncake_transfer_ms": _distribution(values(pd, "mooncake_transfer_ms")),
            "d_kv_readiness_ms": _distribution(values(pd, "d_kv_readiness_ms")),
            "decode_start_after_kv_ready_ms": _distribution(
                values(pd, "decode_start_after_kv_ready_ms")
            ),
        },
    }
    if any(item["mode"] == "aggregated" for item in request_metrics):
        summary["aggregated_node"] = _physical_forward_summary(records, "aggregated")
    if any(item["mode"] == "pd" for item in request_metrics):
        # P/D 的节点级统计分别计算；Mooncake 传输单独输出。
        summary.update(
            {
                "prefill_node": _physical_forward_summary(records, "prefill"),
                "decode_node": _physical_forward_summary(records, "decode"),
                "mooncake_transfer": _transfer_summary(records),
            }
        )
    return summary


def compute_timing_metrics(
    root: Path | str = DEFAULT_TIMING_ROOT,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """计算请求级指标和汇总指标，并附带完整校验报告。"""
    root = Path(root)
    groups, parse_errors = _load_request_groups(root, request_id)
    all_records = [record for records in groups.values() for record in records]
    metrics = [
        _request_metrics(request, records)
        for request, records in sorted(groups.items())
    ]
    return {
        "root": str(root),
        "root_exists": root.exists(),
        "parse_errors": parse_errors,
        "validation": validate_timing_root(root, request_id=request_id),
        "summary": _summary(metrics, all_records),
        "requests": metrics,
    }


def _render_text(report: dict[str, Any]) -> str:
    """渲染人可读的指标报告。"""
    lines = [
        f"Timing root: {report['root']} exists={report['root_exists']}",
        f"Parse errors: {len(report['parse_errors'])}",
        json.dumps(report["summary"], ensure_ascii=False, indent=2),
        "Requests:",
    ]
    for request in report["requests"]:
        if request["mode"] == "pd":
            lines.append(
                f"  {request['request_id']} PD: "
                f"P={_format_ms(request.get('p_compute_ms'))} "
                f"transfer={_format_ms(request.get('mooncake_transfer_ms'))} "
                f"D={_format_ms(request.get('d_compute_ms'))} "
                f"D_kv_ready={_format_ms(request.get('d_kv_readiness_ms'))}"
            )
        else:
            lines.append(
                f"  {request['request_id']} aggregated: "
                f"TTFT={_format_ms(request.get('ttft_local_ms'))} "
                f"prefill={_format_ms(request.get('prefill_compute_ms'))} "
                f"decode={_format_ms(request.get('decode_compute_ms'))}"
            )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 compute_metrics 命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timing-root",
        type=Path,
        default=DEFAULT_TIMING_ROOT,
        help="timing root containing <pid>/<request>.jsonl files",
    )
    parser.add_argument("--request-id", help="only compute metrics for one request")
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="still exit 0 when timing validation reports invalid files",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="still exit 0 when requests have not reached finish",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口；JSON 解析错误或 invalid 数据返回 2。"""
    args = build_arg_parser().parse_args(argv)
    report = compute_timing_metrics(args.timing_root, request_id=args.request_id)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render_text(report))
    if report["parse_errors"]:
        return 2
    validation = report["validation"]
    if validation["invalid_file_count"] and not args.allow_invalid:
        return 2
    if (
        validation["incomplete_or_invalid_request_count"]
        or not validation["file_count"]
    ) and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
