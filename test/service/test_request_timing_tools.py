import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from infinilm.llm.request_timing.compute_metrics import (
    _merged_interval_union,
    _physical_forward_summary,
    compute_timing_metrics,
)
from infinilm.llm.request_timing.validate_logs import (
    expected_timing_filename,
    validate_timing_root,
)


class RequestTimerToolsTest(unittest.TestCase):
    @staticmethod
    def _write(root: Path, pid: str, request_id: str, records: list[dict]):
        """按线上目录结构写一个测试 timing JSONL 文件。"""
        directory = root / pid
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{request_id}.jsonl").write_text(
            "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in records),
            encoding="utf-8",
        )

    @staticmethod
    def _step(role, phase, index, start=100):
        """构造时间戳和 durations 自洽的一条 step 记录。"""
        timestamps = [100, 110, 120, 130, 150, 180]
        return {
            "schema_version": 4,
            "event": "step",
            "request_id": "request-1",
            "role": role,
            "step_index": index,
            "phase": phase,
            "batch_size": 1,
            "server_received_at_ns": timestamps[0],
            "scheduler_enqueued_at_ns": timestamps[1],
            "scheduler_polled_at_ns": timestamps[2],
            "scheduler_batch_ready_at_ns": timestamps[3],
            "scheduler_waiting_queue_depth_before": 4,
            "scheduler_running_queue_depth_before": 2,
            "scheduler_remote_kv_requests_before": 1,
            "scheduler_waiting_queue_depth_after": 3,
            "scheduler_running_queue_depth_after": 1,
            "scheduler_remote_kv_requests_after": 0,
            "engine_forward_started_at_ns": timestamps[4],
            "engine_forward_completed_at_ns": timestamps[5],
            "durations_ns": {
                "queue": 20,
                "scheduler": 10,
                "scheduler_batch_ready_to_engine_forward_start": 20,
                "engine_forward": 30,
            },
        }

    @staticmethod
    def _finish(role, steps: int, forward_ns: int):
        """构造与 step 明细匹配的 finish 汇总。"""
        return {
            "schema_version": 4,
            "event": "finish",
            "role": role,
            "finish_reason": "length",
            "request_id": "request-1",
            "steps": steps,
            "server_received_at_ns": 100,
            "first_scheduler_polled_at_ns": 120,
            "first_engine_forward_started_at_ns": 150,
            "last_engine_forward_completed_at_ns": 180,
            "queue_ns": 20,
            "scheduler_ns": 10,
            "scheduler_batch_ready_to_engine_forward_start_ns": 20,
            "engine_forward_total_ns": forward_ns,
        }

    @staticmethod
    def _transfer(event: str, duration_ns: int):
        """构造 P 发送或 D 接收侧的 KV 传输事件。"""
        role = "prefill" if event == "kv_send" else "decode"
        scope = (
            "producer_batch_transfer"
            if event == "kv_send"
            else "decode_local_kv_readiness"
        )
        return {
            "event": event,
            "request_id": "request-1",
            "role": role,
            "scope": scope,
            "clock_domain": "local-monotonic",
            "cross_node_comparable": False,
            "started_at_ns": 200,
            "completed_at_ns": 200 + duration_ns,
            "duration_ns": duration_ns,
            "block_count": 1,
            "bytes": 1024,
            "batch_size": 1,
            "outcome": "success",
        }

    @staticmethod
    def _with_send_batch_fields(event: dict) -> dict:
        """给 kv_send 事件补充 producer batch 语义字段。"""
        event.update(
            {
                "batch_id": "kv-send-batch-1",
                "batch_request_count": 1,
                "batch_block_count": event["block_count"],
                "batch_bytes": event["bytes"],
                "request_block_count": event["block_count"],
            }
        )
        return event

    def test_physical_forward_and_transfer_union_are_per_pid(self):
        """多 PID 时不混用不同进程或机器的 monotonic 时间。"""
        first = self._step("aggregated", "prefill", 0)
        second = self._step("aggregated", "prefill", 0)
        second["request_id"] = "request-2"
        offset_ns = 10**15
        timestamp_keys = (
            "server_received_at_ns",
            "scheduler_enqueued_at_ns",
            "scheduler_polled_at_ns",
            "scheduler_batch_ready_at_ns",
            "engine_forward_started_at_ns",
            "engine_forward_completed_at_ns",
        )
        for timestamp_key in timestamp_keys:
            second[timestamp_key] += offset_ns
        first.update({"_timing_pid": 100, "forward_batch_id": "forward-1"})
        second.update({"_timing_pid": 200, "forward_batch_id": "forward-2"})

        summary = _physical_forward_summary([first, second], "aggregated")
        self.assertEqual(summary["process_count"], 2)
        self.assertIsNone(summary["forward_busy_ratio"])
        self.assertIsNone(summary["local_wall_ms"])
        self.assertEqual(
            [process["forward_busy_ratio"] for process in summary["processes"]],
            [0.375, 0.375],
        )

        transfer_records = [
            {
                "_timing_pid": 100,
                "started_at_ns": 100,
                "completed_at_ns": 200,
            },
            {
                "_timing_pid": 100,
                "started_at_ns": 150,
                "completed_at_ns": 250,
            },
            {
                "_timing_pid": 200,
                "started_at_ns": offset_ns,
                "completed_at_ns": offset_ns + 100,
            },
        ]
        self.assertEqual(_merged_interval_union(transfer_records), 0.00025)

    def test_aggregated_logs_validate_and_compute_metrics(self):
        """普通服务日志校验完整，并计算本地 TTFT/prefill/decode。"""
        records = [
            self._step("aggregated", "prefill", 0),
            self._step("aggregated", "decode", 1),
            self._finish("aggregated", 2, 60),
        ]
        records[0]["batch_size"] = 2
        records[1]["batch_size"] = 2
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(root, "1000", "request-1", records)
            validation = validate_timing_root(root)
            metrics = compute_timing_metrics(root)

        self.assertEqual(validation["complete_request_count"], 1)
        request = metrics["requests"][0]
        self.assertEqual(request["mode"], "aggregated")
        self.assertEqual(request["ttft_local_ms"], 0.00008)
        self.assertEqual(request["prefill_compute_ms"], 0.00003)
        self.assertEqual(request["decode_compute_ms"], 0.00003)
        self.assertEqual(
            metrics["summary"]["aggregated_node"]["physical_forward_count"],
            1,
        )

    def test_pd_logs_validate_and_compute_metrics(self):
        """PD 日志分别计算 P 计算、D 计算、传输和 KV readiness。"""
        prefill = [
            self._step("prefill", "prefill", 0),
            self._finish("prefill", 1, 30),
            self._with_send_batch_fields(self._transfer("kv_send", 100)),
        ]
        kv_receive = self._transfer("kv_receive", 150)
        decode = self._step("decode", "decode", 0)
        decode.update(
            {
                "scheduler_polled_at_ns": 320,
                "scheduler_batch_ready_at_ns": 330,
                "engine_forward_started_at_ns": 350,
                "engine_forward_completed_at_ns": 370,
                "durations_ns": {
                    "queue": 220,
                    "scheduler": 10,
                    "scheduler_batch_ready_to_engine_forward_start": 20,
                    "engine_forward": 20,
                },
            }
        )
        decode_finish = self._finish("decode", 1, 20)
        decode_finish.update(
            {
                "first_scheduler_polled_at_ns": 320,
                "first_engine_forward_started_at_ns": 350,
                "last_engine_forward_completed_at_ns": 370,
            }
        )
        decode_records = [kv_receive, decode, decode_finish]

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(root, "2000", "request-1", prefill)
            self._write(root, "2001", "request-1", decode_records)
            validation = validate_timing_root(root)
            metrics = compute_timing_metrics(root)

        self.assertEqual(validation["complete_request_count"], 1)
        request = metrics["requests"][0]
        self.assertEqual(request["mode"], "pd")
        self.assertEqual(request["p_compute_ms"], 0.00003)
        self.assertEqual(request["d_compute_ms"], 0.00002)
        self.assertEqual(request["mooncake_transfer_ms"], 0.0001)
        self.assertEqual(request["d_kv_readiness_ms"], 0.00015)
        self.assertEqual(request["decode_start_after_kv_ready_ms"], 0.0)
        mooncake = metrics["summary"]["mooncake_transfer"]["kv_send"]
        self.assertEqual(mooncake["unique_batch_count"], 1)
        self.assertEqual(mooncake["batch_bytes_total"], 1024)
        self.assertEqual(
            metrics["summary"]["prefill_node"]["physical_forward_count"],
            1,
        )
        self.assertEqual(
            metrics["summary"]["decode_node"]["physical_forward_count"],
            1,
        )

    def test_request_metrics_use_first_path_stages_without_queue(self):
        """请求级指标输出六个首调度关键区间，不再输出混合口径 queue_ms。"""
        records = [
            self._step("aggregated", "prefill", 0),
            self._finish("aggregated", 1, 30),
        ]
        records[0]["server_received_at_ns"] = 100
        records[0]["scheduler_enqueued_at_ns"] = 110
        records[0]["scheduler_polled_at_ns"] = 130
        records[0]["scheduler_batch_ready_at_ns"] = 140
        records[0]["engine_forward_started_at_ns"] = 160
        records[0]["engine_forward_completed_at_ns"] = 180

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(root, "1000", "request-1", records)
            metrics = compute_timing_metrics(root)

        role = metrics["requests"][0]["aggregated"]
        self.assertNotIn("queue_ms", role)
        self.assertEqual(role["server_to_scheduler_enqueue_ms"], 0.00001)
        self.assertEqual(role["scheduler_enqueue_to_polled_ms"], 0.00002)
        self.assertEqual(role["scheduler_batch_build_ms"], 0.00001)
        self.assertEqual(
            role["scheduler_batch_ready_to_engine_forward_start_ms"],
            0.00002,
        )
        self.assertEqual(role["server_to_engine_forward_start_ms"], 0.00006)
        self.assertEqual(role["first_engine_forward_ms"], 0.00002)
        self.assertEqual(role["prefill_forward_total_ms"], 0.00003)
        self.assertEqual(role["decode_forward_total_ms"], 0.0)
        self.assertEqual(role["scheduler_waiting_queue_depth_before"], 4)
        self.assertEqual(role["scheduler_running_queue_depth_before"], 2)
        self.assertEqual(role["scheduler_remote_kv_requests_before"], 1)
        self.assertEqual(role["scheduler_waiting_queue_depth_after"], 3)
        self.assertEqual(role["scheduler_running_queue_depth_after"], 1)
        self.assertEqual(role["scheduler_remote_kv_requests_after"], 0)
        snapshots = metrics["summary"]["aggregated_node"]["scheduler_snapshots"]
        self.assertEqual(snapshots["snapshot_count"], 1)
        self.assertEqual(
            snapshots["scheduler_waiting_queue_depth_before"],
            {"count": 1, "avg": 4, "min": 4, "max": 4, "p50": 4, "p95": 4},
        )
        self.assertEqual(
            snapshots["active_backlog_before"],
            {"count": 1, "avg": 7, "min": 7, "max": 7, "p50": 7, "p95": 7},
        )
        self.assertEqual(
            snapshots["active_backlog_after_including_scheduled"],
            {"count": 1, "avg": 5, "min": 5, "max": 5, "p50": 5, "p95": 5},
        )
        self.assertEqual(metrics["summary"]["aggregated"]["ttft_local_ms"]["min_ms"], 0.00008)

    def test_request_id_filter_uses_safe_filename(self):
        """unsafe request id 过滤必须按 recorder 的安全文件名匹配。"""
        request_id = "../unsafe request"
        records = [
            self._step("aggregated", "prefill", 0),
            self._finish("aggregated", 1, 30),
        ]
        records[0]["request_id"] = request_id
        records[1]["request_id"] = request_id
        filename = expected_timing_filename(request_id)

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            directory = root / "1000"
            directory.mkdir(parents=True)
            (directory / filename).write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            validation = validate_timing_root(root, request_id=request_id)
            metrics = compute_timing_metrics(root, request_id=request_id)

        self.assertEqual(validation["file_count"], 1)
        self.assertEqual(metrics["validation"]["file_count"], 1)
        self.assertEqual(metrics["requests"][0]["request_id"], request_id)

    def test_duration_mismatch_is_invalid(self):
        """duration 与时间戳不一致时必须报 invalid。"""
        records = [
            self._step("aggregated", "prefill", 0),
            self._step("aggregated", "decode", 1),
            self._finish("aggregated", 2, 60),
        ]
        records[0]["durations_ns"]["engine_forward"] = 31
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(root, "3000", "request-1", records)
            validation = validate_timing_root(root)

        self.assertEqual(validation["file_status_counts"], {"invalid": 1})
        self.assertTrue(
            any(
                error.startswith("step engine_forward duration mismatch")
                for error in validation["files"][0]["errors"]
            )
        )

    def test_kv_send_batch_consistency_mismatch_is_invalid(self):
        """kv_send batch_request_count 必须等于同一 batch 的 request 事件数。"""
        first = self._with_send_batch_fields(self._transfer("kv_send", 100))
        second = self._with_send_batch_fields(self._transfer("kv_send", 100))
        second["request_id"] = "request-2"
        second["batch_request_count"] = 2
        second["request_block_count"] = 1

        prefill_step = self._step("prefill", "prefill", 0)
        prefill_step["request_id"] = "request-1"
        prefill_finish = self._finish("prefill", 1, 30)
        prefill_finish["request_id"] = "request-1"

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(
                root,
                "4000",
                "request-1",
                [prefill_step, prefill_finish, first],
            )
            self._write(root, "4000", "request-2", [second])
            validation = validate_timing_root(root)

        self.assertGreater(
            validation["batch_consistency_error_count"],
            0,
        )
        self.assertGreater(validation["invalid_file_count"], 0)

    def test_invalid_forward_batch_id_and_schema_version_are_rejected(self):
        """forward_batch_id 和 schema_version 的类型错误必须被判 invalid。"""
        records = [
            self._step("aggregated", "prefill", 0),
            self._finish("aggregated", 1, 30),
        ]
        records[0]["forward_batch_id"] = ["bad-id"]
        records[0]["schema_version"] = 5

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(root, "5000", "request-1", records)
            validation = validate_timing_root(root)

        self.assertEqual(validation["file_status_counts"], {"invalid": 1})
        self.assertTrue(
            any("forward_batch_id" in error for error in validation["files"][0]["errors"])
        )
        self.assertTrue(
            any("schema_version" in error for error in validation["files"][0]["errors"])
        )

    def test_schema_v4_requires_scheduler_snapshots(self):
        """schema v4 的每条 step 都必须有六个非负调度状态快照。"""
        records = [
            self._step("aggregated", "prefill", 0),
            self._finish("aggregated", 1, 30),
        ]
        records[0].pop("scheduler_remote_kv_requests_after")

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(root, "8000", "request-1", records)
            validation = validate_timing_root(root)

        self.assertGreater(validation["invalid_file_count"], 0)
        self.assertTrue(
            any(
                "scheduler_remote_kv_requests_after" in error
                for error in validation["files"][0]["errors"]
            )
        )

    def test_scheduler_snapshots_must_match_inside_forward_batch(self):
        """同一 scheduler batch 内六个状态快照必须完全一致。"""
        first = self._step("aggregated", "prefill", 0)
        second = self._step("aggregated", "prefill", 0)
        first["request_id"] = "request-1"
        second["request_id"] = "request-2"
        first["batch_size"] = 2
        second["batch_size"] = 2
        first["forward_batch_id"] = "forward-batch-1"
        second["forward_batch_id"] = "forward-batch-1"
        second["scheduler_waiting_queue_depth_after"] = 4
        first_finish = self._finish("aggregated", 1, 30)
        first_finish["request_id"] = "request-1"
        second_finish = self._finish("aggregated", 1, 30)
        second_finish["request_id"] = "request-2"

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(
                root,
                "9000",
                "request-1",
                [first, first_finish],
            )
            self._write(
                root,
                "9000",
                "request-2",
                [second, second_finish],
            )
            validation = validate_timing_root(root)

        self.assertGreater(validation["batch_consistency_error_count"], 0)
        self.assertTrue(
            any(
                "scheduler_waiting_queue_depth_after" in error
                for error in validation["requests"][0]["errors"]
            )
        )

    def test_compute_metrics_does_not_crash_on_invalid_durations(self):
        """invalid timing 应返回错误报告，而不是让 metrics 计算先崩溃。"""
        records = [
            self._step("aggregated", "prefill", 0),
            self._finish("aggregated", 1, 30),
        ]
        records[0]["durations_ns"] = None

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(root, "7000", "request-1", records)
            validation = validate_timing_root(root)
            metrics = compute_timing_metrics(root)

        self.assertGreater(validation["invalid_file_count"], 0)
        self.assertEqual(metrics["validation"]["invalid_file_count"], 1)
        self.assertEqual(metrics["requests"][0]["aggregated"]["step_count"], 1)

    def test_pd_decode_before_kv_receive_completion_is_invalid(self):
        """D 首 decode forward 不能早于最终 KV receive 完成。"""
        prefill = [
            self._step("prefill", "prefill", 0),
            self._finish("prefill", 1, 30),
            self._with_send_batch_fields(self._transfer("kv_send", 100)),
        ]
        kv_receive = self._transfer("kv_receive", 150)
        kv_receive["completed_at_ns"] = 500
        kv_receive["duration_ns"] = 300
        decode = self._step("decode", "decode", 0)
        decode["engine_forward_started_at_ns"] = 400
        decode["durations_ns"]["engine_forward"] = 20
        decode["engine_forward_completed_at_ns"] = 420
        decode_finish = self._finish("decode", 1, 20)
        decode_finish["last_engine_forward_completed_at_ns"] = 420

        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write(root, "6000", "request-1", prefill)
            self._write(
                root,
                "6001",
                "request-1",
                [kv_receive, decode, decode_finish],
            )
            validation = validate_timing_root(root)

        self.assertGreater(validation["invalid_file_count"], 0)
        self.assertTrue(
            any(
                "decode starts before" in error
                for error in validation["requests"][0]["errors"]
            )
        )


if __name__ == "__main__":
    unittest.main()
