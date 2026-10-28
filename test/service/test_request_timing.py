import json
import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infinilm.llm.llm import LLMEngine
from infinilm.llm.model_runner.model_runner import ModelForwardTiming, ModelRunner
from infinilm.llm.request import InferenceRequest, RequestStepTiming, monotonic_ns
from infinilm.llm.request_timing.request_timing import (
    RequestTimingFileRecorder,
    request_timing_path,
)
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler


class RequestTimingTest(unittest.TestCase):
    def _request(self, received_at_ns: int) -> InferenceRequest:
        return InferenceRequest(
            request_id="timing-request",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=1),
            server_received_at_ns=received_at_ns,
        )

    def _timing(self, request_id: str, step_index: int) -> RequestStepTiming:
        return RequestStepTiming(
            request_id=request_id,
            step_index=step_index,
            phase="prefill" if step_index == 0 else "decode",
            batch_size=1,
        )

    def test_scheduler_records_prefill_and_decode_steps(self):
        """timing step 覆盖 prefill 和 decode，并且时间顺序单调。"""
        request = self._request(100)
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=8,
            block_size=1,
            enable_prefix_caching=False,
        )
        scheduler.add_request(request)
        prefill = scheduler.schedule()

        # prefill 输出处理后，真实调度器会把仍在 RUNNING 的请求放回队列。
        scheduler.running_queue.sync_q.put(request)
        decode = scheduler.schedule()

        self.assertEqual(len(request.timing_steps), 2)
        self.assertEqual(
            [step.phase for step in request.timing_steps],
            ["prefill", "decode"],
        )
        for step in request.timing_steps:
            self.assertEqual(step.request_id, request.request_id)
            self.assertEqual(step.batch_size, 1)
            self.assertLessEqual(step.server_received_at_ns, 100)
            self.assertLessEqual(
                step.scheduler_enqueued_at_ns,
                step.scheduler_polled_at_ns,
            )
            self.assertLessEqual(
                step.scheduler_polled_at_ns,
                step.scheduler_batch_ready_at_ns,
            )
            self.assertEqual(
                (
                    step.scheduler_waiting_queue_depth_before,
                    step.scheduler_running_queue_depth_before,
                    step.scheduler_remote_kv_requests_before,
                    step.scheduler_waiting_queue_depth_after,
                    step.scheduler_running_queue_depth_after,
                    step.scheduler_remote_kv_requests_after,
                ),
                (1, 0, 0, 0, 0, 0)
                if step.phase == "prefill"
                else (0, 1, 0, 0, 0, 0),
            )
        self.assertIsNotNone(prefill)
        self.assertIsNotNone(decode)

    def test_model_runner_attaches_batch_forward_timing(self):
        """ModelRunner 把 batch forward timing 归到当前 step。"""
        request = self._request(100)
        request.timing_steps = []
        request.scheduler_enqueued_at_ns = 105
        request.scheduler_polled_at_ns = 110
        request.timing_steps.append(
            RequestStepTiming(
                request_id=request.request_id,
                step_index=0,
                phase="prefill",
                batch_size=1,
                server_received_at_ns=100,
                scheduler_enqueued_at_ns=105,
                scheduler_polled_at_ns=110,
                scheduler_batch_ready_at_ns=120,
            )
        )
        runner = ModelRunner.__new__(ModelRunner)
        runner.kv_connector = None
        scheduler_output = SimpleNamespace(
            scheduled_requests=[request],
            num_requests=1,
            is_prefill=True,
        )

        with patch.object(
            ModelRunner,
            "_model_forward",
            return_value=([7], ModelForwardTiming(130, 180)),
        ):
            output = runner.execute_model(scheduler_output)

        timing = request.timing_steps[0]
        self.assertEqual(timing.engine_forward_started_at_ns, 130)
        self.assertEqual(timing.engine_forward_completed_at_ns, 180)
        self.assertEqual(output.forward_timing.source, "model_runner")

    def test_model_runner_times_engine_forward_call(self):
        """forward 计时只包住 engine.forward，不包含输入构造和输出转换。"""
        runner = ModelRunner.__new__(ModelRunner)
        runner.speculative_runner = None
        runner.pipeline_control = None
        engine_forward_at_ns = {}

        class Output:
            @staticmethod
            def to_numpy():
                return SimpleNamespace(tolist=lambda: [7])

        def fake_forward(**kwargs):
            engine_forward_at_ns["at"] = monotonic_ns()
            return Output()

        runner.processor = SimpleNamespace(
            build_model_inputs=lambda *args, **kwargs: {}
        )
        runner.model_engine = SimpleNamespace(forward=fake_forward)
        runner.config = SimpleNamespace(temperature=1, top_p=1, top_k=1)

        sampled_tokens, timing = runner._model_forward(
            SimpleNamespace(scheduled_requests=[])
        )

        self.assertEqual(sampled_tokens, [7])
        self.assertLessEqual(
            timing.engine_forward_started_at_ns,
            engine_forward_at_ns["at"],
        )
        self.assertLessEqual(
            engine_forward_at_ns["at"],
            timing.engine_forward_completed_at_ns,
        )
        self.assertIsNotNone(timing.forward_batch_id)

    def test_recorder_stats_reports_buffers_and_pending_jobs(self):
        """stats 暴露 dropped/pending/buffered，便于诊断观测写入积压。"""
        with TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(root=Path(temporary_dir))
            recorder.record_step(
                RequestStepTiming(
                    request_id="stats-request",
                    step_index=0,
                    phase="prefill",
                    batch_size=1,
                )
            )
            recorder.record_step(
                RequestStepTiming(
                    request_id="stats-request",
                    step_index=1,
                    phase="decode",
                    batch_size=1,
                )
            )
            self.assertEqual(recorder.stats()["buffered_requests"], 1)
            recorder.close()
            self.assertEqual(recorder.stats()["dropped_records"], 0)
            self.assertEqual(recorder.stats()["pending_jobs"], 0)
            self.assertEqual(recorder.stats()["buffered_requests"], 0)

    def test_speculative_path_does_not_report_engine_forward_timing(self):
        """不采集 speculative 路径；它可能包含多次内部 forward。"""
        runner = ModelRunner.__new__(ModelRunner)
        runner.pipeline_control = None
        runner.speculative_runner = SimpleNamespace(
            forward=lambda scheduler_output, model_input: [7]
        )
        runner.processor = SimpleNamespace(
            build_model_inputs=lambda *args, **kwargs: {}
        )
        runner.config = SimpleNamespace(temperature=1, top_p=1, top_k=1)

        sampled_tokens, timing = runner._model_forward(
            SimpleNamespace(scheduled_requests=[])
        )

        self.assertEqual(sampled_tokens, [7])
        self.assertIsNone(timing)

    def test_recorder_appends_steps_and_finish_to_one_file(self):
        """同一请求的多轮 timing 和 finish 汇总写入同一个 JSONL 文件。"""
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            recorder = RequestTimingFileRecorder(root=root)
            request = self._request(100)
            request.timing_steps = []
            request.timing_steps.extend(
                [
                    RequestStepTiming(
                        request_id=request.request_id,
                        step_index=0,
                        phase="prefill",
                        batch_size=1,
                        server_received_at_ns=100,
                        scheduler_enqueued_at_ns=105,
                        scheduler_polled_at_ns=110,
                        scheduler_batch_ready_at_ns=120,
                        engine_forward_started_at_ns=140,
                        engine_forward_completed_at_ns=180,
                    ),
                    RequestStepTiming(
                        request_id=request.request_id,
                        step_index=1,
                        phase="decode",
                        batch_size=1,
                    ),
                ]
            )

            recorder.record_steps(request.timing_steps)
            recorder.record_finish(request)
            recorder.close()

            path = (
                root
                / str(os.getpid())
                / f"{request.request_id}.jsonl"
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([record["event"] for record in records], [
            "step",
            "step",
            "finish",
        ])
        self.assertEqual(records[0]["durations_ns"]["queue"], 10)
        self.assertEqual(records[0]["durations_ns"]["scheduler"], 10)
        self.assertEqual(
            records[0]["durations_ns"][
                "scheduler_batch_ready_to_engine_forward_start"
            ],
            20,
        )
        self.assertEqual(records[0]["durations_ns"]["engine_forward"], 40)
        self.assertEqual(records[-1]["steps"], 2)

    def test_recorder_records_independent_kv_event(self):
        """KV transfer timing 不依赖 step，可作为独立 JSONL event 写出。"""
        with TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(root=Path(temporary_dir))
            payload = {
                "event": "kv_send",
                "request_id": "kv-event-request",
                "role": "prefill",
                "clock_domain": "local-monotonic",
                "cross_node_comparable": False,
                "duration_ns": 123,
            }
            recorder.record_event(payload)
            recorder.close()

            path = (
                Path(temporary_dir)
                / str(os.getpid())
                / "kv-event-request.jsonl"
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(
            records,
            [{**payload, "schema_version": 4}],
        )

    def test_recorder_batches_step_submission(self):
        """一轮 batch 中的多个 request 保持顺序并写入各自文件。"""
        first = RequestStepTiming(
            request_id="batch-first",
            step_index=0,
            phase="prefill",
            batch_size=2,
        )
        second = RequestStepTiming(
            request_id="batch-second",
            step_index=0,
            phase="prefill",
            batch_size=2,
        )

        with TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(root=Path(temporary_dir))
            recorder.record_steps([first, second])
            recorder.close()

            pid_dir = Path(temporary_dir) / str(os.getpid())
            first_records = [
                json.loads(line)
                for line in (pid_dir / "batch-first.jsonl").read_text().splitlines()
            ]
            second_records = [
                json.loads(line)
                for line in (pid_dir / "batch-second.jsonl").read_text().splitlines()
            ]

        self.assertEqual(len(first_records), 1)
        self.assertEqual(len(second_records), 1)
        self.assertEqual(first_records[0]["request_id"], "batch-first")
        self.assertEqual(second_records[0]["request_id"], "batch-second")

    def test_short_request_uses_two_write_jobs(self):
        """未触发阈值的请求只提交首步写任务和终态写任务。"""
        with TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(root=Path(temporary_dir))
            original_write_steps = recorder._write_steps
            original_write_finish = recorder._write_finish
            step_calls = []
            finish_calls = []

            def wrapped_write_steps(timings):
                step_calls.append(timings)
                original_write_steps(timings)

            def wrapped_write_finish(timings, request):
                finish_calls.append((timings, request))
                original_write_finish(timings, request)

            first = self._timing("short-request", 0)
            second = self._timing("short-request", 1)
            request = self._request(100)
            request.request_id = "short-request"
            request.timing_steps = [first, second]

            with (
                patch.object(
                    recorder,
                    "_write_steps",
                    side_effect=wrapped_write_steps,
                ),
                patch.object(
                    recorder,
                    "_write_finish",
                    side_effect=wrapped_write_finish,
                ),
            ):
                recorder.record_steps([first, second])
                recorder.record_finish(request)
                recorder.close()

        self.assertEqual(len(step_calls), 1)
        self.assertEqual(step_calls[0], (first,))
        self.assertEqual(len(finish_calls), 1)
        self.assertEqual(finish_calls[0][0], (second,))

    def test_request_timing_path_is_confined_to_root(self):
        """request id 不能携带路径分隔符逃逸指定目录。"""
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            path = request_timing_path("../malicious/id", root=root)

        self.assertEqual(path.parents[1], root)
        self.assertNotIn("..", path.parts)
        self.assertEqual(path.parent, root / str(os.getpid()))

    def test_recorder_ignores_file_io_failure(self):
        """/tmp 写失败只丢弃观测数据，不向推理线程抛异常。"""
        with TemporaryDirectory() as temporary_dir:
            unavailable_root = Path(temporary_dir) / "not-a-directory"
            unavailable_root.write_text("block directory creation")
            recorder = RequestTimingFileRecorder(root=unavailable_root)

            recorder.record_step(
                RequestStepTiming(
                    request_id="io-failure",
                    step_index=0,
                    phase="prefill",
                    batch_size=1,
                )
            )
            recorder.close()

    def test_recorder_drops_records_when_pending_queue_is_full(self):
        """后台积压时丢弃新 timing，不阻塞推理线程。"""
        started = threading.Event()
        release = threading.Event()

        def blocked_write(timings):
            started.set()
            self.assertTrue(release.wait(timeout=2))

        with TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(
                root=Path(temporary_dir),
                max_pending_jobs=1,
            )
            timing = RequestStepTiming(
                request_id="queued",
                step_index=0,
                phase="decode",
                batch_size=1,
            )

            with patch.object(recorder, "_write_steps", side_effect=blocked_write):
                recorder.record_steps([timing])
                self.assertTrue(started.wait(timeout=2))
                recorder.record_steps([timing])
                self.assertEqual(len(recorder._buffers["queued"]), 1)
                recorder.record_finish(self._request(100))
                self.assertEqual(recorder._dropped_records, 1)
                release.set()
                recorder.close()

    def test_recorder_flushes_after_256_buffered_steps(self):
        """内存缓冲达到 256 条时强制写出，不等待 finish。"""
        timings = [self._timing("batched-request", index) for index in range(257)]
        request = self._request(100)
        request.request_id = "batched-request"
        request.timing_steps = timings

        with TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(root=Path(temporary_dir))
            original_write_steps = recorder._write_steps
            calls = []

            def wrapped_write_steps(timings):
                calls.append(timings)
                original_write_steps(timings)

            with (
                patch.object(
                    recorder,
                    "_start_flush_thread_locked",
                ),
                patch.object(
                    recorder,
                    "_write_steps",
                    side_effect=wrapped_write_steps,
                ),
            ):
                recorder.record_steps(timings)
                self.assertEqual(len(recorder._buffers["batched-request"]), 0)
                recorder.record_finish(request)
                recorder.close()

            path = (
                Path(temporary_dir)
                / str(os.getpid())
                / "batched-request.jsonl"
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([len(call) for call in calls], [1, 256])
        self.assertEqual(len(records), 258)
        self.assertEqual(records[0]["step_index"], 0)
        self.assertEqual(records[-2]["step_index"], 256)
        self.assertEqual(records[-1]["event"], "finish")

    def test_recorder_flushes_after_ten_seconds(self):
        """距上次写出超过 10 秒时，下一条 step 触发强制 flush。"""
        with TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(root=Path(temporary_dir))
            with patch.object(recorder, "_start_flush_thread_locked"):
                recorder.record_step(self._timing("aged-request", 0))
                recorder.record_step(self._timing("aged-request", 1))
                self.assertEqual(len(recorder._buffers["aged-request"]), 1)

                with recorder._lock:
                    recorder._last_flush_at_ns["aged-request"] -= 10_000_000_001

                recorder.record_step(self._timing("aged-request", 2))
                self.assertEqual(len(recorder._buffers["aged-request"]), 0)

                request = self._request(100)
                request.request_id = "aged-request"
                request.timing_steps = [
                    self._timing("aged-request", 0),
                    self._timing("aged-request", 1),
                    self._timing("aged-request", 2),
                ]
                recorder.record_finish(request)
                recorder.close()

            path = (
                Path(temporary_dir)
                / str(os.getpid())
                / "aged-request.jsonl"
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([record["step_index"] for record in records[:3]], [0, 1, 2])
        self.assertEqual(records[-1]["event"], "finish")

    def test_close_flushes_buffered_steps_without_finish(self):
        """引擎退出时即使请求没有 finish，也写出已缓冲 step。"""
        with TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(root=Path(temporary_dir))
            with patch.object(recorder, "_start_flush_thread_locked"):
                recorder.record_step(self._timing("close-request", 0))
                recorder.record_step(self._timing("close-request", 1))
                recorder.close()

            path = (
                Path(temporary_dir)
                / str(os.getpid())
                / "close-request.jsonl"
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([record["event"] for record in records], ["step", "step"])

    def test_recorder_cleans_expired_files_in_background(self):
        """首次写入前清理过期文件，保留未过期文件。"""
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            pid_dir = root / str(os.getpid())
            pid_dir.mkdir(parents=True)
            old_file = pid_dir / "old.jsonl"
            new_file = pid_dir / "new.jsonl"
            old_file.write_text("{}\n")
            new_file.write_text("{}\n")
            expired_timestamp = time.time() - 7200
            os.utime(old_file, (expired_timestamp, expired_timestamp))

            recorder = RequestTimingFileRecorder(
                root=root,
                retention_seconds=3600,
            )
            recorder.record_step(
                RequestStepTiming(
                    request_id="cleanup-request",
                    step_index=0,
                    phase="prefill",
                    batch_size=1,
                )
            )
            recorder.close()

            self.assertFalse(old_file.exists())
            self.assertTrue(new_file.exists())

    def test_llm_engine_submits_timing_to_recorder(self):
        """LLMEngine step 只提交后台写任务，不在当前线程写文件。"""
        engine = LLMEngine.__new__(LLMEngine)
        request = self._request(100)
        request.timing_steps.append(
            RequestStepTiming(
                request_id=request.request_id,
                step_index=0,
                phase="prefill",
                batch_size=1,
            )
        )
        scheduler_output = SimpleNamespace(
            scheduled_requests=[request],
            num_requests=1,
        )
        engine.scheduler = SimpleNamespace(
            schedule=lambda: scheduler_output,
            update_from_output=Mock(),
            pop_timing_events=Mock(return_value=[]),
        )
        engine.model_runner = SimpleNamespace(
            execute_model=Mock(return_value=SimpleNamespace(
                sampled_token_ids=[7],
                kv_connector_output=None,
            ))
        )
        engine.request_timing_recorder = Mock()
        engine._update_requests = Mock(return_value=[])

        did_work, _ = engine.step()

        self.assertTrue(did_work)
        engine.request_timing_recorder.record_steps.assert_called_once_with(
            [request.timing_steps[0]]
        )

    def test_llm_engine_records_kv_transfer_event(self):
        """Scheduler 收集到的 KV transfer timing 会写入独立 event。"""
        engine = LLMEngine.__new__(LLMEngine)
        scheduler_output = SimpleNamespace(
            scheduled_requests=[],
            num_requests=0,
        )
        kv_event = {
            "event": "kv_receive",
            "request_id": "pd-request",
            "role": "decode",
            "clock_domain": "local-monotonic",
            "cross_node_comparable": False,
            "duration_ns": 100,
        }
        engine.scheduler = SimpleNamespace(
            schedule=lambda: scheduler_output,
            update_from_output=Mock(),
            pop_timing_events=Mock(return_value=[kv_event]),
        )
        engine.model_runner = SimpleNamespace(
            execute_model=Mock(return_value=SimpleNamespace(
                sampled_token_ids=[],
                kv_connector_output=SimpleNamespace(
                    finished_sending={"pd-request"},
                ),
            ))
        )
        engine.request_timing_recorder = Mock()
        engine._update_requests = Mock(return_value=[])

        did_work, _ = engine.step()

        self.assertTrue(did_work)
        engine.request_timing_recorder.record_event.assert_called_once_with(kv_event)

    def test_async_engine_derives_timing_role(self):
        """P/D worker 的 KV role 会映射为 timing role。"""
        from infinilm.llm.llm import AsyncLLMEngine

        for kv_role, expected_role in (
            ("kv_producer", "prefill"),
            ("kv_consumer", "decode"),
            (None, "aggregated"),
        ):
            with self.subTest(kv_role=kv_role):
                engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
                engine.config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role=kv_role)
                )
                self.assertEqual(engine._timing_role(), expected_role)

    def test_llm_engine_finish_uses_recorder_without_logging(self):
        """请求完成 timing 写入后台 recorder，而不是输出 logger。"""
        engine = LLMEngine.__new__(LLMEngine)
        request = self._request(100)
        request.timing_steps.append(
            RequestStepTiming(
                request_id=request.request_id,
                step_index=0,
                phase="prefill",
                batch_size=1,
                engine_forward_started_at_ns=100,
                engine_forward_completed_at_ns=180,
            )
        )
        engine.scheduler = SimpleNamespace(
            commit_computed_tokens=Mock(),
            complete_requests=Mock(),
        )
        engine.tokenizer = SimpleNamespace(decode=lambda token_ids: "ok")
        engine.request_timing_recorder = Mock()

        with self.assertNoLogs(level="DEBUG"):
            pending = engine._update_requests([request], [7])

        self.assertEqual(pending, [])
        engine.request_timing_recorder.record_finish.assert_called_once_with(request)

    def test_offline_request_records_finish_only_once(self):
        """offline 生成每轮只写 step，只有最终 token 才写 finish。"""
        engine = LLMEngine.__new__(LLMEngine)
        request = self._request(100)
        request.sampling_params.max_tokens = 2
        engine.eos_token_ids = []
        engine.scheduler = SimpleNamespace(
            commit_computed_tokens=Mock(),
            complete_requests=Mock(),
        )
        engine.tokenizer = SimpleNamespace(decode=lambda token_ids: "ok")
        engine.request_timing_recorder = Mock()

        engine._update_requests([request], [7])
        engine._update_requests([request], [8])

        self.assertEqual(len(request.generated_token_ids), 2)
        engine.request_timing_recorder.record_finish.assert_called_once_with(request)

    def test_aborted_request_records_finish(self):
        """客户端断开等 abort 路径也 flush 缓冲的 timing。"""
        engine = LLMEngine.__new__(LLMEngine)
        request = self._request(100)
        request.abort()
        engine.scheduler = SimpleNamespace(
            commit_computed_tokens=Mock(),
            complete_requests=Mock(),
        )
        engine.request_timing_recorder = Mock()

        pending = engine._update_requests([request], [7])

        self.assertEqual(pending, [])
        engine.request_timing_recorder.record_finish.assert_called_once_with(request)

    def test_llm_engine_closes_recorder_before_runner(self):
        """退出时先 flush timing，再释放 model runner 资源。"""
        engine = LLMEngine.__new__(LLMEngine)
        engine.request_timing_recorder = Mock()
        engine.model_runner = Mock()
        engine.close()

        engine.request_timing_recorder.close.assert_called_once_with()
        engine.model_runner.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
