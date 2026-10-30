import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infinilm.llm.llm import AsyncLLMEngine
from infinilm.llm.request import FinishReason, InferenceRequest
from infinilm.llm.request_timing import (
    DisabledRequestTimingRecorder,
    RequestTimingConfig,
    RequestTimingFileRecorder,
)
from infinilm.llm.request_timing.compute_metrics import compute_timing_metrics
from infinilm.llm.request_timing.validate_logs import validate_timing_root
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler
from infinilm.server.inference_server import InferenceServer


class RequestTimingConfigTest(unittest.TestCase):
    def test_environment_controls_timing_and_root(self):
        with patch.dict("os.environ", {}, clear=True):
            config = RequestTimingConfig.from_environment()
        self.assertTrue(config.enabled)
        self.assertEqual(config.mode, "summary")
        self.assertEqual(config.sample_interval, 32)
        self.assertEqual(config.root, Path("/tmp/infinilm-request-timing"))
        self.assertIsInstance(config.create_recorder(), RequestTimingFileRecorder)

        with patch.dict("os.environ", {"INFINILM_REQUEST_TIMING_MODE": "full"}):
            self.assertEqual(RequestTimingConfig.from_environment().mode, "full")

        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.dict(
                "os.environ",
                {
                    "INFINILM_REQUEST_TIMING": "0",
                    "INFINILM_REQUEST_TIMING_DIR": temporary_dir,
                },
            ):
                config = RequestTimingConfig.from_environment()
            self.assertFalse(config.enabled)
            self.assertEqual(config.root, Path(temporary_dir))
            recorder = config.create_recorder()
            self.assertIsInstance(recorder, DisabledRequestTimingRecorder)

            request = InferenceRequest(
                request_id="disabled-request",
                prompt_token_ids=[1],
                sampling_params=SamplingParams(max_tokens=1),
            )
            recorder.record_finish(request)
            recorder.close()
            self.assertEqual(list(Path(temporary_dir).iterdir()), [])

        with patch.dict("os.environ", {"INFINILM_REQUEST_TIMING": "bad"}):
            with self.assertRaisesRegex(RuntimeError, "must be a boolean"):
                RequestTimingConfig.from_environment()

        with patch.dict("os.environ", {"INFINILM_REQUEST_TIMING_MODE": "bad"}):
            with self.assertRaisesRegex(RuntimeError, "must be full or summary"):
                RequestTimingConfig.from_environment()

        for value in ("1", "4", "8", "16", "32", "64", "128"):
            with patch.dict(
                "os.environ",
                {"INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL": value},
            ):
                self.assertEqual(
                    RequestTimingConfig.from_environment().sample_interval,
                    int(value),
                )
        with patch.dict("os.environ", {"INFINILM_REQUEST_TIMING_SAMPLE_INTERVAL": "3"}):
            with self.assertRaisesRegex(RuntimeError, "SAMPLE_INTERVAL must be one"):
                RequestTimingConfig.from_environment()

    def test_summary_recorder_writes_one_finish_record(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(
                root=Path(temporary_dir),
                mode="summary",
                sample_interval=1,
            )
            request = InferenceRequest(
                request_id="summary-request",
                prompt_token_ids=[1, 2],
                sampling_params=SamplingParams(max_tokens=2),
                server_received_at_ns=100,
            )
            request.request_created_at_ns = 110
            request.scheduler_enqueued_at_ns = 120
            request.timing_mode = "summary"
            request.sample_interval = 1
            request.record_summary_step(140)
            request.record_summary_prefill_forward(150, 160)
            request.append_generated_token_id(1)
            request.record_summary_decode_forward(170, 180)
            request.append_generated_token_id(2)
            request.finish_reason = FinishReason.LENGTH

            recorder.record_steps(
                [Mock(request_id="summary-request", output_timing=None)]
            )
            recorder.record_finish(request)
            recorder.close()

            files = list(Path(temporary_dir).glob("*/*.jsonl"))
            self.assertEqual(len(files), 1)
            lines = files[0].read_text().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["timing_mode"], "summary")
            self.assertEqual(payload["schema_version"], 7)
            self.assertEqual(payload["steps"], 2)
            self.assertEqual(payload["engine_forward_total_ns"], 20)
            self.assertEqual(payload["decode_step_count"], 1)
            self.assertEqual(payload["sampled_decode_step_count"], 1)
            self.assertEqual(payload["first_engine_forward_started_at_ns"], 150)
            self.assertEqual(payload["last_engine_forward_completed_at_ns"], 180)

            validation = validate_timing_root(Path(temporary_dir))
            self.assertEqual(validation["file_status_counts"], {"complete": 1})
            metrics = compute_timing_metrics(Path(temporary_dir))
            self.assertEqual(metrics["summary"]["request_count"], 1)
            aggregated = metrics["requests"][0]["aggregated"]
            self.assertEqual(aggregated["timing_mode"], "summary")
            self.assertEqual(aggregated["engine_forward_total_ms"], 0.00002)

    def test_sampled_summary_writes_estimate_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(
                root=Path(temporary_dir),
                mode="summary",
                sample_interval=32,
            )
            request = InferenceRequest(
                request_id="sampled-request",
                prompt_token_ids=[1, 2],
                sampling_params=SamplingParams(max_tokens=33),
                server_received_at_ns=100,
            )
            request.timing_mode = "summary"
            request.sample_interval = 32
            request.record_summary_step(140)
            request.record_summary_prefill_forward(150, 160)
            request.append_generated_token_id(1)

            for index in range(1, 33):
                duration = 20
                request.record_summary_decode_forward(
                    200 + index * 100,
                    200 + index * 100 + duration,
                )
                request.append_generated_token_id(index + 1)
            request.finish_reason = FinishReason.LENGTH

            recorder.record_finish(request)
            recorder.close()
            payload = json.loads(
                next(Path(temporary_dir).glob("*/*.jsonl")).read_text()
            )

            self.assertEqual(payload["schema_version"], 8)
            self.assertEqual(payload["sample_interval"], 32)
            self.assertEqual(payload["steps"], 33)
            self.assertEqual(payload["decode_step_count"], 32)
            self.assertEqual(payload["sampled_decode_step_count"], 2)
            self.assertIsNone(payload["engine_forward_total_ns"])
            self.assertEqual(payload["engine_forward_total_estimate_ns"], 650)

            validation = validate_timing_root(Path(temporary_dir))
            self.assertEqual(validation["file_status_counts"], {"complete": 1})
            metrics = compute_timing_metrics(Path(temporary_dir))
            aggregated = metrics["requests"][0]["aggregated"]
            self.assertTrue(aggregated["engine_forward_estimated"])
            self.assertEqual(aggregated["engine_forward_total_ms"], 0.00065)

    def test_disabled_scheduler_does_not_create_step_timing(self):
        request = InferenceRequest(
            request_id="disabled-request",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=2),
        )
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=8,
            block_size=1,
            max_num_batched_tokens=4,
            enable_prefix_caching=False,
            request_timing_enabled=False,
        )

        scheduler.add_request(request)
        self.assertIsNone(request.scheduler_enqueued_at_ns)
        prefill = scheduler.schedule()
        scheduler.running_queue.sync_q.put(request)
        decode = scheduler.schedule()

        self.assertIsNotNone(prefill)
        self.assertIsNotNone(decode)
        self.assertEqual(request.timing_steps, [])
        self.assertIsNone(request.scheduler_polled_at_ns)

    def test_summary_scheduler_keeps_only_request_aggregates(self):
        request = InferenceRequest(
            request_id="summary-request",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=2),
        )
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=8,
            block_size=1,
            max_num_batched_tokens=4,
            enable_prefix_caching=False,
            request_timing_enabled=True,
            request_timing_mode="summary",
            request_timing_sample_interval=32,
        )

        scheduler.add_request(request)
        scheduler.schedule()
        request.record_summary_prefill_forward(150, 160)
        scheduler.running_queue.sync_q.put(request)
        scheduler.schedule()
        request.record_summary_decode_forward(170, 180)

        self.assertEqual(request.timing_steps, [])
        self.assertEqual(request.timing_mode, "summary")
        self.assertEqual(request.sample_interval, 32)
        self.assertEqual(request.decode_step_count, 1)
        self.assertEqual(request.sampled_decode_step_count, 1)

    def test_stream_prestart_accepts_disabled_entrance_timing(self):
        server = InferenceServer.__new__(InferenceServer)
        started = SimpleNamespace(tokenized_future=object())
        server.engine = SimpleNamespace(
            start_chat_preprocessing=Mock(return_value=started)
        )

        with patch.object(
            server,
            "_build_sampling_params",
            return_value=SimpleNamespace(ignore_eos=False),
        ):
            result, _ = server._start_stream_chat_request(
                request_id="request-1",
                data={"messages": [{"role": "user", "content": "hello"}]},
                server_received_at_ns=None,
                entrance_timing=None,
            )

        self.assertIs(result, started)
        server.engine.start_chat_preprocessing.assert_called_once_with(
            messages=[{"role": "user", "content": "hello"}],
            request_id="request-1",
            add_generation_prompt=True,
            chat_template_kwargs={},
            server_received_at_ns=None,
            entrance_timing=None,
        )

    def test_disabled_preprocessing_accepts_missing_entrance_timing(self):
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        engine.request_timing_enabled = False
        engine.engine = SimpleNamespace(
            processor=SimpleNamespace(
                supports_fast_text_tokenize=True,
                fast_text_tokenize=Mock(return_value=[1, 2, 3]),
            )
        )
        engine._thread_preprocess_tokenizer = Mock(return_value=object())

        with patch(
            "infinilm.llm.llm.resolve_multimodal_inputs",
            return_value={
                "images": [],
                "image_urls": [],
                "videos": [],
                "video_urls": [],
                "audios": [],
                "audio_urls": [],
            },
        ):
            tokenized = engine._tokenize_chat_input(
                "hello",
                [{"role": "user", "content": "hello"}],
                None,
            )

        self.assertEqual(tokenized.prompt_token_ids, [1, 2, 3])
        self.assertFalse(tokenized.has_multimodal_inputs)


if __name__ == "__main__":
    unittest.main()
