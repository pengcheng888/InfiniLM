import json
import os
import tempfile
import unittest
from pathlib import Path

from infinilm.llm.output_timing import (
    DisabledOutputTimingCollector,
    EventLoopLagMonitor,
    OutputTimingCollector,
    OutputTimingConfig,
    OutputTimingService,
    OutputTokenTiming,
)


class OutputTimingTest(unittest.TestCase):
    def test_collector_aggregates_output_path_segments(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = OutputTimingConfig(
                enabled=True,
                root=Path(temporary_dir),
                trace_sample_rate=0.0,
            )
            collector = OutputTimingCollector(
                "safe-request", config, EventLoopLagMonitor(enabled=False)
            )
            timing = OutputTokenTiming(
                engine_sampled_ready_at_ns=100,
                engine_decode_started_at_ns=110,
                engine_decode_completed_at_ns=120,
                engine_token_ready_at_ns=130,
                loop_callback_scheduled_at_ns=140,
                loop_callback_started_at_ns=150,
                queue_put_started_at_ns=160,
                queue_put_completed_at_ns=170,
                queue_get_returned_at_ns=180,
                timer_cancel_started_at_ns=181,
                timer_cancel_completed_at_ns=182,
                queue_task_done_started_at_ns=183,
                queue_task_done_completed_at_ns=184,
                route_received_at_ns=190,
            )
            collector.record_token(timing, token_id=7)
            collector.record_sse_json(200, 210, body_bytes=30)
            collector.record_route_yield(211, 220, body_bytes=30)
            collector.record_asgi_body_send(212, 219, 30, "http.response.body")
            payload = collector.finish()

            path = Path(temporary_dir) / str(os.getpid()) / "safe-request.jsonl"
            written = json.loads(path.read_text())
            self.assertEqual(payload, written)
            self.assertEqual(written["token_count"], 1)
            self.assertEqual(written["asgi_body_message_count"], 1)
            self.assertEqual(len(written["token_trace"]), 1)
            self.assertEqual(written["token_trace"][0]["token_id"], 7)
            durations = written["durations_ms"]
            self.assertEqual(durations["engine_decode"]["count"], 1)
            self.assertEqual(durations["queue_put_to_queue_get"]["count"], 1)
            self.assertEqual(durations["asgi_body_send"]["count"], 1)

    def test_disabled_service_uses_noop_collector(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = OutputTimingConfig(enabled=False, root=Path(temporary_dir))
            service = OutputTimingService(config)
            self.assertIsInstance(
                service.create_collector("request"), DisabledOutputTimingCollector
            )


if __name__ == "__main__":
    unittest.main()
