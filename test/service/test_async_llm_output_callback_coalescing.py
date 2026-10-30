import asyncio
import threading
import unittest
from unittest.mock import patch

from infinilm.llm.llm import (
    AsyncLLMEngine,
    output_callback_coalescing_from_environment,
    output_flush_max_tokens_from_environment,
    output_token_batch_from_environment,
    stream_ready_drain_items_from_environment,
)
from infinilm.llm.request import TokenBatch, TokenOutput


class AsyncLLMOutputCallbackCoalescingTest(unittest.TestCase):
    @staticmethod
    def _engine(*, enabled: bool = True):
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        engine._coalesce_output_callbacks = enabled
        engine._output_flush_max_tokens = 0
        engine._output_token_batch = False
        engine._pending_stream_outputs = []
        engine._pending_output_flush_scheduled = False
        engine._pending_output_lock = threading.Lock()
        engine._output_producer_batch_count = 0
        engine._output_producer_token_count = 0
        engine._output_flush_callback_count = 0
        engine._output_flush_token_count = 0
        engine._output_flush_max_batch_size = 0
        engine._output_flush_queue_item_count = 0
        engine._output_flush_max_queue_items = 0
        engine._stream_queue_wait_count = 0
        engine._stream_queue_nowait_count = 0
        engine._stream_timer_start_count = 0
        engine._stream_ready_drain_count = 0
        engine._output_max_pending_buffer_size = 0
        return engine

    @staticmethod
    def _output(token_id: int) -> TokenOutput:
        return TokenOutput(
            request_id="request-1",
            token_id=token_id,
            token_text=str(token_id),
        )

    def test_environment_defaults_to_coalescing_and_rejects_invalid_values(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(output_callback_coalescing_from_environment())
        with patch.dict("os.environ", {"INFINILM_OUTPUT_CALLBACK_COALESCE": "0"}):
            self.assertFalse(output_callback_coalescing_from_environment())
        with patch.dict("os.environ", {"INFINILM_OUTPUT_CALLBACK_COALESCE": "bad"}):
            with self.assertRaisesRegex(RuntimeError, "must be a boolean"):
                output_callback_coalescing_from_environment()
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(output_flush_max_tokens_from_environment(), 256)
        with patch.dict("os.environ", {"INFINILM_OUTPUT_FLUSH_MAX_TOKENS": "0"}):
            self.assertEqual(output_flush_max_tokens_from_environment(), 0)
        with patch.dict("os.environ", {"INFINILM_OUTPUT_FLUSH_MAX_TOKENS": "3"}):
            with self.assertRaisesRegex(
                RuntimeError, "must be one of 0, 64, 128, 256, 512"
            ):
                output_flush_max_tokens_from_environment()
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(output_token_batch_from_environment())
        with patch.dict("os.environ", {"INFINILM_OUTPUT_TOKEN_BATCH": "0"}):
            self.assertFalse(output_token_batch_from_environment())
        with patch.dict("os.environ", {"INFINILM_OUTPUT_TOKEN_BATCH": "bad"}):
            with self.assertRaisesRegex(RuntimeError, "must be a boolean"):
                output_token_batch_from_environment()
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(stream_ready_drain_items_from_environment(), 4)
        for value in ("0", "1", "2", "8"):
            with patch.dict("os.environ", {"INFINILM_STREAM_READY_DRAIN_ITEMS": value}):
                self.assertEqual(
                    stream_ready_drain_items_from_environment(), int(value)
                )
        with patch.dict("os.environ", {"INFINILM_STREAM_READY_DRAIN_ITEMS": "3"}):
            with self.assertRaisesRegex(RuntimeError, "must be one of 0, 1, 2, 4, 8"):
                stream_ready_drain_items_from_environment()

    def test_token_batch_groups_outputs_by_queue(self):
        engine = self._engine()
        engine._output_token_batch = True

        class Queue:
            def __init__(self):
                self.items = []

            def put_nowait(self, item):
                self.items.append(item)

        queues = (Queue(), Queue())
        pending = [
            (queues[0], self._output(1)),
            (queues[1], self._output(2)),
            (queues[0], self._output(3)),
        ]
        engine._batch_put(pending, use_token_batch=True)

        self.assertEqual(len(queues[0].items), 1)
        self.assertEqual(len(queues[1].items), 1)
        self.assertEqual(
            [output.token_id for output in queues[0].items[0].outputs], [1, 3]
        )
        self.assertEqual(
            [output.token_id for output in queues[1].items[0].outputs], [2]
        )
        self.assertIsInstance(queues[0].items[0], TokenBatch)
        stats = engine.get_output_flush_stats()
        self.assertEqual(stats["flush_queue_items"], 2)
        self.assertEqual(stats["max_flush_queue_items"], 2)

    def test_bounded_flush_preserves_order_and_reschedules_remainder(self):
        engine = self._engine()
        engine._output_flush_max_tokens = 2
        continuations = []

        class Loop:
            def call_soon_threadsafe(self, callback):
                pass

            def call_soon(self, callback):
                continuations.append(callback)

        engine._loop = Loop()

        class Queue:
            def __init__(self):
                self.outputs = []

            def put_nowait(self, output):
                self.outputs.append(output)

        queues = [Queue() for _ in range(3)]
        engine._submit_pending_outputs(
            [(queue, self._output(index)) for index, queue in enumerate(queues)]
        )
        engine._flush_pending_outputs()

        self.assertEqual(
            [queue.outputs[0].token_id for queue in queues if queue.outputs], [0, 1]
        )
        self.assertEqual(len(continuations), 1)
        self.assertEqual(len(engine._pending_stream_outputs), 1)

    def test_multiple_submissions_share_one_scheduled_flush(self):
        engine = self._engine()
        scheduled = []
        engine._loop = type(
            "Loop",
            (),
            {"call_soon_threadsafe": lambda self, callback: scheduled.append(callback)},
        )()

        class Queue:
            def __init__(self):
                self.outputs = []

            def put_nowait(self, output):
                self.outputs.append(output)

        queues = (Queue(), Queue())
        engine._submit_pending_outputs([(queues[0], self._output(1))])
        engine._submit_pending_outputs([(queues[1], self._output(2))])

        self.assertEqual(len(scheduled), 1)
        self.assertEqual(len(engine._pending_stream_outputs), 2)
        self.assertTrue(engine._pending_output_flush_scheduled)
        stats = engine.get_output_flush_stats()
        self.assertEqual(stats["producer_batches"], 2)
        self.assertEqual(stats["flush_callbacks"], 0)
        self.assertEqual(stats["max_pending_buffer_size"], 2)

        engine._flush_pending_outputs()
        self.assertEqual(
            [queue.outputs[0].token_id for queue in queues],
            [1, 2],
        )

        self.assertEqual(engine._pending_stream_outputs, [])
        self.assertFalse(engine._pending_output_flush_scheduled)
        stats = engine.get_output_flush_stats()
        self.assertEqual(stats["flush_callbacks"], 1)
        self.assertEqual(stats["flush_tokens"], 2)
        self.assertEqual(stats["producer_batches_per_flush"], 2.0)

        engine._submit_pending_outputs([(object(), self._output(3))])
        self.assertEqual(len(scheduled), 2)

    def test_real_loop_flushes_coalesced_outputs_in_order(self):
        async def run():
            engine = self._engine()
            engine._loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            def produce():
                for token_id in range(4):
                    engine._submit_pending_outputs([(queue, self._output(token_id))])

            producer = threading.Thread(target=produce)
            producer.start()
            while producer.is_alive():
                pass
            await asyncio.sleep(0.02)

            outputs = [queue.get_nowait() for _ in range(4)]
            self.assertEqual([output.token_id for output in outputs], [0, 1, 2, 3])
            stats = engine.get_output_flush_stats()
            self.assertEqual(stats["producer_batches"], 4)
            self.assertEqual(stats["flush_callbacks"], 1)
            self.assertEqual(stats["flush_tokens"], 4)
            self.assertFalse(stats["flush_scheduled"])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
