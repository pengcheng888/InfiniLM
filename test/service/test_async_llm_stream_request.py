import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from infinilm.llm import AsyncLLMEngine, FinishReason
from infinilm.llm.llm import logger
from infinilm.llm.request import TokenBatch, TokenOutput


class AsyncLLMStreamRequestTest(unittest.TestCase):
    def _engine(self, request):
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        engine.add_aborted_req = Mock()
        engine._stream_ready_drain_items = 4
        engine._stream_queue_wait_count = 0
        engine._stream_queue_nowait_count = 0
        engine._stream_timer_start_count = 0
        engine._stream_ready_drain_count = 0
        return engine

    def test_ready_output_skips_asyncio_wait_for(self):
        """队列已有输出时使用同步 fast path，不创建 timeout task。"""

        class ReadyQueue:
            value = TokenOutput(
                request_id="request-1",
                token_id=1,
                token_text="ok",
                finished=True,
                finish_reason=FinishReason.LENGTH,
            )

            def get_nowait(self):
                value = self.value
                self.value = None
                if value is None:
                    raise asyncio.QueueEmpty
                return value

            async def get(self):
                raise AssertionError("ready output must use get_nowait")

            def task_done(self):
                pass

        request = SimpleNamespace(
            request_id="request-1",
            output_queue=SimpleNamespace(async_q=ReadyQueue()),
            is_aborted=Mock(return_value=False),
            finish_reason=FinishReason.LENGTH,
            generated_text="ok",
            close=AsyncMock(),
        )

        async def run():
            with patch("asyncio.wait_for", autospec=True) as wait_for:
                wait_for.side_effect = AssertionError("ready queue must not wait")
                engine = self._engine(request)
                outputs = [
                    output
                    async for output in engine.stream_request(
                        request,
                        timeout=100.0,
                    )
                ]

            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].finished)
            wait_for.assert_not_called()
            self.assertEqual(engine._stream_queue_nowait_count, 1)
            self.assertEqual(engine._stream_queue_wait_count, 0)
            self.assertEqual(engine._stream_timer_start_count, 0)
            request.close.assert_awaited_once()

        asyncio.run(run())

    def test_multiple_ready_batches_are_drained_without_timer(self):
        """ready TokenBatch 使用 get_nowait 批量取出，不创建 token timer。"""

        async def run():
            queue = asyncio.Queue()
            request = SimpleNamespace(
                request_id="request-1",
                output_queue=SimpleNamespace(async_q=queue),
                is_aborted=Mock(return_value=False),
                finish_reason=None,
                generated_text="123456",
                close=AsyncMock(),
            )
            engine = self._engine(request)
            engine._stream_ready_drain_items = 8
            for token_id in range(1, 7):
                await queue.put(
                    TokenBatch(
                        "request-1",
                        [
                            TokenOutput(
                                request_id="request-1",
                                token_id=token_id,
                                token_text=str(token_id),
                                finished=(token_id == 6),
                                finish_reason=(
                                    FinishReason.LENGTH if token_id == 6 else None
                                ),
                            )
                        ],
                    )
                )

            outputs = [
                output
                async for output in engine.stream_request(
                    request,
                    timeout=100.0,
                )
            ]

            self.assertEqual(
                [output.token_id for output in outputs], [1, 2, 3, 4, 5, 6]
            )
            self.assertEqual(engine._stream_queue_nowait_count, 6)
            self.assertEqual(engine._stream_queue_wait_count, 0)
            self.assertEqual(engine._stream_timer_start_count, 0)
            self.assertEqual(engine._stream_ready_drain_count, 1)
            request.close.assert_awaited_once()

        asyncio.run(run())

    def test_empty_output_queue_keeps_token_timeout(self):
        """队列空闲时仍使用 timeout，并在 abort 后输出终止 token。"""

        request = SimpleNamespace(
            request_id="request-1",
            output_queue=SimpleNamespace(async_q=asyncio.Queue()),
            is_aborted=Mock(side_effect=[False, True]),
            finish_reason=FinishReason.TIMEOUT,
            generated_text="",
            close=AsyncMock(),
        )

        async def run():
            with patch.object(logger, "warning"):
                outputs = [
                    output
                    async for output in self._engine(request).stream_request(
                        request,
                        timeout=0.001,
                    )
                ]

            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].finished)
            self.assertEqual(outputs[0].finish_reason, FinishReason.TIMEOUT)
            request.close.assert_awaited_once()

        asyncio.run(run())

    def test_token_batch_is_expanded_in_order(self):
        """queue 内部 batch 化后，对外的 token 流和 timeout 语义不变。"""

        async def run():
            queue = asyncio.Queue()
            request = SimpleNamespace(
                request_id="request-1",
                output_queue=SimpleNamespace(async_q=queue),
                is_aborted=Mock(return_value=False),
                finish_reason=None,
                generated_text="123",
                close=AsyncMock(),
            )
            outputs = [
                TokenOutput(
                    request_id="request-1",
                    token_id=index,
                    token_text=str(index),
                    finished=(index == 3),
                    finish_reason=(FinishReason.LENGTH if index == 3 else None),
                )
                for index in range(1, 4)
            ]
            await queue.put(TokenBatch("request-1", outputs))

            received = [
                output
                async for output in self._engine(request).stream_request(
                    request,
                    timeout=100.0,
                )
            ]

            self.assertEqual([output.token_id for output in received], [1, 2, 3])
            self.assertTrue(received[-1].finished)
            self.assertEqual(queue.qsize(), 0)
            request.close.assert_awaited_once()

        asyncio.run(run())

    def test_timeout_timer_does_not_run_while_generator_is_yielded(self):
        """timeout 只覆盖等待 token，不覆盖客户端消费已产出 token 的时间。"""

        async def run():
            queue = asyncio.Queue()
            request = SimpleNamespace(
                request_id="request-1",
                output_queue=SimpleNamespace(async_q=queue),
                is_aborted=Mock(return_value=False),
                finish_reason=None,
                generated_text="",
                close=AsyncMock(),
            )
            engine = self._engine(request)
            await queue.put(
                TokenOutput(request_id="request-1", token_id=1, token_text="1")
            )
            generator = engine.stream_request(request, timeout=0.01)
            first = await anext(generator)

            self.assertEqual(first.token_id, 1)
            self.assertEqual(queue.qsize(), 0)
            await asyncio.sleep(0.05)
            self.assertEqual(queue.qsize(), 0)

            await queue.put(
                TokenOutput(request_id="request-1", token_id=2, token_text="2")
            )
            second = await anext(generator)
            self.assertEqual(second.token_id, 2)
            await generator.aclose()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
