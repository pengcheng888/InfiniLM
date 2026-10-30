import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from infinilm.llm import FinishReason
from infinilm.llm.request import RequestEntranceTiming, TokenOutput
from infinilm.server.inference_server import InferenceServer


class StreamPrestartTest(unittest.TestCase):
    @staticmethod
    def _request():
        return SimpleNamespace(
            get_prompt_length=lambda: 2,
            get_num_generated_tokens=lambda: 1,
            get_total_length=lambda: 3,
            num_local_prefix_cache_hit_tokens=0,
            finish_reason=FinishReason.LENGTH,
            is_finished=lambda: True,
        )

    def test_start_stream_chat_request_submits_work_before_response(self):
        """route 在返回 StreamingResponse 前提交 tokenizer executor work。"""

        async def run():
            tokenized_future = asyncio.get_running_loop().create_future()
            started = SimpleNamespace(tokenized_future=tokenized_future)
            start_preprocessing = Mock(return_value=started)
            server = InferenceServer.__new__(InferenceServer)
            server.engine = SimpleNamespace(
                start_chat_preprocessing=start_preprocessing
            )
            entrance_timing = RequestEntranceTiming()
            data = {"messages": [{"role": "user", "content": "hello"}]}

            with patch.object(
                server,
                "_build_sampling_params",
                return_value=SimpleNamespace(ignore_eos=False),
            ):
                result, sampling_params = server._start_stream_chat_request(
                    request_id="request-1",
                    data=data,
                    server_received_at_ns=123,
                    entrance_timing=entrance_timing,
                )

            self.assertIs(result, started)
            self.assertFalse(tokenized_future.done())
            self.assertIsNotNone(entrance_timing.handler_started_at_ns)
            start_preprocessing.assert_called_once_with(
                messages=data["messages"],
                request_id="request-1",
                add_generation_prompt=True,
                chat_template_kwargs={},
                server_received_at_ns=123,
                entrance_timing=entrance_timing,
            )

        asyncio.run(run())

    def test_sampling_params_are_validated_before_preprocessing_submit(self):
        """sampling 参数非法时不能提交无主的 tokenizer 后台任务。"""

        async def run():
            server = InferenceServer.__new__(InferenceServer)
            server.engine = SimpleNamespace(start_chat_preprocessing=Mock())
            with (
                patch.object(
                    server,
                    "_build_sampling_params",
                    side_effect=ValueError("invalid sampling parameters"),
                ),
                self.assertRaisesRegex(ValueError, "invalid sampling parameters"),
            ):
                server._start_stream_chat_request(
                    request_id="request-1",
                    data={"messages": [{"role": "user", "content": "hello"}]},
                    server_received_at_ns=123,
                    entrance_timing=RequestEntranceTiming(),
                )

            server.engine.start_chat_preprocessing.assert_not_called()

        asyncio.run(run())

    def test_stream_chat_consumes_prestarted_preprocessing(self):
        """stream handler 直接完成 route 预提交的 tokenizer future。"""

        async def run():
            request = self._request()
            tokenized_future = asyncio.get_running_loop().create_future()
            tokenized_future.set_result(object())
            started = SimpleNamespace(tokenized_future=tokenized_future)
            finish_preprocessing = AsyncMock(return_value=request)

            async def stream_request(*args, **kwargs):
                yield TokenOutput(
                    request_id="request-1",
                    token_id=1,
                    token_text="ok",
                    finished=True,
                    finish_reason=FinishReason.LENGTH,
                )

            server = InferenceServer.__new__(InferenceServer)
            server.model_id = "model"
            server.engine = SimpleNamespace(
                finish_chat_preprocessing=finish_preprocessing,
                stream_request=stream_request,
                engine=SimpleNamespace(eos_token_ids=[]),
            )
            sampling_params = SimpleNamespace(ignore_eos=False)
            data = {"messages": [{"role": "user", "content": "hello"}]}

            chunks = [
                chunk
                async for chunk in server._stream_chat(
                    "request-1",
                    data,
                    SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
                    123,
                    RequestEntranceTiming(),
                    started,
                    sampling_params,
                )
            ]

            self.assertTrue(tokenized_future.done())
            finish_preprocessing.assert_awaited_once_with(
                started,
                sampling_params=sampling_params,
                request_data=data,
            )
            self.assertIn('"finish_reason": "length"', chunks[-2])

        asyncio.run(run())

    def test_stream_cancellation_cancels_pending_prestarted_work(self):
        """外层 stream 取消时不能遗留已提交的 tokenizer future。"""

        async def run():
            server = InferenceServer.__new__(InferenceServer)
            server.model_id = "model"

            async def finish_preprocessing(started, **kwargs):
                return await started.tokenized_future

            server.engine = SimpleNamespace(
                finish_chat_preprocessing=finish_preprocessing,
                engine=SimpleNamespace(eos_token_ids=[]),
            )
            prestarted = SimpleNamespace(
                tokenized_future=asyncio.create_task(asyncio.sleep(10))
            )
            generator = server._stream_chat(
                "request-1",
                {"messages": [{"role": "user", "content": "hello"}]},
                SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
                123,
                RequestEntranceTiming(),
                prestarted,
                SimpleNamespace(ignore_eos=False),
            )
            reader = asyncio.create_task(anext(generator))
            await asyncio.sleep(0)
            reader.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await reader

            await asyncio.sleep(0)
            self.assertTrue(prestarted.tokenized_future.cancelled())

        asyncio.run(run())

if __name__ == "__main__":
    unittest.main()
