import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from infinilm.llm import FinishReason
from infinilm.llm.request import TokenOutput
from infinilm.server.inference_server import (
    InferenceServer,
    sse_flush_tokens_from_environment,
)


class StreamSSEFlushTest(unittest.TestCase):
    @staticmethod
    def _request():
        return SimpleNamespace(
            get_prompt_length=lambda: 2,
            num_local_prefix_cache_hit_tokens=0,
            finish_reason=FinishReason.LENGTH,
            is_finished=lambda: True,
        )

    @staticmethod
    def _content_chunks(chunks):
        contents = []
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk == "data: [DONE]\n\n":
                continue
            payload = json.loads(chunk.removeprefix("data: ").strip())
            content = payload["choices"][0].get("delta", {}).get("content")
            if content is not None:
                contents.append(content)
        return contents

    def test_first_token_flushes_then_tokens_group_by_interval(self):
        """首个 token 立即输出，后续按配置聚合。"""

        async def run():
            async def stream_request(*args, **kwargs):
                for index in range(11):
                    yield TokenOutput(
                        request_id="request-1",
                        token_id=index + 1,
                        token_text=str(index + 1),
                        finished=(index == 10),
                        finish_reason=(
                            FinishReason.LENGTH if index == 10 else None
                        ),
                    )

            server = InferenceServer.__new__(InferenceServer)
            server.model_id = "model"
            server.engine = SimpleNamespace(
                add_chat_request_async=AsyncMock(return_value=self._request()),
                stream_request=stream_request,
                engine=SimpleNamespace(eos_token_ids=[]),
            )

            with (
                patch(
                    "infinilm.server.inference_server.SSE_FLUSH_TOKEN_INTERVAL",
                    interval,
                ),
                patch.object(
                    server,
                    "_build_sampling_params",
                    return_value=SimpleNamespace(ignore_eos=False),
                ),
            ):
                chunks = [
                    chunk
                    async for chunk in server._stream_chat(
                        "request-1",
                        {"messages": [{"role": "user", "content": "hello"}]},
                        SimpleNamespace(
                            is_disconnected=AsyncMock(return_value=False)
                        ),
                        123,
                    )
                ]

            self.assertEqual(self._content_chunks(chunks), expected)
            self.assertIn('"finish_reason": "length"', chunks[-2])

        for interval, expected in (
            (4, ["1", "2345", "6789", "1011"]),
            (8, ["1", "23456789", "1011"]),
        ):
            with self.subTest(interval=interval):
                asyncio.run(run())

    def test_timeout_flushes_pending_tokens_before_terminal_chunk(self):
        """timeout 终止 chunk 前先输出未满组的文本。"""

        async def run():
            async def stream_request(*args, **kwargs):
                yield TokenOutput(
                    request_id="request-1",
                    token_id=1,
                    token_text="1",
                )
                yield TokenOutput(
                    request_id="request-1",
                    token_id=2,
                    token_text="2",
                )
                yield TokenOutput(
                    request_id="request-1",
                    token_id=3,
                    token_text="",
                    finish_reason=FinishReason.TIMEOUT,
                    finished=True,
                )

            server = InferenceServer.__new__(InferenceServer)
            server.model_id = "model"
            server.engine = SimpleNamespace(
                add_chat_request_async=AsyncMock(return_value=self._request()),
                stream_request=stream_request,
                engine=SimpleNamespace(eos_token_ids=[]),
            )

            with patch.object(
                server,
                "_build_sampling_params",
                return_value=SimpleNamespace(ignore_eos=False),
            ):
                chunks = [
                    chunk
                    async for chunk in server._stream_chat(
                        "request-1",
                        {"messages": [{"role": "user", "content": "hello"}]},
                        SimpleNamespace(
                            is_disconnected=AsyncMock(return_value=False)
                        ),
                        123,
                    )
                ]

            self.assertEqual(
                self._content_chunks(chunks),
                ["1", "2", "[Request timeout]"],
            )
            self.assertIn('"finish_reason": "timeout"', chunks[-2])

        asyncio.run(run())

    def test_held_back_empty_token_outputs_count_toward_flush_interval(self):
        """空文本 token 生成后也计数，不能让一个 chunk 覆盖过多生成 token。"""

        async def run():
            async def stream_request(*args, **kwargs):
                token_texts = ["", "A", "B", "C", "D", "E", "F"]
                for index, token_text in enumerate(token_texts):
                    yield TokenOutput(
                        request_id="request-1",
                        token_id=index + 1,
                        token_text=token_text,
                        finished=(index == len(token_texts) - 1),
                        finish_reason=(
                            FinishReason.LENGTH
                            if index == len(token_texts) - 1
                            else None
                        ),
                    )

            server = InferenceServer.__new__(InferenceServer)
            server.model_id = "model"
            server.engine = SimpleNamespace(
                add_chat_request_async=AsyncMock(return_value=self._request()),
                stream_request=stream_request,
                engine=SimpleNamespace(eos_token_ids=[]),
            )

            with patch.object(
                server,
                "_build_sampling_params",
                return_value=SimpleNamespace(ignore_eos=False),
            ):
                chunks = [
                    chunk
                    async for chunk in server._stream_chat(
                        "request-1",
                        {"messages": [{"role": "user", "content": "hello"}]},
                        SimpleNamespace(
                            is_disconnected=AsyncMock(return_value=False)
                        ),
                        123,
                    )
                ]

            self.assertEqual(self._content_chunks(chunks), ["A", "BCDE", "F"])

        asyncio.run(run())

    def test_error_flushes_pending_tokens_before_terminal_chunk(self):
        """异常终止 chunk 前同样先输出未满组文本。"""

        async def run():
            request = self._request()
            request.is_finished = lambda: False

            async def stream_request(*args, **kwargs):
                yield TokenOutput(request_id="request-1", token_id=1, token_text="1")
                yield TokenOutput(request_id="request-1", token_id=2, token_text="2")
                raise RuntimeError("engine failed")

            server = InferenceServer.__new__(InferenceServer)
            server.model_id = "model"
            server.engine = SimpleNamespace(
                add_chat_request_async=AsyncMock(return_value=request),
                stream_request=stream_request,
                engine=SimpleNamespace(eos_token_ids=[]),
                add_aborted_req=Mock(),
            )

            with patch.object(
                server,
                "_build_sampling_params",
                return_value=SimpleNamespace(ignore_eos=False),
            ):
                chunks = [
                    chunk
                    async for chunk in server._stream_chat(
                        "request-1",
                        {"messages": [{"role": "user", "content": "hello"}]},
                        SimpleNamespace(
                            is_disconnected=AsyncMock(return_value=False)
                        ),
                        123,
                    )
                ]

            self.assertEqual(
                self._content_chunks(chunks),
                ["1", "2", "[Error: engine failed]"],
            )
            self.assertIn('"finish_reason": "error"', chunks[-2])
            server.engine.add_aborted_req.assert_called_once()

        asyncio.run(run())

    def test_sse_flush_environment_is_restricted(self):
        """配置只允许 1、4、8，防止无界增大流式粒度。"""
        with patch.dict("os.environ", {"INFINILM_SSE_FLUSH_TOKENS": "8"}):
            self.assertEqual(sse_flush_tokens_from_environment(), 8)
        for raw_value in ("", "0", "2", "invalid"):
            with (
                self.assertRaises(RuntimeError),
                patch.dict("os.environ", {"INFINILM_SSE_FLUSH_TOKENS": raw_value}),
            ):
                sse_flush_tokens_from_environment()


if __name__ == "__main__":
    unittest.main()
