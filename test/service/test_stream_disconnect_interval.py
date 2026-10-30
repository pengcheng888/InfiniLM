import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from infinilm.llm import FinishReason
from infinilm.llm.request import TokenOutput
from infinilm.server.inference_server import (
    InferenceServer,
    disconnect_check_tokens_from_environment,
)


class StreamDisconnectIntervalTest(unittest.TestCase):
    def test_disconnect_interval_has_safe_diagnostic_values(self):
        """默认保持 32；1024 仅用于输出路径诊断，finish 时仍检查。"""

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(disconnect_check_tokens_from_environment(), 32)
        with patch.dict("os.environ", {"INFINILM_DISCONNECT_CHECK_TOKENS": "1024"}):
            self.assertEqual(disconnect_check_tokens_from_environment(), 1024)
        with patch.dict("os.environ", {"INFINILM_DISCONNECT_CHECK_TOKENS": "1"}):
            with self.assertRaisesRegex(RuntimeError, "must be 32 or 1024"):
                disconnect_check_tokens_from_environment()

    def test_stream_disconnect_is_checked_every_32_token_outputs(self):
        """流式输出按周期检查断开，避免每个 token 都进入 ASGI receive。"""

        async def run():
            request = SimpleNamespace(
                get_prompt_length=lambda: 2,
                num_local_prefix_cache_hit_tokens=0,
                finish_reason=FinishReason.LENGTH,
                is_finished=lambda: True,
            )

            async def stream_request(*args, **kwargs):
                for index in range(64):
                    yield TokenOutput(
                        request_id="request-1",
                        token_id=index + 1,
                        token_text="a",
                        finished=False,
                        finish_reason=None,
                    )
                yield TokenOutput(
                    request_id="request-1",
                    token_id=65,
                    token_text="a",
                    finished=True,
                    finish_reason=FinishReason.LENGTH,
                )

            server = InferenceServer.__new__(InferenceServer)
            server.model_id = "model"
            server.engine = SimpleNamespace(
                add_chat_request_async=AsyncMock(return_value=request),
                stream_request=stream_request,
                engine=SimpleNamespace(eos_token_ids=[]),
            )
            http_request = SimpleNamespace(
                is_disconnected=AsyncMock(return_value=False)
            )

            with (
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
                        http_request,
                        123,
                    )
                ]

            self.assertEqual(len(chunks), 19)
            self.assertEqual(http_request.is_disconnected.await_count, 3)

        asyncio.run(run())

    def test_non_stream_disconnect_is_checked_every_32_token_outputs(self):
        """非流式长输出也按周期检查断开，避免每 token 进入 ASGI receive。"""

        async def run():
            request = SimpleNamespace(
                get_prompt_length=lambda: 2,
                get_num_generated_tokens=lambda: 65,
                get_total_length=lambda: 67,
                num_local_prefix_cache_hit_tokens=0,
                finish_reason=FinishReason.LENGTH,
                is_finished=lambda: True,
            )

            async def stream_request(*args, **kwargs):
                for index in range(64):
                    yield TokenOutput(
                        request_id="request-1",
                        token_id=index + 1,
                        token_text="a",
                        finished=False,
                        finish_reason=None,
                    )
                yield TokenOutput(
                    request_id="request-1",
                    token_id=65,
                    token_text="a",
                    finished=True,
                    finish_reason=FinishReason.LENGTH,
                )

            server = InferenceServer.__new__(InferenceServer)
            server.model_id = "model"
            server.engine = SimpleNamespace(
                add_chat_request_async=AsyncMock(return_value=request),
                stream_request=stream_request,
                engine=SimpleNamespace(eos_token_ids=[]),
            )
            http_request = SimpleNamespace(
                is_disconnected=AsyncMock(return_value=False)
            )

            with patch.object(
                server,
                "_build_sampling_params",
                return_value=SimpleNamespace(ignore_eos=False),
            ):
                response = await server._chat(
                    "request-1",
                    {"messages": [{"role": "user", "content": "hello"}]},
                    http_request,
                    123,
                )

            self.assertEqual(response["choices"][0]["message"]["content"], "a" * 65)
            self.assertEqual(http_request.is_disconnected.await_count, 3)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
