import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

from infinilm.llm import AsyncLLMEngine, FinishReason
from infinilm.llm.request import (
    InferenceRequest,
    RequestEntranceTiming,
    TokenOutput,
)
from infinilm.server.inference_server import InferenceServer


class ChatTemplateKwargsTest(unittest.TestCase):
    def test_async_engine_forwards_kwargs_to_apply_chat_template(self):
        """AsyncLLMEngine 不再把 chat_template_kwargs 吞进未使用的 kwargs。"""
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        engine.config = SimpleNamespace(max_tokens=2)
        apply_template = Mock(return_value="formatted prompt")
        add_request = Mock()
        engine.engine = SimpleNamespace(
            apply_chat_template=apply_template,
            process=lambda *args, **kwargs: {
                "input_ids": SimpleNamespace(
                    flatten=lambda: SimpleNamespace(tolist=lambda: [1, 2])
                )
            },
            processor=SimpleNamespace(
                get_mm_token_index_list=lambda *args, **kwargs: []
            ),
            eos_token_ids=[0],
            add_request=add_request,
        )

        with patch.object(
            InferenceRequest,
            "output_queue",
            new_callable=PropertyMock,
        ) as output_queue:
            output_queue.return_value = object()
            request = engine.add_request(
                messages=[{"role": "user", "content": "hello"}],
                chat_template_kwargs={"enable_thinking": False},
            )

        apply_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertEqual(request.prompt_token_ids, (1, 2))
        add_request.assert_called_once_with(request)

    def test_async_chat_request_forwards_chat_template_kwargs(self):
        """add_chat_request 必须显式转发模板参数，而不是静默丢弃。"""
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)

        with patch.object(
            AsyncLLMEngine,
            "add_request",
        ) as add_request:
            engine.add_chat_request(
                messages=[{"role": "user", "content": "hello"}],
                add_generation_prompt=False,
                chat_template_kwargs={"enable_thinking": False},
                server_received_at_ns=123,
            )

        add_request.assert_called_once_with(
            messages=[{"role": "user", "content": "hello"}],
            apply_chat_template=True,
            add_generation_prompt=False,
            sampling_params=None,
            request_id=None,
            request_data=None,
            chat_template_kwargs={"enable_thinking": False},
            server_received_at_ns=123,
            entrance_timing=None,
        )

    def test_chat_server_passes_template_kwargs_and_timing(self):
        """OpenAI chat 入口同时保留模板参数和请求接收时间。"""

        async def run():
            request = SimpleNamespace(
                get_prompt_length=lambda: 2,
                get_num_generated_tokens=lambda: 1,
                get_total_length=lambda: 3,
                finish_reason=FinishReason.LENGTH,
                is_finished=lambda: True,
            )

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
                add_chat_request_async=AsyncMock(return_value=request),
                stream_request=stream_request,
                engine=SimpleNamespace(eos_token_ids=[]),
            )
            async def is_disconnected():
                return False

            http_request = SimpleNamespace(is_disconnected=is_disconnected)
            data = {
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            entrance_timing = RequestEntranceTiming()

            with patch.object(
                server,
                "_build_sampling_params",
                return_value=SimpleNamespace(ignore_eos=False),
            ), patch.object(
                InferenceServer,
                "_cache_stats",
                return_value={},
            ):
                response = await server._chat(
                    "request-1",
                    data,
                    http_request,
                    123,
                    entrance_timing,
                )

            self.assertEqual(response["choices"][0]["finish_reason"], "length")
            server.engine.add_chat_request_async.assert_awaited_once_with(
                messages=[{"role": "user", "content": "hello"}],
                sampling_params=SimpleNamespace(ignore_eos=False),
                request_id="request-1",
                request_data=data,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
                server_received_at_ns=123,
                entrance_timing=entrance_timing,
            )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
