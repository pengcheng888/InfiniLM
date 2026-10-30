import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infinilm.llm.llm import (
    AsyncLLMEngine,
    preprocess_workers_from_environment,
)
from infinilm.llm.request import RequestEntranceTiming
from infinilm.processors.basic_llm_processor import BasicLLMProcessor


class AsyncLLMPreprocessPoolTest(unittest.TestCase):
    @staticmethod
    def _engine(process_side_effect, *, workers: int = 1):
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        engine.config = SimpleNamespace(max_tokens=2)
        engine._preprocess_workers = workers
        engine._preprocess_executor = None
        engine._timing_role = Mock(return_value="aggregated")
        engine.engine = SimpleNamespace(
            apply_chat_template=Mock(return_value="rendered prompt"),
            process=Mock(side_effect=process_side_effect),
            processor=SimpleNamespace(get_mm_token_index_list=Mock(return_value=[])),
            eos_token_ids=[0],
            add_request=Mock(),
        )
        return engine

    @staticmethod
    def _processed_inputs():
        return {
            "input_ids": SimpleNamespace(
                flatten=lambda: SimpleNamespace(tolist=lambda: [1, 2])
            )
        }

    def test_chat_preprocessing_runs_off_event_loop_thread(self):
        """async chat 入口把 processor/token ids 转换移入专用线程。"""
        main_thread = threading.get_ident()
        process_threads = []

        def process(*args, **kwargs):
            process_threads.append(threading.get_ident())
            return self._processed_inputs()

        engine = self._engine(process)

        async def run():
            with patch(
                "infinilm.llm.llm.InferenceRequest.output_queue",
                new_callable=lambda: property(lambda self: object()),
            ):
                request = await engine.add_chat_request_async(
                    messages=[{"role": "user", "content": "hello"}],
                    sampling_params=SimpleNamespace(max_tokens=2),
                )
            return request

        request = asyncio.run(run())
        self.assertEqual(request.prompt_token_ids, (1, 2))
        self.assertEqual(len(process_threads), 1)
        self.assertNotEqual(process_threads[0], main_thread)
        engine.engine.add_request.assert_called_once_with(request)
        engine._preprocess_executor.shutdown(wait=True)

        timing = request.entrance_timing
        self.assertLessEqual(
            timing.preprocessing_started_at_ns,
            timing.chat_template_completed_at_ns,
        )
        self.assertLessEqual(
            timing.chat_template_completed_at_ns,
            timing.multimodal_resolved_at_ns,
        )
        self.assertLessEqual(
            timing.multimodal_resolved_at_ns,
            timing.tokenization_completed_at_ns,
        )
        self.assertLessEqual(
            timing.tokenization_completed_at_ns,
            timing.prompt_token_ids_ready_at_ns,
        )
        self.assertLessEqual(
            timing.prompt_token_ids_ready_at_ns,
            request.request_created_at_ns,
        )

    def test_two_workers_can_preprocess_concurrently(self):
        """workers=2 时两个输入预处理可以同时执行。"""
        barrier = threading.Barrier(2, timeout=2)
        process_threads = []

        def process(*args, **kwargs):
            process_threads.append(threading.get_ident())
            barrier.wait()
            return self._processed_inputs()

        engine = self._engine(process, workers=2)

        async def run():
            with patch(
                "infinilm.llm.llm.InferenceRequest.output_queue",
                new_callable=lambda: property(lambda self: object()),
            ):
                requests = await asyncio.gather(
                    engine.add_chat_request_async(
                        messages=[{"role": "user", "content": "one"}],
                        sampling_params=SimpleNamespace(max_tokens=2),
                    ),
                    engine.add_chat_request_async(
                        messages=[{"role": "user", "content": "two"}],
                        sampling_params=SimpleNamespace(max_tokens=2),
                    ),
                )
            return requests

        requests = asyncio.run(run())
        self.assertEqual([request.prompt_token_ids for request in requests], [(1, 2), (1, 2)])
        self.assertEqual(len(process_threads), 2)
        self.assertNotEqual(process_threads[0], process_threads[1])
        engine._preprocess_executor.shutdown(wait=True)

    def test_text_only_input_uses_fast_tokenizer_path(self):
        """纯文本请求跳过 torch tensor 创建和 flatten/tolist 转换。"""
        engine = self._engine(lambda *args, **kwargs: self._processed_inputs())
        fast_tokenize = Mock(return_value=[3, 4, 5])
        engine.engine.tokenizer = object()
        engine.engine.processor.fast_text_tokenize = fast_tokenize
        engine.engine.processor.supports_fast_text_tokenize = True
        timing = RequestEntranceTiming()

        tokenized = engine._tokenize_chat_input(
            "rendered prompt",
            [{"role": "user", "content": "hello"}],
            timing,
        )

        self.assertEqual(tokenized.prompt_token_ids, [3, 4, 5])
        self.assertIsNone(tokenized.processed_inputs)
        self.assertEqual(tokenized.mm_index_mappings, [])
        self.assertFalse(tokenized.has_multimodal_inputs)
        fast_tokenize.assert_called_once()
        self.assertEqual(fast_tokenize.call_args.args, ("rendered prompt",))
        self.assertIsNotNone(fast_tokenize.call_args.kwargs.get("tokenizer"))
        engine.engine.process.assert_not_called()
        engine.engine.processor.get_mm_token_index_list.assert_not_called()

    def test_multimodal_input_keeps_generic_processor_path(self):
        """多模态请求仍必须走通用 processor 和 token index mapping。"""
        engine = self._engine(lambda *args, **kwargs: self._processed_inputs())
        fast_tokenize = Mock(return_value=[3, 4, 5])
        engine.engine.processor.fast_text_tokenize = fast_tokenize
        engine.engine.processor.supports_fast_text_tokenize = True
        engine.engine.processor.get_mm_token_index_list.return_value = [
            {"start_index": 1}
        ]
        timing = RequestEntranceTiming()
        media_inputs = {
            "images": [object()],
            "image_urls": ["image-id"],
            "videos": [],
            "video_urls": [],
            "audios": [],
            "audio_urls": [],
        }

        with patch(
            "infinilm.llm.llm.resolve_multimodal_inputs",
            return_value=media_inputs,
        ):
            tokenized = engine._tokenize_chat_input(
                "rendered prompt",
                [{"role": "user", "content": [{"type": "image_url"}]}],
                timing,
            )

        self.assertEqual(tokenized.prompt_token_ids, [1, 2])
        self.assertEqual(tokenized.mm_index_mappings, [{"start_index": 1}])
        self.assertTrue(tokenized.has_multimodal_inputs)
        fast_tokenize.assert_not_called()
        engine.engine.process.assert_called_once()
        engine.engine.processor.get_mm_token_index_list.assert_called_once()

    def test_custom_processor_call_must_explicitly_opt_into_fast_path(self):
        """override __call__ 的 processor 不能仅靠继承自动进入 fast path。"""

        class CustomCallProcessor(BasicLLMProcessor):
            def __call__(self, *args, **kwargs):
                return {"input_ids": []}

        class ExplicitFastProcessor(BasicLLMProcessor):
            def fast_text_tokenize(self, prompt, *, tokenizer=None):
                return [1]

        self.assertFalse(
            object.__new__(CustomCallProcessor).supports_fast_text_tokenize
        )
        self.assertTrue(
            object.__new__(ExplicitFastProcessor).supports_fast_text_tokenize
        )

    def test_preprocessing_threads_receive_isolated_tokenizers(self):
        """每个预处理线程克隆 tokenizer，避免与 engine 线程共享可变对象。"""

        async def run():
            engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
            lock = threading.Lock()
            engine.engine = SimpleNamespace(
                tokenizer=object(),
                _tokenizer_lock=lock,
            )
            barrier = threading.Barrier(2, timeout=2)
            tokenizers = []

            def load_tokenizer():
                barrier.wait()
                tokenizers.append(engine._thread_preprocess_tokenizer())

            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(lambda _: load_tokenizer(), range(2)))

            self.assertEqual(len(tokenizers), 2)
            self.assertIsNot(tokenizers[0], tokenizers[1])
            self.assertIsNot(tokenizers[0], engine.engine.tokenizer)
            self.assertIsNot(tokenizers[1], engine.engine.tokenizer)

        asyncio.run(run())

    def test_async_messages_path_rejects_disabled_chat_template(self):
        """async messages 路径不能静默忽略 apply_chat_template=False。"""
        engine = self._engine(lambda *args, **kwargs: self._processed_inputs())

        async def run():
            with self.assertRaisesRegex(
                AssertionError,
                "apply_chat_template needs to be true",
            ):
                await engine.add_request_async(
                    messages=[{"role": "user", "content": "hello"}],
                    apply_chat_template=False,
                )

        asyncio.run(run())
        engine.engine.add_request.assert_not_called()

    def test_preprocess_worker_environment_is_restricted(self):
        """配置只允许 1 或 2，避免默认打开过多 tokenizer 并发。"""
        with patch.dict("os.environ", {"INFINILM_PREPROCESS_WORKERS": "2"}):
            self.assertEqual(preprocess_workers_from_environment(), 2)
        for raw_value in ("", "0", "3", "invalid"):
            with (
                self.assertRaises(RuntimeError),
                patch.dict("os.environ", {"INFINILM_PREPROCESS_WORKERS": raw_value}),
            ):
                preprocess_workers_from_environment()


if __name__ == "__main__":
    unittest.main()
