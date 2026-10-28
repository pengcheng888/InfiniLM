import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from infinilm.config.engine_config import EngineConfig
from infinilm.config.kv_transfer import KVTransferConfig
from infinilm.llm import llm as llm_module
from infinilm.llm.llm import LLMEngine
from infinilm.llm.model_runner import communication_model_runner as runner_module
from infinilm.llm.model_runner import create_model_runner
from infinilm.llm.model_runner import model_runner as model_runner_module
from infinilm.llm.model_runner.communication_model_runner import (
    CommunicationModelRunner,
)
from infinilm.llm.request import RequestStepTiming
from infinilm.llm.request_timing.request_timing import (
    RequestTimingFileRecorder,
)
from infinilm.llm.request_timing.validate_logs import validate_timing_root


class ModelRunnerCommunicationModeTest(unittest.TestCase):
    def setUp(self):
        self.hf_config = {
            "model_type": "qwen3",
            "max_position_embeddings": 4096,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "num_attention_heads": 8,
            "hidden_size": 128,
            "head_dim": 16,
        }

    def _config(self, **overrides) -> EngineConfig:
        values = {
            "model_path": "/models/qwen3",
            "device": "cpu",
            "cache_type": "paged",
            "max_num_batched_tokens": 1024,
            "max_cache_len": 256,
            "block_size": 128,
        }
        values.update(overrides)
        return EngineConfig(**values)

    def _runner(self, config: EngineConfig) -> CommunicationModelRunner:
        with (
            patch.object(
                model_runner_module,
                "read_hf_config",
                return_value=self.hf_config,
            ),
            patch.object(
                llm_module,
                "read_hf_config",
                return_value=self.hf_config,
            ),
            patch.object(
                model_runner_module,
                "InferEngine",
                side_effect=AssertionError("InferEngine must not be created"),
            ),
            patch.object(
                model_runner_module,
                "load_model_state_dict_by_file",
                side_effect=AssertionError("weights must not be loaded"),
            ),
        ):
            return create_model_runner(
                config,
                hf_config=self.hf_config,
                initialize_processor=False,
            )

    def test_factory_selects_normal_model_runner_by_default(self):
        """未开启通信测试环境变量时，工厂必须走普通 ModelRunner。"""
        normal_runner = object()

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(
                model_runner_module,
                "ModelRunner",
                return_value=normal_runner,
            ) as normal_factory,
        ):
            runner = create_model_runner(
                self._config(),
                hf_config=self.hf_config,
                initialize_processor=False,
            )

        self.assertIs(runner, normal_runner)
        normal_factory.assert_called_once_with(
            self._config(),
            initialize_processor=False,
        )

    def test_communication_mode_skips_engine_and_returns_constant_tokens(self):
        """通信测试模式不创建引擎、不加载权重，forward 返回常数 token。"""
        with patch.dict(
            "os.environ",
            {
                "INFINILM_CLUSTER_COMMUNICATION_TEST": "1",
                "INFINILM_CLUSTER_COMMUNICATION_TEST_TOKEN_ID": "7",
                "INFINILM_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS": "2",
            },
        ):
            runner = self._runner(self._config())

        self.assertTrue(runner.communication_test_mode)
        self.assertIsNone(runner.model_engine)
        self.assertIsNone(runner.pipeline_control)
        self.assertEqual(runner.model_type, "qwen3")
        self.assertEqual(runner.eos_token_id, [])
        self.assertEqual(runner.config.effective_num_blocks, 2)

        scheduler_output = SimpleNamespace(
            num_requests=3,
            scheduled_requests=[
                SimpleNamespace(request_id="req-0"),
                SimpleNamespace(request_id="req-1"),
                SimpleNamespace(request_id="req-2"),
            ],
        )
        output = runner.execute_model(scheduler_output)

        self.assertEqual(output.req_ids, ["req-0", "req-1", "req-2"])
        self.assertEqual(output.sampled_token_ids, [7, 7, 7])

    def test_communication_mode_produces_valid_schema_v4_timing(self):
        """通信测试假 forward 也生成完整 step timing 并通过 v4 校验。"""
        with patch.dict(
            "os.environ",
            {
                "INFINILM_CLUSTER_COMMUNICATION_TEST": "1",
                "INFINILM_CLUSTER_COMMUNICATION_TEST_TOKEN_ID": "7",
                "INFINILM_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS": "2",
            },
        ):
            runner = self._runner(self._config())

        timing_steps = [
            RequestStepTiming(
                request_id="communication-request",
                role="prefill",
                step_index=0,
                phase="prefill",
                batch_size=1,
                server_received_at_ns=100,
                scheduler_enqueued_at_ns=110,
                scheduler_polled_at_ns=120,
                scheduler_batch_ready_at_ns=130,
                scheduler_waiting_queue_depth_before=1,
                scheduler_running_queue_depth_before=0,
                scheduler_remote_kv_requests_before=0,
                scheduler_waiting_queue_depth_after=0,
                scheduler_running_queue_depth_after=0,
                scheduler_remote_kv_requests_after=0,
            )
        ]
        scheduler_output = SimpleNamespace(
            num_requests=1,
            scheduled_requests=[
                SimpleNamespace(
                    request_id="communication-request",
                    timing_steps=timing_steps,
                )
            ],
        )

        with TemporaryDirectory() as temporary_dir:
            recorder = RequestTimingFileRecorder(root=Path(temporary_dir))
            output = runner.execute_model(scheduler_output)
            recorder.record_steps(timing_steps)
            recorder.close()

            validation = validate_timing_root(Path(temporary_dir))
            timing_file = next(Path(temporary_dir).glob("*/*.jsonl"))
            record = json.loads(timing_file.read_text(encoding="utf-8"))

        timing = timing_steps[0]
        self.assertIsNotNone(output.forward_timing)
        self.assertTrue(
            output.forward_timing.forward_batch_id.startswith(
                "communication-forward-"
            )
        )
        self.assertIsNotNone(timing.engine_forward_started_at_ns)
        self.assertIsNotNone(timing.engine_forward_completed_at_ns)
        self.assertEqual(
            timing.forward_batch_id,
            output.forward_timing.forward_batch_id,
        )
        self.assertEqual(record["schema_version"], 4)
        self.assertEqual(record["forward_batch_id"], timing.forward_batch_id)
        self.assertEqual(validation["invalid_file_count"], 0)

    def test_communication_mode_rejects_parallelism(self):
        """该模式只覆盖单进程上层链路，不创建 TP/PP 通信面。"""
        with patch.dict(
            "os.environ",
            {"INFINILM_CLUSTER_COMMUNICATION_TEST": "true"},
        ):
            with self.assertRaisesRegex(
                ValueError, "only supports tp=1 and pp=1"
            ):
                self._runner(self._config(tensor_parallel_size=2))

            with self.assertRaisesRegex(
                ValueError, "only supports tp=1 and pp=1"
            ):
                self._runner(
                    self._config(
                        pipeline_parallel_size=2,
                        pipeline_parallel_stage=1,
                    )
                )

    def test_communication_mode_uses_kv_blocks_environment_directly(self):
        """KV block 数完全由环境变量决定，不再叠加 max_cache_len 下限。"""
        with patch.dict(
            "os.environ",
            {
                "INFINILM_CLUSTER_COMMUNICATION_TEST": "1",
                "INFINILM_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS": "1",
            },
        ):
            runner = self._runner(self._config())

        self.assertEqual(runner.config.effective_num_blocks, 1)

    def test_llm_engine_initializes_without_infer_engine(self):
        """LLMEngine 在通信测试模式下不依赖真实 InferEngine。"""
        tokenizer = object()
        processor = SimpleNamespace(get_tokenizer=lambda: tokenizer)

        with (
            patch.dict(
                "os.environ",
                {
                    "INFINILM_CLUSTER_COMMUNICATION_TEST": "yes",
                    "INFINILM_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS": "2",
                },
            ),
            patch.object(
                model_runner_module,
                "read_hf_config",
                return_value=self.hf_config,
            ),
            patch.object(
                llm_module,
                "read_hf_config",
                return_value=self.hf_config,
            ),
            patch.object(
                runner_module,
                "AutoInfinilmProcessor",
                SimpleNamespace(from_pretrained=lambda _: processor),
            ),
            patch.object(
                model_runner_module,
                "InferEngine",
                side_effect=AssertionError("InferEngine must not be created"),
            ),
            patch.object(
                model_runner_module,
                "load_model_state_dict_by_file",
                side_effect=AssertionError("weights must not be loaded"),
            ),
        ):
            engine = LLMEngine(self._config())

        self.assertIsNone(engine.model_runner.model_engine)
        self.assertIs(engine.tokenizer, tokenizer)
        self.assertEqual(engine.scheduler.cache_manager.num_blocks, 2)
        self.assertEqual(engine.eos_token_ids, [])

    def test_communication_mode_registers_dedicated_kv_tensors(self):
        """P/D 通信测试会注册专用 KV 地址，但仍不创建真实推理引擎。"""
        tensors = {
            "rank.0.model.layers.0.self_attn.attn": object(),
            "rank.0.model.layers.1.self_attn.attn": object(),
        }
        connector = SimpleNamespace(
            register_kv_caches=lambda value: None,
            shutdown=lambda: None,
        )

        def check_register(value):
            self.assertIs(value, tensors)
            connector.registered = True

        def check_shutdown():
            connector.shutdown_called = True

        connector.register_kv_caches = check_register
        connector.shutdown = check_shutdown

        with (
            patch.dict(
                "os.environ",
                {
                    "INFINILM_CLUSTER_COMMUNICATION_TEST": "1",
                    "INFINILM_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS": "3",
                },
            ),
            patch.object(
                runner_module,
                "create_communication_test_kv_caches",
                return_value=tensors,
            ),
            patch.object(
                runner_module.KVConnectorFactory,
                "create_connector",
                return_value=connector,
            ) as factory_mock,
        ):
            runner = self._runner(
                self._config(
                    kv_transfer_config=KVTransferConfig(
                        kv_connector="MooncakeConnector",
                        kv_role="kv_producer",
                    )
                )
            )
            self.assertIs(runner._communication_test_kv_caches, tensors)
            self.assertIs(runner.kv_connector, connector)
            self.assertTrue(connector.registered)
            self.assertEqual(runner.config.effective_num_blocks, 3)

            factory_mock.assert_called_once_with(
                connector_name="MooncakeConnector",
                role=runner_module.KVConnectorRole.WORKER,
                kv_transfer_config=runner.kv_transfer_config,
            )

            runner.close()
            self.assertIsNone(runner._communication_test_kv_caches)
            self.assertTrue(connector.shutdown_called)

    def test_communication_mode_rejects_non_mooncake_connector(self):
        """当前测试 KV 布局只适配 Mooncake 的注册和传输协议。"""
        with (
            patch.dict(
                "os.environ",
                {
                    "INFINILM_CLUSTER_COMMUNICATION_TEST": "1",
                    "INFINILM_CLUSTER_COMMUNICATION_TEST_KV_BLOCKS": "2",
                },
            ),
            patch.object(
                runner_module.KVConnectorFactory,
                "create_connector",
                side_effect=AssertionError("connector must not be created"),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "only supports MooncakeConnector"
            ):
                self._runner(
                    self._config(
                        kv_transfer_config=KVTransferConfig(
                            kv_connector="UnknownConnector",
                            kv_role="kv_producer",
                        )
                    )
                )

    def test_communication_test_kv_shapes_follow_attention_backend(self):
        """flash-attn 是 BSHD，paged-attn 是 BHSD，两者都要求 bfloat16。"""
        device = object()
        allocated = []

        def fake_zeros(shape, dtype, device):
            allocated.append((shape, dtype, device))
            return SimpleNamespace(shape=shape)

        with patch.object(
            runner_module.infinicore,
            "zeros",
            side_effect=fake_zeros,
        ):
            paged_caches = (
                runner_module.create_communication_test_kv_caches(
                    self.hf_config,
                    self._config(),
                    3,
                    device,
                )
            )
            flash_caches = (
                runner_module.create_communication_test_kv_caches(
                    self.hf_config,
                    self._config(attn_backend="flash-attn"),
                    3,
                    device,
                )
            )

        self.assertEqual(
            list(paged_caches),
            [
                "rank.0.model.layers.0.self_attn.attn",
                "rank.0.model.layers.1.self_attn.attn",
            ],
        )
        self.assertEqual(list(flash_caches), list(paged_caches))
        bfloat16 = runner_module.infinicore.bfloat16
        self.assertEqual(allocated[0], ((2, 3, 4, 128, 16), bfloat16, device))
        self.assertEqual(allocated[2], ((2, 3, 128, 4, 16), bfloat16, device))
        self.assertEqual(
            runner_module.infinicore.bfloat16,
            allocated[0][1],
        )


if __name__ == "__main__":
    unittest.main()
