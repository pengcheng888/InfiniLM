import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from infinilm.base_config import BaseConfig
from infinilm.cluster.aggregated.launch.config import load_cluster_launch_config
from infinilm.cluster.aggregated.launch.launcher import build_launch_commands
from infinilm.cluster.disaggregated.launch.config import load_pd_launch_config
from infinilm.cluster.disaggregated.launch.launcher import build_pd_launch_commands
from infinilm.config.engine_config import EngineConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
AGGREGATED_CONFIG = (
    REPO_ROOT / "python/infinilm/cluster/aggregated/launch/configs/Qwen3-0.6B.yaml"
)
PD_CONFIG = (
    REPO_ROOT / "python/infinilm/cluster/disaggregated/launch/configs/Qwen3-0.6B.yaml"
)


class GpuMemoryConfigTest(unittest.TestCase):
    def test_base_config_uses_utilization_by_default(self):
        """CLI 默认使用显存比例推导 KV 容量。"""
        config = BaseConfig(["--model", "/models/qwen"])

        self.assertEqual(config.gpu_memory_utilization, 0.85)
        self.assertFalse(hasattr(config, "num_blocks"))

    def test_base_config_accepts_custom_utilization(self):
        """用户可覆盖显存比例，但不能再指定固定 blocks。"""
        config = BaseConfig(
            [
                "--model",
                "/models/qwen",
                "--gpu-memory-utilization",
                "0.85",
            ]
        )

        self.assertEqual(config.gpu_memory_utilization, 0.85)

    def test_base_config_rejects_num_blocks(self):
        """旧 --num-blocks 参数必须显式失败并指向新参数。"""
        with self.assertRaisesRegex(ValueError, "--gpu-memory-utilization"):
            BaseConfig(
                [
                    "--model",
                    "/models/qwen",
                    "--num-blocks",
                    "500",
                ]
            )

    def test_engine_config_has_no_user_capacity(self):
        """EngineConfig 不再暴露用户输入的 num_blocks。"""
        config = EngineConfig(model_path="/models/qwen")

        self.assertEqual(config.gpu_memory_utilization, 0.85)
        self.assertEqual(config.effective_num_blocks, 0)
        self.assertFalse(hasattr(config, "num_blocks"))

        for utilization in (0.0, 1.01):
            with self.subTest(utilization=utilization):
                with self.assertRaisesRegex(
                    ValueError, "gpu_memory_utilization must be in"
                ):
                    EngineConfig(
                        model_path="/models/qwen",
                        gpu_memory_utilization=utilization,
                    )

    def test_aggregated_yaml_rejects_num_blocks(self):
        """旧 engine.num_blocks 字段必须显式报错，避免被静默忽略。"""
        source = AGGREGATED_CONFIG.read_text(encoding="utf-8").replace(
            "  gpu_memory_utilization: 0.85",
            "  num_blocks: 500",
        )
        with TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "cluster.yaml"
            config_path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "engine.num_blocks"):
                load_cluster_launch_config(config_path)

    def test_aggregated_yaml_builds_worker_argument(self):
        """聚合式 YAML 始终传递 gpu-memory-utilization。"""
        config = load_cluster_launch_config(AGGREGATED_CONFIG)
        command = build_launch_commands(
            config,
            role="worker",
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            python_executable="python",
        )[0]

        self.assertNotIn("--num-blocks", command.argv)
        self.assertIn("--gpu-memory-utilization", command.argv)
        self.assertEqual(
            command.argv[command.argv.index("--gpu-memory-utilization") + 1],
            "0.85",
        )

    def test_pd_yaml_rejects_num_blocks(self):
        """PD YAML 同样拒绝固定 blocks。"""
        source = PD_CONFIG.read_text(encoding="utf-8").replace(
            "  gpu_memory_utilization: 0.85",
            "  num_blocks: 500",
        )
        with TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "cluster.yaml"
            config_path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "engine.num_blocks"):
                load_pd_launch_config(config_path)

    def test_pd_yaml_builds_worker_argument(self):
        """PD YAML 始终传递 gpu-memory-utilization。"""
        config = load_pd_launch_config(PD_CONFIG)
        command = build_pd_launch_commands(
            config,
            role="prefill",
            environ={"CUDA_VISIBLE_DEVICES": "4", "ENGINE_ID": "auto-test"},
            python_executable="python",
        )[0]

        self.assertNotIn("--num-blocks", command.argv)
        self.assertEqual(
            command.argv[command.argv.index("--gpu-memory-utilization") + 1],
            "0.85",
        )


if __name__ == "__main__":
    unittest.main()
