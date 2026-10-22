import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from infinilm.cluster.aggregated.launch.config import load_cluster_launch_config
from infinilm.cluster.aggregated.launch.launcher import (
    build_launch_commands,
    parse_cuda_visible_devices,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# 使用仓库内示例配置作为启动器行为测试的单一样例。
EXAMPLE_CONFIG = (
    REPO_ROOT
    / "python"
    / "infinilm"
    / "cluster"
    / "aggregated"
    / "launch"
    / "configs"
    / "Qwen3-0.6B.yaml"
)


class ClusterLaunchConfigTest(unittest.TestCase):
    def test_python_package_contains_cluster_runtime_artifacts(self):
        """安装包必须携带 cluster 依赖、入口、文档和启动脚本。"""
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"fastapi>=0.110"', pyproject)
        self.assertIn('"uvicorn>=0.29"', pyproject)
        self.assertIn('"infinilm.pd-frontend"', pyproject)
        self.assertIn('"infinilm.cluster" = ["readme.md"]', pyproject)
        self.assertIn('"infinilm.cluster.aggregated" = ["*.sh"]', pyproject)
        self.assertIn('"infinilm.cluster.disaggregated" = ["*.sh"]', pyproject)

        script_paths = [
            REPO_ROOT / "python/infinilm/cluster/aggregated/run_cluster.sh",
            REPO_ROOT / "python/infinilm/cluster/aggregated/run_worker.sh",
            REPO_ROOT
            / "python/infinilm/cluster/disaggregated/run_pd_discovery_cluster.sh",
            REPO_ROOT / "python/infinilm/cluster/disaggregated/run_pd_frontend.sh",
            REPO_ROOT / "python/infinilm/cluster/disaggregated/run_pd_worker.sh",
        ]
        for script_path in script_paths:
            with self.subTest(script=script_path.name):
                self.assertTrue(script_path.is_file())
                self.assertTrue(
                    os.access(script_path, os.X_OK),
                    f"{script_path} must stay executable",
                )

    def test_example_config_loads_and_builds_three_workers(self):
        """验证三卡环境会展开为 1 个 frontend 和 3 个独立 GPU worker。"""
        config = load_cluster_launch_config(
            EXAMPLE_CONFIG,
            environ={"CUDA_VISIBLE_DEVICES": "0,1,2"},
        )
        commands = build_launch_commands(
            config,
            environ={"CUDA_VISIBLE_DEVICES": "0,1,2"},
            python_executable="python",
        )

        # frontend 必须排在第一位，worker 编号与 GPU 列表顺序一致。
        self.assertEqual(
            [command.name for command in commands],
            [
                "frontend",
                "worker-0",
                "worker-1",
                "worker-2",
            ],
        )
        self.assertEqual(
            [command.environment["CUDA_VISIBLE_DEVICES"] for command in commands[1:]],
            ["0", "1", "2"],
        )
        self.assertEqual(
            commands[1].argv[commands[1].argv.index("--port") + 1], "18091"
        )
        self.assertEqual(
            commands[2].argv[commands[2].argv.index("--port") + 1], "18092"
        )
        self.assertEqual(
            commands[3].argv[commands[3].argv.index("--port") + 1], "18093"
        )

        frontend_argv = commands[0].argv
        # 未显式配置等待数量时，应等待全部可见 GPU 注册完成。
        self.assertEqual(
            frontend_argv[frontend_argv.index("--router-min-initial-workers") + 1],
            "3",
        )

    def test_model_path_can_be_overridden_by_environment(self):
        """验证部署环境可用 MODEL_PATH 覆盖 YAML 中的模型路径。"""
        config = load_cluster_launch_config(
            EXAMPLE_CONFIG,
            environ={
                "MODEL_PATH": "/models/override",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        )

        self.assertEqual(config.model_path, Path("/models/override"))

    def test_cuda_visible_devices_controls_replica_count(self):
        """验证 GPU 列表解析规则和非法值报错行为。"""
        self.assertEqual(parse_cuda_visible_devices("0"), ["0"])
        self.assertEqual(parse_cuda_visible_devices("0, 1,2"), ["0", "1", "2"])

        for value in (None, "", " ", "-1", "0,,"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cuda_visible_devices(value)

    def test_worker_replicas_is_rejected(self):
        """验证第一阶段禁止在 YAML 中配置 workers.replicas。"""
        source = EXAMPLE_CONFIG.read_text(encoding="utf-8")
        # 通过修改示例文本模拟错误配置，不改变仓库中的 YAML 文件。
        source = source.replace(
            "  # 注意：不配置 replicas；worker 数量只来自 CUDA_VISIBLE_DEVICES。",
            "  replicas: 3",
        )
        with mock.patch.object(
            Path,
            "read_text",
            return_value=source,
        ):
            with self.assertRaisesRegex(ValueError, "workers.replicas"):
                load_cluster_launch_config(EXAMPLE_CONFIG)

    def test_boolean_engine_fields_become_flags(self):
        """验证布尔引擎参数会转换为 worker CLI 的开关。"""
        config = load_cluster_launch_config(
            EXAMPLE_CONFIG,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )
        config = replace(config, engine=replace(config.engine, enable_graph=False))
        commands = build_launch_commands(
            config,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            python_executable="python",
        )

        worker_argv = commands[1].argv
        # paged-attn 保持开启，graph 关闭后不应出现对应 flag。
        self.assertIn("--enable-paged-attn", worker_argv)
        self.assertNotIn("--enable-graph", worker_argv)

    def test_module_dry_run_uses_configured_worker_count(self):
        """验证 python -m 入口的 dry-run 输出包含全部 worker 命令。"""
        environment = os.environ.copy()
        # 子进程必须能导入仓库内的 infinilm 包。
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0,1,2",
                "PYTHONPATH": str(REPO_ROOT / "python")
                + (
                    os.pathsep + environment["PYTHONPATH"]
                    if environment.get("PYTHONPATH")
                    else ""
                ),
            }
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "infinilm.cluster.aggregated.launch",
                "--config",
                str(EXAMPLE_CONFIG),
                "--dry-run",
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("infinilm.cluster.aggregated.frontend", result.stdout)
        self.assertEqual(result.stdout.count("infinilm.cluster.worker"), 3)


if __name__ == "__main__":
    unittest.main()
