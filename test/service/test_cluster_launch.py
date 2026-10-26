import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from infinilm.cluster.aggregated.launch.config import load_cluster_launch_config
from infinilm.cluster.aggregated.launch.launcher import (
    build_launch_commands,
    validate_cuda_visible_devices,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
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
        self.assertIn('"infinilm.pd-cluster-launch"', pyproject)
        self.assertIn('"infinilm.cluster" = ["readme.md"]', pyproject)
        self.assertIn('"infinilm.cluster.aggregated" = ["*.sh"]', pyproject)
        self.assertIn('"infinilm.cluster.disaggregated" = ["*.sh"]', pyproject)
        self.assertIn(
            '"infinilm.cluster.disaggregated.launch" = ["configs/*.yaml", "README.md"]',
            pyproject,
        )

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

    def test_frontend_role_builds_only_frontend_command(self):
        """--role frontend 只生成统一入口进程，不生成 worker。"""
        config = load_cluster_launch_config(EXAMPLE_CONFIG)
        commands = build_launch_commands(
            config,
            role="frontend",
            environ={"CUDA_VISIBLE_DEVICES": "0,1,2"},
            python_executable="python",
        )

        self.assertEqual([command.name for command in commands], ["frontend"])
        self.assertIn(
            "infinilm.cluster.aggregated.frontend", commands[0].argv
        )
        frontend_argv = commands[0].argv
        # 三副本部署的等待数量来自 YAML，而不是 GPU 列表长度。
        self.assertEqual(
            frontend_argv[frontend_argv.index("--router-min-initial-workers") + 1],
            "3",
        )

    def test_worker_role_builds_exactly_one_worker(self):
        """多 GPU 列表原样传给唯一 worker，不再按 GPU 数量展开副本。"""
        config = load_cluster_launch_config(EXAMPLE_CONFIG)
        commands = build_launch_commands(
            config,
            role="worker",
            environ={"CUDA_VISIBLE_DEVICES": "0,1,2"},
            python_executable="python",
        )

        self.assertEqual([command.name for command in commands], ["worker"])
        self.assertEqual(
            commands[0].environment["CUDA_VISIBLE_DEVICES"], "0,1,2"
        )
        self.assertIn("infinilm.cluster.worker", commands[0].argv)
        self.assertEqual(
            commands[0].argv[commands[0].argv.index("--port") + 1], "0"
        )

    def test_fixed_base_http_port_builds_one_deterministic_port(self):
        """固定端口模式没有 worker 序号，只叠加固定 offset。"""
        with TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "cluster.yaml"
            config_path.write_text(
                EXAMPLE_CONFIG.read_text().replace("base_http_port: 0", "base_http_port: 18081"),
                encoding="utf-8",
            )
            config = load_cluster_launch_config(config_path)

        commands = build_launch_commands(
            config,
            role="worker",
            environ={"CUDA_VISIBLE_DEVICES": "0,1,2"},
            python_executable="python",
        )
        self.assertEqual(
            commands[0].argv[commands[0].argv.index("--port") + 1], "18091"
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

    def test_cuda_visible_devices_controls_visibility_not_replica_count(self):
        """GPU 列表只校验可见性，逗号数量没有副本语义。"""
        self.assertEqual(validate_cuda_visible_devices("0"), "0")
        self.assertEqual(
            validate_cuda_visible_devices("0, 1,2"), "0, 1,2"
        )
        for value in (None, "", " ", "-1", "0,,"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_cuda_visible_devices(value)

    def test_worker_replicas_is_rejected(self):
        """禁止 YAML 声明副本数，避免与重复执行 worker 的方式冲突。"""
        source = EXAMPLE_CONFIG.read_text(encoding="utf-8").replace(
            "  # 注意：不允许配置 replicas；每次执行 --role worker 只启动一个进程。",
            "  replicas: 3",
        )
        with mock.patch.object(Path, "read_text", return_value=source):
            with self.assertRaisesRegex(ValueError, "workers.replicas"):
                load_cluster_launch_config(EXAMPLE_CONFIG)

    def test_boolean_engine_fields_become_flags(self):
        """验证布尔引擎参数会转换为 worker CLI 的开关。"""
        config = load_cluster_launch_config(EXAMPLE_CONFIG)
        config = replace(config, engine=replace(config.engine, enable_graph=False))
        commands = build_launch_commands(
            config,
            role="worker",
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            python_executable="python",
        )

        worker_argv = commands[0].argv
        self.assertIn("--enable-paged-attn", worker_argv)
        self.assertNotIn("--enable-graph", worker_argv)

    def test_module_dry_run_requires_role_and_prints_one_command(self):
        """python -m 入口必须显式选择角色，并只输出该角色命令。"""
        environment = os.environ.copy()
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
                "--role",
                "worker",
                "--dry-run",
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("infinilm.cluster.worker", result.stdout)
        self.assertEqual(result.stdout.count("infinilm.cluster.worker"), 1)

        missing_role = subprocess.run(
            [
                sys.executable,
                "-m",
                "infinilm.cluster.aggregated.launch",
                "--config",
                str(EXAMPLE_CONFIG),
                "--dry-run",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(missing_role.returncode, 2)


if __name__ == "__main__":
    unittest.main()
