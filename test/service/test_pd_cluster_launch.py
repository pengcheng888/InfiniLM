import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from infinilm.cluster.disaggregated.launch.config import load_pd_launch_config
from infinilm.cluster.disaggregated.launch.launcher import (
    build_pd_launch_commands,
    validate_cuda_visible_devices,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = (
    REPO_ROOT
    / "python"
    / "infinilm"
    / "cluster"
    / "disaggregated"
    / "launch"
    / "configs"
    / "Qwen3-0.6B.yaml"
)


class PDClusterLaunchTest(unittest.TestCase):
    def test_frontend_role_builds_only_pd_frontend(self):
        """--role frontend 只启动 PD 统一入口，不包含推理 worker。"""
        config = load_pd_launch_config(EXAMPLE_CONFIG)
        commands = build_pd_launch_commands(
            config,
            role="frontend",
            environ={"CUDA_VISIBLE_DEVICES": "4,5"},
            python_executable="python",
        )

        self.assertEqual([command.name for command in commands], ["frontend"])
        self.assertIn(
            "infinilm.cluster.disaggregated.frontend", commands[0].argv
        )
        self.assertNotIn("CUDA_VISIBLE_DEVICES", commands[0].environment)

    def test_prefill_role_builds_one_auto_port_producer(self):
        """--role prefill 只启动一个进程，并生成 kv_producer 配置。"""
        config = load_pd_launch_config(EXAMPLE_CONFIG)
        commands = build_pd_launch_commands(
            config,
            role="prefill",
            environ={"CUDA_VISIBLE_DEVICES": "0,1"},
            python_executable="python",
        )

        self.assertEqual([command.name for command in commands], ["prefill"])
        command = commands[0]
        self.assertEqual(command.environment["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(
            command.argv[command.argv.index("--disaggregation-mode") + 1],
            "prefill",
        )
        self.assertEqual(
            command.argv[command.argv.index("--port") + 1], "0"
        )
        self.assertEqual(
            command.argv[command.argv.index("--mooncake-bootstrap-port") + 1],
            "0",
        )
        kv_config = json.loads(
            command.argv[command.argv.index("--kv-transfer-config") + 1]
        )
        self.assertEqual(kv_config["kv_role"], "kv_producer")
        self.assertEqual(kv_config["kv_connector"], "MooncakeConnector")
        self.assertIn("prefill-auto-", kv_config["engine_id"])

    def test_decode_role_builds_one_auto_port_consumer(self):
        """--role decode 只启动一个进程，并生成 kv_consumer 配置。"""
        config = load_pd_launch_config(EXAMPLE_CONFIG)
        commands = build_pd_launch_commands(
            config,
            role="decode",
            environ={
                "CUDA_VISIBLE_DEVICES": "2,3",
                "ENGINE_ID": "decode-explicit",
            },
            python_executable="python",
        )

        self.assertEqual([command.name for command in commands], ["decode"])
        kv_config = json.loads(
            commands[0].argv[commands[0].argv.index("--kv-transfer-config") + 1]
        )
        self.assertEqual(kv_config["kv_role"], "kv_consumer")
        self.assertEqual(kv_config["engine_id"], "decode-explicit")
        self.assertEqual(
            commands[0].environment["INFINILM_HOST_IP"], "127.0.0.1"
        )

    def test_fixed_ports_and_engine_id_are_deterministic(self):
        """固定端口模式按固定偏移计算，engine ID 可来自 YAML。"""
        source = EXAMPLE_CONFIG.read_text(encoding="utf-8")
        source = source.replace("base_http_port: 0", "base_http_port: 18521")
        source = source.replace(
            "base_bootstrap_port: 0", "base_bootstrap_port: 18528"
        )
        source = source.replace(
            "  # 不允许配置 replicas；每次执行一个 worker 角色只启动一个进程。",
            "  engine_id: fixed-engine",
        )
        with mock.patch.object(Path, "read_text", return_value=source):
            config = load_pd_launch_config(EXAMPLE_CONFIG)

        command = build_pd_launch_commands(
            config,
            role="prefill",
            environ={"CUDA_VISIBLE_DEVICES": "4"},
            python_executable="python",
        )[0]
        self.assertEqual(command.argv[command.argv.index("--port") + 1], "18521")
        self.assertEqual(
            command.argv[command.argv.index("--mooncake-bootstrap-port") + 1],
            "18528",
        )
        kv_config = json.loads(
            command.argv[command.argv.index("--kv-transfer-config") + 1]
        )
        self.assertEqual(kv_config["engine_id"], "fixed-engine")

    def test_worker_replicas_is_rejected(self):
        """PD YAML 同样禁止声明 replicas，副本由重复执行命令决定。"""
        source = EXAMPLE_CONFIG.read_text(encoding="utf-8").replace(
            "  # 不允许配置 replicas；每次执行一个 worker 角色只启动一个进程。",
            "  replicas: 2",
        )
        with mock.patch.object(Path, "read_text", return_value=source):
            with self.assertRaisesRegex(ValueError, "workers.replicas"):
                load_pd_launch_config(EXAMPLE_CONFIG)

    def test_cuda_list_is_visibility_not_replica_count(self):
        """多 GPU 字符串原样传递给当前 PD worker。"""
        self.assertEqual(validate_cuda_visible_devices("4,5"), "4,5")
        for value in (None, "", "-1", "4,,5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_cuda_visible_devices(value)

    def test_module_dry_run_requires_role_and_prints_one_command(self):
        """python -m 入口显式选择角色，dry-run 只输出一条命令。"""
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "4",
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
                "infinilm.cluster.disaggregated.launch",
                "--config",
                str(EXAMPLE_CONFIG),
                "--role",
                "prefill",
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
                "infinilm.cluster.disaggregated.launch",
                "--config",
                str(EXAMPLE_CONFIG),
                "--dry-run",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(missing_role.returncode, 2)

    def test_launch_package_artifacts_are_installed(self):
        """editable 安装后必须提供 PD YAML 入口和示例配置。"""
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"infinilm.pd-cluster-launch"', pyproject)
        self.assertIn(
            '"infinilm.cluster.disaggregated.launch" = ["configs/*.yaml", "README.md"]',
            pyproject,
        )
        self.assertTrue(EXAMPLE_CONFIG.is_file())


if __name__ == "__main__":
    unittest.main()
