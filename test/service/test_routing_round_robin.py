"""Tests for the round-robin routing policy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from infinilm.cluster.aggregated.router import AggregatedRouter
from infinilm.cluster.config import parse_frontend_args
from infinilm.cluster.worker_registry import FileWorkerRegistry, WorkerRegistryWatcher


class RoundRobinRoutingTest(unittest.TestCase):
    def test_round_robin_is_default_and_explicitly_accepted(self):
        options = parse_frontend_args([])
        self.assertEqual(options.router_mode, "round-robin")

        options = parse_frontend_args(["--router-mode", "round-robin"])
        self.assertEqual(options.router_mode, "round-robin")

    def test_round_robin_selects_each_worker_evenly(self):
        with TemporaryDirectory() as temporary:
            registry = FileWorkerRegistry(Path(temporary))
            records = [
                registry.new_record(
                    namespace="test",
                    component="infinilm",
                    endpoint="generate",
                    worker_type="aggregated",
                    model="/models/qwen",
                    served_model_name="qwen",
                    host="127.0.0.1",
                    port=18081 + index,
                )
                for index in range(3)
            ]
            watcher = WorkerRegistryWatcher(registry)
            watcher._worker_records = records
            router = AggregatedRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="round-robin",
            )

            selected = [router.select(records) for _ in range(6)]
            counts = {record.instance_id: selected.count(record) for record in records}
            self.assertEqual(list(counts.values()), [2, 2, 2])
