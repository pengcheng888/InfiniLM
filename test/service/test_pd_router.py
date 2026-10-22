import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from infinilm.cluster.disaggregated.router import DisaggregatedRouter
from infinilm.cluster.worker_registry import (
    FileWorkerRegistry,
    WorkerType,
)


class DisaggregatedRouterTest(unittest.TestCase):
    def test_request_lifecycle_updates_inflight_and_role_totals(self):
        """请求生命周期维护 inflight，角色总量只增不减。"""
        with TemporaryDirectory() as temporary:
            registry = FileWorkerRegistry(Path(temporary))
            prefill = registry.new_record(
                namespace="pd-test",
                component="prefill",
                endpoint="generate",
                worker_type=WorkerType.PREFILL,
                model="/models/qwen",
                served_model_name="qwen",
                host="127.0.0.1",
                port=18521,
            )
            decode = registry.new_record(
                namespace="pd-test",
                component="backend",
                endpoint="generate",
                worker_type=WorkerType.DECODE,
                model="/models/qwen",
                served_model_name="qwen",
                host="127.0.0.1",
                port=18522,
            )
            router = DisaggregatedRouter("round-robin")

            router.begin_request(prefill)
            router.begin_request(decode)
            self.assertEqual(router.worker_inflight.count(prefill.instance_id), 1)
            self.assertEqual(router.worker_inflight.count(decode.instance_id), 1)
            self.assertEqual(
                router.role_request_totals,
                {"prefill": 1, "decode": 1},
            )

            router.finish_request(prefill.instance_id)
            router.finish_request(decode.instance_id)
            self.assertEqual(router.worker_inflight.count(prefill.instance_id), 0)
            self.assertEqual(router.worker_inflight.count(decode.instance_id), 0)
            self.assertEqual(
                router.role_request_totals,
                {"prefill": 1, "decode": 1},
            )


if __name__ == "__main__":
    unittest.main()
