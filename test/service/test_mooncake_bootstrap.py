import asyncio
import threading
import unittest

import httpx
from infinilm.kv_connector.mooncake.mooncake_connector_worker import (
    MooncakeConnectorWorker,
    MooncakeParallelConfig,
)
from infinilm.kv_connector.mooncake.mooncake_utils import MooncakeBootstrapServer


class MooncakeBootstrapServerTest(unittest.TestCase):
    def test_bootstrap_port_zero_uses_listener_port(self):
        """bootstrap_port=0 时必须暴露 OS 分配出的真实端口。"""

        async def run():
            server = MooncakeBootstrapServer("127.0.0.1", 0)
            server.start()
            try:
                self.assertEqual(server.requested_port, 0)
                self.assertGreater(server.port, 0)

                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://127.0.0.1:{server.port}/query"
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {})
            finally:
                server.shutdown()

        asyncio.run(run())

    def test_transfer_stats_are_local_and_per_request(self):
        """Mooncake timing 使用本节点 monotonic duration，并按请求导出。"""

        class BareConnectorWorker(MooncakeConnectorWorker):
            def __del__(self):
                return None

        worker = BareConnectorWorker.__new__(BareConnectorWorker)
        worker._kv_transfer_stats = {}
        worker._kv_transfer_stats_lock = threading.Lock()

        worker._record_transfer_stats(
            ["pd-request"],
            event="kv_send",
            role="prefill",
            started_at_ns=100,
            completed_at_ns=180,
            block_count=2,
            bytes_count=4096,
            outcome="success",
        )
        stats = worker.get_kv_connector_stats()

        self.assertEqual(list(stats), ["pd-request"])
        item = stats["pd-request"]
        self.assertEqual(item["event"], "kv_send")
        self.assertEqual(item["duration_ns"], 80)
        self.assertEqual(item["clock_domain"], "local-monotonic")
        self.assertFalse(item["cross_node_comparable"])
        self.assertEqual(worker.get_kv_connector_stats(), {})

    def test_transfer_stats_include_explicit_batch_semantics(self):
        """kv_send 的 bytes/block_count 是 batch 总量，并带 request 粒度。"""

        class BareConnectorWorker(MooncakeConnectorWorker):
            def __del__(self):
                return None

        worker = BareConnectorWorker.__new__(BareConnectorWorker)
        worker._kv_transfer_stats = {}
        worker._kv_transfer_stats_lock = threading.Lock()
        worker._record_transfer_stats(
            ["request-a", "request-b"],
            event="kv_send",
            role="prefill",
            started_at_ns=100,
            completed_at_ns=200,
            block_count=2,
            bytes_count=2048,
            outcome="success",
            batch_size=2,
            batch_id="kv-send-batch-1",
            request_block_counts={"request-a": 1, "request-b": 1},
        )
        stats = worker.get_kv_connector_stats()

        self.assertEqual(stats["request-a"]["batch_id"], "kv-send-batch-1")
        self.assertEqual(stats["request-a"]["batch_request_count"], 2)
        self.assertEqual(stats["request-a"]["batch_block_count"], 2)
        self.assertEqual(stats["request-a"]["batch_bytes"], 2048)
        self.assertEqual(stats["request-a"]["request_block_count"], 1)
        self.assertEqual(stats["request-b"]["request_block_count"], 1)

    def test_producer_registers_to_actual_bootstrap_port(self):
        """producer 自注册不能使用配置值 0，必须读 bootstrap server 实际端口。"""

        async def run():
            bootstrap = MooncakeBootstrapServer("127.0.0.1", 0)
            bootstrap.start()
            class BareConnectorWorker(MooncakeConnectorWorker):
                def __del__(self):
                    return None

            worker = BareConnectorWorker.__new__(BareConnectorWorker)
            worker.bootstrap_server = bootstrap
            worker.parallel_config = MooncakeParallelConfig()
            worker.engine_id = "bootstrap-auto-test"
            worker.hostname = "127.0.0.1"
            worker.side_channel_port = 34567
            worker.tp_rank = 0
            try:
                await worker.register_worker_with_bootstrap()

                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://127.0.0.1:{bootstrap.port}/query"
                    )
                self.assertEqual(
                    response.json(),
                    {
                        "0": {
                            "engine_id": "bootstrap-auto-test",
                            "worker_addr": {"0": {"0": "tcp://127.0.0.1:34567"}},
                        }
                    },
                )
            finally:
                bootstrap.shutdown()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
