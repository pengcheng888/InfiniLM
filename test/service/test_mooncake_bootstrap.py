import asyncio
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
