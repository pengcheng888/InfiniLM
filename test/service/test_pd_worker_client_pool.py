import asyncio
import unittest
from dataclasses import replace
from unittest.mock import patch

from infinilm.cluster.disaggregated.worker_client_pool import WorkerClientPoolManager
from infinilm.cluster.routing import WorkerInflightTracker
from infinilm.cluster.worker_registry import WorkerRecord, WorkerType


def make_record(
    worker_type: str,
    instance_id: str,
    model_name: str = "qwen",
) -> WorkerRecord:
    """构造只包含 client 池测试所需字段的 worker 注册记录。"""
    return WorkerRecord(
        namespace="pd-test",
        component="infinilm",
        endpoint="generate",
        instance_id=instance_id,
        worker_type=worker_type,
        model=f"/models/{model_name}",
        served_model_name=model_name,
        host="127.0.0.1",
        port=18521,
        registered_at=1.0,
        heartbeat_at=1.0,
        lease_expires_at=100.0,
        metadata={
            "pd": {
                "bootstrap_host": "127.0.0.1",
                "bootstrap_port": 18528,
            }
        },
    )


class FakeHttpClient:
    """替代 httpx.AsyncClient，记录资源是否被释放。"""

    def __init__(self, response: object | None = None) -> None:
        self.response = response
        self.closed = False

    async def get(self, url: str):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def aclose(self) -> None:
        self.closed = True


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"0": {"engine_id": "engine-0"}}


class WorkerClientPoolManagerTest(unittest.TestCase):
    def test_reconcile_model_keeps_clients_isolated_by_model(self):
        """同步一个模型时不能清理另一个模型已初始化的 client。"""

        async def run():
            manager = WorkerClientPoolManager(
                connect_timeout=1.0,
                request_timeout=2.0,
                inflight=WorkerInflightTracker(),
            )
            qwen_record = make_record("decode", "decode-qwen", "qwen")
            other_record = make_record("decode", "decode-other", "other")
            clients = [FakeHttpClient(), FakeHttpClient()]

            async def create_client_info(worker):
                index = 0 if worker is qwen_record else 1
                return {
                    "http_client": clients[index],
                    "record": worker,
                    "instance_id": worker.instance_id,
                    "routable": True,
                }

            with patch.object(
                manager,
                "_create_client_info",
                side_effect=create_client_info,
            ):
                await manager.reconcile_model(
                    role=WorkerType.DECODE,
                    model_name="qwen",
                    records=[qwen_record],
                )
                await manager.reconcile_model(
                    role=WorkerType.DECODE,
                    model_name="other",
                    records=[other_record],
                )

            self.assertEqual(
                manager.routable_records("qwen", WorkerType.DECODE),
                [qwen_record],
            )
            self.assertEqual(
                manager.routable_records("other", WorkerType.DECODE),
                [other_record],
            )

            # discovery 中消失的模型由 retain_models 统一清理。
            await manager.retain_models(["qwen"])
            self.assertEqual(
                manager.routable_records("qwen", WorkerType.DECODE),
                [qwen_record],
            )
            self.assertEqual(
                manager.routable_records("other", WorkerType.DECODE),
                [],
            )
            self.assertTrue(clients[1].closed)

        asyncio.run(run())

    def test_concurrent_reconcile_creates_client_once(self):
        """并发同步同一个 worker 时，manager 锁应避免重复初始化。"""

        async def run():
            manager = WorkerClientPoolManager(
                connect_timeout=1.0,
                request_timeout=2.0,
                inflight=WorkerInflightTracker(),
            )
            record = make_record("decode", "decode-1")
            created = 0

            async def create_client_info(worker):
                nonlocal created
                created += 1
                await asyncio.sleep(0)
                return {
                    "http_client": FakeHttpClient(),
                    "record": worker,
                    "instance_id": worker.instance_id,
                    "routable": True,
                }

            with patch.object(
                manager,
                "_create_client_info",
                side_effect=create_client_info,
            ):
                await asyncio.gather(
                    manager.reconcile_model(
                        role=WorkerType.DECODE,
                        model_name="qwen",
                        records=[record],
                    ),
                    manager.reconcile_model(
                        role=WorkerType.DECODE,
                        model_name="qwen",
                        records=[record],
                    ),
                )

            self.assertEqual(created, 1)
            self.assertEqual(
                manager.client_ids("qwen", WorkerType.DECODE),
                {record.instance_id},
            )
            self.assertEqual(
                manager.routable_records("qwen", WorkerType.DECODE),
                [record],
            )

        asyncio.run(run())

    def test_reconcile_updates_existing_record_without_rebuilding_client(self):
        """registry metadata 更新时只刷新 record，不重建 HTTP client。"""

        async def run():
            manager = WorkerClientPoolManager(
                connect_timeout=1.0,
                request_timeout=2.0,
                inflight=WorkerInflightTracker(),
            )
            record = make_record("decode", "decode-1")
            client = FakeHttpClient()
            created = 0

            async def create_client_info(worker):
                nonlocal created
                created += 1
                return {
                    "http_client": client,
                    "record": worker,
                    "instance_id": worker.instance_id,
                    "routable": True,
                }

            with patch.object(
                manager,
                "_create_client_info",
                side_effect=create_client_info,
            ):
                await manager.reconcile_model(
                    role=WorkerType.DECODE,
                    model_name="qwen",
                    records=[record],
                )
                updated = replace(record, heartbeat_at=2.0)
                await manager.reconcile_model(
                    role=WorkerType.DECODE,
                    model_name="qwen",
                    records=[updated],
                )

            self.assertEqual(created, 1)
            self.assertEqual(
                manager.get("qwen", WorkerType.DECODE, record.instance_id)["record"],
                updated,
            )
            self.assertFalse(client.closed)

        asyncio.run(run())

    def test_worker_with_inflight_is_removed_later(self):
        """有存量请求的退出 worker 延迟清理，请求结束后再关闭连接。"""

        async def run():
            inflight = WorkerInflightTracker()
            manager = WorkerClientPoolManager(
                connect_timeout=1.0,
                request_timeout=2.0,
                inflight=inflight,
            )
            record = make_record("decode", "decode-1")
            client = FakeHttpClient()

            async def create_client_info(worker):
                return {
                    "http_client": client,
                    "record": worker,
                    "instance_id": worker.instance_id,
                    "routable": True,
                }

            with patch.object(
                manager,
                "_create_client_info",
                side_effect=create_client_info,
            ):
                await manager.reconcile_model(
                    role=WorkerType.DECODE,
                    model_name="qwen",
                    records=[record],
                )
                self.assertEqual(
                    manager.routable_records("qwen", WorkerType.DECODE), [record]
                )
                inflight.begin_request(record.instance_id)
                await manager.reconcile_model(
                    role=WorkerType.DECODE,
                    model_name="qwen",
                    records=[],
                )
                self.assertEqual(
                    manager.client_ids("qwen", WorkerType.DECODE),
                    {record.instance_id},
                )
                self.assertFalse(client.closed)
                # client 仍在服务存量请求，但不能再进入新请求候选集。
                self.assertEqual(
                    manager.routable_records("qwen", WorkerType.DECODE), []
                )

                inflight.finish_request(record.instance_id)
                await manager.reconcile_model(
                    role=WorkerType.DECODE,
                    model_name="qwen",
                    records=[],
                )

            self.assertEqual(manager.client_ids("qwen", WorkerType.DECODE), set())
            self.assertTrue(client.closed)

        asyncio.run(run())

    def test_prefill_bootstrap_metadata_is_loaded(self):
        """prefill 初始化会查询 bootstrap，并保存 engine ID 映射。"""

        async def run():
            manager = WorkerClientPoolManager(
                connect_timeout=1.0,
                request_timeout=2.0,
                inflight=WorkerInflightTracker(),
            )
            record = make_record("prefill", "prefill-1")
            client = FakeHttpClient(FakeResponse())

            with patch(
                "infinilm.cluster.disaggregated.worker_client_pool.httpx.AsyncClient",
                return_value=client,
            ):
                info = await manager._create_client_info(record)

            self.assertEqual(info["bootstrap_addr"], "http://127.0.0.1:18528")
            self.assertEqual(info["http_base_url"], "http://127.0.0.1:18521")
            self.assertTrue(info["routable"])
            self.assertEqual(info["dp_engine_id"], {0: "engine-0"})
            self.assertEqual(info["dp_size"], 1)

        asyncio.run(run())

    def test_prefill_bootstrap_failure_closes_client(self):
        """bootstrap 失败时关闭新建 client，不让资源泄漏。"""

        async def run():
            manager = WorkerClientPoolManager(
                connect_timeout=1.0,
                request_timeout=2.0,
                inflight=WorkerInflightTracker(),
            )
            record = make_record("prefill", "prefill-1")
            client = FakeHttpClient(RuntimeError("bootstrap unavailable"))

            with patch(
                "infinilm.cluster.disaggregated.worker_client_pool.httpx.AsyncClient",
                return_value=client,
            ):
                with self.assertRaisesRegex(RuntimeError, "bootstrap unavailable"):
                    await manager._create_client_info(record)

            self.assertTrue(client.closed)

        asyncio.run(run())

    def test_aclose_closes_and_clears_pools(self):
        """frontend 退出时 manager 需要释放所有 worker 连接。"""

        async def run():
            manager = WorkerClientPoolManager(
                connect_timeout=1.0,
                request_timeout=2.0,
                inflight=WorkerInflightTracker(),
            )
            clients = [FakeHttpClient(), FakeHttpClient()]
            records = [
                make_record("prefill", "prefill-1"),
                make_record("decode", "decode-1"),
            ]

            async def create_client_info(worker):
                return {
                    "http_client": clients[0]
                    if worker.worker_type is WorkerType.PREFILL
                    else clients[1],
                    "record": worker,
                    "instance_id": worker.instance_id,
                    "routable": True,
                }

            with patch.object(
                manager,
                "_create_client_info",
                side_effect=create_client_info,
            ):
                await manager.reconcile_model(
                    role=WorkerType.PREFILL,
                    model_name="qwen",
                    records=[records[0]],
                )
                await manager.reconcile_model(
                    role=WorkerType.DECODE,
                    model_name="qwen",
                    records=[records[1]],
                )
                await manager.aclose()

            self.assertTrue(all(client.closed for client in clients))
            self.assertEqual(manager.client_ids("qwen", WorkerType.PREFILL), set())
            self.assertEqual(manager.client_ids("qwen", WorkerType.DECODE), set())

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
