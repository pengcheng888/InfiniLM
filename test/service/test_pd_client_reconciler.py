import asyncio
import time
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from infinilm.cluster.disaggregated.client_reconciler import PdClientReconciler
from infinilm.cluster.disaggregated.frontend import (
    DisaggregatedFrontend,
    DisaggregatedFrontendOptions,
)
from infinilm.cluster.disaggregated.worker_client_pool import WorkerClientPoolManager
from infinilm.cluster.routing import WorkerInflightTracker
from infinilm.cluster.worker_registry import WorkerRecord, WorkerType


def make_record(worker_type: str, instance_id: str) -> WorkerRecord:
    return WorkerRecord(
        namespace="pd-test",
        component="infinilm",
        endpoint="generate",
        instance_id=instance_id,
        worker_type=worker_type,
        model="/models/qwen",
        served_model_name="qwen",
        host="127.0.0.1",
        port=18521,
        registered_at=1.0,
        heartbeat_at=1.0,
        lease_expires_at=100.0,
        state="ready",
        engine_healthy=True,
        metadata={},
    )


class StaticWatcher:
    """为真实 HTTP 集成测试提供最小 watcher 接口。"""

    def __init__(self, records):
        self.worker_records = records

    async def refresh(self, namespace: str):
        return self.worker_records

    def start(self, namespace: str, interval: float):
        return None

    async def stop(self):
        return None


async def start_server(app: FastAPI):
    """在随机空闲端口启动 Uvicorn，并等待 socket 可用。"""
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            lifespan="on",
        )
    )
    task = asyncio.create_task(server.serve())
    deadline = time.monotonic() + 5.0
    while not server.started:
        if task.done():
            task.result()
        if time.monotonic() > deadline:
            server.should_exit = True
            await task
            raise TimeoutError("test HTTP server did not start")
        await asyncio.sleep(0.01)
    socket = server.servers[0].sockets[0]
    return server, task, socket.getsockname()[1]


async def stop_server(server: uvicorn.Server, task: asyncio.Task):
    server.should_exit = True
    await task


class PdClientReconcilerTest(unittest.TestCase):
    def test_reconciler_initializes_worker_in_background(self):
        """后台循环会把 watcher snapshot 交给 manager 并完成 client 初始化。"""

        async def run():
            manager = WorkerClientPoolManager(
                connect_timeout=0.2,
                request_timeout=1.0,
                inflight=WorkerInflightTracker(),
            )
            record = make_record("decode", "decode-1")
            reconciler = PdClientReconciler(
                watcher_records=lambda: [record],
                manager=manager,
                interval=0.01,
            )
            reconciler.start()
            try:
                for _ in range(100):
                    if manager.routable_records("qwen", WorkerType.DECODE):
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(
                    manager.routable_records("qwen", WorkerType.DECODE),
                    [record],
                )
            finally:
                await reconciler.stop()
                await manager.aclose()

        asyncio.run(run())

    def test_slow_bootstrap_does_not_block_next_discovery_snapshot(self):
        """慢 bootstrap 不能延迟下一轮 reconcile 调度新 worker。"""

        async def run():
            manager = WorkerClientPoolManager(
                connect_timeout=0.2,
                request_timeout=1.0,
                inflight=WorkerInflightTracker(),
            )
            records = [make_record("decode", "decode-slow")]
            fast_record = make_record("decode", "decode-fast")
            fast_model = "fast"
            fast_record = replace(fast_record, served_model_name=fast_model)
            slow_started = asyncio.Event()

            async def create_client_info(record):
                if record.instance_id == "decode-slow":
                    slow_started.set()
                    await asyncio.sleep(10)
                return {
                    "http_client": httpx.AsyncClient(),
                    "record": record,
                    "instance_id": record.instance_id,
                    "routable": True,
                }

            reconciler = PdClientReconciler(
                watcher_records=lambda: list(records),
                manager=manager,
                interval=0.01,
            )
            with patch.object(
                manager,
                "_create_client_info",
                side_effect=create_client_info,
            ):
                reconciler.start()
                try:
                    await asyncio.wait_for(slow_started.wait(), timeout=1)
                    records.append(fast_record)
                    for _ in range(100):
                        if manager.routable_records(
                            fast_model,
                            WorkerType.DECODE,
                        ):
                            break
                        await asyncio.sleep(0.01)

                    self.assertEqual(
                        manager.routable_records(fast_model, WorkerType.DECODE),
                        [fast_record],
                    )
                    self.assertEqual(
                        manager.initializing_ids("qwen", WorkerType.DECODE),
                        {"decode-slow"},
                    )
                finally:
                    await reconciler.stop()
                    await manager.aclose()

        asyncio.run(run())

    def test_frontend_lifespan_starts_and_stops_reconciler(self):
        """FastAPI lifespan 必须启动/停止后台 client reconcile。"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            frontend = DisaggregatedFrontend(
                DisaggregatedFrontendOptions(
                    namespace="pd-test",
                    discovery_root=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                )
            )
            started = []
            stopped = []

            def start():
                started.append(True)

            async def stop():
                stopped.append(True)

            with (
                patch.object(frontend.client_reconciler, "start", side_effect=start),
                patch.object(frontend.client_reconciler, "stop", side_effect=stop),
            ):
                with TestClient(frontend.create_app()):
                    self.assertEqual(started, [True])
                self.assertEqual(stopped, [True])

    def test_frontend_api_paths_do_not_call_reconcile(self):
        """/health、models 和 chat 只读状态，不触发后台 bootstrap。"""
        import tempfile
        from pathlib import Path

        async def run():
            with tempfile.TemporaryDirectory() as temporary:
                frontend = DisaggregatedFrontend(
                    DisaggregatedFrontendOptions(
                        namespace="pd-test",
                        discovery_root=Path(temporary),
                        router_mode="round-robin",
                        http_host="127.0.0.1",
                        http_port=18520,
                    )
                )
                with (
                    patch.object(frontend.client_reconciler, "start"),
                    patch.object(frontend.client_reconciler, "stop", new=AsyncMock()),
                    patch.object(
                        frontend.worker_client_pool_manager,
                        "reconcile_records",
                        new=AsyncMock(),
                    ) as reconcile_records,
                    patch.object(
                        frontend.worker_client_pool_manager,
                        "reconcile_model",
                        new=AsyncMock(),
                    ) as reconcile_model,
                ):
                    with TestClient(frontend.create_app()) as client:
                        self.assertEqual(client.get("/health").status_code, 200)
                        self.assertEqual(client.get("/v1/models").status_code, 200)
                        self.assertEqual(
                            client.post(
                                "/v1/chat/completions",
                                json={"messages": []},
                            ).status_code,
                            503,
                        )

                reconcile_records.assert_not_called()
                reconcile_model.assert_not_called()
                await frontend.registry.aclose()

        asyncio.run(run())

    def test_real_http_pd_cluster_reconciles_and_serves_chat(self):
        """真实占用本地随机端口，验证后台 reconcile 和 PD chat 全链路。"""

        async def run():
            bootstrap_app = FastAPI()

            @bootstrap_app.get("/query")
            async def query():
                return {"0": {"engine_id": "integration-engine"}}

            prefill_app = FastAPI()

            @prefill_app.post("/v1/chat/completions")
            async def prefill(request: Request):
                assert request.headers["x-infinilm-internal"] == "true"
                return Response(content=b'{"prefill":true}')

            decode_app = FastAPI()

            @decode_app.post("/v1/chat/completions")
            async def decode(request: Request):
                assert request.headers["x-infinilm-internal"] == "true"
                payload = await request.json()
                transfer = payload["kv_transfer_params"]
                assert transfer["remote_bootstrap_addr"].endswith("/query") is False
                assert transfer["remote_engine_id"] == "integration-engine"
                return Response(content=b'{"ok":true}')

            bootstrap_server, bootstrap_task, bootstrap_port = await start_server(
                bootstrap_app
            )
            prefill_server, prefill_task, prefill_port = await start_server(prefill_app)
            decode_server, decode_task, decode_port = await start_server(decode_app)
            frontend_server = None
            frontend_task = None

            try:
                base_prefill = make_record("prefill", "integration-prefill")
                prefill_record = replace(
                    base_prefill,
                    port=prefill_port,
                    metadata={
                        "supported_api_routes": ["/v1/chat/completions"],
                        "pd": {
                            "bootstrap_host": "127.0.0.1",
                            "bootstrap_port": bootstrap_port,
                        },
                    },
                )
                base_decode = make_record("decode", "integration-decode")
                decode_record = replace(
                    base_decode,
                    port=decode_port,
                    metadata={
                        "supported_api_routes": ["/v1/chat/completions"],
                    },
                )

                with TemporaryDirectory() as temporary:
                    with patch.dict(
                        "os.environ",
                        {
                            "INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS": "0.02",
                            "INFINILM_CLUSTER_BOOTSTRAP_TIMEOUT_SECONDS": "1",
                        },
                    ):
                        frontend = DisaggregatedFrontend(
                            DisaggregatedFrontendOptions(
                                namespace="pd-test",
                                discovery_root=Path(temporary),
                                router_mode="round-robin",
                                http_host="127.0.0.1",
                                http_port=0,
                            )
                        )
                    frontend.watcher = StaticWatcher([prefill_record, decode_record])
                    frontend_server, frontend_task, frontend_port = await start_server(
                        frontend.create_app()
                    )

                    async with httpx.AsyncClient() as client:
                        deadline = time.monotonic() + 5.0
                        health = None
                        while time.monotonic() < deadline:
                            health = await client.get(
                                f"http://127.0.0.1:{frontend_port}/health"
                            )
                            if health.json().get("ready"):
                                break
                            await asyncio.sleep(0.02)
                        assert health is not None and health.json()["ready"]

                        response = await client.post(
                            f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
                            json={"model": "qwen", "messages": []},
                        )
                        assert response.status_code == 200
                        assert response.json() == {"ok": True}

                    await frontend.registry.aclose()
            finally:
                if frontend_server is not None:
                    await stop_server(frontend_server, frontend_task)
                await stop_server(decode_server, decode_task)
                await stop_server(prefill_server, prefill_task)
                await stop_server(bootstrap_server, bootstrap_task)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
