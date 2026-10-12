import asyncio
import json
import os
import time
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from infinilm.cluster.config import FrontendOptions
from infinilm.cluster.frontend import ClusterFrontend
from infinilm.cluster.http_worker_client_factory import HttpWorkerClientFactory
from infinilm.cluster.registry import FileServiceRegistry, FileServiceWatcher
from infinilm.cluster.router import RoundRobinRouter
from infinilm.cluster.worker_discovery import WorkerDiscoveryService


class FakeEngine:
    def is_healthy(self) -> bool:
        return True


class FileRegistryTest(unittest.TestCase):
    def test_register_snapshot_and_lease_expiry(self):
        with TemporaryDirectory() as temporary:
            # 打印本次测试使用的 discovery 根目录；with 块结束后会自动删除。
            print(f"temporary directory: {temporary}")
            registry = FileServiceRegistry(Path(temporary), lease_seconds=10)
            record = registry.new_record(
                namespace="test",
                component="infinilm",
                endpoint="generate",
                worker_type="aggregated",
                model="/models/qwen",
                served_model_name="qwen",
                host="127.0.0.1",
                port=18081,
                transport="http",
            )

            # registry.register(record) 只创建 coroutine 对象；asyncio.run()
            # 对当前测试线程同步等待注册完成：
            #
            # 测试线程
            #   -> registry.register(record)
            #      创建 coroutine 对象，但不执行函数体
            #   -> asyncio.run(...)
            #      创建临时 event loop
            #      执行 register()，直到原子写入完成
            #      关闭临时 event loop
            #   -> 测试函数继续执行
            asyncio.run(registry.register(record))
            record_files = list(Path(temporary).glob("*.json"))
            print(f"registered record files: {[str(path) for path in record_files]}")
            self.assertEqual(len(asyncio.run(registry.snapshot("test"))), 1)

            expired = replace(record, lease_expires_at=time.time() - 1)
            asyncio.run(registry.register(expired))

            self.assertEqual(asyncio.run(registry.snapshot("test")), [])
        
            self.assertEqual(list(Path(temporary).glob("*.json")), [])

            # 需要人工检查临时文件时设置 INFINILM_TEST_PAUSE_SECONDS。
            pause_seconds = float(os.getenv("INFINILM_TEST_PAUSE_SECONDS", "0"))
            if pause_seconds > 0:
                print(f"pausing for {pause_seconds} seconds")
                time.sleep(pause_seconds)


class RoundRobinRouterTest(unittest.TestCase):
    def test_three_workers_receive_two_requests_each(self):
        with TemporaryDirectory() as temporary:
            registry = FileServiceRegistry(Path(temporary))
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
                    transport="http",
                )
                for index in range(3)
            ]
            watcher = FileServiceWatcher(registry)
            watcher._workers = records
            router = RoundRobinRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
            )

            selected = [router.select(records) for _ in range(6)]
            counts = {
                record.instance_id: selected.count(record) for record in records
            }
            self.assertEqual(list(counts.values()), [2, 2, 2])


class WorkerDiscoveryServiceTest(unittest.TestCase):
    def test_register_heartbeat_and_unregister(self):
        async def run():
            with TemporaryDirectory() as temporary:
                registry = FileServiceRegistry(Path(temporary), lease_seconds=0.1)
                record = registry.new_record(
                    namespace="test",
                    component="infinilm",
                    endpoint="generate",
                    worker_type="aggregated",
                    model="/models/qwen",
                    served_model_name="qwen",
                    host="127.0.0.1",
                    port=18081,
                    transport="http",
                )
                service = WorkerDiscoveryService(
                    engine=FakeEngine(),
                    registry=registry,
                    record=record,
                    heartbeat_interval=0.01,
                )
                await service.start()

                registered = await registry.snapshot("test")
                self.assertEqual(len(registered), 1)
                self.assertEqual(registered[0].transport, "http")

                self.assertGreaterEqual(
                    registered[0].heartbeat_at,
                    record.registered_at,
                )

                await service.stop()
                self.assertEqual(await registry.snapshot("test"), [])

        asyncio.run(run())


class HttpRequestPlaneTest(unittest.TestCase):
    def test_frontend_non_stream_uses_http_worker(self):
        def worker_transport(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "qwen")
            return httpx.Response(
                200,
                json={
                    "id": "worker-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "qwen",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Ready."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )

        async def run():
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                registry = FileServiceRegistry(root)
                record = registry.new_record(
                    namespace="test",
                    component="infinilm",
                    endpoint="generate",
                    worker_type="aggregated",
                    model="/models/qwen",
                    served_model_name="qwen",
                    host="127.0.0.1",
                    port=18081,
                    transport="http",
                )
                await registry.register(record)

                frontend = ClusterFrontend(
                    FrontendOptions(
                        namespace="test",
                        file_kv=root,
                        router_mode="round-robin",
                        router_min_initial_workers=1,
                        http_host="127.0.0.1",
                        http_port=18000,
                        connect_timeout=1.0,
                        request_timeout=2.0,
                        health_interval=0.1,
                        registry_ttl=10.0,
                        heartbeat_interval=2.0,
                    )
                )
                await frontend.router.aclose()
                
                worker_http = httpx.AsyncClient(
                    transport=httpx.MockTransport(worker_transport)
                )
                frontend.router.worker_client_factory = HttpWorkerClientFactory(
                    connect_timeout=1.0,
                    request_timeout=2.0,
                    http_client=worker_http,
                )
                await frontend.watcher.refresh("test")
                app = frontend.create_app()

                try:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://frontend",
                    ) as client:
                        ready = await client.get("/v1/models/qwen/ready")
                        self.assertEqual(ready.status_code, 200)
                        self.assertTrue(ready.json()["ready"])

                        response = await client.post(
                            "/v1/chat/completions",
                            json={
                                "model": "qwen",
                                "messages": [{"role": "user", "content": "hi"}],
                            },
                        )
                        self.assertEqual(response.status_code, 200)
                        body = response.json()
                        self.assertEqual(
                            body["choices"][0]["message"]["content"],
                            "Ready.",
                        )
                        self.assertEqual(body["usage"]["total_tokens"], 5)
                finally:
                    await worker_http.aclose()

        asyncio.run(run())

    def test_http_stream_yields_delta_and_finish(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            stream_text = (
                'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
                'data: {"choices":[{"index":0,"delta":{"content":"Ready"}}]}\n\n'
                'data: {"choices":[{"index":0,"delta":{"content":"."}}]}\n\n'
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                text=stream_text,
                headers={"Content-Type": "text/event-stream"},
            )

        async def run():
            with TemporaryDirectory() as temporary:
                registry = FileServiceRegistry(Path(temporary))
                record = registry.new_record(
                    namespace="test",
                    component="infinilm",
                    endpoint="generate",
                    worker_type="aggregated",
                    model="/models/qwen",
                    served_model_name="qwen",
                    host="127.0.0.1",
                    port=18081,
                    transport="http",
                )
                watcher = FileServiceWatcher(registry)
                watcher._workers = [record]
                router = RoundRobinRouter(
                    watcher,
                    connect_timeout=1.0,
                    request_timeout=2.0,
                )
                await router.aclose()
                transport_http = httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                )
                router.worker_client_factory = HttpWorkerClientFactory(
                    connect_timeout=1.0,
                    request_timeout=2.0,
                    http_client=transport_http,
                )

                try:
                    frames = [
                        frame
                        async for frame in router.stream_chat(
                            {
                                "model": "qwen",
                                "messages": [{"role": "user", "content": "hi"}],
                                "stream": True,
                            }
                        )
                    ]
                    self.assertEqual(
                        [frame["type"] for frame in frames],
                        ["delta", "delta", "finish"],
                    )
                    self.assertEqual(
                        "".join(frame["content"] for frame in frames[:2]),
                        "Ready.",
                    )
                    self.assertEqual(frames[-1]["finish_reason"], "stop")
                finally:
                    await transport_http.aclose()

        asyncio.run(run())

    def test_http_worker_5xx_retries_next_worker(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.port == 18091:
                return httpx.Response(
                    500,
                    json={"error": {"message": "worker failed"}},
                )
            return httpx.Response(
                200,
                json={
                    "id": "worker-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "qwen",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Ready."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )

        async def run():
            with TemporaryDirectory() as temporary:
                registry = FileServiceRegistry(Path(temporary))
                records = [
                    registry.new_record(
                        namespace="test",
                        component="infinilm",
                        endpoint="generate",
                        worker_type="aggregated",
                        model="/models/qwen",
                        served_model_name="qwen",
                        host="127.0.0.1",
                        port=port,
                        transport="http",
                    )
                    for port in (18091, 18092)
                ]
                bad, good = records
                watcher = FileServiceWatcher(registry)
                watcher._workers = records
                router = RoundRobinRouter(
                    watcher,
                    connect_timeout=1.0,
                    request_timeout=2.0,
                )
                await router.aclose()
                transport_http = httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                )
                router.worker_client_factory = HttpWorkerClientFactory(
                    connect_timeout=1.0,
                    request_timeout=2.0,
                    http_client=transport_http,
                )

                try:
                    frames = [
                        frame
                        async for frame in router.stream_chat(
                            {
                                "model": "qwen",
                                "messages": [{"role": "user", "content": "hi"}],
                            }
                        )
                    ]
                    self.assertEqual(frames[-1]["type"], "finish")
                    self.assertEqual(frames[-1]["content"], "Ready.")
                    self.assertEqual(router.metrics.errors_total, 1)
                    self.assertEqual(
                        router.metrics.worker_errors[bad.instance_id],
                        1,
                    )
                finally:
                    await transport_http.aclose()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
