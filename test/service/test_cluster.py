import asyncio
import json
import os
import time
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from infinilm.cluster.config import (
    FrontendOptions,
    parse_frontend_args,
    parse_worker_cluster_args,
    remove_cluster_worker_args,
)
from infinilm.cluster.frontend import ClusterFrontend
from infinilm.cluster.frontend import main as frontend_main
from infinilm.cluster.http_worker_client_factory import HttpWorkerClientFactory
from infinilm.cluster.metrics import REQUEST_DURATION_BUCKETS, Histogram
from infinilm.cluster.registry import FileServiceRegistry, FileServiceWatcher
from infinilm.cluster.router import ClusterRouter
from infinilm.cluster.worker import ClusterInferenceServer, DrainingUvicornServer
from infinilm.cluster.worker_discovery import WorkerDiscoveryService
from infinilm.cluster.worker_traffic import WorkerTrafficMetrics
from infinilm.llm import AsyncLLMEngine
from infinilm.server.inference_server import InferenceServer


class FakeEngine:
    def is_healthy(self) -> bool:
        return True


class FakeLoadEngine(FakeEngine):
    def get_load_stats(self) -> dict:
        return {
            "waiting_requests": 2,
            "running_requests": 3,
            "max_batch_size": 16,
            "kv_capacity": 100,
            "kv_used": 40,
            "kv_usage_ratio": 0.4,
        }


class HistogramTest(unittest.TestCase):
    def test_quantile_uses_cumulative_buckets(self):
        histogram = Histogram()
        for value in (0.1, 0.2, 0.3):
            histogram.observe(value, REQUEST_DURATION_BUCKETS)

        self.assertAlmostEqual(
            histogram.quantile(0.50, REQUEST_DURATION_BUCKETS), 0.175
        )
        self.assertAlmostEqual(
            histogram.quantile(0.95, REQUEST_DURATION_BUCKETS), 0.4625
        )
        self.assertEqual(histogram.quantile(0.99, REQUEST_DURATION_BUCKETS) > 0.2, True)


class WorkerTrafficMetricsTest(unittest.TestCase):
    def test_counters_track_active_success_and_failure(self):
        metrics = WorkerTrafficMetrics()
        metrics.begin(streaming=True)
        metrics.begin(streaming=False)
        self.assertEqual(metrics.snapshot()["active_requests"], 2)

        metrics.finish()
        metrics.finish(failed=True)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["received_requests_total"], 2)
        self.assertEqual(snapshot["streaming_requests_total"], 1)
        self.assertEqual(snapshot["non_streaming_requests_total"], 1)
        self.assertEqual(snapshot["successful_requests_total"], 1)
        self.assertEqual(snapshot["failed_requests_total"], 1)
        self.assertEqual(snapshot["active_requests"], 0)

    def test_cluster_inference_server_counts_chat_traffic(self):
        async def fake_chat(self, request_id, data, http_request):
            return SimpleNamespace(status_code=200)

        async def fake_stream_chat(self, request_id, data, http_request):
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"Ready\"}}]}\n\n"
            yield "data: {\"choices\":[{\"finish_reason\": \"stop\"}]}\n\n"

        server = ClusterInferenceServer.__new__(ClusterInferenceServer)
        server.worker_traffic = WorkerTrafficMetrics()

        async def run():
            with (
                patch.object(InferenceServer, "_chat", new=fake_chat),
                patch.object(InferenceServer, "_stream_chat", new=fake_stream_chat),
            ):
                await server._chat("request-1", {}, object())
                chunks = [chunk async for chunk in server._stream_chat("request-2", {}, object())]

            self.assertEqual(len(chunks), 2)
            snapshot = server.worker_traffic.snapshot()
            self.assertEqual(snapshot["received_requests_total"], 2)
            self.assertEqual(snapshot["streaming_requests_total"], 1)
            self.assertEqual(snapshot["non_streaming_requests_total"], 1)
            self.assertEqual(snapshot["successful_requests_total"], 2)
            self.assertEqual(snapshot["active_requests"], 0)

        asyncio.run(run())


class WorkerDrainingTest(unittest.TestCase):
    def test_worker_options_include_drain_grace(self):
        options = parse_worker_cluster_args(["--drain-grace-seconds", "3"])
        self.assertEqual(options.drain_grace_seconds, 3.0)
        self.assertEqual(parse_worker_cluster_args([]).drain_grace_seconds, 1.0)
        remaining = remove_cluster_worker_args(
            ["--model", "m", "--drain-grace-seconds", "3", "--tp", "1"]
        )
        self.assertEqual(remaining, ["--model", "m", "--tp", "1"])

    def test_uvicorn_server_drains_before_exit(self):
        async def run():
            server = DrainingUvicornServer.__new__(DrainingUvicornServer)
            server.drain_grace_seconds = 0.01
            server.should_exit = False
            called = asyncio.Event()

            async def drain():
                called.set()

            server.on_drain_start = drain
            await server._drain_and_exit()

            self.assertTrue(called.is_set())
            self.assertTrue(server.should_exit)

        asyncio.run(run())


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


class RouterModeTest(unittest.TestCase):
    def test_least_loaded_is_default_and_round_robin_is_accepted(self):
        options = parse_frontend_args([])
        self.assertEqual(options.router_mode, "round-robin")

        options = parse_frontend_args(["--router-mode", "round-robin"])
        self.assertEqual(options.router_mode, "round-robin")

        with self.assertRaises(SystemExit):
            parse_frontend_args(["--router-mode", "least-inflight"])

    def test_frontend_startup_logs_monitoring_url(self):
        from unittest.mock import patch

        with (
            patch("infinilm.cluster.frontend.uvicorn.run") as uvicorn_run,
            self.assertLogs("infinilm.cluster.frontend", level="INFO") as logs,
        ):
            frontend_main([])

        uvicorn_run.assert_called_once()
        self.assertIn(
            "router_mode=round-robin",
            logs.output[0],
        )
        self.assertIn(
            "INFO:infinilm.cluster.frontend:Monitoring page: "
            "http://127.0.0.1:18000/monitoring",
            logs.output,
        )

    def test_round_robin_selects_each_worker_evenly(self):
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
            router = ClusterRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="round-robin",
            )

            selected = [router.select(records) for _ in range(6)]
            counts = {
                record.instance_id: selected.count(record) for record in records
            }
            self.assertEqual(list(counts.values()), [2, 2, 2])

    def test_router_excludes_draining_worker(self):
        with TemporaryDirectory() as temporary:
            registry = FileServiceRegistry(Path(temporary))
            ready = registry.new_record(
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
            draining = registry.new_record(
                namespace="test",
                component="infinilm",
                endpoint="generate",
                worker_type="aggregated",
                model="/models/qwen",
                served_model_name="qwen",
                host="127.0.0.1",
                port=18082,
                transport="http",
                state="draining",
            )
            watcher = FileServiceWatcher(registry)
            watcher._workers = [draining, ready]
            router = ClusterRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
            )

            self.assertEqual(router.available_workers(), [ready])
            self.assertIs(router.select([draining, ready]), ready)


class LoadAwareRouterTest(unittest.TestCase):
    def test_least_loaded_inflight_is_dominant(self):
        with TemporaryDirectory() as temporary:
            registry = FileServiceRegistry(Path(temporary))
            busy, medium, idle = [
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
            watcher._workers = [busy, medium, idle]
            router = ClusterRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="least-loaded",
            )
            router.worker_inflight.update(
                {
                    busy.instance_id: 5,
                    medium.instance_id: 2,
                    idle.instance_id: 0,
                }
            )

            self.assertIs(router.select([busy, medium, idle]), idle)
            self.assertAlmostEqual(router.load_score(busy), 4.5)
            self.assertAlmostEqual(router.load_score(medium), 1.8)
            self.assertAlmostEqual(router.load_score(idle), 0.0)
            rendered = router.metrics.render(3, [busy, medium, idle])
            self.assertIn("infinilm_router_mode{mode=\"least-loaded\"} 1", rendered)
            self.assertIn("infinilm_router_worker_inflight", rendered)

    def test_least_loaded_routes_around_active_request(self):
        class FakeClient:
            def __init__(self, release: asyncio.Event) -> None:
                self.release = release

            async def stream_chat(self, payload):
                yield {"type": "delta", "content": "partial"}
                await self.release.wait()
                yield {"type": "finish", "finish_reason": "stop"}

        class FakeFactory:
            def __init__(self) -> None:
                self.release = asyncio.Event()

            def create(self, worker):
                return FakeClient(self.release)

            async def aclose(self) -> None:
                return None

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
                        port=18081 + index,
                        transport="http",
                    )
                    for index in range(2)
                ]
                watcher = FileServiceWatcher(registry)
                watcher._workers = records
                router = ClusterRouter(
                    watcher,
                    connect_timeout=1.0,
                    request_timeout=2.0,
                    router_mode="least-loaded",
                )
                await router.aclose()
                factory = FakeFactory()
                router.worker_client_factory = factory

                first = router.stream_chat(
                    {"model": "qwen", "messages": [{"role": "user", "content": "1"}]}
                )
                first_frame = await first.__anext__()
                first_worker = first_frame["worker_id"]
                self.assertEqual(router.metrics.inflight, 1)

                second = router.stream_chat(
                    {"model": "qwen", "messages": [{"role": "user", "content": "2"}]}
                )
                second_frame = await second.__anext__()
                second_worker = second_frame["worker_id"]
                self.assertNotEqual(first_worker, second_worker)
                self.assertEqual(router.metrics.inflight, 2)
                self.assertEqual(router.metrics.worker_inflight[first_worker], 1)
                self.assertEqual(router.metrics.worker_inflight[second_worker], 1)

                factory.release.set()
                async for _ in first:
                    pass
                async for _ in second:
                    pass

                self.assertEqual(router.metrics.inflight, 0)
                self.assertEqual(router.metrics.successful_requests_total, 2)
                self.assertEqual(router.metrics.worker_inflight[first_worker], 0)
                self.assertEqual(router.metrics.worker_inflight[second_worker], 0)

        asyncio.run(run())

    def test_least_loaded_uses_worker_metadata(self):
        with TemporaryDirectory() as temporary:
            registry = FileServiceRegistry(Path(temporary))
            light, medium, heavy = [
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
            light = replace(
                light,
                metadata={
                    "load": {
                        "waiting_requests": 0,
                        "running_requests": 1,
                        "max_batch_size": 8,
                        "kv_capacity": 100,
                        "kv_used": 10,
                    }
                },
            )
            medium = replace(
                medium,
                metadata={
                    "load": {
                        "waiting_requests": 1,
                        "running_requests": 2,
                        "max_batch_size": 8,
                        "kv_capacity": 100,
                        "kv_used": 40,
                    }
                },
            )
            heavy = replace(
                heavy,
                metadata={
                    "load": {
                        "waiting_requests": 4,
                        "running_requests": 8,
                        "max_batch_size": 8,
                        "kv_capacity": 100,
                        "kv_used": 90,
                    }
                },
            )
            watcher = FileServiceWatcher(registry)
            watcher._workers = [light, medium, heavy]
            router = ClusterRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="least-loaded",
            )

            self.assertIs(router.select([light, medium, heavy]), light)
            self.assertLess(
                router.load_score(light),
                router.load_score(medium),
            )
            self.assertLess(
                router.load_score(medium),
                router.load_score(heavy),
            )
            rendered = router.metrics.render(3, [light, medium, heavy])
            self.assertIn("infinilm_router_worker_load_score", rendered)
            self.assertIn("infinilm_router_worker_waiting_requests", rendered)
            self.assertIn("infinilm_router_worker_running_requests", rendered)
            self.assertIn("infinilm_router_worker_kv_usage_ratio", rendered)


class WorkerDiscoveryServiceTest(unittest.TestCase):
    def test_register_load_metadata_heartbeat_and_unregister(self):
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
                    engine=FakeLoadEngine(),
                    registry=registry,
                    record=record,
                    heartbeat_interval=0.01,
                    traffic_metrics=WorkerTrafficMetrics(),
                )
                await service.start()

                registered = await registry.snapshot("test")
                self.assertEqual(len(registered), 1)
                self.assertEqual(registered[0].transport, "http")
                self.assertEqual(
                    registered[0].metadata["load"]["waiting_requests"],
                    2,
                )
                self.assertEqual(
                    registered[0].metadata["load"]["running_requests"],
                    3,
                )
                self.assertAlmostEqual(
                    registered[0].metadata["load"]["kv_usage_ratio"],
                    0.4,
                )
                self.assertEqual(
                    registered[0].metadata["traffic"]["received_requests_total"],
                    0,
                )
                self.assertEqual(registered[0].state, "ready")

                await service.start_draining()
                draining = await registry.snapshot("test")
                self.assertEqual(draining[0].state, "draining")

                self.assertGreaterEqual(
                    registered[0].heartbeat_at,
                    record.registered_at,
                )

                await service.stop()
                self.assertEqual(await registry.snapshot("test"), [])

        asyncio.run(run())


class AsyncLLMEngineLoadStatsTest(unittest.TestCase):
    def test_get_load_stats_reads_scheduler_state(self):
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        engine.config = SimpleNamespace(max_batch_size=16)
        engine.engine = SimpleNamespace(
            scheduler=SimpleNamespace(
                waiting_queue=SimpleNamespace(sync_q=SimpleNamespace(qsize=lambda: 2)),
                running_queue=SimpleNamespace(sync_q=SimpleNamespace(qsize=lambda: 3)),
                get_cache_stats=lambda: {
                    "usable_blocks": 100,
                    "num_used_blocks": 40,
                    "num_free_blocks": 60,
                },
            )
        )

        stats = engine.get_load_stats()
        self.assertEqual(stats["waiting_requests"], 2)
        self.assertEqual(stats["running_requests"], 3)
        self.assertEqual(stats["max_batch_size"], 16)
        self.assertEqual(stats["kv_capacity"], 100)
        self.assertEqual(stats["kv_used"], 40)
        self.assertAlmostEqual(stats["kv_usage_ratio"], 0.4)


class MonitoringDashboardTest(unittest.TestCase):
    def test_monitoring_page_and_api_are_served_by_frontend(self):
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
                    metadata={
                        "cuda_visible_devices": "2",
                        "tp": 1,
                        "max_batch_size": 8,
                        "load": {
                            "waiting_requests": 1,
                            "running_requests": 2,
                            "max_batch_size": 8,
                            "kv_capacity": 100,
                            "kv_used": 40,
                            "kv_usage_ratio": 0.4,
                        },
                        "traffic": {
                            "received_requests_total": 9,
                            "streaming_requests_total": 4,
                            "non_streaming_requests_total": 5,
                            "successful_requests_total": 8,
                            "failed_requests_total": 1,
                            "canceled_requests_total": 0,
                            "active_requests": 2,
                        },
                    },
                )
                await registry.register(record)

                frontend = ClusterFrontend(
                    FrontendOptions(
                        namespace="test",
                        file_kv=root,
                        router_mode="least-loaded",
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
                await frontend.watcher.refresh("test")
                frontend.router.metrics.requests_total = 7
                frontend.router.metrics.mark_request_mode(True)
                frontend.router.metrics.mark_request_mode(False)
                frontend.router.metrics.mark_request_mode(False)
                frontend.router.metrics.worker_requests[record.instance_id] = 7
                frontend.router.metrics.record_usage(
                    record.instance_id,
                    {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                )
                frontend.router.metrics.mark_worker_success(record.instance_id)
                
                app = frontend.create_app()

                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://frontend",
                ) as client:
                    page = await client.get("/monitoring")
                    self.assertEqual(page.status_code, 200)
                    self.assertIn("text/html", page.headers["content-type"])
                    self.assertIn("InfiniLM Cluster Monitor", page.text)
                    self.assertIn("/api/monitoring", page.text)
                    self.assertIn('id="auto"', page.text)
                    self.assertIn('id="refresh"', page.text)
                    self.assertIn("duration-p95", page.text)
                    self.assertIn("duration-p99", page.text)
                    self.assertNotIn("https://cdn", page.text)

                    response = await client.get("/api/monitoring")
                    self.assertEqual(response.status_code, 200)
                    snapshot = response.json()
                    self.assertTrue(snapshot["ready"])
                    self.assertEqual(snapshot["router_mode"], "least-loaded")
                    self.assertEqual(snapshot["metrics_source"], "frontend-router")
                    self.assertEqual(snapshot["worker_count"], 1)
                    self.assertEqual(snapshot["worker_states"]["ready"], 1)
                    self.assertEqual(snapshot["worker_states"]["draining"], 0)
                    self.assertEqual(snapshot["global"]["requests_total"], 7)
                    self.assertEqual(
                        snapshot["global"]["worker_received_requests_total"], 9
                    )
                    self.assertEqual(
                        snapshot["global"]["worker_active_requests"], 2
                    )
                    self.assertEqual(
                        snapshot["global"]["untracked_requests_total"], 2
                    )
                    self.assertEqual(snapshot["global"]["streaming_requests_total"], 1)
                    self.assertEqual(
                        snapshot["global"]["non_streaming_requests_total"], 2
                    )
                    self.assertEqual(
                        snapshot["global"]["usage_prompt_tokens_total"], 11
                    )
                    self.assertEqual(
                        snapshot["global"]["usage_completion_tokens_total"], 7
                    )
                    self.assertEqual(snapshot["global"]["usage_total_tokens_total"], 18)

                    worker = snapshot["workers"][0]
                    self.assertEqual(worker["instance_id"], record.instance_id)
                    self.assertEqual(worker["served_model_name"], "qwen")
                    self.assertEqual(worker["endpoint"], "127.0.0.1:18081")
                    self.assertEqual(worker["gpu"], "2")
                    self.assertEqual(worker["waiting_requests"], 1)
                    self.assertEqual(worker["running_requests"], 2)
                    self.assertEqual(worker["max_batch_size"], 8)
                    self.assertAlmostEqual(worker["kv_usage_ratio"], 0.4)
                    self.assertEqual(worker["requests_total"], 7)
                    self.assertEqual(worker["state"], "ready")
                    self.assertEqual(worker["worker_received_requests_total"], 9)
                    self.assertEqual(worker["untracked_requests_total"], 2)
                    self.assertEqual(worker["worker_active_requests"], 2)
                    self.assertEqual(worker["worker_failed_requests_total"], 1)
                    self.assertEqual(worker["usage_total_tokens_total"], 18)
                    self.assertGreaterEqual(worker["lease_remaining_seconds"], 0)
                    self.assertGreaterEqual(worker["load_snapshot_age_seconds"], 0)
                    self.assertGreater(worker["last_success_timestamp"], 0)
                    self.assertIn("p95", snapshot["latency"]["request_duration"])
                    self.assertIn("p99", snapshot["latency"]["ttft"])

                    metrics_response = await client.get("/metrics")
                    self.assertEqual(metrics_response.status_code, 200)
                    self.assertIn("infinilm_worker_received_requests_total", metrics_response.text)
                    self.assertIn("infinilm_worker_untracked_requests_total", metrics_response.text)
                    self.assertIn('infinilm_worker_state{', metrics_response.text)

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
                        router_mode="least-loaded",
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

                        metrics = await client.get("/metrics")
                        self.assertEqual(metrics.status_code, 200)
                        self.assertIn(
                            'infinilm_router_mode{mode="least-loaded"} 1',
                            metrics.text,
                        )
                        self.assertIn(
                            "infinilm_router_request_duration_seconds_bucket",
                            metrics.text,
                        )

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

                        snapshot = await client.get("/api/monitoring")
                        self.assertEqual(snapshot.status_code, 200)
                        monitoring = snapshot.json()
                        self.assertEqual(monitoring["global"]["requests_total"], 1)
                        self.assertEqual(
                            monitoring["global"]["non_streaming_requests_total"], 1
                        )
                        self.assertEqual(
                            monitoring["global"]["streaming_requests_total"], 0
                        )
                        self.assertEqual(
                            monitoring["global"]["usage_prompt_tokens_total"], 3
                        )
                        self.assertEqual(
                            monitoring["global"]["usage_completion_tokens_total"], 2
                        )
                        self.assertEqual(
                            monitoring["global"]["usage_total_tokens_total"], 5
                        )
                        self.assertEqual(monitoring["workers"][0]["requests_total"], 1)
                        self.assertEqual(
                            monitoring["workers"][0]["usage_total_tokens_total"], 5
                        )
                        rendered_metrics = await client.get("/metrics")
                        self.assertIn(
                            "infinilm_router_usage_total_tokens_total 5",
                            rendered_metrics.text,
                        )
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
                router = ClusterRouter(
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
                    self.assertEqual(router.metrics.requests_total, 1)
                    self.assertEqual(router.metrics.streaming_requests_total, 1)
                    self.assertEqual(router.metrics.non_streaming_requests_total, 0)
                    self.assertEqual(router.metrics.attempts_total, 1)
                    self.assertEqual(router.metrics.successful_requests_total, 1)
                    self.assertEqual(router.metrics.tokens_total, 2)
                    self.assertEqual(router.metrics.ttft.count, 1)
                    self.assertEqual(router.metrics.inter_token_latency.count, 1)
                    self.assertEqual(router.metrics.request_duration.count, 1)
                    self.assertIn(
                        "infinilm_router_ttft_seconds_bucket",
                        router.metrics.render(1, [record]),
                    )
                    self.assertIn(
                        "infinilm_router_inter_token_latency_seconds_bucket",
                        router.metrics.render(1, [record]),
                    )
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
                router = ClusterRouter(
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
                    self.assertEqual(router.metrics.retries_total, 1)
                    self.assertEqual(router.metrics.attempts_total, 2)
                    self.assertEqual(router.metrics.requests_total, 1)
                    self.assertEqual(
                        router.metrics.worker_errors[bad.instance_id],
                        1,
                    )
                finally:
                    await transport_http.aclose()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
