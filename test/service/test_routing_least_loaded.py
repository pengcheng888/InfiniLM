"""Tests for the least-loaded routing policy."""

import asyncio
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from infinilm.cluster.config import parse_frontend_args
from infinilm.cluster.router import ClusterRouter
from infinilm.cluster.worker_registry import FileWorkerRegistry, WorkerRegistryWatcher


class LeastLoadedModeTest(unittest.TestCase):
    def test_least_loaded_mode_is_accepted(self):
        options = parse_frontend_args(["--router-mode", "least-loaded"])
        self.assertEqual(options.router_mode, "least-loaded")


class LoadAwareRouterTest(unittest.TestCase):
    def test_least_loaded_inflight_is_dominant(self):
        with TemporaryDirectory() as temporary:
            registry = FileWorkerRegistry(Path(temporary))
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
            watcher = WorkerRegistryWatcher(registry)
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
            self.assertIn('infinilm_router_mode{mode="least-loaded"} 1', rendered)
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
                        transport="http",
                    )
                    for index in range(2)
                ]
                watcher = WorkerRegistryWatcher(registry)
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
            registry = FileWorkerRegistry(Path(temporary))
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
            watcher = WorkerRegistryWatcher(registry)
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
