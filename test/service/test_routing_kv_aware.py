"""Tests for KV-aware routing and prefix-cache indexing."""

import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from infinilm.cluster.cache_index import WorkerCacheIndexExporter
from infinilm.cluster.config import FrontendOptions, parse_frontend_args
from infinilm.cluster.environment import kv_router_min_tokens
from infinilm.cluster.frontend import ClusterFrontend
from infinilm.cluster.http_worker_client_factory import HttpWorkerClientFactory
from infinilm.cluster.kv_index import KvIndexStore
from infinilm.cluster.kv_prefix import PrefixHashCalculator, prefix_hashes
from infinilm.cluster.router import ClusterRouter
from infinilm.cluster.worker import ClusterInferenceServer
from infinilm.cluster.worker_registry import FileWorkerRegistry, WorkerRegistryWatcher
from infinilm.llm.cache_manager import BlockManager
from infinilm.llm.request import InferenceRequest


class PrefixHashCalculatorTest(unittest.TestCase):
    def test_extracts_tokens_from_processor_batch_encoding(self):
        class BatchEncoding:
            def __getitem__(self, key):
                if key != "input_ids":
                    raise KeyError(key)
                return [[1, 2, 3], [4, 5, 6]]

        self.assertEqual(
            PrefixHashCalculator._input_ids(BatchEncoding()),
            [1, 2, 3],
        )

    def test_hashes_text_chat_prompt_with_worker_hash_algorithm(self):
        model_path = "/data-aisoft/mechdancer/models/Qwen3-0.6B"
        calculator = PrefixHashCalculator()
        payload = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {
                    "role": "user",
                    "content": "hello world, this prompt spans multiple blocks",
                },
            ],
        }
        result = calculator.calculate(payload, model_path=model_path, block_size=8)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["token_count"], len(result["hashes"]) * 8)
        self.assertGreater(len(result["hashes"]), 0)
        self.assertEqual(len(result["namespace"]), 64)

        multimodal = {
            "model": "qwen",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://invalid"}}
                    ],
                }
            ],
        }
        self.assertIsNone(
            calculator.calculate(multimodal, model_path=model_path, block_size=8)
        )


class CacheIndexTest(unittest.TestCase):
    def _request_with_tokens(self, token_ids: list[int]) -> InferenceRequest:
        request = InferenceRequest(
            request_id=str(len(token_ids)), prompt_token_ids=token_ids
        )
        request.initialize_block_hashes(block_size=4, enable_prefix_caching=True)
        return request

    def test_worker_cache_export_paths_are_isolated_by_instance(self):
        with TemporaryDirectory() as temporary:
            registry = FileWorkerRegistry(Path(temporary))
            manager = BlockManager(num_blocks=4, block_size=4)
            # SimpleNamespace 用作轻量测试替身，只提供被测代码需要的属性，
            # 模拟 engine.engine.scheduler 和 engine.config.block_size 结构。
            scheduler = SimpleNamespace(
                cache_manager=manager,
                enable_prefix_caching=True,
            )
            engine = SimpleNamespace(
                engine=SimpleNamespace(scheduler=scheduler),
                config=SimpleNamespace(block_size=4),
            )
            server = ClusterInferenceServer.__new__(ClusterInferenceServer)
            server.registry = registry
            server.model_path = "/models/qwen"
            server.block_size = 4

            first = server._create_cache_index_exporter(engine, "worker-a")
            second = server._create_cache_index_exporter(engine, "worker-b")

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertNotEqual(first.output_dir, second.output_dir)
            self.assertEqual(
                first.output_dir,
                Path(temporary).parent / "cache-index" / "worker-a",
            )

    def test_block_manager_prefix_lookup_and_events(self):
        async def run():
            with TemporaryDirectory() as temporary:
                manager = BlockManager(num_blocks=8, block_size=4)
                exporter = WorkerCacheIndexExporter(
                    block_manager=manager,
                    output_dir=Path(temporary),
                    namespace="test-namespace",
                    block_size=4,
                )
                first = self._request_with_tokens(list(range(8)))
                first.block_table, first.slot_mapping = manager.allocate_slots(8)
                await asyncio.to_thread(exporter.start)
                manager.publish_computed_blocks(
                    first.block_table,
                    first.block_hashes,
                    0,
                    8,
                )

                second = self._request_with_tokens(list(range(12)))
                cached, cached_tokens = manager.get_computed_blocks(
                    second.block_hashes, 11
                )
                self.assertEqual(len(cached), 2)
                self.assertEqual(cached_tokens, 8)

                manager.free_blocks(cached)
                await asyncio.to_thread(exporter.write_snapshot)
                for _ in range(50):
                    if exporter.metadata()["entries"] >= 2:
                        break
                    await asyncio.sleep(0.01)
                self.assertGreaterEqual(exporter.metadata()["entries"], 2)

                snapshot = json.loads(
                    exporter.snapshot_path.read_text(encoding="utf-8")
                )
                self.assertEqual(snapshot["namespace"], "test-namespace")
                self.assertEqual(len(snapshot["hashes"]), 2)

                manager.free_blocks(first.block_table)
                manager.try_free_blocks(8)
                self.assertEqual(manager.snapshot_prefix_index(), [])
                for _ in range(50):
                    if exporter.metadata()["entries"] == 0:
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(exporter.metadata()["entries"], 0)
                events = []
                for _ in range(50):
                    if exporter.events_path.exists():
                        events = [
                            json.loads(line)
                            for line in exporter.events_path.read_text(
                                encoding="utf-8"
                            ).splitlines()
                        ]
                        if "evicted" in {event["type"] for event in events}:
                            break
                    await asyncio.sleep(0.01)
                event_types = [event["type"] for event in events]
                self.assertIn("published", event_types)
                self.assertIn("evicted", event_types)
                await asyncio.to_thread(exporter.stop)

        asyncio.run(run())

    def test_frontend_index_reads_snapshot_and_events(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = FileWorkerRegistry(root)
            namespace = "test-namespace"
            hashes = prefix_hashes(list(range(12)), block_size=4)
            records = []
            for index, matched_hash_count in enumerate((3, 1)):
                output_dir = root / f"worker-{index}"
                output_dir.mkdir()
                snapshot_path = output_dir / "snapshot.json"
                events_path = output_dir / "events.jsonl"
                snapshot_path.write_text(
                    json.dumps(
                        {
                            "namespace": namespace,
                            "block_size": 4,
                            "sequence": 1,
                            "hashes": hashes[:matched_hash_count],
                        }
                    ),
                    encoding="utf-8",
                )
                events_path.write_text("", encoding="utf-8")
                record = registry.new_record(
                    namespace="test",
                    component="infinilm",
                    endpoint="generate",
                    worker_type="aggregated",
                    model="/models/qwen",
                    served_model_name="qwen",
                    host="127.0.0.1",
                    port=18081 + index,
                    metadata={
                        "kv_index": {
                            "namespace": namespace,
                            "block_size": 4,
                            "snapshot": str(snapshot_path),
                            "events": str(events_path),
                        }
                    },
                )
                records.append(record)

            index = KvIndexStore()
            index.refresh(records)
            matches = index.lookup(
                {
                    "namespace": namespace,
                    "block_size": 4,
                    "hashes": hashes,
                },
                records,
            )
            self.assertEqual(matches[0].worker_id, records[0].instance_id)
            self.assertEqual(matches[0].matched_blocks, 3)
            self.assertEqual(matches[0].matched_tokens, 12)
            self.assertEqual(index.entries(), 4)

            with Path(records[0].metadata["kv_index"]["events"]).open(
                "a", encoding="utf-8"
            ) as events:
                events.write(
                    json.dumps({"sequence": 2, "type": "evicted", "hash": hashes[2]})
                    + "\n"
                )
            index.refresh(records)
            matches = index.lookup(
                {"namespace": namespace, "block_size": 4, "hashes": hashes},
                records,
            )
            first = next(
                match for match in matches if match.worker_id == records[0].instance_id
            )
            self.assertEqual(first.matched_blocks, 2)
            self.assertEqual(index.entries(), 3)


class KvAwareRoutingTest(unittest.TestCase):
    def test_kv_aware_mode_is_accepted(self):
        options = parse_frontend_args(["--router-mode", "kv-aware"])
        self.assertEqual(options.router_mode, "kv-aware")

    def test_kv_router_min_tokens_reads_environment(self):
        self.assertEqual(kv_router_min_tokens({}), 1)
        self.assertEqual(
            kv_router_min_tokens({"INFINILM_CLUSTER_KV_ROUTER_MIN_TOKENS": ""}),
            1,
        )
        self.assertEqual(
            kv_router_min_tokens({"INFINILM_CLUSTER_KV_ROUTER_MIN_TOKENS": "256"}),
            256,
        )

        for raw_value in ("invalid", "0", "-1", "1.5"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                kv_router_min_tokens(
                    {"INFINILM_CLUSTER_KV_ROUTER_MIN_TOKENS": raw_value}
                )

    def test_kv_router_min_tokens_is_read_only_for_kv_policy(self):
        watcher = SimpleNamespace(workers=[])

        with patch(
            "infinilm.cluster.routing.policies.kv_aware.kv_router_min_tokens",
            return_value=8,
        ) as read_min_tokens:
            ClusterRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="least-loaded",
            )
            read_min_tokens.assert_not_called()

            ClusterRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="kv-aware",
            )
            read_min_tokens.assert_called_once_with()

    def test_kv_aware_routes_to_longest_cached_prefix(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = FileWorkerRegistry(root)
            namespace = "kv-router-test"
            hashes = prefix_hashes(list(range(12)), block_size=4)
            records = []
            for index, count in enumerate((1, 3)):
                output_dir = root / f"worker-{index}"
                output_dir.mkdir()
                snapshot_path = output_dir / "snapshot.json"
                events_path = output_dir / "events.jsonl"
                snapshot_path.write_text(
                    json.dumps(
                        {
                            "namespace": namespace,
                            "block_size": 4,
                            "sequence": 1,
                            "hashes": hashes[:count],
                        }
                    ),
                    encoding="utf-8",
                )
                events_path.write_text("", encoding="utf-8")
                records.append(
                    registry.new_record(
                        namespace="test",
                        component="infinilm",
                        endpoint="generate",
                        worker_type="aggregated",
                        model="/models/qwen",
                        served_model_name="qwen",
                        host="127.0.0.1",
                        port=18091 + index,
                        metadata={
                            "kv_index": {
                                "namespace": namespace,
                                "block_size": 4,
                                "snapshot": str(snapshot_path),
                                "events": str(events_path),
                            }
                        },
                    )
                )

            watcher = WorkerRegistryWatcher(registry)
            watcher._workers = records
            router = ClusterRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="kv-aware",
            )
            selected = router.select(
                records,
                kv_prefix={
                    "namespace": namespace,
                    "block_size": 4,
                    "hashes": hashes,
                },
            )

            self.assertIs(selected, records[1])
            self.assertEqual(router.metrics.kv_aware_requests_total, 1)
            self.assertEqual(router.metrics.kv_hit_requests_total, 1)
            self.assertEqual(router.metrics.kv_miss_requests_total, 0)
            self.assertEqual(router.metrics.kv_predicted_saved_tokens_total, 12)
            self.assertEqual(router.kv_index.entries(), 4)

    def test_kv_aware_frontend_routes_and_hides_internal_payload(self):
        visited_ports: list[int] = []

        def worker_transport(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertNotIn("__infinilm_kv_prefix__", payload)
            self.assertNotIn("__infinilm_routing_key", payload)
            visited_ports.append(request.url.port)
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
                        "prompt_tokens": 12,
                        "completion_tokens": 2,
                        "total_tokens": 14,
                    },
                    "cache": {
                        "hit": True,
                        "matched_tokens": 12,
                        "prefill_tokens": 0,
                        "prompt_tokens": 12,
                    },
                },
            )

        async def run():
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                registry = FileWorkerRegistry(root)
                namespace = "frontend-kv-test"
                hashes = prefix_hashes(list(range(12)), block_size=4)
                records = []
                for index, count in enumerate((0, 3)):
                    output_dir = root / f"worker-{index}"
                    output_dir.mkdir()
                    snapshot_path = output_dir / "snapshot.json"
                    events_path = output_dir / "events.jsonl"
                    snapshot_path.write_text(
                        json.dumps(
                            {
                                "namespace": namespace,
                                "block_size": 4,
                                "sequence": 1,
                                "hashes": hashes[:count],
                            }
                        ),
                        encoding="utf-8",
                    )
                    events_path.write_text("", encoding="utf-8")
                    records.append(
                        registry.new_record(
                            namespace="test",
                            component="infinilm",
                            endpoint="generate",
                            worker_type="aggregated",
                            model="/models/qwen",
                            served_model_name="qwen",
                            host="127.0.0.1",
                            port=18091 + index,
                            metadata={
                                "kv_index": {
                                    "namespace": namespace,
                                    "block_size": 4,
                                    "snapshot": str(snapshot_path),
                                    "events": str(events_path),
                                }
                            },
                        )
                    )
                for record in records:
                    await registry.register(record)

                with patch.dict(
                    os.environ,
                    {"INFINILM_CLUSTER_KV_ROUTER_MIN_TOKENS": "4"},
                ):
                    frontend = ClusterFrontend(
                        FrontendOptions(
                            namespace="test",
                            file_kv=root,
                            router_mode="kv-aware",
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

                prefix = {
                    "namespace": namespace,
                    "block_size": 4,
                    "hashes": hashes,
                    "token_count": 12,
                }
                try:
                    with patch.object(
                        frontend.prefix_calculator,
                        "calculate",
                        return_value=prefix,
                    ):
                        async with httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=app),
                            base_url="http://frontend",
                        ) as client:
                            response = await client.post(
                                "/v1/chat/completions",
                                json={
                                    "model": "qwen",
                                    "messages": [{"role": "user", "content": "hi"}],
                                },
                            )
                            self.assertEqual(response.status_code, 200)

                    self.assertEqual(visited_ports, [18092])
                    metrics = frontend.router.metrics
                    self.assertEqual(metrics.kv_aware_requests_total, 1)
                    self.assertEqual(metrics.kv_hit_requests_total, 1)
                    self.assertEqual(metrics.kv_predicted_saved_tokens_total, 12)
                    self.assertEqual(metrics.kv_actual_saved_tokens_total, 12)
                finally:
                    await worker_http.aclose()

        asyncio.run(run())


"""
cd /workspace_codex/InfiniLM
PYTHONPATH=python python -m pytest -q \
  test/service/test_routing_kv_aware.py
"""
