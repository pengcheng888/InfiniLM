"""Tests for session-affinity routing and its key store."""

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from infinilm.cluster.aggregated.affinity import AffinityStore, affinity_key
from infinilm.cluster.aggregated.frontend import AggregatedFrontend
from infinilm.cluster.aggregated.http_worker_client_factory import (
    HttpWorkerClientFactory,
)
from infinilm.cluster.aggregated.router import AggregatedRouter
from infinilm.cluster.config import FrontendOptions, parse_frontend_args
from infinilm.cluster.environment import affinity_capacity, affinity_ttl_seconds
from infinilm.cluster.worker_registry import FileWorkerRegistry, WorkerRegistryWatcher


class AffinityTest(unittest.TestCase):
    def test_affinity_store_expires_and_evicts_lru_entries(self):
        now = 100.0

        def clock() -> float:
            return now

        store = AffinityStore(ttl_seconds=10, capacity=2, clock=clock)
        store.assign("a", "worker-a")
        now = 105.0
        store.assign("b", "worker-b")
        store.get("a", {"worker-a"})

        now = 106.0
        store.assign("c", "worker-c")
        self.assertNotIn("b", store._entries)
        self.assertIn("a", store._entries)

        now = 116.0
        lookup = store.get("a", {"worker-a"})
        self.assertFalse(lookup.hit)
        self.assertTrue(lookup.expired)
        self.assertNotIn("a", store._entries)
        self.assertIn("c", store._entries)

    def test_affinity_key_uses_identity_and_stable_conversation_prefix(self):
        identity = affinity_key(
            {"user": "user-1"},
            {"x-session-id": "session-1"},
        )
        self.assertEqual(
            identity,
            affinity_key({"user": "other-user"}, {"x-session-id": "session-1"}),
        )
        self.assertNotEqual(identity, affinity_key({"user": "user-2"}, {}))

        first_turn = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "hello"},
            ],
        }
        next_turn = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "continue"},
            ],
        }
        self.assertEqual(affinity_key(first_turn), affinity_key(next_turn))
        self.assertIsNotNone(affinity_key(first_turn))


class SessionAffinityRoutingTest(unittest.TestCase):
    def test_session_affinity_mode_is_accepted(self):
        options = parse_frontend_args(["--router-mode", "session-affinity"])
        self.assertEqual(options.router_mode, "session-affinity")

    def test_affinity_capacity_reads_environment(self):
        self.assertEqual(affinity_capacity({}), 4096)
        self.assertEqual(
            affinity_capacity({"INFINILM_CLUSTER_AFFINITY_CAPACITY": ""}),
            4096,
        )
        self.assertEqual(
            affinity_capacity({"INFINILM_CLUSTER_AFFINITY_CAPACITY": "128"}),
            128,
        )

        for raw_value in ("invalid", "0", "-1", "1.5"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                affinity_capacity({"INFINILM_CLUSTER_AFFINITY_CAPACITY": raw_value})

    def test_affinity_ttl_seconds_reads_environment(self):
        self.assertEqual(affinity_ttl_seconds({}), 300.0)
        self.assertEqual(
            affinity_ttl_seconds({"INFINILM_CLUSTER_AFFINITY_TTL_SECONDS": ""}),
            300.0,
        )
        self.assertEqual(
            affinity_ttl_seconds({"INFINILM_CLUSTER_AFFINITY_TTL_SECONDS": "60"}),
            60.0,
        )
        self.assertEqual(
            affinity_ttl_seconds({"INFINILM_CLUSTER_AFFINITY_TTL_SECONDS": "0"}),
            0.0,
        )

        for raw_value in ("invalid", "-1", "nan", "inf"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                affinity_ttl_seconds(
                    {"INFINILM_CLUSTER_AFFINITY_TTL_SECONDS": raw_value}
                )

    def test_affinity_ttl_is_read_only_for_session_policy(self):
        watcher = SimpleNamespace(worker_records=[])

        with patch(
            "infinilm.cluster.aggregated.routing.policies.session_affinity.affinity_ttl_seconds",
            return_value=60.0,
        ) as read_ttl:
            AggregatedRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="round-robin",
            )
            read_ttl.assert_not_called()

            AggregatedRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="session-affinity",
            )
            read_ttl.assert_called_once_with()

    def test_affinity_capacity_is_read_only_for_session_policy(self):
        watcher = SimpleNamespace(worker_records=[])

        with patch(
            "infinilm.cluster.aggregated.routing.policies.session_affinity.affinity_capacity",
            return_value=128,
        ) as read_capacity:
            AggregatedRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="least-loaded",
            )
            read_capacity.assert_not_called()

            AggregatedRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode="session-affinity",
            )
            read_capacity.assert_called_once_with()

    def test_affinity_store_is_created_only_for_session_affinity(self):
        watcher = SimpleNamespace(worker_records=[])

        for router_mode in ("round-robin", "least-loaded", "kv-aware"):
            router = AggregatedRouter(
                watcher,
                connect_timeout=1.0,
                request_timeout=2.0,
                router_mode=router_mode,
            )
            self.assertEqual(router.affinity_entries, 0)
            router.update_worker_metrics([])

        router = AggregatedRouter(
            watcher,
            connect_timeout=1.0,
            request_timeout=2.0,
            router_mode="session-affinity",
        )
        self.assertEqual(router.affinity_entries, 0)
        router.update_worker_metrics([])

    def test_session_affinity_reuses_and_falls_back(self):
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
                router_mode="session-affinity",
            )
            # 用 user 生成稳定的哈希路由 key，模拟同一会话的连续请求。
            key = affinity_key({"user": "user-1"})

            first = router.select(records, routing_key=key)
            second = router.select(records, routing_key=key)
            self.assertIs(first, second)
            self.assertEqual(router.metrics.affinity_requests_total, 2)
            self.assertEqual(router.metrics.affinity_hits_total, 1)
            self.assertEqual(router.metrics.affinity_misses_total, 1)
            self.assertEqual(router.affinity_entries, 1)

            watcher._worker_records = [
                record for record in records if record.instance_id != first.instance_id
            ]
            fallback = router.select(watcher.worker_records, routing_key=key)
            self.assertIsNot(fallback, first)
            self.assertEqual(router.metrics.affinity_misses_total, 2)

            for _ in range(5):
                router.worker_inflight.begin_request(fallback.instance_id)
            rebalanced = router.select(watcher.worker_records, routing_key=key)
            self.assertIsNot(rebalanced, fallback)
            self.assertEqual(router.metrics.affinity_misses_total, 3)

    def test_session_affinity_uses_frontend_header_and_hides_internal_key(self):
        visited_ports: list[int] = []

        def worker_transport(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
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
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )

        async def run():
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                registry = FileWorkerRegistry(root)
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
                    )
                    for port in (18091, 18092)
                ]
                for record in records:
                    await registry.register(record)

                frontend = AggregatedFrontend(
                    FrontendOptions(
                        namespace="test",
                        file_kv=root,
                        router_mode="session-affinity",
                        router_min_initial_workers=1,
                        http_host="127.0.0.1",
                        http_port=18000,
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
                        headers = {"X-Session-ID": "session-1"}
                        for _ in range(2):
                            response = await client.post(
                                "/v1/chat/completions",
                                headers=headers,
                                json={
                                    "model": "qwen",
                                    "messages": [{"role": "user", "content": "hi"}],
                                },
                            )
                            self.assertEqual(response.status_code, 200)

                    self.assertEqual(len(visited_ports), 2)
                    self.assertEqual(visited_ports[0], visited_ports[1])
                    self.assertEqual(frontend.router.metrics.affinity_requests_total, 2)
                    self.assertEqual(frontend.router.metrics.affinity_hits_total, 1)
                    self.assertEqual(frontend.router.metrics.affinity_misses_total, 1)
                finally:
                    await worker_http.aclose()

        asyncio.run(run())
