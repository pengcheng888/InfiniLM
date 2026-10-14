import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi.testclient import TestClient
from infinilm.cluster.config import (
    WorkerOptions,
    parse_worker_cluster_args,
)
from infinilm.cluster.pd_frontend import (
    PdClusterFrontend,
    PdFrontendOptions,
    parse_pd_frontend_args,
)
from infinilm.cluster.registry import FileServiceRegistry, FileServiceWatcher
from infinilm.cluster.worker import (
    WORKER_CAPABILITIES,
    WORKER_NEEDS,
    ClusterInferenceServer,
    _resolve_kv_transfer_config,
)
from infinilm.cluster.worker_traffic import WorkerTrafficMetrics
from infinilm.config import KVTransferConfig


class FakeEngine:
    def is_healthy(self) -> bool:
        return True


def make_record(registry: FileServiceRegistry, worker_type: str, model="qwen"):
    return registry.new_record(
        namespace="pd-test",
        component="infinilm",
        endpoint="generate",
        worker_type=worker_type,
        model="/models/qwen",
        served_model_name=model,
        host="127.0.0.1",
        port=1000,
        metadata={
            "capabilities": list(WORKER_CAPABILITIES[worker_type]),
            "needs": list(WORKER_NEEDS[worker_type]),
        },
    )


class PdDiscoveryTest(unittest.TestCase):
    def test_registry_and_watcher_filter_worker_roles(self):
        async def run():
            with TemporaryDirectory() as temporary:
                registry = FileServiceRegistry(Path(temporary))
                records = [
                    make_record(registry, "aggregated"),
                    make_record(registry, "prefill"),
                    make_record(registry, "decode"),
                ]
                for record in records:
                    await registry.register(record)

                self.assertEqual(
                    [record.worker_type for record in await registry.snapshot("pd-test")],
                    ["aggregated"],
                )

                watcher = FileServiceWatcher(
                    registry,
                    worker_types=("prefill", "decode"),
                )
                workers = await watcher.refresh("pd-test")
                self.assertEqual(
                    sorted(worker.worker_type for worker in workers),
                    ["decode", "prefill"],
                )

        asyncio.run(run())

    def test_pd_worker_exports_mooncake_metadata(self):
        with TemporaryDirectory() as temporary:
            server = ClusterInferenceServer.__new__(ClusterInferenceServer)
            server.cluster_options = WorkerOptions(
                namespace="pd-test",
                file_kv=Path(temporary),
                registry_ttl=10,
                heartbeat_interval=2,
                served_model_name="qwen",
                advertise_host="127.0.0.1",
                worker_type="prefill",
                mooncake_bootstrap_port=18528,
            )
            server.registry = FileServiceRegistry(Path(temporary))
            server.model_path = "/models/qwen"
            server.host = "127.0.0.1"
            server.port = 18521
            server.tensor_parallel_size = 1
            server.max_batch_size = 8
            server.worker_traffic = WorkerTrafficMetrics()
            server.kv_transfer_config = KVTransferConfig(
                kv_connector="MooncakeConnector",
                kv_role="kv_producer",
                engine_id="prefill-1",
            )

            discovery = server._create_discovery_service(FakeEngine())
            metadata = discovery.record.metadata["pd"]

            self.assertEqual(metadata["engine_id"], "prefill-1")
            self.assertEqual(metadata["kv_role"], "kv_producer")
            self.assertEqual(metadata["bootstrap_port"], 18528)
            self.assertEqual(
                discovery.record.metadata["capabilities"],
                ["internal-prefill"],
            )
            self.assertEqual(discovery.record.metadata["needs"], ["decode"])
            self.assertIsNone(discovery.cache_index_exporter)

    def test_pd_frontend_model_and_load_selection(self):
        with TemporaryDirectory() as temporary:
            registry = FileServiceRegistry(Path(temporary))
            prefill = make_record(registry, "prefill")
            decode = make_record(registry, "decode")
            loaded = replace_load(decode)
            frontend = PdClusterFrontend(
                PdFrontendOptions(
                    namespace="pd-test",
                    file_kv=Path(temporary),
                    router_mode="least-loaded",
                    http_host="127.0.0.1",
                    http_port=18520,
                    connect_timeout=1,
                    request_timeout=1,
                    health_interval=0.1,
                    registry_ttl=10,
                    heartbeat_interval=2,
                )
            )
            frontend.watcher = SimpleNamespace(workers=[prefill, loaded])

            self.assertEqual(frontend._ready_models(), ["qwen"])
            self.assertEqual(frontend._select_record([decode, loaded]).instance_id, decode.instance_id)

    def test_cli_parses_worker_and_frontend_roles(self):
        mode = parse_worker_cluster_args(
            [
                "--disaggregation-mode",
                "prefill",
                "--mooncake-bootstrap-port",
                "18528",
            ]
        )
        self.assertEqual(mode.disaggregation_mode, "prefill")
        self.assertEqual(mode.worker_type, "prefill")
        self.assertEqual(mode.component, "prefill")

        worker = parse_worker_cluster_args(
            [
                "--worker-type",
                "prefill",
                "--mooncake-bootstrap-port",
                "18528",
            ]
        )
        self.assertEqual(worker.worker_type, "prefill")
        self.assertEqual(worker.disaggregation_mode, "prefill")
        self.assertEqual(worker.mooncake_bootstrap_port, 18528)

        with self.assertRaises(ValueError):
            parse_worker_cluster_args(
                [
                    "--disaggregation-mode",
                    "decode",
                    "--worker-type",
                    "prefill",
                ]
            )

        frontend = parse_pd_frontend_args(
            [
                "--router-mode",
                "least-loaded",
                "--min-prefill-workers",
                "2",
            ]
        )
        self.assertEqual(frontend.router_mode, "least-loaded")
        self.assertEqual(frontend.min_prefill_workers, 2)

    def test_pd_frontend_round_robin_selects_each_role(self):
        with TemporaryDirectory() as temporary:
            registry = FileServiceRegistry(Path(temporary))
            workers = [
                make_record(registry, "prefill"),
                make_record(registry, "prefill"),
                make_record(registry, "decode"),
                make_record(registry, "decode"),
            ]
            frontend = PdClusterFrontend(
                PdFrontendOptions(
                    namespace="pd-test",
                    file_kv=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                    connect_timeout=1,
                    request_timeout=1,
                    health_interval=0.1,
                    registry_ttl=10,
                    heartbeat_interval=2,
                )
            )
            frontend.watcher = SimpleNamespace(workers=workers)

            prefill_ids = [
                frontend._select_record(
                    frontend._role_workers(workers, "prefill")
                ).instance_id
                for _ in range(2)
            ]
            decode_ids = [
                frontend._select_record(
                    frontend._role_workers(workers, "decode")
                ).instance_id
                for _ in range(2)
            ]

            self.assertNotEqual(prefill_ids[0], prefill_ids[1])
            self.assertNotEqual(decode_ids[0], decode_ids[1])

    def test_pd_monitoring_routes_expose_role_workers(self):
        with TemporaryDirectory() as temporary:
            registry = FileServiceRegistry(Path(temporary))
            prefill = replace_pd(make_record(registry, "prefill"))
            decode = replace_pd(make_record(registry, "decode"))
            frontend = PdClusterFrontend(
                PdFrontendOptions(
                    namespace="pd-test",
                    file_kv=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                    connect_timeout=1,
                    request_timeout=1,
                    health_interval=0.1,
                    registry_ttl=10,
                    heartbeat_interval=2,
                )
            )
            frontend.watcher = SimpleNamespace(workers=[prefill, decode])
            client = TestClient(frontend.create_app())

            page = client.get("/monitoring")
            snapshot = client.get("/api/monitoring").json()

            self.assertEqual(page.status_code, 200)
            self.assertIn("InfiniLM PD Monitor", page.text)
            self.assertEqual(snapshot["namespace"], "pd-test")
            self.assertEqual(
                [worker["role"] for worker in snapshot["workers"]],
                ["prefill", "decode"],
            )
            self.assertEqual(snapshot["requests_total"]["prefill"], 0)

    def test_disaggregation_mode_derives_and_validates_kv_role(self):
        config = _resolve_kv_transfer_config(
            '{"kv_connector":"MooncakeConnector","engine_id":"p"}',
            "prefill",
        )
        self.assertEqual(config.kv_role, "kv_producer")

        with self.assertRaises(ValueError):
            _resolve_kv_transfer_config(
                '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}',
                "prefill",
            )

    def test_prefill_openai_surface_requires_internal_header(self):
        with TemporaryDirectory() as temporary:
            server = ClusterInferenceServer.__new__(ClusterInferenceServer)
            server.cluster_options = WorkerOptions(
                namespace="pd-test",
                file_kv=Path(temporary),
                registry_ttl=10,
                heartbeat_interval=2,
                served_model_name="qwen",
                advertise_host="127.0.0.1",
                disaggregation_mode="prefill",
                worker_type="prefill",
            )
            server.kv_transfer_config = KVTransferConfig(
                kv_connector="MooncakeConnector",
                kv_role="kv_producer",
                engine_id="prefill-guard",
            )
            response = TestClient(server._create_app()).post(
                "/v1/chat/completions",
                json={"model": "qwen", "messages": []},
            )

            self.assertEqual(response.status_code, 403)
            self.assertIn("only accepts internal requests", response.text)

    def test_frontend_starts_without_workers_and_reports_not_ready(self):
        with TemporaryDirectory() as temporary:
            frontend = PdClusterFrontend(
                PdFrontendOptions(
                    namespace="pd-test",
                    file_kv=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                    connect_timeout=1,
                    request_timeout=1,
                    health_interval=0.05,
                    registry_ttl=1,
                    heartbeat_interval=0.2,
                )
            )

            with TestClient(frontend.create_app()) as client:
                health = client.get("/health").json()
                request = client.post(
                    "/v1/chat/completions",
                    json={"messages": []},
                )

            self.assertEqual(health["status"], "healthy")
            self.assertFalse(health["ready"])
            self.assertEqual(request.status_code, 503)

    def test_frontend_filters_workers_without_required_capability(self):
        with TemporaryDirectory() as temporary:
            registry = FileServiceRegistry(Path(temporary))
            capable = make_record(registry, "decode")
            legacy = replace_metadata(make_record(registry, "decode"), {})
            frontend = PdClusterFrontend(
                PdFrontendOptions(
                    namespace="pd-test",
                    file_kv=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                    connect_timeout=1,
                    request_timeout=1,
                    health_interval=0.1,
                    registry_ttl=10,
                    heartbeat_interval=2,
                )
            )

            workers = frontend._role_workers(
                [capable, legacy],
                "decode",
                "qwen",
                "openai-chat",
            )

            self.assertEqual([worker.instance_id for worker in workers], [capable.instance_id])


def replace_load(record):
    from dataclasses import replace

    return replace(
        record,
        metadata={
            **record.metadata,
            "load": {
                "waiting_requests": 2,
                "running_requests": 1,
                "max_batch_size": 4,
            },
        },
    )


def replace_pd(record):
    from dataclasses import replace

    return replace(
        record,
        metadata={
            **record.metadata,
            "pd": {
                "engine_id": f"{record.worker_type}-engine",
                "kv_role": record.worker_type,
                "bootstrap_host": "127.0.0.1",
                "bootstrap_port": 18528,
            },
        },
    )


def replace_metadata(record, metadata):
    from dataclasses import replace

    return replace(record, metadata=metadata)


if __name__ == "__main__":
    unittest.main()
