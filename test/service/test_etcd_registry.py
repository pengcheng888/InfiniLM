import asyncio
import json
import unittest

from infinilm.cluster.config import parse_frontend_args, parse_worker_cluster_args
from infinilm.cluster.etcd_registry import EtcdServiceRegistry, _parse_endpoint
from infinilm.cluster.registry import FileServiceRegistry
from infinilm.cluster.registry_factory import create_service_registry


class FakeLease:
    def __init__(self, client, lease_id, ttl):
        self.id = lease_id
        self.ttl = ttl
        self.etcd_client = client
        self.refresh_count = 0
        self.revoked = False

    def refresh(self):
        self.refresh_count += 1
        return []

    def revoke(self):
        self.revoked = True
        self.etcd_client.expire_lease(self.id)


class FakeEtcdClient:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 2379
        self.items = {}
        self.next_lease_id = 1
        self.closed = False

    def lease(self, ttl):
        lease = FakeLease(self, self.next_lease_id, ttl)
        self.next_lease_id += 1
        return lease

    def put(self, key, value, lease=None):
        self.items[key] = (
            value.encode(),
            {"key": key.encode(), "lease_id": lease.id if lease else 0},
        )

    def get_prefix(self, prefix):
        return [
            (value, metadata)
            for key, (value, metadata) in self.items.items()
            if key.startswith(prefix)
        ]

    def delete(self, key):
        return self.items.pop(key, None) is not None

    def expire_lease(self, lease_id):
        for key in list(self.items):
            if self.items[key][1]["lease_id"] == lease_id:
                del self.items[key]

    def close(self):
        self.closed = True


class EtcdRegistryTest(unittest.TestCase):
    def test_discovery_backend_configuration_and_factory(self):
        worker = parse_worker_cluster_args(
            [
                "--discovery-backend",
                "etcd",
                "--etcd-endpoints",
                "127.0.0.1:12379",
                "--etcd-prefix",
                "/infinilm-test",
            ]
        )
        frontend = parse_frontend_args(
            [
                "--discovery-backend",
                "etcd",
                "--etcd-endpoints",
                "127.0.0.1:12379",
                "--etcd-prefix",
                "/infinilm-test",
            ]
        )

        self.assertIsInstance(create_service_registry(worker), EtcdServiceRegistry)
        self.assertIsInstance(create_service_registry(frontend), EtcdServiceRegistry)
        self.assertIsInstance(
            create_service_registry(parse_worker_cluster_args([])),
            FileServiceRegistry,
        )

    def test_parse_endpoint_accepts_host_and_url(self):
        self.assertEqual(_parse_endpoint("127.0.0.1:2379"), ("127.0.0.1", 2379))
        self.assertEqual(
            _parse_endpoint("http://etcd.example:2380"),
            ("etcd.example", 2380),
        )
        with self.assertRaises(ValueError):
            _parse_endpoint("http://")

    def test_register_snapshot_heartbeat_and_unregister(self):
        async def run():
            client = FakeEtcdClient()
            registry = EtcdServiceRegistry(
                endpoints="127.0.0.1:2379",
                prefix="/infinilm-test",
                lease_seconds=10,
            )
            registry._clients = [client]
            registry._client_index = 0

            record = registry.new_record(
                namespace="pd",
                component="prefill",
                endpoint="generate",
                worker_type="prefill",
                model="/models/qwen",
                served_model_name="qwen",
                host="127.0.0.1",
                port=1000,
            )
            registered = await registry.register(record)

            self.assertEqual(
                await registry.snapshot("pd", worker_types=("prefill", "decode")),
                [registered],
            )
            self.assertEqual(
                await registry.snapshot("pd", worker_types=("decode",)), []
            )
            self.assertEqual(await registry.snapshot("other"), [])

            heartbeat = await registry.heartbeat(
                registered,
                engine_healthy=False,
            )
            snapshot = await registry.snapshot("pd")
            self.assertEqual(snapshot, [])
            self.assertTrue(client.items[registry._key(registered)][1]["lease_id"] > 0)
            lease = list(registry._leases.values())[0]
            self.assertEqual(lease.refresh_count, 1)
            self.assertGreaterEqual(heartbeat.heartbeat_at, registered.heartbeat_at)

            await registry.unregister(heartbeat)
            self.assertEqual(await registry.snapshot("pd"), [])
            self.assertTrue(lease.revoked)

            await registry.aclose()
            self.assertTrue(client.closed)

        asyncio.run(run())

    def test_snapshot_ignores_malformed_values_and_lease_expiry_removes_worker(self):
        async def run():
            client = FakeEtcdClient()
            registry = EtcdServiceRegistry(
                endpoints="127.0.0.1:2379",
                prefix="/infinilm-test",
            )
            registry._clients = [client]
            record = registry.new_record(
                namespace="pd",
                component="backend",
                endpoint="generate",
                worker_type="decode",
                model="/models/qwen",
                served_model_name="qwen",
                host="127.0.0.1",
                port=1001,
            )
            await registry.register(record)
            client.items["/infinilm-test/discovery/bad"] = (
                b"{not-json",
                {"lease_id": 0},
            )

            workers = await registry.snapshot("pd", worker_types=("prefill", "decode"))
            self.assertEqual(
                [worker.instance_id for worker in workers],
                [record.instance_id],
            )

            lease = list(registry._leases.values())[0]
            client.expire_lease(lease.id)
            self.assertEqual(
                await registry.snapshot("pd", worker_types=("prefill", "decode")),
                [],
            )

            payload = json.dumps({"not": "a-worker-record"}).encode()
            client.items["/infinilm-test/discovery/invalid"] = (
                payload,
                {"lease_id": 0},
            )
            self.assertEqual(
                await registry.snapshot("pd", worker_types=("prefill", "decode")),
                [],
            )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
