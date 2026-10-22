import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from infinilm.cluster.aggregated.frontend import AggregatedFrontend
from infinilm.cluster.aggregated.router import AggregatedRouter
from infinilm.cluster.config import (
    FrontendOptions,
    parse_frontend_args,
    parse_worker_cluster_args,
)
from infinilm.cluster.disaggregated.frontend import (
    DisaggregatedFrontend,
    DisaggregatedFrontendOptions,
    parse_disaggregated_frontend_args,
)
from infinilm.cluster.environment import (
    bootstrap_timeout_seconds,
    connect_timeout,
    drain_grace_seconds,
    health_interval_seconds,
    heartbeat_interval_seconds,
    registry_ttl_seconds,
    request_timeout,
)
from infinilm.cluster.worker_registry import create_worker_registry


class ClusterTimeoutEnvironmentTest(unittest.TestCase):
    def test_connect_timeout_reads_environment(self):
        self.assertEqual(connect_timeout({}), 3.0)
        self.assertEqual(
            connect_timeout({"INFINILM_CLUSTER_CONNECT_TIMEOUT": ""}),
            3.0,
        )
        self.assertEqual(
            connect_timeout({"INFINILM_CLUSTER_CONNECT_TIMEOUT": "5"}),
            5.0,
        )

        for raw_value in ("invalid", "0", "-1", "nan", "inf"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                connect_timeout({"INFINILM_CLUSTER_CONNECT_TIMEOUT": raw_value})

    def test_request_timeout_reads_environment(self):
        self.assertEqual(request_timeout({}), 1000.0)
        self.assertEqual(
            request_timeout({"INFINILM_CLUSTER_REQUEST_TIMEOUT": ""}),
            1000.0,
        )
        self.assertEqual(
            request_timeout({"INFINILM_CLUSTER_REQUEST_TIMEOUT": "120"}),
            120.0,
        )

        for raw_value in ("invalid", "0", "-1", "nan", "inf"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                request_timeout({"INFINILM_CLUSTER_REQUEST_TIMEOUT": raw_value})

    def test_bootstrap_timeout_reads_environment(self):
        self.assertEqual(bootstrap_timeout_seconds({}), 3.0)
        self.assertEqual(
            bootstrap_timeout_seconds(
                {"INFINILM_CLUSTER_BOOTSTRAP_TIMEOUT_SECONDS": ""}
            ),
            3.0,
        )
        self.assertEqual(
            bootstrap_timeout_seconds(
                {"INFINILM_CLUSTER_BOOTSTRAP_TIMEOUT_SECONDS": "0.5"}
            ),
            0.5,
        )

        for raw_value in ("invalid", "0", "-1", "nan", "inf"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                bootstrap_timeout_seconds(
                    {"INFINILM_CLUSTER_BOOTSTRAP_TIMEOUT_SECONDS": raw_value}
                )

    def test_health_interval_reads_environment(self):
        self.assertEqual(health_interval_seconds({}), 1.0)
        self.assertEqual(
            health_interval_seconds({"INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS": ""}),
            1.0,
        )
        self.assertEqual(
            health_interval_seconds(
                {"INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS": "0.5"}
            ),
            0.5,
        )

        for raw_value in ("invalid", "0", "-1", "nan", "inf"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                health_interval_seconds(
                    {"INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS": raw_value}
                )

    def test_drain_grace_reads_environment(self):
        self.assertEqual(drain_grace_seconds({}), 1.0)
        self.assertEqual(
            drain_grace_seconds({"INFINILM_CLUSTER_DRAIN_GRACE_SECONDS": ""}),
            1.0,
        )
        self.assertEqual(
            drain_grace_seconds({"INFINILM_CLUSTER_DRAIN_GRACE_SECONDS": "3"}),
            3.0,
        )
        self.assertEqual(
            drain_grace_seconds({"INFINILM_CLUSTER_DRAIN_GRACE_SECONDS": "0"}),
            0.0,
        )

        for raw_value in ("invalid", "-1", "nan", "inf"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                drain_grace_seconds({"INFINILM_CLUSTER_DRAIN_GRACE_SECONDS": raw_value})

    def test_registry_ttl_reads_environment(self):
        self.assertEqual(registry_ttl_seconds({}), 10.0)
        self.assertEqual(
            registry_ttl_seconds({"INFINILM_CLUSTER_REGISTRY_TTL_SECONDS": ""}),
            10.0,
        )
        self.assertEqual(
            registry_ttl_seconds({"INFINILM_CLUSTER_REGISTRY_TTL_SECONDS": "30"}),
            30.0,
        )

        for raw_value in ("invalid", "0", "-1", "nan", "inf"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                registry_ttl_seconds(
                    {"INFINILM_CLUSTER_REGISTRY_TTL_SECONDS": raw_value}
                )

    def test_heartbeat_interval_reads_environment(self):
        self.assertEqual(heartbeat_interval_seconds({}), 2.0)
        self.assertEqual(
            heartbeat_interval_seconds(
                {"INFINILM_CLUSTER_HEARTBEAT_INTERVAL_SECONDS": ""}
            ),
            2.0,
        )
        self.assertEqual(
            heartbeat_interval_seconds(
                {"INFINILM_CLUSTER_HEARTBEAT_INTERVAL_SECONDS": "0.5"}
            ),
            0.5,
        )

        for raw_value in ("invalid", "0", "-1", "nan", "inf"):
            with self.subTest(raw_value=raw_value), self.assertRaises(ValueError):
                heartbeat_interval_seconds(
                    {"INFINILM_CLUSTER_HEARTBEAT_INTERVAL_SECONDS": raw_value}
                )

    def test_timeout_cli_options_are_removed(self):
        for option in (
            "--connect-timeout",
            "--request-timeout",
            "--registry-ttl",
            "--heartbeat-interval",
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parse_frontend_args([option, "5"])

        for option in (
            "--connect-timeout",
            "--request-timeout",
            "--registry-ttl",
            "--heartbeat-interval",
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parse_disaggregated_frontend_args([option, "5"])

        for parser in (parse_frontend_args, parse_disaggregated_frontend_args):
            with self.subTest(parser=parser), self.assertRaises(SystemExit):
                parser(["--health-interval", "1"])

        with self.assertRaises(ValueError):
            parse_worker_cluster_args(["--drain-grace-seconds", "3"])

        for option in ("--registry-ttl", "--heartbeat-interval"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                parse_worker_cluster_args([option, "5"])

    def test_router_reads_timeouts_from_environment(self):
        watcher = SimpleNamespace(worker_records=[])
        environment = {
            "INFINILM_CLUSTER_CONNECT_TIMEOUT": "4",
            "INFINILM_CLUSTER_REQUEST_TIMEOUT": "120",
        }

        with patch.dict("os.environ", environment):
            router = AggregatedRouter(watcher)

        self.assertEqual(router.connect_timeout, 4.0)
        self.assertEqual(router.request_timeout, 120.0)

    def test_registry_factory_reads_ttl_from_environment(self):
        with TemporaryDirectory() as temporary:
            options = SimpleNamespace(
                discovery_backend="file",
                file_kv=Path(temporary),
            )

            with patch.dict(
                "os.environ",
                {"INFINILM_CLUSTER_REGISTRY_TTL_SECONDS": "17"},
            ):
                registry = create_worker_registry(options)

            self.assertEqual(registry.lease_seconds, 17.0)

    def test_pd_frontend_reads_timeouts_from_environment(self):
        with TemporaryDirectory() as temporary:
            options = DisaggregatedFrontendOptions(
                namespace="test",
                discovery_root=Path(temporary),
                router_mode="round-robin",
                http_host="127.0.0.1",
                http_port=18520,
            )
            environment = {
                "INFINILM_CLUSTER_CONNECT_TIMEOUT": "4",
                "INFINILM_CLUSTER_REQUEST_TIMEOUT": "120",
                "INFINILM_CLUSTER_BOOTSTRAP_TIMEOUT_SECONDS": "0.5",
                "INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS": "0.1",
            }

            with patch.dict("os.environ", environment):
                frontend = DisaggregatedFrontend(options)

            self.assertEqual(frontend._connect_timeout, 4.0)
            self.assertEqual(frontend._request_timeout, 120.0)
            self.assertEqual(frontend._bootstrap_timeout, 0.5)
            self.assertEqual(frontend._discovery_refresh_interval_seconds, 0.1)
            asyncio.run(frontend.registry.aclose())

    def test_frontend_reads_health_interval_from_environment(self):
        with TemporaryDirectory() as temporary:
            options = FrontendOptions(
                namespace="test",
                file_kv=Path(temporary),
                router_mode="round-robin",
                router_min_initial_workers=1,
                http_host="127.0.0.1",
                http_port=18000,
            )

            with patch.dict(
                "os.environ",
                {"INFINILM_CLUSTER_HEALTH_INTERVAL_SECONDS": "0.5"},
            ):
                frontend = AggregatedFrontend(options)

            self.assertEqual(frontend._health_interval_seconds, 0.5)
            asyncio.run(frontend.registry.aclose())


if __name__ == "__main__":
    unittest.main()
