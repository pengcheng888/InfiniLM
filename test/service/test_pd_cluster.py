import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from infinilm.cluster.config import (
    WorkerOptions,
    parse_worker_cluster_args,
)
from infinilm.cluster.disaggregated.frontend import (
    DisaggregatedFrontend,
    DisaggregatedFrontendOptions,
    _PrefetchedAsyncIterator,
    parse_disaggregated_frontend_args,
)
from infinilm.cluster.worker import (
    WORKER_NEEDS,
    WORKER_SUPPORTED_API_ROUTES,
    ClusterInferenceServer,
    _resolve_kv_transfer_config,
)
from infinilm.cluster.worker_registry import (
    FileWorkerRegistry,
    WorkerRegistryWatcher,
    WorkerType,
)
from infinilm.cluster.worker_traffic import WorkerTrafficMetrics
from infinilm.config import KVTransferConfig
from infinilm.server.inference_server import InferenceServer


class FakeEngine:
    def is_healthy(self) -> bool:
        return True


def make_record(registry: FileWorkerRegistry, worker_type: str, model="qwen"):
    resolved_worker_type = WorkerType(worker_type)
    return registry.new_record(
        namespace="pd-test",
        component="infinilm",
        endpoint="generate",
        worker_type=resolved_worker_type,
        model="/models/qwen",
        served_model_name=model,
        host="127.0.0.1",
        port=1000,
        metadata={
            "supported_api_routes": list(
                WORKER_SUPPORTED_API_ROUTES[resolved_worker_type]
            ),
            "needs": list(WORKER_NEEDS[resolved_worker_type]),
        },
    )


class PdDiscoveryTest(unittest.TestCase):
    def test_registry_and_watcher_filter_worker_roles(self):
        """默认 registry 只看聚合式 worker，PD watcher 单独过滤 P/D 角色。"""

        async def run():
            with TemporaryDirectory() as temporary:
                registry = FileWorkerRegistry(Path(temporary))
                # 同时注册三种角色，验证默认 snapshot 的角色隔离。
                records = [
                    make_record(registry, "aggregated"),
                    make_record(registry, "prefill"),
                    make_record(registry, "decode"),
                ]
                for record in records:
                    await registry.register(record)

                # 未显式指定 worker_types 时，registry 默认只返回 aggregated。
                self.assertEqual(
                    [
                        record.worker_type.value
                        for record in await registry.snapshot("pd-test")
                    ],
                    ["aggregated"],
                )

                watcher = WorkerRegistryWatcher(
                    registry,
                    worker_types=(WorkerType.PREFILL, WorkerType.DECODE),
                )
                # PD frontend 使用独立 watcher，只发现 prefill 和 decode。
                workers = await watcher.refresh("pd-test")
                self.assertEqual(
                    sorted(worker.worker_type.value for worker in workers),
                    ["decode", "prefill"],
                )

        asyncio.run(run())

    def test_pd_worker_exports_mooncake_metadata(self):
        """验证 prefill worker 会把 Mooncake 元数据和 API surface 写入 registry。"""
        with TemporaryDirectory() as temporary:
            # 绕过完整初始化，只构造 discovery metadata 生成所需的字段。
            server = ClusterInferenceServer.__new__(ClusterInferenceServer)
            server.cluster_options = WorkerOptions(
                namespace="pd-test",
                file_kv=Path(temporary),
                served_model_name="qwen",
                advertise_host="127.0.0.1",
                worker_type="prefill",
                mooncake_bootstrap_port=18528,
            )
            server.registry = FileWorkerRegistry(Path(temporary))
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

            # FakeEngine 避免加载真实模型，仅保证健康检查可用。
            discovery = server._create_discovery_service(FakeEngine())
            metadata = discovery.record.metadata["pd"]

            # frontend 依赖这些字段选择 bootstrap 地址并识别 KV 角色。
            self.assertEqual(metadata["engine_id"], "prefill-1")
            self.assertEqual(metadata["kv_role"], "kv_producer")
            self.assertEqual(metadata["bootstrap_port"], 18528)
            self.assertEqual(
                discovery.record.metadata["supported_api_routes"],
                ["/v1/chat/completions", "/chat/completions"],
            )
            self.assertEqual(discovery.record.metadata["needs"], ["decode"])
            # PD worker 不导出聚合式 KV-aware 的本地 prefix index。
            self.assertIsNone(discovery.cache_index_exporter)

    def test_pd_frontend_model_and_load_selection(self):
        """验证 PD 模型 ready 判断和 least-loaded 对重负载 worker 的避让。"""
        with TemporaryDirectory() as temporary:
            registry = FileWorkerRegistry(Path(temporary))
            # 构造同一模型的 P/D 记录，并额外构造一个高负载 decode。
            prefill = make_record(registry, "prefill")
            decode = make_record(registry, "decode")
            loaded = replace_load(decode)
            frontend = DisaggregatedFrontend(
                DisaggregatedFrontendOptions(
                    namespace="pd-test",
                    discovery_root=Path(temporary),
                    router_mode="least-loaded",
                    http_host="127.0.0.1",
                    http_port=18520,
                )
            )
            # 直接注入 watcher，避免测试依赖后台 discovery 轮询。
            frontend.watcher = SimpleNamespace(worker_records=[prefill, loaded])

            # 同一模型同时存在 P/D 时才 ready。
            self.assertEqual(frontend._candidate_model_names(), ["qwen"])
            # least-loaded 应避开带负载快照的 decode。
            self.assertEqual(
                frontend.router.select_record(
                    WorkerType.DECODE,
                    [decode, loaded],
                ).instance_id,
                decode.instance_id,
            )

    def test_worker_chat_cancels_internal_request_on_client_disconnect(self):
        async def run():
            disconnected = asyncio.Event()
            internal_canceled = asyncio.Event()

            class DisconnectingRequest:
                async def is_disconnected(self) -> bool:
                    # Event 让测试精确控制“客户端何时断开”。
                    return disconnected.is_set()

            async def slow_inference_chat(
                self,
                request_id,
                data,
                http_request,
            ):
                # 模拟 InferenceServer 内部请求正在等待 engine 返回。
                # 若取消传播正确，sleep 会被 CancelledError 打断。
                    try:
                        await asyncio.sleep(10)
                    except asyncio.CancelledError:
                        internal_canceled.set()
                        raise

            server = ClusterInferenceServer.__new__(ClusterInferenceServer)
            server.worker_traffic = WorkerTrafficMetrics()
            with patch.object(InferenceServer, "_chat", slow_inference_chat):
                # 外层 ClusterInferenceServer._chat() 会每 100ms 检查
                # DisconnectingRequest；这里不需要真实 HTTP 连接。
                request_task = asyncio.create_task(
                    server._chat("test-request", {}, DisconnectingRequest())
                )
                await asyncio.sleep(0.02)
                # 触发客户端断开，内部推理任务应随之取消。
                disconnected.set()

                with self.assertRaises(asyncio.CancelledError):
                    await request_task

            self.assertTrue(internal_canceled.is_set())
            snapshot = server.worker_traffic.snapshot()
            self.assertEqual(snapshot["canceled_requests_total"], 1)
            self.assertEqual(snapshot["active_requests"], 0)

        asyncio.run(run())

    def test_concurrent_client_reconcile_initializes_worker_once(self):
        async def run():
            with TemporaryDirectory() as temporary:
                frontend = DisaggregatedFrontend(
                    DisaggregatedFrontendOptions(
                        namespace="pd-test",
                        discovery_root=Path(temporary),
                        router_mode="round-robin",
                        http_host="127.0.0.1",
                        http_port=18520,
                    )
                )
                decode = make_record(frontend.registry, "decode")
                frontend.watcher = SimpleNamespace(worker_records=[decode])
                created = 0

                async def create_client_info(record):
                    nonlocal created
                    created += 1
                    await asyncio.sleep(0)
                    return {
                        "http_client": SimpleNamespace(),
                        "record": record,
                        "instance_id": record.instance_id,
                        "routable": True,
                    }

                with patch.object(
                    frontend.worker_client_pool_manager,
                    "_create_client_info",
                    side_effect=create_client_info,
                ):
                    await asyncio.gather(
                        frontend.worker_client_pool_manager.reconcile_records([decode]),
                        frontend.worker_client_pool_manager.reconcile_records([decode]),
                    )
                    await frontend.worker_client_pool_manager.wait_for_creates()

                self.assertEqual(created, 1)
                self.assertEqual(
                    frontend.worker_client_pool_manager.client_ids(
                        "qwen", WorkerType.DECODE
                    ),
                    {decode.instance_id},
                )

        asyncio.run(run())

    def test_completion_request_selects_both_roles_without_bootstrap(self):
        async def run():
            with TemporaryDirectory() as temporary:
                frontend = DisaggregatedFrontend(
                    DisaggregatedFrontendOptions(
                        namespace="pd-test",
                        discovery_root=Path(temporary),
                        router_mode="round-robin",
                        http_host="127.0.0.1",
                        http_port=18520,
                    )
                )
                prefill = make_record(frontend.registry, "prefill")
                decode = make_record(frontend.registry, "decode")
                frontend.watcher = SimpleNamespace(worker_records=[prefill, decode])
                prefill_info = {
                    "instance_id": prefill.instance_id,
                    "record": prefill,
                }
                decode_info = {
                    "instance_id": decode.instance_id,
                    "record": decode,
                }

                async def select_clients(model_name, api_route):
                    self.assertEqual(model_name, "qwen")
                    return prefill_info, decode_info

                async def stream_decode(*args, **kwargs):
                    yield b'{"ok":true}'

                with (
                    patch.object(
                        frontend,
                        "_select_worker_clients",
                        side_effect=select_clients,
                    ) as select_clients,
                    patch.object(frontend, "_send_prefill_background", new=AsyncMock()),
                    patch.object(
                        frontend,
                        "_stream_decode",
                        side_effect=stream_decode,
                    ),
                ):
                    client = TestClient(frontend.create_app())
                    response = client.post(
                        "/v1/chat/completions",
                        json={"model": "qwen", "messages": []},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"ok": True})
                select_clients.assert_awaited_once_with("qwen", "/v1/chat/completions")

        asyncio.run(run())

    def test_streaming_response_uses_sse_media_type(self):
        async def run():
            with TemporaryDirectory() as temporary:
                frontend = DisaggregatedFrontend(
                    DisaggregatedFrontendOptions(
                        namespace="pd-test",
                        discovery_root=Path(temporary),
                        router_mode="round-robin",
                        http_host="127.0.0.1",
                        http_port=18520,
                    )
                )
                prefill = make_record(frontend.registry, "prefill")
                decode = make_record(frontend.registry, "decode")
                frontend.watcher = SimpleNamespace(
                    worker_records=[prefill, decode]
                )
                prefill_info = {"instance_id": "prefill", "record": object()}
                decode_info = {"instance_id": "decode", "record": object()}

                async def select_clients(model_name, api_route):
                    return prefill_info, decode_info

                async def stream_decode(*args, **kwargs):
                    # 返回标准 SSE bytes，让测试只关注 frontend 的
                    # Content-Type 处理，而不依赖真实 decode worker。
                    yield b'data: {"ok":true}\n\n'

                with (
                    patch.object(
                        frontend,
                        "_select_worker_clients",
                        side_effect=select_clients,
                    ),
                    patch.object(frontend, "_send_prefill_background", new=AsyncMock()),
                    patch.object(
                        frontend,
                        "_stream_decode",
                        side_effect=stream_decode,
                    ),
                ):
                    client = TestClient(frontend.create_app())
                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "qwen",
                            "messages": [],
                            "stream": True,
                        },
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.headers["content-type"].split(";", 1)[0],
                    "text/event-stream",
                )
                self.assertIn(b'"ok":true', response.content)

        asyncio.run(run())

    def test_initial_pd_error_returns_real_http_status(self):
        """首个正常 chunk 前，backend 错误不能伪装成 HTTP 200。"""

        async def run():
            with TemporaryDirectory() as temporary:
                frontend = DisaggregatedFrontend(
                    DisaggregatedFrontendOptions(
                        namespace="pd-test",
                        discovery_root=Path(temporary),
                        router_mode="round-robin",
                        http_host="127.0.0.1",
                        http_port=18520,
                    )
                )
                prefill = make_record(frontend.registry, "prefill")
                decode = make_record(frontend.registry, "decode")
                frontend.watcher = SimpleNamespace(
                    worker_records=[prefill, decode]
                )
                decode_closed = asyncio.Event()

                async def select_clients(model_name, api_route):
                    return (
                        {"instance_id": prefill.instance_id, "record": prefill},
                        {"instance_id": decode.instance_id, "record": decode},
                    )

                async def failing_prefill(*args, **kwargs):
                    raise HTTPException(
                        status_code=429,
                        detail="prefill overloaded",
                    )

                async def decode_stream(*args, **kwargs):
                    try:
                        yield b'{"ok":true}'
                    finally:
                        decode_closed.set()

                with (
                    patch.object(
                        frontend,
                        "_select_worker_clients",
                        side_effect=select_clients,
                    ),
                    patch.object(
                        frontend,
                        "_send_prefill_background",
                        side_effect=failing_prefill,
                    ),
                    patch.object(
                        frontend,
                        "_stream_decode",
                        side_effect=decode_stream,
                    ),
                ):
                    client = TestClient(frontend.create_app())
                    response = client.post(
                        "/v1/chat/completions",
                        json={"model": "qwen", "messages": [], "stream": True},
                    )

                self.assertEqual(response.status_code, 429)
                self.assertEqual(
                    response.json(),
                    {"detail": "prefill overloaded"},
                )
                self.assertNotIn(b'"ok":true', response.content)
                self.assertTrue(decode_closed.is_set())

        asyncio.run(run())

    def test_prefill_failure_is_propagated_and_cancels_decode(self):
        async def run():
            with TemporaryDirectory() as temporary:
                frontend = DisaggregatedFrontend(
                    DisaggregatedFrontendOptions(
                        namespace="pd-test",
                        discovery_root=Path(temporary),
                        router_mode="round-robin",
                        http_host="127.0.0.1",
                        http_port=18520,
                    )
                )
                decode_closed = asyncio.Event()

                async def failing_prefill():
                    # 模拟 prefill worker 返回明确的 HTTP 错误。
                    # 该异常必须穿透 P/D 编排层，不能被后台任务吞掉。
                    raise HTTPException(
                        status_code=502,
                        detail="injected prefill failure",
                    )

                async def successful_decode():
                    # decode 先产生一个“成功 chunk”再长时间挂起。
                    # 若 prefill 失败，该 chunk 不允许发送给客户端，
                    # 且 decode generator 必须被 aclose()。
                    try:
                        yield b'{"ok":true}'
                        await asyncio.sleep(10)
                    finally:
                        decode_closed.set()

                prefill_task = asyncio.create_task(failing_prefill())
                decode_stream = successful_decode()
                with self.assertRaises(HTTPException) as raised:
                    async for _chunk in frontend._generate_pd_stream(
                        prefill_task,
                        decode_stream,
                        "test-request",
                        streaming=True,
                    ):
                        pass

                self.assertEqual(raised.exception.status_code, 502)
                self.assertEqual(
                    raised.exception.detail,
                    "injected prefill failure",
                )
                self.assertTrue(decode_closed.is_set())

        asyncio.run(run())

    def test_prefetch_closing_response_still_closes_pd_stream(self):
        """StreamingResponse 未继续迭代时也必须触发内层资源清理。"""

        async def run():
            cleanup_called = asyncio.Event()

            async def source():
                try:
                    yield b'{"ok":true}'
                    await asyncio.sleep(10)
                finally:
                    cleanup_called.set()

            from infinilm.cluster.disaggregated.frontend import (
                _CancellationSafeAsyncIterator,
            )
            inner = _CancellationSafeAsyncIterator(source())
            first_chunk = await inner.__anext__()
            iterator = _PrefetchedAsyncIterator(first_chunk, inner)

            self.assertEqual(await iterator.__anext__(), first_chunk)
            await iterator.aclose()

            self.assertTrue(cleanup_called.is_set())

        asyncio.run(run())

    def test_client_disconnect_cancels_prefill_and_decode(self):
        async def run():
            with TemporaryDirectory() as temporary:
                frontend = DisaggregatedFrontend(
                    DisaggregatedFrontendOptions(
                        namespace="pd-test",
                        discovery_root=Path(temporary),
                        router_mode="round-robin",
                        http_host="127.0.0.1",
                        http_port=18520,
                    )
                )
                prefill_started = asyncio.Event()
                decode_started = asyncio.Event()
                prefill_canceled = asyncio.Event()
                decode_canceled = asyncio.Event()

                async def slow_prefill():
                    # 模拟 prefill 正在远端计算，客户端断开时应被取消。
                    prefill_started.set()
                    try:
                        await asyncio.sleep(10)
                    except asyncio.CancelledError:
                        prefill_canceled.set()
                        raise

                async def slow_decode():
                    # decode 首 chunk 可先就绪，但 prefill 成功前不得输出；
                    # 后续长时间挂起用于模拟仍在等待远端 KV/生成。
                    decode_started.set()
                    try:
                        yield b'{"ok":true}'
                        await asyncio.sleep(10)
                    except (asyncio.CancelledError, GeneratorExit):
                        decode_canceled.set()
                        raise

                prefill_task = asyncio.create_task(slow_prefill())
                decode_stream = slow_decode()
                generator = frontend._generate_pd_stream(
                    prefill_task,
                    decode_stream,
                    "test-request",
                )
                # reader 模拟 StreamingResponse 外层迭代任务；取消它
                # 等价于 Starlette 在客户端断开时取消响应迭代。
                reader = asyncio.create_task(anext(generator))

                await asyncio.wait_for(
                    asyncio.gather(prefill_started.wait(), decode_started.wait()),
                    timeout=1,
                )
                # 在 prefill 未成功、任何 decode chunk 尚未暴露给客户端前断开。
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                await generator.aclose()

                self.assertTrue(prefill_canceled.is_set())
                self.assertTrue(decode_canceled.is_set())

        asyncio.run(run())

    def test_select_worker_client_uses_manager_routable_records(self):
        """请求选择只使用 manager 中已初始化且可路由的 worker。"""

        async def run():
            with TemporaryDirectory() as temporary:
                frontend = DisaggregatedFrontend(
                    DisaggregatedFrontendOptions(
                        namespace="pd-test",
                        discovery_root=Path(temporary),
                        router_mode="round-robin",
                        http_host="127.0.0.1",
                        http_port=18520,
                    )
                )
                initialized_prefill = make_record(frontend.registry, "prefill")
                initialized = make_record(frontend.registry, "decode")
                registry_only = make_record(frontend.registry, "decode")
                frontend.watcher = SimpleNamespace(
                    worker_records=[initialized, registry_only]
                )

                async def create_client_info(record):
                    if record is registry_only:
                        raise RuntimeError("client initialization failed")
                    return {
                        "http_client": SimpleNamespace(),
                        "record": record,
                        "instance_id": record.instance_id,
                        "routable": True,
                    }

                with patch.object(
                    frontend.worker_client_pool_manager,
                    "_create_client_info",
                    side_effect=create_client_info,
                ):
                    await frontend.worker_client_pool_manager.reconcile_records(
                        [initialized_prefill, initialized]
                    )
                    await frontend.worker_client_pool_manager.wait_for_creates()

                _, info = await frontend._select_worker_clients(
                    "qwen",
                    "/v1/chat/completions",
                )
                self.assertIs(info["record"], initialized)

                # registry 可见但 client 未初始化时，不能进入请求候选集。
                frontend.worker_client_pool_manager._pools["qwen"][WorkerType.DECODE][
                    initialized.instance_id
                ]["routable"] = False
                with self.assertRaises(HTTPException):
                    await frontend._select_worker_clients(
                        "qwen",
                        "/v1/chat/completions",
                    )

        asyncio.run(run())

    def test_cli_parses_worker_and_frontend_roles(self):
        """验证 PD worker/frontend 角色参数解析，并拒绝冲突的 worker 角色。"""
        # 权威角色参数会同时推导 worker_type 和 registry component。
        mode = parse_worker_cluster_args(
            [
                "--disaggregation-mode",
                "prefill",
                "--mooncake-bootstrap-port",
                "18528",
            ]
        )
        self.assertEqual(mode.worker_type, WorkerType.PREFILL)
        self.assertEqual(mode.component, "prefill")

        # 兼容旧 worker-type 入口，但推导结果必须一致。
        worker = parse_worker_cluster_args(
            [
                "--worker-type",
                "prefill",
                "--mooncake-bootstrap-port",
                "18528",
            ]
        )
        self.assertEqual(worker.worker_type, WorkerType.PREFILL)
        self.assertEqual(worker.mooncake_bootstrap_port, 18528)

        # 两个角色参数同时出现且冲突时必须快速失败。
        with self.assertRaises(ValueError):
            parse_worker_cluster_args(
                [
                    "--disaggregation-mode",
                    "decode",
                    "--worker-type",
                    "prefill",
                ]
            )

        frontend = parse_disaggregated_frontend_args(
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
        """验证 prefill 和 decode 各自维护独立 round-robin 游标。"""
        with TemporaryDirectory() as temporary:
            registry = FileWorkerRegistry(Path(temporary))
            # 每个角色准备两个 worker，用于观察两次选择是否轮换。
            workers = [
                make_record(registry, "prefill"),
                make_record(registry, "prefill"),
                make_record(registry, "decode"),
                make_record(registry, "decode"),
            ]
            frontend = DisaggregatedFrontend(
                DisaggregatedFrontendOptions(
                    namespace="pd-test",
                    discovery_root=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                )
            )
            frontend.watcher = SimpleNamespace(worker_records=workers)

            # 分别在 P/D 候选集内连续选择两次。
            prefill_ids = [
                frontend.router.select_record(
                    WorkerType.PREFILL,
                    frontend._filter_worker_records(workers, WorkerType.PREFILL),
                ).instance_id
                for _ in range(2)
            ]
            decode_ids = [
                frontend.router.select_record(
                    WorkerType.DECODE,
                    frontend._filter_worker_records(workers, WorkerType.DECODE),
                ).instance_id
                for _ in range(2)
            ]

            # 每个角色内部的两次选择都应不同，说明游标按角色独立推进。
            self.assertNotEqual(prefill_ids[0], prefill_ids[1])
            self.assertNotEqual(decode_ids[0], decode_ids[1])

    def test_pd_monitoring_routes_expose_role_workers(self):
        """验证 PD 监控页面和 API 会按 prefill/decode 角色展示 worker。"""
        with TemporaryDirectory() as temporary:
            registry = FileWorkerRegistry(Path(temporary))
            # monitoring 需要 PD bootstrap metadata 来展示 engine 信息。
            prefill = replace_pd(make_record(registry, "prefill"))
            decode = replace_pd(make_record(registry, "decode"))
            frontend = DisaggregatedFrontend(
                DisaggregatedFrontendOptions(
                    namespace="pd-test",
                    discovery_root=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                )
            )
            frontend.watcher = SimpleNamespace(worker_records=[prefill, decode])
            client = TestClient(frontend.create_app())

            # 同时验证 HTML 页面和页面消费的 JSON 快照。
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
        """验证 disaggregation mode 能推导 KV 角色，并拒绝冲突配置。"""
        # 未显式配置 kv_role 时，由 prefill 角色推导 producer。
        config = _resolve_kv_transfer_config(
            '{"kv_connector":"MooncakeConnector","engine_id":"p"}',
            WorkerType.PREFILL,
        )
        self.assertEqual(config.kv_role, "kv_producer")

        # 显式 KV 角色与 worker 角色冲突时必须拒绝启动。
        with self.assertRaises(ValueError):
            _resolve_kv_transfer_config(
                '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}',
                WorkerType.PREFILL,
            )

    def test_prefill_openai_surface_requires_internal_header(self):
        """验证 producer-only prefill OpenAI 接口拒绝外部直连。"""
        with TemporaryDirectory() as temporary:
            # 构造最小的 prefill server，避免加载模型和启动真实引擎。
            server = ClusterInferenceServer.__new__(ClusterInferenceServer)
            server.cluster_options = WorkerOptions(
                namespace="pd-test",
                file_kv=Path(temporary),
                served_model_name="qwen",
                advertise_host="127.0.0.1",
                worker_type="prefill",
            )
            server.kv_transfer_config = KVTransferConfig(
                kv_connector="MooncakeConnector",
                kv_role="kv_producer",
                engine_id="prefill-guard",
            )
            # 不携带 X-InfiniLM-Internal，模拟客户端绕过 PD frontend 直连。
            response = TestClient(server._create_app()).post(
                "/v1/chat/completions",
                json={"model": "qwen", "messages": []},
            )

            self.assertEqual(response.status_code, 403)
            self.assertIn("only accepts internal requests", response.text)

    def test_frontend_starts_without_workers_and_reports_not_ready(self):
        """验证空集群时 PD frontend 保持 healthy 但 ready=false。"""
        with TemporaryDirectory() as temporary:
            # 空目录代表 discovery 中没有 P/D worker。
            frontend = DisaggregatedFrontend(
                DisaggregatedFrontendOptions(
                    namespace="pd-test",
                    discovery_root=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                )
            )

            # TestClient 会执行 lifespan，验证服务启动后的真实 HTTP 行为。
            with TestClient(frontend.create_app()) as client:
                health = client.get("/health").json()
                request = client.post(
                    "/v1/chat/completions",
                    json={"messages": []},
                )

            self.assertEqual(health["status"], "healthy")
            # frontend 进程存活不等于已有可服务模型。
            self.assertFalse(health["ready"])
            self.assertEqual(request.status_code, 503)

    def test_discovery_launch_script_waits_for_frontend_ready_flag(self):
        """启动脚本不能把 /health 200 误当成 PD 模型已经 ready。"""
        script = (
            Path(__file__).resolve().parents[2]
            / "python/infinilm/cluster/disaggregated/run_pd_discovery_cluster.sh"
        )
        source = script.read_text(encoding="utf-8")

        self.assertIn('wait_for_pd_ready "PD frontend"', source)
        self.assertIn('.get("ready") is True', source)
        self.assertNotIn('wait_for_url "PD frontend"', source)

    def test_frontend_filters_workers_without_required_api_route(self):
        """验证缺少必需 API route 的 worker 不会进入 PD 路由候选。"""
        with TemporaryDirectory() as temporary:
            registry = FileWorkerRegistry(Path(temporary))
            # 一个 worker 带完整 API route 声明，另一个模拟缺少 metadata 的旧记录。
            capable = make_record(registry, "decode")
            legacy = replace_metadata(make_record(registry, "decode"), {})
            frontend = DisaggregatedFrontend(
                DisaggregatedFrontendOptions(
                    namespace="pd-test",
                    discovery_root=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                )
            )

            # 显式要求请求路径，避免把不能服务该 API 的记录选入。
            workers = frontend._filter_worker_records(
                [capable, legacy],
                WorkerType.DECODE,
                "qwen",
                "/v1/chat/completions",
            )

            self.assertEqual(
                [worker.instance_id for worker in workers], [capable.instance_id]
            )

    def test_frontend_does_not_route_unadvertised_completions_api(self):
        """验证当前不暴露未实现的 legacy completions API。"""
        with TemporaryDirectory() as temporary:
            registry = FileWorkerRegistry(Path(temporary))
            decode = make_record(registry, "decode")
            frontend = DisaggregatedFrontend(
                DisaggregatedFrontendOptions(
                    namespace="pd-test",
                    discovery_root=Path(temporary),
                    router_mode="round-robin",
                    http_host="127.0.0.1",
                    http_port=18520,
                )
            )

            # 当前 worker 只实现 chat completions，不能广告 legacy completions。
            self.assertEqual(
                frontend._filter_worker_records(
                    [decode],
                    WorkerType.DECODE,
                    "qwen",
                    "/v1/completions",
                ),
                [],
            )

            with TestClient(frontend.create_app()) as client:
                response = client.post(
                    "/v1/completions",
                    json={"model": "qwen", "prompt": "hello"},
                )
            self.assertEqual(response.status_code, 404)


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
                "engine_id": f"{record.worker_type.value}-engine",
                "kv_role": record.worker_type.value,
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
"""
cd /workspace_codex/InfiniLM
PYTHONPATH=python python -m pytest -q \
  test/service/test_pd_cluster.py
"""
