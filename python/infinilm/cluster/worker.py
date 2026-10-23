"""Cluster-aware entrypoint for one aggregated InfiniLM inference worker."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from infinilm.base_config import BaseConfig
from infinilm.config import KVTransferConfig
from infinilm.moe_config import configure_moe_ep_backend
from infinilm.server.inference_server import (
    InferenceServer,
    parse_kv_transfer_config,
    setup_logging,
)

from .cache_index import WorkerCacheIndexExporter
from .config import WorkerOptions, parse_worker_cluster_args, remove_cluster_worker_args
from .environment import drain_grace_seconds, heartbeat_interval_seconds
from .kv_prefix import model_cache_namespace
from .worker_discovery import WorkerDiscoveryService
from .worker_registry import WorkerRegistry, create_worker_registry
from .worker_registry.types import WorkerType, coerce_worker_type
from .worker_traffic import WorkerTrafficMetrics

logger = logging.getLogger(__name__)

WORKER_SUPPORTED_API_ROUTES = {
    WorkerType.AGGREGATED: ("/v1/chat/completions", "/chat/completions"),
    WorkerType.PREFILL: ("/v1/chat/completions", "/chat/completions"),
    WorkerType.DECODE: ("/v1/chat/completions", "/chat/completions"),
}
WORKER_NEEDS = {
    WorkerType.AGGREGATED: (),
    WorkerType.PREFILL: ("decode",),
    WorkerType.DECODE: ("prefill",),
}
PD_KV_ROLES = {
    WorkerType.PREFILL: "kv_producer",
    WorkerType.DECODE: "kv_consumer",
}


class DrainingUvicornServer(uvicorn.Server):
    """收到退出信号后先广播 DRAINING，再优雅停止 Uvicorn。

    完整调用链：
    SIGTERM/SIGINT -> handle_exit() -> _drain_and_exit()
    -> on_drain_start() -> ClusterInferenceServer.start_draining()
    -> WorkerDiscoveryService.start_draining()
    -> registry.heartbeat() 写入 state=draining
    -> frontend watcher 下一次 snapshot 后停止路由该 worker
    -> 等待 drain_grace_seconds -> should_exit=True
    -> FastAPI lifespan cleanup -> WorkerDiscoveryService.stop()
    -> registry.unregister() 删除注册记录
    """

    def __init__(
        self,
        config: uvicorn.Config,
        *,
        drain_grace_seconds: float,
        on_ready=None,
    ) -> None:
        super().__init__(config)
        self.drain_grace_seconds = max(0.0, drain_grace_seconds)
        # 该回调由 ClusterWorkerRuntime 绑定为推理服务的 start_draining 方法。
        self.on_drain_start = None
        # Uvicorn lifespan startup 先于 socket bind；此回调在两者都完成后执行。
        self.on_ready = on_ready
        # 区分第一次优雅退出和第二次运维强制退出。
        self._exit_requested = False

    async def startup(self, sockets=None) -> None:
        """socket bind 完成后触发 worker discovery 注册回调。"""
        await super().startup(sockets=sockets)
        if not self.started or self.should_exit or self.on_ready is None:
            return

        try:
            await self.on_ready(self)
        except Exception:
            # 让 Uvicorn _serve() 统一进入 shutdown，避免 engine lifespan、
            # listener socket 或 registry 资源因 startup 异常而泄漏。
            logger.exception("Worker ready callback failed; stopping server")
            self.should_exit = True

    def handle_exit(self, sig, frame) -> None:
        # 第二次信号跳过 drain 宽限期，交给 Uvicorn 原生逻辑立即退出。
        if self._exit_requested:
            super().handle_exit(sig, frame)
            return

        self._exit_requested = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有 running loop 时无法调度异步 drain，只能直接退出。
            self.should_exit = True
            return
        loop.create_task(self._drain_and_exit(), name="infinilm-worker-drain")

    async def _drain_and_exit(self) -> None:
        try:
            # 先把 registry 记录改成 draining，让 frontend 尽快停止新请求路由。
            if self.on_drain_start is not None:
                await self.on_drain_start()
            # 宽限期内存量请求可以继续完成。
            if self.drain_grace_seconds:
                await asyncio.sleep(self.drain_grace_seconds)
        except Exception:
            logger.exception("Worker draining failed; proceeding with shutdown")
        finally:
            self.should_exit = True


class ClusterInferenceServer(InferenceServer):
    """Existing InferenceServer plus discoverable cluster request metadata."""

    def __init__(
        self,
        *,
        cluster_options: WorkerOptions,
        registry: WorkerRegistry,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cluster_options = cluster_options
        self.registry = registry
        self.worker_discovery: WorkerDiscoveryService | None = None
        self.worker_traffic = WorkerTrafficMetrics()
        if cluster_options.served_model_name:
            self.model_id = cluster_options.served_model_name

    async def start_draining(self) -> None:
        """把当前 worker 的 discovery 状态切换为 draining。"""
        if self.worker_discovery is None:
            logger.warning("Ignoring worker drain request before discovery is ready")
            return
        await self.worker_discovery.start_draining()

    async def start_discovery(self, *, http_port: int) -> None:
        """HTTP socket 绑定完成后注册 discovery。

        FastAPI/Uvicorn lifespan startup 发生在 socket bind 之前，因此
        ``port=0`` 分配出的真实端口必须由 runtime 回调后传入这里。
        """
        if self.worker_discovery is not None:
            return
        discovery = self._create_discovery_service(
            self.engine,
            http_port=http_port,
        )
        self.worker_discovery = discovery
        try:
            await discovery.start()
        except Exception:
            self.worker_discovery = None
            raise

    @staticmethod
    async def _client_disconnected(http_request) -> bool:
        """读取 HTTP 客户端是否断开，兼容异步和同步两种测试 fake。

        Starlette Request.is_disconnected() 返回 coroutine；部分单元测试
        注入的是同步 callable。两种形态都统一转换成 bool，避免包装层
        依赖具体测试对象类型。
        """
        is_disconnected = getattr(http_request, "is_disconnected", None)
        if not callable(is_disconnected):
            # 没有 disconnect API 时视为仍连接，交由正常超时/完成路径处理。
            return False
        result = is_disconnected()
        if inspect.isawaitable(result):
            # 异步实现必须 await 后才能得到真实连接状态。
            result = await result
        return bool(result)

    async def _wait_for_request(
        self,
        request_task: asyncio.Task,
        http_request,
    ):
        """等待内部 InferenceServer 任务完成，并把断开转换为任务取消。

        ``request_task`` 包住真正的 ``_chat()`` 或 ``__anext__()``。
        使用任务而不是直接 await，是为了每 100ms 有机会检查客户端是否
        断开；一旦断开，就取消内部任务，让 InferenceServer 的 finally
        调用 add_aborted_req() 终止 engine 请求。
        """
        try:
            while not request_task.done():
                # 0.1 秒是断开检测频率，不是请求超时；任务可以持续运行很久。
                done, _pending = await asyncio.wait(
                    {request_task},
                    timeout=0.1,
                )
                if done:
                    break
                if await self._client_disconnected(http_request):
                    # 对外统一抛 CancelledError；下方 except 会统一取消
                    # 内部任务并传播取消信号。
                    raise asyncio.CancelledError
        except asyncio.CancelledError:
            # 这里有两个取消来源：
            # 1. worker 轮询发现 HTTP 客户端断开；
            # 2. Uvicorn/Starlette 直接取消外层 streaming task。
            # 无论来源如何，都不能把 request_task 遗留在 pending 状态；
            # 否则流式路径稍后调用 internal_stream.aclose() 时，内部
            # generator 仍在一个 __anext__() task 中运行，会触发
            # "aclose(): asynchronous generator is already running"。
            await self._cancel_request_task(request_task)
            raise
        # 任务完成或抛错都在这里恢复结果/异常。
        return request_task.result()

    @staticmethod
    async def _cancel_request_task(request_task: asyncio.Task) -> None:
        """取消并等待内部请求任务结束，屏蔽它的取消型异常。

        ``return_exceptions=True`` 会同时收集 ``CancelledError``、
        ``StopAsyncIteration`` 和普通异常，避免内部任务的退出异常覆盖
        外层正在传播的取消信号。调用方仍必须重新抛出原始
        ``CancelledError``，因此这里不能吞掉外层取消本身。
        """
        if not request_task.done():
            request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)

    async def _chat(self, request_id: str, data: dict, http_request):
        """包装非流式 InferenceServer 请求，统计流量并传播客户端断开。"""
        self.worker_traffic.begin(streaming=False)
        canceled = False
        failed = False
        try:
            # 内部请求放入独立 task，由 _wait_for_request() 周期检查断开。
            response = await self._wait_for_request(
                asyncio.create_task(super()._chat(request_id, data, http_request)),
                http_request,
            )
            failed = int(getattr(response, "status_code", 200)) >= 400
            if isinstance(response, dict):
                self.worker_traffic.record_cache(response.get("cache"))
            return response
        except asyncio.CancelledError:
            canceled = True
            raise
        except Exception:
            failed = True
            raise
        finally:
            # begin/finish 一一对应，保证 active_requests 不会因断开泄漏。
            self.worker_traffic.finish(canceled=canceled, failed=failed)

    async def _stream_chat(self, request_id: str, data: dict, http_request):
        """包装流式 InferenceServer generator，统计流量并传播取消。"""
        self.worker_traffic.begin(streaming=True)
        canceled = False
        failed = False
        try:
            # internal_stream 是原有 InferenceServer._stream_chat()；
            # 这里逐 chunk 包装，才能在等待下一个 token 时检查断开。
            internal_stream = super()._stream_chat(request_id, data, http_request)
            while True:
                try:
                    # 每个 chunk 都通过独立 task 等待，等待期间可检测断开。
                    chunk = await self._wait_for_request(
                        asyncio.create_task(internal_stream.__anext__()),
                        http_request,
                    )
                except StopAsyncIteration:
                    # 正常结束，进入 traffic finish 统计。
                    break
                except asyncio.CancelledError:
                    # 客户端断开时显式关闭内层 generator，触发原有
                    # engine abort 和资源释放逻辑。
                    await internal_stream.aclose()
                    raise
                if "finish_reason" in chunk and (
                    '"error"' in chunk or '"timeout"' in chunk
                ):
                    failed = True
                if "finish_reason" in chunk:
                    try:
                        message = json.loads(chunk.removeprefix("data: ").strip())
                        self.worker_traffic.record_cache(message.get("cache"))
                    except ValueError:
                        pass
                yield chunk
        except asyncio.CancelledError:
            canceled = True
            raise
        except GeneratorExit:
            canceled = True
            raise
        except Exception:
            failed = True
            raise
        finally:
            # 流式请求的 active 状态必须覆盖整个响应 generator 生命周期。
            self.worker_traffic.finish(canceled=canceled, failed=failed)

    def _create_discovery_service(
        self,
        engine: object,
        *,
        http_port: int | None = None,
    ) -> WorkerDiscoveryService:
        """在 engine ready 后构造 worker 的 registry discovery 服务。"""
        actual_http_port = self.port if http_port is None else int(http_port)
        if actual_http_port <= 0:
            raise ValueError(
                "worker HTTP port must be resolved before discovery registration"
            )
        served_model_name = (
            self.cluster_options.served_model_name
            or os.path.basename(os.path.normpath(self.model_path))
            or "model"
        )
        kv_config = self.kv_transfer_config
        bootstrap_port = self.cluster_options.mooncake_bootstrap_port
        if (
            self.cluster_options.worker_type is WorkerType.PREFILL
            and bootstrap_port == 0
        ):
            bootstrap_port = self._actual_bootstrap_port(engine)
        metadata = {
            "http_host": self.host,
            "http_port": actual_http_port,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "tp": self.tensor_parallel_size,
            "max_batch_size": self.max_batch_size,
            "supported_api_routes": list(
                WORKER_SUPPORTED_API_ROUTES[self.cluster_options.worker_type]
            ),
            "needs": list(WORKER_NEEDS[self.cluster_options.worker_type]),
        }
        if kv_config is not None:
            metadata["pd"] = {
                "engine_id": kv_config.engine_id,
                "kv_role": kv_config.kv_role,
                "bootstrap_host": self.cluster_options.mooncake_bootstrap_host,
                "bootstrap_port": bootstrap_port,
            }
        record = self.registry.new_record(
            namespace=self.cluster_options.namespace,
            component=self.cluster_options.component,
            endpoint=self.cluster_options.endpoint,
            worker_type=self.cluster_options.worker_type,
            model=self.model_path,
            served_model_name=served_model_name,
            host=self.cluster_options.advertise_host,
            port=actual_http_port,
            engine_healthy=True,
            metadata=metadata,
        )
        cache_exporter = self._create_cache_index_exporter(engine, record.instance_id)
        return WorkerDiscoveryService(
            engine=engine,
            registry=self.registry,
            record=record,
            heartbeat_interval=heartbeat_interval_seconds(),
            traffic_metrics=self.worker_traffic,
            cache_index_exporter=cache_exporter,
        )

    def _actual_bootstrap_port(self, engine: object) -> int:
        """读取 prefill Mooncake bootstrap server 绑定后的真实端口。"""
        connector = getattr(
            getattr(getattr(engine, "engine", None), "model_runner", None),
            "kv_connector",
            None,
        )
        port = getattr(connector, "bootstrap_port", None)
        if port is None or int(port) <= 0:
            raise RuntimeError(
                "Mooncake bootstrap port=0 requires a bound bootstrap server"
            )
        return int(port)

    def _create_cache_index_exporter(
        self,
        engine: object,
        instance_id: str,
    ) -> WorkerCacheIndexExporter | None:
        """为 KV-aware 路由创建 prefix cache 索引导出器。

        使用 getattr 按鸭子类型读取 engine 和 cluster_options：真实
        AsyncLLMEngine 与测试中的 SimpleNamespace 都可以传入。结构不完整、
        prefix cache 未开启或 etcd 缺少共享目录时返回 None；索引只服务路由
        优化，不能影响推理服务启动。
        """
        worker_type = coerce_worker_type(
            getattr(
                getattr(self, "cluster_options", None),
                "worker_type",
                WorkerType.AGGREGATED,
            )
        )
        cluster_options = getattr(self, "cluster_options", None)
        # 只有完整聚合式 worker 拥有可复用的本地 prompt prefix cache。
        if worker_type is not WorkerType.AGGREGATED:
            return None
        # 从真实 engine 结构中读取 BlockManager 和 block size。
        scheduler = getattr(getattr(engine, "engine", None), "scheduler", None)
        block_manager = getattr(scheduler, "cache_manager", None)
        block_size = int(getattr(engine, "config", None).block_size or self.block_size)
        # prefix cache 未开启或 manager 不存在时，没有索引可导出。
        if block_manager is None or not getattr(
            scheduler, "enable_prefix_caching", False
        ):
            return None
        # etcd 跨节点部署要求共享目录，否则 frontend 无法读取 worker 本地文件。
        if (
            getattr(cluster_options, "discovery_backend", "file") == "etcd"
            and getattr(cluster_options, "cache_index_dir", None) is None
        ):
            return None

        cache_index_dir = getattr(cluster_options, "cache_index_dir", None)
        # 每个 worker 实例使用独立子目录，避免 snapshot/event 相互覆盖。
        output_dir = (
            cache_index_dir / instance_id
            if cache_index_dir is not None
            else self.registry.root.parent / "cache-index" / instance_id
        )
        # namespace 绑定模型元数据和 block size，防止不同哈希空间被误匹配。
        return WorkerCacheIndexExporter(
            block_manager=block_manager,
            output_dir=output_dir,
            namespace=model_cache_namespace(self.model_path, block_size),
            block_size=block_size,
        )

    def _create_app(self) -> FastAPI:
        app = super()._create_app()
        engine_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def cluster_lifespan(app: FastAPI):
            async with engine_lifespan(app):
                try:
                    yield
                finally:
                    if self.worker_discovery is not None:
                        await self.worker_discovery.stop()
                        self.worker_discovery = None
                    await self.registry.aclose()

        app.router.lifespan_context = cluster_lifespan
        return app


class ClusterWorkerRuntime:
    """一个 cluster worker 进程的运行时协调者。

    它只负责组装和生命周期编排：
    1. 让 ``ClusterInferenceServer`` 创建 FastAPI 应用和 discovery 服务；
    2. 创建 ``DrainingUvicornServer`` 承载 HTTP 服务；
    3. 把退出信号触发的 drain 回调绑定到推理服务的 discovery 状态更新。
    """

    def __init__(self, inference_server: ClusterInferenceServer) -> None:
        self.inference_server = inference_server
        self.http_server = self._create_http_server()

    @staticmethod
    def _listening_port(http_server: uvicorn.Server) -> int:
        """从 asyncio listener socket 读取 OS 分配的真实 TCP 端口。"""
        for listener in http_server.servers or []:
            for sock in listener.sockets or []:
                sockname = sock.getsockname()
                if len(sockname) >= 2:
                    return int(sockname[1])
        raise RuntimeError("worker HTTP server has no TCP listener")

    async def _on_http_server_ready(self, http_server: uvicorn.Server) -> None:
        """HTTP socket 绑定后注册 discovery，避免 registry 记录 port=0。"""
        actual_port = self._listening_port(http_server)
        self.inference_server.port = actual_port
        await self.inference_server.start_discovery(http_port=actual_port)

    def _create_http_server(self) -> DrainingUvicornServer:
        config = uvicorn.Config(
            self.inference_server._create_app(),
            host=self.inference_server.host,
            port=self.inference_server.port,
        )
        http_server = DrainingUvicornServer(
            config,
            # 生产路径从环境变量读取；DrainingUvicornServer 保留显式参数便于测试。
            drain_grace_seconds=drain_grace_seconds(),
            on_ready=self._on_http_server_ready,
        )
        http_server.on_drain_start = self.inference_server.start_draining
        return http_server

    def run(self) -> None:
        self.http_server.run()


def _resolve_kv_transfer_config(
    kv_transfer_config_str: str,
    worker_type: WorkerType,
) -> KVTransferConfig:
    raw_config = json.loads(kv_transfer_config_str)
    if not isinstance(raw_config, dict):
        raise ValueError("--kv-transfer-config must be a JSON object")

    expected_kv_role = PD_KV_ROLES.get(worker_type)
    if expected_kv_role is not None:
        configured_role = raw_config.get("kv_role")
        if configured_role is not None and configured_role != expected_kv_role:
            raise ValueError(
                f"--disaggregation-mode {worker_type.value} requires "
                f"kv_role={expected_kv_role!r}, got {configured_role!r}"
            )
        raw_config["kv_role"] = expected_kv_role

    return parse_kv_transfer_config(json.dumps(raw_config))


def _build_server(argv: list[str]) -> ClusterInferenceServer:
    cluster_options = parse_worker_cluster_args(argv)
    inference_argv = remove_cluster_worker_args(argv)
    cfg = BaseConfig(inference_argv)

    if cfg.pp > 1:
        raise ValueError("the cluster worker currently requires --pp 1")

    if cluster_options.worker_type is WorkerType.PREFILL:
        os.environ["INFINILM_HOST_IP"] = cluster_options.advertise_host
        os.environ["INFINILM_MOONCAKE_BOOTSTRAP_HOST"] = (
            cluster_options.mooncake_bootstrap_host
        )
        os.environ["INFINILM_MOONCAKE_BOOTSTRAP_PORT"] = str(
            cluster_options.mooncake_bootstrap_port
        )
    elif cluster_options.worker_type is WorkerType.DECODE:
        os.environ["INFINILM_HOST_IP"] = cluster_options.advertise_host

    if cfg.use_legacy_moe:
        moe_ep_backend, expert_parallel_size = "disabled", 1
    else:
        moe_ep_backend, expert_parallel_size = configure_moe_ep_backend(
            cfg.tp,
            cfg.dp,
            cfg.ep,
            cfg.moe_ep_backend,
            cfg.model,
        )

    kv_transfer_config: KVTransferConfig | None = None
    if cfg.kv_transfer_config:
        kv_transfer_config = _resolve_kv_transfer_config(
            cfg.kv_transfer_config,
            cluster_options.worker_type,
        )
    elif cluster_options.worker_type in PD_KV_ROLES:
        raise ValueError(
            f"--disaggregation-mode {cluster_options.worker_type.value} "
            "requires --kv-transfer-config"
        )

    if cluster_options.worker_type in PD_KV_ROLES and cfg.tp != 1:
        raise ValueError("PD cluster workers currently require --tp 1")

    registry = create_worker_registry(cluster_options)
    server = ClusterInferenceServer(
        cluster_options=cluster_options,
        registry=registry,
        model_path=cfg.model,
        device=cfg.get_device_str(cfg.device),
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tp,
        pipeline_parallel_size=cfg.pp,
        pipeline_parallel_stage=cfg.node_rank,
        master_addr=cfg.master_addr,
        master_port=cfg.master_port,
        moe_ep_backend=moe_ep_backend,
        moe_ep_size=expert_parallel_size,
        use_legacy_moe=cfg.use_legacy_moe,
        cache_type="paged" if cfg.enable_paged_attn else "static",
        max_tokens=cfg.max_new_tokens,
        max_batch_size=cfg.max_batch_size,
        num_blocks=cfg.num_blocks,
        block_size=cfg.block_size,
        max_cache_len=cfg.max_cache_len,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        host=cfg.host,
        port=cfg.port,
        enable_graph=cfg.enable_graph,
        enable_workspace=cfg.enable_workspace,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        attn_backend=cfg.attn,
        use_mla=cfg.use_mla,
        skip_load=cfg.skip_load,
        weight_load_mode=cfg.weight_load_mode,
        ignore_eos=cfg.ignore_eos,
        kv_transfer_config=kv_transfer_config,
        enable_prefix_caching=cfg.enable_prefix_caching,
        pre_transpose=cfg.pre_transpose,
    )
    server.log_level = cfg.log_level
    return server


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    server: ClusterInferenceServer = _build_server(args)
    setup_logging(server.log_level)
    logger.info(
        "Starting InfiniLM %s worker: namespace=%s endpoint=%s:%s",
        server.cluster_options.worker_type.value,
        server.cluster_options.namespace,
        server.cluster_options.advertise_host,
        server.port,
    )
    runtime = ClusterWorkerRuntime(server)
    runtime.run()


if __name__ == "__main__":
    main()
