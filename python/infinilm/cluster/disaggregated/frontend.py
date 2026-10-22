"""Discovery-aware OpenAI frontend for a prefill/decode separated cluster."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

from ..config import _add_cluster_arguments
from ..environment import (
    bootstrap_timeout_seconds,
    connect_timeout,
    health_interval_seconds,
    request_timeout,
)
from ..worker_registry import (
    WorkerRecord,
    WorkerRegistryWatcher,
    WorkerType,
    create_worker_registry,
)
from .client_reconciler import PdClientReconciler
from .monitoring import PD_MONITORING_HTML, pd_monitoring_snapshot
from .pd_http_transport import send_prefill_request, stream_decode_response
from .router import DisaggregatedRouter
from .worker_client_pool import WorkerClientPoolManager

logger = logging.getLogger(__name__)


class _CancellationSafeAsyncIterator:
    """确保外层响应任务取消时会关闭内层 async generator。

    Starlette 在客户端断开时会取消 StreamingResponse 的外层迭代任务，
    但不总是显式关闭 body iterator。这里捕获外层取消并主动 ``aclose()``
    内层 generator，让 P/D 任务的 finally 清理逻辑可靠执行。
    """

    def __init__(self, inner: AsyncIterator[bytes]) -> None:
        """保存 StreamingResponse 实际消费的内层 async generator。"""
        self._inner = inner

    def __aiter__(self) -> AsyncIterator[bytes]:
        """满足 AsyncIterator 协议，StreamingResponse 会调用它取得迭代器。"""
        return self

    async def __anext__(self) -> bytes:
        """读取下一个响应 chunk，并在外层任务取消时关闭内层资源。"""
        try:
            return await self._inner.__anext__()
        except asyncio.CancelledError:
            # 必须在重新抛出 CancelledError 前完成清理；否则外层任务已经
            # 结束，内层 generator 可能再也不会获得调度机会。
            await self._inner.aclose()
            raise

    async def aclose(self) -> None:
        """响应正常结束或框架主动关闭 body iterator 时释放内层资源。"""
        await self._inner.aclose()


class _PrefetchedAsyncIterator:
    """先取得首 chunk，再把剩余响应交给 StreamingResponse。

    handler 会在构造 ``StreamingResponse`` 前调用一次内层 iterator。这样，
    backend 在首个 token 前返回 4xx/5xx 时，HTTP status 仍可正常返回；
    同时也能保证 P/D orchestration generator 已经启动。若框架随后尚未
    迭代 body 就关闭响应，``aclose()`` 会继续关闭内层 generator。
    """

    def __init__(
        self,
        first_chunk: bytes,
        inner: _CancellationSafeAsyncIterator,
    ) -> None:
        """缓存已取得的首 chunk，并保存可继续迭代的内层响应流。"""
        self._first_chunk = first_chunk
        self._inner = inner

    def __aiter__(self) -> AsyncIterator[bytes]:
        """满足 AsyncIterator 协议。"""
        return self

    async def __anext__(self) -> bytes:
        """先返回缓存 chunk，再继续读取内层 P/D 响应流。"""
        if self._first_chunk is not None:
            chunk = self._first_chunk
            self._first_chunk = None
            return chunk
        try:
            return await self._inner.__anext__()
        except asyncio.CancelledError:
            await self._inner.aclose()
            raise

    async def aclose(self) -> None:
        """关闭内层 generator，释放 P/D task、HTTP response 和 inflight。"""
        await self._inner.aclose()


@dataclass(frozen=True)
class DisaggregatedFrontendOptions:
    namespace: str
    # 共享 CLI 仍叫 --file-kv；这里按实际语义命名为 discovery 根目录。
    discovery_root: Path
    router_mode: str
    http_host: str
    http_port: int
    discovery_backend: str = "file"
    etcd_endpoints: str = "127.0.0.1:2379"
    etcd_prefix: str = "/infinilm"
    etcd_username: str | None = None
    etcd_password_env: str = "ETCD_PASSWORD"
    etcd_ca_cert: str | None = None
    etcd_cert_file: str | None = None
    etcd_key_file: str | None = None
    etcd_timeout: float = 3.0
    min_prefill_workers: int = 1
    min_decode_workers: int = 1


def parse_disaggregated_frontend_args(
    argv: list[str] | None = None,
) -> DisaggregatedFrontendOptions:
    parser = argparse.ArgumentParser(
        description="Start a discovery-aware InfiniLM PD cluster frontend"
    )
    _add_cluster_arguments(parser)
    parser.add_argument(
        "--router-mode",
        choices=["round-robin", "least-loaded"],
        default="round-robin",
    )
    parser.add_argument("--min-prefill-workers", type=int, default=1)
    parser.add_argument("--min-decode-workers", type=int, default=1)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=18520)
    args = parser.parse_args(argv)
    return DisaggregatedFrontendOptions(
        namespace=args.namespace,
        discovery_root=Path(args.file_kv),
        router_mode=args.router_mode,
        http_host=args.http_host,
        http_port=args.http_port,
        discovery_backend=args.discovery_backend,
        etcd_endpoints=args.etcd_endpoints,
        etcd_prefix=args.etcd_prefix,
        etcd_username=args.etcd_username,
        etcd_password_env=args.etcd_password_env,
        etcd_ca_cert=args.etcd_ca_cert,
        etcd_cert_file=args.etcd_cert_file,
        etcd_key_file=args.etcd_key_file,
        etcd_timeout=args.etcd_timeout,
        min_prefill_workers=max(1, args.min_prefill_workers),
        min_decode_workers=max(1, args.min_decode_workers),
    )


class DisaggregatedFrontend:
    """PD 分离集群的统一 frontend，负责发现 worker 并编排 P/D 请求链路。"""

    def __init__(self, options: DisaggregatedFrontendOptions) -> None:
        """初始化 PD frontend 的 registry、watcher、HTTP client 和路由状态。"""
        self.options = options
        # 启动时读取一次，保证配置稳定并让非法环境变量立即暴露。
        self._connect_timeout = connect_timeout()
        self._request_timeout = request_timeout()
        self._bootstrap_timeout = bootstrap_timeout_seconds()
        self._discovery_refresh_interval_seconds = health_interval_seconds()
        self.registry = create_worker_registry(options)
        # PD frontend 只关心 prefill 和 decode，不能把 aggregated worker 当作候选。
        self.watcher = WorkerRegistryWatcher(
            self.registry,
            worker_types=(WorkerType.PREFILL, WorkerType.DECODE),
        )
        # PD router 只负责角色选择和请求计数，不管理 discovery 或 HTTP client。
        self.router = DisaggregatedRouter(options.router_mode)

        # client 池独立管理；这里注入 inflight，用于延迟清理仍有请求的连接。
        self.worker_client_pool_manager = WorkerClientPoolManager(
            connect_timeout=self._connect_timeout,
            request_timeout=self._request_timeout,
            inflight=self.router.worker_inflight,
            bootstrap_timeout=self._bootstrap_timeout,
            retry_delay=max(
                3.0 * self._discovery_refresh_interval_seconds,
                self._bootstrap_timeout,
            ),
        )
        # reconcile 只属于后台生命周期；API 请求路径不触发 bootstrap。
        self.client_reconciler = PdClientReconciler(
            watcher_records=lambda: self.watcher.worker_records,
            manager=self.worker_client_pool_manager,
            interval=self._discovery_refresh_interval_seconds,
        )
        self._prefill_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _filter_worker_records(
        records: Iterable[WorkerRecord],
        role: WorkerType,
        model_name: str | None = None,
        api_route: str | None = None,
    ) -> list[WorkerRecord]:
        """在 frontend 层过滤模型和 API route，路由决策保留在 frontend。"""
        result = [
            record
            for record in records
            if record.worker_type is role
            and record.state == "ready"
            and record.engine_healthy
            and (model_name is None or record.served_model_name == model_name)
            and (
                bool(record.metadata.get("supported_api_routes"))
                and (
                    api_route is None
                    or api_route in record.metadata["supported_api_routes"]
                )
            )
        ]
        return sorted(result, key=lambda record: record.instance_id)

    def _candidate_model_names(self) -> list[str]:
        """返回 discovery 中同时存在健康 P/D worker 的候选模型名。"""
        records = self.watcher.worker_records
        # 先分别求集合，再取交集，避免某一边缺失时误判模型 ready。
        prefill_model_names = {
            record.served_model_name
            for record in self._filter_worker_records(records, WorkerType.PREFILL)
        }
        decode_model_names = {
            record.served_model_name
            for record in self._filter_worker_records(records, WorkerType.DECODE)
        }
        return sorted(prefill_model_names & decode_model_names)

    def _has_minimum_routable_workers(self, model_name: str) -> bool:
        """判断指定模型是否已达到 P/D 最小可用 client 数。"""
        records = self.watcher.worker_records
        for role, minimum in (
            (WorkerType.PREFILL, self.options.min_prefill_workers),
            (WorkerType.DECODE, self.options.min_decode_workers),
        ):
            routable_ids = {
                record.instance_id
                for record in self.worker_client_pool_manager.routable_records(
                    model_name, role
                )
            }
            eligible_records = [
                record
                for record in self._filter_worker_records(records, role, model_name)
                if record.instance_id in routable_ids
            ]
            if len(eligible_records) < minimum:
                return False
        return True

    def _ready_model_names(self) -> list[str]:
        """纯读返回当前已有可用 P/D client 的模型。

        该方法不触发 reconcile/bootstrap，因此可以作为 health、models 和
        chat 请求的统一快照视图。
        """
        return [
            model_name
            for model_name in self._candidate_model_names()
            if self._has_minimum_routable_workers(model_name)
        ]

    async def _select_worker_clients(
        self,
        model_name: str,
        api_route: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """从已初始化 client 中选择 P/D，并原子登记本次请求。"""
        selected: dict[WorkerType, WorkerRecord] = {}
        for role in (WorkerType.PREFILL, WorkerType.DECODE):
            records = self._filter_worker_records(
                self.worker_client_pool_manager.routable_records(model_name, role),
                role,
                model_name,
                api_route,
            )
            if not records:
                raise HTTPException(
                    status_code=503,
                    detail=f"No ready {role.value} workers for model {model_name}",
                )
            selected[role] = self.router.select_record(role, records)

        # P/D 必须同时拿到；acquire_workers 在 manager 状态锁内登记 inflight，
        # 防止后台 reconcile 在本请求开始前关闭任一连接。
        acquired = await self.worker_client_pool_manager.acquire_workers(
            model_name,
            [
                (WorkerType.PREFILL, selected[WorkerType.PREFILL].instance_id),
                (WorkerType.DECODE, selected[WorkerType.DECODE].instance_id),
            ],
        )
        if len(acquired) != 2:
            raise HTTPException(
                status_code=503,
                detail=f"PD workers for model {model_name} are not ready",
            )
        return acquired[0], acquired[1]

    async def _send_prefill_background(
        self,
        client_info: dict[str, Any],
        prefill_dp_rank: int,
        api_route: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> None:
        """执行 prefill 请求，并维护 prefill worker 的 inflight 指标。

        这里不再吞掉异常。prefill task 由 ``_generate_pd_stream_impl()``
        统一等待：成功后才允许输出 decode chunk，失败则转换成请求错误。
        """
        # inflight 已由 acquire_workers() 原子登记；这里只推进角色总量。
        self.router.begin_request(client_info["record"], count_inflight=False)
        try:
            # 复制 payload，避免异步 prefill 与 decode 流同时修改同一个字典。
            await send_prefill_request(
                client_info,
                prefill_dp_rank,
                api_route,
                payload.copy(),
                request_id,
            )
        finally:
            # finish_request() 必须与 begin_request() 一一对应；否则
            # DRAINING worker 的延迟清理和负载路由都会误判仍有请求。
            self.router.finish_request(client_info["instance_id"])

    async def _stream_decode(
        self,
        prefill_client_info: dict[str, Any],
        prefill_dp_rank: int,
        decode_client_info: dict[str, Any],
        api_route: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> AsyncIterator[bytes]:
        """通过 decode worker 转发流式响应，并维护 decode inflight。"""
        self.router.begin_request(
            decode_client_info["record"],
            count_inflight=False,
        )
        try:
            # PD HTTP transport 会携带 prefill bootstrap 信息，让 decode
            # worker 通过 Mooncake 拉取远端 KV。
            async for chunk in stream_decode_response(
                prefill_client_info,
                prefill_dp_rank,
                decode_client_info,
                api_route,
                payload,
                request_id=request_id,
            ):
                yield chunk
        finally:
            self.router.finish_request(decode_client_info["instance_id"])

    async def _generate_pd_stream_impl(
        self,
        prefill_task: asyncio.Task[None],
        decode_stream: AsyncIterator[bytes],
        request_id: str,
        *,
        streaming: bool = False,
        http_request: Request | None = None,
    ) -> AsyncIterator[bytes]:
        """协调并行的 prefill/decode 任务，并统一清理请求生命周期。

        decode task 会先启动，以保持 Mooncake 并行拉取 KV 的行为；但在
        prefill 成功前，任何 decode 输出都不会发给客户端。这样 prefill
        失败不会伪装成成功响应，客户端断开时也能同时取消两侧任务。

        P/D stream 编排逻辑：

        ```text
        decode 先启动，保持 Mooncake 并行拉取 KV 的行为
              |
              v
        decode 的首个 chunk 会被缓存
              |
              v
        等待 prefill 成功
              |
              +-- prefill 成功 -> 才向客户端输出 decode chunk
              |
              +-- prefill 失败 -> 输出错误，取消 decode
              |
              +-- 客户端断开 -> 取消 prefill 和 decode
        ```

        内部状态分为五类：

        1. ``pending_chunk``：decode 已产生的首 chunk，但尚未暴露给客户端；
        2. ``prefill_task``：远端 prefill 的成功/失败结果；
        3. ``decode_task``：当前正在等待的 decode ``__anext__()`` 任务；
        4. ``response_started``：是否已经向客户端暴露过正常响应 bytes；
        5. ``error_body``/``initial_error``：流中错误和首 token 前错误。

        所有正常、失败、取消和断开路径都会进入 finally，统一清理
        prefill task、decode task 和 decode async generator。
        """
        # 立即启动 decode 的第一个 __anext__()，而不是等待 prefill 完成。
        # Mooncake PD 流程依赖 decode 侧并行发起远端 KV 拉取；串行等待
        # 会失去 prefill/decode 重叠执行的机会。
        decode_task = asyncio.create_task(decode_stream.__anext__())
        pending_chunk: bytes | None = None
        error_body: bytes | None = None
        initial_error: Exception | None = None
        response_started = False

        async def client_disconnected() -> bool:
            """主动查询客户端连接是否已经断开。"""
            # StreamingResponse 不总会在 body 产出首 chunk 前取消迭代
            # 任务，因此等待 P/D 任务时需要周期性检查 HTTP receive 状态。
            return http_request is not None and await http_request.is_disconnected()

        try:
            # 第一阶段：等待 prefill 完成，同时允许 decode 首 chunk 先就绪。
            # timeout=0.1 保证两个任务都未完成时也有机会检查 disconnect。
            while not prefill_task.done():
                done, _pending = await asyncio.wait(
                    {prefill_task, decode_task},
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done and await client_disconnected():
                    # 客户端已经离开；return 会触发 finally 取消 P/D 任务。
                    return
                if decode_task not in done:
                    continue

                try:
                    # 先缓存 decode 首个 chunk，但不向客户端暴露；
                    # 必须等待 prefill 成功后再输出。
                    pending_chunk = await decode_task
                except StopAsyncIteration:
                    # decode 在 prefill 完成前结束，不能证明 PD 请求成功；
                    # 直接结束外层响应，并让 finally 取消仍在运行的 prefill。
                    return
                decode_task = None
                if not prefill_task.done():
                    # 首 chunk 已缓存但 prefill 未完成，此时只等待 prefill，
                    # 不继续拉取第二个 decode chunk，避免提前扩大生成范围。
                    while not prefill_task.done():
                        done, _pending = await asyncio.wait(
                            {prefill_task},
                            timeout=0.1,
                        )
                        if not done and await client_disconnected():
                            return

            # 第二阶段：prefill task 已结束。这里 await 会重新抛出 prefill
            # 的异常；成功时返回 None，进入 decode 输出阶段。
            await prefill_task

            # 第三阶段：prefill 已成功，此时才允许把 decode 输出交给客户端。
            if decode_task is None and pending_chunk is None:
                # 第一阶段 decode 尚未产出首 chunk，需要继续等待。如果首
                # chunk 已缓存，不能再创建第二个 __anext__() task。
                decode_task = asyncio.create_task(decode_stream.__anext__())
            if pending_chunk is None:
                while not decode_task.done():
                    done, _pending = await asyncio.wait(
                        {decode_task},
                        timeout=0.1,
                    )
                    if not done and await client_disconnected():
                        return
                try:
                    # decode 正常结束，取出已缓冲的首 chunk。
                    pending_chunk = decode_task.result()
                except StopAsyncIteration:
                    return
                # 首 chunk 的 __anext__() task 必须显式丢弃。它仍可能持有
                # decode generator 的 asend 协程；若保留 task，后续 async for
                # 会并发调用同一个 generator 并触发 already running。
                decode_task = None

            # prefill 已成功，首 chunk 可以安全暴露给客户端。
            # response_started 必须在 yield 前置位：yield 被消费并恢复后，
            # HTTP 200 头已经可能发出，后续错误只能作为流内错误输出。
            response_started = True
            yield pending_chunk
            # 后续 chunk 不需要缓存 prefill 结果，直接透传 decode 响应。
            async for chunk in decode_stream:
                yield chunk
        except HTTPException as exc:
            logger.error(
                "PD request %s failed: HTTP %s - %s",
                request_id,
                exc.status_code,
                exc.detail,
            )
            if response_started:
                error_body = json.dumps(
                    {"error": {"message": exc.detail, "code": exc.status_code}}
                ).encode()
            else:
                # 首个正常 chunk 尚未返回。handler 会先预取首 chunk 再构造
                # StreamingResponse，因此这里抛出后仍能保留真实 HTTP status。
                initial_error = exc
        except Exception as exc:
            logger.exception("PD stream %s failed", request_id)
            if response_started:
                error_body = json.dumps(
                    {"error": {"message": str(exc), "code": 500}}
                ).encode()
            else:
                initial_error = HTTPException(status_code=500, detail=str(exc))
        finally:
            # 统一清理：覆盖正常完成、prefill 失败、decode 失败和客户端断开。
            prefill_task.cancel()
            cleanup_tasks = [prefill_task]
            if decode_task is not None and not decode_task.done():
                decode_task.cancel()
                cleanup_tasks.append(decode_task)
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            # 关闭 decode generator，触发 _stream_decode() 的 finally，
            # 释放 decode inflight 并关闭 Mooncake proxy 的 HTTP response。
            await decode_stream.aclose()

        if error_body is not None:
            # 错误在资源清理完成后输出；streaming 模式保持 SSE 语法，
            # 非流式模式输出普通 JSON。
            if streaming:
                yield b"data: " + error_body + b"\n\n"
            else:
                yield error_body
        if initial_error is not None:
            raise initial_error

    def _generate_pd_stream(
        self,
        prefill_task: asyncio.Task[None],
        decode_stream: AsyncIterator[bytes],
        request_id: str,
        *,
        streaming: bool = False,
        http_request: Request | None = None,
    ) -> AsyncIterator[bytes]:
        """构造外层响应实际使用的 P/D 流迭代器。

        业务状态机在 ``_generate_pd_stream_impl()`` 中实现；本方法只负责
        套一层 cancellation-safe wrapper，把 Starlette 的外层取消转换成
        内层 generator 的显式 ``aclose()``。
        """
        return _CancellationSafeAsyncIterator(
            self._generate_pd_stream_impl(
                prefill_task,
                decode_stream,
                request_id,
                streaming=streaming,
                http_request=http_request,
            )
        )

    def create_app(self) -> FastAPI:
        """创建 PD frontend 的 FastAPI 应用和 OpenAI/运维接口。"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """启动 discovery watcher 和后台 client reconcile 生命周期。"""
            # 先同步刷新一次 discovery，再启动周期 watcher；不等待 bootstrap，
            # frontend 可以先监听 HTTP，并在 client 就绪前返回 503。
            await self.watcher.refresh(self.options.namespace)
            self.watcher.start(
                self.options.namespace,
                interval=max(0.1, self._discovery_refresh_interval_seconds),
            )
            self.client_reconciler.start()
            logger.info(
                "PD frontend is listening; prefill=%d decode=%d models=%s",
                len(
                    self._filter_worker_records(
                        self.watcher.worker_records,
                        WorkerType.PREFILL,
                    )
                ),
                len(
                    self._filter_worker_records(
                        self.watcher.worker_records,
                        WorkerType.DECODE,
                    )
                ),
                self._candidate_model_names(),
            )
            try:
                yield
            finally:
                # 先停止后台 reconcile，再停止 watcher，避免清理过程中读到新快照。
                await self.client_reconciler.stop()
                # 停止 watcher 后，再取消尚未完成的 prefill 后台任务。
                await self.watcher.stop()
                for task in self._prefill_tasks:
                    task.cancel()
                if self._prefill_tasks:
                    await asyncio.gather(
                        *self._prefill_tasks,
                        return_exceptions=True,
                    )
                # manager 统一关闭每个 worker 独立持有的 HTTP client。
                await self.worker_client_pool_manager.aclose()
                await self.registry.aclose()

        app = FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health():
            """返回进程健康、discovery 候选数和当前已 ready 模型。"""
            # 请求路径纯读；bootstrap 由 client_reconciler 后台完成。
            ready_models = self._ready_model_names()
            records = self.watcher.worker_records
            return {
                "status": "healthy",
                "namespace": self.options.namespace,
                "ready": bool(ready_models),
                "models": ready_models,
                # 对外字段保持兼容；这里统计 discovery 候选，不等同于已初始化 client。
                "prefill_workers": len(
                    self._filter_worker_records(records, WorkerType.PREFILL)
                ),
                "decode_workers": len(
                    self._filter_worker_records(records, WorkerType.DECODE)
                ),
            }

        @app.get("/metrics")
        async def metrics():
            """输出 PD frontend 的 Prometheus 指标。"""
            # 先输出角色维度 worker 数，再输出请求和 inflight 指标。
            lines = [
                "# HELP infinilm_pd_workers Discovered healthy worker candidates by role.",
                "# TYPE infinilm_pd_workers gauge",
            ]
            for role in (WorkerType.PREFILL, WorkerType.DECODE):
                lines.append(
                    f'infinilm_pd_workers{{role="{role.value}"}} '
                    f"{len(self._filter_worker_records(self.watcher.worker_records, role))}"
                )
            lines.extend(
                [
                    "# HELP infinilm_pd_frontend_requests_total Requests selected by role.",
                    "# TYPE infinilm_pd_frontend_requests_total counter",
                ]
            )
            for role, count in self.router.role_request_totals.items():
                lines.append(
                    f'infinilm_pd_frontend_requests_total{{role="{role}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP infinilm_pd_frontend_inflight_requests Active requests by worker.",
                    "# TYPE infinilm_pd_frontend_inflight_requests gauge",
                ]
            )
            for role in (WorkerType.PREFILL, WorkerType.DECODE):
                for worker in self._filter_worker_records(
                    self.watcher.worker_records, role
                ):
                    lines.append(
                        f'infinilm_pd_frontend_inflight_requests{{role="{role.value}",'
                        f'worker_id="{worker.instance_id}"}} '
                        f"{self.router.worker_inflight.count(worker.instance_id)}"
                    )
            return PlainTextResponse(
                "\n".join(lines) + "\n",
                media_type="text/plain; version=0.0.4",
            )

        @app.get("/v1/models")
        async def models():
            """返回已满足 P/D 可用条件的 OpenAI models 列表。"""
            ready_models = self._ready_model_names()
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "infinilm",
                    }
                    for model_name in ready_models
                ],
            }

        @app.get("/monitoring", response_class=HTMLResponse)
        async def monitoring_page():
            """返回内置 PD 监控页面。"""
            return HTMLResponse(
                PD_MONITORING_HTML,
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/api/monitoring")
        async def monitoring_api():
            """返回 PD 监控页面使用的 JSON 快照。"""
            return pd_monitoring_snapshot(self)

        async def handle_completions(api_route: str, request: Request):
            """统一处理 completions/chat completions 请求。"""
            try:
                # 先保证请求体是 JSON object，后续逻辑才能安全读取字段。
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise HTTPException(
                    status_code=400, detail="Request body must be an object"
                )

            requested_model = payload.get("model")
            if requested_model:
                # 显式指定模型只检查 discovery 是否存在；client 尚未初始化时
                # 由后续选择阶段返回 503，而不是在请求路径触发 bootstrap。
                model_name = requested_model
                if model_name not in self._candidate_model_names():
                    raise HTTPException(
                        status_code=404, detail=f"model not found: {model_name}"
                    )
            else:
                # 未指定模型时选择排序后的第一个 ready 模型。
                ready_models = self._ready_model_names()
                if not ready_models:
                    raise HTTPException(status_code=503, detail="No ready PD models")
                model_name = ready_models[0]
            payload = {**payload, "model": model_name}

            # 只从已初始化 client 选择；P/D 必须同时声明支持当前请求路径。
            prefill_client_info, decode_client_info = (
                await self._select_worker_clients(model_name, api_route)
            )
            # 当前 PD 流程固定使用 prefill worker 的 data-parallel rank 0。
            prefill_dp_rank = 0

            request_id = f"cmpl-{uuid.uuid4().hex}"
            # PD 请求需要两个角色同时参与：
            # 1. prefill task 负责计算 prompt 并发布 KV；
            # 2. decode stream 负责拉取 KV 并生成增量 token。
            # _generate_pd_stream_impl() 会让 decode 先启动以并行拉取 KV，
            # 但会等 prefill 成功后才把 decode 输出暴露给客户端。
            prefill_task = asyncio.create_task(
                self._send_prefill_background(
                    prefill_client_info,
                    prefill_dp_rank,
                    api_route,
                    payload,
                    request_id,
                )
            )
            # _prefill_tasks 只用于 frontend shutdown 时的兜底清理；
            # 正常请求断开由 cancellation-safe iterator 的 finally 清理。
            self._prefill_tasks.add(prefill_task)
            prefill_task.add_done_callback(self._prefill_tasks.discard)
            # async generator 只有被迭代时才开始执行；这里先创建对象，
            # 后续由 StreamingResponse 驱动 _stream_decode()。
            decode_stream = self._stream_decode(
                prefill_client_info,
                prefill_dp_rank,
                decode_client_info,
                api_route,
                payload,
                request_id=request_id,
            )
            media_type = (
                "text/event-stream"
                if payload.get("stream", False)
                else "application/json"
            )
            # 先预取首个响应 chunk，再构造 StreamingResponse。这个顺序解决
            # 两个问题：backend 首 token 前错误可以返回真实 HTTP status；
            # P/D generator 已必然启动，不会出现“创建后从未迭代”的泄漏。
            response_stream = self._generate_pd_stream(
                prefill_task,
                decode_stream,
                request_id,
                streaming=bool(payload.get("stream", False)),
                http_request=request,
            )
            try:
                first_chunk = await response_stream.__anext__()
            except StopAsyncIteration:
                raise HTTPException(
                    status_code=502,
                    detail="PD backend returned an empty response",
                ) from None

            # worker 的流式 body 是 SSE；非流式请求虽然也经 StreamingResponse
            # 分块转发，但协议本体仍是 JSON，Content-Type 必须与请求匹配。
            return StreamingResponse(
                _PrefetchedAsyncIterator(first_chunk, response_stream),
                media_type=media_type,
            )

        @app.post("/chat/completions")
        async def chat_completions(request: Request):
            """处理兼容路径 chat completions API。"""
            return await handle_completions("/chat/completions", request)

        @app.post("/v1/chat/completions")
        async def v1_chat_completions(request: Request):
            """处理 OpenAI-compatible chat completions API。"""
            return await handle_completions("/v1/chat/completions", request)

        return app

def main(argv: list[str] | None = None) -> None:
    options = parse_disaggregated_frontend_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    frontend = DisaggregatedFrontend(options)
    display_host = "127.0.0.1" if options.http_host == "0.0.0.0" else options.http_host
    logger.info(
        "Starting InfiniLM PD frontend at %s:%s namespace=%s discovery_root=%s router_mode=%s",
        options.http_host,
        options.http_port,
        options.namespace,
        options.discovery_root,
        options.router_mode,
    )
    logger.info("PD API: http://%s:%s/v1", display_host, options.http_port)
    uvicorn.run(
        frontend.create_app(),
        host=options.http_host,
        port=options.http_port,
    )


if __name__ == "__main__":
    main()
