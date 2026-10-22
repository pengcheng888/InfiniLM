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

from ...server.mooncake_proxy_server import send_request, stream_response
from ..config import _add_cluster_arguments
from ..environment import (
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
from .monitoring import PD_MONITORING_HTML, pd_monitoring_snapshot
from .router import DisaggregatedRouter
from .worker_client_pool import WorkerClientPoolManager

logger = logging.getLogger(__name__)


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

    async def _reconcile_worker_clients_for_model(self, model_name: str) -> None:
        """按模型筛选期望 worker，并同步 P/D HTTP client 池。"""
        records = self.watcher.worker_records

        # 针对同一个 discovery 快照，分别核对 prefill 和 decode 两个 client 池。
        prefill_records = self._filter_worker_records(
                records,
                WorkerType.PREFILL,
                model_name,
            )
        
        await self.worker_client_pool_manager.reconcile_model(
            role=WorkerType.PREFILL,
            model_name=model_name,
            records=prefill_records,
        )

        decode_records = self._filter_worker_records(
                        records,
                        WorkerType.DECODE,
                        model_name,
                    )
        
        await self.worker_client_pool_manager.reconcile_model(
            role=WorkerType.DECODE,
            model_name=model_name,
            records=decode_records,
        )

    def _select_worker_client(
        self,
        role: WorkerType,
        model_name: str,
        api_route: str | None = None,
    ) -> dict[str, Any]:
        """在已同步的 client 集合中选择指定角色的 worker。"""
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
        record = self.router.select_record(role, records)
        # routable_records 已保证 client 初始化成功；这里保留防御式读取。
        client_info = self.worker_client_pool_manager.get(
            model_name,
            role,
            record.instance_id,
        )
        if client_info is None:
            raise HTTPException(
                status_code=503,
                detail=f"{role.value} worker {record.instance_id} is not ready",
            )
        return client_info

    async def _send_prefill_background(
        self,
        client_info: dict[str, Any],
        prefill_dp_rank: int,
        api_route: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> None:
        """异步执行 prefill 请求，让 decode 可以并行拉取远端 KV。"""
        self.router.begin_request(client_info["record"])
        try:
            # 复制 payload，避免异步 prefill 与 decode 流同时修改同一个字典。
            await send_request(
                client_info,
                prefill_dp_rank,
                api_route,
                payload.copy(),
                request_id,
            )
        except Exception:
            # prefill 后台任务不在请求协程内等待，这里记录异常供运维排查。
            logger.exception("Prefill request %s failed", request_id)
        finally:
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
        self.router.begin_request(decode_client_info["record"])
        try:
            # Mooncake proxy 层会携带 prefill bootstrap 信息并拉取远端 KV。
            async for chunk in stream_response(
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

    def create_app(self) -> FastAPI:
        """创建 PD frontend 的 FastAPI 应用和 OpenAI/运维接口。"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """启动 discovery watcher，退出时清理后台任务和 HTTP client。"""
            # 先同步刷新一次，再启动周期 watcher，保证初始模型列表尽量及时。
            await self.watcher.refresh(self.options.namespace)
            self.watcher.start(
                self.options.namespace,
                interval=max(0.1, self._discovery_refresh_interval_seconds),
            )
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
            """返回 frontend 存活状态、P/D discovery 候选数和 ready 模型。"""
            ready_models = await self._reconcile_and_collect_ready_models()
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
            ready_models = await self._reconcile_and_collect_ready_models()
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
                # 显式指定模型时必须已经 ready，避免请求落到半可用 P/D 组合。
                model_name = requested_model
                if model_name not in self._candidate_model_names():
                    raise HTTPException(
                        status_code=404, detail=f"model not found: {model_name}"
                    )
            else:
                # 未指定模型时选择排序后的第一个 ready 模型。
                ready_models = await self._reconcile_and_collect_ready_models()
                if not ready_models:
                    raise HTTPException(status_code=503, detail="No ready PD models")
                model_name = ready_models[0]
            payload = {**payload, "model": model_name}

            # 每个请求只 reconcile 一次 P/D client 集合，避免重复 bootstrap 查询。
            await self._reconcile_worker_clients_for_model(model_name)
            
            # P/D 都必须声明支持当前请求路径，避免把请求转发给不兼容 worker。
            prefill_client_info = self._select_worker_client(
                WorkerType.PREFILL, model_name, api_route
            )
            # 当前 PD 流程固定使用 prefill worker 的 data-parallel rank 0。
            prefill_dp_rank = 0

            decode_client_info = self._select_worker_client(
                WorkerType.DECODE, model_name, api_route
            )
            
            request_id = f"cmpl-{uuid.uuid4().hex}"
            # prefill 与 decode 并行启动；decode 侧按 Mooncake 流程拉取 KV。
            prefill_task = asyncio.create_task(
                self._send_prefill_background(
                    prefill_client_info,
                    prefill_dp_rank,
                    api_route,
                    payload,
                    request_id,
                )
            )
            self._prefill_tasks.add(prefill_task)
            prefill_task.add_done_callback(self._prefill_tasks.discard)

            async def generate_stream():
                """包装 decode 流，并把异常转换成响应内 JSON 错误。"""
                try:
                    async for chunk in self._stream_decode(
                        prefill_client_info,
                        prefill_dp_rank,
                        decode_client_info,
                        api_route,
                        payload,
                        request_id=request_id,
                    ):
                        yield chunk
                except HTTPException as exc:
                    # StreamingResponse 已开始后不能再修改状态码，只能输出错误块。
                    logger.error(
                        "Decode error for %s: HTTP %s - %s",
                        request_id,
                        exc.status_code,
                        exc.detail,
                    )
                    yield json.dumps(
                        {"error": {"message": exc.detail, "code": exc.status_code}}
                    ).encode()
                except Exception as exc:
                    logger.exception("PD stream %s failed", request_id)
                    yield json.dumps(
                        {"error": {"message": str(exc), "code": 500}}
                    ).encode()

            return StreamingResponse(generate_stream(), media_type="application/json")

        @app.post("/v1/completions")
        async def v1_completions(request: Request):
            """处理 OpenAI-compatible completions API。"""
            return await handle_completions("/v1/completions", request)

        @app.post("/chat/completions")
        async def chat_completions(request: Request):
            """处理兼容路径 chat completions API。"""
            return await handle_completions("/chat/completions", request)

        @app.post("/v1/chat/completions")
        async def v1_chat_completions(request: Request):
            """处理 OpenAI-compatible chat completions API。"""
            return await handle_completions("/v1/chat/completions", request)

        return app

    async def _reconcile_and_collect_ready_models(self) -> list[str]:
        """reconcile 各模型 client，并返回满足最小 P/D 副本数的模型。"""
        ready_models: list[str] = []
        candidate_model_names = self._candidate_model_names()
        # discovery 中已消失的模型不再保留 client；有存量请求的连接延迟清理。
        await self.worker_client_pool_manager.retain_models(candidate_model_names)
        for model_name in candidate_model_names:
            # client reconcile 可能触发 bootstrap 查询，因此这里需要 async 等待。
            await self._reconcile_worker_clients_for_model(model_name)
            if self._has_minimum_routable_workers(model_name):
                ready_models.append(model_name)
        return ready_models


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
