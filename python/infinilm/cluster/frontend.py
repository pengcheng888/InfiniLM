"""Unified OpenAI API frontend for a multi-worker InfiniLM cluster."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)

from .config import FrontendOptions, parse_frontend_args
from .metrics import FrontendMetrics
from .monitoring import MONITORING_HTML, monitoring_snapshot
from .registry import FileServiceRegistry, FileServiceWatcher
from .router import ClusterRouter, NoWorkersAvailable

logger = logging.getLogger(__name__)


class ClusterFrontend:
    """多 worker InfiniLM 集群的统一 OpenAI API frontend。"""

    def __init__(self, options: FrontendOptions) -> None:
        """根据配置初始化 discovery、watcher、router 和指标对象。"""
        self.options = options
        self.registry = FileServiceRegistry(
            options.file_kv,
            lease_seconds=options.registry_ttl,
        )
        self.watcher = FileServiceWatcher(self.registry)
        self.router = ClusterRouter(
            self.watcher,
            connect_timeout=options.connect_timeout,
            request_timeout=options.request_timeout,
            request_plane=options.request_plane,
            router_mode=options.router_mode,
        )
        self.metrics = FrontendMetrics(router=self.router.metrics)

    def _resolve_model(self, requested: str | None) -> tuple[str, list] | None:
        """解析请求中的模型名，并返回该模型可用的 worker 列表。

        如果调用方没有指定模型，则使用第一个可用 worker 的 served_model_name。
        返回 ``None`` 表示没有找到可服务的 worker。
        """
        workers = self.router.available_workers(requested)
        if workers:
            return requested or workers[0].served_model_name, workers
        if not requested and self.watcher.workers:
            ready_workers = [worker for worker in self.watcher.workers if worker.state == "ready"]
            if ready_workers:
                return ready_workers[0].served_model_name, ready_workers
        return None

    def create_app(self) -> FastAPI:
        """创建对外提供 OpenAI API 和运维接口的 FastAPI 应用。"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """管理 frontend 生命周期：等待初始 worker、启动 watcher、退出清理。"""
            await self.watcher.refresh(self.options.namespace)
            while (
                len(self.watcher.workers)
                < self.options.router_min_initial_workers
            ):
                logger.info(
                    "Waiting for %d workers; currently %d",
                    self.options.router_min_initial_workers,
                    len(self.watcher.workers),
                )
                await asyncio.sleep(
                    max(0.1, self.options.health_interval)
                )
                await self.watcher.refresh(self.options.namespace)

            self.watcher.start(
                self.options.namespace,
                interval=max(0.1, self.options.health_interval),
            )
            try:
                yield
            finally:
                await self.watcher.stop()
                await self.router.aclose()

        app = FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health():
            """返回 frontend 自身健康状态和当前发现的 worker 数量。"""
            worker_count = len(self.watcher.workers)
            ready_count = len(self.router.available_workers())
            return {
                "status": "healthy",
                "namespace": self.options.namespace,
                "workers": worker_count,
                "ready": ready_count >= self.options.router_min_initial_workers,
                "ready_workers": ready_count,
            }

        @app.get("/v1/models")
        async def models():
            """以 OpenAI models API 格式返回当前可用模型名列表。"""
            names = sorted({worker.served_model_name for worker in self.watcher.workers})
            return {
                "object": "list",
                "data": [
                    {
                        "id": name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "infinilm",
                    }
                    for name in names
                ],
            }

        @app.get("/v1/models/{model_name}/ready")
        async def model_ready(model_name: str):
            """查询指定模型是否已经满足最小 worker 副本数。"""
            workers = self.router.available_workers(model_name)
            ready = len(workers) >= self.options.router_min_initial_workers
            return {
                "ready": ready,
                "model": model_name,
                "namespace": self.options.namespace,
                "worker_type": "aggregated",
                "workers": len(workers),
                "worker_ids": [worker.instance_id for worker in workers],
            }

        @app.get("/metrics")
        async def metrics():
            """输出 Prometheus 文本格式的路由和 worker 指标。"""
            self.router.update_worker_metrics(self.watcher.workers)
            return PlainTextResponse(
                self.metrics.render(len(self.watcher.workers), self.watcher.workers),
                media_type="text/plain; version=0.0.4",
            )

        @app.get("/monitoring", response_class=HTMLResponse)
        async def monitoring_page():
            """返回内置 worker 监控页面。"""
            return HTMLResponse(
                MONITORING_HTML,
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/api/monitoring")
        async def monitoring_api():
            """返回监控页面使用的集群状态 JSON 快照。"""
            return monitoring_snapshot(
                options=self.options,
                watcher=self.watcher,
                router=self.router,
            )

        @app.post("/v1/chat/completions")
        @app.post("/chat/completions")
        async def chat_completions(request: Request):
            """统一处理 OpenAI-compatible chat completions 请求。"""
            try:
                payload = await request.json()
            except Exception:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                )
            if not isinstance(payload, dict) or not payload.get("messages"):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "messages is required",
                            "type": "invalid_request_error",
                        }
                    },
                )

            requested_model = payload.get("model")
            resolved = self._resolve_model(requested_model)
            if resolved is None:
                return JSONResponse(
                    status_code=503 if not requested_model else 404,
                    content={
                        "error": {
                            "message": f"model is not ready: {requested_model or '<default>'}",
                            "type": "service_unavailable" if not requested_model else "invalid_request_error",
                        }
                    },
                )
            model_name, _ = resolved
            payload = {**payload, "model": model_name}
            request_id = f"cmpl-{uuid.uuid4().hex}"

            if payload.get("stream", False):
                return StreamingResponse(
                    self._stream_response(request_id, model_name, payload),
                    media_type="text/event-stream",
                )

            try:
                response = await self._non_stream_response(request_id, model_name, payload)
            except NoWorkersAvailable as exc:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": str(exc), "type": "service_unavailable"}},
                )
            except Exception as exc:
                logger.exception("Chat request %s failed", request_id)
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": str(exc), "type": "server_error"}},
                )
            return JSONResponse(content=response)

        return app

    async def _stream_response(
        self,
        request_id: str,
        model_name: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[str]:
        """把 router 输出的内部帧转换为 OpenAI streaming SSE 响应。"""

        def sse(chunk: dict[str, Any]) -> str:
            """将一个 OpenAI chunk 编码为一行 SSE data 事件。"""
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        first = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        try:
            yield sse(first)
            async for frame in self.router.stream_chat(payload, model_name=model_name):
                if frame.get("type") != "delta":
                    continue
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": frame.get("content", "")},
                            "finish_reason": None,
                        }
                    ],
                }
                yield sse(chunk)
            yield sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
        except Exception as exc:
            logger.exception("Streaming request %s failed", request_id)
            yield sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error",
                        }
                    ],
                    "error": {"message": str(exc)},
                }
            )
        yield "data: [DONE]\n\n"

    async def _non_stream_response(
        self,
        request_id: str,
        model_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """聚合内部 delta/finish 帧，生成 OpenAI 非流式 chat completion。"""
        content_parts: list[str] = []
        finish_content: str | None = None
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        finish_reason = "stop"
        async for frame in self.router.stream_chat(payload, model_name=model_name):
            if frame.get("type") == "delta":
                content_parts.append(frame.get("content", ""))
            elif frame.get("type") == "finish":
                if isinstance(frame.get("usage"), dict):
                    usage = frame["usage"]
                finish_reason = frame.get("finish_reason") or "stop"
                if frame.get("content") is not None:
                    finish_content = str(frame["content"])
        content = finish_content if finish_content is not None else "".join(content_parts)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }


def main(argv: list[str] | None = None) -> None:
    options = parse_frontend_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    frontend = ClusterFrontend(options)
    display_host = "127.0.0.1" if options.http_host == "0.0.0.0" else options.http_host
    logger.info(
        "Starting InfiniLM cluster frontend at %s:%s namespace=%s file_kv=%s request_plane=%s router_mode=%s",
        options.http_host,
        options.http_port,
        options.namespace,
        options.file_kv,
        options.request_plane,
        options.router_mode,
    )
    logger.info(
        "Monitoring page: http://%s:%s/monitoring",
        display_host,
        options.http_port,
    )
    
    uvicorn.run(
        frontend.create_app(),
        host=options.http_host,
        port=options.http_port,
    )


if __name__ == "__main__":
    '''
     -> main()
     -> parse_frontend_args()
     -> logging.basicConfig()
     -> ClusterFrontend(options)
        -> FileServiceRegistry
        -> FileServiceWatcher
        -> ClusterRouter
        -> HttpWorkerClientFactory
        -> FrontendMetrics
     -> logger.info()
     -> uvicorn.run(frontend.create_app())
        -> FastAPI lifespan
           -> 读取 file discovery
           -> 等待最小 worker 数
           -> 启动 watcher 周期任务
           -> 注册 API 路由
           -> 对外提供 OpenAI API
        -> 请求 /v1/chat/completions
           -> ClusterRouter
           -> HttpWorkerClient
           -> worker HTTP API
           -> AsyncLLMEngine
        -> 退出时清理 watcher 和 httpx 连接池
    '''
    main()
