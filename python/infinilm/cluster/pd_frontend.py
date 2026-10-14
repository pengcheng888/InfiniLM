"""Discovery-aware OpenAI frontend for a prefill/decode separated cluster."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

from ..server.mooncake_proxy_server import (
    maybe_wrap_ipv6_address,
    send_request,
    stream_response,
)
from .config import _add_cluster_arguments
from .monitoring import PD_MONITORING_HTML, pd_monitoring_snapshot
from .registry import FileServiceRegistry, FileServiceWatcher, WorkerRecord
from .router import worker_load

logger = logging.getLogger(__name__)

ROLE_CAPABILITIES = {
    "prefill": "internal-prefill",
    "decode": "openai-chat",
}


@dataclass(frozen=True)
class PdFrontendOptions:
    namespace: str
    file_kv: Path
    router_mode: str
    http_host: str
    http_port: int
    connect_timeout: float
    request_timeout: float
    health_interval: float
    registry_ttl: float
    heartbeat_interval: float
    request_plane: str = "http"
    min_prefill_workers: int = 1
    min_decode_workers: int = 1


def parse_pd_frontend_args(argv: list[str] | None = None) -> PdFrontendOptions:
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
    parser.add_argument("--connect-timeout", type=float, default=3.0)
    parser.add_argument("--request-timeout", type=float, default=1000.0)
    parser.add_argument("--health-interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    return PdFrontendOptions(
        namespace=args.namespace,
        file_kv=Path(args.file_kv),
        router_mode=args.router_mode,
        http_host=args.http_host,
        http_port=args.http_port,
        connect_timeout=args.connect_timeout,
        request_timeout=args.request_timeout,
        health_interval=args.health_interval,
        registry_ttl=args.registry_ttl,
        heartbeat_interval=args.heartbeat_interval,
        request_plane=args.request_plane,
        min_prefill_workers=max(1, args.min_prefill_workers),
        min_decode_workers=max(1, args.min_decode_workers),
    )


class PdClusterFrontend:
    """Routes one request through one prefill worker and one decode worker."""

    def __init__(self, options: PdFrontendOptions) -> None:
        self.options = options
        self.registry = FileServiceRegistry(
            options.file_kv,
            lease_seconds=options.registry_ttl,
        )
        self.watcher = FileServiceWatcher(
            self.registry,
            worker_types=("prefill", "decode"),
        )
        self.clients: dict[str, dict[str, dict[str, Any]]] = {
            "prefill": {},
            "decode": {},
        }
        self.inflight: dict[str, int] = defaultdict(int)
        self.cursors: dict[str, int] = {"prefill": 0, "decode": 0}
        self.request_counters: dict[str, int] = {
            "prefill": 0,
            "decode": 0,
        }
        self._prefill_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _role_workers(
        workers: list[WorkerRecord],
        role: str,
        model_name: str | None = None,
        capability: str | None = None,
    ) -> list[WorkerRecord]:
        required_capability = capability or ROLE_CAPABILITIES.get(role)
        result = [
            worker
            for worker in workers
            if worker.worker_type == role
            and worker.transport == "http"
            and worker.state == "ready"
            and worker.engine_healthy
            and (model_name is None or worker.served_model_name == model_name)
            and (
                required_capability is None
                or required_capability
                in (worker.metadata.get("capabilities") or [])
            )
        ]
        return sorted(result, key=lambda worker: worker.instance_id)

    def _ready_models(self) -> list[str]:
        workers = self.watcher.workers
        prefill_models = {
            worker.served_model_name
            for worker in self._role_workers(workers, "prefill")
        }
        decode_models = {
            worker.served_model_name
            for worker in self._role_workers(workers, "decode")
        }
        return sorted(prefill_models & decode_models)

    def _is_model_ready(self, model_name: str) -> bool:
        workers = self.watcher.workers
        for role, minimum in (
            ("prefill", self.options.min_prefill_workers),
            ("decode", self.options.min_decode_workers),
        ):
            record_ids = {
                worker.instance_id
                for worker in self._role_workers(workers, role, model_name)
            }
            if len(record_ids & set(self.clients[role])) < minimum:
                return False
        return True

    @staticmethod
    def _http_url(host: str, port: int) -> str:
        return f"http://{maybe_wrap_ipv6_address(host)}:{port}"

    async def _create_client_info(
        self,
        record: WorkerRecord,
    ) -> dict[str, Any]:
        base_url = self._http_url(record.host, record.port)
        client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=self.options.connect_timeout,
                read=self.options.request_timeout,
                write=self.options.connect_timeout,
                pool=self.options.connect_timeout,
            ),
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=None,
            ),
        )
        info: dict[str, Any] = {
            "client": client,
            "url": base_url,
            "record": record,
            "instance_id": record.instance_id,
        }

        if record.worker_type == "prefill":
            pd_metadata = record.metadata.get("pd")
            if not isinstance(pd_metadata, dict):
                await client.aclose()
                raise ValueError(f"prefill worker {record.instance_id} has no PD metadata")
            bootstrap_host = str(pd_metadata.get("bootstrap_host") or record.host)
            bootstrap_port = int(pd_metadata.get("bootstrap_port", 0))
            info["bootstrap_addr"] = self._http_url(bootstrap_host, bootstrap_port)
            try:
                response = await client.get(f"{info['bootstrap_addr']}/query")
                response.raise_for_status()
                engines = response.json()
            except Exception:
                await client.aclose()
                raise
            if not isinstance(engines, dict) or not engines:
                await client.aclose()
                raise ValueError(
                    f"prefill worker {record.instance_id} bootstrap returned no engines"
                )
            info["dp_engine_id"] = {
                int(rank): engine["engine_id"]
                for rank, engine in engines.items()
                if isinstance(engine, dict) and "engine_id" in engine
            }
            info["dp_size"] = len(info["dp_engine_id"])
            if not info["dp_engine_id"]:
                await client.aclose()
                raise ValueError(
                    f"prefill worker {record.instance_id} bootstrap returned no engines"
                )

        return info

    async def _sync_clients(self, model_name: str) -> None:
        for role in ("prefill", "decode"):
            records = self._role_workers(
                self.watcher.workers,
                role,
                model_name,
            )
            desired = {record.instance_id: record for record in records}
            clients = self.clients[role]

            for instance_id, info in list(clients.items()):
                if instance_id in desired:
                    info["record"] = desired[instance_id]
                    continue
                if self.inflight[instance_id] == 0:
                    await info["client"].aclose()
                    del clients[instance_id]

            for instance_id, record in desired.items():
                if instance_id in clients:
                    continue
                try:
                    clients[instance_id] = await self._create_client_info(record)
                except Exception as exc:
                    logger.warning(
                        "Could not initialize %s worker %s: %s",
                        role,
                        instance_id,
                        exc,
                    )

    def _select_record(self, records: list[WorkerRecord]) -> WorkerRecord:
        if self.options.router_mode == "least-loaded":
            def score(record: WorkerRecord) -> tuple[float, str]:
                load = worker_load(record)
                return (
                    0.90 * self.inflight[record.instance_id]
                    + 0.05 * load.waiting_requests
                    + 0.03 * load.running_requests / load.max_batch_size
                    + 0.02 * load.kv_usage,
                    record.instance_id,
                )

            return min(records, key=score)

        cursor = self.cursors["prefill" if records[0].worker_type == "prefill" else "decode"]
        selected = records[cursor % len(records)]
        role = selected.worker_type
        self.cursors[role] = (self.cursors[role] + 1) % max(1, len(records))
        return selected

    async def _choose_client(
        self,
        role: str,
        model_name: str,
        capability: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        await self._sync_clients(model_name)
        records = self._role_workers(
            self.watcher.workers,
            role,
            model_name,
            capability,
        )
        if not records:
            raise HTTPException(
                status_code=503,
                detail=f"No ready {role} workers for model {model_name}",
            )
        record = self._select_record(records)
        info = self.clients[role].get(record.instance_id)
        if info is None:
            raise HTTPException(
                status_code=503,
                detail=f"{role} worker {record.instance_id} is not ready",
            )
        return info, 0

    def _begin_request(self, info: dict[str, Any]) -> None:
        instance_id = info["instance_id"]
        self.inflight[instance_id] += 1
        role = info["record"].worker_type
        self.request_counters[role] += 1

    def _finish_request(self, info: dict[str, Any]) -> None:
        instance_id = info["instance_id"]
        self.inflight[instance_id] -= 1
        if self.inflight[instance_id] <= 0:
            self.inflight[instance_id] = 0

    async def _send_prefill(
        self,
        info: dict[str, Any],
        dp_rank: int,
        api: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> None:
        self._begin_request(info)
        try:
            await send_request(info, dp_rank, api, payload.copy(), request_id)
        except Exception:
            logger.exception("Prefill request %s failed", request_id)
        finally:
            self._finish_request(info)

    async def _stream_decode(
        self,
        prefill_info: dict[str, Any],
        prefill_rank: int,
        decode_info: dict[str, Any],
        api: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> AsyncIterator[bytes]:
        self._begin_request(decode_info)
        try:
            async for chunk in stream_response(
                prefill_info,
                prefill_rank,
                decode_info,
                api,
                payload,
                request_id=request_id,
            ):
                yield chunk
        finally:
            self._finish_request(decode_info)

    def create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.watcher.refresh(self.options.namespace)
            self.watcher.start(
                self.options.namespace,
                interval=max(0.1, self.options.health_interval),
            )
            logger.info(
                "PD frontend is listening; prefill=%d decode=%d models=%s",
                len(self._role_workers(self.watcher.workers, "prefill")),
                len(self._role_workers(self.watcher.workers, "decode")),
                self._ready_models(),
            )
            try:
                yield
            finally:
                await self.watcher.stop()
                for task in self._prefill_tasks:
                    task.cancel()
                if self._prefill_tasks:
                    await asyncio.gather(
                        *self._prefill_tasks,
                        return_exceptions=True,
                    )
                for role_clients in self.clients.values():
                    for info in role_clients.values():
                        await info["client"].aclose()

        app = FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health():
            ready_models = await self._ready_model_names()
            workers = self.watcher.workers
            return {
                "status": "healthy",
                "namespace": self.options.namespace,
                "ready": bool(ready_models),
                "models": ready_models,
                "prefill_workers": len(self._role_workers(workers, "prefill")),
                "decode_workers": len(self._role_workers(workers, "decode")),
            }

        @app.get("/metrics")
        async def metrics():
            lines = [
                "# HELP infinilm_pd_workers Ready workers by role.",
                "# TYPE infinilm_pd_workers gauge",
            ]
            for role in ("prefill", "decode"):
                lines.append(
                    f'infinilm_pd_workers{{role="{role}"}} '
                    f'{len(self._role_workers(self.watcher.workers, role))}'
                )
            lines.extend(
                [
                    "# HELP infinilm_pd_frontend_requests_total Requests selected by role.",
                    "# TYPE infinilm_pd_frontend_requests_total counter",
                ]
            )
            for role, count in self.request_counters.items():
                lines.append(f'infinilm_pd_frontend_requests_total{{role="{role}"}} {count}')
            lines.extend(
                [
                    "# HELP infinilm_pd_frontend_inflight_requests Active requests by worker.",
                    "# TYPE infinilm_pd_frontend_inflight_requests gauge",
                ]
            )
            for role in ("prefill", "decode"):
                for worker in self._role_workers(self.watcher.workers, role):
                    lines.append(
                        f'infinilm_pd_frontend_inflight_requests{{role="{role}",'
                        f'worker_id="{worker.instance_id}"}} '
                        f'{self.inflight[worker.instance_id]}'
                    )
            return PlainTextResponse(
                "\n".join(lines) + "\n",
                media_type="text/plain; version=0.0.4",
            )

        @app.get("/v1/models")
        async def models():
            ready_models = await self._ready_model_names()
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
            return HTMLResponse(
                PD_MONITORING_HTML,
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/api/monitoring")
        async def monitoring_api():
            return pd_monitoring_snapshot(self)

        async def handle_completions(api: str, request: Request):
            try:
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise HTTPException(status_code=400, detail="Request body must be an object")

            requested_model = payload.get("model")
            if requested_model:
                model_name = requested_model
                if model_name not in self._ready_models():
                    raise HTTPException(status_code=404, detail=f"model not found: {model_name}")
            else:
                ready_models = await self._ready_model_names()
                if not ready_models:
                    raise HTTPException(status_code=503, detail="No ready PD models")
                model_name = ready_models[0]
            payload = {**payload, "model": model_name}

            decode_capability = (
                "openai-chat" if api.endswith("chat/completions") else "openai-completions"
            )
            prefill_info, prefill_rank = await self._choose_client("prefill", model_name)
            decode_info, _ = await self._choose_client(
                "decode",
                model_name,
                decode_capability,
            )
            request_id = f"cmpl-{uuid.uuid4().hex}"
            prefill_task = asyncio.create_task(
                self._send_prefill(
                    prefill_info,
                    prefill_rank,
                    api,
                    payload,
                    request_id,
                )
            )
            self._prefill_tasks.add(prefill_task)
            prefill_task.add_done_callback(self._prefill_tasks.discard)

            async def generate_stream():
                try:
                    async for chunk in self._stream_decode(
                        prefill_info,
                        prefill_rank,
                        decode_info,
                        api,
                        payload,
                        request_id=request_id,
                    ):
                        yield chunk
                except HTTPException as exc:
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
            return await handle_completions("/v1/completions", request)

        @app.post("/chat/completions")
        async def chat_completions(request: Request):
            return await handle_completions("/chat/completions", request)

        @app.post("/v1/chat/completions")
        async def v1_chat_completions(request: Request):
            return await handle_completions("/v1/chat/completions", request)

        return app

    async def _ready_model_names(self) -> list[str]:
        ready_models: list[str] = []
        for model_name in self._ready_models():
            await self._sync_clients(model_name)
            if self._is_model_ready(model_name):
                ready_models.append(model_name)
        return ready_models


def main(argv: list[str] | None = None) -> None:
    options = parse_pd_frontend_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    frontend = PdClusterFrontend(options)
    display_host = "127.0.0.1" if options.http_host == "0.0.0.0" else options.http_host
    logger.info(
        "Starting InfiniLM PD frontend at %s:%s namespace=%s file_kv=%s router_mode=%s",
        options.http_host,
        options.http_port,
        options.namespace,
        options.file_kv,
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
