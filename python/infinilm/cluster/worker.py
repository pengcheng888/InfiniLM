"""Cluster-aware entrypoint for one aggregated InfiniLM inference worker."""

from __future__ import annotations

import asyncio
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

from .config import WorkerOptions, parse_worker_cluster_args, remove_cluster_worker_args
from .registry import FileServiceRegistry
from .worker_discovery import WorkerDiscoveryService
from .worker_traffic import WorkerTrafficMetrics

logger = logging.getLogger(__name__)


class DrainingUvicornServer(uvicorn.Server):
    """Uvicorn server that advertises DRAINING before it stops accepting."""

    def __init__(
        self,
        config: uvicorn.Config,
        *,
        drain_grace_seconds: float,
    ) -> None:
        super().__init__(config)
        self.drain_grace_seconds = max(0.0, drain_grace_seconds)
        self.on_drain_start = None
        self._exit_requested = False

    def handle_exit(self, sig, frame) -> None:
        if self._exit_requested:
            super().handle_exit(sig, frame)
            return

        self._exit_requested = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.should_exit = True
            return
        loop.create_task(self._drain_and_exit(), name="infinilm-worker-drain")

    async def _drain_and_exit(self) -> None:
        try:
            if self.on_drain_start is not None:
                await self.on_drain_start()
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
        registry: FileServiceRegistry,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cluster_options = cluster_options
        self.registry = registry
        self.worker_discovery: WorkerDiscoveryService | None = None
        self.uvicorn_server: DrainingUvicornServer | None = None
        self.worker_traffic = WorkerTrafficMetrics()
        if cluster_options.served_model_name:
            self.model_id = cluster_options.served_model_name

    async def _chat(self, request_id: str, data: dict, http_request):
        """Count non-streaming traffic before delegating to InferenceServer."""
        self.worker_traffic.begin(streaming=False)
        canceled = False
        failed = False
        try:
            response = await super()._chat(request_id, data, http_request)
            failed = int(getattr(response, "status_code", 200)) >= 400
            return response
        except asyncio.CancelledError:
            canceled = True
            raise
        except Exception:
            failed = True
            raise
        finally:
            self.worker_traffic.finish(canceled=canceled, failed=failed)

    async def _stream_chat(self, request_id: str, data: dict, http_request):
        """Count streaming traffic for the full response-generator lifetime."""
        self.worker_traffic.begin(streaming=True)
        canceled = False
        failed = False
        try:
            async for chunk in super()._stream_chat(request_id, data, http_request):
                if (
                    "finish_reason" in chunk
                    and ('"error"' in chunk or '"timeout"' in chunk)
                ):
                    failed = True
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
            self.worker_traffic.finish(canceled=canceled, failed=failed)

    def _create_discovery_service(
        self,
        engine: object,
    ) -> WorkerDiscoveryService:
        """在 engine ready 后构造 worker 的 file discovery 服务。"""
        served_model_name = (
            self.cluster_options.served_model_name
            or os.path.basename(os.path.normpath(self.model_path))
            or "model"
        )
        record = self.registry.new_record(
            namespace=self.cluster_options.namespace,
            component=self.cluster_options.component,
            endpoint=self.cluster_options.endpoint,
            worker_type=self.cluster_options.worker_type,
            model=self.model_path,
            served_model_name=served_model_name,
            host=self.cluster_options.advertise_host,
            port=self.port,
            engine_healthy=True,
            transport=self.cluster_options.request_plane,
            metadata={
                "request_plane": self.cluster_options.request_plane,
                "http_host": self.host,
                "http_port": self.port,
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "tp": self.tensor_parallel_size,
                "max_batch_size": self.max_batch_size,
            },
        )
        return WorkerDiscoveryService(
            engine=engine,
            registry=self.registry,
            record=record,
            heartbeat_interval=self.cluster_options.heartbeat_interval,
            traffic_metrics=self.worker_traffic,
        )

    def _create_app(self) -> FastAPI:
        app = super()._create_app()
        engine_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def cluster_lifespan(app: FastAPI):
            async with engine_lifespan(app):
                discovery = self._create_discovery_service(self.engine)
                self.worker_discovery = discovery
                if self.uvicorn_server is not None:
                    self.uvicorn_server.on_drain_start = discovery.start_draining
                await discovery.start()
                try:
                    yield
                finally:
                    await discovery.stop()
                    self.worker_discovery = None

        app.router.lifespan_context = cluster_lifespan
        return app


def _build_server(argv: list[str]) -> ClusterInferenceServer:
    cluster_options = parse_worker_cluster_args(argv)
    inference_argv = remove_cluster_worker_args(argv)
    cfg = BaseConfig(inference_argv)

    if cfg.pp > 1:
        raise ValueError("the aggregated cluster worker currently requires --pp 1")

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
        kv_transfer_config = parse_kv_transfer_config(cfg.kv_transfer_config)

    registry = FileServiceRegistry(
        cluster_options.file_kv,
        lease_seconds=cluster_options.registry_ttl,
    )
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


def _run_server(server: ClusterInferenceServer) -> None:
    config = uvicorn.Config(
        server._create_app(),
        host=server.host,
        port=server.port,
    )
    uvicorn_server = DrainingUvicornServer(
        config,
        drain_grace_seconds=server.cluster_options.drain_grace_seconds,
    )
    server.uvicorn_server = uvicorn_server
    uvicorn_server.run()


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    server = _build_server(args)
    setup_logging(server.log_level)
    logger.info(
        "Starting aggregated InfiniLM worker: namespace=%s request_plane=%s endpoint=%s:%s",
        server.cluster_options.namespace,
        server.cluster_options.request_plane,
        server.cluster_options.advertise_host,
        server.port,
    )
    _run_server(server)


if __name__ == "__main__":
    main()
