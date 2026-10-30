"""
Inference Server - HTTP API server for LLM inference.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from infinilm.base_config import BaseConfig
from infinilm.config import KVTransferConfig
from infinilm.llm import AsyncLLMEngine, FinishReason, SamplingParams
from infinilm.llm.output_timing import (
    DisabledOutputTimingCollector,
    OutputTimingService,
)
from infinilm.llm.request import RequestEntranceTiming, monotonic_ns
from infinilm.llm.request_timing import RequestTimingConfig
from infinilm.moe_config import configure_moe_ep_backend

logger = logging.getLogger(__name__)

DEFAULT_STREAM_TIMEOUT = 100.0
DEFAULT_REQUEST_TIMEOUT = 1000.0
DEFAULT_DISCONNECT_CHECK_TOKEN_INTERVAL = 32
DEFAULT_SSE_FLUSH_TOKEN_INTERVAL = 4
INTERNAL_REQUEST_HEADER = "X-InfiniLM-Internal"


def sse_flush_tokens_from_environment() -> int:
    raw_value = os.environ.get(
        "INFINILM_SSE_FLUSH_TOKENS",
        str(DEFAULT_SSE_FLUSH_TOKEN_INTERVAL),
    )
    try:
        interval = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"INFINILM_SSE_FLUSH_TOKENS must be an integer: {raw_value!r}"
        ) from exc
    if interval not in {1, 4, 8}:
        raise RuntimeError("INFINILM_SSE_FLUSH_TOKENS must be 1, 4, or 8")
    return interval


SSE_FLUSH_TOKEN_INTERVAL = sse_flush_tokens_from_environment()


def disconnect_check_tokens_from_environment() -> int:
    raw_value = os.environ.get(
        "INFINILM_DISCONNECT_CHECK_TOKENS",
        str(DEFAULT_DISCONNECT_CHECK_TOKEN_INTERVAL),
    )
    try:
        interval = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"INFINILM_DISCONNECT_CHECK_TOKENS must be an integer: {raw_value!r}"
        ) from exc
    if interval not in {32, 1024}:
        raise RuntimeError("INFINILM_DISCONNECT_CHECK_TOKENS must be 32 or 1024")
    return interval


DISCONNECT_CHECK_TOKEN_INTERVAL = disconnect_check_tokens_from_environment()


def chunk_json(
    id_,
    content=None,
    role=None,
    finish_reason=None,
    model: str = "unknown",
    cache: dict | None = None,
):
    """Generate JSON chunk for streaming response."""
    delta = {}
    if content:
        delta["content"] = content
    if role:
        delta["role"] = role
    chunk = {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "text": content,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if cache is not None:
        chunk["cache"] = cache
    return chunk


def completion_json(
    id_,
    content,
    role="assistant",
    finish_reason="stop",
    model: str = "unknown",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cache: dict | None = None,
):
    """Generate JSON response for non-streaming completion."""
    response = {
        "id": id_,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": role,
                    "content": content,
                },
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    if cache is not None:
        response["cache"] = cache
    return response


class TimedStreamingResponse(StreamingResponse):
    """StreamingResponse with a diagnostic wrapper around ASGI body sends."""

    def __init__(self, *args, timing_collector=None, timing_service=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._timing_collector = timing_collector
        self._timing_service = timing_service

    async def __call__(self, scope, receive, send):
        if self._timing_collector is None or not self._timing_collector.enabled:
            await super().__call__(scope, receive, send)
            return

        async def timed_send(message):
            started_at_ns = monotonic_ns()
            await send(message)
            self._timing_collector.record_asgi_body_send(
                started_at_ns,
                monotonic_ns(),
                len(message.get("body", b"")),
                message.get("type", ""),
            )

        try:
            await super().__call__(scope, receive, timed_send)
        finally:
            payload = self._timing_collector.finish()
            if self._timing_service is not None:
                self._timing_service.record_finished_collector(payload)


class InferenceServer:
    """HTTP server for LLM inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_stage: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        moe_ep_backend: str = "disabled",
        moe_ep_size: int = 1,
        use_legacy_moe: bool = False,
        cache_type: str = "paged",
        max_tokens: int = 4096,
        max_batch_size: int = 16,
        gpu_memory_utilization: float = 0.85,
        block_size: int = 256,
        max_cache_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_graph: bool = False,
        enable_workspace: bool = True,
        max_num_batched_tokens: int = 9216,
        attn_backend: str = "default",
        use_mla: bool = False,
        skip_load: bool = False,
        weight_load_mode: str = "async",
        ignore_eos: bool = False,
        kv_transfer_config: Optional[KVTransferConfig] = None,
        enable_prefix_caching: bool = True,
        pre_transpose: bool = False,
    ):
        """Initialize inference server.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            moe_ep_backend: MoE expert-parallel backend.
            moe_ep_size: MoE expert-parallel size.
            use_legacy_moe: Whether to use the legacy Qwen3 MoE implementation.
            cache_type: Cache type ('paged' or 'static').
            max_tokens: Default maximum tokens to generate.
            max_batch_size: Maximum batch size for inference (only for paged cache).
            gpu_memory_utilization: Fraction of device memory used to size KV cache.
            block_size: Size of each KV cache block (only for paged cache).
            max_cache_len: Maximum sequence length (only for static cache).
            temperature: Default sampling temperature.
            top_p: Default top-p sampling parameter.
            top_k: Default top-k sampling parameter.
            host: Server host address.
            port: Server port number.
            enable_graph: Whether to enable graph compiling.
            enable_workspace: Whether to preallocate reusable inference activations.
            max_num_batched_tokens: Maximum scheduler batch and workspace token count.
            attn_backend: Attention backend to use ('default', 'flash-attn').
            use_mla: Whether to use DeepSeek V2 MLA attention when supported.
            skip_load: Whether to skip loading model weights.
            weight_load_mode: Weight loading mode across tensor-parallel workers.
            ignore_eos: Whether to ignore EOS tokens during generation.
            kv_transfer_config: Optional configuration for the KV transfer mechanism.
        """
        self.model_path = model_path
        # vLLM-like served model id: directory name of model_path
        self.model_id = os.path.basename(os.path.normpath(model_path)) or "model"
        self.device = device
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_stage = pipeline_parallel_stage
        self.master_addr = master_addr
        self.master_port = master_port
        self.moe_ep_backend = moe_ep_backend
        self.moe_ep_size = moe_ep_size
        self.use_legacy_moe = use_legacy_moe
        self.cache_type = cache_type
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.block_size = block_size
        self.max_cache_len = max_cache_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.host = host
        self.port = port
        self.enable_graph = enable_graph
        self.enable_workspace = enable_workspace
        self.max_num_batched_tokens = max_num_batched_tokens
        self.attn_backend = attn_backend
        self.use_mla = use_mla
        self.skip_load = skip_load
        self.weight_load_mode = weight_load_mode
        self.ignore_eos = ignore_eos
        self.kv_transfer_config = kv_transfer_config
        self.enable_prefix_caching = enable_prefix_caching
        self.pre_transpose = pre_transpose

        self.engine: AsyncLLMEngine = None
        self.output_timing = OutputTimingService()
        self.request_timing_enabled = RequestTimingConfig.from_environment().enabled

    def start(self):
        """Start the HTTP server."""
        app = self._create_app()
        logger.info(f"Starting API Server at {self.host}:{self.port}...")
        uvicorn.run(app, host=self.host, port=self.port)
        logger.info("Inference Server stopped")

    def _create_app(self):
        """Create FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.output_timing.start()
            try:
                self.engine = AsyncLLMEngine(
                    model_path=self.model_path,
                    device=self.device,
                    dtype=self.dtype,
                    tensor_parallel_size=self.tensor_parallel_size,
                    pipeline_parallel_size=self.pipeline_parallel_size,
                    pipeline_parallel_stage=self.pipeline_parallel_stage,
                    master_addr=self.master_addr,
                    master_port=self.master_port,
                    moe_ep_backend=self.moe_ep_backend,
                    moe_ep_size=self.moe_ep_size,
                    use_legacy_moe=self.use_legacy_moe,
                    cache_type=self.cache_type,
                    max_batch_size=self.max_batch_size,
                    max_tokens=self.max_tokens,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    block_size=self.block_size,
                    max_cache_len=self.max_cache_len,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    enable_graph=self.enable_graph,
                    enable_workspace=self.enable_workspace,
                    max_num_batched_tokens=self.max_num_batched_tokens,
                    attn_backend=self.attn_backend,
                    use_mla=self.use_mla,
                    skip_load=self.skip_load,
                    weight_load_mode=self.weight_load_mode,
                    kv_transfer_config=self.kv_transfer_config,
                    enable_prefix_caching=self.enable_prefix_caching,
                    pre_transpose=self.pre_transpose,
                )
                self.engine.start()
                logger.info(f"Engine initialized with model at {self.model_path}")
                logger.info(f"  enable_graph: {self.enable_graph}")
                logger.info(f"  enable_workspace: {self.enable_workspace}")
                yield
            finally:
                if self.engine is not None:
                    self.engine.stop()
                await self.output_timing.stop()

        app = FastAPI(lifespan=lifespan)
        self._register_routes(app)
        self._register_prefill_api_guard(app)
        return app

    def _register_prefill_api_guard(self, app: FastAPI) -> None:
        """Keep producer-only workers off the public OpenAI surface."""
        if (
            self.kv_transfer_config is None
            or self.kv_transfer_config.kv_role != "kv_producer"
        ):
            return

        @app.middleware("http")
        async def protect_prefill_openai_api(request: Request, call_next):
            internal_surface = request.url.path.startswith("/v1/") or (
                request.url.path in {"/chat/completions", "/completions"}
            )
            if internal_surface and request.headers.get(
                INTERNAL_REQUEST_HEADER, ""
            ).lower() not in {"true", "1"}:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": {
                            "message": "prefill worker only accepts internal requests",
                            "type": "permission_error",
                        }
                    },
                )
            return await call_next(request)

    def _register_routes(self, app: FastAPI):
        """Register API routes."""

        # OpenAI-compatible chat completions endpoint.
        # Support both legacy path and OpenAI-style /v1 prefix for proxy/router compatibility.
        @app.post("/chat/completions")
        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            # 该时间代表 worker 应用层收到 HTTP request 的时刻，早于 body
            # 读取、JSON 解析、tokenize 和 InferenceRequest 创建。
            server_received_at_ns = (
                monotonic_ns() if self.request_timing_enabled else None
            )
            entrance_timing = (
                RequestEntranceTiming() if self.request_timing_enabled else None
            )
            try:
                body = await request.body()
                if entrance_timing is not None:
                    entrance_timing.body_read_completed_at_ns = monotonic_ns()
                data = json.loads(body)
                if entrance_timing is not None:
                    entrance_timing.json_parsed_at_ns = monotonic_ns()
            except Exception as e:
                logger.error(f"Failed to parse request JSON: {e}")
                return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

            if not data.get("messages"):
                if not data.get("prompt"):
                    return JSONResponse(
                        content={"error": "No message provided"}, status_code=400
                    )
                else:
                    data["messages"] = [{"role": "user", "content": data.get("prompt")}]

            # Normalize messages to handle multimodal content (list format)
            data["messages"] = data.get("messages", [])

            stream = data.get("stream", False)
            # PD frontend 会把同一个 X-Request-Id 发给 prefill/decode。
            # worker 尊重该 ID，使两台机器上的 timing 文件可以用同一个
            # request_id 关联；普通客户端未传 header 时仍自动生成。
            request_id = (
                request.headers.get("X-Request-Id") or f"cmpl-{uuid.uuid4().hex}"
            )
            if entrance_timing is not None:
                entrance_timing.route_prepared_at_ns = monotonic_ns()

            if stream:
                prestarted_request, prebuilt_sampling_params = (
                    self._start_stream_chat_request(
                        request_id=request_id,
                        data=data,
                        server_received_at_ns=server_received_at_ns,
                        entrance_timing=entrance_timing,
                    )
                )
                timing_collector = self.output_timing.create_collector(request_id)
                return TimedStreamingResponse(
                    self._stream_chat(
                        request_id,
                        data,
                        request,
                        server_received_at_ns,
                        entrance_timing,
                        prestarted_request,
                        prebuilt_sampling_params,
                        output_collector=timing_collector,
                    ),
                    media_type="text/event-stream",
                    timing_collector=timing_collector,
                    timing_service=self.output_timing,
                )
            else:
                timing_collector = self.output_timing.create_collector(request_id)
                response = await self._chat(
                    request_id,
                    data,
                    request,
                    server_received_at_ns,
                    entrance_timing,
                    output_collector=timing_collector,
                )
                payload = timing_collector.finish()
                self.output_timing.record_finished_collector(payload)
                if isinstance(response, JSONResponse):
                    return response
                return JSONResponse(content=response)

        @app.get("/health")
        async def health():
            # Expose engine health so babysitter/registry can treat backend as unhealthy.
            if (
                self.engine is not None
                and hasattr(self.engine, "is_healthy")
                and not self.engine.is_healthy()
            ):
                return JSONResponse(content={"status": "unhealthy"}, status_code=503)
            return {"status": "healthy"}

        def _models_payload():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "infinilm",
                    }
                ],
            }

        # Support both /v1/models (OpenAI) and /models (common legacy) for compatibility.
        @app.get("/v1/models")
        async def list_models():
            return _models_payload()

        @app.get("/models")
        async def list_models_legacy():
            return _models_payload()

    def _normalize_messages(self, messages: list) -> list:
        """Normalize messages to handle multimodal content (list format).

        Converts content from list format [{"type": "text", "text": "..."}]
        to string format for chat template compatibility.
        """
        normalized = []
        for msg in messages:
            if not isinstance(msg, dict):
                normalized.append(msg)
                continue

            content = msg.get("content")
            if isinstance(content, list):
                # Extract text from multimodal content list
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    elif isinstance(part, str):
                        text_parts.append(part)
                # Join all text parts
                normalized_msg = msg.copy()
                normalized_msg["content"] = "".join(text_parts) if text_parts else ""
                normalized.append(normalized_msg)
            else:
                normalized.append(msg)

        return normalized

    def _build_sampling_params(self, data: dict) -> SamplingParams:
        """Build SamplingParams from request data."""
        # Support both:
        # - top-level OpenAI-ish fields: temperature/top_p/top_k/max_tokens/stop
        # - nested dict: sampling_params: { ... }
        sp = data.get("sampling_params") or {}
        if not isinstance(sp, dict):
            sp = {}

        def pick(key: str, default):
            # Priority: explicit top-level field > nested sampling_params > server default
            if key in data and data.get(key) is not None:
                return data.get(key)
            if key in sp and sp.get(key) is not None:
                return sp.get(key)
            return default

        # Accept common alias
        max_tokens = pick("max_tokens", self.max_tokens)
        if max_tokens is None:
            # Some clients use max_new_tokens
            max_tokens = pick("max_new_tokens", self.max_tokens)

        stop = pick("stop", None)
        if isinstance(stop, str):
            stop = [stop]

        return SamplingParams(
            temperature=float(pick("temperature", self.temperature)),
            top_p=float(pick("top_p", self.top_p)),
            top_k=int(pick("top_k", self.top_k)),
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            stop=stop,
            ignore_eos=self.ignore_eos,
        )

    def _start_stream_chat_request(
        self,
        *,
        request_id: str,
        data: dict,
        server_received_at_ns: int | None,
        entrance_timing: RequestEntranceTiming,
    ):
        """Submit preprocessing before Starlette schedules the stream body."""
        if entrance_timing is not None:
            entrance_timing.handler_started_at_ns = monotonic_ns()
        sampling_params = self._build_sampling_params(data)
        prestarted_request = self.engine.start_chat_preprocessing(
            messages=data.get("messages", []),
            request_id=request_id,
            add_generation_prompt=bool(data.get("add_generation_prompt", True)),
            chat_template_kwargs=data.get("chat_template_kwargs") or {},
            server_received_at_ns=server_received_at_ns,
            entrance_timing=entrance_timing,
        )
        return prestarted_request, sampling_params

    async def _stream_chat(
        self,
        request_id: str,
        data: dict,
        http_request: Request,
        server_received_at_ns: int | None = None,
        entrance_timing: RequestEntranceTiming | None = None,
        prestarted_request=None,
        prebuilt_sampling_params: SamplingParams | None = None,
        output_collector=None,
    ):
        """Handle streaming chat request."""
        req = None
        _abort_reason = FinishReason.CANCELED
        received_token_outputs = 0
        pending_token_texts: list[str] = []
        pending_token_count = 0
        has_flushed_token_text = False
        if output_collector is None:
            timing_service = getattr(self, "output_timing", None)
            output_collector = (
                timing_service.create_collector(request_id)
                if timing_service is not None
                else DisabledOutputTimingCollector()
            )
        output_timing_enabled = output_collector.enabled
        output_timing_enabled = output_collector.enabled

        def render_sse_chunk(payload: dict) -> str:
            json_started_at_ns = monotonic_ns() if output_timing_enabled else None
            chunk = json.dumps(payload, ensure_ascii=False)
            frame = f"data: {chunk}\n\n"
            if output_timing_enabled:
                output_collector.record_sse_json(
                    json_started_at_ns,
                    monotonic_ns(),
                    len(frame.encode("utf-8")),
                )
            return frame

        def take_pending_sse_chunk() -> str | None:
            nonlocal pending_token_texts, pending_token_count
            if not pending_token_texts:
                return None
            content = "".join(pending_token_texts)
            pending_token_texts = []
            pending_token_count = 0
            chunk = render_sse_chunk(
                chunk_json(
                    request_id,
                    content=content,
                    model=self.model_id,
                )
            )
            return chunk

        if (
            entrance_timing is not None
            and entrance_timing.handler_started_at_ns is None
        ):
            entrance_timing.handler_started_at_ns = monotonic_ns()

        try:
            messages = data.get("messages", [])
            sampling_params = prebuilt_sampling_params or self._build_sampling_params(
                data
            )

            if prestarted_request is None:
                req = await self.engine.add_chat_request_async(
                    messages=messages,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    request_data=data,
                    add_generation_prompt=bool(data.get("add_generation_prompt", True)),
                    chat_template_kwargs=data.get("chat_template_kwargs") or {},
                    server_received_at_ns=server_received_at_ns,
                    entrance_timing=entrance_timing,
                )
            else:
                req = await self.engine.finish_chat_preprocessing(
                    prestarted_request,
                    sampling_params=sampling_params,
                    request_data=data,
                )

            async for token_output in self.engine.stream_request(
                req,
                timeout=DEFAULT_STREAM_TIMEOUT,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
            ):
                received_token_outputs += 1
                token_timing = getattr(token_output, "output_timing", None)
                if token_timing is not None:
                    token_timing.route_received_at_ns = monotonic_ns()
                    output_collector.record_token(token_timing, token_output.token_id)

                # ASGI disconnect probes are measurable at high token rates. Check
                # periodically instead of once per token; Starlette still cancels
                # this generator when it observes a transport-level disconnect.
                should_check_disconnect = (
                    received_token_outputs % DISCONNECT_CHECK_TOKEN_INTERVAL == 0
                    or token_output.finished
                )
                if should_check_disconnect:
                    disconnect_started_at_ns = (
                        monotonic_ns() if output_timing_enabled else None
                    )
                    if await http_request.is_disconnected():
                        logger.info(f"Client disconnected for request {request_id}")
                        break
                    if output_timing_enabled:
                        output_collector.record_disconnect(
                            disconnect_started_at_ns, monotonic_ns()
                        )

                if token_output.finish_reason == FinishReason.TIMEOUT:
                    logger.warning(
                        f"Request {request_id} timed out after {DEFAULT_REQUEST_TIMEOUT}s"
                    )
                    pending_chunk = take_pending_sse_chunk()
                    if pending_chunk is not None:
                        yield_started_at_ns = (
                            monotonic_ns() if output_timing_enabled else None
                        )
                        yield pending_chunk
                        if output_timing_enabled:
                            output_collector.record_route_yield(
                                yield_started_at_ns,
                                monotonic_ns(),
                                len(pending_chunk.encode()),
                            )
                    error_chunk = render_sse_chunk(
                        chunk_json(
                            request_id,
                            content="[Request timeout]",
                            finish_reason="timeout",
                            model=self.model_id,
                        )
                    )
                    yield_started_at_ns = (
                        monotonic_ns() if output_timing_enabled else None
                    )
                    yield error_chunk
                    if output_timing_enabled:
                        output_collector.record_route_yield(
                            yield_started_at_ns,
                            monotonic_ns(),
                            len(error_chunk.encode()),
                        )
                    break

                # Skip EOS token text for OpenAI API compatibility
                # Check if this token is an EOS token by comparing token_id with eos_token_ids
                eos_token_ids = self.engine.engine.eos_token_ids
                is_eos_token = (
                    not sampling_params.ignore_eos
                    and eos_token_ids
                    and token_output.token_id in eos_token_ids
                )

                pending_token_count += 1

                if not is_eos_token and token_output.token_text:
                    pending_token_texts.append(token_output.token_text)

                    # Preserve first-token latency, then amortize JSON and ASGI
                    # sends over a small token group.
                    should_flush = (
                        not has_flushed_token_text
                        or pending_token_count >= SSE_FLUSH_TOKEN_INTERVAL
                    )
                    if should_flush:
                        chunk = take_pending_sse_chunk()
                        assert chunk is not None
                        has_flushed_token_text = True
                        yield_started_at_ns = (
                            monotonic_ns() if output_timing_enabled else None
                        )
                        yield chunk
                        if output_timing_enabled:
                            output_collector.record_route_yield(
                                yield_started_at_ns,
                                monotonic_ns(),
                                len(chunk.encode()),
                            )

                if token_output.finished:
                    pending_chunk = take_pending_sse_chunk()
                    if pending_chunk is not None:
                        yield_started_at_ns = (
                            monotonic_ns() if output_timing_enabled else None
                        )
                        yield pending_chunk
                        if output_timing_enabled:
                            output_collector.record_route_yield(
                                yield_started_at_ns,
                                monotonic_ns(),
                                len(pending_chunk.encode()),
                            )
                    finish_reason = self._convert_finish_reason(
                        token_output.finish_reason
                    )
                    cache_stats = self._cache_stats(req)
                    chunk = render_sse_chunk(
                        chunk_json(
                            request_id,
                            finish_reason=finish_reason,
                            model=self.model_id,
                            cache=cache_stats,
                        )
                    )
                    yield_started_at_ns = (
                        monotonic_ns() if output_timing_enabled else None
                    )
                    yield chunk
                    if output_timing_enabled:
                        output_collector.record_route_yield(
                            yield_started_at_ns,
                            monotonic_ns(),
                            len(chunk.encode()),
                        )
                    break

        except asyncio.CancelledError:
            # Starlette cancelled us (client disconnected); stream_request will be
            # aclose()'d automatically via the async-for destructor.
            logger.info(f"Request {request_id} was cancelled")
            raise

        except Exception as e:
            logger.error(f"Stream error for {request_id}: {e}", exc_info=True)
            _abort_reason = FinishReason.ERROR
            pending_chunk = take_pending_sse_chunk()
            if pending_chunk is not None:
                yield_started_at_ns = monotonic_ns() if output_timing_enabled else None
                yield pending_chunk
                if output_timing_enabled:
                    output_collector.record_route_yield(
                        yield_started_at_ns,
                        monotonic_ns(),
                        len(pending_chunk.encode()),
                    )
            error_chunk = render_sse_chunk(
                chunk_json(
                    request_id,
                    content=f"[Error: {str(e)}]",
                    finish_reason="error",
                    model=self.model_id,
                )
            )
            yield_started_at_ns = monotonic_ns() if output_timing_enabled else None
            yield error_chunk
            if output_timing_enabled:
                output_collector.record_route_yield(
                    yield_started_at_ns,
                    monotonic_ns(),
                    len(error_chunk.encode()),
                )

        finally:
            if (
                prestarted_request is not None
                and not prestarted_request.tokenized_future.done()
            ):
                prestarted_request.tokenized_future.cancel()
            # Unified abort: reason is ERROR if we got here via Exception, else CANCELED.
            # req.close() is handled by stream_request.finally.
            if req and not req.is_finished():
                self.engine.add_aborted_req(req, _abort_reason)
        done_chunk = "data: [DONE]\n\n"
        yield_started_at_ns = monotonic_ns() if output_timing_enabled else None
        yield done_chunk
        if output_timing_enabled:
            output_collector.record_route_yield(
                yield_started_at_ns,
                monotonic_ns(),
                len(done_chunk.encode()),
            )

    async def _chat(
        self,
        request_id: str,
        data: dict,
        http_request: Request,
        server_received_at_ns: int | None = None,
        entrance_timing: RequestEntranceTiming | None = None,
        output_collector=None,
    ):
        """Handle non-streaming chat request."""
        req = None
        _abort_reason = FinishReason.CANCELED
        if output_collector is None:
            timing_service = getattr(self, "output_timing", None)
            output_collector = (
                timing_service.create_collector(request_id)
                if timing_service is not None
                else DisabledOutputTimingCollector()
            )
        output_timing_enabled = output_collector.enabled
        if entrance_timing is not None and (
            entrance_timing.handler_started_at_ns is None
        ):
            entrance_timing.handler_started_at_ns = monotonic_ns()

        try:
            messages = data.get("messages", [])
            sampling_params = self._build_sampling_params(data)

            req = await self.engine.add_chat_request_async(
                messages=messages,
                sampling_params=sampling_params,
                request_id=request_id,
                request_data=data,
                add_generation_prompt=bool(data.get("add_generation_prompt", True)),
                chat_template_kwargs=data.get("chat_template_kwargs") or {},
                server_received_at_ns=server_received_at_ns,
                entrance_timing=entrance_timing,
            )

            # Collect all generated tokens
            output_text = ""
            received_token_outputs = 0
            async for token_output in self.engine.stream_request(
                req,
                timeout=DEFAULT_STREAM_TIMEOUT,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
            ):
                received_token_outputs += 1
                token_timing = getattr(token_output, "output_timing", None)
                if token_timing is not None:
                    token_timing.route_received_at_ns = monotonic_ns()
                    output_collector.record_token(token_timing, token_output.token_id)

                # Keep the non-streaming disconnect probe on the same cadence as
                # SSE. Per-token receive probes dominate long chat completions.
                should_check_disconnect = (
                    received_token_outputs % DISCONNECT_CHECK_TOKEN_INTERVAL == 0
                    or token_output.finished
                )
                if should_check_disconnect:
                    disconnect_started_at_ns = (
                        monotonic_ns() if output_timing_enabled else None
                    )
                    if await http_request.is_disconnected():
                        logger.info(f"Client disconnected for request {request_id}")
                        break
                    if output_timing_enabled:
                        output_collector.record_disconnect(
                            disconnect_started_at_ns, monotonic_ns()
                        )

                # Request-level timeout is handled inside stream_request.
                if token_output.finish_reason == FinishReason.TIMEOUT:
                    logger.warning(f"Request {request_id} timed out")
                    break

                # Skip EOS token text for OpenAI API compatibility
                # Check if this token is an EOS token by comparing token_id with eos_token_ids
                eos_token_ids = self.engine.engine.eos_token_ids
                is_eos_token = eos_token_ids and token_output.token_id in eos_token_ids

                if not is_eos_token and token_output.token_text:
                    output_text += token_output.token_text

                if token_output.finished:
                    break

            output_text = output_text.strip()
            finish_reason = self._convert_finish_reason(req.finish_reason)

            response = completion_json(
                request_id,
                content=output_text,
                role="assistant",
                finish_reason=finish_reason or "stop",
                model=self.model_id,
                prompt_tokens=req.get_prompt_length(),
                completion_tokens=req.get_num_generated_tokens(),
                total_tokens=req.get_total_length(),
                cache=self._cache_stats(req),
            )
            return response

        except asyncio.CancelledError:
            logger.info(f"Request {request_id} was cancelled")
            raise

        except Exception as e:
            logger.error(f"Chat error for {request_id}: {e}", exc_info=True)
            _abort_reason = FinishReason.ERROR
            return JSONResponse(content={"error": str(e)}, status_code=500)

        finally:
            # Unified abort: reason is ERROR if we got here via Exception, else CANCELED.
            # req.close() is handled by stream_request.finally.
            if req and not req.is_finished():
                self.engine.add_aborted_req(req, _abort_reason)

    def _convert_finish_reason(self, reason: FinishReason) -> str:
        """Convert FinishReason enum to string."""
        if reason is None:
            return None
        if reason in (FinishReason.EOS_TOKEN, FinishReason.STOP_STRING):
            return "stop"

        return reason.value

    @staticmethod
    def _cache_stats(req) -> dict:
        """Return the local prefix-cache hit observed for this request."""
        prompt_tokens = req.get_prompt_length() if req is not None else 0
        cached_tokens = req.num_local_prefix_cache_hit_tokens if req is not None else 0
        return {
            "hit": cached_tokens > 0,
            "matched_tokens": cached_tokens,
            "prefill_tokens": max(0, prompt_tokens - cached_tokens),
            "prompt_tokens": prompt_tokens,
        }


def setup_logging(log_level: str = "INFO"):
    """Configure logging system with proper formatting and handlers."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def parse_kv_transfer_config(kv_transfer_config_str: str) -> KVTransferConfig:
    """Parse JSON string into KVTransferConfig."""
    kv_dict = json.loads(kv_transfer_config_str)
    if not isinstance(kv_dict, dict):
        raise ValueError("--kv-transfer-config must be a JSON object")

    return KVTransferConfig(
        kv_connector=kv_dict.get("kv_connector", None),
        engine_id=kv_dict.get("engine_id", None),
        kv_role=kv_dict.get("kv_role", None),
        kv_connector_extra_config=kv_dict.get("kv_connector_extra_config", None),
    )


def main():
    cfg = BaseConfig()
    setup_logging(cfg.log_level)
    if cfg.pp > 1 and cfg.node_rank > 0:
        from infinilm.server.pipeline_worker import run_worker

        run_worker(cfg)
        return

    device = cfg.get_device_str(cfg.device)

    kv_transfer_config = None
    if cfg.kv_transfer_config:
        kv_transfer_config = parse_kv_transfer_config(cfg.kv_transfer_config)

    if cfg.use_legacy_moe:
        moe_ep_backend, ep = "disabled", 1
    else:
        moe_ep_backend, ep = configure_moe_ep_backend(
            cfg.tp, cfg.dp, cfg.ep, cfg.moe_ep_backend, cfg.model
        )
    logger.info(
        "MoE EP backend: %s  TP=%s  DP=%s  EP=%s",
        moe_ep_backend,
        cfg.tp,
        cfg.dp,
        ep,
    )

    server = InferenceServer(
        model_path=cfg.model,
        device=device,
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tp,
        pipeline_parallel_size=cfg.pp,
        pipeline_parallel_stage=cfg.node_rank,
        master_addr=cfg.master_addr,
        master_port=cfg.master_port,
        moe_ep_backend=moe_ep_backend,
        moe_ep_size=ep,
        use_legacy_moe=cfg.use_legacy_moe,
        cache_type="paged" if cfg.enable_paged_attn else "static",
        max_tokens=cfg.max_new_tokens,
        max_batch_size=cfg.max_batch_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
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
    server.start()


if __name__ == "__main__":
    main()
