"""
LLM Engine - Main interface for LLM inference.

This module provides:
- LLM class for batch generation (offline use)
- AsyncLLM class for asynchronous streaming (server use)
"""

import asyncio
import functools
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Union

import janus

from infinilm.config.engine_config import EngineConfig
from infinilm.config.kv_transfer import KVTransferConfig
from infinilm.infer_engine import model_uses_mamba_cache, read_hf_config
from infinilm.kv_connector import KVConnectorFactory, KVConnectorRole
from infinilm.llm.model_runner import create_model_runner
from infinilm.llm.output_timing import OutputTimingConfig, OutputTokenTiming
from infinilm.llm.request import (
    FinishReason,
    InferenceRequest,
    RequestEntranceTiming,
    RequestOutput,
    TokenBatch,
    TokenOutput,
    monotonic_ns,
)
from infinilm.llm.request_timing.request_timing import RequestTimingConfig
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler
from infinilm.llm.static_scheduler import StaticScheduler
from infinilm.multimodal.multimodal import resolve_multimodal_inputs

logger = logging.getLogger(__name__)

DEFAULT_PREPROCESS_WORKERS = 1


def preprocess_workers_from_environment() -> int:
    raw_value = os.environ.get(
        "INFINILM_PREPROCESS_WORKERS", str(DEFAULT_PREPROCESS_WORKERS)
    )
    try:
        workers = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"INFINILM_PREPROCESS_WORKERS must be an integer: {raw_value!r}"
        ) from exc
    if workers not in {1, 2}:
        raise RuntimeError("INFINILM_PREPROCESS_WORKERS must be 1 or 2")
    return workers


def output_callback_coalescing_from_environment() -> bool:
    raw_value = os.environ.get("INFINILM_OUTPUT_CALLBACK_COALESCE", "true")
    normalized = raw_value.strip().lower()
    if normalized not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise RuntimeError(
            f"INFINILM_OUTPUT_CALLBACK_COALESCE must be a boolean: {raw_value!r}"
        )
    return normalized in {"1", "true", "yes", "on"}


def output_flush_max_tokens_from_environment() -> int:
    raw_value = os.environ.get("INFINILM_OUTPUT_FLUSH_MAX_TOKENS", "256")
    try:
        max_tokens = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"INFINILM_OUTPUT_FLUSH_MAX_TOKENS must be an integer: {raw_value!r}"
        ) from exc
    if max_tokens not in {0, 64, 128, 256, 512}:
        raise RuntimeError(
            "INFINILM_OUTPUT_FLUSH_MAX_TOKENS must be one of 0, 64, 128, 256, 512"
        )
    return max_tokens


def output_token_batch_from_environment() -> bool:
    raw_value = os.environ.get("INFINILM_OUTPUT_TOKEN_BATCH", "true")
    normalized = raw_value.strip().lower()
    if normalized not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise RuntimeError(
            f"INFINILM_OUTPUT_TOKEN_BATCH must be a boolean: {raw_value!r}"
        )
    return normalized in {"1", "true", "yes", "on"}


def stream_ready_drain_items_from_environment() -> int:
    raw_value = os.environ.get("INFINILM_STREAM_READY_DRAIN_ITEMS", "4")
    try:
        drain_items = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"INFINILM_STREAM_READY_DRAIN_ITEMS must be an integer: {raw_value!r}"
        ) from exc
    if drain_items not in {0, 1, 2, 4, 8}:
        raise RuntimeError(
            "INFINILM_STREAM_READY_DRAIN_ITEMS must be one of 0, 1, 2, 4, 8"
        )
    return drain_items


@dataclass(frozen=True)
class _TokenizedChatInput:
    processed_inputs: dict | None
    prompt_token_ids: list[int]
    mm_index_mappings: list[dict] | None
    has_multimodal_inputs: bool


@dataclass(frozen=True)
class _StartedChatPreprocessing:
    request_id: str
    prompt: str
    tokenized_future: asyncio.Future[_TokenizedChatInput]
    server_received_at_ns: Optional[int]
    entrance_timing: RequestEntranceTiming | None


class _LockedTokenizerProxy:
    """Serialize fallback tokenization when a tokenizer cannot be cloned."""

    def __init__(self, tokenizer, lock: threading.Lock):
        self._tokenizer = tokenizer
        self._lock = lock

    def encode(self, text, **kwargs):
        with self._lock:
            return self._tokenizer.encode(text, **kwargs)


class LLMEngine:
    """Low-level LLM engine that handles inference execution."""

    def __init__(self, config: EngineConfig):
        self.config = config
        request_timing_config = RequestTimingConfig.from_environment()
        self.request_timing_enabled = request_timing_config.enabled
        self.request_timing_mode = request_timing_config.mode
        self.request_timing_sample_interval = request_timing_config.sample_interval
        self.request_timing_recorder = request_timing_config.create_recorder()
        self._output_timing_enabled = OutputTimingConfig.from_environment().enabled
        self._tokenizer_lock = threading.Lock()
        hf_config = read_hf_config(config.model_path)
        has_mamba_cache = model_uses_mamba_cache(hf_config)
        if has_mamba_cache and config.enable_prefix_caching:
            model_type = hf_config["model_type"]
            raise RuntimeError(
                "Prefix caching is not supported for Mamba-cache model "
                f"{model_type!r} yet. Restart with "
                "--disable-prefix-caching."
            )

        if config.pipeline_parallel_size > 1 and config.pipeline_parallel_stage != 0:
            raise ValueError(
                "LLMEngine can only run pipeline stage 0; launch non-host nodes "
                "with inference_server.py --node-rank=N"
            )

        # hf_config 已在 Engine 初始化时读取；轻量通信 runner 直接复用它。
        self.model_runner = create_model_runner(config, hf_config=hf_config)

        self.device = self.model_runner.device
        self.dtype = self.model_runner.dtype

        # Initialize processor
        self.processor = self.model_runner.processor
        self.tokenizer = self.processor.get_tokenizer()

        # Initialize KV cache based on cache type
        if config.cache_type == "static":
            self.scheduler = StaticScheduler(
                max_cache_len=config.max_cache_len,
                max_num_batched_tokens=(
                    config.max_num_batched_tokens if config.enable_workspace else None
                ),
                enable_prefix_caching=config.enable_prefix_caching,
            )
            logger.info(
                f"Using Static KV Cache with max_cache_len={config.max_cache_len}"
            )
        elif config.cache_type == "paged":
            connector = None
            if config.kv_transfer_config and config.kv_transfer_config.kv_connector:
                connector = KVConnectorFactory.create_connector(
                    connector_name=config.kv_transfer_config.kv_connector,
                    role=KVConnectorRole.SCHEDULER,
                    kv_transfer_config=config.kv_transfer_config,
                )
                logger.info(
                    f"KV Connector created: {config.kv_transfer_config.kv_connector} "
                    f"(role={config.kv_transfer_config.kv_role})"
                )
            llm_config = self.model_runner.hf_config
            if "text_config" in llm_config:
                llm_config = llm_config["text_config"]

            # max_position_embeddings 不参与 scheduler 初始化；paged 容量以
            # BlockManager 的 num_blocks 为准，static 模式使用 max_cache_len。
            effective_num_blocks = config.effective_num_blocks
            num_mamba_cache_blocks = max(2, effective_num_blocks // 4)

            self.scheduler = Scheduler(
                max_batch_size=config.max_batch_size,
                num_blocks=effective_num_blocks,
                block_size=config.block_size,
                max_num_batched_tokens=config.max_num_batched_tokens,
                connector=connector,
                has_mamba_cache=has_mamba_cache,
                num_mamba_cache_blocks=num_mamba_cache_blocks,
                enable_prefix_caching=config.enable_prefix_caching,
                request_timing_enabled=self.request_timing_enabled,
                request_timing_mode=self.request_timing_mode,
                request_timing_sample_interval=self.request_timing_sample_interval,
            )
            logger.info(f"Using Paged KV Cache with num_blocks={effective_num_blocks}")
            if has_mamba_cache:
                logger.info(
                    "Using Mamba cache with num_blocks=%s, zero_state_index=0",
                    num_mamba_cache_blocks,
                )
        else:
            raise ValueError(f"Unsupported cache_type: {config.cache_type}")

        self.cache_type = config.cache_type

        # Get EOS token IDs from model config
        self.eos_token_ids = self.model_runner.eos_token_id or []
        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = [self.eos_token_ids]

        logger.info(
            f"LLMEngine initialized with model at {config.model_path} "
            f"on device {config.device}, "
            f"enable_graph={config.enable_graph}"
        )

    def add_request(self, request: InferenceRequest):
        """Add a request to the scheduler."""
        if request.timing_role == "aggregated":
            # 请求进入引擎前根据 KV 角色细化 timing 标签：
            # producer 对应 P/prefill，consumer 对应 D/decode。
            kv_config = getattr(self.config, "kv_transfer_config", None)
            kv_role = kv_config and kv_config.kv_role
            if kv_role == "kv_producer":
                request.timing_role = "prefill"
            elif kv_role == "kv_consumer":
                request.timing_role = "decode"
        self.scheduler.add_request(request)

    def close(self):
        self.request_timing_recorder.close()
        self.model_runner.close()

    def step(self) -> tuple[bool, list[tuple]]:
        """Run one inference step.

        Returns:
            A tuple of:
            - did_work
            - pending: Pending streaming outputs as (async_queue, TokenOutput) pairs.
        """
        # Schedule the next unit of work, which may be model execution,
        # connector control metadata, or both.
        scheduler_output = self.scheduler.schedule()
        if scheduler_output is None:
            return False, []

        runner_output = self.model_runner.execute_model(scheduler_output)
        output_timing_enabled = getattr(self, "_output_timing_enabled", False)
        sampled_ready_at_ns = monotonic_ns() if output_timing_enabled else None
        if (
            getattr(self, "request_timing_enabled", True)
            and getattr(self, "request_timing_mode", "full") == "full"
        ):
            timing_steps = [
                request.timing_steps[-1]
                for request in scheduler_output.scheduled_requests
                if request.timing_steps
            ]
            self.request_timing_recorder.record_steps(timing_steps)
        sampled_token_ids = runner_output.sampled_token_ids
        # update_from_output() 会让 connector 汇报本 step 完成的 KV 事件；
        # 随后统一取出并交给 recorder，保持 step 与传输事件的落盘顺序。
        self.scheduler.update_from_output(runner_output)
        for timing_event in self.scheduler.pop_timing_events():
            self.request_timing_recorder.record_event(timing_event)
        pending = self._update_requests(
            scheduler_output.scheduled_requests,
            sampled_token_ids,
            sampled_ready_at_ns=sampled_ready_at_ns,
        )

        # Return False (no immediate work) only when no requests were scheduled
        # and no KV transfers completed in this step.
        if not scheduler_output.scheduled_requests:
            if not runner_output.kv_connector_output or (
                not getattr(runner_output.kv_connector_output, "finished_sending", None)
                and not getattr(
                    runner_output.kv_connector_output, "finished_recving", None
                )
            ):
                return False, pending

        return True, pending

    def _update_requests(
        self,
        requests: List[InferenceRequest],
        sampled_token_ids: list[int | list[int]],
        sampled_ready_at_ns: Optional[int] = None,
    ) -> List[tuple]:
        """Apply sampled tokens and publish their target-model KV boundary."""
        output_timing_enabled = getattr(self, "_output_timing_enabled", False)
        if len(requests) != len(sampled_token_ids):
            raise RuntimeError(
                "model output count does not match the scheduled request count: "
                f"requests={len(requests)}, outputs={len(sampled_token_ids)}"
            )
        pending = []
        for req, token_ids in zip(requests, sampled_token_ids):
            if req.is_finished():
                # scheduler 可能在输出更新前已把请求标记为 failed/timeout。
                self.request_timing_recorder.record_finish(req)
                continue

            # The model successfully consumed the request's current logical tokens.
            # Commit this boundary before observing a concurrent client abort.
            pre_output_computed_tokens = req.get_total_length()
            self.scheduler.commit_computed_tokens(req, pre_output_computed_tokens)

            if req.is_aborted():
                logger.info(
                    f"Request {req.request_id} aborted by client, skipping update"
                )
                # close() may have set _aborted=True without setting a terminal status
                # (status still RUNNING).
                if not req.is_finished():
                    req.mark_canceled()
                self.request_timing_recorder.record_finish(req)
                continue

            if isinstance(token_ids, list):
                num_computed_output_tokens = max(len(token_ids) - 1, 0)
            else:
                num_computed_output_tokens = 0
                token_ids = [token_ids]

            num_appended_tokens = 0
            for token_id in token_ids:
                if req.is_finished():
                    break
                req.append_generated_token_id(token_id)
                num_appended_tokens += 1
                pending_tokens = req.generated_token_ids[req._token_decode_offset :]
                decode_started_at_ns = monotonic_ns() if output_timing_enabled else None
                with self._tokenizer_lock:
                    delta = self.tokenizer.decode(pending_tokens)
                decode_completed_at_ns = (
                    monotonic_ns() if output_timing_enabled else None
                )
                holds_back = bool(delta) and delta.endswith("\ufffd")

                last_committed_text = req.generated_text

                if not holds_back:
                    req.generated_text = last_committed_text + delta
                    req._token_decode_offset = len(req.generated_token_ids)

                is_finished = self._check_request_finished(req, token_id)

                # vLLM-style replacement character handling is primarily relevant for streaming.
                # For offline generation (no output queue), keep the fast incremental path.
                if req._output_queue is None:
                    if is_finished:
                        req.mark_finished(req.finish_reason)
                        self.request_timing_recorder.record_finish(req)

                else:
                    if holds_back and not is_finished:
                        token_text = ""
                    else:
                        if is_finished and req.finish_reason in (
                            FinishReason.EOS_TOKEN,
                            FinishReason.LENGTH,
                            FinishReason.STOP_STRING,
                        ):
                            token_text = ""
                        else:
                            token_text = req.generated_text[req._text_output_offset :]
                            if token_text:
                                req._text_output_offset = len(req.generated_text)

                        if is_finished:
                            req.mark_finished(req.finish_reason)
                            self.request_timing_recorder.record_finish(req)

                    output_timing = (
                        OutputTokenTiming(
                            engine_sampled_ready_at_ns=sampled_ready_at_ns,
                            engine_decode_started_at_ns=decode_started_at_ns,
                            engine_decode_completed_at_ns=decode_completed_at_ns,
                            engine_token_ready_at_ns=monotonic_ns(),
                        )
                        if output_timing_enabled
                        else None
                    )
                    output = TokenOutput(
                        request_id=req.request_id,
                        token_id=token_id,
                        token_text=token_text,
                        finished=is_finished,
                        finish_reason=req.finish_reason if is_finished else None,
                        generated_text=req.generated_text,
                        output_timing=output_timing,
                    )
                    if req.is_aborted():
                        logger.info(
                            f"Request {req.request_id} aborted before putting token"
                        )
                        continue
                    pending.append((req.output_queue.async_q, output))

            post_output_computed_tokens = pre_output_computed_tokens + min(
                num_appended_tokens,
                num_computed_output_tokens,
            )
            if post_output_computed_tokens > pre_output_computed_tokens:
                self.scheduler.commit_computed_tokens(req, post_output_computed_tokens)

        self.scheduler.complete_requests(requests)
        return pending

    def _check_request_finished(self, req: InferenceRequest, token_id: int) -> bool:
        """Check if request generation is finished."""
        max_tokens = req.sampling_params.max_tokens
        if max_tokens and req.get_num_generated_tokens() >= max_tokens:
            req.finish_reason = FinishReason.LENGTH
            return True

        if not req.sampling_params.ignore_eos:
            # Check EOS token - only stop if ignore_eos is False
            eos_ids = req.eos_token_ids or self.eos_token_ids
            if eos_ids and token_id in eos_ids:
                req.finish_reason = FinishReason.EOS_TOKEN
                return True

            # While ignoring EOS, stop strings are also ignored to avoid requiring additional arguments for benchmarking.
            # Check stop strings
            # Remove stop string from generated_text if STOP_STRING is the finishing reason
            stop_strings = req.sampling_params.stop or []
            for stop_str in stop_strings:
                if req.generated_text.endswith(stop_str):
                    req.generated_text = req.generated_text[: -len(stop_str)]
                    req.finish_reason = FinishReason.STOP_STRING
                    return True

        return False

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        with self._tokenizer_lock:
            return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        """Detokenize token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def process(self, prompt, images, videos, audios, **kwargs) -> dict:
        """Process the input prompt and media into final model inputs."""
        with self._tokenizer_lock:
            return self.processor(
                prompt, images=images, videos=videos, audios=audios, **kwargs
            )

    def apply_chat_template(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
        chat_template_kwargs: Optional[dict] = None,
    ) -> str:
        """Apply chat template to messages."""
        chat_template_kwargs = chat_template_kwargs or {}
        return self.processor.apply_chat_template(
            conversation=messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **chat_template_kwargs,
        )


class LLM:
    """High-level LLM interface for batch generation."""

    def __init__(
        self,
        model_path: str,
        draft_model_path: Optional[str] = None,
        num_draft_tokens: int = 4,
        device: str = "cuda",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_stage: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        moe_ep_backend: str = "disabled",
        moe_ep_size: int = 1,
        cache_type: str = "paged",
        max_batch_size: int = 16,
        max_tokens: int = 4096,
        gpu_memory_utilization: float = 0.85,
        block_size: int = 256,
        max_cache_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        enable_graph: bool = False,
        enable_workspace: bool = True,
        max_num_batched_tokens: int = 9216,
        attn_backend: str = "default",
        use_mla: bool = False,
        pre_transpose: bool = False,
        weight_load_mode: str = "async",
        skip_load: bool = False,
        use_legacy_moe: bool = False,
        enable_prefix_caching: bool = True,
    ):
        """Initialize LLM.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            cache_type: Cache type ('paged' or 'static').
            max_batch_size: Maximum batch size (only for paged cache).
            max_tokens: Default maximum tokens to generate.
            gpu_memory_utilization: Fraction of device memory used to size KV cache.
            block_size: Size of each KV cache block (only for paged cache).
            max_cache_len: Maximum sequence length (only for static cache).
            temperature: Default sampling temperature.
            top_p: Default top-p sampling parameter.
            top_k: Default top-k sampling parameter.
            enable_graph: Whether to enable graph compiling.
            enable_workspace: Whether to preallocate reusable inference activations.
            max_num_batched_tokens: Maximum scheduler batch and workspace token count.
            attn_backend: Attention backend to use ('default', 'flash-attn').
            use_mla: Whether to use DeepSeek V2 MLA attention when supported.
            weight_load_mode: Weight loading mode across tensor-parallel workers.
        """
        config = EngineConfig(
            model_path=model_path,
            draft_model_path=draft_model_path,
            num_draft_tokens=num_draft_tokens,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            pipeline_parallel_stage=pipeline_parallel_stage,
            master_addr=master_addr,
            master_port=master_port,
            moe_ep_backend=moe_ep_backend,
            moe_ep_size=moe_ep_size,
            cache_type=cache_type,
            max_batch_size=max_batch_size,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            block_size=block_size,
            max_cache_len=max_cache_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_graph=enable_graph,
            enable_workspace=enable_workspace,
            max_num_batched_tokens=max_num_batched_tokens,
            attn_backend=attn_backend,
            use_mla=use_mla,
            pre_transpose=pre_transpose,
            weight_load_mode=weight_load_mode,
            skip_load=skip_load,
            use_legacy_moe=use_legacy_moe,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.engine = LLMEngine(config)
        self.config = config

    def close(self):
        self.engine.close()

    def generate(
        self,
        prompts: Union[str, List[str]] = None,
        messages: Union[List[dict], List[List[dict]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: A single prompt string or list of prompt strings.
            sampling_params: Sampling parameters for generation.
            use_tqdm: Whether to show progress bar.

        Returns:
            List of RequestOutput objects containing generated text.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]

        contents = prompts
        apply_chat_template = False
        if messages:
            contents = messages
            apply_chat_template = True

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=self.config.max_tokens)
        elif sampling_params.max_tokens is None:
            sampling_params = sampling_params.clone()
            sampling_params.max_tokens = self.config.max_tokens

        requests = []
        for content in contents:
            request_id = f"cmpl-{uuid.uuid4().hex}"
            processed_inputs = None
            mm_index_mappings = None
            has_multimodal_inputs = False
            if apply_chat_template:
                prompt = self.engine.apply_chat_template(
                    content, add_generation_prompt=True
                )

                mm_inputs = resolve_multimodal_inputs(content)

                has_multimodal_inputs = any(
                    mm_inputs[key] for key in ("images", "videos", "audios")
                )
                processed_inputs = self.engine.process(
                    prompt,
                    mm_inputs["images"],
                    mm_inputs["videos"],
                    mm_inputs["audios"],
                    return_tensors="pt",
                )

                prompt_token_ids = processed_inputs.get("input_ids").flatten().tolist()
                mm_index_mappings = self.engine.processor.get_mm_token_index_list(
                    prompt_token_ids,
                    image_ids=mm_inputs["image_urls"],
                    video_ids=mm_inputs["video_urls"],
                    audio_ids=mm_inputs["audio_urls"],
                )
            else:
                prompt = content
                prompt_token_ids = self.engine.tokenize(prompt)

            req = InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                processed_inputs=processed_inputs,
                mm_token_index_mappings=mm_index_mappings,
                has_multimodal_inputs=has_multimodal_inputs,
                sampling_params=sampling_params,
                eos_token_ids=self.engine.eos_token_ids,
            )
            requests.append(req)
            self.engine.add_request(req)

        # Run inference until all requests are finished
        if use_tqdm:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(requests), desc="Generating")
            except ImportError:
                pbar = None
                use_tqdm = False
        else:
            pbar = None

        finished_count = 0
        while finished_count < len(requests):
            self.engine.step()

            new_finished = sum(1 for req in requests if req.is_finished())
            if use_tqdm and pbar and new_finished > finished_count:
                pbar.update(new_finished - finished_count)
            finished_count = new_finished

        if pbar:
            pbar.close()

        outputs = [req.to_request_output() for req in requests]
        return outputs

    def chat(
        self,
        messages: Union[List[dict], List[List[dict]]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generate chat completions for the given messages.

        Args:
            messages: A single conversation (list of message dicts) or
                     a list of conversations.
            sampling_params: Sampling parameters for generation.
            use_tqdm: Whether to show progress bar.

        Returns:
            List of RequestOutput objects containing generated responses.
        """
        if messages and isinstance(messages[0], dict):
            messages = [messages]

        return self.generate(
            messages=messages, sampling_params=sampling_params, use_tqdm=use_tqdm
        )


class AsyncLLMEngine:
    """Asynchronous LLM engine for server use with streaming support."""

    def __init__(
        self,
        model_path: str,
        draft_model_path: Optional[str] = None,
        num_draft_tokens: int = 4,
        device: str = "cuda",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_stage: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        moe_ep_backend: str = "disabled",
        moe_ep_size: int = 1,
        cache_type: str = "paged",
        max_batch_size: int = 16,
        max_tokens: int = 512,
        gpu_memory_utilization: float = 0.85,
        block_size: int = 256,
        max_cache_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        enable_graph: bool = False,
        enable_workspace: bool = True,
        max_num_batched_tokens: int = 9216,
        attn_backend: str = "default",
        kv_transfer_config: Optional[KVTransferConfig] = None,
        use_mla: bool = False,
        pre_transpose: bool = False,
        skip_load: bool = False,
        weight_load_mode: str = "async",
        use_legacy_moe: bool = False,
        enable_prefix_caching: bool = True,
        preprocess_workers: int | None = None,
    ):
        """Initialize AsyncLLMEngine.

        Args:
            model_path: Path to the model directory.
            device: Device type ('cpu', 'cuda', 'mlu', 'moore').
            dtype: Data type ('float16', 'bfloat16', 'float32').
            tensor_parallel_size: Number of devices for tensor parallelism.
            cache_type: Cache type ('paged' or 'static').
            max_batch_size: Maximum batch size (only for paged cache).
            max_tokens: Default maximum tokens to generate.
            gpu_memory_utilization: Fraction of device memory used to size KV cache.
            block_size: Size of each KV cache block (only for paged cache).
            max_cache_len: Maximum sequence length (only for static cache).
            temperature: Default sampling temperature.
            top_p: Default top-p sampling parameter.
            top_k: Default top-k sampling parameter.
            enable_graph: Whether to enable graph compiling.
            enable_workspace: Whether to preallocate reusable inference activations.
            max_num_batched_tokens: Maximum scheduler batch and workspace token count.
            attn_backend: Attention backend to use ('default', 'flash-attn').
            kv_connector: KV connector type ('MooncakeConnector').
            kv_role: Role in KV connector ('kv_producer' or 'kv_consumer').
            kv_connector_extra_config: Extra config dict for KV connector.
            use_mla: Whether to use DeepSeek V2 MLA attention when supported.
            skip_load: Whether to skip loading model weights.
            weight_load_mode: Weight loading mode across tensor-parallel workers.
        """
        config = EngineConfig(
            model_path=model_path,
            draft_model_path=draft_model_path,
            num_draft_tokens=num_draft_tokens,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            pipeline_parallel_stage=pipeline_parallel_stage,
            master_addr=master_addr,
            master_port=master_port,
            moe_ep_backend=moe_ep_backend,
            moe_ep_size=moe_ep_size,
            cache_type=cache_type,
            max_batch_size=max_batch_size,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            block_size=block_size,
            max_cache_len=max_cache_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_graph=enable_graph,
            enable_workspace=enable_workspace,
            max_num_batched_tokens=max_num_batched_tokens,
            attn_backend=attn_backend,
            kv_transfer_config=kv_transfer_config,
            use_mla=use_mla,
            pre_transpose=pre_transpose,
            skip_load=skip_load,
            weight_load_mode=weight_load_mode,
            use_legacy_moe=use_legacy_moe,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.engine = LLMEngine(config)
        self.config = config

        self._running = False
        self._step_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._healthy = True
        self._abort_queue: Optional[janus.Queue] = None
        self._preprocess_workers = (
            preprocess_workers
            if preprocess_workers is not None
            else preprocess_workers_from_environment()
        )
        if self._preprocess_workers not in {1, 2}:
            raise ValueError("preprocess_workers must be 1 or 2")
        self._preprocess_executor: ThreadPoolExecutor | None = None
        self._preprocess_thread_local = threading.local()
        self._coalesce_output_callbacks = output_callback_coalescing_from_environment()
        self._output_flush_max_tokens = output_flush_max_tokens_from_environment()
        self._output_token_batch = output_token_batch_from_environment()
        self._stream_ready_drain_items = stream_ready_drain_items_from_environment()
        self._pending_stream_outputs: list[tuple] = []
        self._pending_output_flush_scheduled = False
        self._pending_output_lock = threading.Lock()
        self._output_producer_batch_count = 0
        self._output_producer_token_count = 0
        self._output_flush_callback_count = 0
        self._output_flush_token_count = 0
        self._output_flush_max_batch_size = 0
        self._output_flush_queue_item_count = 0
        self._output_flush_max_queue_items = 0
        self._output_max_pending_buffer_size = 0
        self._stream_queue_wait_count = 0
        self._stream_queue_nowait_count = 0
        self._stream_timer_start_count = 0
        self._stream_ready_drain_count = 0

    def is_healthy(self) -> bool:
        return bool(self._healthy)

    def get_load_stats(self) -> dict:
        """Return a best-effort snapshot for cluster load-aware routing.

        Queue sizes and cache counters are read from the scheduler while the
        engine step thread may be mutating them.  This is intentional: the
        values are periodic routing hints, not exact accounting snapshots.
        """
        scheduler = self.engine.scheduler
        stats: dict[str, Union[int, float]] = {
            "waiting_requests": 0,
            "running_requests": 0,
            "max_batch_size": self.config.max_batch_size,
        }

        waiting_queue = getattr(scheduler, "waiting_queue", None)
        if waiting_queue is not None:
            stats["waiting_requests"] = waiting_queue.sync_q.qsize()

        running_queue = getattr(scheduler, "running_queue", None)
        if running_queue is not None:
            stats["running_requests"] = running_queue.sync_q.qsize()
        elif getattr(scheduler, "running_request", None) is not None:
            stats["running_requests"] = 1

        try:
            cache_stats = scheduler.get_cache_stats()
            if "usable_blocks" in cache_stats:
                capacity = max(1, int(cache_stats["usable_blocks"]))
                used = max(0, int(cache_stats.get("num_used_blocks", 0)))
                stats["kv_capacity"] = capacity
                stats["kv_used"] = used
                stats["num_free_blocks"] = max(
                    0, int(cache_stats.get("num_free_blocks", capacity - used))
                )
                stats["kv_usage_ratio"] = min(1.0, used / capacity)
            elif "max_cache_len" in cache_stats:
                capacity = max(1, int(cache_stats["max_cache_len"]))
                used = max(0, int(cache_stats.get("cached_tokens", 0)))
                stats["kv_capacity"] = capacity
                stats["kv_used"] = used
                stats["kv_usage_ratio"] = min(1.0, used / capacity)
        except Exception as exc:
            logger.debug("Could not read scheduler cache stats: %s", exc)

        return stats

    def start(self):
        """Start the background inference loop."""
        if self._running:
            logger.warning("AsyncLLMEngine is already running")
            return

        self._loop = asyncio.get_running_loop()
        self._abort_queue = janus.Queue()
        self._pending_stream_outputs = []
        self._pending_output_flush_scheduled = False
        self._output_producer_batch_count = 0
        self._output_producer_token_count = 0
        self._output_flush_callback_count = 0
        self._output_flush_token_count = 0
        self._output_flush_max_batch_size = 0
        self._output_flush_queue_item_count = 0
        self._output_flush_max_queue_items = 0
        self._output_max_pending_buffer_size = 0
        self._stream_queue_wait_count = 0
        self._stream_queue_nowait_count = 0
        self._stream_timer_start_count = 0
        self._stream_ready_drain_count = 0
        self._preprocess_thread_local = threading.local()
        self._preprocess_executor = ThreadPoolExecutor(
            max_workers=self._preprocess_workers,
            thread_name_prefix="infinilm-chat-preprocess",
        )
        self._running = True
        self._step_thread = threading.Thread(
            target=self._step_loop, daemon=True, name="AsyncLLMEngineStepThread"
        )
        self._step_thread.start()
        logger.info("AsyncLLMEngine started")

    def stop(self):
        """Stop the background inference loop."""
        if not self._running:
            logger.warning("AsyncLLMEngine is not running")
            return

        self._running = False
        if self._step_thread:
            self._step_thread.join(timeout=5)
        if self._preprocess_executor is not None:
            self._preprocess_executor.shutdown(wait=True, cancel_futures=True)
            self._preprocess_executor = None
        self._flush_pending_outputs_if_on_loop()
        logger.info("Output callback stats: %s", self.get_output_flush_stats())
        self.engine.close()
        logger.info("AsyncLLMEngine stopped")

    def get_output_flush_stats(self) -> dict[str, int | float]:
        with self._pending_output_lock:
            producer_batches = self._output_producer_batch_count
            producer_tokens = self._output_producer_token_count
            flush_callbacks = self._output_flush_callback_count
            flush_tokens = self._output_flush_token_count
            max_flush_batch = self._output_flush_max_batch_size
            flush_queue_items = self._output_flush_queue_item_count
            max_flush_queue_items = self._output_flush_max_queue_items
            max_pending = self._output_max_pending_buffer_size
            stream_queue_waits = self._stream_queue_wait_count
            stream_queue_nowaits = self._stream_queue_nowait_count
            stream_timer_starts = self._stream_timer_start_count
            stream_ready_drains = self._stream_ready_drain_count
            pending = len(self._pending_stream_outputs)
            scheduled = self._pending_output_flush_scheduled

        return {
            "enabled": self._coalesce_output_callbacks,
            "max_tokens_per_flush": self._output_flush_max_tokens,
            "token_batch_enabled": self._output_token_batch,
            "producer_batches": producer_batches,
            "producer_tokens": producer_tokens,
            "flush_callbacks": flush_callbacks,
            "flush_tokens": flush_tokens,
            "flush_queue_items": flush_queue_items,
            "producer_batches_per_flush": (
                producer_batches / flush_callbacks if flush_callbacks else None
            ),
            "producer_tokens_per_flush": (
                producer_tokens / flush_callbacks if flush_callbacks else None
            ),
            "max_flush_batch_size": max_flush_batch,
            "max_flush_queue_items": max_flush_queue_items,
            "stream_queue_waits": stream_queue_waits,
            "stream_queue_nowaits": stream_queue_nowaits,
            "stream_timer_starts": stream_timer_starts,
            "stream_ready_drains": stream_ready_drains,
            "max_pending_buffer_size": max_pending,
            "pending_buffer_size": pending,
            "flush_scheduled": scheduled,
        }

    def add_aborted_req(
        self,
        req: InferenceRequest,
        reason: FinishReason = FinishReason.CANCELED,
    ):
        """Submit an abort request from async side to the step thread.

        The step thread processes this in _drain_abort_queue() before each schedule().
        """
        if self._abort_queue is not None:
            self._abort_queue.sync_q.put((req, reason))

    def _drain_abort_queue(self):
        """Process all pending abort requests before each schedule() call.

        Runs in the step thread (sync context). Guarantees mark_*() is called
        before schedule() so is_finished() checks in waiting/running queue loops
        work correctly. Puts a final TokenOutput into the output queue to unblock
        stream_request when _stream_chat is still alive after abort.
        """
        if self._abort_queue is None:
            return
        while True:
            try:
                req, reason = self._abort_queue.sync_q.get_nowait()
            except Exception:
                break

            if req.is_finished():
                continue

            if reason == FinishReason.CANCELED:
                req.mark_canceled()
            elif reason == FinishReason.TIMEOUT:
                req.mark_timeout()
            else:
                req.mark_failed(reason)

            self.engine.request_timing_recorder.record_finish(req)

            # Put a final token to unblock stream_request.
            # If Starlette already cancelled _stream_chat, aclose() may have closed
            # the queue; put_nowait will raise and we silently ignore it.
            if req._output_queue is not None:
                final = TokenOutput(
                    request_id=req.request_id,
                    token_id=-1,
                    token_text="",
                    finished=True,
                    finish_reason=req.finish_reason,
                    generated_text=req.generated_text,
                )
                try:
                    queue_item = (
                        TokenBatch(final.request_id, [final])
                        if self._output_token_batch
                        else final
                    )
                    req.output_queue.sync_q.put_nowait(queue_item)
                except Exception:
                    pass

    def _step_loop(self):
        """Background loop that runs inference steps."""
        while self._running:
            try:
                self._drain_abort_queue()
                did_work, pending = self.engine.step()
                if not did_work:
                    time.sleep(0.003)
                elif pending:
                    self._submit_pending_outputs(pending)
            except Exception as e:
                logger.error(f"Error in step loop: {e}", exc_info=True)
                self._healthy = False
                self._running = False
                break

    @staticmethod
    def _mark_outputs_scheduled(pending, scheduled_at_ns: int) -> None:
        for _, output in pending:
            if output.output_timing is not None:
                output.output_timing.loop_callback_scheduled_at_ns = scheduled_at_ns

    def _submit_pending_outputs(self, pending: list[tuple]) -> None:
        """Publish step outputs while retaining loop ownership rules."""
        if not pending:
            return

        scheduled_at_ns = monotonic_ns()
        if not self._coalesce_output_callbacks:
            with self._pending_output_lock:
                self._output_producer_batch_count += 1
                self._output_producer_token_count += len(pending)
                self._output_flush_callback_count += 1
                self._output_flush_token_count += len(pending)
                self._output_flush_max_batch_size = max(
                    self._output_flush_max_batch_size, len(pending)
                )
            self._mark_outputs_scheduled(pending, scheduled_at_ns)
            self._loop.call_soon_threadsafe(
                self._batch_put,
                pending,
                scheduled_at_ns,
                False,
            )
            return

        with self._pending_output_lock:
            self._mark_outputs_scheduled(pending, scheduled_at_ns)
            self._pending_stream_outputs.extend(pending)
            self._output_producer_batch_count += 1
            self._output_producer_token_count += len(pending)
            self._output_max_pending_buffer_size = max(
                self._output_max_pending_buffer_size,
                len(self._pending_stream_outputs),
            )
            should_schedule_flush = not self._pending_output_flush_scheduled
            if should_schedule_flush:
                self._pending_output_flush_scheduled = True

        if should_schedule_flush:
            self._loop.call_soon_threadsafe(self._flush_pending_outputs)

    def _flush_pending_outputs(self) -> None:
        continue_flush = False
        with self._pending_output_lock:
            pending = self._pending_stream_outputs
            if not pending:
                self._pending_output_flush_scheduled = False
                return
            flush_size = (
                len(pending)
                if self._output_flush_max_tokens == 0
                else min(len(pending), self._output_flush_max_tokens)
            )
            pending, self._pending_stream_outputs = (
                pending[:flush_size],
                pending[flush_size:],
            )
            continue_flush = bool(self._pending_stream_outputs)
            if not continue_flush:
                self._pending_output_flush_scheduled = False
            self._output_flush_callback_count += 1
            self._output_flush_token_count += len(pending)
            self._output_flush_max_batch_size = max(
                self._output_flush_max_batch_size, len(pending)
            )

        self._batch_put(pending, monotonic_ns(), self._output_token_batch)
        if continue_flush:
            # Schedule after the queue notifications so already-running stream
            # consumers get a turn before the next output slice.
            self._loop.call_soon(self._flush_pending_outputs)

    def _flush_pending_outputs_if_on_loop(self) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._loop is None or running_loop is not self._loop:
            return
        original_max_tokens = self._output_flush_max_tokens
        self._output_flush_max_tokens = 0
        try:
            self._flush_pending_outputs()
        finally:
            self._output_flush_max_tokens = original_max_tokens

    def _batch_put(
        self,
        pending,
        callback_started_at_ns: int | None = None,
        use_token_batch: bool | None = None,
    ):
        if callback_started_at_ns is None:
            callback_started_at_ns = monotonic_ns()

        use_token_batch = (
            self._output_token_batch if use_token_batch is None else use_token_batch
        )
        queue_items: list[tuple] = []
        if use_token_batch:
            grouped_outputs: dict[Any, list[TokenOutput]] = {}
            for async_q, output in pending:
                grouped_outputs.setdefault(async_q, []).append(output)
            queue_items = [
                (async_q, outputs) for async_q, outputs in grouped_outputs.items()
            ]
        else:
            queue_items = [(async_q, [output]) for async_q, output in pending]

        with self._pending_output_lock:
            self._output_flush_queue_item_count += len(queue_items)
            self._output_flush_max_queue_items = max(
                self._output_flush_max_queue_items, len(queue_items)
            )

        for async_q, outputs in queue_items:
            try:
                queue_put_started_at_ns = monotonic_ns()
                for output in outputs:
                    if output.output_timing is not None:
                        output.output_timing.loop_callback_started_at_ns = (
                            callback_started_at_ns
                        )
                        output.output_timing.queue_put_started_at_ns = (
                            queue_put_started_at_ns
                        )

                queue_item = (
                    TokenBatch(
                        request_id=outputs[0].request_id,
                        outputs=outputs,
                    )
                    if use_token_batch
                    else outputs[0]
                )
                async_q.put_nowait(queue_item)
                queue_put_completed_at_ns = monotonic_ns()
                for output in outputs:
                    if output.output_timing is not None:
                        output.output_timing.queue_put_completed_at_ns = (
                            queue_put_completed_at_ns
                        )
            except Exception as e:
                request_id = outputs[0].request_id if outputs else "unknown"
                logger.warning(
                    f"Failed to put token for request {request_id}: {e}. "
                    f"Likely due to client disconnecting or request cancelation."
                )

    def _tokenize_chat_input(
        self,
        prompt: str,
        messages: List[dict],
        entrance_timing: RequestEntranceTiming | None,
    ) -> _TokenizedChatInput:
        mm_inputs = resolve_multimodal_inputs(messages)
        if entrance_timing is not None:
            entrance_timing.multimodal_resolved_at_ns = monotonic_ns()

        has_multimodal_inputs = any(
            mm_inputs[key] for key in ("images", "videos", "audios")
        )

        processor = self.engine.processor
        if not has_multimodal_inputs and getattr(
            processor, "supports_fast_text_tokenize", False
        ):
            prompt_token_ids = processor.fast_text_tokenize(
                prompt,
                tokenizer=self._thread_preprocess_tokenizer(),
            )
            if entrance_timing is not None:
                entrance_timing.tokenization_completed_at_ns = monotonic_ns()
                entrance_timing.prompt_token_ids_ready_at_ns = monotonic_ns()
            return _TokenizedChatInput(
                processed_inputs=None,
                prompt_token_ids=prompt_token_ids,
                mm_index_mappings=[],
                has_multimodal_inputs=False,
            )

        processed_inputs = self.engine.process(
            prompt,
            mm_inputs["images"],
            mm_inputs["videos"],
            mm_inputs["audios"],
            return_tensors="pt",
        )
        if entrance_timing is not None:
            entrance_timing.tokenization_completed_at_ns = monotonic_ns()

        prompt_token_ids = processed_inputs.get("input_ids").flatten().tolist()
        mm_index_mappings = self.engine.processor.get_mm_token_index_list(
            prompt_token_ids,
            image_ids=mm_inputs["image_urls"],
            video_ids=mm_inputs["video_urls"],
            audio_ids=mm_inputs["audio_urls"],
        )
        if entrance_timing is not None:
            entrance_timing.prompt_token_ids_ready_at_ns = monotonic_ns()
        return _TokenizedChatInput(
            processed_inputs=processed_inputs,
            prompt_token_ids=prompt_token_ids,
            mm_index_mappings=mm_index_mappings,
            has_multimodal_inputs=has_multimodal_inputs,
        )

    def _request_from_tokenized_input(
        self,
        *,
        request_id: str,
        prompt: str,
        tokenized: _TokenizedChatInput,
        sampling_params: Optional[SamplingParams],
        server_received_at_ns: Optional[int],
        entrance_timing: RequestEntranceTiming | None,
        request_data: Optional[dict],
    ) -> InferenceRequest:
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=self.config.max_tokens)
        elif sampling_params.max_tokens is None:
            sampling_params = sampling_params.clone()
            sampling_params.max_tokens = self.config.max_tokens

        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=tokenized.prompt_token_ids,
            processed_inputs=tokenized.processed_inputs,
            mm_token_index_mappings=tokenized.mm_index_mappings,
            has_multimodal_inputs=tokenized.has_multimodal_inputs,
            sampling_params=sampling_params,
            eos_token_ids=self.engine.eos_token_ids,
            request_data=request_data,
            server_received_at_ns=server_received_at_ns,
            timing_role=self._timing_role(),
            entrance_timing=entrance_timing,
        )
        if getattr(self, "request_timing_enabled", True):
            request.request_created_at_ns = monotonic_ns()

        if request_data and "kv_transfer_params" in request_data:
            request.kv_transfer_params = request_data["kv_transfer_params"]

        # janus.Queue is bound to the caller's event loop; initialize it after
        # preprocessing has returned to that loop.
        _ = request.output_queue
        self.engine.add_request(request)
        return request

    def _ensure_preprocess_executor(self) -> ThreadPoolExecutor:
        if self._preprocess_executor is None:
            self._preprocess_executor = ThreadPoolExecutor(
                max_workers=self._preprocess_workers,
                thread_name_prefix="infinilm-chat-preprocess",
            )
        return self._preprocess_executor

    def _thread_preprocess_tokenizer(self):
        thread_local = getattr(self, "_preprocess_thread_local", None)
        if thread_local is None:
            thread_local = threading.local()
            self._preprocess_thread_local = thread_local
        tokenizer = getattr(thread_local, "tokenizer", None)
        if tokenizer is None:
            try:
                tokenizer = deepcopy(self.engine.tokenizer)
            except Exception as exc:
                logger.warning(
                    "Could not clone tokenizer for preprocessing thread; "
                    "falling back to the shared tokenizer lock: %s",
                    exc,
                )
                tokenizer = _LockedTokenizerProxy(
                    self.engine.tokenizer,
                    self.engine._tokenizer_lock,
                )
            thread_local.tokenizer = tokenizer
        return tokenizer

    def add_request(
        self,
        messages: Optional[List[dict]],
        apply_chat_template: bool = True,
        add_generation_prompt: bool = True,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        server_received_at_ns: Optional[int] = None,
        entrance_timing: Optional[RequestEntranceTiming] = None,
        # For server use
        request_data: Optional[dict] = None,
    ) -> InferenceRequest:
        """Add a request to the engine.

        Args:
            messages: List of message dicts (chat conversation). Following this format:
                    [
                        {
                            "role": "user",
                            "content": [
                            {
                                "type": "text",
                                "text": "xxxxxxxxx"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "xxx.jpg"
                                }
                            },
                            ]
                        },
                    ]
            apply_chat_template: Whether to apply the chat template.
            add_generation_prompt: Whether to add a generation prompt.
            prompt: Text prompt for generation. If provided, it will be used directly after encoded by tokenizer, ignoring messages.
            prompt_token_ids: Pre-tokenized prompt. If provided, it will be used directly as input.
            sampling_params: Sampling parameters.
            request_id: Optional request ID.
            request_data: Optional request data dict (for server use).

        Returns:
            The created InferenceRequest object.
        """
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"

        mm_index_mappings = None
        processed_inputs = None
        has_multimodal_inputs = False
        if entrance_timing is None and getattr(self, "request_timing_enabled", True):
            entrance_timing = RequestEntranceTiming()

        if prompt_token_ids is not None:
            prompt = self.engine.detokenize(prompt_token_ids)
        elif prompt is not None:
            prompt_token_ids = self.engine.tokenize(prompt)
        else:
            assert messages is not None, (
                "Either messages or prompt/prompt_token_ids must be provided"
            )

            assert apply_chat_template, (
                "apply_chat_template needs to be true for multi-role conversation"
            )

            if entrance_timing is not None:
                entrance_timing.preprocessing_started_at_ns = monotonic_ns()
            prompt = self.engine.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs,
            )
            if entrance_timing is not None:
                entrance_timing.chat_template_completed_at_ns = monotonic_ns()

            mm_inputs = resolve_multimodal_inputs(messages)
            if entrance_timing is not None:
                entrance_timing.multimodal_resolved_at_ns = monotonic_ns()

            has_multimodal_inputs = any(
                mm_inputs[key] for key in ("images", "videos", "audios")
            )
            processed_inputs = self.engine.process(
                prompt,
                mm_inputs["images"],
                mm_inputs["videos"],
                mm_inputs["audios"],
                return_tensors="pt",
            )
            if entrance_timing is not None:
                entrance_timing.tokenization_completed_at_ns = monotonic_ns()

            prompt_token_ids = processed_inputs.get("input_ids").flatten().tolist()
            mm_index_mappings = self.engine.processor.get_mm_token_index_list(
                prompt_token_ids,
                image_ids=mm_inputs["image_urls"],
                video_ids=mm_inputs["video_urls"],
                audio_ids=mm_inputs["audio_urls"],
            )
            if entrance_timing is not None:
                entrance_timing.prompt_token_ids_ready_at_ns = monotonic_ns()

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=self.config.max_tokens)
        elif sampling_params.max_tokens is None:
            sampling_params = sampling_params.clone()
            sampling_params.max_tokens = self.config.max_tokens

        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            processed_inputs=processed_inputs,
            mm_token_index_mappings=mm_index_mappings,
            has_multimodal_inputs=has_multimodal_inputs,
            sampling_params=sampling_params,
            eos_token_ids=self.engine.eos_token_ids,
            request_data=request_data,
            server_received_at_ns=server_received_at_ns,
            timing_role=self._timing_role(),
            entrance_timing=entrance_timing,
        )
        if getattr(self, "request_timing_enabled", True):
            request.request_created_at_ns = monotonic_ns()

        if request_data and "kv_transfer_params" in request_data:
            kv_params = request_data["kv_transfer_params"]
            request.kv_transfer_params = kv_params

        # Initialize output queue for streaming
        _ = request.output_queue

        self.engine.add_request(request)
        return request

    async def add_request_async(
        self,
        messages: Optional[List[dict]],
        apply_chat_template: bool = True,
        add_generation_prompt: bool = True,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        server_received_at_ns: Optional[int] = None,
        entrance_timing: Optional[RequestEntranceTiming] = None,
        request_data: Optional[dict] = None,
    ) -> InferenceRequest:
        """Add a chat request while keeping tokenization off the event loop."""
        if messages is None or prompt is not None or prompt_token_ids is not None:
            return self.add_request(
                messages=messages,
                apply_chat_template=apply_chat_template,
                add_generation_prompt=add_generation_prompt,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                chat_template_kwargs=chat_template_kwargs,
                server_received_at_ns=server_received_at_ns,
                entrance_timing=entrance_timing,
                request_data=request_data,
            )

        assert apply_chat_template, (
            "apply_chat_template needs to be true for multi-role conversation"
        )
        started = self.start_chat_preprocessing(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            request_id=request_id,
            chat_template_kwargs=chat_template_kwargs,
            server_received_at_ns=server_received_at_ns,
            entrance_timing=entrance_timing,
        )
        return await self.finish_chat_preprocessing(
            started,
            sampling_params=sampling_params,
            request_data=request_data,
        )

    def start_chat_preprocessing(
        self,
        *,
        messages: List[dict],
        add_generation_prompt: bool = True,
        request_id: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        server_received_at_ns: Optional[int] = None,
        entrance_timing: Optional[RequestEntranceTiming] = None,
    ) -> _StartedChatPreprocessing:
        """Render the prompt and submit tokenizer work to the executor now."""
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex}"
        if entrance_timing is None and getattr(self, "request_timing_enabled", True):
            entrance_timing = RequestEntranceTiming()

        if entrance_timing is not None:
            entrance_timing.preprocessing_started_at_ns = monotonic_ns()
        rendered_prompt = self.engine.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        if entrance_timing is not None:
            entrance_timing.chat_template_completed_at_ns = monotonic_ns()

        loop = asyncio.get_running_loop()
        tokenized_future = loop.run_in_executor(
            self._ensure_preprocess_executor(),
            functools.partial(
                self._tokenize_chat_input,
                prompt=rendered_prompt,
                messages=messages,
                entrance_timing=entrance_timing,
            ),
        )
        return _StartedChatPreprocessing(
            request_id=request_id,
            prompt=rendered_prompt,
            tokenized_future=tokenized_future,
            server_received_at_ns=server_received_at_ns,
            entrance_timing=entrance_timing,
        )

    async def finish_chat_preprocessing(
        self,
        started: _StartedChatPreprocessing,
        *,
        sampling_params: Optional[SamplingParams],
        request_data: Optional[dict],
    ) -> InferenceRequest:
        tokenized = await started.tokenized_future
        return self._request_from_tokenized_input(
            request_id=started.request_id,
            prompt=started.prompt,
            tokenized=tokenized,
            sampling_params=sampling_params,
            server_received_at_ns=started.server_received_at_ns,
            entrance_timing=started.entrance_timing,
            request_data=request_data,
        )

    def add_chat_request(
        self,
        messages: List[dict],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        request_data: Optional[dict] = None,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Optional[dict] = None,
        server_received_at_ns: Optional[int] = None,
        entrance_timing: Optional[RequestEntranceTiming] = None,
    ) -> InferenceRequest:
        """Add a chat request to the engine.

        Args:
            messages: List of message dicts (chat conversation).
            sampling_params: Sampling parameters.
            request_id: Optional request ID.
            request_data: Optional request data dict.

        Returns:
            The created InferenceRequest object.
        """

        return self.add_request(
            messages=messages,
            apply_chat_template=True,
            add_generation_prompt=add_generation_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            request_data=request_data,
            chat_template_kwargs=chat_template_kwargs,
            server_received_at_ns=server_received_at_ns,
            entrance_timing=entrance_timing,
        )

    async def add_chat_request_async(
        self,
        messages: List[dict],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        request_data: Optional[dict] = None,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Optional[dict] = None,
        server_received_at_ns: Optional[int] = None,
        entrance_timing: Optional[RequestEntranceTiming] = None,
    ) -> InferenceRequest:
        """Add an OpenAI chat request through the preprocessing executor."""
        return await self.add_request_async(
            messages=messages,
            apply_chat_template=True,
            add_generation_prompt=add_generation_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            request_data=request_data,
            chat_template_kwargs=chat_template_kwargs,
            server_received_at_ns=server_received_at_ns,
            entrance_timing=entrance_timing,
        )

    def _timing_role(self) -> str:
        """返回当前 worker 在 timing 日志中的角色标签。"""
        # 角色来自静态 KV 配置，而不是请求参数；同一个 worker 保持一致。
        kv_config = getattr(self.config, "kv_transfer_config", None)
        kv_role = kv_config and kv_config.kv_role
        if kv_role == "kv_producer":
            return "prefill"
        if kv_role == "kv_consumer":
            return "decode"
        return "aggregated"

    async def stream_request(
        self,
        request: InferenceRequest,
        timeout: float = 100.0,
        request_timeout: Optional[float] = None,
    ) -> AsyncIterator[TokenOutput]:
        """Stream tokens from a request.

        Args:
            request: The inference request to stream from.
            timeout: Timeout for waiting on each token.

        Yields:
            TokenOutput objects for each generated token.
        """
        import asyncio

        class TimeoutMarker:
            __slots__ = ()

        start = time.time()
        output_queue = request.output_queue.async_q
        loop = asyncio.get_running_loop()
        timeout_handle: asyncio.TimerHandle | None = None
        ready_drain_items = getattr(self, "_stream_ready_drain_items", 4)

        def record_stream_stat(name: str) -> None:
            setattr(self, name, getattr(self, name, 0) + 1)

        def start_token_timer() -> None:
            nonlocal timeout_handle
            if timeout is None:
                return

            def on_token_timeout():
                nonlocal timeout_handle
                timeout_handle = None
                output_queue.put_nowait(TimeoutMarker())

            timeout_handle = loop.call_later(float(timeout), on_token_timeout)
            record_stream_stat("_stream_timer_start_count")

        def cancel_token_timer() -> None:
            nonlocal timeout_handle
            if timeout_handle is not None:
                timeout_handle.cancel()
                timeout_handle = None

        def expand_queue_item(queue_item):
            if isinstance(queue_item, TokenBatch):
                return queue_item.outputs
            return [queue_item]

        def accept_queue_item(queue_item, timer_was_active: bool):
            token_outputs = expand_queue_item(queue_item)
            token_timings = [
                getattr(token_output, "output_timing", None)
                for token_output in token_outputs
                if getattr(token_output, "output_timing", None) is not None
            ]
            timestamp_ns = monotonic_ns()
            for token_timing in token_timings:
                token_timing.queue_get_returned_at_ns = timestamp_ns
                if timer_was_active:
                    token_timing.timer_cancel_started_at_ns = timestamp_ns
                    token_timing.timer_cancel_completed_at_ns = timestamp_ns
                token_timing.queue_task_done_started_at_ns = timestamp_ns

            output_queue.task_done()
            timestamp_ns = monotonic_ns()
            for token_timing in token_timings:
                token_timing.queue_task_done_completed_at_ns = timestamp_ns
            return token_outputs

        async def retrieve_queue_items() -> list[tuple[object, bool]]:
            queue_items: list[tuple[object, bool]] = []
            timer_was_active = False

            if ready_drain_items == 0:
                start_token_timer()
                timer_was_active = timeout_handle is not None
                queue_items.append((await output_queue.get(), timer_was_active))
                cancel_token_timer()
                record_stream_stat("_stream_queue_wait_count")
            else:
                try:
                    queue_items.append((output_queue.get_nowait(), False))
                    record_stream_stat("_stream_queue_nowait_count")
                except asyncio.QueueEmpty:
                    start_token_timer()
                    timer_was_active = timeout_handle is not None
                    queue_items.append((await output_queue.get(), timer_was_active))
                    cancel_token_timer()
                    record_stream_stat("_stream_queue_wait_count")

                while len(queue_items) < ready_drain_items:
                    try:
                        queue_items.append((output_queue.get_nowait(), False))
                        record_stream_stat("_stream_queue_nowait_count")
                    except asyncio.QueueEmpty:
                        break

            if len(queue_items) > 1:
                record_stream_stat("_stream_ready_drain_count")
            return queue_items

        try:
            pending_queue_items: list[tuple[object, bool]] = []
            while True:
                try:
                    if request_timeout and time.time() - start > float(request_timeout):
                        logger.warning(
                            f"Request {request.request_id} exceeded request timeout of {request_timeout} seconds"
                        )
                        self.add_aborted_req(request, FinishReason.TIMEOUT)

                    if not pending_queue_items:
                        pending_queue_items = await retrieve_queue_items()

                    queue_item, timer_was_active = pending_queue_items.pop(0)
                    token_outputs = accept_queue_item(queue_item, timer_was_active)

                    if isinstance(queue_item, TimeoutMarker):
                        logger.warning(
                            f"Timeout while waiting for token from request {request.request_id}"
                        )
                        if request.is_aborted():
                            for pending_item, _ in pending_queue_items:
                                for token_output in accept_queue_item(
                                    pending_item, False
                                ):
                                    yield token_output
                            pending_queue_items.clear()
                            while not output_queue.empty():
                                try:
                                    queue_item = output_queue.get_nowait()
                                    output_queue.task_done()
                                    if isinstance(queue_item, TimeoutMarker):
                                        continue
                                    for token_output in expand_queue_item(queue_item):
                                        yield token_output
                                except asyncio.QueueEmpty:
                                    break

                            yield TokenOutput(
                                request_id=request.request_id,
                                token_id=-1,
                                token_text="",
                                finished=True,
                                finish_reason=request.finish_reason,
                                generated_text=request.generated_text,
                            )
                            break
                        continue

                    for token_output in token_outputs:
                        yield token_output

                        if token_output.finished:
                            break
                    else:
                        continue
                    break
                except Exception as e:
                    logger.error(
                        f"Error while streaming request {request.request_id}: {e}"
                    )
                    break
        finally:
            # Unified cleanup point: runs whether the loop exits normally,
            # via exception, or via aclose() (GeneratorExit from Starlette).
            cancel_token_timer()
            await request.close()
