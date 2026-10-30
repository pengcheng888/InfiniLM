"""
Request and Output - Data structures for inference requests and outputs.
"""

import asyncio
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import janus

from infinilm.llm.output_timing import OutputTokenTiming
from infinilm.llm.prefix_cache import (
    EMPTY_BLOCK_HASH,
    BlockHash,
    hash_block_tokens,
)
from infinilm.llm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


def monotonic_ns() -> int:
    """返回与 C++ forward timing 使用的同一 CLOCK_MONOTONIC 时间。"""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


class _SequenceView(Sequence):
    """Live read-only view over an internally mutable list."""

    def __init__(self, values: list) -> None:
        self._values = values

    def __getitem__(self, index):
        return self._values[index]

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(self._values)

    def __eq__(self, other: object) -> bool:
        return (
            list(self._values) == list(other) if isinstance(other, Sequence) else False
        )


class RequestStatus(Enum):
    """Status of an inference request."""

    # Pending
    WAITING = "waiting"
    WAITING_FOR_REMOTE_KVS = "waiting_for_remote_kvs"

    # Active
    RUNNING = "running"

    # Successful terminal
    FINISHED = "finished"

    # Abnormal terminal
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    FAILED = "failed"


class FinishReason(Enum):
    """Reason for finishing generation."""

    # Normal completion
    EOS_TOKEN = "eos_token"
    STOP_STRING = "stop_string"
    STOP = "stop"

    # Controlled truncation
    LENGTH = "length"

    # Abnormal termination
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class RequestOutput:
    """Output from a single generation request.

    Attributes:
        request_id: Unique identifier for the request.
        prompt: Original prompt text.
        prompt_token_ids: Token IDs of the prompt.
        outputs: List of generated outputs (for beam search, multiple outputs possible).
        finished: Whether generation is complete.
        finish_reason: Reason for finishing.
    """

    request_id: str
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    outputs: List["CompletionOutput"] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[FinishReason] = None


@dataclass
class CompletionOutput:
    """Single completion output.

    Attributes:
        index: Index of this output (for beam search).
        text: Generated text.
        token_ids: Generated token IDs.
        finish_reason: Reason for finishing.
    """

    index: int = 0
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[FinishReason] = None


@dataclass
class TokenOutput:
    """Output for a single generated token.

    Attributes:
        request_id: Unique identifier for the request.
        token_id: Generated token ID.
        token_text: Decoded text of the token.
        finished: Whether generation is complete.
        finish_reason: Reason for finishing.
        generated_text: Full generated text so far.
    """

    request_id: str
    token_id: int
    token_text: str
    finished: bool = False
    finish_reason: Optional[FinishReason] = None
    generated_text: str = ""
    output_timing: Optional[OutputTokenTiming] = None


@dataclass
class TokenBatch:
    """Multiple outputs from one request, published as one queue item."""

    request_id: str
    outputs: List[TokenOutput] = field(default_factory=list)


@dataclass
class RequestEntranceTiming:
    """HTTP 入口到 InferenceRequest 创建前的分段时间点。

    These timestamps are captured on the worker's monotonic clock and are only
    subtracted within the same process.
    """

    body_read_completed_at_ns: int | None = None
    json_parsed_at_ns: int | None = None
    route_prepared_at_ns: int | None = None
    handler_started_at_ns: int | None = None
    preprocessing_started_at_ns: int | None = None
    chat_template_completed_at_ns: int | None = None
    multimodal_resolved_at_ns: int | None = None
    tokenization_completed_at_ns: int | None = None
    prompt_token_ids_ready_at_ns: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "body_read_completed_at_ns": self.body_read_completed_at_ns,
            "json_parsed_at_ns": self.json_parsed_at_ns,
            "route_prepared_at_ns": self.route_prepared_at_ns,
            "handler_started_at_ns": self.handler_started_at_ns,
            "preprocessing_started_at_ns": self.preprocessing_started_at_ns,
            "chat_template_completed_at_ns": self.chat_template_completed_at_ns,
            "multimodal_resolved_at_ns": self.multimodal_resolved_at_ns,
            "tokenization_completed_at_ns": self.tokenization_completed_at_ns,
            "prompt_token_ids_ready_at_ns": self.prompt_token_ids_ready_at_ns,
        }


@dataclass
class RequestStepTiming:
    """一个请求在一次 prefill/decode step 中的关键时间点。

    所有时间都使用 ``CLOCK_MONOTONIC`` 纳秒。C++ forward 与 Python 侧
    打点必须使用同一时钟，不能混用 wall clock 和 perf_counter。
    """

    # 请求唯一 ID；同一个请求的每一轮 timing 都写入同一个 JSONL 文件。
    request_id: str
    # 当前请求第几次被调度执行；prefill 为 0，后续 decode 依次递增。
    step_index: int
    # 当前执行阶段：prefill 表示首段计算，decode 表示逐 token 增量生成。
    phase: str
    # 本轮 scheduler 构建的 batch 内请求数；engine forward 是 batch 级调用。
    batch_size: int
    # 同一次物理 engine forward 的批次 ID。同一 batch 内所有 request step
    # 共享该 ID，离线分析用它去重 request step，得到真实 forward 调用数。
    forward_batch_id: str | None = None
    # 记录产生者角色。P/D 可能跨机器部署，各角色只使用本进程
    # CLOCK_MONOTONIC，时间戳不能跨节点比较。
    role: str = "aggregated"
    # worker HTTP route 入口时间，早于 body 读取、JSON 解析和 tokenize。
    server_received_at_ns: int | None = None
    # InferenceRequest 完成创建的时间；v5 用于拆分入口预处理和入队前处理。
    request_created_at_ns: int | None = None
    # 请求完成预处理并进入 scheduler waiting queue 的时间。
    scheduler_enqueued_at_ns: int | None = None
    # scheduler 从 waiting/running queue 取出该请求、开始处理的时间。
    scheduler_polled_at_ns: int | None = None
    # scheduler 完成本轮 batch 构建、准备交给 ModelRunner 的时间。
    scheduler_batch_ready_at_ns: int | None = None
    # schedule() 入口处的三个 scheduler 状态深度。waiting 表示待 prefill
    # 调度，running 表示待 decode 调度，remote_kv 表示等待远端 KV ready。
    scheduler_waiting_queue_depth_before: int | None = None
    scheduler_running_queue_depth_before: int | None = None
    scheduler_remote_kv_requests_before: int | None = None
    # 本轮 batch 构建完成、即将交给 ModelRunner 时的状态深度。此时已被
    # 选中的请求不在三个队列中，数量由 batch_size 单独表示。
    scheduler_waiting_queue_depth_after: int | None = None
    scheduler_running_queue_depth_after: int | None = None
    scheduler_remote_kv_requests_after: int | None = None
    # ModelRunner 真正调用 self.model_engine.forward() 前的时间；
    # speculative 路径不采集该字段。
    engine_forward_started_at_ns: int | None = None
    # self.model_engine.forward() 返回后的时间；该值是 ModelRunner
    # 视角的 engine forward 完成点，不包含输出 tensor 转 Python list。
    engine_forward_completed_at_ns: int | None = None
    # v6 入口分段时间点。写日志时展开到 step 顶层，保持离线工具简单。
    entrance_timing: RequestEntranceTiming | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        """转换为可直接写入日志或导出的字典。"""
        payload = {
            "request_id": self.request_id,
            "role": self.role,
            "step_index": self.step_index,
            "phase": self.phase,
            "batch_size": self.batch_size,
            "forward_batch_id": self.forward_batch_id,
            "server_received_at_ns": self.server_received_at_ns,
            "request_created_at_ns": self.request_created_at_ns,
            "scheduler_enqueued_at_ns": self.scheduler_enqueued_at_ns,
            "scheduler_polled_at_ns": self.scheduler_polled_at_ns,
            "scheduler_batch_ready_at_ns": self.scheduler_batch_ready_at_ns,
            "scheduler_waiting_queue_depth_before": self.scheduler_waiting_queue_depth_before,
            "scheduler_running_queue_depth_before": self.scheduler_running_queue_depth_before,
            "scheduler_remote_kv_requests_before": self.scheduler_remote_kv_requests_before,
            "scheduler_waiting_queue_depth_after": self.scheduler_waiting_queue_depth_after,
            "scheduler_running_queue_depth_after": self.scheduler_running_queue_depth_after,
            "scheduler_remote_kv_requests_after": self.scheduler_remote_kv_requests_after,
            "engine_forward_started_at_ns": self.engine_forward_started_at_ns,
            "engine_forward_completed_at_ns": self.engine_forward_completed_at_ns,
        }
        if self.entrance_timing is not None:
            payload.update(self.entrance_timing.to_dict())
        return payload


class InferenceRequest:
    """Internal inference request object for managing generation state and resources."""

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        processed_inputs: Optional[dict] = None,
        mm_token_index_mappings: Optional[List[dict]] = None,
        sampling_params: Optional[SamplingParams] = None,
        eos_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        server_received_at_ns: Optional[int] = None,
        # timing 日志中的 worker 角色；未显式指定时按普通 aggregated 服务处理。
        timing_role: Optional[str] = None,
        # For server use
        request_data: Optional[dict] = None,
        *,
        entrance_timing: Optional[RequestEntranceTiming] = None,
        has_multimodal_inputs: bool = False,
    ):
        self.arrival_time: float = arrival_time or time.time()
        self.finished_time: Optional[float] = None
        # HTTP route 在解析 body 前记录该时间；arrival_time 是内部 request
        # 创建时间，二者不能混用。
        self.server_received_at_ns: Optional[int] = server_received_at_ns
        self.request_created_at_ns: Optional[int] = None
        self.scheduler_enqueued_at_ns: Optional[int] = None
        self.scheduler_polled_at_ns: Optional[int] = None
        self.first_scheduler_polled_at_ns: Optional[int] = None
        self.scheduler_batch_ready_at_ns: Optional[int] = None
        self.first_engine_forward_started_at_ns: Optional[int] = None
        self.first_engine_forward_completed_at_ns: Optional[int] = None
        self.last_engine_forward_completed_at_ns: Optional[int] = None
        self.engine_forward_total_ns: Optional[int] = None
        self.prefill_forward_ns: Optional[int] = None
        self.decode_step_count: int = 0
        self.sampled_decode_step_count: int = 0
        self.sampled_decode_forward_ns: int = 0
        self.sample_interval: int = 1
        self.timing_step_count: int = 0
        self.timing_mode: str = "full"
        self.timing_steps: List[RequestStepTiming] = []

        # Request metadata
        self.request_id: str = request_id
        # PD 场景最终会在 engine 侧根据 KV producer/consumer 配置校正角色。
        self.timing_role: str = timing_role or "aggregated"
        self.prompt: Optional[str] = prompt
        self._prompt_token_ids: tuple[int, ...] = (
            tuple(prompt_token_ids) if prompt_token_ids is not None else ()
        )
        self.prompt_length: int = len(self._prompt_token_ids)
        self.processed_inputs: Optional[dict] = processed_inputs
        self.mm_token_index_mappings: Optional[List[dict]] = mm_token_index_mappings
        self.has_multimodal_inputs: bool = has_multimodal_inputs or bool(
            mm_token_index_mappings
        )
        self.priority: int = 0

        # Sampling & stopping criteria
        self.sampling_params: SamplingParams = sampling_params or SamplingParams()
        self.eos_token_ids: List[int] = (
            eos_token_ids if eos_token_ids is not None else []
        )

        # Generation state
        self._generated_token_ids: List[int] = []
        self._generated_token_ids_view = _SequenceView(self._generated_token_ids)
        self.generated_text: str = ""  # generated_text == tokenizer.decode(generated_token_ids[:_token_decode_offset])
        self.status: RequestStatus = RequestStatus.WAITING
        self.finish_reason: Optional[FinishReason] = None

        # KV cache state
        self.block_table: List[int] = []  # Logical block to physical page ID
        self.slot_mapping: List[int] = []
        self.num_local_cached_tokens: int = (
            0  # Number of cached tokens visible to the current model step
        )
        self.num_computed_tokens: int = 0  # Total tokens computed (local + remote)
        # Longest local prefix-cache hit observed when the prompt was first
        # scheduled. Unlike num_local_cached_tokens, this value is not advanced
        # as decode produces additional tokens.
        self.num_local_prefix_cache_hit_tokens: int = 0
        self.num_blocks: int = 0
        # Incremental hashes for complete logical token blocks.
        self._block_hashes: List[BlockHash] = []
        self._block_hashes_view = _SequenceView(self._block_hashes)
        self._hash_block_size: Optional[int] = None
        self._prefix_caching_enabled: bool = False
        self._hash_tail_token_ids: List[int] = []
        # Number of leading full blocks published to the prefix-cache index.
        self.num_cache_indexed_blocks: int = 0

        # Mamba cache management. None means no mamba cache row is currently owned.
        self.mamba_cache_index: Optional[int] = None

        # Qwen-style MRoPE decode offset. It is zero for pure text requests.
        self.mrope_position_delta: int = 0

        # PD disaggregation support
        self.kv_transfer_params: Optional[dict] = (
            None  # KV transfer parameters from the router
        )

        # For server use
        self.request_data: Optional[dict] = request_data
        self.entrance_timing: Optional[RequestEntranceTiming] = entrance_timing

        # Async output & streaming
        self._output_queue: Optional[janus.Queue] = None
        self._aborted: bool = False
        self._text_output_offset: int = 0
        self._token_decode_offset: int = 0

    @property
    def output_queue(self) -> janus.Queue:
        """Lazy initialization of output queue."""
        if self._output_queue is None:
            self._output_queue = janus.Queue()
        return self._output_queue

    @property
    def prompt_token_ids(self) -> tuple[int, ...]:
        return self._prompt_token_ids

    @property
    def generated_token_ids(self) -> Sequence[int]:
        return self._generated_token_ids_view

    @property
    def block_hashes(self) -> Sequence[BlockHash]:
        return self._block_hashes_view

    def get_prompt_length(self) -> int:
        return self.prompt_length

    def get_input_tokens(self) -> Sequence[int]:
        return self._prompt_token_ids

    def get_num_generated_tokens(self) -> int:
        return len(self._generated_token_ids)

    def get_total_length(self) -> int:
        return self.prompt_length + len(self._generated_token_ids)

    def get_all_token_ids(self) -> List[int]:
        return list(self._prompt_token_ids) + self._generated_token_ids

    def initialize_block_hashes(
        self, block_size: int, enable_prefix_caching: bool
    ) -> None:
        """Initialize incremental block hashing for this request."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if self._hash_block_size is not None:
            if (
                self._hash_block_size != block_size
                or self._prefix_caching_enabled != enable_prefix_caching
            ):
                raise RuntimeError("request block hashing is already initialized")
            return

        self._hash_block_size = block_size
        self._prefix_caching_enabled = enable_prefix_caching
        if not enable_prefix_caching:
            return

        num_full_blocks = self.prompt_length // block_size
        parent_hash = EMPTY_BLOCK_HASH
        for block_idx in range(num_full_blocks):
            start = block_idx * block_size
            end = start + block_size
            parent_hash = hash_block_tokens(
                self.prompt_token_ids[start:end], parent_hash
            )
            self._block_hashes.append(parent_hash)

        tail_start = num_full_blocks * block_size
        self._hash_tail_token_ids = list(self._prompt_token_ids[tail_start:])

    def append_generated_token_id(self, token_id: int) -> None:
        """Append one output token and extend the hash chain at block boundaries."""
        self._generated_token_ids.append(token_id)
        if not self._prefix_caching_enabled:
            return

        self._hash_tail_token_ids.append(token_id)
        if len(self._hash_tail_token_ids) == self._hash_block_size:
            parent_hash = (
                self._block_hashes[-1] if self._block_hashes else EMPTY_BLOCK_HASH
            )
            self._block_hashes.append(
                hash_block_tokens(self._hash_tail_token_ids, parent_hash)
            )
            self._hash_tail_token_ids.clear()

    def get_num_blocks_required(self, block_size: int) -> int:
        total_tokens = self.get_total_length()
        return (total_tokens + block_size - 1) // block_size

    def get_max_tokens(self) -> Optional[int]:
        return self.sampling_params.max_tokens

    def get_mm_token_index_mappings(self) -> Optional[List[dict]]:
        return self.mm_token_index_mappings

    def get_timing_summary(self) -> dict[str, int | str | None]:
        """汇总首步排队路径和全部 C++ forward 耗时。"""
        if not self.timing_steps:
            if self.timing_mode != "summary":
                return {"request_id": self.request_id, "steps": 0}

            queue_ns = (
                self.first_scheduler_polled_at_ns - self.server_received_at_ns
                if self.first_scheduler_polled_at_ns is not None
                and self.server_received_at_ns is not None
                else None
            )
            scheduler_ns = (
                self.scheduler_batch_ready_at_ns - self.first_scheduler_polled_at_ns
                if self.scheduler_batch_ready_at_ns is not None
                and self.first_scheduler_polled_at_ns is not None
                else None
            )
            scheduler_batch_ready_to_engine_forward_start_ns = (
                self.first_engine_forward_started_at_ns
                - self.scheduler_batch_ready_at_ns
                if self.first_engine_forward_started_at_ns is not None
                and self.scheduler_batch_ready_at_ns is not None
                else None
            )
            generated_steps = self.get_num_generated_tokens()
            exact_total_ns = None
            estimated_total_ns = None
            if self.sample_interval == 1:
                exact_total_ns = (
                    self.prefill_forward_ns or 0
                ) + self.sampled_decode_forward_ns
            elif self.decode_step_count == 0:
                estimated_total_ns = self.prefill_forward_ns or 0
            elif self.sampled_decode_step_count > 0:
                sampled_decode_average_ns = (
                    self.sampled_decode_forward_ns / self.sampled_decode_step_count
                )
                estimated_total_ns = int(
                    (self.prefill_forward_ns or 0)
                    + sampled_decode_average_ns * self.decode_step_count
                )
            return {
                "request_id": self.request_id,
                "role": self.timing_role,
                "steps": generated_steps,
                "server_received_at_ns": self.server_received_at_ns,
                "request_created_at_ns": self.request_created_at_ns,
                **(
                    self.entrance_timing.to_dict()
                    if self.entrance_timing is not None
                    else {}
                ),
                "scheduler_enqueued_at_ns": self.scheduler_enqueued_at_ns,
                "first_scheduler_polled_at_ns": self.first_scheduler_polled_at_ns,
                "scheduler_batch_ready_at_ns": self.scheduler_batch_ready_at_ns,
                "first_engine_forward_started_at_ns": (
                    self.first_engine_forward_started_at_ns
                ),
                "first_engine_forward_completed_at_ns": (
                    self.first_engine_forward_completed_at_ns
                ),
                "last_engine_forward_completed_at_ns": (
                    self.last_engine_forward_completed_at_ns
                ),
                "queue_ns": queue_ns,
                "scheduler_ns": scheduler_ns,
                "scheduler_batch_ready_to_engine_forward_start_ns": (
                    scheduler_batch_ready_to_engine_forward_start_ns
                ),
                "sample_interval": self.sample_interval,
                "decode_step_count": self.decode_step_count,
                "sampled_decode_step_count": self.sampled_decode_step_count,
                "sampled_decode_forward_ns": self.sampled_decode_forward_ns,
                "engine_forward_total_ns": exact_total_ns,
                **(
                    {"engine_forward_total_estimate_ns": estimated_total_ns}
                    if self.sample_interval > 1
                    else {}
                ),
            }

        first = self.timing_steps[0]
        last = self.timing_steps[-1]
        engine_forward_total_ns = sum(
            step.engine_forward_completed_at_ns - step.engine_forward_started_at_ns
            for step in self.timing_steps
            if step.engine_forward_completed_at_ns is not None
            and step.engine_forward_started_at_ns is not None
        )
        queue_ns = (
            first.scheduler_polled_at_ns - first.server_received_at_ns
            if first.scheduler_polled_at_ns is not None
            and first.server_received_at_ns is not None
            else None
        )
        scheduler_ns = (
            first.scheduler_batch_ready_at_ns - first.scheduler_polled_at_ns
            if first.scheduler_batch_ready_at_ns is not None
            and first.scheduler_polled_at_ns is not None
            else None
        )
        scheduler_batch_ready_to_engine_forward_start_ns = (
            first.engine_forward_started_at_ns - first.scheduler_batch_ready_at_ns
            if first.engine_forward_started_at_ns is not None
            and first.scheduler_batch_ready_at_ns is not None
            else None
        )
        return {
            "request_id": self.request_id,
            "role": self.timing_role,
            "steps": len(self.timing_steps),
            "server_received_at_ns": first.server_received_at_ns,
            "request_created_at_ns": first.request_created_at_ns,
            **(
                first.entrance_timing.to_dict()
                if first.entrance_timing is not None
                else {}
            ),
            "first_scheduler_polled_at_ns": first.scheduler_polled_at_ns,
            "first_engine_forward_started_at_ns": first.engine_forward_started_at_ns,
            "last_engine_forward_completed_at_ns": (
                last.engine_forward_completed_at_ns
            ),
            "queue_ns": queue_ns,
            "scheduler_ns": scheduler_ns,
            "scheduler_batch_ready_to_engine_forward_start_ns": (
                scheduler_batch_ready_to_engine_forward_start_ns
            ),
            "engine_forward_total_ns": engine_forward_total_ns,
        }

    def record_summary_step(self, batch_ready_at_ns: int) -> None:
        """Record the first scheduler boundary without per-step retention."""
        if self.scheduler_polled_at_ns is None:
            self.scheduler_polled_at_ns = batch_ready_at_ns
        if self.first_scheduler_polled_at_ns is None:
            self.first_scheduler_polled_at_ns = self.scheduler_polled_at_ns
        if self.scheduler_batch_ready_at_ns is None:
            self.scheduler_batch_ready_at_ns = batch_ready_at_ns

    def record_summary_prefill_forward(
        self,
        engine_forward_started_at_ns: int,
        engine_forward_completed_at_ns: int,
    ) -> None:
        """Record the exact first prefill forward."""
        duration_ns = engine_forward_completed_at_ns - engine_forward_started_at_ns
        if self.first_engine_forward_started_at_ns is None:
            self.first_engine_forward_started_at_ns = engine_forward_started_at_ns
            self.first_engine_forward_completed_at_ns = engine_forward_completed_at_ns
        self.last_engine_forward_completed_at_ns = engine_forward_completed_at_ns
        self.prefill_forward_ns = duration_ns

    def record_summary_decode_forward(
        self,
        engine_forward_started_at_ns: int,
        engine_forward_completed_at_ns: int,
    ) -> None:
        """Count every decode step but sample only periodic durations."""
        self.decode_step_count += 1
        if self.first_engine_forward_started_at_ns is None:
            self.first_engine_forward_started_at_ns = engine_forward_started_at_ns
            self.first_engine_forward_completed_at_ns = engine_forward_completed_at_ns
        self.last_engine_forward_completed_at_ns = engine_forward_completed_at_ns

        should_sample = (
            self.sampled_decode_step_count == 0
            or self.decode_step_count % self.sample_interval == 0
        )
        if should_sample:
            duration_ns = engine_forward_completed_at_ns - engine_forward_started_at_ns
            self.sampled_decode_forward_ns += duration_ns
            self.sampled_decode_step_count += 1

    def is_finished(self) -> bool:
        return self.status in [
            RequestStatus.FINISHED,
            RequestStatus.CANCELED,
            RequestStatus.FAILED,
            RequestStatus.TIMEOUT,
        ]

    def abort(self):
        """Signal that the request has been aborted and should stop generation."""
        self._aborted = True

    def is_aborted(self) -> bool:
        """Check if the request has been aborted."""
        return self._aborted

    def mark_finished(self, reason: FinishReason):
        """Mark the request as finished with the given reason."""
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.finished_time = time.time()

    def mark_failed(self, reason: FinishReason = FinishReason.ERROR):
        """Mark the request as failed."""
        self.abort()
        self.status = RequestStatus.FAILED
        self.finish_reason = reason
        self.finished_time = time.time()

    def mark_canceled(self):
        """Mark the request as canceled."""
        self.abort()
        self.status = RequestStatus.CANCELED
        self.finish_reason = FinishReason.CANCELED
        self.finished_time = time.time()

    def mark_timeout(self):
        """Mark the request as timed out."""
        self.abort()
        self.status = RequestStatus.TIMEOUT
        self.finish_reason = FinishReason.TIMEOUT
        self.finished_time = time.time()

    async def close(self):
        """Close the output queue and clean up resources."""
        if self._output_queue is not None:
            self.abort()
            try:
                while not self._output_queue.async_q.empty():
                    try:
                        self._output_queue.async_q.get_nowait()
                        self._output_queue.async_q.task_done()
                    except asyncio.QueueEmpty:
                        break
            except Exception as e:
                logger.error(
                    f"Error while clearing output queue for request {self.request_id}: {e}"
                )
                pass

            self._output_queue.close()
            try:
                await asyncio.wait_for(self._output_queue.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("wait_closed timeout, force close")

    def to_request_output(self) -> RequestOutput:
        """Convert to RequestOutput for external use."""
        return RequestOutput(
            request_id=self.request_id,
            prompt=self.prompt,
            prompt_token_ids=list(self._prompt_token_ids),
            outputs=[
                CompletionOutput(
                    index=0,
                    text=self.generated_text,
                    token_ids=list(self._generated_token_ids),
                    finish_reason=self.finish_reason,
                )
            ],
            finished=self.is_finished(),
            finish_reason=self.finish_reason,
        )
