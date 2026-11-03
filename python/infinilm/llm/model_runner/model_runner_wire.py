"""ModelRunner 子进程通信协议层。

该模块负责 request full/diff snapshot、batch layout、Mooncake metadata 和
ModelRunnerOutput 的跨进程编解码。传输细节由 model_runner_channel.py 提供。
"""

from __future__ import annotations

import os
from array import array
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from infinilm.llm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from infinilm.llm.model_runner.model_runner import ModelRunnerOutput
    from infinilm.llm.request import InferenceRequest
    from infinilm.llm.scheduler import SchedulerOutput


def model_runner_request_cache_enabled() -> bool:
    """读取是否在 worker 侧缓存请求快照并只发送 decode 增量。"""
    value = os.environ.get("INFINILM_MODEL_RUNNER_REQUEST_CACHE", "1")
    normalized = value.strip().lower()
    if normalized not in {
        "0",
        "1",
        "false",
        "true",
        "no",
        "yes",
        "off",
        "on",
    }:
        raise RuntimeError(
            f"INFINILM_MODEL_RUNNER_REQUEST_CACHE must be a boolean, got {value!r}"
        )
    return normalized in {"1", "true", "yes", "on"}


def _int64_array(values) -> "array[int]":
    """Convert an integer sequence to a compact pickle representation."""
    return array("q", values)


def _int64_array_values(value) -> list[int]:
    return value.tolist() if isinstance(value, array) else list(value)


def model_runner_fixed_array_enabled() -> bool:
    """读取是否使用平铺 int 数组 decode wire layout。

    该布局为 shared-memory/zero-copy 协议预留；在当前 Pipe + pickle 路径
    中，每次创建 array 和 offsets 会抵消 payload 缩减收益，因此默认关闭。
    """
    value = os.environ.get("INFINILM_MODEL_RUNNER_FIXED_ARRAY", "0")
    normalized = value.strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise RuntimeError(
            f"INFINILM_MODEL_RUNNER_FIXED_ARRAY must be a boolean, got {value!r}"
        )
    return normalized in {"1", "true", "yes", "on"}


def _sampling_to_dict(params: SamplingParams) -> dict[str, Any]:
    """将采样参数转换成 Pipe 可以传输的字典。"""
    return asdict(params)


def metadata_to_wire(metadata: Any) -> dict[str, Any] | None:
    """序列化 scheduler 侧的 Mooncake 元数据。

    P/D 场景中，scheduler 会先为请求决定发送或接收哪些 KV block。元数据
    中包含远端 engine、transfer id、远端 bootstrap 地址和本地 block id，
    worker 侧需要这些信息才能真正执行 Mooncake transfer。

    这里不能直接传输 MooncakeConnectorMetadata 实例：它包含默认字典、集合
    以及 connector 专用对象。显式转换为字典后，协议内容也更容易审查和扩展。
    """
    if metadata is None:
        return None
    return {
        "reqs_to_recv": {
            engine_id: {
                request_id: {
                    "d_req_id": meta.d_req_id,
                    "transfer_id": meta.transfer_id,
                    "local_block_ids": list(meta.local_block_ids),
                    "remote_engine_id": meta.remote_engine_id,
                    "remote_bootstrap_addr": meta.remote_bootstrap_addr,
                }
                for request_id, meta in requests.items()
            }
            for engine_id, requests in metadata.reqs_to_recv.items()
        },
        "reqs_to_send": {
            request_id: [transfer_id, list(block_ids)]
            for request_id, (transfer_id, block_ids) in metadata.reqs_to_send.items()
        },
        "reqs_not_processed": list(metadata.reqs_not_processed),
    }


def metadata_from_wire(payload: dict[str, Any] | None):
    """将 Pipe 中的 Mooncake 元数据恢复成 connector 使用的对象。"""
    from infinilm.kv_connector.mooncake.mooncake_connector import (
        MooncakeConnectorMetadata,
        PullReqMeta,
    )

    if payload is None:
        return None
    metadata = MooncakeConnectorMetadata()
    for engine_id, requests in payload.get("reqs_to_recv", {}).items():
        for request_id, raw in requests.items():
            metadata.reqs_to_recv[engine_id][request_id] = PullReqMeta(
                d_req_id=raw["d_req_id"],
                transfer_id=raw["transfer_id"],
                local_block_ids=list(raw["local_block_ids"]),
                remote_engine_id=raw["remote_engine_id"],
                remote_bootstrap_addr=raw["remote_bootstrap_addr"],
            )
    metadata.reqs_to_send = {
        request_id: (raw[0], list(raw[1]))
        for request_id, raw in payload.get("reqs_to_send", {}).items()
    }
    metadata.reqs_not_processed = set(payload.get("reqs_not_processed", []))
    return metadata


def request_to_wire(request: "InferenceRequest") -> dict[str, Any]:
    """提取 ModelRunner 执行一个请求所需的最小状态。

    ``InferenceRequest`` 由 scheduler 管理，不能直接跨进程传递。这里保留
    prompt、已生成 token、paged KV cache 的 block table/slot mapping、采样
    参数和 KV transfer 参数；请求对象中的锁、future、回调等运行时状态全部
    留在 scheduler 进程。

    当前进程模式只支持纯文本请求。多模态输入通常包含不可直接 pickle 的
    tensor 或处理器状态，暂时由旧的单进程路径处理。
    """
    if request.has_multimodal_inputs:
        raise NotImplementedError(
            "INFINILM_MODEL_RUNNER_PROCESS currently supports text requests only"
        )
    if request.processed_inputs is not None:
        raise NotImplementedError(
            "INFINILM_MODEL_RUNNER_PROCESS does not transfer processed multimodal inputs"
        )
    return {
        "kind": "full",
        "request_id": request.request_id,
        "prompt_token_ids": list(request.prompt_token_ids),
        "generated_token_ids": list(request.generated_token_ids),
        "block_table": list(request.block_table),
        "slot_mapping": list(request.slot_mapping),
        "num_local_cached_tokens": request.num_local_cached_tokens,
        "num_computed_tokens": request.num_computed_tokens,
        "num_blocks": request.num_blocks,
        "kv_transfer_params": request.kv_transfer_params,
        "sampling_params": _sampling_to_dict(request.sampling_params),
        "is_multimodal": False,
    }


def request_to_diff(
    request: "InferenceRequest", known_generated_tokens: int
) -> dict[str, Any]:
    """构造已缓存请求的 decode 增量。"""
    return {
        "kind": "diff",
        "request_id": request.request_id,
        "base_generated_tokens": known_generated_tokens,
        "new_generated_token_ids": list(
            request.generated_token_ids[known_generated_tokens:]
        ),
        "block_table": list(request.block_table),
        "slot_mapping": list(request.slot_mapping),
        "num_local_cached_tokens": request.num_local_cached_tokens,
        "num_computed_tokens": request.num_computed_tokens,
        "num_blocks": request.num_blocks,
        "kv_transfer_params": request.kv_transfer_params,
    }


class WireRequest:
    """子进程中供 processor、ModelRunner 和 connector 使用的轻量请求视图。

    这不是 ``InferenceRequest`` 的完整替代品，只实现当前 forward 和 KV
    connector 访问到的字段及方法。这样可以避免把 scheduler 的请求对象、
    asyncio 状态和线程相关对象带入子进程。
    """

    def __init__(self, payload: dict[str, Any]):
        self.request_id = payload["request_id"]
        self.prompt_token_ids = tuple(payload["prompt_token_ids"])
        self.generated_token_ids = list(payload["generated_token_ids"])
        self.block_table = list(payload["block_table"])
        self.slot_mapping = list(payload["slot_mapping"])
        self.num_local_cached_tokens = payload["num_local_cached_tokens"]
        self.num_computed_tokens = payload["num_computed_tokens"]
        self.num_blocks = payload["num_blocks"]
        self.kv_transfer_params = payload["kv_transfer_params"]
        self.sampling_params = SamplingParams(**payload["sampling_params"])
        self.has_multimodal_inputs = payload.get("is_multimodal", False)
        self.processed_inputs = None
        self.timing_steps = []
        self.timing_mode = "full"
        self.mamba_cache_index = None

    def get_prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    def get_input_tokens(self):
        return self.prompt_token_ids

    def get_total_length(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def apply_diff(self, payload: dict[str, Any]) -> None:
        """将 scheduler 发来的 decode 增量应用到 worker 缓存对象。"""
        self.apply_diff_values(
            new_generated_token_ids=payload["new_generated_token_ids"],
            block_table=payload["block_table"],
            slot_mapping=payload["slot_mapping"],
            num_local_cached_tokens=payload["num_local_cached_tokens"],
            num_computed_tokens=payload["num_computed_tokens"],
            num_blocks=payload["num_blocks"],
            kv_transfer_params=payload["kv_transfer_params"],
            base_generated_tokens=payload["base_generated_tokens"],
        )

    def apply_diff_values(
        self,
        *,
        new_generated_token_ids,
        block_table,
        slot_mapping,
        num_local_cached_tokens: int,
        num_computed_tokens: int,
        num_blocks: int,
        kv_transfer_params,
        base_generated_tokens: int,
    ) -> None:
        if len(self.generated_token_ids) != base_generated_tokens:
            raise RuntimeError(
                "ModelRunner worker request cache generation mismatch for "
                f"{self.request_id}: expected={base_generated_tokens}, "
                f"actual={len(self.generated_token_ids)}"
            )
        self.generated_token_ids.extend(new_generated_token_ids)
        self.block_table = list(block_table)
        self.slot_mapping = list(slot_mapping)
        self.num_local_cached_tokens = num_local_cached_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_blocks = num_blocks
        self.kv_transfer_params = kv_transfer_params


def batch_to_diff_array(
    scheduled_requests, request_state: dict[str, int] | None
) -> dict[str, Any] | None:
    """将全量 decode diff 编码为平铺 int 数组；混合 full/diff 时返回 None。"""
    if request_state is None or not scheduled_requests:
        return None

    request_ids = []
    base_generated_tokens = []
    new_generated_token_ids = []
    new_generated_token_offsets = [0]
    block_table = []
    block_table_offsets = [0]
    slot_mapping = []
    slot_mapping_offsets = [0]
    num_local_cached_tokens = []
    num_computed_tokens = []
    num_blocks = []
    kv_transfer_params = []

    for request in scheduled_requests:
        known = request_state.get(request.request_id)
        if known is None or len(request.generated_token_ids) < known:
            return None
        request_ids.append(request.request_id)
        base_generated_tokens.append(known)
        new_tokens = request.generated_token_ids[known:]
        new_generated_token_ids.extend(new_tokens)
        new_generated_token_offsets.append(len(new_generated_token_ids))
        block_table.extend(request.block_table)
        block_table_offsets.append(len(block_table))
        slot_mapping.extend(request.slot_mapping)
        slot_mapping_offsets.append(len(slot_mapping))
        num_local_cached_tokens.append(request.num_local_cached_tokens)
        num_computed_tokens.append(request.num_computed_tokens)
        num_blocks.append(request.num_blocks)
        kv_transfer_params.append(request.kv_transfer_params)

    for request in scheduled_requests:
        request_state[request.request_id] = len(request.generated_token_ids)

    return {
        "layout": "diff_array",
        "request_ids": request_ids,
        "base_generated_tokens": _int64_array(base_generated_tokens),
        "new_generated_token_ids": _int64_array(new_generated_token_ids),
        "new_generated_token_offsets": _int64_array(new_generated_token_offsets),
        "block_table": _int64_array(block_table),
        "block_table_offsets": _int64_array(block_table_offsets),
        "slot_mapping": _int64_array(slot_mapping),
        "slot_mapping_offsets": _int64_array(slot_mapping_offsets),
        "num_local_cached_tokens": _int64_array(num_local_cached_tokens),
        "num_computed_tokens": _int64_array(num_computed_tokens),
        "num_blocks": _int64_array(num_blocks),
        "kv_transfer_params": kv_transfer_params,
    }


def batch_to_wire(
    scheduler_output: "SchedulerOutput",
    request_state: dict[str, int] | None = None,
) -> dict[str, Any]:
    """将一次 scheduler 调度结果编码成跨进程 batch 消息。"""
    diff_array = (
        batch_to_diff_array(scheduler_output.scheduled_requests, request_state)
        if model_runner_fixed_array_enabled()
        else None
    )
    if diff_array is not None:
        diff_array.update(
            {
                "is_prefill": scheduler_output.is_prefill,
                "request_cache": {
                    "full": 0,
                    "diff": len(scheduler_output.scheduled_requests),
                },
                "kv_connector_metadata": metadata_to_wire(
                    scheduler_output.kv_connector_metadata
                ),
            }
        )
        return diff_array

    requests = []
    for request in scheduler_output.scheduled_requests:
        known_generated_tokens = (
            request_state.get(request.request_id) if request_state is not None else None
        )
        if (
            request_state is not None
            and known_generated_tokens is not None
            and len(request.generated_token_ids) >= known_generated_tokens
        ):
            requests.append(request_to_diff(request, known_generated_tokens))
        else:
            requests.append(request_to_wire(request))
        if request_state is not None:
            request_state[request.request_id] = len(request.generated_token_ids)

    return {
        "layout": "per_request",
        "is_prefill": scheduler_output.is_prefill,
        "requests": requests,
        "request_cache": {
            "full": sum(item["kind"] == "full" for item in requests),
            "diff": sum(item["kind"] == "diff" for item in requests),
        },
        "kv_connector_metadata": metadata_to_wire(
            scheduler_output.kv_connector_metadata
        ),
    }


def requests_from_wire(
    batch: dict[str, Any], cache: dict[str, WireRequest]
) -> list[WireRequest]:
    """从 full/diff 消息恢复本批请求，并在 worker 侧维护请求缓存。"""
    if batch.get("layout", "per_request") == "diff_array":
        request_ids = batch["request_ids"]
        bases = _int64_array_values(batch["base_generated_tokens"])
        new_tokens = _int64_array_values(batch["new_generated_token_ids"])
        token_offsets = _int64_array_values(batch["new_generated_token_offsets"])
        blocks = _int64_array_values(batch["block_table"])
        block_offsets = _int64_array_values(batch["block_table_offsets"])
        slots = _int64_array_values(batch["slot_mapping"])
        slot_offsets = _int64_array_values(batch["slot_mapping_offsets"])
        local_cached = _int64_array_values(batch["num_local_cached_tokens"])
        computed = _int64_array_values(batch["num_computed_tokens"])
        block_counts = _int64_array_values(batch["num_blocks"])

        requests = []
        for index, request_id in enumerate(request_ids):
            request = cache.get(request_id)
            if request is None:
                raise RuntimeError(
                    "ModelRunner worker request cache miss for "
                    f"{request_id}; a full snapshot is required"
                )
            request.apply_diff_values(
                new_generated_token_ids=new_tokens[
                    token_offsets[index] : token_offsets[index + 1]
                ],
                block_table=blocks[block_offsets[index] : block_offsets[index + 1]],
                slot_mapping=slots[slot_offsets[index] : slot_offsets[index + 1]],
                num_local_cached_tokens=local_cached[index],
                num_computed_tokens=computed[index],
                num_blocks=block_counts[index],
                kv_transfer_params=batch["kv_transfer_params"][index],
                base_generated_tokens=bases[index],
            )
            requests.append(request)
        return requests

    requests = []
    for payload in batch["requests"]:
        kind = payload.get("kind", "full")
        if kind == "full":
            request = WireRequest(payload)
            cache[request.request_id] = request
        elif kind == "diff":
            request_id = payload["request_id"]
            request = cache.get(request_id)
            if request is None:
                raise RuntimeError(
                    "ModelRunner worker request cache miss for "
                    f"{request_id}; a full snapshot is required"
                )
            request.apply_diff(payload)
        else:
            raise RuntimeError(f"unknown ModelRunner request wire kind: {kind}")
        requests.append(request)
    return requests


def response_to_wire(output: "ModelRunnerOutput") -> dict[str, Any]:
    """将 ModelRunner 输出编码成 scheduler 可以消费的普通数据。

    除采样 token 外，还要返回 KV connector 的完成集合，否则 scheduler 无法
    更新 P/D 请求状态。forward timing 也必须跨进程带回，才能继续写入请求
    timing 日志并分析 scheduler、forward 和 KV transfer 的时间关系。
    """
    connector = output.kv_connector_output
    timing = output.forward_timing
    sampled_token_ids = output.sampled_token_ids
    flat_sampled_token_ids: list[int] | None = None
    if model_runner_fixed_array_enabled():
        flat_sampled_token_ids = []
        for token_ids in sampled_token_ids:
            if isinstance(token_ids, int):
                flat_sampled_token_ids.append(token_ids)
            else:
                flat_sampled_token_ids = None
                break

    return {
        "req_ids": list(output.req_ids),
        "sampled_layout": (
            "int_array" if flat_sampled_token_ids is not None else "per_request"
        ),
        "sampled_token_ids": (
            _int64_array(flat_sampled_token_ids)
            if flat_sampled_token_ids is not None
            else sampled_token_ids
        ),
        "kv_connector_output": (
            None
            if connector is None
            else {
                "finished_sending": list(connector.finished_sending or []),
                "finished_recving": list(connector.finished_recving or []),
                "failed_recving": list(connector.failed_recving or []),
                "invalid_block_ids": list(connector.invalid_block_ids),
                "kv_connector_stats": connector.kv_connector_stats,
            }
        ),
        "forward_timing": (
            None
            if timing is None
            else {
                "started_at_ns": timing.engine_forward_started_at_ns,
                "completed_at_ns": timing.engine_forward_completed_at_ns,
                "forward_batch_id": timing.forward_batch_id,
                "source": timing.source,
            }
        ),
    }


def response_from_wire(payload: dict[str, Any]) -> ModelRunnerOutput:
    """恢复 scheduler 侧的 ModelRunnerOutput、KV 输出和 forward timing。"""
    from infinilm.llm.model_runner.model_runner import (
        KVConnectorOutput,
        ModelForwardTiming,
        ModelRunnerOutput,
    )

    raw_connector = payload.get("kv_connector_output")
    connector = None
    if raw_connector is not None:
        connector = KVConnectorOutput(
            finished_sending=set(raw_connector["finished_sending"]),
            finished_recving=set(raw_connector["finished_recving"]),
            failed_recving=set(raw_connector["failed_recving"]),
            invalid_block_ids=set(raw_connector["invalid_block_ids"]),
            kv_connector_stats=raw_connector["kv_connector_stats"],
        )
    raw_timing = payload.get("forward_timing")
    timing = None
    if raw_timing is not None:
        timing = ModelForwardTiming(
            engine_forward_started_at_ns=raw_timing["started_at_ns"],
            engine_forward_completed_at_ns=raw_timing["completed_at_ns"],
            forward_batch_id=raw_timing["forward_batch_id"],
            source=raw_timing["source"],
        )
    return ModelRunnerOutput(
        req_ids=list(payload["req_ids"]),
        sampled_token_ids=(
            _int64_array_values(payload["sampled_token_ids"])
            if payload.get("sampled_layout", "per_request") == "int_array"
            else payload["sampled_token_ids"]
        ),
        kv_connector_output=connector,
        forward_timing=timing,
    )
