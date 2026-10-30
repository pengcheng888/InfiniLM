"""轻量集群通信模式的 KV 分配和专用 model runner。"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from time import monotonic_ns
from typing import Any, Generator, Mapping

import infinicore

from infinilm.cluster.environment import (
    cluster_communication_test_kv_blocks,
    cluster_communication_test_token_id,
)
from infinilm.config.engine_config import EngineConfig
from infinilm.infer_engine import read_hf_config
from infinilm.kv_connector import KVConnectorFactory, KVConnectorRole
from infinilm.llm.model_runner.model_runner import (
    KVConnectorOutput,
    ModelForwardTiming,
    ModelRunnerOutput,
)
from infinilm.processors import AutoInfinilmProcessor

logger = logging.getLogger(__name__)


def _required_positive_int(config: Mapping[str, Any], name: str) -> int:
    """从 HF 配置读取必填正整数，缺失或非法时给出明确错误。"""
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            "cluster communication test KV registration requires a positive "
            f"integer {name} in model config, got {value!r}"
        )
    return value


def create_communication_test_kv_caches(
    hf_config: Mapping[str, Any],
    config: EngineConfig,
    num_blocks: int,
    device: infinicore.device,
) -> dict[str, infinicore.Tensor]:
    """分配通信测试注册所需的每层真实 KV tensor。

    通信测试模式不创建 InferEngine，因此不能调用 ``get_kv_cache()``。这里
    按照 HF 配置和当前 attention backend 手工分配与真实 Paged KV 相同布局的
    bfloat16 tensor。Mooncake 只负责注册/传输这些地址，不执行 attention，
    所以 tensor 内容保持为零即可。
    """
    text_config = hf_config.get("text_config", hf_config)
    num_layers = _required_positive_int(text_config, "num_hidden_layers")
    num_kv_heads = _required_positive_int(text_config, "num_key_value_heads")

    # 大多数 Llama 系配置没有显式 head_dim，此时由 hidden_size 与 Q 头数推导。
    head_dim = text_config.get("head_dim")
    if head_dim is None:
        hidden_size = _required_positive_int(text_config, "hidden_size")
        num_attention_heads = _required_positive_int(text_config, "num_attention_heads")
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "cluster communication test requires hidden_size divisible by "
                "num_attention_heads when head_dim is absent"
            )
        head_dim = hidden_size // num_attention_heads
    elif not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise ValueError(
            f"cluster communication test requires positive head_dim, got {head_dim!r}"
        )

    if config.attn_backend == "flash-attn":
        # FLASH_ATTN 使用 BSHD: [K/V, block, token, kv_head, head_dim]。
        shape = (2, num_blocks, config.block_size, num_kv_heads, head_dim)
    else:
        # PAGED_ATTN 使用 BHSD: [K/V, block, kv_head, token, head_dim]。
        shape = (2, num_blocks, num_kv_heads, config.block_size, head_dim)

    kv_caches: dict[str, infinicore.Tensor] = {}
    for layer_idx in range(num_layers):
        layer_name = f"rank.0.model.layers.{layer_idx}.self_attn.attn"
        kv_caches[layer_name] = infinicore.zeros(
            shape,
            dtype=infinicore.bfloat16,
            device=device,
        )
    return kv_caches


class CommunicationModelRunner:
    """不加载模型、只验证集群和 KV 传输链路的轻量通信 runner。"""

    def __init__(
        self,
        config: EngineConfig,
        hf_config: Mapping[str, Any] | None = None,
        initialize_processor: bool = True,
    ) -> None:
        # hf_config 由 LLMEngine 传入可以避免同一配置文件被重复读取；单独
        # 构造测试对象时也允许 runner 自己读取一次。
        self.config = config
        self.hf_config = dict(hf_config or read_hf_config(config.model_path))
        self._closed = False
        self.communication_test_mode = True
        self.kv_transfer_config = config.kv_transfer_config
        # 通信测试没有真实 engine forward，但仍需要稳定 batch ID 供 timing
        # 去重和 Mooncake 事件关联使用。
        self._forward_batch_sequence = 0

        self._validate_config()
        self._init_device()

        self.model_engine = None
        self.speculative_runner = None
        self.pipeline_control = None
        self.processor = (
            AutoInfinilmProcessor.from_pretrained(config.model_path)
            if initialize_processor
            else None
        )

        self._communication_test_token_id = cluster_communication_test_token_id()
        self.kv_connector = None
        self._communication_test_kv_caches = None
        self._register_kv_caches_for_transfer()

        # scheduler 与 Mooncake 注册侧必须看到同一份 KV 容量。
        self.config.effective_num_blocks = cluster_communication_test_kv_blocks()
        logger.info(
            "Cluster communication runner enabled: "
            "constant sampled token id=%s, logical num_blocks=%s, "
            "registered_kv_layers=%s",
            self._communication_test_token_id,
            self.config.effective_num_blocks,
            len(self._communication_test_kv_caches or {}),
        )

    def _validate_config(self) -> None:
        """校验轻量通信模式支持的推理配置范围。"""
        text_config = self.hf_config.get("text_config", self.hf_config)
        max_position_embeddings = text_config.get(
            "max_position_embeddings",
            self.config.max_cache_len,
        )
        if not (1024 <= self.config.max_num_batched_tokens <= max_position_embeddings):
            raise ValueError(
                f"max_num_batched_tokens must be in [1024, {max_position_embeddings}]"
            )

        if self.config.cache_type != "paged":
            raise ValueError("cluster communication test requires cache_type='paged'")
        if (
            self.config.tensor_parallel_size != 1
            or self.config.pipeline_parallel_size != 1
        ):
            raise ValueError("cluster communication test only supports tp=1 and pp=1")
        if self.config.draft_model_path is not None:
            raise ValueError("cluster communication test does not support draft models")
        if self.config.use_mla:
            raise ValueError(
                "cluster communication test KV registration does not support MLA"
            )

    def _init_device(self) -> None:
        """初始化真实 infinicore device，测试 KV 需要真实设备地址。"""
        supported_devices = ["cpu", "cuda", "mlu", "musa", "npu"]
        device_str = self.config.device
        if device_str not in supported_devices:
            raise ValueError(
                f"Unsupported device: '{device_str}'. "
                f"Supported devices: {supported_devices}"
            )
        self.device = infinicore.device(device_str, 0)

        dtype_map = {
            "float32": infinicore.float32,
            "float16": infinicore.float16,
            "bfloat16": infinicore.bfloat16,
        }
        if self.config.dtype not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: '{self.config.dtype}'. "
                f"Supported dtypes: {list(dtype_map.keys())}"
            )
        self.dtype = dtype_map[self.config.dtype]

    def _register_kv_caches_for_transfer(self) -> None:
        """分配专用测试 KV 并注册到 Mooncake worker connector。"""
        if self.kv_transfer_config is None or not self.kv_transfer_config.kv_connector:
            return

        connector_name = self.kv_transfer_config.kv_connector
        if connector_name != "MooncakeConnector":
            raise ValueError(
                "cluster communication test only supports MooncakeConnector "
                "for KV registration"
            )

        num_blocks = cluster_communication_test_kv_blocks()
        self._communication_test_kv_caches = create_communication_test_kv_caches(
            self.hf_config,
            self.config,
            num_blocks,
            self.device,
        )
        self.kv_connector = KVConnectorFactory.create_connector(
            connector_name=connector_name,
            role=KVConnectorRole.WORKER,
            kv_transfer_config=self.kv_transfer_config,
        )

        try:
            self.kv_connector.register_kv_caches(self._communication_test_kv_caches)
        except BaseException:
            # 注册失败时立即释放 connector 和 tensor 引用，避免残留后台线程。
            self.kv_connector.shutdown()
            self.kv_connector = None
            self._communication_test_kv_caches = None
            raise

    @property
    def model_type(self) -> str:
        """返回模型类型；轻量模式不创建 InferEngine。"""
        return self.hf_config.get("model_type", "communication-test")

    @property
    def eos_token_id(self):
        """测试 token 不作为 EOS，保证请求不会被提前终止。"""
        return []

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        """执行一次假 forward，并保留 KV connector 的收发状态收集。"""
        sampled_tokens_list = []
        kv_connector_output = None
        forward_timing: ModelForwardTiming | None = None

        if self.kv_connector is None:
            engine_forward_started_at_ns = monotonic_ns()
            sampled_tokens_list = self._model_forward(scheduler_output)
            engine_forward_completed_at_ns = monotonic_ns()
            forward_timing = self._make_forward_timing(
                engine_forward_started_at_ns,
                engine_forward_completed_at_ns,
            )
        else:
            with self.maybe_get_kv_connector_output(
                scheduler_output,
            ) as kv_connector_output:
                if scheduler_output.num_requests > 0:
                    engine_forward_started_at_ns = monotonic_ns()
                    sampled_tokens_list = self._model_forward(scheduler_output)
                    engine_forward_completed_at_ns = monotonic_ns()
                    forward_timing = self._make_forward_timing(
                        engine_forward_started_at_ns,
                        engine_forward_completed_at_ns,
                    )

        # 和普通 ModelRunner 一样，把 batch 级 forward timing 复制到本轮
        # 每个 request 的 step；假 forward 耗时只代表测试调度开销，不代表计算。
        if forward_timing is not None:
            for request in scheduler_output.scheduled_requests:
                timing_steps = getattr(request, "timing_steps", ())
                for timing in timing_steps[-1:]:
                    timing.engine_forward_started_at_ns = (
                        forward_timing.engine_forward_started_at_ns
                    )
                    timing.engine_forward_completed_at_ns = (
                        forward_timing.engine_forward_completed_at_ns
                    )
                    timing.forward_batch_id = forward_timing.forward_batch_id
                if (
                    not timing_steps
                    and getattr(request, "timing_mode", "full") == "summary"
                ):
                    if scheduler_output.is_prefill:
                        request.record_summary_prefill_forward(
                            forward_timing.engine_forward_started_at_ns,
                            forward_timing.engine_forward_completed_at_ns,
                        )
                    else:
                        request.record_summary_decode_forward(
                            forward_timing.engine_forward_started_at_ns,
                            forward_timing.engine_forward_completed_at_ns,
                        )

        req_ids = [
            request.request_id for request in scheduler_output.scheduled_requests
        ]
        return ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled_tokens_list,
            kv_connector_output=kv_connector_output,
            forward_timing=forward_timing,
        )

    def _make_forward_timing(
        self,
        engine_forward_started_at_ns: int,
        engine_forward_completed_at_ns: int,
    ) -> ModelForwardTiming:
        """为一次假 forward 生成 schema v4 需要的 batch timing。"""
        forward_sequence = getattr(self, "_forward_batch_sequence", 0) + 1
        self._forward_batch_sequence = forward_sequence
        return ModelForwardTiming(
            engine_forward_started_at_ns=engine_forward_started_at_ns,
            engine_forward_completed_at_ns=engine_forward_completed_at_ns,
            forward_batch_id=(
                f"communication-forward-{os.getpid()}-{forward_sequence}"
            ),
            source="communication_model_runner",
        )

    def _model_forward(self, scheduler_output) -> list[int]:
        """返回固定 token id，只驱动 scheduler 和 KV 传输流程。"""
        return [self._communication_test_token_id] * scheduler_output.num_requests

    @contextmanager
    def maybe_get_kv_connector_output(
        self,
        scheduler_output: Any,
    ) -> Generator[KVConnectorOutput, None, None]:
        """在假 forward 前后绑定 Mooncake metadata 并收集传输结果。"""
        output = KVConnectorOutput()
        assert scheduler_output.kv_connector_metadata is not None

        self.kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata
        )
        self.kv_connector.start_load_kv()
        try:
            yield output
        finally:
            (
                output.finished_sending,
                output.failed_recving,
                output.finished_recving,
            ) = self.kv_connector.get_finished("finished_req_ids")
            output.invalid_block_ids = (
                self.kv_connector.get_block_ids_with_load_errors()
            )
            output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()

    def close(self) -> None:
        """先关闭 connector，再释放测试 KV tensor 的本地引用。"""
        if self._closed:
            return
        if self.pipeline_control is not None:
            self.pipeline_control.close()
        if self.kv_connector is not None:
            self.kv_connector.shutdown()
        # connector 释放注册信息后再清除本地引用，保证 shutdown 期间地址仍有效。
        self._communication_test_kv_caches = None
        self._closed = True
