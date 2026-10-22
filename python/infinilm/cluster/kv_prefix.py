"""Frontend-side prefix hashing for KV-aware routing."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infinilm.llm.prefix_cache import hash_block_tokens
from infinilm.multimodal.multimodal import resolve_multimodal_inputs
from infinilm.processors import AutoInfinilmProcessor

_FINGERPRINT_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def model_cache_namespace(model_path: str, block_size: int) -> str:
    """Build a stable cache namespace from model/tokenizer metadata."""
    root = Path(model_path)
    digest = hashlib.sha256()
    digest.update(b"InfiniLM-kv-cache-namespace-v1\0")
    digest.update(str(block_size).encode("utf-8"))
    for name in _FINGERPRINT_FILES:
        digest.update(name.encode("utf-8"))
        path = root / name
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def prefix_hashes(token_ids: list[int], block_size: int) -> list[str]:
    """Return chained hashes for all full token blocks in a prompt."""
    parent = b""
    result: list[str] = []
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        # 只把完整 block 纳入前缀索引，末尾不足一个 block 的 token 不参与。
        if len(block) != block_size:
            break
        # 每个块的哈希都包含父前缀哈希，形成从开头到当前块的链式前缀。
        parent = hash_block_tokens(block, parent)
        result.append(parent.hex())
    return result


class PrefixHashCalculator:
    """缓存多个模型的 processor，并计算文本 chat prompt 的前缀哈希。"""

    def __init__(self) -> None:
        """初始化 processor 缓存和懒加载锁。"""
        self._processors: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _get_processor(self, model_path: str):
        """获取指定模型 processor；首次访问时加载并写入缓存。"""
        with self._lock:
            processor = self._processors.get(model_path)
            if processor is None:
                # processor 加载可能较慢；锁避免并发请求重复加载同一模型。
                processor = AutoInfinilmProcessor.from_pretrained(model_path)
                self._processors[model_path] = processor
            return processor

    @staticmethod
    def _input_ids(processed: Any) -> list[int]:
        """从 processor 输出中提取一维 token id 列表。"""
        if isinstance(processed, Mapping):
            value = processed.get("input_ids")
        else:
            try:
                value = processed["input_ids"]
            except (KeyError, TypeError):
                return []

        flatten = getattr(value, "flatten", None)
        if callable(flatten):
            # 兼容 torch/tensor-like 输出，先展平再转成 Python list。
            value = flatten()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        if not isinstance(value, list):
            return []
        if value and isinstance(value[0], list):
            # processor 可能返回 batch 维度；prefix 路由只使用当前单条请求。
            value = value[0]
        return [int(token_id) for token_id in value]

    def calculate(
        self,
        payload: dict[str, Any],
        *,
        model_path: str,
        block_size: int,
    ) -> dict[str, Any] | None:
        """计算文本 chat 请求的 KV 前缀信息；不支持时返回 None。"""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None

        try:
            # KV-aware 当前只处理文本 prompt，多模态请求交由 fallback 路由。
            multimodal = resolve_multimodal_inputs(messages)
            if any(multimodal[name] for name in ("images", "videos", "audios")):
                return None

            processor = self._get_processor(model_path)
            # 先渲染 chat template，再交给 processor tokenize，保证与 worker 侧一致。
            prompt = processor.apply_chat_template(
                conversation=messages,
                add_generation_prompt=bool(payload.get("add_generation_prompt", True)),
                tokenize=False,
                **(payload.get("chat_template_kwargs") or {}),
            )
            token_ids = self._input_ids(processor(prompt, return_tensors="pt"))
        except Exception:
            # 前缀哈希只是路由提示；失败时不能影响推理请求继续路由。
            return None

        return {
            "namespace": model_cache_namespace(model_path, block_size),
            "hashes": prefix_hashes(token_ids, block_size),
            "token_count": len(token_ids),
            "block_size": block_size,
        }
