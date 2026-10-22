"""frontend 到 worker 的 HTTP 客户端与共享连接池管理。"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx
import openai
from openai import AsyncOpenAI

from ..worker_registry import WorkerRecord

logger = logging.getLogger(__name__)

# 这些字段可以作为 OpenAI SDK 的强类型参数传入；其余 InfiniLM 扩展字段
# 会通过 extra_body 合并到 worker 收到的 JSON 顶层。
_STANDARD_FIELDS = {
    "model",
    "messages",
    "stream",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "__infinilm_routing_key",
    "__infinilm_kv_prefix__",
}


def _usage(value: Any) -> dict[str, int]:
    """把 OpenAI SDK 的 usage 对象转换成普通 int 字典。"""
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    if not isinstance(value, dict):
        return {}
    return {
        key: int(value[key])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if value.get(key) is not None
    }


class HttpWorkerClient:
    """访问一个 HTTP worker，并把 OpenAI API 响应适配为内部帧。

    该对象只服务于 router 已选中的单个 worker，不负责 worker 选择、
    重试或连接池生命周期管理。
    """

    def __init__(
        self,
        worker: WorkerRecord,
        *,
        http_client: httpx.AsyncClient,
        request_timeout: float,
    ) -> None:
        self.worker = worker
        base_url = f"http://{worker.host}:{worker.port}/v1"
        # worker 本身不校验鉴权，但 OpenAI SDK 要求提供 api_key；
        # 使用固定占位值可以让集群部署不依赖外部凭据。
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="infinilm",
            http_client=http_client,
            max_retries=0,
            timeout=request_timeout,
        )

    def _create_args(
        self,
        payload: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """构造 OpenAI SDK chat.completions.create 参数。"""
        args: dict[str, Any] = {
            "model": payload.get("model") or self.worker.served_model_name,
            "messages": payload.get("messages") or [],
            "stream": stream,
        }
        for name in ("temperature", "top_p", "max_tokens", "stop"):
            if payload.get(name) is not None:
                args[name] = payload[name]

        # InfiniLM 扩展字段（如 add_generation_prompt、chat_template_kwargs、
        # sampling_params）不能作为 SDK 强类型参数传入，统一放入 extra_body，
        # SDK 最终会把它们合并到 worker 收到的 JSON 顶层。
        args["extra_body"] = {
            key: value for key, value in payload.items() if key not in _STANDARD_FIELDS
        }
        return args

    async def stream_chat(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """调用 OpenAI-compatible API 并返回内部 delta/finish 帧。"""
        if payload.get("stream", False):
            stream = await self.client.chat.completions.create(
                **self._create_args(payload, stream=True)
            )
            finish_reason = "stop"
            usage: dict[str, Any] = {}
            stream_cache = None
            try:
                async for chunk in stream:
                    usage.update(_usage(chunk.usage))
                    for choice in chunk.choices or []:
                        content = choice.delta.content
                        if content:
                            yield {"type": "delta", "content": content}
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                        if getattr(chunk, "cache", None) is not None:
                            stream_cache = chunk.cache
                yield {
                    "type": "finish",
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "cache": stream_cache,
                }
            finally:
                # 关闭 SSE response 也是取消路径。frontend 客户端断开后，
                # InferenceServer 会观察到 HTTP disconnect，并在 finally 中
                # 调用 add_aborted_req() 终止底层 AsyncLLMEngine 请求。
                await stream.close()
            return

        completion = await self.client.chat.completions.create(
            **self._create_args(payload, stream=False)
        )
        choices = completion.choices or []
        choice = choices[0] if choices else None
        content = choice.message.content if choice else ""
        yield {
            "type": "finish",
            "content": content or "",
            "finish_reason": (choice.finish_reason if choice else None) or "stop",
            "usage": _usage(completion.usage),
            "cache": getattr(completion, "cache", None),
        }


class HttpWorkerClientFactory:
    """为 discovery 中的 worker 创建 HTTP 客户端。

    工厂本身持有 frontend 共享的 ``httpx.AsyncClient`` 连接池。不同
    worker 的 ``HttpWorkerClient`` 都复用这同一个连接池，避免连接数
    随 worker 数或请求数无限膨胀。
    """

    def __init__(
        self,
        *,
        connect_timeout: float,
        request_timeout: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """初始化共享 HTTP 连接池和超时配置。

        ``http_client`` 主要用于测试注入 MockTransport。生产路径不传入
        该参数，由工厂创建并拥有真实的 ``httpx.AsyncClient``。
        """
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=request_timeout,
                write=connect_timeout,
                pool=connect_timeout,
            ),
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
        )

    def create(self, worker: WorkerRecord) -> HttpWorkerClient:
        """根据一条 WorkerRecord 创建访问对应 worker 的客户端。"""
        return HttpWorkerClient(
            worker,
            http_client=self._http_client,
            request_timeout=self.request_timeout,
        )

    async def aclose(self) -> None:
        """释放工厂拥有的共享 httpx 连接池。"""
        if self._owns_client:
            await self._http_client.aclose()


def is_retryable_http_error(exc: Exception) -> bool:
    """判断可安全切换 worker 的连接、超时和 5xx 错误。"""
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500
