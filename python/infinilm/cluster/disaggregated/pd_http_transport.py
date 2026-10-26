"""PD frontend 专用的 prefill/decode HTTP 转发层。

该模块只服务 discovery-aware PD frontend，不包含 standalone proxy 的
CLI、FastAPI app、readiness loop 或 Uvicorn 生命周期逻辑。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

from ..http_url import http_url, maybe_wrap_ipv6_address

logger = logging.getLogger(__name__)

INTERNAL_REQUEST_HEADER = "X-InfiniLM-Internal"

__all__ = [
    "INTERNAL_REQUEST_HEADER",
    "forward_request",
    "http_url",
    "maybe_wrap_ipv6_address",
    "send_prefill_request",
    "stream_decode_response",
]


async def forward_request(
    client_info: dict[str, Any],
    api: str,
    req_data: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """把 PD 内部请求转发给 worker 并透传响应 bytes。

    只有 2xx 是正常响应。4xx/5xx 都会读取错误 body 并转换成
    ``HTTPException``，让 PD stream 编排层能够取消另一侧任务；在尚未
    输出内容前，frontend 可以把该状态码作为真实 HTTP status 返回。
    """
    try:
        async with client_info["http_client"].stream(
            "POST",
            api,
            json=req_data,
            headers=headers,
        ) as response:
            if 200 <= response.status_code < 300:
                async for chunk in response.aiter_bytes():
                    yield chunk
                return

            # 非 2xx 不应作为正常响应透传。特别是 prefill 4xx 不能被
            # send_prefill_request() 当成“KV 已发布”的成功结果。
            error_bytes = await response.aread()
            try:
                error_body = json.loads(error_bytes)
            except Exception:
                error_body = error_bytes.decode(errors="replace")

            logger.error(
                "PD backend %s%s returned status %d: %s",
                client_info.get("http_base_url", ""),
                api,
                response.status_code,
                error_body,
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend error {response.status_code}: {error_body}",
            )
    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        logger.error(
            "PD backend %s%s timed out: %s",
            client_info.get("http_base_url", ""),
            api,
            exc,
        )
        raise HTTPException(
            status_code=504,
            detail=f"Backend timeout: {exc}",
        ) from exc
    except (
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.ReadError,
    ) as exc:
        logger.error(
            "PD backend connection error %s%s: %s",
            client_info.get("http_base_url", ""),
            api,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Backend connection error: {exc}",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.error(
            "PD backend HTTP error %s%s: %s",
            client_info.get("http_base_url", ""),
            api,
            exc,
        )
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected PD backend forwarding error %s%s",
            client_info.get("http_base_url", ""),
            api,
        )
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


async def send_prefill_request(
    client_info: dict[str, Any],
    dp_rank: int,
    api: str,
    req_data: dict[str, Any],
    request_id: str,
) -> None:
    """向 prefill worker 发送内部预填充请求。

    请求会被复制并修改为“本地 prefill、远端 decode、只生成一个 token”。
    prefill worker 计算 prompt 并发布 KV；decode worker 之后通过同一个
    ``transfer_id`` 拉取 KV。该函数只消费 prefill 响应，不把内容返回客户端。
    """
    # prefill 与 decode 并行使用同一 payload；浅拷贝即可避免新增/覆盖
    # kv_transfer_params 时影响 decode 侧。
    request_data = req_data.copy()
    request_data["kv_transfer_params"] = {
        "do_remote_prefill": False,
        "do_remote_decode": True,
        "transfer_id": f"xfer-{request_id}",
    }
    request_data["stream"] = False

    # PD 协议只需要一个 token 来触发 KV 发布；同时兼容新旧 OpenAI 参数。
    request_data["max_tokens"] = 1
    if "max_completion_tokens" in request_data:
        request_data["max_completion_tokens"] = 1
    request_data.pop("stream_options", None)

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        "X-data-parallel-rank": str(dp_rank),
        INTERNAL_REQUEST_HEADER: "true",
        # 保持每个内部 prefill 请求独立连接，避免取消时连接状态影响复用。
        "Connection": "close",
    }

    # 显式持有并关闭内层 generator；外层 prefill task 取消时也必须释放
    # httpx response，否则 worker 侧可能看不到连接断开。
    response_stream = forward_request(client_info, api, request_data, headers)
    try:
        # 消费完整响应，确保 prefill worker 完成 KV 发布流程。
        async for _ in response_stream:
            pass
    finally:
        await response_stream.aclose()


async def stream_decode_response(
    prefill_client_info: dict[str, Any],
    prefill_dp_rank: int,
    decode_client_info: dict[str, Any],
    api: str,
    req_data: dict[str, Any],
    request_id: str,
) -> AsyncIterator[bytes]:
    """向 decode worker 转发请求并透传增量生成结果。

    decode 请求携带 prefill bootstrap 地址和 engine ID；Mooncake connector
    在 worker 内查询远端 KV 传输信息并拉取 prefill 已发布的 KV。
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        INTERNAL_REQUEST_HEADER: "true",
        # 保持每个 decode 流独立连接，取消时立即释放 worker 连接。
        "Connection": "close",
    }

    request_data = req_data.copy()
    request_data["kv_transfer_params"] = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_bootstrap_addr": prefill_client_info["bootstrap_addr"],
        "remote_engine_id": prefill_client_info["dp_engine_id"][prefill_dp_rank],
        "transfer_id": f"xfer-{request_id}",
    }

    # 显式持有并关闭内层 generator。客户端断开时，外层 P/D stream 取消，
    # 这里同步关闭 decode HTTP response，避免连接和生成任务泄漏。
    response_stream = forward_request(
        decode_client_info,
        api,
        request_data,
        headers,
    )
    try:
        # 保持 bytes 透传，不在 frontend 内重新解析或拼装 SSE。
        async for chunk in response_stream:
            yield chunk
    finally:
        await response_stream.aclose()
