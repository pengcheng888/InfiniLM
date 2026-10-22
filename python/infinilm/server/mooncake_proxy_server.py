import argparse
import asyncio
import ipaddress
import itertools
import json
import logging
import os
import urllib
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
INTERNAL_REQUEST_HEADER = "X-InfiniLM-Internal"


def maybe_wrap_ipv6_address(hostname):
    try:
        ipaddress.IPv6Address(hostname)
        return f"[{hostname}]"
    except ValueError:
        return hostname


def make_http_address(hostname, port):
    return f"http://{hostname}:{port}"


async def wait_for_cluster_ready(prefill_clients, decode_clients, ready_event):
    for prefill_client in prefill_clients:
        while True:
            try:
                response = await prefill_client["http_client"].get("/health")
                response.raise_for_status()
            except Exception:
                await asyncio.sleep(1)  # Wait before retrying
                logger.warning(
                    "waiting for handshake with prefill server: %s",
                    prefill_client,
                )
                continue

            response = await prefill_client["http_client"].get(
                prefill_client["bootstrap_addr"] + "/query"
            )
            response.raise_for_status()
            data = response.json()
            break

        logger.warning("successfully handshake with prefill server!")

        for dp_rank, engine_info in data.items():
            prefill_client["dp_engine_id"][int(dp_rank)] = engine_info["engine_id"]
        prefill_client["dp_size"] = len(data)

    for decode_client in decode_clients:
        while True:
            try:
                response = await decode_client["http_client"].get("/health")
                response.raise_for_status()
                logger.warning("successfully handshake with decode server!")
                break
            except Exception:
                await asyncio.sleep(1)  # Wait before retrying
                logger.warning(
                    "waiting for handshake with decode server: %s",
                    decode_client,
                )

    ready_event.set()  # Signal that all prefiller and decoder info has been collected


def prefiller_cycle(prefill_clients):
    while True:
        for prefill_client in prefill_clients:
            dp_size = prefill_client["dp_size"]
            for dp_rank in range(dp_size):
                yield prefill_client, dp_rank


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.ready = asyncio.Event()

    # Create prefill clients
    for prefill_url, bootstrap_port in global_args.prefill:
        parsed_url = urllib.parse.urlparse(prefill_url)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        app.state.prefill_clients.append(
            {
                "http_client": httpx.AsyncClient(
                    timeout=None,
                    base_url=prefill_url,
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "http_base_url": prefill_url,
                "bootstrap_addr": make_http_address(hostname, bootstrap_port or 9600),
                "dp_engine_id": {},
            }
        )

    logger.info("global_args: %s", global_args)
    # Create decode clients
    for decode_url in global_args.decode:
        parsed_url = urllib.parse.urlparse(decode_url)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        app.state.decode_clients.append(
            {
                "http_client": httpx.AsyncClient(
                    timeout=None,
                    base_url=decode_url,
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "http_base_url": decode_url,
            }
        )

    asyncio.create_task(
        wait_for_cluster_ready(
            app.state.prefill_clients, app.state.decode_clients, app.state.ready
        )
    )
    app.state.prefill_iterator = prefiller_cycle(app.state.prefill_clients)
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    yield

    for client_info in app.state.prefill_clients:
        await client_info["http_client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["http_client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser(description="Mooncake Proxy Server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    # Prefill services
    parser.add_argument(
        "--prefill",
        nargs="+",
        action="append",
        dest="prefill_url_list",
        metavar=("URL", "Bootstrap"),
        help=(
            "Prefill service URL and optional bootstrap file"
            "Can be specified multiple times for multiple services "
            "(e.g., --prefill http://localhost:9000/ prefill_bootstrap.json)"
        ),
    )

    # Decoder services
    parser.add_argument(
        "--decode",
        nargs=1,
        action="append",
        dest="decode_url_list",
        metavar="URL",
        help=(
            "Decoder service URL. Can be specified multiple times for multiple services "
            "(e.g., --decode http://localhost:9001/)"
        ),
    )

    args = parser.parse_args()
    args.prefill = _parse_prefill_urls(args.prefill_url_list)
    args.decode = _parse_decode_urls(args.decode_url_list)

    return args


def _parse_prefill_urls(prefill_url_list):
    if not prefill_url_list:
        return []

    prefill_urls = []

    for url in prefill_url_list:
        prefill_url = url[0]

        if len(url) > 1:
            bootstrap_port_str = url[1]
            if bootstrap_port_str.lower() == "none":
                bootstrap_port = None
            else:
                try:
                    bootstrap_port = int(bootstrap_port_str)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid bootstrap port value: {bootstrap_port_str}. Must be an integer or 'none'."
                    ) from e
        else:
            bootstrap_port = None

        prefill_urls.append((prefill_url, bootstrap_port))

    return prefill_urls


def _parse_decode_urls(decode_url_list):
    if not decode_url_list:
        return []

    return [url[0] for url in decode_url_list]


def get_next_client(app: FastAPI, service_type: str):
    if service_type == "prefill":
        return next(app.state.prefill_iterator)
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def forward_request(client_info: dict, api: str, req_data: dict, headers: dict):
    """Forward a request to a backend node and yield response bytes.
    Modeled after disagg_proxy_demo.forward_request, adapted for httpx.
    """
    try:
        async with client_info["http_client"].stream(
            "POST", api, json=req_data, headers=headers
        ) as response:
            if 200 <= response.status_code < 300 or 400 <= response.status_code < 500:
                async for chunk in response.aiter_bytes():
                    yield chunk
            else:
                error_bytes = await response.aread()
                try:
                    error_body = json.loads(error_bytes)
                except Exception:
                    error_body = error_bytes.decode(errors="replace")
                logger.error(
                    "Backend %s%s returned status %d: %s",
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
    except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
        logger.error(
            "Connection error to %s%s: %s",
            client_info.get("http_base_url", ""),
            api,
            e,
        )
        raise HTTPException(
            status_code=502, detail=f"Backend connection error: {e}"
        ) from e
    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP error from %s%s: %s",
            client_info.get("http_base_url", ""),
            api,
            e,
        )
        raise HTTPException(status_code=e.response.status_code, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Unexpected error forwarding to %s%s: %s",
            client_info.get("http_base_url", ""),
            api,
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


async def send_request(
    client_info: dict, dp_rank: int, api: str, req_data: dict, request_id: str
):
    """向 prefill worker 发送内部预填充请求。

    PD frontend 会把原请求复制一份并改成“只做 prefill、只生成一个 token”：
    prefill worker 计算并发布 KV，decode worker 随后通过 transfer_id 拉取。
    该函数只消费 prefill 响应，不把响应内容返回给客户端。
    """
    # PD frontend 和 decode 可能并发读取原 payload；复制后修改不会影响 decode 侧。
    req_data = req_data.copy()
    # do_remote_prefill=False 表示本请求在当前 worker 本地执行 prompt prefill；
    # do_remote_decode=True 表示生成的 KV 会被远端 decode 消费。
    req_data["kv_transfer_params"] = {
        "do_remote_prefill": False,
        "do_remote_decode": True,
        "transfer_id": f"xfer-{request_id}",
    }
    req_data["stream"] = False
    # PD 协议只需要 prefill 完成一个 token，用于触发 KV 发布。
    req_data["max_tokens"] = 1

    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        "X-data-parallel-rank": str(dp_rank),
        INTERNAL_REQUEST_HEADER: "true",
        # Uncomment in high-concurrency or long-context scenarios (minor performance overhead):
        "Connection": "close",
    }
    # 显式持有 forward_request() generator。若不持有引用，外层 prefill
    # task 被取消时 Starlette/httpx 不一定关闭内层 async generator，
    # 对应 HTTP response 连接可能继续占用 worker。
    response_stream = forward_request(client_info, api, req_data, headers)
    try:
        # 消费完整响应，确保 prefill worker 已经完成 KV 发布流程。
        async for _ in response_stream:
            pass
    finally:
        # 正常完成、异常和任务取消都会关闭 response，释放底层连接。
        await response_stream.aclose()


async def stream_response(
    prefill_client_info: dict,
    prefill_dp_rank: int,
    decode_client_info: dict,
    api: str,
    req_data: dict,
    request_id: str,
):
    """向 decode worker 转发请求，并透传增量生成结果。

    decode 请求会携带 prefill bootstrap 地址和 engine ID；Mooncake
    connector 先查询远端 KV 传输信息，再拉取 prefill 已发布的 KV。
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        INTERNAL_REQUEST_HEADER: "true",
        # Uncomment in high-concurrency or long-context scenarios (minor performance overhead):
        "Connection": "close",
    }

    req_data["kv_transfer_params"] = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_bootstrap_addr": prefill_client_info["bootstrap_addr"],
        "remote_engine_id": prefill_client_info["dp_engine_id"][prefill_dp_rank],
        "transfer_id": f"xfer-{request_id}",
    }

    # 显式持有并关闭转发 generator：客户端断开时，外层 P/D stream 被
    # 取消，这里同步关闭 decode HTTP response，避免连接和生成任务泄漏。
    response_stream = forward_request(decode_client_info, api, req_data, headers)
    try:
        # 保持原始 bytes 透传，不在 proxy 层重新拼装或解析 SSE。
        async for chunk in response_stream:
            yield chunk
    finally:
        # 无论正常结束、worker 异常还是客户端取消，都释放 decode 连接。
        await response_stream.aclose()


async def _handle_completions(api: str, request: Request):
    if not request.app.state.ready.is_set():
        raise HTTPException(status_code=503, detail="Service Unavailable")

    try:
        req_data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    request_id = f"cmpl-{uuid.uuid4().hex}"

    prefill_client_info, prefill_dp_rank = get_next_client(request.app, "prefill")
    asyncio.create_task(
        send_request(prefill_client_info, prefill_dp_rank, api, req_data, request_id)
    )

    decode_client_info = get_next_client(request.app, "decode")

    async def generate_stream():
        try:
            async for chunk in stream_response(
                prefill_client_info,
                prefill_dp_rank,
                decode_client_info,
                api,
                req_data,
                request_id=request_id,
            ):
                yield chunk
        except HTTPException as e:
            logger.error(
                "Decode error for %s: HTTP %d - %s", request_id, e.status_code, e.detail
            )
            yield json.dumps(
                {"error": {"message": e.detail, "code": e.status_code}}
            ).encode()
        except Exception as e:
            logger.error("Stream error for %s: %s", request_id, e, exc_info=True)
            yield json.dumps({"error": {"message": str(e), "code": 500}}).encode()

    return StreamingResponse(generate_stream(), media_type="application/json")


@app.post("/v1/completions")
async def handle_v1_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.post("/v1/chat/completions")
async def handle_v1_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    uvicorn.run(app, host=global_args.host, port=global_args.port)
