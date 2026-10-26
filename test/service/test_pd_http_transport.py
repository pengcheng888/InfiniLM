import ast
import asyncio
import unittest
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException
from infinilm.cluster.disaggregated.pd_http_transport import (
    forward_request,
    http_url,
    send_prefill_request,
    stream_decode_response,
)


class FakeHTTPResponse:
    def __init__(
        self,
        *,
        chunks: list[bytes] | None = None,
        status_code: int = 200,
        delay: float | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.status_code = status_code
        self.delay = delay
        self.stream_error = stream_error
        self.closed = False
        self.read_body = b'{"error":"backend failed"}'

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.closed = True

    async def aiter_bytes(self):
        try:
            for chunk in self.chunks:
                if self.delay is not None:
                    await asyncio.sleep(self.delay)
                yield chunk
            if self.stream_error is not None:
                raise self.stream_error
        except Exception:
            if self.stream_error is None:
                raise
            raise self.stream_error

    async def aread(self) -> bytes:
        return self.read_body


class FakeHTTPClient:
    def __init__(self, response: FakeHTTPResponse):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def stream(self, method: str, api: str, json: dict, headers: dict):
        self.calls.append(
            {
                "method": method,
                "api": api,
                "json": json,
                "headers": headers,
            }
        )
        return self.response


class PdHttpTransportTest(unittest.TestCase):
    def test_http_url_wraps_ipv6_hostname(self):
        self.assertEqual(http_url("2001:db8::1", 18000), "http://[2001:db8::1]:18000")
        self.assertEqual(http_url("127.0.0.1", 18000), "http://127.0.0.1:18000")

    def test_send_prefill_request_builds_internal_producer_payload(self):
        async def run():
            response = FakeHTTPResponse(chunks=[b'{"ok":true}'])
            client = FakeHTTPClient(response)
            original = {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "max_tokens": 32,
                "max_completion_tokens": 32,
                "stream_options": {"include_usage": True},
            }

            await send_prefill_request(
                {"http_client": client, "http_base_url": "http://prefill"},
                3,
                "/v1/chat/completions",
                original,
                "request-1",
            )

            self.assertEqual(len(client.calls), 1)
            call = client.calls[0]
            payload = call["json"]
            self.assertEqual(call["method"], "POST")
            self.assertEqual(call["api"], "/v1/chat/completions")
            self.assertEqual(
                payload["kv_transfer_params"],
                {
                    "do_remote_prefill": False,
                    "do_remote_decode": True,
                    "transfer_id": "xfer-request-1",
                },
            )
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["max_tokens"], 1)
            self.assertEqual(payload["max_completion_tokens"], 1)
            self.assertNotIn("stream_options", payload)
            self.assertEqual(call["headers"]["X-data-parallel-rank"], "3")
            self.assertEqual(call["headers"]["X-InfiniLM-Internal"], "true")

            # 原 payload 不能被 prefill 请求修改，decode 侧仍使用原设置。
            self.assertTrue(original["stream"])
            self.assertEqual(original["max_tokens"], 32)
            self.assertIn("stream_options", original)
            self.assertTrue(response.closed)

        asyncio.run(run())

    def test_stream_decode_response_builds_remote_prefill_payload(self):
        async def run():
            response = FakeHTTPResponse(
                chunks=[b'data: {"ok":true}\n\n', b"data: [DONE]\n\n"]
            )
            client = FakeHTTPClient(response)
            payload = {"model": "qwen", "messages": [], "stream": True}

            chunks = [
                chunk
                async for chunk in stream_decode_response(
                    {
                        "bootstrap_addr": "http://127.0.0.1:18528",
                        "dp_engine_id": {0: "engine-0", 1: "engine-1"},
                    },
                    1,
                    {"http_client": client, "http_base_url": "http://decode"},
                    "/v1/chat/completions",
                    payload,
                    "request-2",
                )
            ]

            self.assertEqual(
                chunks,
                [b'data: {"ok":true}\n\n', b"data: [DONE]\n\n"],
            )
            call = client.calls[0]
            transfer = call["json"]["kv_transfer_params"]
            self.assertTrue(transfer["do_remote_prefill"])
            self.assertFalse(transfer["do_remote_decode"])
            self.assertEqual(
                transfer["remote_bootstrap_addr"],
                "http://127.0.0.1:18528",
            )
            self.assertEqual(transfer["remote_engine_id"], "engine-1")
            self.assertEqual(transfer["transfer_id"], "xfer-request-2")
            self.assertEqual(call["headers"]["X-InfiniLM-Internal"], "true")
            self.assertNotIn("kv_transfer_params", payload)
            self.assertTrue(response.closed)

        asyncio.run(run())

    def test_prefill_cancellation_closes_http_response(self):
        async def run():
            response = FakeHTTPResponse(delay=10)
            client = FakeHTTPClient(response)
            task = asyncio.create_task(
                send_prefill_request(
                    {"http_client": client, "http_base_url": "http://prefill"},
                    0,
                    "/v1/chat/completions",
                    {"messages": []},
                    "request-3",
                )
            )
            await asyncio.sleep(0.02)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

            self.assertTrue(response.closed)

        asyncio.run(run())

    def test_worker_non_2xx_is_converted_to_http_exception(self):
        """4xx 与 5xx 一样不能作为正常 P/D 响应透传。"""
        async def run():
            for status_code in (400, 404, 429, 500):
                with self.subTest(status_code=status_code):
                    response = FakeHTTPResponse(status_code=status_code)
                    client = FakeHTTPClient(response)
                    stream = forward_request(
                        {"http_client": client, "http_base_url": "http://worker"},
                        "/v1/chat/completions",
                        {"messages": []},
                        {},
                    )

                    with self.assertRaises(HTTPException) as raised:
                        async for _ in stream:
                            pass

                    self.assertEqual(raised.exception.status_code, status_code)
                    self.assertIn("backend failed", raised.exception.detail)
                    self.assertTrue(response.closed)

        asyncio.run(run())

    def test_worker_timeout_is_converted_to_http_504(self):
        """read timeout 不应落入通用 500 分支。"""

        async def run():
            response = FakeHTTPResponse(
                chunks=[b"partial"],
                stream_error=httpx.ReadTimeout("worker read timeout"),
            )
            client = FakeHTTPClient(response)
            stream = forward_request(
                {"http_client": client, "http_base_url": "http://worker"},
                "/v1/chat/completions",
                {"messages": []},
                {},
            )

            with self.assertRaises(HTTPException) as raised:
                async for _ in stream:
                    pass

            self.assertEqual(raised.exception.status_code, 504)
            self.assertIn("worker read timeout", raised.exception.detail)
            self.assertTrue(response.closed)

        asyncio.run(run())


class ClusterDependencyTest(unittest.TestCase):
    def test_cluster_code_does_not_import_mooncake_proxy_server(self):
        """确保 cluster 包不依赖 standalone proxy 的内部实现。"""
        cluster_root = (
            Path(__file__).resolve().parents[2]
            / "python"
            / "infinilm"
            / "cluster"
        )
        forbidden_modules = {
            "infinilm.server.mooncake_proxy_server",
            "...server.mooncake_proxy_server",
        }

        for path in sorted(cluster_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.endswith("mooncake_proxy_server"):
                            self.fail(
                                f"{path} imports forbidden module: {alias.name}"
                            )
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if (
                        module in forbidden_modules
                        or module.endswith(".mooncake_proxy_server")
                    ):
                        self.fail(f"{path} imports forbidden module: {module}")


if __name__ == "__main__":
    unittest.main()
