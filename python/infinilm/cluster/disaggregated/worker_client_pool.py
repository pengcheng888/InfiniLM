"""PD frontend 的 worker HTTP client 池管理。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable, Sequence

import httpx

from ...server.mooncake_proxy_server import maybe_wrap_ipv6_address
from ..routing import WorkerInflightTracker
from ..worker_registry import WorkerRecord, WorkerType

logger = logging.getLogger(__name__)


class WorkerClientPoolManager:
    """按模型管理 prefill/decode worker 的 HTTP client 生命周期。

    registry 只描述“worker 是否注册可见”；这个类维护的是 frontend 本地
    已成功初始化、可实际转发请求的 client 集合。池子按 model -> role ->
    instance_id 分层，因此同步某个模型不会影响其他模型的可用 client。
    """

    def __init__(
        self,
        *,
        connect_timeout: float,
        request_timeout: float,
        inflight: WorkerInflightTracker,
    ) -> None:
        self._pools: dict[
            str,
            dict[WorkerType, dict[str, dict[str, Any]]],
        ] = {}
        # client 初始化可能包含 bootstrap 网络请求；锁避免并发请求重复创建。
        self._lock = asyncio.Lock()
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout
        # inflight 属于请求生命周期，由 frontend 维护；这里只读取以延迟清理。
        self._inflight = inflight

    @staticmethod
    def _http_url(host: str, port: int) -> str:
        """构造 worker HTTP URL，并兼容 IPv6 host。"""
        # IPv6 地址必须加方括号，避免 httpx 把冒号解析成端口分隔符。
        return f"http://{maybe_wrap_ipv6_address(host)}:{port}"

    async def _create_client_info(
        self,
        record: WorkerRecord,
    ) -> dict[str, Any]:
        """为一个 worker 创建 HTTP client，并加载 PD bootstrap 元数据。"""
        base_url = self._http_url(record.host, record.port)
        # 超时配置来自环境变量；连接池不设上限，由 worker 数和请求并发决定。
        client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=self._connect_timeout,
                read=self._request_timeout,
                write=self._connect_timeout,
                pool=self._connect_timeout,
            ),
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=None,
            ),
        )
        info: dict[str, Any] = {
            # 显式说明这是 HTTP 传输对象和 HTTP base URL，避免与逻辑 worker 混淆。
            "http_client": client,
            "http_base_url": base_url,
            "record": record,
            "instance_id": record.instance_id,
            # 新建成功即对 desired 集合内的 worker 开启新请求路由。
            "routable": True,
        }

        if record.worker_type is not WorkerType.PREFILL:
            return info

        try:
            # prefill 必须提供 Mooncake bootstrap 信息，decode 才能查询 KV 地址。
            pd_metadata = record.metadata.get("pd")
            if not isinstance(pd_metadata, dict):
                raise ValueError(
                    f"prefill worker {record.instance_id} has no PD metadata"
                )
            bootstrap_host = str(pd_metadata.get("bootstrap_host") or record.host)
            bootstrap_port = int(pd_metadata.get("bootstrap_port", 0))
            info["bootstrap_addr"] = self._http_url(
                bootstrap_host,
                bootstrap_port,
            )
            # 查询 bootstrap server，拿到各 pipeline rank 对应的 engine ID。
            response = await client.get(f"{info['bootstrap_addr']}/query")
            response.raise_for_status()
            engines = response.json()
            if not isinstance(engines, dict) or not engines:
                raise ValueError(
                    f"prefill worker {record.instance_id} bootstrap returned no engines"
                )
            info["dp_engine_id"] = {
                int(rank): engine["engine_id"]
                for rank, engine in engines.items()
                if isinstance(engine, dict) and "engine_id" in engine
            }
            info["dp_size"] = len(info["dp_engine_id"])
            # 没有 engine ID 时 decode 无法定位远端 KV，视为该 worker 初始化失败。
            if not info["dp_engine_id"]:
                raise ValueError(
                    f"prefill worker {record.instance_id} bootstrap returned no engines"
                )
            return info
        except Exception:
            await client.aclose()
            raise

    async def reconcile_model(
        self,
        *,
        role: WorkerType,
        model_name: str,
        records: Sequence[WorkerRecord],
    ) -> None:
        """按期望 worker 集合同步一个模型角色的 HTTP client 池。"""
        # 同一时间只允许一个 reconcile 过程修改 client 池。
        async with self._lock:
            model_pools = self._pools.setdefault(
                model_name,
                {
                    WorkerType.PREFILL: {},
                    WorkerType.DECODE: {},
                },
            )
            pool = model_pools[role]
            # desired 表示当前期望进入该角色 client 池的 worker 集合。
            desired = {record.instance_id: record for record in records}

            # 先刷新仍然存在 worker 的 metadata；退出或 draining 的连接稍后清理。
            for instance_id, info in list(pool.items()):
                if instance_id in desired:
                    info["record"] = desired[instance_id]
                    # desired 内的 worker 可以接收新请求，并同步最新 metadata。
                    info["routable"] = True
                    continue
                # 不在 desired 内但仍有存量请求时，仅延迟清理，不再路由新请求。
                info["routable"] = False
                if not self._inflight.has_inflight_requests(instance_id):
                    # 有存量请求时不关闭 client，避免请求过程中连接被释放。
                    await info["http_client"].aclose()
                    del pool[instance_id]

            for instance_id, record in desired.items():
                if instance_id in pool:
                    continue
                try:
                    # bootstrap 查询失败只跳过该 worker，不影响其他副本继续服务。
                    pool[instance_id] = await self._create_client_info(record)
                except Exception as exc:
                    logger.warning(
                        "Could not initialize %s worker %s for model %s: %s",
                        role.value,
                        instance_id,
                        model_name,
                        exc,
                    )

    async def retain_models(self, model_names: Iterable[str]) -> None:
        """只保留指定模型，延迟清理已从 discovery 消失的模型 client。"""
        active_models = set(model_names)
        async with self._lock:
            for model_name in list(self._pools):
                if model_name in active_models:
                    continue
                model_pools = self._pools[model_name]
                for pool in model_pools.values():
                    for instance_id, info in list(pool.items()):
                        info["routable"] = False
                        if self._inflight.has_inflight_requests(instance_id):
                            continue
                        await info["http_client"].aclose()
                        del pool[instance_id]
                if all(not pool for pool in model_pools.values()):
                    del self._pools[model_name]

    def client_ids(self, model_name: str, role: WorkerType) -> set[str]:
        """返回指定模型和角色仍在 client 池中的 worker ID 集合。"""
        return set(self._pools.get(model_name, {}).get(role, ()))

    def routable_records(
        self,
        model_name: str,
        role: WorkerType,
    ) -> list[WorkerRecord]:
        """返回指定模型中当前可接收新请求且 client 已初始化的 records。"""
        return sorted(
            (
                info["record"]
                for info in self._pools.get(model_name, {}).get(role, {}).values()
                if info["routable"]
            ),
            key=lambda record: record.instance_id,
        )

    def get(
        self,
        model_name: str,
        role: WorkerType,
        instance_id: str,
    ) -> dict[str, Any] | None:
        """读取指定模型中已初始化成功的 worker client 信息。"""
        return self._pools.get(model_name, {}).get(role, {}).get(instance_id)

    async def aclose(self) -> None:
        """关闭所有池内 HTTP client，并清空本地池。"""
        async with self._lock:
            for model_pools in self._pools.values():
                for pool in model_pools.values():
                    for info in pool.values():
                        await info["http_client"].aclose()
                    pool.clear()
            self._pools.clear()
