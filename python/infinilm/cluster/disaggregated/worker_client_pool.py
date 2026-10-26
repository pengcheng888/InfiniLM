"""PD frontend 的 worker HTTP client 池管理。"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable, Sequence
from typing import Any

import httpx

from ..routing import WorkerInflightTracker
from ..worker_registry import WorkerRecord, WorkerType
from .pd_http_transport import http_url

logger = logging.getLogger(__name__)

# key 依次为 model、role 和 instance_id。同一个 worker 如果服务多个 model
# 名，也需要在每个 model 下维护独立状态，避免 ready 状态互相污染。
WorkerClientKey = tuple[str, WorkerType, str]


class WorkerClientPoolManager:
    """按模型管理 prefill/decode worker 的 HTTP client 生命周期。

    registry 只说明 worker 在 discovery 中可见；这个类维护 frontend 本地
    已成功 bootstrap、可实际转发请求的 client。核心设计是：

    1. ``_state_lock`` 只保护 desired/pending/pool 等内存状态；
    2. HTTP client 创建和 bootstrap 查询全部在锁外执行；
    3. 每个 worker 只有一个 pending task，慢 worker 不阻塞其他 worker；
    4. 请求通过 ``acquire_workers()`` 原子登记 inflight，避免后台清理在
       “选中 client”和“登记请求”之间关闭连接。
    """

    def __init__(
        self,
        *,
        connect_timeout: float,
        request_timeout: float,
        inflight: WorkerInflightTracker,
        bootstrap_timeout: float = 3.0,
        retry_delay: float = 3.0,
    ) -> None:
        self._pools: dict[
            str,
            dict[WorkerType, dict[str, dict[str, Any]]],
        ] = {}
        # desired 是最近 discovery 快照希望进入池子的 worker；pending 是仍在
        # bootstrap 的 worker。两者共同驱动状态机，网络 I/O 不持有状态锁。
        self._desired: dict[WorkerClientKey, WorkerRecord] = {}
        self._pending: dict[WorkerClientKey, asyncio.Task[None]] = {}
        self._failed_at: dict[WorkerClientKey, float] = {}
        self._state_lock = asyncio.Lock()
        self._closing = False
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout
        self._bootstrap_timeout = bootstrap_timeout
        self._retry_delay = retry_delay
        self._inflight = inflight

    @staticmethod
    def _http_url(host: str, port: int) -> str:
        """构造 worker HTTP URL，并兼容 IPv6 host。"""
        # IPv6 地址必须加方括号，避免 httpx 把冒号解析成端口分隔符。
        return http_url(host, port)

    @staticmethod
    def _transport_signature(record: WorkerRecord) -> tuple[Any, ...]:
        """提取会影响 HTTP/KV bootstrap 连接的字段。

        heartbeat、负载等普通 metadata 变化只需要刷新 record；host、端口或
        bootstrap 地址变化则必须重建 client，不能继续复用旧连接目标。
        """
        pd_metadata = record.metadata.get("pd")
        bootstrap = None
        if isinstance(pd_metadata, dict):
            bootstrap = (
                str(pd_metadata.get("bootstrap_host") or record.host),
                int(pd_metadata.get("bootstrap_port", 0)),
            )
        return record.host, record.port, bootstrap

    async def _create_client_info(
        self,
        record: WorkerRecord,
    ) -> dict[str, Any]:
        """为一个 worker 创建 HTTP client，并加载 PD bootstrap 元数据。"""
        base_url = self._http_url(record.host, record.port)
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
            "http_client": client,
            "http_base_url": base_url,
            "record": record,
            "instance_id": record.instance_id,
            "routable": True,
        }

        if record.worker_type is not WorkerType.PREFILL:
            return info

        try:
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
            # bootstrap 是后台初始化而不是推理请求，必须使用独立短超时，
            # 避免 REQUEST_TIMEOUT=1000s 时异常 worker 长期占住初始化任务。
            response = await asyncio.wait_for(
                client.get(
                    f"{info['bootstrap_addr']}/query",
                    timeout=httpx.Timeout(
                        connect=self._connect_timeout,
                        read=self._bootstrap_timeout,
                        write=self._connect_timeout,
                        pool=self._connect_timeout,
                    ),
                ),
                timeout=self._bootstrap_timeout,
            )
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
            if not info["dp_engine_id"]:
                raise ValueError(
                    f"prefill worker {record.instance_id} bootstrap returned no engines"
                )
            return info
        except asyncio.CancelledError:
            # shutdown 或 worker 消失会取消 pending task；此时 client 还没有
            # 入池，manager 无法从 _pools 找到它，必须在这里释放。
            await client.aclose()
            raise
        except Exception:
            # bootstrap 失败时新建 client 尚未入池，也必须立即释放。
            await client.aclose()
            raise

    async def _initialize_worker(
        self,
        key: WorkerClientKey,
        record: WorkerRecord,
    ) -> None:
        """在锁外执行 worker 网络初始化，成功后再短锁提交状态。

        流程是：bootstrap -> 短锁确认仍是 desired -> 提交 client。如果
        初始化期间 discovery 移除了 worker，则新 client 不入池并立即关闭。
        """
        info: dict[str, Any] | None = None
        try:
            info = await self._create_client_info(record)
            stale_info: dict[str, Any] | None = info
            async with self._state_lock:
                current_task = asyncio.current_task()
                if self._pending.get(key) is current_task:
                    self._pending.pop(key, None)
                self._failed_at.pop(key, None)

                latest_record = self._desired.get(key)
                transport_same = (
                    latest_record is not None
                    and self._transport_signature(latest_record)
                    == self._transport_signature(record)
                )
                if not self._closing and transport_same and latest_record is not None:
                    # bootstrap 期间 heartbeat 或负载 metadata 可能更新；
                    # 传输目标未变时复用新 client，并把 record 刷成最新值。
                    info["record"] = latest_record
                    _, role, instance_id = key
                    model_pools = self._pools.setdefault(
                        key[0],
                        {WorkerType.PREFILL: {}, WorkerType.DECODE: {}},
                    )
                    model_pools[role][instance_id] = info
                    info["routable"] = True
                    stale_info = None
            if stale_info is not None:
                await stale_info["http_client"].aclose()
        except asyncio.CancelledError:
            async with self._state_lock:
                current_task = asyncio.current_task()
                if self._pending.get(key) is current_task:
                    self._pending.pop(key, None)
            raise
        except Exception as exc:
            async with self._state_lock:
                current_task = asyncio.current_task()
                if self._pending.get(key) is current_task:
                    self._pending.pop(key, None)
                self._failed_at[key] = time.monotonic()
            model_name, role, instance_id = key
            logger.warning(
                "Could not initialize %s worker %s for model %s: %s",
                role.value,
                instance_id,
                model_name,
                exc,
            )

    def _schedule_create_locked(
        self,
        key: WorkerClientKey,
        record: WorkerRecord,
    ) -> None:
        """为缺失 worker 创建去重后的初始化 task；调用方必须持有状态锁。"""
        if self._closing or key in self._pending:
            return
        model_name, role, instance_id = key
        model_pools = self._pools.setdefault(
            model_name,
            {WorkerType.PREFILL: {}, WorkerType.DECODE: {}},
        )
        if instance_id in model_pools[role]:
            return
        failed_at = self._failed_at.get(key)
        if failed_at is not None:
            if time.monotonic() < failed_at + self._retry_delay:
                return
            self._failed_at.pop(key, None)
        self._pending[key] = asyncio.create_task(
            self._initialize_worker(key, record),
            name=(
                "infinilm-pd-client-bootstrap-"
                f"{model_name}-{role.value}-{instance_id}"
            ),
        )

    @staticmethod
    def _desired_key(record: WorkerRecord) -> WorkerClientKey:
        """把 discovery record 映射为 client 池状态机的唯一 key。"""
        return record.served_model_name, record.worker_type, record.instance_id

    async def _reconcile_desired(
        self,
        desired: dict[WorkerClientKey, WorkerRecord],
        *,
        prune_scope: bool,
        model_name: str | None = None,
        role: WorkerType | None = None,
    ) -> None:
        """用 desired 快照同步 pending task、存量 client 和延迟清理。

        ``prune_scope=True`` 只替换指定 model/role 的 desired 子集；
        ``prune_scope=False`` 表示传入参数就是整个集群的 desired 全集。
        """
        stale_clients: list[dict[str, Any]] = []
        cancelled_tasks: list[asyncio.Task[None]] = []

        async with self._state_lock:
            if self._closing:
                return

            if prune_scope:
                assert model_name is not None and role is not None
                merged_desired = {
                    key: record
                    for key, record in self._desired.items()
                    if not (key[0] == model_name and key[1] is role)
                }
            else:
                merged_desired = {}
            merged_desired.update(desired)
            self._desired = merged_desired

            # desired 消失的 pending task 必须取消，避免完成后再提交旧 worker。
            for key, task in list(self._pending.items()):
                if key not in merged_desired:
                    cancelled_tasks.append(task)
                    self._pending.pop(key, None)
                    self._failed_at.pop(key, None)

            for candidate_model, model_pools in self._pools.items():
                for pool_role, pool in model_pools.items():
                    for instance_id, info in list(pool.items()):
                        key = (candidate_model, pool_role, instance_id)
                        desired_record = merged_desired.get(key)
                        if desired_record is None:
                            info["routable"] = False
                            if not self._inflight.has_inflight_requests(instance_id):
                                pool.pop(instance_id, None)
                                stale_clients.append(info)
                            continue

                        if self._transport_signature(info["record"]) == self._transport_signature(
                            desired_record
                        ):
                            info["record"] = desired_record
                            info["routable"] = True
                            continue

                        # 地址变化时旧 client 不再承接新请求；若无 inflight
                        # 可立即关闭并调度重建，有 inflight 则等下一次循环。
                        info["routable"] = False
                        if not self._inflight.has_inflight_requests(instance_id):
                            pool.pop(instance_id, None)
                            stale_clients.append(info)

            for candidate_model in list(self._pools):
                model_pools = self._pools[candidate_model]
                if all(not pool for pool in model_pools.values()):
                    del self._pools[candidate_model]

            for key, record in merged_desired.items():
                self._schedule_create_locked(key, record)

        if cancelled_tasks:
            for task in cancelled_tasks:
                task.cancel()
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)
        if stale_clients:
            await asyncio.gather(
                *(info["http_client"].aclose() for info in stale_clients),
                return_exceptions=True,
            )

    async def reconcile_model(
        self,
        *,
        role: WorkerType,
        model_name: str,
        records: Sequence[WorkerRecord],
    ) -> None:
        """调度指定模型/角色的初始化，不等待网络 bootstrap 完成。"""
        desired = {
            self._desired_key(record): record
            for record in records
            if record.worker_type is role
            and record.served_model_name == model_name
        }
        await self._reconcile_desired(
            desired,
            prune_scope=True,
            model_name=model_name,
            role=role,
        )

    async def reconcile_records(
        self,
        records: Iterable[WorkerRecord],
    ) -> None:
        """以后台 reconciler 的全量 discovery 快照驱动 client 池状态机。"""
        desired = {
            self._desired_key(record): record
            for record in records
            if record.worker_type in (WorkerType.PREFILL, WorkerType.DECODE)
        }
        await self._reconcile_desired(desired, prune_scope=False)

    async def retain_models(self, model_names: Iterable[str]) -> None:
        """清理 discovery 中消失的 model；保留该接口用于测试和兼容调用。"""
        active_models = set(model_names)
        stale_clients: list[dict[str, Any]] = []
        cancelled_tasks: list[asyncio.Task[None]] = []
        async with self._state_lock:
            if self._closing:
                return
            for key in list(self._desired):
                if key[0] in active_models:
                    continue
                task = self._pending.pop(key, None)
                if task is not None:
                    cancelled_tasks.append(task)
                self._failed_at.pop(key, None)
                del self._desired[key]
            for model_name in list(self._pools):
                if model_name in active_models:
                    continue
                model_pools = self._pools[model_name]
                for pool in model_pools.values():
                    for instance_id, info in list(pool.items()):
                        info["routable"] = False
                        if not self._inflight.has_inflight_requests(instance_id):
                            pool.pop(instance_id, None)
                            stale_clients.append(info)
                if all(not pool for pool in model_pools.values()):
                    del self._pools[model_name]

        if cancelled_tasks:
            for task in cancelled_tasks:
                task.cancel()
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)
        if stale_clients:
            await asyncio.gather(
                *(info["http_client"].aclose() for info in stale_clients),
                return_exceptions=True,
            )

    async def wait_for_creates(self) -> None:
        """等待当前已调度的 bootstrap task 结束，用于测试和受控停机。"""
        while True:
            async with self._state_lock:
                tasks = list(self._pending.values())
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    def client_ids(self, model_name: str, role: WorkerType) -> set[str]:
        """返回指定模型和角色仍在 client 池中的 worker ID 集合。"""
        return set(self._pools.get(model_name, {}).get(role, ()))

    def initializing_ids(self, model_name: str, role: WorkerType) -> set[str]:
        """返回指定模型和角色仍在后台初始化的 worker ID 集合。"""
        return {
            instance_id
            for candidate_model, candidate_role, instance_id in self._pending
            if candidate_model == model_name and candidate_role is role
        }

    def routable_records(
        self,
        model_name: str,
        role: WorkerType,
    ) -> list[WorkerRecord]:
        """返回指定模型中可接收新请求且 client 已初始化的 records。"""
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

    async def acquire_workers(
        self,
        model_name: str,
        selections: Sequence[tuple[WorkerType, str]],
    ) -> list[dict[str, Any]]:
        """原子获取多个 worker client 并登记 inflight。

        PD 请求需要 prefill 和 decode 同时参与。这里在一次状态锁内完成
        “检查 routable + 登记 inflight”，要么两个 client 都拿到，要么全部
        回滚；后台 reconcile 因看到 inflight 而不会立即关闭被选中的连接。
        """
        async with self._state_lock:
            acquired: list[dict[str, Any]] = []
            for role, instance_id in selections:
                info = self._pools.get(model_name, {}).get(role, {}).get(instance_id)
                if info is None or not info.get("routable", False):
                    for acquired_info in acquired:
                        self._inflight.finish_request(acquired_info["instance_id"])
                    return []
                self._inflight.begin_request(instance_id)
                acquired.append(info)
            return acquired

    async def aclose(self) -> None:
        """取消 pending bootstrap，并关闭所有池内 HTTP client。"""
        async with self._state_lock:
            self._closing = True
            pending = list(self._pending.values())
            self._pending.clear()
            clients: list[dict[str, Any]] = []
            for model_pools in self._pools.values():
                for pool in model_pools.values():
                    clients.extend(pool.values())
            self._pools.clear()
            self._desired.clear()
            self._failed_at.clear()

        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if clients:
            await asyncio.gather(
                *(info["http_client"].aclose() for info in clients),
                return_exceptions=True,
            )
