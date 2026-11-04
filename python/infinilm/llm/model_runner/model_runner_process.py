"""Scheduler 与 ModelRunner 之间的可选进程边界。

主进程负责 HTTP 服务、请求生命周期、调度器、BlockManager，以及 scheduler
侧的 KV connector。子进程负责模型权重、GPU KV cache、GPU forward 和 worker
侧的 KV connector。两个进程之间只传输可序列化的普通 Python 数据，不传递
GPU tensor 或完整的 ``InferenceRequest`` 对象。

传输实现位于 ``model_runner_channel.py``；request/batch/response 协议编解码
位于 ``model_runner_wire.py``。本模块只负责子进程生命周期、worker 循环和
scheduler 侧代理。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import statistics
import time
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any, Deque

from infinilm.llm.model_runner.model_runner_channel import (
    PipeChannel,
    VLLMMessageQueueChannel,
    model_runner_ipc_mode,
)
from infinilm.llm.model_runner.model_runner_wire import (
    WireRequest,
    batch_to_wire,
    connector_control_response_from_wire,
    connector_control_response_to_wire,
    metadata_from_wire,
    metadata_to_wire,
    model_runner_request_cache_enabled,
    requests_from_wire,
    response_from_wire,
    response_to_wire,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from infinilm.config.engine_config import EngineConfig
    from infinilm.llm.model_runner.model_runner import ModelRunnerOutput


def model_runner_process_enabled() -> bool:
    """读取是否启用独立 ModelRunner 进程。

    使用严格的布尔值解析，避免把拼写错误的环境变量静默当成 False。普通
    服务和 P/D 服务都通过这个函数决定是否使用进程模式。
    """
    value = os.environ.get("INFINILM_MODEL_RUNNER_PROCESS", "0").strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise RuntimeError(
            f"INFINILM_MODEL_RUNNER_PROCESS must be a boolean, got {value!r}"
        )
    return value in {"1", "true", "yes", "on"}


def _worker_main(
    conn: Connection,
    config: EngineConfig,
    ipc_mode: str = "pipe",
    request_name: str | None = None,
    response_name: str | None = None,
    request_address: str | None = None,
    response_address: str | None = None,
    slot_bytes: int = 0,
) -> None:
    """ModelRunner 子进程入口。

    子进程启动后只初始化一次模型和 worker-side connector，然后循环处理
    ``execute`` 消息。收到 ``close`` 后优雅释放 runner 并退出。异常会先通过
    Pipe 返回给主进程，再重新抛出，便于保留完整的子进程 traceback。

    ``spawn`` 会继承父进程环境变量。父进程启用进程模式时，如果这里直接
    调用 ``create_model_runner``，工厂会再次创建 ModelRunnerProcessClient，
    形成递归拉起子进程。因此这里只在创建真实 runner 的短暂阶段将环境变量
    设为 0，创建完成后恢复原值。
    """
    from infinilm.llm.model_runner import create_model_runner
    from infinilm.llm.scheduler import SchedulerOutput

    channel: PipeChannel | VLLMMessageQueueChannel | None = None
    request_slot = 0
    command: dict[str, Any] = {}
    ready_sent = False
    request_cache: dict[str, WireRequest] = {}
    try:
        if ipc_mode == "vllm_message_queue":
            channel = VLLMMessageQueueChannel.attach_worker(
                request_name,
                response_name,
                request_address,
                response_address,
                slot_bytes,
            )
        else:
            channel = PipeChannel(conn)

        # worker 本身就是计算进程，必须在其中创建真实 ModelRunner，而不是
        # 再通过工厂创建另一个 ModelRunnerProcessClient。
        previous_process_mode = os.environ.get("INFINILM_MODEL_RUNNER_PROCESS")
        os.environ["INFINILM_MODEL_RUNNER_PROCESS"] = "0"
        try:
            runner = create_model_runner(config, initialize_processor=True)
        finally:
            if previous_process_mode is None:
                os.environ.pop("INFINILM_MODEL_RUNNER_PROCESS", None)
            else:
                os.environ["INFINILM_MODEL_RUNNER_PROCESS"] = previous_process_mode

        # ready 消息用于把子进程计算出的有效 block 数和 EOS token id 同步
        # 回 scheduler 侧。主进程收到该消息后才会继续初始化 scheduler。
        # 通知父进程：子进程创建完毕了
        conn.send(
            {
                "ok": True,
                "effective_num_blocks": config.effective_num_blocks,
                "eos_token_ids": runner.eos_token_id,
            }
        )
        ready_sent = True

        while True:
            request_slot, command = channel.recv_request()
            worker_received_at_ns = time.monotonic_ns()
            if command["op"] == "close":
                # 正常关闭时让 runner 释放模型、KV cache 和 connector 资源。
                runner.close()
                request_cache.clear()
                channel.send_response(request_slot, {"ok": True})
                return

            if command["op"] == "release":
                for request_id in command["request_ids"]:
                    request_cache.pop(request_id, None)
                continue

            if command["op"] == "connector_control":
                # connector-only 控制步没有 scheduled request，也不做模型
                # forward，因此跳过 request cache 和完整 batch/response 协议。
                metadata_started_at_ns = time.monotonic_ns()
                metadata = metadata_from_wire(command["kv_connector_metadata"])
                metadata_completed_at_ns = time.monotonic_ns()
                control_scheduler_output = SchedulerOutput(
                    scheduled_requests=[],
                    is_prefill=False,
                )
                control_scheduler_output.kv_connector_metadata = metadata
                runner_execute_started_at_ns = time.monotonic_ns()
                output = runner.execute_connector_control(control_scheduler_output)
                runner_execute_completed_at_ns = time.monotonic_ns()
                response_wire_started_at_ns = time.monotonic_ns()
                control_payload = connector_control_response_to_wire(output)
                response_wire_completed_at_ns = time.monotonic_ns()
                response_send_started_at_ns = time.monotonic_ns()
                channel.send_response(
                    request_slot,
                    {
                        "ok": True,
                        "batch_id": command.get("batch_id"),
                        "control_output": control_payload,
                        "ipc_timing": {
                            "worker_received_at_ns": worker_received_at_ns,
                            "metadata_from_wire_ns": (
                                metadata_completed_at_ns - metadata_started_at_ns
                            ),
                            "worker_recv_to_execute_start_ns": (
                                runner_execute_started_at_ns - worker_received_at_ns
                            ),
                            "runner_execute_ns": (
                                runner_execute_completed_at_ns
                                - runner_execute_started_at_ns
                            ),
                            "response_wire_ns": (
                                response_wire_completed_at_ns
                                - response_wire_started_at_ns
                            ),
                            "response_send_started_at_ns": response_send_started_at_ns,
                        },
                    },
                )
                continue

            if command["op"] != "execute":
                raise RuntimeError(f"unknown ModelRunner command: {command['op']}")

            # execute 消息只包含普通 Python 数据；GPU tensor 在子进程内由
            # ModelRunner 根据这些 token/block 信息重新构造。
            batch = command["batch"]
            request_rebuild_started_at_ns = time.monotonic_ns()
            requests = requests_from_wire(batch, request_cache)
            request_rebuild_completed_at_ns = time.monotonic_ns()
            scheduler_output = SchedulerOutput(
                scheduled_requests=requests,
                is_prefill=batch["is_prefill"],
            )
            metadata_started_at_ns = time.monotonic_ns()
            scheduler_output.kv_connector_metadata = metadata_from_wire(
                batch["kv_connector_metadata"]
            )
            metadata_completed_at_ns = time.monotonic_ns()
            runner_execute_started_at_ns = time.monotonic_ns()
            output = runner.execute_model(scheduler_output)
            runner_execute_completed_at_ns = time.monotonic_ns()
            response_wire_started_at_ns = time.monotonic_ns()
            response_payload = response_to_wire(output)
            response_wire_completed_at_ns = time.monotonic_ns()
            response_send_started_at_ns = time.monotonic_ns()
            channel.send_response(
                request_slot,
                {
                    "ok": True,
                    "batch_id": command.get("batch_id"),
                    "output": response_payload,
                    "ipc_timing": {
                        "worker_received_at_ns": worker_received_at_ns,
                        "request_rebuild_ns": (
                            request_rebuild_completed_at_ns
                            - request_rebuild_started_at_ns
                        ),
                        "metadata_from_wire_ns": (
                            metadata_completed_at_ns - metadata_started_at_ns
                        ),
                        "worker_recv_to_execute_start_ns": (
                            runner_execute_started_at_ns - worker_received_at_ns
                        ),
                        "runner_execute_ns": (
                            runner_execute_completed_at_ns
                            - runner_execute_started_at_ns
                        ),
                        "response_wire_ns": (
                            response_wire_completed_at_ns - response_wire_started_at_ns
                        ),
                        "response_send_started_at_ns": response_send_started_at_ns,
                    },
                },
            )
    except BaseException as exc:
        try:
            if ready_sent:
                channel.send_response(
                    request_slot,
                    {
                        "ok": False,
                        "batch_id": command.get("batch_id"),
                        "error": repr(exc),
                    },
                )
            else:
                conn.send({"ok": False, "error": repr(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        if channel is not None:
            channel.close()


class ModelRunnerProcessClient:
    """scheduler 侧的 ModelRunner 进程代理。

    对调用方而言，该类保留与 ModelRunner 相同的核心接口，例如
    ``execute_model``、``processor``、``device``、``dtype`` 和 ``close``。
    真实的模型计算发生在 ``InfiniLMModelRunner`` 子进程中，本对象只负责
    启动进程、发送 wire batch、接收结果和同步 timing。
    """

    supports_non_block = True
    supports_connector_control = True
    startup_timeout_s = 300.0
    close_timeout_s = 10.0

    def _record_ipc_timing(self, name: str, duration_ns: int | None) -> None:
        if duration_ns is not None:
            self._ipc_timing_samples.setdefault(name, []).append(duration_ns)

    def __init__(self, config: "EngineConfig", hf_config: dict[str, Any]):
        """创建 Pipe、启动子进程，并等待子进程完成初始化。

        processor 仍在 scheduler 侧保留一份，因为请求预处理和 tokenizer 属于
        服务调度路径。模型权重、GPU KV cache 以及 worker-side connector 则
        只在子进程中创建，避免和 scheduler 共享同一 Python 运行时负载。
        """
        if config.cache_type != "paged":
            raise NotImplementedError(
                "INFINILM_MODEL_RUNNER_PROCESS currently requires paged cache"
            )
        self.config = config
        self.hf_config = hf_config
        from infinilm.processors import AutoInfinilmProcessor

        self.processor = AutoInfinilmProcessor.from_pretrained(config.model_path)
        self.device = config.device
        self.dtype = config.dtype
        # 显式使用 spawn，避免 fork 继承已初始化的 GPU runtime、线程和网络
        # 状态。代价是子进程需要重新加载模型，但进程边界更明确、更可靠。
        self._ctx = mp.get_context("spawn")
        # Pipe() 返回两个连接对象，构成一条双向管道：
        # self._parent_conn：留给父进程用，之后通过 send() / recv() 与子进程通信。
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._pending_futures: Deque[ModelRunnerFuture] = deque()
        self._next_batch_id = 0
        self._closed = False
        self._ipc_mode = model_runner_ipc_mode()
        self._request_cache_enabled = model_runner_request_cache_enabled()
        self._worker_request_state: dict[str, int] = {}
        self._ipc_timing_samples: dict[str, list[int]] = {}
        self._ipc_timing_batches = 0
        self._ipc_timing_execute_batches = 0
        self._ipc_timing_connector_control_batches = 0
        self._ipc_timing_connector_metadata_empty = 0
        self._ipc_timing_connector_metadata_nonempty = 0
        self._ipc_timing_connector_empty_output = 0
        self._ipc_timing_connector_nonempty_output = 0
        self._ipc_timing_connector_finished_sending = 0
        self._ipc_timing_connector_finished_recving = 0
        self._ipc_timing_connector_failed_recving = 0
        self._ipc_timing_requests = 0
        self._ipc_timing_full_requests = 0
        self._ipc_timing_diff_requests = 0

        process_args: tuple[Any, ...] = (child_conn, config)
        if self._ipc_mode == "vllm_message_queue":
            self._channel = VLLMMessageQueueChannel.create_parent()
            process_args = (
                child_conn,
                config,
                self._ipc_mode,
                self._channel.request_ring.shm.name,
                self._channel.response_ring.shm.name,
                self._channel.request_address,
                self._channel.response_address,
                self._channel.slot_bytes,
            )
        else:
            self._channel = PipeChannel(self._parent_conn)

        # 创建子进程对象
        self._process = self._ctx.Process(
            target=_worker_main,  # 子进程启动后执行的函数
            args=process_args,
            name="InfiniLMModelRunner",
            daemon=True,  # 守护进程：主进程退出时，该子进程会被自动终止，不会留下孤儿进程。
        )
        try:
            self._process.start()

            # 初始化阶段采用同步握手：没有 ready 消息就不能继续创建
            # scheduler，否则后续 block 数和 EOS token 配置可能与子进程不一致。
            startup_deadline = time.monotonic() + self.startup_timeout_s
            while not self._parent_conn.poll(0.1):
                if not self._process.is_alive():
                    self._process.join(timeout=5)
                    raise RuntimeError(
                        "ModelRunner process exited during startup: "
                        f"exitcode={self._process.exitcode}"
                    )
                if time.monotonic() >= startup_deadline:
                    self._process.terminate()
                    self._process.join(timeout=5)
                    raise TimeoutError(
                        "timed out waiting for ModelRunner process startup"
                    )
            ready = self._parent_conn.recv()
            if not ready.get("ok"):
                raise RuntimeError(f"ModelRunner process failed: {ready.get('error')}")
        except BaseException:
            # start/handshake 任一阶段失败都可能已经创建子进程、shared memory
            # 或 ZMQ IPC 文件；统一走 close() 清理后再抛出原始异常。
            self.close()
            raise

        config.effective_num_blocks = int(ready["effective_num_blocks"])
        self._eos_token_ids = ready.get("eos_token_ids") or []

    @property
    def eos_token_id(self):
        return self._eos_token_ids

    def _resolve_next(self, timeout: float | None = None) -> None:
        """等待队首响应，并在等待期间检查 worker 是否存活。"""
        if not self._pending_futures:
            raise RuntimeError("no pending ModelRunner future")
        future = self._pending_futures[0]
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                if deadline is None:
                    poll_timeout = 0.1
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise FutureTimeoutError(
                            "timed out waiting for ModelRunner response"
                        )
                    poll_timeout = min(0.1, remaining)
                if self._channel.poll(poll_timeout):
                    receive_started_at_ns = time.monotonic_ns()
                    response = self._channel.recv()
                    response_received_at_ns = time.monotonic_ns()
                    break
                if not self._process.is_alive():
                    raise RuntimeError(
                        "ModelRunner process exited while executing batch "
                        f"{future.batch_id}: exitcode={self._process.exitcode}"
                    )
            if not response.get("ok"):
                raise RuntimeError(
                    f"ModelRunner process failed: {response.get('error')}"
                )
            if response.get("batch_id") != future.batch_id:
                raise RuntimeError(
                    "ModelRunner response batch id mismatch: "
                    f"expected={future.batch_id}, got={response.get('batch_id')}"
                )
            worker_timing = response.get("ipc_timing", {})
            parent_timing = future.parent_ipc_timing
            is_connector_control = future.command == "connector_control"
            if is_connector_control:
                output = connector_control_response_from_wire(
                    response.get("control_output")
                )
            else:
                output = response_from_wire(response["output"])
            self._ipc_timing_batches += 1
            if is_connector_control:
                self._ipc_timing_connector_control_batches += 1
                if future.connector_metadata_empty:
                    self._ipc_timing_connector_metadata_empty += 1
                else:
                    self._ipc_timing_connector_metadata_nonempty += 1
                connector = output.kv_connector_output
                if connector is None:
                    self._ipc_timing_connector_empty_output += 1
                    finished_sending = 0
                    finished_recving = 0
                    failed_recving = 0
                else:
                    if connector.is_empty():
                        self._ipc_timing_connector_empty_output += 1
                    else:
                        self._ipc_timing_connector_nonempty_output += 1
                    finished_sending = len(connector.finished_sending or [])
                    finished_recving = len(connector.finished_recving or [])
                    failed_recving = len(connector.failed_recving or [])
                self._ipc_timing_connector_finished_sending += finished_sending
                self._ipc_timing_connector_finished_recving += finished_recving
                self._ipc_timing_connector_failed_recving += failed_recving
            else:
                self._ipc_timing_execute_batches += 1
                self._ipc_timing_requests += len(
                    future.scheduler_output.scheduled_requests
                )
                cache_stats = future.request_cache_stats
                self._ipc_timing_full_requests += cache_stats.get("full", 0)
                self._ipc_timing_diff_requests += cache_stats.get("diff", 0)
            timing_prefix = "connector_control_" if is_connector_control else ""
            self._record_ipc_timing(
                f"{timing_prefix}parent_wire", parent_timing.get("wire_duration_ns")
            )
            self._record_ipc_timing(
                f"{timing_prefix}parent_send", parent_timing.get("send_duration_ns")
            )
            self._record_ipc_timing(
                f"{timing_prefix}send_start_to_worker_recv",
                (
                    worker_timing.get("worker_received_at_ns")
                    - parent_timing.get("send_started_at_ns")
                    if worker_timing.get("worker_received_at_ns") is not None
                    and parent_timing.get("send_started_at_ns") is not None
                    else None
                ),
            )
            worker_timing_names = (
                "request_rebuild",
                "metadata_from_wire",
                "worker_recv_to_execute_start",
                "runner_execute",
                "response_wire",
            )
            if is_connector_control:
                # 控制步没有 request rebuild。
                worker_timing_names = worker_timing_names[1:]
            for name in worker_timing_names:
                self._record_ipc_timing(
                    f"{timing_prefix}{name}", worker_timing.get(f"{name}_ns")
                )
            self._record_ipc_timing(
                f"{timing_prefix}parent_response_decode",
                response_received_at_ns - receive_started_at_ns,
            )
            if worker_timing.get("response_send_started_at_ns") is not None:
                self._record_ipc_timing(
                    f"{timing_prefix}response_send_start_to_parent_recv",
                    response_received_at_ns
                    - worker_timing["response_send_started_at_ns"],
                )
            self._record_ipc_timing(
                f"{timing_prefix}submit_to_response_decoded",
                response_received_at_ns - future.submit_started_at_ns,
            )
            _update_forward_timing(future.scheduler_output, output)
            self._pending_futures.popleft()
            future.set_result(output)
        except FutureTimeoutError:
            # Keep the Future at the head of the queue so a later result() call
            # can continue waiting for the same response.
            raise
        except BaseException as exc:
            self._fail_pending(exc)
            raise

    def _fail_pending(self, exc: BaseException) -> None:
        """将 worker 故障传播给当前及所有尚未完成的 Future。"""
        for pending in self._pending_futures:
            if not pending.done():
                pending.set_exception(exc)
        self._pending_futures.clear()

    def execute_connector_control(self, scheduler_output):
        """同步执行一个 connector-only 控制步。

        该路径只传输 Mooncake metadata 和 completion output，不构造 request
        cache diff，也不走完整 ``execute_model`` batch/response 协议。
        """
        if self._closed:
            raise RuntimeError("ModelRunner process client is closed")

        batch_id = self._next_batch_id
        self._next_batch_id += 1
        wire_started_at_ns = time.monotonic_ns()
        metadata = metadata_to_wire(scheduler_output.kv_connector_metadata)
        wire_completed_at_ns = time.monotonic_ns()
        send_started_at_ns = time.monotonic_ns()
        future = ModelRunnerFuture(
            self,
            scheduler_output,
            batch_id,
            command="connector_control",
        )
        future.connector_metadata_empty = metadata is None
        self._channel.send(
            {
                "op": "connector_control",
                "batch_id": batch_id,
                "kv_connector_metadata": metadata,
                "ipc_timing": {
                    "wire_started_at_ns": wire_started_at_ns,
                    "wire_completed_at_ns": wire_completed_at_ns,
                    "send_started_at_ns": send_started_at_ns,
                },
            }
        )
        future.submit_started_at_ns = wire_started_at_ns
        future.parent_ipc_timing = {
            "wire_duration_ns": wire_completed_at_ns - wire_started_at_ns,
            "send_started_at_ns": send_started_at_ns,
            "send_duration_ns": time.monotonic_ns() - send_started_at_ns,
        }
        future.request_cache_stats = {"full": 0, "diff": 0}
        self._pending_futures.append(future)
        return future.result()

    def execute_model(
        self, scheduler_output, non_block: bool = False
    ) -> "ModelRunnerOutput | ModelRunnerFuture":
        """执行一个 batch；``non_block`` 时返回尚未完成的 Future。

        请求发送、Future 创建和 FIFO pending 队列管理集中在这个入口，
        与 vLLM ``collective_rpc(..., non_block=True)`` 的接口形态一致。
        """
        if self._closed:
            raise RuntimeError("ModelRunner process client is closed")

        batch_id = self._next_batch_id
        self._next_batch_id += 1
        future = ModelRunnerFuture(self, scheduler_output, batch_id)
        wire_started_at_ns = time.monotonic_ns()
        batch = batch_to_wire(
            scheduler_output,
            self._worker_request_state if self._request_cache_enabled else None,
        )
        wire_completed_at_ns = time.monotonic_ns()
        send_started_at_ns = time.monotonic_ns()
        self._channel.send(
            {
                "op": "execute",
                "batch_id": batch_id,
                "batch": batch,
                "ipc_timing": {
                    "wire_started_at_ns": wire_started_at_ns,
                    "wire_completed_at_ns": wire_completed_at_ns,
                    "send_started_at_ns": send_started_at_ns,
                },
            }
        )
        future.submit_started_at_ns = wire_started_at_ns
        future.parent_ipc_timing = {
            "wire_duration_ns": wire_completed_at_ns - wire_started_at_ns,
            "send_started_at_ns": send_started_at_ns,
            "send_duration_ns": time.monotonic_ns() - send_started_at_ns,
        }
        future.request_cache_stats = batch["request_cache"]
        self._pending_futures.append(future)
        return future if non_block else future.result()

    def release_requests(self, request_ids: list[str]) -> None:
        """释放已经结束的请求在父/子进程中的增量同步状态。"""
        if self._closed or not request_ids:
            return
        for request_id in request_ids:
            self._worker_request_state.pop(request_id, None)
        self._channel.send({"op": "release", "request_ids": list(request_ids)})

    def close(self) -> None:
        """优雅关闭子进程，超时后再强制终止。

        正常路径先发送 ``close`` 并等待 worker 回复；如果 worker 已经异常
        退出或 Pipe 已断开，则忽略通信错误。最多等待 10 秒，仍未退出时
        使用 ``terminate``，避免服务关闭时永久阻塞。
        """
        if not getattr(self, "_process", None) or getattr(self, "_closed", False):
            return
        # Worker 必须先消费所有已提交 batch，否则 close response 可能与某个
        # execute response 错位。正常服务关闭时队列通常为空，这里只覆盖正在
        # drain 的情况。
        try:
            while self._pending_futures:
                self._pending_futures[0].result()
        except BaseException:
            # worker 已退出或响应失败时，直接进入 terminate/join 清理路径。
            pass
        if self._process.is_alive():
            try:
                self._channel.send({"op": "close"})
                close_deadline = time.monotonic() + self.close_timeout_s
                while not self._channel.poll(0.1):
                    if not self._process.is_alive():
                        break
                    if time.monotonic() >= close_deadline:
                        break
                if self._channel.poll(0):
                    self._channel.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._process.join(timeout=self.close_timeout_s)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        if self._ipc_timing_batches:
            logger.info(
                "ModelRunner IPC timing summary: %s",
                self.get_ipc_timing_stats(),
            )
        self._channel.close()
        self._closed = True

    def get_ipc_timing_stats(self) -> dict[str, Any]:
        """返回进程通信路径的聚合耗时，单位毫秒。"""
        result: dict[str, Any] = {
            "batches": self._ipc_timing_batches,
            "execute_batches": self._ipc_timing_execute_batches,
            "connector_control_batches": self._ipc_timing_connector_control_batches,
            "connector_metadata_empty": self._ipc_timing_connector_metadata_empty,
            "connector_metadata_nonempty": (
                self._ipc_timing_connector_metadata_nonempty
            ),
            "connector_empty_output": self._ipc_timing_connector_empty_output,
            "connector_nonempty_output": self._ipc_timing_connector_nonempty_output,
            "connector_finished_sending_requests": (
                self._ipc_timing_connector_finished_sending
            ),
            "connector_finished_recving_requests": (
                self._ipc_timing_connector_finished_recving
            ),
            "connector_failed_recving_requests": (
                self._ipc_timing_connector_failed_recving
            ),
            "requests": self._ipc_timing_requests,
            "full_requests": self._ipc_timing_full_requests,
            "diff_requests": self._ipc_timing_diff_requests,
        }
        for name, values in self._ipc_timing_samples.items():
            if not values:
                continue
            ordered = sorted(values)
            result[name] = {
                "count": len(values),
                "mean_ms": statistics.mean(values) / 1_000_000,
                "median_ms": statistics.median(values) / 1_000_000,
                "p95_ms": ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))]
                / 1_000_000,
                "max_ms": ordered[-1] / 1_000_000,
            }
        return result


def _update_forward_timing(scheduler_output, output: "ModelRunnerOutput") -> None:
    """把异步返回的 forward timing 写回对应 batch 的原始请求。"""
    if output.forward_timing is None:
        return
    for request in scheduler_output.scheduled_requests:
        if request.timing_steps:
            timing = request.timing_steps[-1]
            timing.engine_forward_started_at_ns = (
                output.forward_timing.engine_forward_started_at_ns
            )
            timing.engine_forward_completed_at_ns = (
                output.forward_timing.engine_forward_completed_at_ns
            )
            timing.forward_batch_id = output.forward_timing.forward_batch_id
            continue

        if getattr(request, "timing_mode", "full") != "summary":
            continue

        if scheduler_output.is_prefill:
            request.record_summary_prefill_forward(
                output.forward_timing.engine_forward_started_at_ns,
                output.forward_timing.engine_forward_completed_at_ns,
            )
        else:
            request.record_summary_decode_forward(
                output.forward_timing.engine_forward_started_at_ns,
                output.forward_timing.engine_forward_completed_at_ns,
            )


class ModelRunnerFuture(Future):
    """按提交顺序从 ModelRunnerProcessClient 取结果的 Future。"""

    def __init__(
        self,
        client: ModelRunnerProcessClient,
        scheduler_output,
        batch_id: int,
        command: str = "execute",
    ):
        super().__init__()
        self.client = client
        self.scheduler_output = scheduler_output
        self.batch_id = batch_id
        self.command = command
        self.connector_metadata_empty = False
        self.submit_started_at_ns = 0
        self.parent_ipc_timing: dict[str, int] = {}
        self.request_cache_stats: dict[str, int] = {}

    def result(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.done():
            if not self.client._pending_futures:
                raise RuntimeError("ModelRunner future queue is inconsistent")
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            self.client._resolve_next(remaining)
        return super().result()
