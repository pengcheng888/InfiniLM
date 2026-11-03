"""Scheduler 与 ModelRunner 之间的可选进程边界。

主进程负责 HTTP 服务、请求生命周期、调度器、BlockManager，以及 scheduler
侧的 KV connector。子进程负责模型权重、GPU KV cache、GPU forward 和 worker
侧的 KV connector。两个进程之间只传输可序列化的普通 Python 数据，不传递
GPU tensor 或完整的 ``InferenceRequest`` 对象。

该模块采用 ``multiprocessing`` 的 ``spawn`` 模式，并通过 ``Pipe`` 传递两类
消息：主进程发送待执行 batch，子进程返回 forward 和 KV connector 的结果。
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import asdict
from multiprocessing.connection import Connection
import time
from typing import TYPE_CHECKING, Any, Deque

from infinilm.llm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from infinilm.config.engine_config import EngineConfig
    from infinilm.llm.model_runner.model_runner import ModelRunnerOutput
    from infinilm.llm.request import InferenceRequest
    from infinilm.llm.scheduler import SchedulerOutput


def model_runner_process_enabled() -> bool:
    """读取是否启用独立 ModelRunner 进程。

    使用严格的布尔值解析，避免把拼写错误的环境变量静默当成 False。普通
    服务和 P/D 服务都通过这个函数决定是否使用进程模式。
    """
    value = os.environ.get("INFINILM_MODEL_RUNNER_PROCESS", "0").strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise RuntimeError(
            "INFINILM_MODEL_RUNNER_PROCESS must be a boolean, got "
            f"{value!r}"
        )
    return value in {"1", "true", "yes", "on"}


def _sampling_to_dict(params: SamplingParams) -> dict[str, Any]:
    """将采样参数转换成 Pipe 可以传输的字典。"""
    return asdict(params)


def _metadata_to_wire(metadata: Any) -> dict[str, Any] | None:
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


def _metadata_from_wire(payload: dict[str, Any] | None):
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


def _request_to_wire(request: "InferenceRequest") -> dict[str, Any]:
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


class _WireRequest:
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


def _batch_to_wire(scheduler_output: "SchedulerOutput") -> dict[str, Any]:
    """将一次 scheduler 调度结果编码成跨进程 batch 消息。

    消息中的 ``is_prefill`` 决定 ModelRunner 执行 prefill 还是 decode；
    ``requests`` 保存轻量请求状态；``kv_connector_metadata`` 保存 P/D
    transfer 所需的发送/接收计划。
    """
    return {
        "is_prefill": scheduler_output.is_prefill,
        "requests": [_request_to_wire(req) for req in scheduler_output.scheduled_requests],
        "kv_connector_metadata": _metadata_to_wire(
            scheduler_output.kv_connector_metadata
        ),
    }


def _response_to_wire(output: "ModelRunnerOutput") -> dict[str, Any]:
    """将 ModelRunner 输出编码成 scheduler 可以消费的普通数据。

    除采样 token 外，还要返回 KV connector 的完成集合，否则 scheduler 无法
    更新 P/D 请求状态。forward timing 也必须跨进程带回，才能继续写入请求
    timing 日志并分析 scheduler、forward 和 KV transfer 的时间关系。
    """
    connector = output.kv_connector_output
    timing = output.forward_timing
    return {
        "req_ids": list(output.req_ids),
        "sampled_token_ids": output.sampled_token_ids,
        "kv_connector_output": None
        if connector is None
        else {
            "finished_sending": list(connector.finished_sending or []),
            "finished_recving": list(connector.finished_recving or []),
            "failed_recving": list(connector.failed_recving or []),
            "invalid_block_ids": list(connector.invalid_block_ids),
            "kv_connector_stats": connector.kv_connector_stats,
        },
        "forward_timing": None
        if timing is None
        else {
            "started_at_ns": timing.engine_forward_started_at_ns,
            "completed_at_ns": timing.engine_forward_completed_at_ns,
            "forward_batch_id": timing.forward_batch_id,
            "source": timing.source,
        },
    }


def _response_from_wire(payload: dict[str, Any]) -> ModelRunnerOutput:
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
        sampled_token_ids=payload["sampled_token_ids"],
        kv_connector_output=connector,
        forward_timing=timing,
    )


def _worker_main(conn: Connection, config: EngineConfig) -> None:
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

    try:
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

        while True:
            command = conn.recv()
            if command["op"] == "close":
                # 正常关闭时让 runner 释放模型、KV cache 和 connector 资源。
                runner.close()
                conn.send({"ok": True})
                return

            if command["op"] != "execute":
                raise RuntimeError(f"unknown ModelRunner command: {command['op']}")

            # execute 消息只包含普通 Python 数据；GPU tensor 在子进程内由
            # ModelRunner 根据这些 token/block 信息重新构造。
            batch = command["batch"]
            requests = [_WireRequest(item) for item in batch["requests"]]
            scheduler_output = SchedulerOutput(
                scheduled_requests=requests,
                is_prefill=batch["is_prefill"],
            )
            scheduler_output.kv_connector_metadata = _metadata_from_wire(
                batch["kv_connector_metadata"]
            )
            output = runner.execute_model(scheduler_output)
            conn.send(
                {
                    "ok": True,
                    "batch_id": command.get("batch_id"),
                    "output": _response_to_wire(output),
                }
            )
    except BaseException as exc:
        try:
            conn.send({"ok": False, "error": repr(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise


class ModelRunnerProcessClient:
    """scheduler 侧的 ModelRunner 进程代理。

    对调用方而言，该类保留与 ModelRunner 相同的核心接口，例如
    ``execute_model``、``processor``、``device``、``dtype`` 和 ``close``。
    真实的模型计算发生在 ``InfiniLMModelRunner`` 子进程中，本对象只负责
    启动进程、发送 wire batch、接收结果和同步 timing。
    """

    supports_non_block = True
    startup_timeout_s = 300.0
    close_timeout_s = 10.0

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

        # 创建子进程对象
        self._process = self._ctx.Process(
            target=_worker_main, # 子进程启动后执行的函数
            args=(child_conn, config),
            name="InfiniLMModelRunner",
            daemon=True, # 守护进程：主进程退出时，该子进程会被自动终止，不会留下孤儿进程。
        )
        self._process.start()

        # 初始化阶段采用同步握手：没有 ready 消息就不能继续创建 scheduler，
        # 否则后续 block 数和 EOS token 配置可能与子进程不一致。
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
            self.close()
            raise RuntimeError(f"ModelRunner process failed: {ready.get('error')}")

        config.effective_num_blocks = int(ready["effective_num_blocks"])
        self._eos_token_ids = ready.get("eos_token_ids") or []

        self._pending_futures: Deque[ModelRunnerFuture] = deque()
        self._next_batch_id = 0
        self._closed = False

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
                if self._parent_conn.poll(poll_timeout):
                    response = self._parent_conn.recv()
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
            output = _response_from_wire(response["output"])
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
        self._parent_conn.send(
            {
                "op": "execute",
                "batch_id": batch_id,
                "batch": _batch_to_wire(scheduler_output),
            }
        )
        self._pending_futures.append(future)
        return future if non_block else future.result()

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
                self._parent_conn.send({"op": "close"})
                close_deadline = time.monotonic() + self.close_timeout_s
                while not self._parent_conn.poll(0.1):
                    if not self._process.is_alive():
                        break
                    if time.monotonic() >= close_deadline:
                        break
                if self._parent_conn.poll(0):
                    self._parent_conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._process.join(timeout=self.close_timeout_s)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._closed = True


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


class ModelRunnerFuture(Future):
    """按提交顺序从 ModelRunnerProcessClient 取结果的 Future。"""

    def __init__(
        self, client: ModelRunnerProcessClient, scheduler_output, batch_id: int
    ):
        super().__init__()
        self.client = client
        self.scheduler_output = scheduler_output
        self.batch_id = batch_id

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
