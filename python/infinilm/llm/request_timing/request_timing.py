"""请求 timing 的后台缓冲式 JSONL 写入器。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from infinilm.llm.request import InferenceRequest, RequestStepTiming

DEFAULT_REQUEST_TIMING_DIR = Path("/tmp/infinilm-request-timing")
# 观测文件只用于诊断，默认 24 小时后清理，防止长期运行占满 /tmp。
DEFAULT_REQUEST_TIMING_RETENTION_SECONDS = 24 * 60 * 60
# 后台写队列采用有界容量；超过后丢弃 timing，不影响推理请求。
DEFAULT_REQUEST_TIMING_MAX_PENDING_JOBS = 1024
# schema 字段用于离线工具识别字段语义；修改日志结构时必须递增。
REQUEST_TIMING_SCHEMA_VERSION = 4
# 过期文件清理与请求写入解耦，固定每小时执行一次。
REQUEST_TIMING_CLEANUP_INTERVAL_SECONDS = 60 * 60

# 这些是观测数据的固定保护阈值：内存中最多积累 256 条未写出的 step，
# 或距离上次写出超过 10 秒，就强制提交到后台写线程。阈值不做配置接口，
# 避免运行期被调大后影响推理进程内存稳定性。
REQUEST_TIMING_FLUSH_STEP_COUNT = 256
REQUEST_TIMING_FLUSH_INTERVAL_SECONDS = 10.0
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def request_timing_path(
    request_id: str,
    *,
    root: Path | str = DEFAULT_REQUEST_TIMING_DIR,
) -> Path:
    """返回一个请求专属的 JSONL 文件路径。

    request id 来自外部输入，不能直接拼路径。先替换非法字符，再在原值
    被修改或过长时追加 hash，防止不同 request id 清洗后发生文件名冲突。
    """
    safe_prefix = _UNSAFE_FILENAME_CHARS.sub("_", request_id)[:80]
    if safe_prefix != request_id or not safe_prefix:
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:12]
        safe_prefix = f"{safe_prefix}-{digest}"
    return Path(root) / str(os.getpid()) / f"{safe_prefix}.jsonl"


class RequestTimingFileRecorder:
    """缓存 request step，并由单个后台线程分块写出 JSONL。

    正常短请求只修改文件两次：首条 prefill step 立即写出，finish 时把
    后续 decode step 和汇总一起写出。长请求按 256 条或 10 秒强制分段
    写出，避免异常退出时丢失全部观测数据，也限制内存中的未落盘记录数。
    """

    def __init__(
        self,
        *,
        root: Path | str = DEFAULT_REQUEST_TIMING_DIR,
        retention_seconds: float = DEFAULT_REQUEST_TIMING_RETENTION_SECONDS,
        max_pending_jobs: int = DEFAULT_REQUEST_TIMING_MAX_PENDING_JOBS,
    ) -> None:
        """初始化 recorder；线程池和后台线程都等到首次写入时创建。"""
        if max_pending_jobs <= 0:
            raise ValueError("max_pending_jobs must be positive")
        self._root = Path(root)
        # 单 worker 保证同一请求的 JSONL append 顺序与提交顺序一致。
        self._executor: ThreadPoolExecutor | None = None
        # _lock 保护 buffers/closed 等状态；flush condition 复用同一把锁。
        self._lock = threading.Lock()
        self._flush_condition = threading.Condition(self._lock)
        self._closed = False
        self._retention_seconds = retention_seconds
        self._pending_slots = threading.BoundedSemaphore(max_pending_jobs)
        self._dropped_records = 0
        self._pending_jobs = 0
        # flush 线程持有 condition lock 时也会提交任务；统计锁必须独立，
        # 否则在 _submit_locked() 中重复获取非重入 state lock 会死锁。
        self._stats_lock = threading.Lock()

        # request_id -> 尚未落盘的后续 step；首条 step 不进入该缓冲。
        self._buffers: dict[str, list[RequestStepTiming]] = {}
        # request_id -> 上次提交写任务时的 monotonic 时间，用于 10 秒强刷。
        self._last_flush_at_ns: dict[str, int] = {}
        self._flush_thread: threading.Thread | None = None
        self._cleanup_stop = threading.Event()
        self._cleanup_thread: threading.Thread | None = None

    @staticmethod
    def _duration_ns(started_at: int | None, completed_at: int | None) -> int | None:
        """计算两个 monotonic 时间点差值；任一时间缺失时返回 None。"""
        if started_at is None or completed_at is None:
            return None
        return completed_at - started_at

    @classmethod
    def _step_payload(cls, timing: RequestStepTiming) -> dict[str, Any]:
        """把一轮 prefill/decode timing 转成可序列化的 JSON 对象。"""
        return {
            "schema_version": REQUEST_TIMING_SCHEMA_VERSION,
            "event": "step",
            **timing.to_dict(),
            "durations_ns": {
                "queue": cls._duration_ns(
                    timing.server_received_at_ns,
                    timing.scheduler_polled_at_ns,
                ),
                "scheduler": cls._duration_ns(
                    timing.scheduler_polled_at_ns,
                    timing.scheduler_batch_ready_at_ns,
                ),
                "scheduler_batch_ready_to_engine_forward_start": cls._duration_ns(
                    timing.scheduler_batch_ready_at_ns,
                    timing.engine_forward_started_at_ns,
                ),
                "engine_forward": cls._duration_ns(
                    timing.engine_forward_started_at_ns,
                    timing.engine_forward_completed_at_ns,
                ),
            },
        }

    @staticmethod
    def _finish_payload(request: InferenceRequest) -> dict[str, Any]:
        """请求进入终态时输出一条汇总记录。"""
        return {
            "schema_version": REQUEST_TIMING_SCHEMA_VERSION,
            "event": "finish",
            "role": request.timing_role,
            "finish_reason": (
                request.finish_reason.value
                if request.finish_reason is not None
                else None
            ),
            "prompt_tokens": request.get_prompt_length(),
            "generated_tokens": request.get_num_generated_tokens(),
            "finish_local_monotonic_ns": time.monotonic_ns(),
            **request.get_timing_summary(),
        }

    def _cleanup_expired_files(self) -> None:
        """后台删除过期 JSONL，避免 /tmp 中 timing 文件无限增长。"""
        if self._retention_seconds <= 0 or not self._root.exists():
            return
        cutoff = time.time() - self._retention_seconds
        try:
            for path in self._root.rglob("*.jsonl"):
                try:
                    if not path.is_symlink() and path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    continue
        except OSError:
            return

    def _write_records(
        self,
        path: Path,
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        """后台线程用一次文件打开写出一组 JSON line。"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
                for payload in payloads
            ]
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write("".join(lines))
        except (OSError, TypeError, ValueError):
            # /tmp 不可写、磁盘满或序列化失败都只丢弃观测数据。
            return

    def _write_steps(self, timings: Sequence[RequestStepTiming]) -> None:
        """后台写出一个请求的一段 step 记录。"""
        if not timings:
            return
        path = request_timing_path(timings[0].request_id, root=self._root)
        self._write_records(path, [self._step_payload(timing) for timing in timings])

    def _write_finish(
        self,
        timings: Sequence[RequestStepTiming],
        request: InferenceRequest,
    ) -> None:
        """后台一次写出剩余 step 和 finish 汇总。"""
        path = request_timing_path(request.request_id, root=self._root)
        payloads = [self._step_payload(timing) for timing in timings]
        payloads.append(self._finish_payload(request))
        self._write_records(path, payloads)

    def _run_job(self, callback, *args) -> None:
        """执行后台任务并释放有界队列额度。"""
        try:
            callback(*args)
        finally:
            self._pending_slots.release()
            with self._stats_lock:
                self._pending_jobs = max(0, self._pending_jobs - 1)

    def _drop_records(self, count: int) -> None:
        """记录因后台队列满而丢弃的观测记录数。"""
        with self._stats_lock:
            self._dropped_records += count

    def _ensure_executor_locked(self) -> ThreadPoolExecutor:
        """惰性创建单 worker 线程池，并启动周期清理线程。"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="infinilm-request-timing",
            )
        # 首次真正产生 timing 时才开始清理，空载服务不额外创建线程。
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_stop.clear()
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="infinilm-request-timing-cleanup",
                daemon=True,
            )
            self._cleanup_thread.start()
        return self._executor

    def _cleanup_loop(self) -> None:
        """启动时和每小时清理一次过期 timing 文件。"""
        while True:
            self._cleanup_expired_files()
            if self._cleanup_stop.wait(
                REQUEST_TIMING_CLEANUP_INTERVAL_SECONDS
            ):
                return

    def _submit_locked(
        self,
        record_count: int,
        callback,
        *args,
        force: bool = False,
    ) -> None:
        """在持有 state lock 时非阻塞提交写任务。

        正常路径使用有界信号量限制待写任务；close() 的最后 flush 使用
        ``force=True`` 直接排队，保证退出前尽量写出已缓冲的观测数据。
        """
        executor = self._ensure_executor_locked()
        # 推理热路径不能阻塞等待磁盘；队列满时选择丢弃观测数据。
        if not force and not self._pending_slots.acquire(blocking=False):
            self._drop_records(record_count)
            return

        try:
            if force:
                executor.submit(callback, *args)
            else:
                with self._stats_lock:
                    self._pending_jobs += 1
                executor.submit(self._run_job, callback, *args)
        except RuntimeError:
            if not force:
                self._pending_slots.release()
                with self._stats_lock:
                    self._pending_jobs = max(0, self._pending_jobs - 1)
            self._drop_records(record_count)

    def _start_flush_thread_locked(self) -> None:
        """首次出现内存缓冲时启动后台 10 秒强刷线程。"""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._flush_thread = threading.Thread(
            target=self._flush_expired_buffers,
            name="infinilm-request-timing-flush",
            daemon=True,
        )
        self._flush_thread.start()

    def _flush_request_locked(self, request_id: str, now_ns: int) -> None:
        """把一个请求的当前内存缓冲提交给后台写线程。"""
        timings = tuple(self._buffers.get(request_id, ()))
        if not timings:
            return
        self._buffers[request_id] = []
        self._last_flush_at_ns[request_id] = now_ns
        self._submit_locked(len(timings), self._write_steps, timings)

    def _flush_expired_buffers(self) -> None:
        """后台定期检查超过 10 秒未写出的请求缓冲。"""
        while True:
            # 没有缓冲时休眠，record_steps() 提交新缓冲后再唤醒。
            with self._flush_condition:
                while not self._closed and not any(self._buffers.values()):
                    self._flush_condition.wait()
                if self._closed:
                    return

                now_ns = time.monotonic_ns()
                deadline_ns = min(
                    self._last_flush_at_ns[request_id]
                    + int(REQUEST_TIMING_FLUSH_INTERVAL_SECONDS * 1_000_000_000)
                    for request_id, buffer in self._buffers.items()
                    if buffer
                )
                timeout_seconds = max(0.001, (deadline_ns - now_ns) / 1_000_000_000)
                self._flush_condition.wait(timeout=timeout_seconds)
                if self._closed:
                    return

                now_ns = time.monotonic_ns()
                interval_ns = int(
                    REQUEST_TIMING_FLUSH_INTERVAL_SECONDS * 1_000_000_000
                )
                for request_id, buffer in list(self._buffers.items()):
                    if buffer and now_ns - self._last_flush_at_ns[request_id] >= interval_ns:
                        self._flush_request_locked(request_id, now_ns)

    def record_steps(self, timings: Sequence[RequestStepTiming]) -> None:
        """记录一轮 scheduler batch；同批次内按 request 分别处理。"""
        timings = tuple(timings)
        if not timings:
            return

        with self._lock:
            if self._closed:
                return

            for timing in timings:
                request_id = timing.request_id
                if request_id not in self._last_flush_at_ns:
                    # 首条 step 立即写出，保证文件能尽早创建并保留首包路径。
                    self._last_flush_at_ns[request_id] = time.monotonic_ns()
                    self._buffers[request_id] = []
                    self._submit_locked(1, self._write_steps, (timing,))
                    continue

                buffer = self._buffers.setdefault(request_id, [])
                buffer.append(timing)
                now_ns = time.monotonic_ns()
                interval_ns = int(
                    REQUEST_TIMING_FLUSH_INTERVAL_SECONDS * 1_000_000_000
                )
                if (
                    len(buffer) >= REQUEST_TIMING_FLUSH_STEP_COUNT
                    or now_ns - self._last_flush_at_ns[request_id] >= interval_ns
                ):
                    self._flush_request_locked(request_id, now_ns)

            if any(self._buffers.values()):
                self._start_flush_thread_locked()
                self._flush_condition.notify_all()

    def record_step(self, timing: RequestStepTiming) -> None:
        """兼容单条记录接口；内部复用 batch 提交路径。"""
        self.record_steps([timing])

    def record_finish(self, request: InferenceRequest) -> None:
        """请求终态时一次写出剩余 step 和 finish 汇总。"""
        with self._lock:
            if self._closed:
                return
            request_id = request.request_id
            # finish 负责清空该请求的全部残留状态，防止后续事件复用旧时间。
            timings = tuple(self._buffers.pop(request_id, ()))
            self._last_flush_at_ns.pop(request_id, None)
            self._submit_locked(
                len(timings) + 1,
                self._write_finish,
                timings,
                request,
            )

    def record_event(self, payload: dict[str, Any]) -> None:
        """立即记录一个 KV transfer 等非 step 事件。

        事件使用同一个单 worker 写线程，因此与该 request 的 step/finish
        写任务保持提交顺序。事件时间只属于当前节点，不能与另一台 P/D
        节点的时间戳相减。
        """
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return
        with self._lock:
            if self._closed:
                return
            event_payload = dict(payload)
            event_payload.setdefault("schema_version", REQUEST_TIMING_SCHEMA_VERSION)
            self._submit_locked(1, self._write_event, event_payload)

    def stats(self) -> dict[str, int]:
        """返回 recorder 内部观测状态，用于诊断后台写入是否积压。"""
        with self._lock:
            buffered_requests = sum(
                bool(buffer) for buffer in self._buffers.values()
            )
        with self._stats_lock:
            return {
                "dropped_records": self._dropped_records,
                "pending_jobs": self._pending_jobs,
                "buffered_requests": buffered_requests,
            }

    def _write_event(self, payload: dict[str, Any]) -> None:
        """后台写出一个独立 timing event。"""
        request_id = payload.get("request_id")
        if not isinstance(request_id, str):
            return
        path = request_timing_path(request_id, root=self._root)
        self._write_records(path, [payload])

    def close(self) -> None:
        """写出未完成请求缓冲，停止后台线程并关闭线程池。"""
        with self._lock:
            if self._closed:
                return

            # close 只处理已经缓存的 step；随后不再接受新的观测记录。
            for request_id, buffer in list(self._buffers.items()):
                if buffer:
                    self._submit_locked(
                        len(buffer),
                        self._write_steps,
                        tuple(buffer),
                        force=True,
                    )
                self._buffers.pop(request_id, None)
                self._last_flush_at_ns.pop(request_id, None)

            self._closed = True
            executor = self._executor
            self._executor = None
            self._flush_condition.notify_all()
            self._cleanup_stop.set()
            flush_thread = self._flush_thread
            self._flush_thread = None
            cleanup_thread = self._cleanup_thread
            self._cleanup_thread = None

        if flush_thread is not None and flush_thread is not threading.current_thread():
            flush_thread.join(timeout=REQUEST_TIMING_FLUSH_INTERVAL_SECONDS + 1)
        if cleanup_thread is not None and cleanup_thread is not threading.current_thread():
            cleanup_thread.join(timeout=1)
        if executor is not None:
            executor.shutdown(wait=True)
