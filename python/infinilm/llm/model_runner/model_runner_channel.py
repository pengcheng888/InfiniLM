"""ModelRunner 子进程通信传输层。

该模块只负责本机 transport：Pipe、shared-memory ring、ZMQ notification 和
短窗口 spin/notify。它不依赖 InferenceRequest、SchedulerOutput 或
ModelRunnerOutput 等推理领域对象。
"""

from __future__ import annotations

import os
import pickle
import tempfile
import time
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import zmq


def model_runner_ipc_mode() -> str:
    """读取 ModelRunner 子进程的数据面通信模式。"""
    mode = os.environ.get("INFINILM_MODEL_RUNNER_IPC", "pipe").strip().lower()
    aliases = {
        "pipe": "pipe",
        "shared_memory": "vllm_message_queue",
        "vllm_message_queue": "vllm_message_queue",
    }
    if mode not in aliases:
        raise RuntimeError(
            "INFINILM_MODEL_RUNNER_IPC must be pipe, shared_memory, "
            f"or vllm_message_queue, got {mode!r}"
        )
    return aliases[mode]


def model_runner_slot_bytes() -> int:
    raw_value = os.environ.get("INFINILM_MODEL_RUNNER_SHM_SLOT_BYTES", "8388608")
    try:
        slot_bytes = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "INFINILM_MODEL_RUNNER_SHM_SLOT_BYTES must be an integer"
        ) from exc
    if slot_bytes < 64 * 1024:
        raise RuntimeError(
            "INFINILM_MODEL_RUNNER_SHM_SLOT_BYTES must be at least 65536"
        )
    return slot_bytes


def model_runner_ipc_busy_loop_us() -> int:
    """读取 IPC reader 的短窗口 busy-poll 时间，单位微秒。"""
    raw_value = os.environ.get("INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US", "0")
    try:
        busy_loop_us = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US must be an integer"
        ) from exc
    if not 0 <= busy_loop_us <= 1_000_000:
        raise RuntimeError(
            "INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US must be between 0 and 1000000"
        )
    return busy_loop_us


def _sched_yield() -> None:
    try:
        os.sched_yield()
    except AttributeError:  # pragma: no cover - non-POSIX fallback
        time.sleep(0)


def _spin_then_poll(poll, timeout: float | None, busy_loop_us: int):
    """vLLM SpinCondition 风格的短窗口忙等，然后回落到阻塞 poll。"""
    if busy_loop_us <= 0:
        return poll(timeout)

    started_at = time.monotonic()
    budget = busy_loop_us / 1_000_000
    if timeout is not None:
        budget = min(budget, timeout)
    deadline = started_at + budget
    while time.monotonic() < deadline:
        if poll(0):
            return True
        _sched_yield()

    if timeout is None:
        return poll(None)
    remaining = timeout - (time.monotonic() - started_at)
    if remaining <= 0:
        return poll(0)
    return poll(remaining)


class SharedMemoryRing:
    """单生产者/单消费者的固定 slot shared-memory ring。"""

    header_bytes = 8

    def __init__(self, shm: SharedMemory, slots: int, slot_bytes: int):
        self.shm = shm
        self.slots = slots
        self.slot_bytes = slot_bytes
        self.state_offset = slots * slot_bytes

    @classmethod
    def create(cls, slots: int, slot_bytes: int) -> "SharedMemoryRing":
        shm = SharedMemory(create=True, size=slots * slot_bytes + slots)
        shm.buf[:] = b"\x00" * len(shm.buf)
        return cls(shm, slots, slot_bytes)

    @classmethod
    def attach(cls, name: str, slots: int, slot_bytes: int) -> "SharedMemoryRing":
        return cls(SharedMemory(name=name), slots, slot_bytes)

    def state(self, slot: int) -> int:
        return self.shm.buf[self.state_offset + slot]

    def set_state(self, slot: int, value: int) -> None:
        self.shm.buf[self.state_offset + slot] = value

    def write(self, slot: int, payload: bytes) -> None:
        if len(payload) > self.slot_bytes - self.header_bytes:
            raise ValueError(
                f"ModelRunner IPC message is {len(payload)} bytes, "
                f"shared-memory slot supports {self.slot_bytes - self.header_bytes}"
            )
        offset = slot * self.slot_bytes
        self.shm.buf[offset : offset + self.header_bytes] = len(payload).to_bytes(
            self.header_bytes, "little"
        )
        self.shm.buf[
            offset + self.header_bytes : offset + self.header_bytes + len(payload)
        ] = payload
        self.set_state(slot, 1)

    def read(self, slot: int) -> bytes:
        offset = slot * self.slot_bytes
        size = int.from_bytes(
            self.shm.buf[offset : offset + self.header_bytes], "little"
        )
        if size > self.slot_bytes - self.header_bytes:
            raise ValueError(f"invalid shared-memory message size: {size}")
        return bytes(
            self.shm.buf[offset + self.header_bytes : offset + self.header_bytes + size]
        )

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()


def _ipc_endpoint(prefix: str) -> str:
    path = os.path.join(
        tempfile.gettempdir(),
        f"infinilm-model-runner-{prefix}-{os.getpid()}-{time.time_ns()}.ipc",
    )
    return f"ipc://{path}"


class PipeChannel:
    """保留当前 Pipe 数据面，作为性能和行为的基线。"""

    def __init__(self, conn: Connection, busy_loop_us: int | None = None):
        self.conn = conn
        self.busy_loop_us = (
            model_runner_ipc_busy_loop_us() if busy_loop_us is None else busy_loop_us
        )

    def send(self, message: dict[str, Any]) -> None:
        self.conn.send(message)

    def poll(self, timeout: float) -> bool:
        return _spin_then_poll(self.conn.poll, timeout, self.busy_loop_us)

    def recv(self) -> dict[str, Any]:
        self.poll(None)
        return self.conn.recv()

    def recv_request(self) -> tuple[int, dict[str, Any]]:
        """Pipe transport has no slot id; use zero as the placeholder."""
        return 0, self.recv()

    def send_response(self, slot: int, message: dict[str, Any]) -> None:
        self.conn.send(message)

    def close(self) -> None:
        pass


class VLLMMessageQueueChannel:
    """vLLM MessageQueue 风格的本机数据面。

    vLLM 的 MessageQueue 使用 shared-memory ring 携带消息，并用轻量通知唤醒
    reader；本实现同样只把 pickle 后的 batch 放入 shared memory，ZMQ IPC
    通知仅携带 4 字节 slot 编号。它保留 local reader、ring 和 notification
    的核心结构，未实现 remote reader、out-of-band tensor arena 和 overflow。
    """

    slots = 2

    def __init__(
        self,
        *,
        role: str,
        request_ring: SharedMemoryRing | None = None,
        response_ring: SharedMemoryRing | None = None,
        request_address: str | None = None,
        response_address: str | None = None,
    ):
        self.role = role
        self.owns_shm = role == "parent"
        self.request_ring = request_ring
        self.response_ring = response_ring
        self.request_address = request_address
        self.response_address = response_address
        self.context = zmq.Context()
        self.next_slot = 0
        self.busy_loop_us = model_runner_ipc_busy_loop_us()
        self.request_socket = None
        self.response_socket = None

        try:
            if role == "parent":
                assert request_address and response_address
                self.request_socket = self.context.socket(zmq.PUSH)
                self.request_socket.bind(request_address)
                self.response_socket = self.context.socket(zmq.PULL)
                self.response_socket.bind(response_address)
            else:
                assert request_ring and response_ring
                assert request_address and response_address
                self.request_socket = self.context.socket(zmq.PULL)
                self.request_socket.connect(request_address)
                self.response_socket = self.context.socket(zmq.PUSH)
                self.response_socket.connect(response_address)
        except BaseException:
            if self.request_socket is not None:
                self.request_socket.close(linger=0)
            if self.response_socket is not None:
                self.response_socket.close(linger=0)
            self.context.term()
            raise

    @classmethod
    def create_parent(cls) -> "VLLMMessageQueueChannel":
        request_ring = None
        response_ring = None
        request_address = None
        response_address = None
        try:
            slot_bytes = model_runner_slot_bytes()
            request_ring = SharedMemoryRing.create(cls.slots, slot_bytes)
            response_ring = SharedMemoryRing.create(cls.slots, slot_bytes)
            request_address = _ipc_endpoint("request")
            response_address = _ipc_endpoint("response")
            return cls(
                role="parent",
                request_ring=request_ring,
                response_ring=response_ring,
                request_address=request_address,
                response_address=response_address,
            )
        except BaseException:
            if request_ring is not None:
                request_ring.close()
                request_ring.unlink()
            if response_ring is not None:
                response_ring.close()
                response_ring.unlink()
            for address in (request_address, response_address):
                if address and address.startswith("ipc://"):
                    try:
                        os.unlink(address.removeprefix("ipc://"))
                    except FileNotFoundError:
                        pass
            raise

    @classmethod
    def attach_worker(
        cls,
        request_name: str,
        response_name: str,
        request_address: str,
        response_address: str,
        slot_bytes: int,
    ) -> "VLLMMessageQueueChannel":
        return cls(
            role="worker",
            request_ring=SharedMemoryRing.attach(request_name, cls.slots, slot_bytes),
            response_ring=SharedMemoryRing.attach(response_name, cls.slots, slot_bytes),
            request_address=request_address,
            response_address=response_address,
        )

    @property
    def slot_bytes(self) -> int:
        return self.request_ring.slot_bytes

    def send(self, message: dict[str, Any]) -> None:
        slot = self.next_slot % self.slots
        deadline = time.monotonic() + 10.0
        while self.request_ring.state(slot) != 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("ModelRunner request ring is full")
            time.sleep(0.0001)

        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        self.request_ring.write(slot, payload)
        self.request_socket.send(slot.to_bytes(4, "little"))
        self.next_slot += 1

    def poll(self, timeout: float) -> bool:
        return bool(
            _spin_then_poll(
                lambda poll_timeout: self.response_socket.poll(
                    None if poll_timeout is None else poll_timeout * 1000
                ),
                timeout,
                self.busy_loop_us,
            )
        )

    def recv(self) -> dict[str, Any]:
        slot = int.from_bytes(self.response_socket.recv(), "little")
        payload = self.response_ring.read(slot)
        self.response_ring.set_state(slot, 0)
        return pickle.loads(payload)

    def recv_request(self) -> tuple[int, dict[str, Any]]:
        _spin_then_poll(
            lambda timeout: self.request_socket.poll(
                None if timeout is None else timeout * 1000
            ),
            None,
            self.busy_loop_us,
        )
        slot = int.from_bytes(self.request_socket.recv(), "little")
        payload = self.request_ring.read(slot)
        self.request_ring.set_state(slot, 0)
        return slot, pickle.loads(payload)

    def send_response(self, slot: int, message: dict[str, Any]) -> None:
        deadline = time.monotonic() + 10.0
        while self.response_ring.state(slot) != 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("ModelRunner response ring is full")
            time.sleep(0.0001)
        self.response_ring.write(
            slot, pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        )
        self.response_socket.send(slot.to_bytes(4, "little"))

    def close(self) -> None:
        self.request_socket.close(linger=0)
        self.response_socket.close(linger=0)
        self.context.term()
        if self.request_ring is not None:
            self.request_ring.close()
        if self.response_ring is not None:
            self.response_ring.close()
        if self.role != "parent" or not self.owns_shm:
            return
        self.request_ring.unlink()
        self.response_ring.unlink()
        for address in (self.request_address, self.response_address):
            if address and address.startswith("ipc://"):
                try:
                    os.unlink(address.removeprefix("ipc://"))
                except FileNotFoundError:
                    pass
