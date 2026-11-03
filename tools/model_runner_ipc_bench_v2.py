#!/usr/bin/env python3
"""对比三种 ModelRunner step IPC 实现。

三种实现：

1. ``pipe_pickle``：当前生产实现的 Pipe + pickle 基线；
2. ``shared_memory_msgpack``：v1 benchmark 的 shared memory + Semaphore + msgpack；
3. ``vllm_message_queue``：参考 vLLM 本机 MessageQueue 的简化版本：
   shared-memory ring buffer + slot 状态位 + ZMQ notification + pickle。

vLLM 的真实 MessageQueue 还包含多 reader、overflow、大 tensor 的 out-of-band
buffer、memory fence 和 remote reader 路径。本脚本只模拟当前普通服务的单生产者、
单消费者、同步 request/response 场景，用于隔离比较三种 IPC 成本。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import statistics
import tempfile
import threading
import time
from dataclasses import asdict
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import zmq

from model_runner_ipc_bench import (
    HEADER_BYTES,
    BenchConfig,
    _result,
    _simulate_compute,
    build_batch,
    build_response,
    run_pipe,
    run_shared_memory,
)


# 与 vLLM shm_broadcast.py 中 memory_fence 的意图相同：在发布 slot 状态前，
# 先完成当前进程中对共享内存 payload 的写入。这里使用本地锁作为 Python 层
# 的顺序屏障；生产化实现仍应使用经过验证的跨进程原子状态更新。
_memory_fence_lock = threading.Lock()


def memory_fence() -> None:
    with _memory_fence_lock:
        pass


class SharedMemoryRing:
    """单生产者/单消费者 shared-memory ring buffer。

    每个 slot 的 data 区域前 8 字节保存消息长度，ring 尾部保存每个 slot 的
    状态：0 表示 writer 可写，1 表示 reader 可读。当前 benchmark 是同步往返，
    因此不会出现多个未消费 request，但保留多个 slot 以模拟 vLLM 的 ring 结构。
    """

    def __init__(self, shm: SharedMemory, slot_count: int, slot_size: int):
        self.shm = shm
        self.slot_count = slot_count
        self.slot_size = slot_size
        self.metadata_offset = slot_count * slot_size
        self.next_write = 0
        self.next_read = 0

    @classmethod
    def create(cls, slot_count: int, slot_size: int) -> "SharedMemoryRing":
        shm = SharedMemory(create=True, size=slot_count * slot_size + slot_count)
        shm.buf[:] = b"\x00" * len(shm.buf)
        return cls(shm, slot_count, slot_size)

    @classmethod
    def attach(cls, name: str, slot_count: int, slot_size: int) -> "SharedMemoryRing":
        return cls(SharedMemory(name=name), slot_count, slot_size)

    def _state(self, index: int) -> int:
        return self.shm.buf[self.metadata_offset + index]

    def _set_state(self, index: int, state: int) -> None:
        self.shm.buf[self.metadata_offset + index] = state

    def write(self, payload: bytes) -> None:
        if len(payload) > self.slot_size - HEADER_BYTES:
            raise ValueError(
                f"message is {len(payload)} bytes, slot supports "
                f"{self.slot_size - HEADER_BYTES} bytes"
            )
        index = self.next_write
        if self._state(index) != 0:
            raise RuntimeError("shared-memory ring is full")
        offset = index * self.slot_size
        self.shm.buf[offset : offset + HEADER_BYTES] = len(payload).to_bytes(
            HEADER_BYTES, "little"
        )
        self.shm.buf[offset + HEADER_BYTES : offset + HEADER_BYTES + len(payload)] = payload
        memory_fence()
        self._set_state(index, 1)
        self.next_write = (index + 1) % self.slot_count

    def read(self) -> bytes:
        index = self.next_read
        if self._state(index) != 1:
            raise RuntimeError("shared-memory ring has no ready message")
        offset = index * self.slot_size
        size = int.from_bytes(self.shm.buf[offset : offset + HEADER_BYTES], "little")
        if size > self.slot_size - HEADER_BYTES:
            raise ValueError(f"invalid shared-memory message size: {size}")
        payload = bytes(
            self.shm.buf[offset + HEADER_BYTES : offset + HEADER_BYTES + size]
        )
        memory_fence()
        self._set_state(index, 0)
        self.next_read = (index + 1) % self.slot_count
        return payload

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()


def _vllm_like_worker(
    control_conn: Connection,
    request_name: str,
    response_name: str,
    notify_address: str,
    slot_count: int,
    slot_size: int,
    config: BenchConfig,
) -> None:
    request_ring = SharedMemoryRing.attach(request_name, slot_count, slot_size)
    response_ring = SharedMemoryRing.attach(response_name, slot_count, slot_size)
    context = zmq.Context()
    notify_socket = context.socket(zmq.PAIR)
    notify_socket.connect(notify_address)
    control_conn.send({"ready": True})
    try:
        while True:
            # vLLM 的 SpinCondition 会在高频场景快速检查共享内存，空闲时通过
            # ZMQ notification 唤醒。同步 benchmark 直接等待 notification，避免
            # 忙等影响 benchmark 的可重复性。
            notify_socket.recv()
            command = pickle.loads(request_ring.read())
            if command["op"] == "close":
                response = pickle.dumps({"closed": True}, protocol=pickle.HIGHEST_PROTOCOL)
                response_ring.write(response)
                notify_socket.send(b"response")
                return
            _simulate_compute(config.compute_us)
            response = pickle.dumps(
                build_response(config, command["step"]),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            response_ring.write(response)
            notify_socket.send(b"response")
    finally:
        notify_socket.close(linger=0)
        context.term()
        request_ring.close()
        response_ring.close()
        control_conn.close()


def _notify_endpoint() -> tuple[zmq.Context, zmq.Socket, str]:
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    path = os.path.join(tempfile.gettempdir(), f"infinilm-ipc-{os.getpid()}-{time.time_ns()}.ipc")
    address = f"ipc://{path}"
    socket.bind(address)
    return context, socket, address


def run_vllm_like(config: BenchConfig, slot_count: int = 4) -> dict[str, Any]:
    """运行 vLLM MessageQueue 风格的本机 IPC benchmark。"""
    context = mp.get_context("spawn")
    max_request = pickle.dumps(
        {"op": "execute", "step": config.output_len, "batch": build_batch(config, config.output_len)},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    max_response = pickle.dumps(
        build_response(config, config.output_len), protocol=pickle.HIGHEST_PROTOCOL
    )
    slot_size = max(config.max_message_bytes, len(max_request), len(max_response)) + HEADER_BYTES
    request_ring = SharedMemoryRing.create(slot_count, slot_size)
    response_ring = SharedMemoryRing.create(slot_count, slot_size)
    notify_context, notify_socket, notify_address = _notify_endpoint()
    parent_conn, child_conn = context.Pipe()
    process = context.Process(
        target=_vllm_like_worker,
        args=(
            child_conn,
            request_ring.shm.name,
            response_ring.shm.name,
            notify_address,
            slot_count,
            slot_size,
            config,
        ),
    )
    process.start()
    try:
        parent_conn.recv()
        samples = []
        request_size = response_size = 0
        for step in range(config.iterations):
            command = {"op": "execute", "step": step, "batch": build_batch(config, step)}
            started = time.perf_counter_ns()
            encoded_command = pickle.dumps(command, protocol=pickle.HIGHEST_PROTOCOL)
            request_ring.write(encoded_command)
            notify_socket.send(b"request")
            notify_socket.recv()
            response = pickle.loads(response_ring.read())
            elapsed = time.perf_counter_ns() - started
            if response["req_ids"][-1] != f"request-{config.batch_size - 1}":
                raise RuntimeError("vLLM-like response validation failed")
            samples.append(elapsed)
            request_size = len(encoded_command)
            response_size = len(
                pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
            )

        close_message = pickle.dumps({"op": "close"}, protocol=pickle.HIGHEST_PROTOCOL)
        request_ring.write(close_message)
        notify_socket.send(b"request")
        notify_socket.recv()
        return _result(
            "vllm_message_queue",
            config,
            samples,
            {"request": request_size, "response": response_size},
        )
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent_conn.close()
        child_conn.close()
        notify_socket.close(linger=0)
        notify_context.term()
        request_ring.close()
        response_ring.close()
        request_ring.unlink()
        response_ring.unlink()
        if notify_address.startswith("ipc://"):
            try:
                os.unlink(notify_address.removeprefix("ipc://"))
            except FileNotFoundError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--compute-us", type=int, default=0)
    parser.add_argument("--max-message-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--ring-slots", type=int, default=4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.input_len, args.output_len, args.iterations) <= 0:
        parser.error("batch-size/input-len/output-len/iterations must be positive")
    if min(args.compute_us, args.max_message_bytes, args.ring_slots) < 0:
        parser.error("compute-us/max-message-bytes/ring-slots must be non-negative")
    if args.ring_slots == 0:
        parser.error("ring-slots must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = BenchConfig(
        batch_size=args.batch_size,
        input_len=args.input_len,
        output_len=args.output_len,
        iterations=args.iterations,
        compute_us=args.compute_us,
        max_message_bytes=args.max_message_bytes,
    )
    results = [run_pipe(config), run_shared_memory(config), run_vllm_like(config, args.ring_slots)]
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    baseline = results[0]
    print("\n=== 三组对比 ===")
    for result in results:
        improvement = (baseline["mean_ms"] - result["mean_ms"]) / baseline["mean_ms"] * 100
        print(
            f"{result['mode']}: mean={result['mean_ms']:.4f} ms, "
            f"p99={result['p99_ms']:.4f} ms, "
            f"相对 Pipe 改善={improvement:.2f}%"
        )
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
