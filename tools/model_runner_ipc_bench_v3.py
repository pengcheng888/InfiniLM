#!/usr/bin/env python3
"""比较同步 IPC、同步 MessageQueue 和异步 in-flight MessageQueue。

v3 的重点是模拟 vLLM ``MultiprocExecutor`` 的使用方式，而不只是把消息
放进 shared memory：

* 请求端允许 ``--inflight`` 个未完成 step；
* worker 通过共享内存 ring 读取请求并顺序执行；
* parent 的 receiver 线程独立接收完成通知并填充 Future；
* 完成通知使用独立的 PUSH/PULL socket，生产和消费不会共用一个 socket；
* receiver 可以先短暂 spin，再进入阻塞通知，模拟 spin/notify 混合等待。

结果中的 ``mean_ms`` 是每个 request 从提交到完成的延迟，
``wall_ms`` 是整个 batch 流水线的墙钟时间。异步窗口增大后，单条延迟
可能上升，但在存在 scheduler CPU 工作或 worker 计算时，wall throughput
可能提高。该脚本只测 IPC/调度重叠，不启动模型或 GPU。
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
from dataclasses import asdict, dataclass
from concurrent.futures import Future
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import zmq

from model_runner_ipc_bench import (
    HEADER_BYTES,
    BenchConfig,
    _simulate_compute,
    build_batch,
    build_response,
    run_pipe,
    run_shared_memory,
)
from model_runner_ipc_bench_v2 import run_vllm_like


class AsyncRing:
    """带显式 slot 编号的单生产者/单消费者共享内存 ring。"""

    def __init__(self, shm: SharedMemory, slots: int, slot_size: int):
        self.shm = shm
        self.slots = slots
        self.slot_size = slot_size
        self.state_offset = slots * slot_size

    @classmethod
    def create(cls, slots: int, slot_size: int) -> "AsyncRing":
        shm = SharedMemory(create=True, size=slots * slot_size + slots)
        shm.buf[:] = b"\x00" * len(shm.buf)
        return cls(shm, slots, slot_size)

    @classmethod
    def attach(cls, name: str, slots: int, slot_size: int) -> "AsyncRing":
        return cls(SharedMemory(name=name), slots, slot_size)

    def state(self, slot: int) -> int:
        return self.shm.buf[self.state_offset + slot]

    def set_state(self, slot: int, value: int) -> None:
        self.shm.buf[self.state_offset + slot] = value

    def write(self, slot: int, payload: bytes) -> None:
        if len(payload) > self.slot_size - HEADER_BYTES:
            raise ValueError("message exceeds shared-memory slot")
        offset = slot * self.slot_size
        self.shm.buf[offset : offset + HEADER_BYTES] = len(payload).to_bytes(
            HEADER_BYTES, "little"
        )
        self.shm.buf[offset + HEADER_BYTES : offset + HEADER_BYTES + len(payload)] = payload
        self.set_state(slot, 1)

    def read(self, slot: int) -> bytes:
        offset = slot * self.slot_size
        size = int.from_bytes(self.shm.buf[offset : offset + HEADER_BYTES], "little")
        if size > self.slot_size - HEADER_BYTES:
            raise ValueError("invalid shared-memory message size")
        return bytes(self.shm.buf[offset + HEADER_BYTES : offset + HEADER_BYTES + size])

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()


def _endpoint(prefix: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"infinilm-{prefix}-{os.getpid()}-{time.time_ns()}.ipc")
    return f"ipc://{path}"


def _async_worker(
    control: Connection,
    request_name: str,
    response_name: str,
    request_address: str,
    response_address: str,
    slots: int,
    slot_size: int,
    config: BenchConfig,
) -> None:
    request_ring = AsyncRing.attach(request_name, slots, slot_size)
    response_ring = AsyncRing.attach(response_name, slots, slot_size)
    context = zmq.Context()
    request_socket = context.socket(zmq.PULL)
    response_socket = context.socket(zmq.PUSH)
    request_socket.connect(request_address)
    response_socket.connect(response_address)
    control.send({"ready": True})
    try:
        while True:
            slot = int.from_bytes(request_socket.recv(), "little")
            command = pickle.loads(request_ring.read(slot))
            request_ring.set_state(slot, 0)
            if command["op"] == "close":
                response_ring.write(slot, pickle.dumps({"closed": True}))
                response_socket.send(slot.to_bytes(4, "little"))
                return
            _simulate_compute(config.compute_us)
            response = pickle.dumps(
                {"slot": slot, "seq": command["seq"], **build_response(config, command["step"])},
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            response_ring.write(slot, response)
            response_socket.send(slot.to_bytes(4, "little"))
    finally:
        request_socket.close(linger=0)
        response_socket.close(linger=0)
        context.term()
        request_ring.close()
        response_ring.close()
        control.close()


@dataclass
class AsyncResult:
    config: dict[str, Any]
    inflight: int
    spin_us: int
    scheduler_us: int
    mean_ms: float
    median_ms: float
    p99_ms: float
    wall_ms: float
    throughput: float
    request_bytes: int
    response_bytes: int


def _percentile(values: list[float], ratio: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * ratio))]


def run_async(config: BenchConfig, inflight: int, spin_us: int, scheduler_us: int) -> dict[str, Any]:
    """异步提交 + receiver Future 测试。"""
    context = mp.get_context("spawn")
    max_request = pickle.dumps({"op": "execute", "seq": 0, "step": config.output_len}, protocol=pickle.HIGHEST_PROTOCOL)
    max_response = pickle.dumps(build_response(config, config.output_len), protocol=pickle.HIGHEST_PROTOCOL)
    slot_size = max(config.max_message_bytes, len(max_request), len(max_response)) + HEADER_BYTES
    request_ring = AsyncRing.create(inflight, slot_size)
    response_ring = AsyncRing.create(inflight, slot_size)
    req_address = _endpoint("req")
    resp_address = _endpoint("resp")
    parent, child = context.Pipe()
    process = context.Process(
        target=_async_worker,
        args=(child, request_ring.shm.name, response_ring.shm.name, req_address, resp_address, inflight, slot_size, config),
    )
    req_context = zmq.Context()
    req_socket = req_context.socket(zmq.PUSH)
    resp_socket = req_context.socket(zmq.PULL)
    req_socket.bind(req_address)
    resp_socket.bind(resp_address)
    process.start()
    try:
        parent.recv()
        pending: dict[int, Future[dict[str, Any]]] = {}
        submitted_at: dict[int, int] = {}
        lock = threading.Lock()
        stop = threading.Event()

        def receiver() -> None:
            while not stop.is_set():
                started_spin = time.perf_counter_ns()
                while spin_us and (time.perf_counter_ns() - started_spin) < spin_us * 1000:
                    if resp_socket.poll(0):
                        break
                if not resp_socket.poll(1):
                    continue
                slot = int.from_bytes(resp_socket.recv(), "little")
                response = pickle.loads(response_ring.read(slot))
                response_ring.set_state(slot, 0)
                seq = response.get("seq", -1)
                with lock:
                    future = pending.get(seq)
                if future is not None:
                    future.set_result(response)

        thread = threading.Thread(
            target=receiver, name="ipc-response-receiver", daemon=True
        )
        thread.start()
        latencies: list[float] = []
        request_size = response_size = 0
        next_seq = 0
        next_complete = 0
        wall_start = time.perf_counter_ns()
        while next_complete < config.iterations:
            while next_seq < config.iterations and next_seq - next_complete < inflight:
                _simulate_compute(scheduler_us)
                slot = next_seq % inflight
                command = {"op": "execute", "seq": next_seq, "step": next_seq, "batch": build_batch(config, next_seq)}
                payload = pickle.dumps(command, protocol=pickle.HIGHEST_PROTOCOL)
                if request_ring.state(slot) != 0:
                    raise RuntimeError("request ring slot is still busy")
                request_ring.write(slot, payload)
                future: Future[dict[str, Any]] = Future()
                with lock:
                    pending[next_seq] = future
                submitted_at[next_seq] = time.perf_counter_ns()
                req_socket.send(slot.to_bytes(4, "little"))
                if next_seq == 0:
                    request_size = len(payload)
                next_seq += 1

            with lock:
                future = pending[next_complete]
            response = future.result()
            with lock:
                pending.pop(next_complete)
            latencies.append(
                (time.perf_counter_ns() - submitted_at.pop(next_complete)) / 1_000_000
            )
            response_size = max(response_size, len(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)))
            next_complete += 1

        wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000
        stop.set()
        thread.join(timeout=2)
        close_slot = 0
        while request_ring.state(close_slot) != 0:
            time.sleep(0.001)
        request_ring.write(close_slot, pickle.dumps({"op": "close"}))
        req_socket.send(close_slot.to_bytes(4, "little"))
        return asdict(AsyncResult(asdict(config), inflight, spin_us, scheduler_us,
                                  statistics.mean(latencies), statistics.median(latencies),
                                  _percentile(latencies, 0.99), wall_ms,
                                  config.iterations / (wall_ms / 1000), request_size, response_size))
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent.close()
        child.close()
        req_socket.close(linger=0)
        resp_socket.close(linger=0)
        req_context.term()
        request_ring.close(); response_ring.close()
        request_ring.unlink(); response_ring.unlink()
        for address in (req_address, resp_address):
            try:
                os.unlink(address.removeprefix("ipc://"))
            except FileNotFoundError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--compute-us", type=int, default=1000)
    parser.add_argument("--scheduler-us", type=int, default=500)
    parser.add_argument("--inflight", type=int, default=4)
    parser.add_argument("--spin-us", type=int, default=50)
    parser.add_argument("--max-message-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--ring-slots", type=int, default=4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.input_len,
        args.output_len,
        args.iterations,
        args.inflight,
        args.ring_slots,
    ) <= 0:
        parser.error("batch-size/input-len/output-len/iterations/inflight/ring-slots must be positive")
    if min(args.compute_us, args.scheduler_us, args.spin_us, args.max_message_bytes) < 0:
        parser.error("compute-us/scheduler-us/spin-us/max-message-bytes must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    config = BenchConfig(args.batch_size, args.input_len, args.output_len, args.iterations, args.compute_us, args.max_message_bytes)
    results: list[dict[str, Any]] = []
    results.append(run_pipe(config))
    results.append(run_shared_memory(config))
    results.append(run_vllm_like(config, args.ring_slots))
    results.append({"mode": "v3_async", **run_async(config, args.inflight, args.spin_us, args.scheduler_us)})
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n=== 四组对比 ===")
    for result in results:
        if result["mode"] == "v3_async":
            print(f"{result['mode']}: request_mean={result['mean_ms']:.4f} ms, wall={result['wall_ms']:.4f} ms, throughput={result['throughput']:.2f} step/s")
        else:
            print(f"{result['mode']}: mean={result['mean_ms']:.4f} ms, steps/s={result['steps_per_second']:.2f}")
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
