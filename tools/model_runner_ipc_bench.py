#!/usr/bin/env python3
"""独立测试 ModelRunner 每个 step 的进程间通信开销。

该脚本不启动模型、不使用 GPU，也不修改生产代码。它只模拟当前
``ModelRunnerProcessClient.execute_model`` 的数据边界：

* baseline：``multiprocessing.Pipe`` + pickle + 同步 request/response；
* optimized：shared-memory 单槽消息队列 + Semaphore + msgpack。

optimized 的设计参考 vLLM 的 ``MessageQueue``：消息主体写入共享内存，
进程间只同步槽位状态，不把每个 step 的大字典再次经过 OS pipe。当前服务
每次 step 是同步往返，因此这里使用 request/response 双槽；后续如果要支持
生产者和消费者重叠，可将双槽扩展成 ring buffer。

示例：

    python tools/model_runner_ipc_bench.py --mode all --iterations 2000
    python tools/model_runner_ipc_bench.py --mode all --compute-us 1000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import statistics
import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

try:
    import msgspec.msgpack
except ImportError as exc:  # pragma: no cover - environment validation
    raise SystemExit("optimized mode requires msgspec: pip install msgspec") from exc


HEADER_BYTES = 8


@dataclass
class BenchConfig:
    batch_size: int
    input_len: int
    output_len: int
    iterations: int
    compute_us: int
    max_message_bytes: int


def build_batch(config: BenchConfig, step: int) -> dict[str, Any]:
    """构造接近当前 wire batch 的 request 数据。

    普通 decode 服务中，prompt token、已生成 token、block table、slot mapping
    和采样参数会随 step 进入 ``_batch_to_wire``。这里保留相同的主要字段，且
    让 generated_token_ids 随 step 增长，避免 benchmark 只测固定小消息。
    """
    generated_len = step % (config.output_len + 1)
    requests = []
    for request_index in range(config.batch_size):
        requests.append(
            {
                "request_id": f"request-{request_index}",
                "prompt_token_ids": list(range(config.input_len)),
                "generated_token_ids": list(range(generated_len)),
                "block_table": list(range((config.input_len + generated_len + 255) // 256)),
                "slot_mapping": list(range(config.input_len + generated_len)),
                "num_local_cached_tokens": 0,
                "num_computed_tokens": config.input_len + generated_len,
                "num_blocks": (config.input_len + generated_len + 255) // 256,
                "kv_transfer_params": None,
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 0.8,
                    "top_k": 1,
                    "max_tokens": config.output_len,
                },
                "is_multimodal": False,
            }
        )
    return {
        "is_prefill": False,
        "requests": requests,
        "kv_connector_metadata": None,
        "step": step,
    }


def build_response(config: BenchConfig, step: int) -> dict[str, Any]:
    """构造接近当前 ModelRunnerOutput 的 response 数据。"""
    return {
        "req_ids": [f"request-{i}" for i in range(config.batch_size)],
        "sampled_token_ids": [[step % 32000] for _ in range(config.batch_size)],
        "kv_connector_output": {
            "finished_sending": [],
            "finished_recving": [],
            "failed_recving": [],
            "invalid_block_ids": [],
            "kv_connector_stats": None,
        },
        "forward_timing": {
            "started_at_ns": 0,
            "completed_at_ns": 0,
            "forward_batch_id": step,
            "source": "ipc-bench",
        },
    }


def _simulate_compute(compute_us: int) -> None:
    if compute_us > 0:
        time.sleep(compute_us / 1_000_000)


def _pipe_worker(conn: Connection, config: BenchConfig) -> None:
    """baseline worker：复现 Pipe + pickle 的生产通信路径。"""
    conn.send({"ready": True})
    try:
        while True:
            command = conn.recv()
            if command["op"] == "close":
                conn.send({"closed": True})
                return
            step = command["step"]
            _simulate_compute(config.compute_us)
            conn.send(build_response(config, step))
    finally:
        conn.close()


def _write_shared_memory(shm: SharedMemory, payload: bytes) -> None:
    if len(payload) > len(shm.buf) - HEADER_BYTES:
        raise ValueError(
            f"message is {len(payload)} bytes, shared-memory slot only supports "
            f"{len(shm.buf) - HEADER_BYTES} bytes"
        )
    shm.buf[:HEADER_BYTES] = len(payload).to_bytes(HEADER_BYTES, "little")
    shm.buf[HEADER_BYTES : HEADER_BYTES + len(payload)] = payload


def _read_shared_memory(shm: SharedMemory) -> bytes:
    size = int.from_bytes(shm.buf[:HEADER_BYTES], "little")
    if size > len(shm.buf) - HEADER_BYTES:
        raise ValueError(f"invalid shared-memory message size: {size}")
    return bytes(shm.buf[HEADER_BYTES : HEADER_BYTES + size])


def _shared_memory_worker(
    control_conn: Connection,
    request_name: str,
    response_name: str,
    request_ready: Any,
    response_ready: Any,
    config: BenchConfig,
) -> None:
    """optimized worker：共享内存传输消息，Semaphore 只负责通知。"""
    request_shm = SharedMemory(name=request_name)
    response_shm = SharedMemory(name=response_name)
    control_conn.send({"ready": True})
    try:
        while True:
            request_ready.acquire()
            command = msgspec.msgpack.decode(_read_shared_memory(request_shm))
            if command["op"] == "close":
                response = msgspec.msgpack.encode({"closed": True})
                _write_shared_memory(response_shm, response)
                response_ready.release()
                return
            step = command["step"]
            _simulate_compute(config.compute_us)
            response = msgspec.msgpack.encode(build_response(config, step))
            _write_shared_memory(response_shm, response)
            response_ready.release()
    finally:
        request_shm.close()
        response_shm.close()
        control_conn.close()


def _percentile(values: list[float], percentile: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, int((len(values) - 1) * percentile))
    return values[index]


def _result(mode: str, config: BenchConfig, samples_ns: list[int], payload_bytes: dict[str, int]):
    samples_ms = [value / 1_000_000 for value in samples_ns]
    return {
        "mode": mode,
        "config": asdict(config),
        "request_bytes": payload_bytes["request"],
        "response_bytes": payload_bytes["response"],
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p99_ms": _percentile(samples_ms, 0.99),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "steps_per_second": 1000 / statistics.mean(samples_ms),
    }


def run_pipe(config: BenchConfig) -> dict[str, Any]:
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe()
    process = context.Process(target=_pipe_worker, args=(child_conn, config))
    process.start()
    try:
        parent_conn.recv()
        samples = []
        request_size = response_size = 0
        for step in range(config.iterations):
            command = {"op": "execute", "step": step, "batch": build_batch(config, step)}
            started = time.perf_counter_ns()
            parent_conn.send(command)
            response = parent_conn.recv()
            elapsed = time.perf_counter_ns() - started
            if response["req_ids"][-1] != f"request-{config.batch_size - 1}":
                raise RuntimeError("Pipe response validation failed")
            samples.append(elapsed)
            request_size = len(pickle.dumps(command, protocol=pickle.HIGHEST_PROTOCOL))
            response_size = len(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
        parent_conn.send({"op": "close"})
        parent_conn.recv()
        return _result("pipe_pickle", config, samples, {"request": request_size, "response": response_size})
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent_conn.close()


def run_shared_memory(config: BenchConfig) -> dict[str, Any]:
    context = mp.get_context("spawn")
    max_request = msgspec.msgpack.encode(
        {"op": "execute", "step": config.output_len, "batch": build_batch(config, config.output_len)}
    )
    max_response = msgspec.msgpack.encode(build_response(config, config.output_len))
    slot_size = max(config.max_message_bytes, len(max_request), len(max_response)) + HEADER_BYTES
    request_shm = SharedMemory(create=True, size=slot_size)
    response_shm = SharedMemory(create=True, size=slot_size)
    request_ready = context.Semaphore(0)
    response_ready = context.Semaphore(0)
    parent_conn, child_conn = context.Pipe()
    process = context.Process(
        target=_shared_memory_worker,
        args=(child_conn, request_shm.name, response_shm.name, request_ready, response_ready, config),
    )
    process.start()
    try:
        parent_conn.recv()
        samples = []
        request_size = response_size = 0
        for step in range(config.iterations):
            command = {"op": "execute", "step": step, "batch": build_batch(config, step)}
            started = time.perf_counter_ns()
            encoded_command = msgspec.msgpack.encode(command)
            _write_shared_memory(request_shm, encoded_command)
            request_ready.release()
            response_ready.acquire()
            response = msgspec.msgpack.decode(_read_shared_memory(response_shm))
            elapsed = time.perf_counter_ns() - started
            if response["req_ids"][-1] != f"request-{config.batch_size - 1}":
                raise RuntimeError("shared-memory response validation failed")
            samples.append(elapsed)
            request_size = len(encoded_command)
            response_size = len(msgspec.msgpack.encode(response))

        close_message = msgspec.msgpack.encode({"op": "close"})
        _write_shared_memory(request_shm, close_message)
        request_ready.release()
        response_ready.acquire()
        return _result(
            "shared_memory_msgpack",
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
        request_shm.close()
        response_shm.close()
        request_shm.unlink()
        response_shm.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pipe", "shared-memory", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--compute-us",
        type=int,
        default=0,
        help="每个 step 在 worker 中模拟的计算时间；默认 0，只测 IPC",
    )
    parser.add_argument(
        "--max-message-bytes",
        type=int,
        default=8 * 1024 * 1024,
        help="optimized 模式每个共享内存槽的最小容量",
    )
    parser.add_argument("--json", type=Path, help="将结果额外写入 JSON 文件")
    args = parser.parse_args()
    for name in ("batch_size", "input_len", "output_len", "iterations", "compute_us"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.batch_size == 0 or args.input_len == 0 or args.output_len == 0 or args.iterations == 0:
        parser.error("batch-size/input-len/output-len/iterations must be positive")
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
    modes = ["pipe", "shared-memory"] if args.mode == "all" else [args.mode]
    results = []
    for mode in modes:
        result = run_pipe(config) if mode == "pipe" else run_shared_memory(config)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if len(results) == 2:
        baseline, optimized = results
        print("\n=== 对比 ===")
        print(f"baseline mean:  {baseline['mean_ms']:.3f} ms")
        print(f"optimized mean: {optimized['mean_ms']:.3f} ms")
        print(
            "mean improvement: "
            f"{(baseline['mean_ms'] - optimized['mean_ms']) / baseline['mean_ms'] * 100:.2f}%"
        )
        print(
            "p99 improvement: "
            f"{(baseline['p99_ms'] - optimized['p99_ms']) / baseline['p99_ms'] * 100:.2f}%"
        )
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
