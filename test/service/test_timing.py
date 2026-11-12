import json
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from infinilm.llm.llm import AsyncLLMEngine
from infinilm.llm.llm import LLMEngine
from infinilm.llm import llm as llm_module
from infinilm.llm.request import FinishReason, InferenceRequest
from infinilm.llm.request import TokenOutput
from infinilm.llm.scheduler import Scheduler
from infinilm.llm.static_scheduler import StaticScheduler
from infinilm.llm.sampling_params import SamplingParams
from infinilm.timing import RequestTiming, TokenTiming
from infinilm.timing.timestamps import ForwardTiming
from infinilm.timing.config import timing_enabled_from_environment
from infinilm.timing.metrics import compute_timing_metrics
from infinilm.timing.recorder import (
    MAX_PENDING_RECORDS,
    MAX_WRITE_BATCH_RECORDS,
    SCHEMA_VERSION,
    TimingRecorder,
)
from infinilm.timing.validate import validate_timing_root


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("0", False),
        ("false", False),
        ("1", True),
        ("TRUE", True),
        (" Yes ", True),
    ],
)
def test_timing_enabled_from_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("INFINILM_TIMING", raising=False)
    else:
        monkeypatch.setenv("INFINILM_TIMING", value)

    assert timing_enabled_from_environment() is expected


def test_timing_enabled_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("INFINILM_TIMING", "enabled")

    with pytest.raises(RuntimeError, match="INFINILM_TIMING"):
        timing_enabled_from_environment()


def test_scheduler_records_timestamp_before_waiting_queue_put():
    request = InferenceRequest(
        request_id="request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=2),
        timing=RequestTiming(),
    )
    scheduler = Scheduler(
        num_blocks=8,
        block_size=1,
        max_batch_size=4,
        max_num_batched_tokens=16,
        enable_prefix_caching=False,
    )

    scheduler.add_request(request)

    assert request.timing is not None
    assert request.timing.scheduler_enqueue_before_at_ns is not None
    assert scheduler.waiting_queue.sync_q.get_nowait() is request


def test_scheduler_does_not_create_timing_for_unsupported_request():
    request = InferenceRequest(
        request_id="request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler = Scheduler(
        num_blocks=8,
        block_size=1,
        max_batch_size=4,
        max_num_batched_tokens=16,
        enable_prefix_caching=False,
    )

    scheduler.add_request(request)

    assert request.timing is None


def test_static_scheduler_does_not_support_timing():
    request = InferenceRequest(
        request_id="request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=2),
        timing=RequestTiming(),
    )
    scheduler = StaticScheduler(enable_prefix_caching=False)

    scheduler.add_request(request)

    assert request.timing.scheduler_enqueue_before_at_ns is None


def test_chat_signature_has_no_request_timing_parameter():
    import inspect

    from infinilm.server.inference_server import InferenceServer

    assert "request_timing" not in inspect.signature(InferenceServer._chat).parameters


def test_speculative_and_pipeline_forward_do_not_return_timing():
    from infinilm.llm.model_runner.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=1)
    runner.processor = SimpleNamespace(
        build_model_inputs=lambda scheduler_output, *args, **kwargs: {}
    )
    runner._is_pd_service = False
    runner.speculative_runner = SimpleNamespace(
        forward=lambda scheduler_output, model_input: [7]
    )

    scheduler_output = SimpleNamespace(scheduled_requests=[], is_prefill=True)
    assert runner._model_forward(scheduler_output)[1] is None

    runner.speculative_runner = None
    runner.pipeline_control = SimpleNamespace(
        dispatch_forward=lambda model_input: None,
        wait_forward=lambda: None,
        abort=lambda: None,
    )
    runner.model_engine = SimpleNamespace(
        forward=lambda **kwargs: SimpleNamespace(
            to_numpy=lambda: SimpleNamespace(tolist=lambda: [7])
        )
    )
    assert runner._model_forward(scheduler_output)[1] is None


def test_forward_timing_is_disabled_when_requests_do_not_participate():
    from infinilm.llm.model_runner.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=1)
    runner.processor = SimpleNamespace(
        build_model_inputs=lambda scheduler_output, *args, **kwargs: {}
    )
    runner._is_pd_service = False
    runner.speculative_runner = None
    runner.pipeline_control = None
    runner._forward_timing_enabled = False
    runner.model_engine = SimpleNamespace(
        forward=lambda **kwargs: SimpleNamespace(
            to_numpy=lambda: SimpleNamespace(tolist=lambda: [7])
        )
    )
    request = SimpleNamespace(timing=RequestTiming())
    scheduler_output = SimpleNamespace(scheduled_requests=[request], is_prefill=True)

    assert runner._model_forward(scheduler_output)[1] is None


def test_kv_connector_output_records_get_finished_boundaries(monkeypatch):
    from infinilm.llm.model_runner import model_runner as model_runner_module
    from infinilm.llm.model_runner.model_runner import ModelRunner

    timestamps = iter(range(100, 200))
    monkeypatch.setattr(
        model_runner_module.time,
        "monotonic_ns",
        lambda: next(timestamps),
    )
    runner = object.__new__(ModelRunner)
    runner._forward_timing_enabled = True
    runner.kv_connector = SimpleNamespace(
        bind_connector_metadata=lambda metadata: None,
        start_load_kv=lambda: None,
        get_finished=lambda name: ({"sent"}, {"failed"}, {"received"}),
        get_block_ids_with_load_errors=lambda: set(),
        get_kv_connector_stats=lambda: {},
    )

    with runner.maybe_get_kv_connector_output(
        SimpleNamespace(kv_connector_metadata={})
    ) as output:
        pass

    assert output.kv_connector_finalize_started_at_ns == 100
    assert output.kv_connector_get_finished_started_at_ns == 101
    assert output.kv_connector_get_finished_completed_at_ns == 102
    assert output.kv_connector_finalize_completed_at_ns == 103
    assert output.finished_sending == {"sent"}
    assert output.failed_recving == {"failed"}
    assert output.finished_recving == {"received"}


def test_batch_put_records_api_loop_boundaries():
    engine = object.__new__(AsyncLLMEngine)
    output_queue = SimpleNamespace(put_nowait=lambda item: None)
    token_timing = TokenTiming(token_index=0)
    output = SimpleNamespace(timing=token_timing)

    engine._batch_put([(output_queue, output)])

    assert token_timing.batch_put_started_at_ns is not None
    assert (
        token_timing.batch_put_completed_at_ns >= token_timing.batch_put_started_at_ns
    )


def test_batch_put_does_not_call_clock_when_no_output_is_sampled(monkeypatch):
    engine = object.__new__(AsyncLLMEngine)

    def fail_monotonic():
        raise AssertionError("timing-disabled output must not read the clock")

    monkeypatch.setattr(llm_module.time, "monotonic_ns", fail_monotonic)
    output_queue = SimpleNamespace(put_nowait=lambda item: None)
    output = SimpleNamespace(timing=None)

    engine._batch_put([(output_queue, output)])


def test_token_timing_is_created_only_for_sampled_tokens():
    engine = object.__new__(LLMEngine)
    engine.timing_recorder = SimpleNamespace(
        enabled=True,
        should_record_token=lambda token_index, finished: (
            token_index == 0 or finished or (token_index + 1) % 32 == 0
        ),
    )
    request = InferenceRequest(
        request_id="request",
        prompt_token_ids=[1],
        timing=RequestTiming(),
    )
    forward_timing = ForwardTiming(
        batch_id="batch",
        forward_started_at_ns=100,
        forward_completed_at_ns=200,
    )

    assert (
        engine._make_token_timing(
            request,
            token_index=0,
            forward_timing=forward_timing,
            phase="prefill",
            finished=False,
        )
        is not None
    )
    assert (
        engine._make_token_timing(
            request,
            token_index=1,
            forward_timing=forward_timing,
            phase="decode",
            finished=False,
        )
        is None
    )
    assert (
        engine._make_token_timing(
            request,
            token_index=31,
            forward_timing=forward_timing,
            phase="decode",
            finished=False,
        )
        is not None
    )
    assert (
        engine._make_token_timing(
            request,
            token_index=5,
            forward_timing=forward_timing,
            phase="decode",
            finished=True,
        )
        is not None
    )


def test_step_loop_records_submit_timestamp_before_call_soon_threadsafe():
    engine = object.__new__(AsyncLLMEngine)
    engine._running = True
    engine._step_wakeup = threading.Event()
    engine._drain_abort_queue = lambda: None
    engine.engine = SimpleNamespace(step=lambda: (True, []))

    token_timing = TokenTiming(token_index=0)
    output = SimpleNamespace(timing=token_timing)
    pending = [(SimpleNamespace(), output)]

    observed = []

    def call_soon_threadsafe(callback, args):
        observed.append(
            (
                token_timing.loop_submit_started_at_ns,
                time.monotonic_ns(),
            )
        )
        engine._running = False

    engine._loop = SimpleNamespace(call_soon_threadsafe=call_soon_threadsafe)
    engine.engine.step = lambda: (True, pending)

    engine._step_loop()

    assert len(observed) == 1
    submit_started_at_ns, call_entered_at_ns = observed[0]
    assert submit_started_at_ns is not None
    assert submit_started_at_ns <= call_entered_at_ns


def test_record_timing_token_populates_first_api_loop_timestamps():
    engine = object.__new__(AsyncLLMEngine)
    recorded = []
    engine.engine = SimpleNamespace(
        timing_enabled=True,
        timing_recorder=SimpleNamespace(
            should_record_token=lambda token_index, finished: True,
            record_token=recorded.append,
        ),
    )
    request = InferenceRequest(
        request_id="request",
        prompt_token_ids=[1],
        timing=RequestTiming(),
    )
    output = TokenOutput(
        request_id="request",
        token_id=1,
        token_text="x",
        timing=TokenTiming(
            token_index=0,
            loop_submit_started_at_ns=100,
            batch_put_started_at_ns=110,
            batch_put_completed_at_ns=120,
            route_token_received_at_ns=130,
            route_token_end_at_ns=140,
        ),
    )

    engine.record_timing_token(request, output)

    assert request.timing.first_loop_submit_started_at_ns == 100
    assert request.timing.first_batch_put_started_at_ns == 110
    assert request.timing.first_batch_put_completed_at_ns == 120
    assert len(recorded) == 1


def test_finish_route_token_timing_is_idempotent():
    from infinilm.server.inference_server import InferenceServer

    server = object.__new__(InferenceServer)
    recorded = []
    server.engine = SimpleNamespace(
        record_timing_token=lambda request, output: recorded.append(output)
    )
    request = InferenceRequest(
        request_id="request",
        prompt_token_ids=[1],
        timing=RequestTiming(),
    )
    output = TokenOutput(
        request_id="request",
        token_id=1,
        token_text="x",
        timing=TokenTiming(token_index=0),
    )

    server._finish_route_token_timing(request, output)
    first_end_at_ns = output.timing.route_token_end_at_ns
    server._finish_route_token_timing(request, output)

    assert output.timing.route_token_end_at_ns == first_end_at_ns
    assert recorded == [output]


def test_stream_route_records_empty_and_finished_tokens():
    from infinilm.server.inference_server import InferenceServer

    server = object.__new__(InferenceServer)
    server.model_id = "model"
    server._build_sampling_params = lambda data: SamplingParams(ignore_eos=True)
    recorded = []

    async def run():
        request = InferenceRequest(
            request_id="request",
            prompt_token_ids=[1],
            timing=RequestTiming(),
        )

        async def stream_request(req, timeout, request_timeout):
            yield TokenOutput(
                request_id=req.request_id,
                token_id=1,
                token_text="",
                timing=TokenTiming(token_index=0),
            )
            req.mark_finished(FinishReason.LENGTH)
            yield TokenOutput(
                request_id=req.request_id,
                token_id=2,
                token_text="",
                finished=True,
                finish_reason=FinishReason.LENGTH,
                timing=TokenTiming(token_index=1),
            )

        server.engine = SimpleNamespace(
            add_chat_request=lambda **kwargs: request,
            stream_request=stream_request,
            record_timing_token=lambda req, output: recorded.append(output),
            record_timing_finish=lambda req: None,
            add_aborted_req=lambda req, reason: None,
            engine=SimpleNamespace(eos_token_ids=[]),
        )

        async def is_disconnected():
            return False

        http_request = SimpleNamespace(is_disconnected=is_disconnected)

        return [
            chunk
            async for chunk in server._stream_chat(
                "request", {}, http_request, RequestTiming()
            )
        ]

    chunks = asyncio.run(run())

    assert len(recorded) == 2
    assert recorded[0].token_text == ""
    assert all(output.timing.route_token_end_at_ns is not None for output in recorded)
    assert chunks[-1] == "data: [DONE]\n\n"


def test_connector_only_step_without_forward_timing_keeps_step_thread_alive():
    engine = object.__new__(LLMEngine)
    engine.timing_recorder = SimpleNamespace(enabled=False)
    scheduler_output = SimpleNamespace(
        scheduled_requests=[],
        num_requests=0,
        is_prefill=False,
    )
    engine.scheduler = SimpleNamespace(
        schedule=lambda: scheduler_output,
        update_from_output=lambda output: None,
    )
    engine.model_runner = SimpleNamespace(
        execute_model=lambda output: SimpleNamespace(
            forward_timing=None,
            sampled_token_ids=[],
            kv_connector_output=SimpleNamespace(
                finished_sending={"request"},
                finished_recving=None,
            ),
        )
    )
    captured = {}

    def update_requests(requests, sampled_token_ids, forward_timing, timing_phase):
        captured["forward_timing"] = forward_timing
        captured["timing_phase"] = timing_phase
        return []

    engine._update_requests = update_requests

    did_work, pending = engine.step()

    assert did_work is True
    assert pending == []
    assert captured == {"forward_timing": None, "timing_phase": "decode"}


def test_connector_only_step_records_pending_timing_events():
    engine = object.__new__(LLMEngine)
    recorded = []
    event = {"event": "kv_receive", "request_id": "request"}
    engine.timing_recorder = SimpleNamespace(
        enabled=True,
        record_event=lambda event_name, payload: recorded.append((event_name, payload)),
    )
    engine.scheduler = SimpleNamespace(
        schedule=lambda: SimpleNamespace(
            scheduled_requests=[],
            num_requests=0,
            is_prefill=False,
        ),
        update_from_output=lambda output: None,
        pop_timing_events=lambda: [event],
    )
    engine.model_runner = SimpleNamespace(
        execute_model=lambda output: SimpleNamespace(
            forward_timing=None,
            sampled_token_ids=[],
            kv_connector_output=SimpleNamespace(
                finished_recving={"request"},
                kv_connector_stats={"request": event},
            ),
        )
    )
    engine._update_requests = lambda *args: []

    engine.step()

    assert recorded == [("kv_receive", event)]


def test_timing_enabled_engine_rejects_untimed_scheduled_request():
    engine = object.__new__(LLMEngine)
    untimed_request = InferenceRequest(
        request_id="untimed",
        prompt_token_ids=[1],
    )
    scheduler_output = SimpleNamespace(
        scheduled_requests=[untimed_request],
        num_requests=2,
        is_prefill=True,
    )
    engine.timing_recorder = SimpleNamespace(
        enabled=True,
        record_batch=lambda payload: None,
    )
    engine.scheduler = SimpleNamespace(
        schedule=lambda: scheduler_output,
        update_from_output=lambda output: None,
    )
    engine.model_runner = SimpleNamespace(
        execute_model=lambda output: SimpleNamespace(
            forward_timing=ForwardTiming(
                batch_id="batch",
                forward_started_at_ns=100,
                forward_completed_at_ns=200,
            ),
            sampled_token_ids=[],
            kv_connector_output=None,
        )
    )
    engine._update_requests = lambda *args: []

    with pytest.raises(RuntimeError, match="without RequestTiming"):
        engine.step()


def test_non_stream_request_returns_400_when_timing_enabled():
    from fastapi.testclient import TestClient
    from infinilm.server.inference_server import InferenceServer

    server = object.__new__(InferenceServer)
    server.engine = SimpleNamespace(timing_enabled=True)
    app = server._create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": False},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_chat_defensively_rejects_timing_enabled_mode():
    from infinilm.server.inference_server import InferenceServer

    server = object.__new__(InferenceServer)
    server.engine = SimpleNamespace(timing_enabled=True)

    with pytest.raises(RuntimeError, match="Non-stream chat is unsupported"):
        asyncio.run(server._chat("request", {}, SimpleNamespace()))


def test_recorder_close_stops_thread_even_when_queue_is_full(tmp_path):
    recorder = TimingRecorder(root=tmp_path)
    thread = threading.Thread(target=recorder._stop_event.wait)
    recorder._thread = thread
    thread.start()

    for index in range(MAX_PENDING_RECORDS):
        recorder.record_token({"request_id": f"request-{index}"})

    assert recorder._queue.qsize() == MAX_PENDING_RECORDS
    recorder.close()
    thread.join(timeout=2)

    assert not thread.is_alive()


def test_recorder_writes_batch_token_and_finish_records(tmp_path):
    recorder = TimingRecorder(root=tmp_path, sample_interval=32)
    timestamps = {
        "service_entered_at_ns": 100,
        "scheduler_enqueue_before_at_ns": 110,
        "forward_started_at_ns": 120,
        "forward_completed_at_ns": 130,
        "loop_submit_started_at_ns": 140,
        "batch_put_started_at_ns": 150,
        "batch_put_completed_at_ns": 150,
        "route_token_received_at_ns": 160,
        "route_token_end_at_ns": 170,
    }

    recorder.record_batch(
        {
            "request_ids": ["request"],
            "phase": "prefill",
            "batch_size": 1,
            "batch_id": "batch",
            "forward_started_at_ns": 120,
            "forward_completed_at_ns": 130,
        }
    )
    recorder.record_token({"request_id": "request", **timestamps})
    recorder.record_finish({"request_id": "request", **timestamps})
    recorder.close()

    files = list(tmp_path.glob("*/timing.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [record["event"] for record in records] == ["batch", "token", "finish"]
    assert records[1] == {
        "schema_version": SCHEMA_VERSION,
        "event": "token",
        "pid": records[0]["pid"],
        "request_id": "request",
        **timestamps,
    }
    assert recorder.stats() == {"dropped_records": 0, "pending_records": 0}


def test_recorder_writes_generic_timing_event(tmp_path):
    recorder = TimingRecorder(root=tmp_path)
    recorder.record_event("kv_receive", {"request_id": "request"})
    recorder.close()

    records = [
        json.loads(line)
        for line in next(tmp_path.glob("*/timing.jsonl")).read_text().splitlines()
    ]

    assert records[0]["event"] == "kv_receive"
    assert records[0]["request_id"] == "request"
    assert records[0]["schema_version"] == SCHEMA_VERSION


def test_recorder_writes_pending_records_in_batches(tmp_path):
    recorder = TimingRecorder(root=tmp_path)
    recorder._thread = threading.current_thread()

    for index in range(MAX_WRITE_BATCH_RECORDS + 2):
        recorder._queue.put_nowait({"request_id": f"request-{index}"})

    write_sizes = []
    recorder._write_records = lambda records: write_sizes.append(len(records))
    recorder._stop_event.set()
    recorder._write_loop()

    assert write_sizes == [MAX_WRITE_BATCH_RECORDS, 2]
    assert recorder._queue.empty()


def test_validate_and_compute_timing_records(tmp_path):
    pid_dir = tmp_path / "123"
    pid_dir.mkdir()
    first_timestamps = {
        "service_entered_at_ns": 100,
        "scheduler_enqueue_before_at_ns": 110,
        "forward_started_at_ns": 120,
        "forward_completed_at_ns": 130,
        "loop_submit_started_at_ns": 140,
        "batch_put_started_at_ns": 150,
        "batch_put_completed_at_ns": 150,
        "route_token_received_at_ns": 160,
        "route_token_end_at_ns": 170,
    }
    second_timestamps = {
        "service_entered_at_ns": 200,
        "scheduler_enqueue_before_at_ns": 210,
        "forward_started_at_ns": 220,
        "forward_completed_at_ns": 240,
        "loop_submit_started_at_ns": 250,
        "batch_put_started_at_ns": 260,
        "batch_put_completed_at_ns": 260,
        "route_token_received_at_ns": 280,
        "route_token_end_at_ns": 320,
    }
    records = [
        {
            "schema_version": 3,
            "event": "batch",
            "request_ids": ["request"],
            "phase": "prefill",
            "batch_size": 2,
            "batch_id": "batch",
            "forward_started_at_ns": 120,
            "forward_completed_at_ns": 130,
        },
        {
            "schema_version": 3,
            "event": "token",
            "request_id": "request",
            "token_index": 0,
            "finished": True,
            **first_timestamps,
        },
        {
            "schema_version": 3,
            "event": "finish",
            "request_id": "request",
            **first_timestamps,
        },
        {
            "schema_version": 3,
            "event": "token",
            "request_id": "request-2",
            "token_index": 0,
            "finished": True,
            **second_timestamps,
        },
        {
            "schema_version": 3,
            "event": "finish",
            "request_id": "request-2",
            **second_timestamps,
        },
    ]
    (pid_dir / "timing.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    validation = validate_timing_root(tmp_path)
    assert validation["valid"] is True

    metrics = compute_timing_metrics(tmp_path)
    assert metrics["record_count"] == 5
    assert metrics["batch"]["prefill"]["batch_size_counts"] == {"2": 1}
    assert metrics["token_metrics_ms"]["service_to_enqueue_ms"]["count"] == 2
    assert metrics["token_metrics_ms"]["forward_ms"] == pytest.approx(
        {
            "count": 2,
            "avg_ms": 0.000015,
            "min_ms": 0.000010,
            "max_ms": 0.000020,
            "p50_ms": 0.000015,
            "p95_ms": 0.0000195,
            "p99_ms": 0.0000199,
        }
    )
    assert metrics["request_metrics_ms"]["request_first_token_e2e_ms"] == pytest.approx(
        {
            "count": 2,
            "avg_ms": 0.000070,
            "min_ms": 0.000060,
            "max_ms": 0.000080,
            "p50_ms": 0.000070,
            "p95_ms": 0.000079,
            "p99_ms": 0.0000798,
        }
    )
    assert metrics["request_metrics_ms"][
        "request_e2e_to_final_route_ms"
    ] == pytest.approx(
        {
            "count": 2,
            "avg_ms": 0.000095,
            "min_ms": 0.000070,
            "max_ms": 0.000120,
            "p50_ms": 0.000095,
            "p95_ms": 0.0001175,
            "p99_ms": 0.0001195,
        }
    )
    assert metrics["request_metrics_ms"]["request_prefill_forward_ms"][
        "p99_ms"
    ] == pytest.approx(0.0000199)
    assert metrics["request_metrics_ms"]["request_enqueue_to_prefill_start_ms"][
        "p99_ms"
    ] == pytest.approx(0.000010)

    distribution_fields = {
        "count",
        "avg_ms",
        "min_ms",
        "max_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
    }
    for distribution in metrics["token_metrics_ms"].values():
        assert set(distribution) == distribution_fields
    for distribution in metrics["request_metrics_ms"].values():
        assert set(distribution) == distribution_fields
    for phase_metrics in metrics["batch"].values():
        assert set(phase_metrics["forward_duration_ms"]) == distribution_fields


def test_validate_and_compute_kv_connector_get_finished_timing(tmp_path):
    pid_dir = tmp_path / "123"
    pid_dir.mkdir()
    token_timestamps = {
        "service_entered_at_ns": 100,
        "scheduler_enqueue_before_at_ns": 110,
        "forward_started_at_ns": 500,
        "forward_completed_at_ns": 600,
        "loop_submit_started_at_ns": 700,
        "batch_put_started_at_ns": 710,
        "batch_put_completed_at_ns": 711,
        "route_token_received_at_ns": 720,
        "route_token_end_at_ns": 730,
    }
    records = [
        {
            "schema_version": 5,
            "event": "batch",
            "pid": 123,
            "request_ids": ["request"],
            "phase": "decode",
            "batch_size": 1,
            "batch_id": "batch",
            "forward_started_at_ns": 500,
            "forward_completed_at_ns": 600,
            "kv_connector_finalize_started_at_ns": 610,
            "kv_connector_get_finished_started_at_ns": 611,
            "kv_connector_get_finished_completed_at_ns": 620,
            "kv_connector_finalize_completed_at_ns": 650,
        },
        {
            "schema_version": 5,
            "event": "token",
            "pid": 123,
            "request_id": "request",
            "token_index": 0,
            "finished": False,
            "batch_id": "batch",
            **token_timestamps,
        },
    ]
    (pid_dir / "timing.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    validation = validate_timing_root(tmp_path)
    assert validation["valid"] is True

    metrics = compute_timing_metrics(tmp_path)
    post_forward = metrics["post_forward_metrics_ms"]
    assert post_forward["kv_connector_finalize_ms"]["avg_ms"] == pytest.approx(0.00004)
    assert post_forward["kv_connector_get_finished_ms"]["avg_ms"] == pytest.approx(
        0.000009
    )
    assert post_forward["kv_connector_finalize_remaining_ms"][
        "avg_ms"
    ] == pytest.approx(0.000031)
    decode_batch = metrics["batch"]["decode"]
    assert decode_batch["kv_connector_get_finished_ms"]["avg_ms"] == pytest.approx(
        0.000009
    )
    assert decode_batch["kv_connector_finalize_remaining_ms"][
        "avg_ms"
    ] == pytest.approx(0.000031)


def test_validate_and_compute_mooncake_timing_records(tmp_path):
    pid_dir = tmp_path / "123"
    pid_dir.mkdir()
    send_common = {
        "schema_version": 3,
        "event": "kv_send",
        "pid": 123,
        "role": "prefill",
        "scope": "producer_batch_transfer",
        "clock_domain": "local-monotonic",
        "cross_node_comparable": False,
        "batch_transfer_sync_write_started_at_ns": 200,
        "batch_transfer_sync_write_completed_at_ns": 300,
        "batch_size": 2,
        "batch_id": "kv-send-123-1",
        "batch_request_count": 2,
        "batch_block_count": 3,
        "batch_bytes": 300,
        "block_count": 3,
        "bytes": 300,
        "mooncake_kv_transfer_bypassed": False,
        "outcome": "success",
    }
    records = [
        {
            **send_common,
            "request_id": "request-1",
            "kv_cache_send_ready_at_ns": 100,
            "request_block_count": 1,
        },
        {
            **send_common,
            "request_id": "request-2",
            "kv_cache_send_ready_at_ns": 150,
            "request_block_count": 2,
        },
        {
            "schema_version": 3,
            "event": "kv_receive",
            "pid": 123,
            "request_id": "request-1",
            "role": "decode",
            "scope": "decode_local_kv_readiness",
            "clock_domain": "local-monotonic",
            "cross_node_comparable": False,
            "kv_cache_pull_started_at_ns": 350,
            "kv_cache_pull_confirmed_at_ns": 500,
            "block_count": 1,
            "bytes": 100,
            "mooncake_kv_transfer_bypassed": False,
            "outcome": "success",
        },
        {
            "schema_version": 3,
            "event": "batch",
            "pid": 123,
            "request_ids": ["request-1"],
            "phase": "decode",
            "batch_size": 1,
            "batch_id": "decode",
            "forward_started_at_ns": 600,
            "forward_completed_at_ns": 700,
        },
    ]
    (pid_dir / "timing.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    validation = validate_timing_root(tmp_path)
    assert validation["valid"] is True

    metrics = compute_timing_metrics(tmp_path)
    mooncake = metrics["mooncake"]
    assert mooncake["kv_send"]["unique_batch_count"] == 1
    assert mooncake["kv_send"]["request_event_count"] == 2
    assert mooncake["kv_send"]["batch_transfer_sync_write_batch_weighted_ms"] == {
        "count": 1,
        "avg_ms": pytest.approx(0.0001),
        "min_ms": pytest.approx(0.0001),
        "max_ms": pytest.approx(0.0001),
        "p50_ms": pytest.approx(0.0001),
        "p95_ms": pytest.approx(0.0001),
        "p99_ms": pytest.approx(0.0001),
    }
    assert mooncake["kv_receive"]["kv_cache_pull_readiness_ms"]["count"] == 1
    assert mooncake["kv_receive"]["decode_start_after_kv_cache_pull_confirmed_ms"][
        "avg_ms"
    ] == pytest.approx(0.0001)


def test_validate_rejects_timestamp_order(tmp_path):
    pid_dir = tmp_path / "123"
    pid_dir.mkdir()
    record = {
        "schema_version": 3,
        "event": "token",
        "request_id": "request",
        "service_entered_at_ns": 200,
        "scheduler_enqueue_before_at_ns": 100,
        "forward_started_at_ns": 110,
        "forward_completed_at_ns": 120,
        "loop_submit_started_at_ns": 130,
        "batch_put_started_at_ns": 140,
        "batch_put_completed_at_ns": 140,
        "route_token_received_at_ns": 150,
        "route_token_end_at_ns": 160,
    }
    (pid_dir / "timing.jsonl").write_text(json.dumps(record) + "\n")

    report = validate_timing_root(tmp_path)
    assert report["valid"] is False
    assert any("StepThread timestamps" in error for error in report["errors"])
