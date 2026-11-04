from collections import deque
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from infinilm.kv_connector.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    PullReqMeta,
)
from infinilm.kv_connector.mooncake.mooncake_connector_scheduler import (
    MooncakeConnectorScheduler,
)
from infinilm.llm.model_runner.model_runner import (
    KVConnectorOutput,
    ModelForwardTiming,
    ModelRunner,
    ModelRunnerOutput,
)
from infinilm.llm.model_runner.model_runner_channel import (
    model_runner_ipc_busy_loop_us,
)
from infinilm.llm.model_runner.model_runner_process import (
    ModelRunnerFuture,
    ModelRunnerProcessClient,
    _update_forward_timing,
    model_runner_process_enabled,
)
from infinilm.llm.model_runner.model_runner_wire import (
    WireRequest,
    batch_to_wire,
    connector_control_response_from_wire,
    connector_control_response_to_wire,
    metadata_from_wire,
    metadata_to_wire,
    model_runner_fixed_array_enabled,
    model_runner_request_cache_enabled,
    requests_from_wire,
    response_from_wire,
    response_to_wire,
)
from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler, SchedulerOutput


def _request(request_id: str) -> InferenceRequest:
    request = InferenceRequest(
        request_id=request_id,
        prompt_token_ids=[11, 12, 13],
        sampling_params=SamplingParams(max_tokens=8),
    )
    request._generated_token_ids.append(14)
    request.block_table = [3, 4]
    request.slot_mapping = [30]
    request.num_local_cached_tokens = 2
    request.num_computed_tokens = 2
    request.num_blocks = 2
    return request


def test_model_batch_wire_round_trip_preserves_text_execution_state():
    request = _request("request-1")
    batch = SchedulerOutput([request], is_prefill=False)

    wire = batch_to_wire(batch)
    restored = WireRequest(wire["requests"][0])

    assert restored.request_id == request.request_id
    assert list(restored.prompt_token_ids) == [11, 12, 13]
    assert restored.generated_token_ids == [14]
    assert restored.block_table == [3, 4]
    assert restored.slot_mapping == [30]
    assert restored.num_local_cached_tokens == 2
    assert restored.get_total_length() == 4


def test_request_cache_wire_sends_full_once_then_decode_diff(monkeypatch):
    monkeypatch.setenv("INFINILM_MODEL_RUNNER_FIXED_ARRAY", "1")
    request = _request("request-cache")
    batch = SchedulerOutput([request], is_prefill=False)
    request_state = {"request-cache": 1}

    wire = batch_to_wire(batch, request_state)
    cache = {"request-cache": WireRequest(_request_to_wire_cached(request))}
    assert wire["layout"] == "diff_array"
    restored = requests_from_wire(wire, cache)[0]

    assert wire["request_cache"] == {"full": 0, "diff": 1}
    assert restored.prompt_token_ids == tuple(request.prompt_token_ids)
    assert restored.sampling_params.max_tokens == request.sampling_params.max_tokens
    assert restored.generated_token_ids == list(request.generated_token_ids)


def test_decode_diff_array_preserves_variable_blocks_and_tokens(monkeypatch):
    monkeypatch.setenv("INFINILM_MODEL_RUNNER_FIXED_ARRAY", "1")
    first = _request("request-array-1")
    second = _request("request-array-2")
    first._generated_token_ids.append(21)
    second._generated_token_ids.append(22)
    second.block_table.append(9)
    second.slot_mapping.append(31)

    batch = SchedulerOutput([first, second], is_prefill=False)
    state = {
        first.request_id: len(first.generated_token_ids) - 1,
        second.request_id: len(second.generated_token_ids) - 1,
    }
    cache = {
        first.request_id: WireRequest(_request_to_wire_cached(first)),
        second.request_id: WireRequest(_request_to_wire_cached(second)),
    }
    # 缓存对象先回到上一步的状态，再应用本步 diff。
    cache[first.request_id].generated_token_ids.pop()
    cache[second.request_id].generated_token_ids.pop()
    cache[second.request_id].block_table.pop()

    wire = batch_to_wire(batch, state)
    restored_first, restored_second = requests_from_wire(wire, cache)

    assert wire["layout"] == "diff_array"
    assert restored_first.generated_token_ids[-1] == 21
    assert restored_second.generated_token_ids[-1] == 22
    assert restored_second.block_table[-1] == 9
    assert restored_second.slot_mapping == [30, 31]


def test_request_cache_rolls_back_to_full_snapshot():
    request = _request("request-rollback")
    batch = SchedulerOutput([request], is_prefill=False)
    request_state = {"request-rollback": 2}

    wire = batch_to_wire(batch, request_state)

    assert wire["request_cache"] == {"full": 1, "diff": 0}
    assert wire["requests"][0]["kind"] == "full"
    assert request_state["request-rollback"] == len(request.generated_token_ids)


def test_request_cache_generation_mismatch_reports_expected_generation():
    request = WireRequest(_request_to_wire_cached(_request("request-mismatch")))

    with pytest.raises(
        RuntimeError,
        match="request-mismatch: expected=2, actual=1",
    ):
        request.apply_diff_values(
            new_generated_token_ids=[15],
            block_table=[3],
            slot_mapping=[30],
            num_local_cached_tokens=2,
            num_computed_tokens=2,
            num_blocks=2,
            kv_transfer_params=request.kv_transfer_params,
            base_generated_tokens=2,
        )


def _request_to_wire_cached(request):
    from infinilm.llm.model_runner.model_runner_wire import (
        request_to_wire as _request_to_wire,
    )

    return _request_to_wire(request)


def test_connector_metadata_wire_round_trip_preserves_pd_send_and_receive():
    metadata = MooncakeConnectorMetadata()
    metadata.reqs_to_send["p-request"] = ("transfer-p", [7, 8])
    metadata.reqs_to_recv["prefill-engine"]["d-request"] = PullReqMeta(
        d_req_id="d-request",
        transfer_id="transfer-d",
        local_block_ids=[9, 10],
        remote_engine_id="prefill-engine",
        remote_bootstrap_addr="http://127.0.0.1:9600",
    )
    metadata.reqs_not_processed.add("transfer-aborted")

    restored = metadata_from_wire(metadata_to_wire(metadata))

    assert restored.reqs_to_send == {"p-request": ("transfer-p", [7, 8])}
    received = restored.reqs_to_recv["prefill-engine"]["d-request"]
    assert received.transfer_id == "transfer-d"
    assert received.local_block_ids == [9, 10]
    assert restored.reqs_not_processed == {"transfer-aborted"}


def test_empty_mooncake_connector_metadata_is_not_dispatched():
    scheduler = MooncakeConnectorScheduler(
        SimpleNamespace(kv_role="kv_producer"),
        engine_id="prefill-engine",
    )

    assert scheduler.build_connector_meta() is None
    assert MooncakeConnectorMetadata().is_empty()


def test_single_process_connector_control_accepts_empty_metadata_step():
    runner = object.__new__(ModelRunner)
    runner.kv_connector = Mock()
    runner.kv_connector.get_finished.return_value = (set(), set(), set())
    runner.kv_connector.get_block_ids_with_load_errors.return_value = set()
    runner.kv_connector.get_kv_connector_stats.return_value = {}
    scheduler_output = SchedulerOutput(scheduled_requests=[])

    output = runner.execute_connector_control(scheduler_output)

    runner.kv_connector.bind_connector_metadata.assert_not_called()
    runner.kv_connector.start_load_kv.assert_not_called()
    runner.kv_connector.get_finished.assert_called_once_with(
        "finished_req_ids"
    )
    assert output.req_ids == []
    assert output.sampled_token_ids == []
    assert output.kv_connector_output.is_empty()


def test_connector_control_response_omits_empty_output_and_round_trips_events():
    empty_payload = connector_control_response_to_wire(
        ModelRunnerOutput(kv_connector_output=KVConnectorOutput())
    )
    assert empty_payload is None
    restored_empty = connector_control_response_from_wire(empty_payload)
    assert restored_empty.kv_connector_output is None

    output = ModelRunnerOutput(
        kv_connector_output=KVConnectorOutput(
            finished_sending={"p-request"},
            kv_connector_stats={"p-request": {"duration_ns": 123}},
        )
    )
    payload = connector_control_response_to_wire(output)
    restored = connector_control_response_from_wire(payload)

    assert payload is not None
    assert restored.kv_connector_output.finished_sending == {"p-request"}
    assert restored.kv_connector_output.kv_connector_stats == {
        "p-request": {"duration_ns": 123}
    }


def test_model_response_omits_empty_connector_output():
    output = ModelRunnerOutput(
        req_ids=["request-1"],
        sampled_token_ids=[7],
        kv_connector_output=KVConnectorOutput(),
    )

    payload = response_to_wire(output)
    restored = response_from_wire(payload)

    assert payload["kv_connector_output"] is None
    assert restored.sampled_token_ids == [7]
    assert restored.kv_connector_output is None


def test_process_mode_accepts_boolean_environment_values(monkeypatch):

    monkeypatch.setenv("INFINILM_MODEL_RUNNER_PROCESS", "yes")
    assert model_runner_process_enabled()
    monkeypatch.setenv("INFINILM_MODEL_RUNNER_PROCESS", "0")
    assert not model_runner_process_enabled()


def test_request_cache_mode_accepts_boolean_environment_values(monkeypatch):
    monkeypatch.setenv("INFINILM_MODEL_RUNNER_REQUEST_CACHE", "off")
    assert not model_runner_request_cache_enabled()
    monkeypatch.setenv("INFINILM_MODEL_RUNNER_REQUEST_CACHE", "true")
    assert model_runner_request_cache_enabled()


def test_process_mode_updates_summary_forward_timing():
    request = SimpleNamespace(
        timing_steps=[],
        timing_mode="summary",
        record_summary_prefill_forward=Mock(),
        record_summary_decode_forward=Mock(),
    )
    scheduler_output = SchedulerOutput([request], is_prefill=False)
    timing = ModelForwardTiming(
        engine_forward_started_at_ns=10,
        engine_forward_completed_at_ns=20,
        forward_batch_id="forward-1",
    )

    _update_forward_timing(
        scheduler_output,
        SimpleNamespace(forward_timing=timing),
    )

    request.record_summary_decode_forward.assert_called_once_with(10, 20)
    request.record_summary_prefill_forward.assert_not_called()


def test_ipc_busy_loop_configuration(monkeypatch):
    monkeypatch.setenv("INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US", "20000")
    assert model_runner_ipc_busy_loop_us() == 20000
    monkeypatch.setenv("INFINILM_MODEL_RUNNER_IPC_BUSY_LOOP_US", "0")
    assert model_runner_ipc_busy_loop_us() == 0


def test_fixed_array_layout_is_opt_in(monkeypatch):
    monkeypatch.delenv("INFINILM_MODEL_RUNNER_FIXED_ARRAY", raising=False)
    assert not model_runner_fixed_array_enabled()
    request = _request("request-per-request-layout")
    batch = SchedulerOutput([request], is_prefill=False)
    state = {request.request_id: len(request.generated_token_ids)}
    wire = batch_to_wire(batch, state)
    assert wire["layout"] == "per_request"


def test_response_wire_uses_flat_int_array_for_single_token_outputs(monkeypatch):
    monkeypatch.setenv("INFINILM_MODEL_RUNNER_FIXED_ARRAY", "1")
    from infinilm.llm.model_runner.model_runner import ModelRunnerOutput

    output = ModelRunnerOutput(
        req_ids=["request-1", "request-2"],
        sampled_token_ids=[7, 8],
    )
    payload = response_to_wire(output)
    restored = response_from_wire(payload)

    assert payload["sampled_layout"] == "int_array"
    assert restored.sampled_token_ids == [7, 8]


class _FakeConnection:
    def poll(self, timeout):
        return False


class _FakeProcess:
    def __init__(self, alive):
        self._alive = alive
        self.exitcode = 17

    def is_alive(self):
        return self._alive


def _pending_future(process_alive):
    client = object.__new__(ModelRunnerProcessClient)
    client._pending_futures = deque()
    client._parent_conn = _FakeConnection()
    client._channel = _FakeConnection()
    client._process = _FakeProcess(process_alive)
    future = ModelRunnerFuture(client, object(), batch_id=3)
    client._pending_futures.append(future)
    return client, future


def _batch_queue_engine(scheduler, model_runner, queue_size=2):
    from infinilm.llm.llm import LLMEngine

    engine = object.__new__(LLMEngine)
    engine._inflight_decode_batches = deque()
    engine._decode_batch_queue_size = queue_size
    engine.scheduler = scheduler
    engine.model_runner = model_runner
    return engine


def test_model_runner_future_timeout_keeps_pending_response():
    client, future = _pending_future(process_alive=True)

    with pytest.raises(FutureTimeoutError):
        future.result(timeout=0.01)

    assert list(client._pending_futures) == [future]
    assert not future.done()


def test_model_runner_future_fails_when_worker_dies():
    client, future = _pending_future(process_alive=False)

    with pytest.raises(RuntimeError, match="exited while executing batch 3"):
        future.result(timeout=1)

    assert not client._pending_futures
    assert isinstance(future.exception(), RuntimeError)


def test_connector_control_empty_output_updates_ipc_counters():
    class ResponseConnection:
        def __init__(self):
            self._response = {
                "ok": True,
                "batch_id": 9,
                "control_output": None,
                "ipc_timing": {
                    "worker_received_at_ns": 101,
                    "runner_execute_ns": 3,
                },
            }

        def poll(self, timeout):
            return True

        def recv(self):
            return self._response

    client = object.__new__(ModelRunnerProcessClient)
    client._pending_futures = deque()
    client._channel = ResponseConnection()
    client._process = _FakeProcess(True)
    client._ipc_timing_samples = {}
    client._ipc_timing_batches = 0
    client._ipc_timing_execute_batches = 0
    client._ipc_timing_connector_control_batches = 0
    client._ipc_timing_connector_metadata_empty = 0
    client._ipc_timing_connector_metadata_nonempty = 0
    client._ipc_timing_connector_empty_output = 0
    client._ipc_timing_connector_nonempty_output = 0
    client._ipc_timing_connector_finished_sending = 0
    client._ipc_timing_connector_finished_recving = 0
    client._ipc_timing_connector_failed_recving = 0
    client._ipc_timing_requests = 0
    client._ipc_timing_full_requests = 0
    client._ipc_timing_diff_requests = 0

    scheduler_output = SimpleNamespace(scheduled_requests=[])
    future = ModelRunnerFuture(
        client,
        scheduler_output,
        batch_id=9,
        command="connector_control",
    )
    future.connector_metadata_empty = True
    future.parent_ipc_timing = {
        "wire_duration_ns": 10,
        "send_started_at_ns": 100,
        "send_duration_ns": 5,
    }
    future.request_cache_stats = {"full": 0, "diff": 0}
    future.submit_started_at_ns = 90
    client._pending_futures.append(future)

    client._resolve_next()

    assert future.result().kv_connector_output is None
    assert client._ipc_timing_connector_control_batches == 1
    assert client._ipc_timing_connector_metadata_empty == 1
    assert client._ipc_timing_connector_empty_output == 1
    assert client._ipc_timing_connector_nonempty_output == 0
    assert "connector_control_runner_execute" in client._ipc_timing_samples


def test_batch_queue_prefill_executes_synchronously(monkeypatch):
    scheduler = Mock()
    scheduler.has_waiting_requests.return_value = False
    scheduler.schedule.return_value = Mock(
        is_prefill=True,
        scheduled_requests=[Mock()],
    )
    model_runner = Mock()
    engine = _batch_queue_engine(scheduler, model_runner)
    committed = []
    monkeypatch.setattr(
        engine,
        "_commit_model_step",
        lambda output, runner_output: (
            committed.append((output, runner_output)) or (True, [])
        ),
    )

    result = engine.step_with_batch_queue()

    assert result == (True, [])
    assert not engine._inflight_decode_batches
    model_runner.execute_model.assert_called_once_with(
        scheduler.schedule.return_value,
        non_block=False,
    )
    assert committed == [
        (
            scheduler.schedule.return_value,
            model_runner.execute_model.return_value,
        )
    ]


def test_batch_queue_decode_executes_non_blocking():
    scheduler = Mock()
    scheduler.has_waiting_requests.return_value = False
    scheduler.schedule.return_value = Mock(
        is_prefill=False,
        scheduled_requests=[Mock()],
    )
    future = Mock()
    model_runner = Mock()
    model_runner.execute_model.return_value = future
    engine = _batch_queue_engine(scheduler, model_runner)

    result = engine.step_with_batch_queue()

    assert result == (True, [])
    assert list(engine._inflight_decode_batches) == [
        (future, scheduler.schedule.return_value)
    ]
    model_runner.execute_model.assert_called_once_with(
        scheduler.schedule.return_value,
        non_block=True,
    )


def test_batch_queue_commits_decode_before_new_prefill(monkeypatch):
    future = Mock()
    future.result.return_value = Mock()
    old_output = Mock(is_prefill=False, scheduled_requests=[Mock()])
    scheduler = Mock()
    scheduler.has_waiting_requests.return_value = True
    model_runner = Mock()
    engine = _batch_queue_engine(scheduler, model_runner)
    engine._inflight_decode_batches.append((future, old_output))
    committed = []
    monkeypatch.setattr(
        engine,
        "_commit_model_step",
        lambda output, runner_output: (
            committed.append((output, runner_output)) or (True, [])
        ),
    )

    result = engine.step_with_batch_queue()

    assert result == (True, [])
    assert not engine._inflight_decode_batches
    assert committed == [(old_output, future.result.return_value)]
    scheduler.schedule.assert_not_called()
    model_runner.execute_model.assert_not_called()


def test_batch_queue_connector_only_step_preserves_pending_output(monkeypatch):
    future = Mock()
    future.result.return_value = Mock()
    old_output = Mock(is_prefill=False, scheduled_requests=[Mock()])
    connector_output = Mock(is_prefill=False, scheduled_requests=[])
    scheduler = Mock()
    scheduler.has_waiting_requests.return_value = False
    scheduler.schedule.return_value = connector_output
    model_runner = Mock()
    engine = _batch_queue_engine(scheduler, model_runner)
    engine._inflight_decode_batches.append((future, old_output))
    old_pending = [Mock()]
    new_pending = [Mock()]

    def commit(output, runner_output):
        if output is old_output:
            return True, old_pending
        return False, new_pending

    monkeypatch.setattr(engine, "_commit_model_step", commit)

    result = engine.step_with_batch_queue()

    assert result == (True, old_pending + new_pending)
    model_runner.execute_connector_control.assert_called_once_with(connector_output)
    model_runner.execute_model.assert_not_called()


def test_scheduler_records_requests_completed_outside_model_batch():
    scheduler = Scheduler(
        max_batch_size=8,
        num_blocks=128,
        block_size=16,
        request_timing_enabled=False,
    )
    request = _request("request-terminal")
    request.block_table = []
    request.slot_mapping = []
    request.num_blocks = 0
    request.mark_canceled()

    scheduler.complete_requests([request])

    assert scheduler.pop_completed_request_ids() == ["request-terminal"]
    assert scheduler.pop_completed_request_ids() == []


def test_engine_releases_all_scheduler_terminal_requests():
    scheduler = Mock()
    scheduler.pop_completed_request_ids.return_value = ["request-a", "request-b"]
    model_runner = Mock()
    engine = _batch_queue_engine(scheduler, model_runner)

    engine._release_completed_request_cache()

    model_runner.release_requests.assert_called_once_with(["request-a", "request-b"])
