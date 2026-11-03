from collections import deque
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest

from infinilm.kv_connector.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    PullReqMeta,
)
from infinilm.llm.model_runner.model_runner_process import (
    ModelRunnerFuture,
    ModelRunnerProcessClient,
    _WireRequest,
    _batch_to_wire,
    _metadata_from_wire,
    _metadata_to_wire,
)
from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import SchedulerOutput


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

    wire = _batch_to_wire(batch)
    restored = _WireRequest(wire["requests"][0])

    assert restored.request_id == request.request_id
    assert list(restored.prompt_token_ids) == [11, 12, 13]
    assert restored.generated_token_ids == [14]
    assert restored.block_table == [3, 4]
    assert restored.slot_mapping == [30]
    assert restored.num_local_cached_tokens == 2
    assert restored.get_total_length() == 4


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

    restored = _metadata_from_wire(_metadata_to_wire(metadata))

    assert restored.reqs_to_send == {"p-request": ("transfer-p", [7, 8])}
    received = restored.reqs_to_recv["prefill-engine"]["d-request"]
    assert received.transfer_id == "transfer-d"
    assert received.local_block_ids == [9, 10]
    assert restored.reqs_not_processed == {"transfer-aborted"}


def test_process_mode_accepts_boolean_environment_values(monkeypatch):
    from infinilm.llm.model_runner.model_runner_process import (
        model_runner_process_enabled,
    )

    monkeypatch.setenv("INFINILM_MODEL_RUNNER_PROCESS", "yes")
    assert model_runner_process_enabled()
    monkeypatch.setenv("INFINILM_MODEL_RUNNER_PROCESS", "0")
    assert not model_runner_process_enabled()


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
    client._process = _FakeProcess(process_alive)
    future = ModelRunnerFuture(client, object(), batch_id=3)
    client._pending_futures.append(future)
    return client, future


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
