import asyncio
import threading

import pytest

from infinilm.kv_connector.mooncake.mooncake_connector_worker import (
    MooncakeConnectorWorker,
)
from infinilm.infinilm_environment import (
    MOONCAKE_GET_FINISHED_LEGACY_MODE,
    MOONCAKE_GET_FINISHED_LOCK_SNAPSHOT_MODE,
    MOONCAKE_GET_FINISHED_MODE_ENV,
    get_mooncake_get_finished_mode,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, MOONCAKE_GET_FINISHED_LOCK_SNAPSHOT_MODE),
        ("", MOONCAKE_GET_FINISHED_LOCK_SNAPSHOT_MODE),
        ("legacy", MOONCAKE_GET_FINISHED_LEGACY_MODE),
        (" LOCK_SNAPSHOT ", MOONCAKE_GET_FINISHED_LOCK_SNAPSHOT_MODE),
    ],
)
def test_mooncake_get_finished_mode_selection(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(MOONCAKE_GET_FINISHED_MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(MOONCAKE_GET_FINISHED_MODE_ENV, value)

    assert get_mooncake_get_finished_mode() == expected


@pytest.mark.parametrize("value", ["eventloop", "snapshot", "true", "none"])
def test_mooncake_get_finished_mode_rejects_unknown_value(monkeypatch, value):
    monkeypatch.setenv(MOONCAKE_GET_FINISHED_MODE_ENV, value)

    with pytest.raises(RuntimeError, match="must be one of"):
        get_mooncake_get_finished_mode()


def make_decode_worker(
    mode: str = MOONCAKE_GET_FINISHED_LOCK_SNAPSHOT_MODE,
) -> MooncakeConnectorWorker:
    worker = object.__new__(MooncakeConnectorWorker)
    worker.is_kv_producer = False
    worker.is_kv_consumer = True
    worker.tp_rank = 0
    worker.get_finished_mode = mode
    worker.finished_recving_reqs = set()
    worker.xfer_failed_recving_reqs_ids = set()
    worker._receiver_finished_lock = threading.Lock()
    worker.shutdown = lambda: None
    return worker


def test_get_finished_takes_local_receiver_snapshot_without_event_loop():
    worker = make_decode_worker()

    worker._publish_receiver_finished_state({"finished"}, {"failed"})

    assert worker.get_finished() == (None, {"failed"}, {"finished"})
    assert worker.finished_recving_reqs == set()
    assert worker.xfer_failed_recving_reqs_ids == set()
    assert worker.get_finished() == (None, None, None)


def test_receiver_publishes_finished_and_failed_states_together():
    worker = make_decode_worker()
    started = threading.Event()
    writer_finished = threading.Event()

    def publish_states():
        started.set()
        for index in range(2000):
            request_id = f"request-{index}"
            worker._publish_receiver_finished_state({request_id}, {request_id})
        writer_finished.set()

    writer = threading.Thread(target=publish_states)
    writer.start()
    started.wait()

    while not writer_finished.is_set():
        finished_sending, failed, finished = worker.get_finished()
        assert finished_sending is None
        assert finished == failed

    writer.join()


def test_legacy_get_finished_uses_receiver_event_loop():
    worker = make_decode_worker(MOONCAKE_GET_FINISHED_LEGACY_MODE)
    receiver_loop = asyncio.new_event_loop()
    receiver_thread = threading.Thread(
        target=worker._async_loop,
        args=(receiver_loop,),
        daemon=True,
    )
    receiver_thread.start()
    worker.receiver_loop = receiver_loop

    try:
        worker._publish_receiver_finished_state({"finished"}, {"failed"})

        assert worker.get_finished() == (None, {"failed"}, {"finished"})
        assert worker.finished_recving_reqs == set()
        assert worker.xfer_failed_recving_reqs_ids == set()
        assert worker.get_finished() == (None, None, None)
    finally:
        receiver_loop.call_soon_threadsafe(receiver_loop.stop)
        receiver_thread.join(timeout=5)
