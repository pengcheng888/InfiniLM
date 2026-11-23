from types import SimpleNamespace

import pytest
import threading

from infinilm.kv_connector.mooncake import mooncake_connector_worker
from infinilm.infinilm_environment import (
    MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV,
    get_mooncake_kv_transfer_bypass_delay_ms,
)


def test_bypass_delay_environment_name_is_spelled_mooncake():
    assert MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV == (
        "INFINILM_MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS"
    )


@pytest.mark.parametrize("value", [None, "", "none", "NONE", "None"])
def test_bypass_delay_is_disabled_for_unset_or_none(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV, raising=False)
    else:
        monkeypatch.setenv(MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV, value)

    assert get_mooncake_kv_transfer_bypass_delay_ms() is None


@pytest.mark.parametrize("value", ["1", " 5 ", "100"])
def test_bypass_delay_accepts_positive_integers(monkeypatch, value):
    monkeypatch.setenv(MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV, value)

    assert get_mooncake_kv_transfer_bypass_delay_ms() == int(value)


@pytest.mark.parametrize(
    "value",
    ["true", "false", "0", "-1", "1.5", "1ms"],
)
def test_bypass_delay_rejects_non_positive_integers(monkeypatch, value):
    monkeypatch.setenv(MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV, value)

    with pytest.raises(RuntimeError, match="must be"):
        get_mooncake_kv_transfer_bypass_delay_ms()


def test_send_blocks_performs_real_transfer_when_bypass_is_disabled(monkeypatch):
    worker = SimpleNamespace(
        mooncake_kv_transfer_bypass_delay_ms=None,
        engine=SimpleNamespace(
            batch_transfer_sync_write=(
                lambda remote_session, src, dst, lengths: 7
            )
        ),
    )

    result = mooncake_connector_worker.MooncakeConnectorWorker._send_blocks(
        worker,
        remote_session="127.0.0.1:1234",
        src_ptrs=[1],
        dst_ptrs=[2],
        lengths=[3],
        xfer_reqs_ids=["req"],
        xfer_block_ids=[0],
        kv_flag_addr=[],
    )

    assert result == 7


def test_send_blocks_sleeps_and_skips_engine_when_bypass_is_enabled(monkeypatch):
    def fail_transfer(*args, **kwargs):
        raise AssertionError("real Mooncake transfer must be skipped")

    worker = SimpleNamespace(
        mooncake_kv_transfer_bypass_delay_ms=5,
        engine=SimpleNamespace(batch_transfer_sync_write=fail_transfer),
    )

    sleeps = []
    monkeypatch.setattr(
        mooncake_connector_worker.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = mooncake_connector_worker.MooncakeConnectorWorker._send_blocks(
        worker,
        remote_session="127.0.0.1:1234",
        src_ptrs=[1],
        dst_ptrs=[2],
        lengths=[3],
        xfer_reqs_ids=["req"],
        xfer_block_ids=[0],
        kv_flag_addr=[],
    )

    assert result == 0
    assert sleeps == [0.005]


def test_send_blocks_records_boundary_timestamps_when_timing_is_enabled(monkeypatch):
    lock = threading.Lock()
    worker = SimpleNamespace(
        _timing_enabled=True,
        mooncake_kv_transfer_bypass_delay_ms=None,
        engine=SimpleNamespace(
            batch_transfer_sync_write=(
                lambda remote_session, src, dst, lengths: 0
            )
        ),
        _kv_transfer_stats={},
        _kv_transfer_stats_lock=lock,
    )
    worker._record_kv_send_stats = (
        mooncake_connector_worker.MooncakeConnectorWorker._record_kv_send_stats.__get__(
            worker
        )
    )
    timestamps = iter([100, 200])
    monkeypatch.setattr(
        mooncake_connector_worker.time,
        "monotonic_ns",
        lambda: next(timestamps),
    )

    result = mooncake_connector_worker.MooncakeConnectorWorker._send_blocks(
        worker,
        remote_session="127.0.0.1:1234",
        src_ptrs=[1],
        dst_ptrs=[2],
        lengths=[3],
        xfer_reqs_ids=["req"],
        xfer_block_ids=[0],
        kv_flag_addr=[],
        transfer_batch_id="kv-send-123-1",
        request_block_counts={"req": 1},
        request_send_ready_at_ns={"req": 50},
    )

    assert result == 0
    event = worker._kv_transfer_stats["req"]
    assert event["kv_cache_send_ready_at_ns"] == 50
    assert event["batch_transfer_sync_write_started_at_ns"] == 100
    assert event["batch_transfer_sync_write_completed_at_ns"] == 200
