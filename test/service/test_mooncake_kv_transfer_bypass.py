from types import SimpleNamespace

import pytest

from infinilm.kv_connector.mooncake import mooncake_connector_worker
from infinilm.infinilm_environment import (
    MOONCAKE_KV_TRANSFER_BYPASS_DELAY_MS_ENV,
    get_mooncake_kv_transfer_bypass_delay_ms,
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
