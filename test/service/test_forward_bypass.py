from types import SimpleNamespace

import pytest

import infinilm.infer_engine as infer_engine_module
from infinilm.infer_engine import InferEngine
from infinilm.infinilm_environment import (
    FORWARD_BYPASS_DELAY_MS_ENV,
    get_forward_bypass_delay_ms,
)
from infinilm.llm.model_runner import model_runner


@pytest.mark.parametrize("value", [None, "", "none", "NONE", "None"])
def test_forward_bypass_delay_is_disabled_for_unset_or_none(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(FORWARD_BYPASS_DELAY_MS_ENV, raising=False)
    else:
        monkeypatch.setenv(FORWARD_BYPASS_DELAY_MS_ENV, value)

    assert get_forward_bypass_delay_ms() is None


@pytest.mark.parametrize("value", ["1", " 5 ", "100"])
def test_forward_bypass_delay_accepts_positive_integers(monkeypatch, value):
    monkeypatch.setenv(FORWARD_BYPASS_DELAY_MS_ENV, value)

    assert get_forward_bypass_delay_ms() == int(value)


@pytest.mark.parametrize("value", ["true", "false", "0", "-1", "1.5", "1ms"])
def test_forward_bypass_delay_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv(FORWARD_BYPASS_DELAY_MS_ENV, value)

    with pytest.raises(RuntimeError, match="must be"):
        get_forward_bypass_delay_ms()


def test_forward_bypass_sleeps_and_does_not_enter_real_forward(monkeypatch):
    engine = InferEngine.__new__(InferEngine)
    engine.forward_bypass_delay_ms = 5
    engine._forward_bypass_token_id = 10
    input_offsets = SimpleNamespace(shape=(3,))
    sleeps = []
    created_tensors = []

    def fake_from_list(values, dtype):
        created_tensors.append((values, dtype))
        return created_tensors[-1]

    monkeypatch.setattr(infer_engine_module.infinicore, "from_list", fake_from_list)
    monkeypatch.setattr(
        infer_engine_module.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = InferEngine.forward(
        engine,
        input_ids=None,
        input_offsets=input_offsets,
    )

    assert result == ([10, 10], infer_engine_module.infinicore.int64)
    assert sleeps == [0.005]


def test_forward_raw_rejects_bypass(monkeypatch):
    engine = SimpleNamespace(forward_bypass_delay_ms=1)

    with pytest.raises(NotImplementedError, match="forward_raw"):
        InferEngine.forward_raw(engine, input_ids=SimpleNamespace(shape=(1, 1)))


def test_model_runner_passes_graph_flag_and_skips_weight_loading(monkeypatch):
    config = SimpleNamespace(
        model_path="/model",
        cache_type="paged",
        max_cache_len=1024,
        num_blocks=128,
        block_size=256,
        max_batch_size=1,
        tensor_parallel_size=1,
        moe_ep_backend="disabled",
        moe_ep_size=1,
        pipeline_parallel_size=1,
        pipeline_parallel_stage=0,
        master_addr="127.0.0.1",
        master_port=29500,
        max_num_batched_tokens=1024,
        enable_graph=True,
        attention_backend="default",
        attn_backend="default",
        use_mla=False,
        weight_load_mode="async",
        use_legacy_moe=False,
        pre_transpose=False,
        enable_workspace=True,
        skip_load=False,
        draft_model_path=None,
        kv_transfer_config=None,
    )
    captured_kwargs = {}

    class FakeInferEngine:
        model_type = "fake"
        dtype = object()

        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setenv(FORWARD_BYPASS_DELAY_MS_ENV, "1")
    monkeypatch.setattr(
        model_runner,
        "read_hf_config",
        lambda path: {"text_config": {"max_position_embeddings": 1024}},
    )

    def fake_init_device(self):
        self.device = object()
        self.dtype = object()

    monkeypatch.setattr(model_runner.ModelRunner, "_init_device", fake_init_device)
    monkeypatch.setattr(model_runner, "InferEngine", FakeInferEngine)

    def fail_load(*args, **kwargs):
        raise AssertionError("weights must not load during forward bypass")

    monkeypatch.setattr(model_runner, "load_model_state_dict_by_file", fail_load)

    runner = model_runner.ModelRunner(config, initialize_processor=False)

    assert captured_kwargs["enable_graph_compiling"] is True


def test_model_runner_rejects_speculative_forward_bypass(monkeypatch):
    config = SimpleNamespace(
        draft_model_path="/draft",
        kv_transfer_config=None,
        pipeline_parallel_size=1,
    )
    monkeypatch.setenv(FORWARD_BYPASS_DELAY_MS_ENV, "1")

    with pytest.raises(NotImplementedError, match="speculative"):
        model_runner.ModelRunner(config, initialize_processor=False)
