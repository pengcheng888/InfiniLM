import importlib.util
from pathlib import Path


def load_benchmark_module(monkeypatch):
    path = Path(__file__).parents[2] / "examples" / "bench.py"
    # bench.py resolves its default prompt relative to the repository root.
    monkeypatch.chdir(path.parents[1])
    spec = importlib.util.spec_from_file_location("infinilm_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_prefill_layout(monkeypatch):
    module = load_benchmark_module(monkeypatch)

    layout = module.direct_input_layout(
        "prefill", 3, input_len=1024, past_kv_length=1024, block_size=256
    )

    assert layout["input_ids"] == [[10] * 3072]
    assert layout["position_ids"] == list(range(1024)) * 3
    assert layout["past_kv_lengths"] == [0, 0, 0]
    assert layout["total_kv_lengths"] == [1024, 1024, 1024]
    assert layout["input_offsets"] == [0, 1024, 2048, 3072]
    assert layout["block_tables"][2] == [8, 9, 10, 11]
    assert layout["slot_mapping"][-1] == 2 * 4 * 256 + 1023


def test_direct_decode_layout(monkeypatch):
    module = load_benchmark_module(monkeypatch)

    layout = module.direct_input_layout(
        "decode", 3, input_len=1024, past_kv_length=1024, block_size=256
    )

    assert layout["input_ids"] == [[10, 10, 10]]
    assert layout["position_ids"] == [1024, 1024, 1024]
    assert layout["past_kv_lengths"] == [1024, 1024, 1024]
    assert layout["total_kv_lengths"] == [1025, 1025, 1025]
    assert layout["input_offsets"] == [0, 1, 2, 3]
    assert layout["cu_seqlens"] == [0, 1025, 2050, 3075]
    assert layout["block_tables"][2] == [10, 11, 12, 13, 14]
    assert layout["slot_mapping"] == [1024, 2304, 3584]


def test_direct_decode_step_layout(monkeypatch):
    module = load_benchmark_module(monkeypatch)

    layout = module.direct_input_layout(
        "decode",
        2,
        input_len=1024,
        past_kv_length=1024,
        block_size=256,
        decode_step=3,
        blocks_per_request=5,
    )

    assert layout["position_ids"] == [1027, 1027]
    assert layout["past_kv_lengths"] == [1027, 1027]
    assert layout["total_kv_lengths"] == [1028, 1028]
    assert layout["cu_seqlens"] == [0, 1028, 2056]
    assert layout["blocks_per_request"] == 5
    assert layout["slot_mapping"] == [1027, 2307]


def test_decode_warmup_uses_sampled_tokens(monkeypatch):
    module = load_benchmark_module(monkeypatch)

    class FakeTokens:
        def __init__(self, token_id):
            self.token_id = token_id

        def view(self, shape):
            assert shape == [1, 2]
            return self

    class FakeEngine:
        def __init__(self):
            self.inputs = []

        def forward(self, **kwargs):
            self.inputs.append(kwargs["input_ids"])
            return FakeTokens(len(self.inputs) + 10)

    engine = FakeEngine()
    step_inputs = [
        {"input_ids": [10, 10]},
        {"input_ids": [10, 10]},
        {"input_ids": [10, 10]},
        {"input_ids": [10, 10]},
    ]

    original_sync = module.infinicore.sync_device
    module.infinicore.sync_device = lambda: None
    try:
        next_tokens = module.run_direct_decode_warmup(
            engine, step_inputs, 2, 1.0, 1.0, 1
        )
    finally:
        module.infinicore.sync_device = original_sync

    assert [getattr(value, "token_id", value) for value in engine.inputs] == [
        [10, 10],
        11,
        12,
        13,
    ]
    assert next_tokens.token_id == 14


def test_summarize_durations(monkeypatch):
    module = load_benchmark_module(monkeypatch)

    summary = module.summarize_durations([1.0, 2.0, 3.0], token_count=3)

    assert summary["count"] == 3
    assert summary["duration_ms_mean"] == 2.0
    assert summary["duration_ms_p50"] == 2.0
    assert summary["duration_ms_p95"] == 2.9
    assert summary["duration_ms_p99"] == 2.98
