import json
import sys
import types

from infinilm.processors.deepseek_v4_processor import DeepSeekV4Processor


def test_prefill_metadata_reuses_scheduler_builder(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 256}), encoding="utf-8"
    )
    fake_infinicore = types.SimpleNamespace(
        int32=object(),
        from_list=lambda values, dtype: values,
    )
    monkeypatch.setitem(sys.modules, "infinicore", fake_infinicore)

    metadata = DeepSeekV4Processor.build_prefill_attention_metadata(
        str(tmp_path), [0], 4, 256
    )

    assert len(metadata["dsv4_swa_indices"]) == 4
    assert all(len(row) == 128 for row in metadata["dsv4_swa_indices"])
    assert metadata["dsv4_swa_topk_lengths"] == [1, 2, 3, 4]
    assert metadata["dsv4_raw_out_loc"] == [0, 1, 2, 3]
    assert metadata["dsv4_seq_lens_casual"] == [1, 2, 3, 4]
    assert metadata["dsv4_positions_casual"] == [0, 1, 2, 3]
    assert all(len(row) == 512 for row in metadata["dsv4_c4_indices"])
    assert all(len(row) == 128 for row in metadata["dsv4_c128_indices"])
