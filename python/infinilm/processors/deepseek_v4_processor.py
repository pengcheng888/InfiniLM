import json
import os
import time

from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("deepseek_v4")
class DeepSeekV4Processor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path)
        self._dsv4_c128_metadata_width = self._load_c128_metadata_width(model_dir_path)
        self._fix_missing_chat_template()

    @classmethod
    def _load_c128_metadata_width(cls, model_dir_path: str) -> int:
        config_path = os.path.join(model_dir_path, "config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except OSError:
            return 64

        max_position_embeddings = int(config.get("max_position_embeddings") or 0)
        if max_position_embeddings <= 0:
            return 64

        # SGLang materializes C128 page indices from the full page table, not
        # from the currently visible compressed length. Its offline DSv4 path
        # carries one extra 64-slot guard tile beyond the HF max position.
        c128_slots = (max_position_embeddings + 127) // 128 + 64
        return cls._dsv4_align(c128_slots, 64)

    def _fix_missing_chat_template(self):
        if getattr(self.tokenizer, "chat_template", None):
            return

        user_token = self.tokenizer.convert_ids_to_tokens(128803)
        assistant_token = self.tokenizer.convert_ids_to_tokens(128804)
        thinking_end_token = "</think>"

        self.tokenizer.chat_template = (
            "{{ bos_token }}"
            "{%- for message in messages -%}"
            "{%- if message['role'] == 'system' -%}"
            "{{ message['content'] }}"
            "{%- elif message['role'] == 'developer' -%}"
            f"{user_token}{{{{ message['content'] }}}}"
            "{%- if loop.last and add_generation_prompt -%}"
            f"{assistant_token}{thinking_end_token}"
            "{%- endif -%}"
            "{%- elif message['role'] == 'user' -%}"
            f"{user_token}{{{{ message['content'] }}}}"
            "{%- if (not loop.last and messages[loop.index0 + 1]['role'] == 'assistant') or (loop.last and add_generation_prompt) -%}"
            f"{assistant_token}{thinking_end_token}"
            "{%- endif -%}"
            "{%- elif message['role'] == 'assistant' -%}"
            "{{ message['content'] }}{{ eos_token }}"
            "{%- endif -%}"
            "{%- endfor -%}"
        )

    def _build_model_input_from_batch_scheduler_output(
        self, scheduler_output, temperature, top_p, top_k
    ) -> dict:
        profile = os.getenv("INFINILM_DSV4_PROCESSOR_PROFILE", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        start = time.perf_counter()
        result = super()._build_model_input_from_batch_scheduler_output(
            scheduler_output, temperature, top_p, top_k
        )
        base_ms = (time.perf_counter() - start) * 1000.0
        meta_start = time.perf_counter()
        metadata = self._build_dsv4_attention_metadata(scheduler_output)
        meta_ms = (time.perf_counter() - meta_start) * 1000.0
        result.update(metadata)
        if profile:
            phase = "prefill" if scheduler_output.is_prefill else "decode"
            print(
                f"[INFINILM_DSV4_PROCESSOR_PROFILE] phase={phase} "
                f"requests={len(scheduler_output.scheduled_requests)} "
                f"base_ms={base_ms:.3f} dsv4_meta_ms={meta_ms:.3f} "
                f"total_ms={(time.perf_counter() - start) * 1000.0:.3f}",
                flush=True,
            )
        return result

    @staticmethod
    def _dsv4_slot_for_position(block_table, position: int, block_size: int = 256):
        if position < 0:
            return -1
        block_idx = position // block_size
        block_offset = position % block_size
        if block_idx >= len(block_table):
            return -1
        block_id = block_table[block_idx]
        return block_id * block_size + block_offset if block_id >= 0 else -1

    @staticmethod
    def _dsv4_full_slot_to_swa_slot(slot: int, full_to_swa_block_ids, block_size: int = 256):
        if slot < 0:
            return -1
        if full_to_swa_block_ids is None:
            return slot
        full_block_id = slot // block_size
        block_offset = slot % block_size
        if full_block_id >= len(full_to_swa_block_ids):
            return -1
        swa_block_id = full_to_swa_block_ids[full_block_id]
        return swa_block_id * block_size + block_offset if swa_block_id >= 0 else -1

    @classmethod
    def _dsv4_swa_indices_for_position(
        cls,
        block_table,
        position: int,
        window_size: int = 128,
        block_size: int = 256,
        full_to_swa_block_ids=None,
    ):
        indices = []
        for offset in range(window_size):
            full_slot = cls._dsv4_slot_for_position(block_table, position - offset, block_size)
            indices.append(cls._dsv4_full_slot_to_swa_slot(full_slot, full_to_swa_block_ids, block_size))
        return indices

    @classmethod
    def _dsv4_compressed_indices_for_position(cls, block_table, position: int, ratio: int, width: int, block_size: int = 256):
        visible = (position + 1) // ratio
        indices = []
        for compressed_idx in range(width):
            if compressed_idx >= visible:
                indices.append(-1)
                continue
            raw_position = (compressed_idx + 1) * ratio - 1
            raw_slot = cls._dsv4_slot_for_position(block_table, raw_position, block_size)
            indices.append(raw_slot // ratio if raw_slot >= 0 else -1)
        clamp1 = max(visible, 1)
        return indices, visible, clamp1

    @classmethod
    def _dsv4_compress_locations_for_position(cls, block_table, slot: int, position: int, ratio: int, block_size: int = 256):
        ring_size = 8 if ratio == 4 else 128

        def state_loc(raw_slot: int) -> int:
            if raw_slot < 0:
                return -1
            return ((raw_slot // block_size) * ring_size + (raw_slot % ring_size)) // ratio

        state_index = state_loc(slot)
        out_loc = slot // ratio if slot >= 0 and (position + 1) % ratio == 0 else -1
        clipped_position = (position // ratio) * ratio
        clipped_slot = cls._dsv4_slot_for_position(block_table, clipped_position, block_size)
        write_loc = state_loc(clipped_slot)
        if write_loc < 0:
            write_loc = state_index
        prev_position = clipped_position - ratio
        prev_slot = cls._dsv4_slot_for_position(block_table, prev_position, block_size)
        prev_state_index = state_loc(prev_slot)
        if prev_state_index < 0:
            prev_state_index = write_loc if write_loc >= 0 else 0
        return out_loc, clipped_position, write_loc, prev_state_index, state_index

    @staticmethod
    def _dsv4_align(value: int, multiple: int):
        return ((value + multiple - 1) // multiple) * multiple

    @classmethod
    def build_prefill_attention_metadata(
        cls,
        model_dir_path: str,
        block_table,
        sequence_length: int,
        block_size: int = 256,
    ) -> dict:
        """Build the same DSv4 prefill metadata used by the batch scheduler."""
        from types import SimpleNamespace

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        block_table = list(block_table)
        if not block_table:
            raise ValueError("block_table must not be empty")

        processor = cls.__new__(cls)
        processor._dsv4_c128_metadata_width = cls._load_c128_metadata_width(
            model_dir_path
        )
        request = SimpleNamespace(
            block_table=block_table,
            num_local_cached_tokens=0,
            slot_mapping=[
                cls._dsv4_slot_for_position(block_table, position, block_size)
                for position in range(sequence_length)
            ],
            get_input_tokens=lambda: [0] * sequence_length,
        )
        scheduler_output = SimpleNamespace(
            scheduled_requests=[request],
            is_prefill=True,
            dsv4_full_to_swa_block_ids=None,
            dsv4_swa_block_size=block_size,
        )
        return processor._build_dsv4_attention_metadata(scheduler_output)

    def _build_dsv4_attention_metadata(self, scheduler_output) -> dict:
        import infinicore

        start = time.perf_counter()
        query_rows = []
        max_block_table_len = 1
        max_c128_visible = 1
        full_to_swa_block_ids = getattr(scheduler_output, "dsv4_full_to_swa_block_ids", None)
        block_size = getattr(scheduler_output, "dsv4_swa_block_size", 256) or 256
        for req in scheduler_output.scheduled_requests:
            max_block_table_len = max(max_block_table_len, len(req.block_table))
            num_cached = req.num_local_cached_tokens
            if scheduler_output.is_prefill:
                compute_len = len(req.get_input_tokens()) - num_cached
                positions = range(num_cached, num_cached + compute_len)
            else:
                positions = [req.get_total_length() - 1]
            slots = list(req.slot_mapping)
            if len(slots) != len(list(positions)):
                positions = list(positions)
            else:
                positions = list(positions)
            for idx, position in enumerate(positions):
                slot = slots[idx] if idx < len(slots) else self._dsv4_slot_for_position(req.block_table, position)
                max_c128_visible = max(max_c128_visible, (position + 1) // 128)
                query_rows.append((req.block_table, position, slot))

        c128_width = max(self._dsv4_c128_metadata_width, self._dsv4_align(max(max_c128_visible, 1), 64))
        c4_sparse_topk = 512

        swa_indices = []
        swa_topk_lengths = []
        raw_out_loc = []
        page_table = []
        seq_lens_casual = []
        positions_casual = []

        c4_indices = []
        c4_topk_lengths = []
        c4_topk_lengths_raw = []
        c4_topk_lengths_clamp1 = []
        c4_sparse_topk_lengths = []
        c4_out_loc = []
        c4_positions = []
        c4_compress_write_loc = []
        c4_compress_extra_loc = []
        c4_compress_state_indices = []

        c128_indices = []
        c128_topk_lengths = []
        c128_topk_lengths_clamp1 = []
        c128_out_loc = []
        c128_positions = []
        c128_compress_write_loc = []
        c128_compress_state_indices = []

        for block_table, position, slot in query_rows:
            padded_page_table = block_table + [-1] * (max_block_table_len - len(block_table))
            page_table.append(padded_page_table)
            raw_out_loc.append(slot)
            seq_lens_casual.append(position + 1)
            positions_casual.append(position)

            swa_indices.append(
                self._dsv4_swa_indices_for_position(
                    block_table,
                    position,
                    block_size=block_size,
                    full_to_swa_block_ids=full_to_swa_block_ids,
                )
            )
            swa_topk_lengths.append(min(position + 1, 128))
            raw_out_loc[-1] = self._dsv4_full_slot_to_swa_slot(raw_out_loc[-1], full_to_swa_block_ids, block_size)

            c4_row, c4_raw_len, c4_clamp1 = self._dsv4_compressed_indices_for_position(
                block_table, position, 4, c4_sparse_topk
            )
            c4_indices.append(c4_row)
            c4_topk_lengths_raw.append(c4_raw_len)
            c4_topk_lengths_clamp1.append(c4_clamp1)
            c4_sparse_topk_lengths.append(min(c4_clamp1, c4_sparse_topk))
            c4_topk_lengths.append(min(c4_clamp1, c4_sparse_topk))
            c4_loc, c4_pos, c4_write, c4_prev, c4_state = self._dsv4_compress_locations_for_position(
                block_table, slot, position, 4
            )
            c4_out_loc.append(c4_loc)
            c4_positions.append(c4_pos)
            c4_compress_write_loc.append(c4_write)
            c4_compress_extra_loc.append([c4_prev])
            c4_compress_state_indices.append(c4_state)

            c128_row, c128_raw_len, c128_clamp1 = self._dsv4_compressed_indices_for_position(
                block_table, position, 128, c128_width
            )
            c128_indices.append(c128_row)
            c128_topk_lengths_clamp1.append(c128_clamp1)
            c128_topk_lengths.append(c128_clamp1)
            c128_loc, c128_pos, c128_write, _c128_prev, c128_state = self._dsv4_compress_locations_for_position(
                block_table, slot, position, 128
            )
            c128_out_loc.append(c128_loc)
            c128_positions.append(c128_pos)
            c128_compress_write_loc.append(c128_write)
            c128_compress_state_indices.append(c128_state)

        list_done = time.perf_counter()
        result = {
            "dsv4_swa_indices": infinicore.from_list(swa_indices, dtype=infinicore.int32),
            "dsv4_swa_topk_lengths": infinicore.from_list(swa_topk_lengths, dtype=infinicore.int32),
            "dsv4_c4_indices": infinicore.from_list(c4_indices, dtype=infinicore.int32),
            "dsv4_c4_topk_lengths": infinicore.from_list(c4_topk_lengths, dtype=infinicore.int32),
            "dsv4_c128_indices": infinicore.from_list(c128_indices, dtype=infinicore.int32),
            "dsv4_c128_topk_lengths": infinicore.from_list(c128_topk_lengths, dtype=infinicore.int32),
            "dsv4_raw_out_loc": infinicore.from_list(raw_out_loc, dtype=infinicore.int32),
            "dsv4_page_table": infinicore.from_list(page_table, dtype=infinicore.int32),
            "dsv4_seq_lens_casual": infinicore.from_list(seq_lens_casual, dtype=infinicore.int32),
            "dsv4_positions_casual": infinicore.from_list(positions_casual, dtype=infinicore.int32),
            "dsv4_c4_out_loc": infinicore.from_list(c4_out_loc, dtype=infinicore.int32),
            "dsv4_c4_positions": infinicore.from_list(c4_positions, dtype=infinicore.int32),
            "dsv4_c4_topk_lengths_raw": infinicore.from_list(c4_topk_lengths_raw, dtype=infinicore.int32),
            "dsv4_c4_topk_lengths_clamp1": infinicore.from_list(c4_topk_lengths_clamp1, dtype=infinicore.int32),
            "dsv4_c4_sparse_indices": infinicore.from_list(c4_indices, dtype=infinicore.int32),
            "dsv4_c4_sparse_topk_lengths": infinicore.from_list(c4_sparse_topk_lengths, dtype=infinicore.int32),
            "dsv4_c128_out_loc": infinicore.from_list(c128_out_loc, dtype=infinicore.int32),
            "dsv4_c128_positions": infinicore.from_list(c128_positions, dtype=infinicore.int32),
            "dsv4_c128_page_indices": infinicore.from_list(c128_indices, dtype=infinicore.int32),
            "dsv4_c128_topk_lengths_clamp1": infinicore.from_list(c128_topk_lengths_clamp1, dtype=infinicore.int32),
            "dsv4_c4_compress_write_loc": infinicore.from_list(c4_compress_write_loc, dtype=infinicore.int32),
            "dsv4_c4_compress_extra_loc": infinicore.from_list(c4_compress_extra_loc, dtype=infinicore.int32),
            "dsv4_c4_compress_state_indices": infinicore.from_list(c4_compress_state_indices, dtype=infinicore.int32),
            "dsv4_c128_compress_write_loc": infinicore.from_list(c128_compress_write_loc, dtype=infinicore.int32),
            "dsv4_c128_compress_state_indices": infinicore.from_list(c128_compress_state_indices, dtype=infinicore.int32),
        }
        if os.getenv("INFINILM_DSV4_PROCESSOR_PROFILE", "0").lower() in ("1", "true", "yes", "on"):
            print(
                f"[INFINILM_DSV4_METADATA_PROFILE] phase={'prefill' if scheduler_output.is_prefill else 'decode'} "
                f"rows={len(query_rows)} c128_width={c128_width} "
                f"list_ms={(list_done - start) * 1000.0:.3f} "
                f"from_list_ms={(time.perf_counter() - list_done) * 1000.0:.3f}",
                flush=True,
            )
        return result
