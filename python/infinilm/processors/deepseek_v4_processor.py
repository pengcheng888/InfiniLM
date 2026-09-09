import json
import os

from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("deepseek_v4")
class DeepSeekV4Processor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        with open(
            os.path.join(model_dir_path, "config.json"), encoding="utf-8"
        ) as config_file:
            config = json.load(config_file)
        self.swa_topk = int(config.get("sliding_window", 0))
        if self.swa_topk <= 0:
            raise ValueError("DeepSeek-V4 sliding_window must be positive")

        super().__init__(model_dir_path)
        self._fix_missing_chat_template()

    def _fix_missing_chat_template(self):
        if getattr(self.tokenizer, "chat_template", None):
            return

        user_token = self.tokenizer.convert_ids_to_tokens(128803)
        assistant_token = self.tokenizer.convert_ids_to_tokens(128804)
        self.tokenizer.chat_template = (
            "{{ bos_token }}"
            "{%- for message in messages -%}"
            "{%- if message['role'] == 'system' -%}"
            "{{ message['content'] }}"
            "{%- elif message['role'] == 'user' -%}"
            f"{user_token}{{{{ message['content'] }}}}"
            "{%- if loop.last and add_generation_prompt -%}"
            f"{assistant_token}</think>"
            "{%- endif -%}"
            "{%- elif message['role'] == 'assistant' -%}"
            "{{ message['content'] }}{{ eos_token }}"
            "{%- endif -%}"
            "{%- endfor -%}"
        )

    @staticmethod
    def _slot_for_position(block_table, position: int, block_size: int):
        if position < 0:
            return -1
        block_idx = position // block_size
        block_offset = position % block_size
        if block_idx >= len(block_table):
            return -1
        block_id = block_table[block_idx]
        return block_id * block_size + block_offset if block_id >= 0 else -1

    @staticmethod
    def _full_slot_to_swa_slot(slot: int, full_to_swa_block_ids, block_size: int):
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
    def _swa_indices_for_position(
        cls,
        block_table,
        position: int,
        window_size: int,
        block_size: int,
        full_to_swa_block_ids=None,
    ):
        indices = []
        for offset in range(window_size):
            full_slot = cls._slot_for_position(block_table, position - offset, block_size)
            indices.append(cls._full_slot_to_swa_slot(full_slot, full_to_swa_block_ids, block_size))
        return indices

    def _build_model_input_from_batch_scheduler_output(
        self, scheduler_output, temperature, top_p, top_k
    ) -> dict:
        import infinicore

        result = super()._build_model_input_from_batch_scheduler_output(
            scheduler_output, temperature, top_p, top_k
        )

        window_size = self.swa_topk
        block_size = getattr(scheduler_output, "dsv4_swa_block_size", 256) or 256
        full_to_swa_block_ids = getattr(
            scheduler_output, "dsv4_full_to_swa_block_ids", None
        )

        query_rows = []
        for req in scheduler_output.scheduled_requests:
            num_cached = req.num_local_cached_tokens
            if scheduler_output.is_prefill:
                compute_len = len(req.get_input_tokens()) - num_cached
                positions = range(num_cached, num_cached + compute_len)
            else:
                positions = [req.get_total_length() - 1]
            for idx, position in enumerate(positions):
                slot = (
                    req.slot_mapping[idx]
                    if idx < len(req.slot_mapping)
                    else self._slot_for_position(req.block_table, position, block_size)
                )
                query_rows.append((req.block_table, position, slot))

        swa_indices = []
        swa_topk_lengths = []
        raw_out_loc = []
        for block_table, position, slot in query_rows:
            swa_indices.append(
                self._swa_indices_for_position(
                    block_table,
                    position,
                    window_size,
                    block_size,
                    full_to_swa_block_ids,
                )
            )
            swa_topk_lengths.append(min(position + 1, window_size))
            raw_out_loc.append(
                self._full_slot_to_swa_slot(slot, full_to_swa_block_ids, block_size)
            )

        result.update(
            {
                "swa_indices": infinicore.from_list(
                    swa_indices, dtype=infinicore.int32
                ),
                "swa_topk_lengths": infinicore.from_list(
                    swa_topk_lengths, dtype=infinicore.int32
                ),
                "raw_out_loc": infinicore.from_list(
                    raw_out_loc, dtype=infinicore.int32
                ),
            }
        )
        return result
