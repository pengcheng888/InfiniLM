from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor
from ..llm.scheduler import SchedulerOutput
from typing_extensions import override


@register_processor("glm4_moe_lite")
class Glm4MoeLiteProcessor(BasicLLMProcessor):
    @override
    def _build_model_input_from_batch_scheduler_output(
        self, scheduler_output: SchedulerOutput, temperature, top_p, top_k
    ) -> dict:
        import infinicore

        if not scheduler_output.scheduled_requests:
            raise RuntimeError(
                "build_model_inputs called with empty scheduled_requests"
            )

        tokens = []
        seq_lens = []
        seq_offsets = [0]
        block_tables = []
        slot_mapping = []
        cached_lens = []
        position_ids = []
        cu_seqlens = [0]

        max_block_table_len = max(
            len(req.block_table) for req in scheduler_output.scheduled_requests
        )
        current_offset = 0

        for req in scheduler_output.scheduled_requests:
            num_cached = req.num_local_cached_tokens
            if scheduler_output.is_prefill:
                req_tokens = req.get_input_tokens()
                tokens_to_compute = req_tokens[num_cached:]
                tokens.extend(tokens_to_compute)

                compute_len = len(tokens_to_compute)
                seq_len = len(req_tokens)
                current_offset += compute_len
                position_ids.extend(range(num_cached, num_cached + compute_len))
            else:
                seq_len = req.get_total_length()
                last_token = (
                    req.generated_token_ids[-1]
                    if req.generated_token_ids
                    else req.prompt_token_ids[-1]
                )
                tokens.append(last_token)

                current_offset += 1
                position_ids.append(seq_len - 1)

            seq_lens.append(seq_len)
            seq_offsets.append(current_offset)
            slot_mapping.extend(req.slot_mapping)
            cached_lens.append(num_cached)
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

            padded_block_table = req.block_table + [-1] * (
                max_block_table_len - len(req.block_table)
            )
            block_tables.append(padded_block_table)

        return {
            "input_ids": infinicore.from_list([tokens], dtype=infinicore.int64),
            "position_ids": infinicore.from_list(
                position_ids, dtype=self._position_ids_dtype()
            ),
            "past_kv_lengths": infinicore.from_list(
                cached_lens, dtype=infinicore.int32
            ),
            "total_kv_lengths": infinicore.from_list(seq_lens, dtype=infinicore.int32),
            "input_offsets": infinicore.from_list(seq_offsets, dtype=infinicore.int32),
            "cu_seqlens": infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
            "block_tables": infinicore.from_list(block_tables, dtype=infinicore.int32),
            "slot_mapping": infinicore.from_list(slot_mapping, dtype=infinicore.int64),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
