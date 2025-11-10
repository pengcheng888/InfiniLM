import torch
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from ..cache_utils import (
    Cache,
    DynamicCache
)


class GenerationMixin:
    def __init__(self):
        pass

    def _get_initial_cache_position(self, seq_length, device, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""

        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            return model_kwargs
        if "inputs_embeds" in model_kwargs and not self.config.is_encoder_decoder:
            cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        elif "decoder_inputs_embeds" in model_kwargs and self.config.is_encoder_decoder:
            cache_position = (
                    torch.ones_like(model_kwargs["decoder_inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            )
        else:
            cache_position = torch.ones(seq_length, dtype=torch.int64, device=device).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            # Support for BC tuple cache format
            if isinstance(cache, tuple):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length"):
                past_length = cache.get_seq_length()

            cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position

        #
        import infinicore
        cache_position_list = list(range(0, seq_length))
        cache_position_infini = infinicore.convert_list_to_infini_tensor(cache_position_list, shape=[seq_length])
        model_kwargs["cache_position_infini"] = cache_position_infini
        #
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ):
        """
        Prepare the model inputs for generation. It includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """
        import infinicore
        # 1. Handle BC:
        model_inputs = {}
        model_inputs["inputs_embeds"] = None
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        if False:

            model_inputs["cache_position"] = cache_position

            # 2. Generic cache-dependent input preparation
            if past_key_values is not None:
                if input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                    input_ids = input_ids[:, cache_position]
                if input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                    input_ids = input_ids[:, cache_position]

            # 3. Prepare base model inputs
            model_inputs["input_ids"] = input_ids.clone(memory_format=torch.contiguous_format)  # [[1234]]

        # --------------------------------------------------- #
        if True:
            model_inputs["cache_position_infini"] = kwargs.get("cache_position_infini", None)
            if past_key_values is not None:
                if kwargs.get("next_token", None) is not None:
                    next_token = kwargs.get("next_token", None)
                    input_ids_infini = infinicore.convert_list_to_infini_tensor([next_token], shape=[1, 1])
                    model_inputs["input_ids_infini"] = input_ids_infini

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        return model_inputs

    def _sample(
            self,
            input_ids: torch.LongTensor,
            **model_kwargs,
    ) -> Union[torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.

 
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).

            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        """

        do_sample = False

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__

        max_new_tokens = 10
        cur_count = 0

        import infinicore
        output_tokens_list = []
        while (cur_count < max_new_tokens) and (not this_peer_finished):

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)  # input_ids: Tensor [[1,1128,...]]

            # outputs = model_forward(**model_inputs, return_dict=True)
            outputs = self(**model_inputs, return_dict=True)  # => CausalLMOutputWithPast

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            if model_kwargs.get("use_cache", True):
                origin_cache_position = model_kwargs["cache_position"]
                cache_position = model_kwargs["cache_position"][-1:] + 1
                model_kwargs["cache_position"] = cache_position

                # TODO
                cache_position_infini = infinicore.convert_torch_to_infini_tensor(origin_cache_position)
                last_pos_value = infinicore.get_index_value(cache_position_infini, [-1])
                cache_position_infini = infinicore.convert_list_to_infini_tensor([last_pos_value + 1])
                model_kwargs["cache_position_infini"] = cache_position_infini

            else:
                raise KeyError("cache_position error")

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            if outputs.next_token_logits is not None:
                next_token_logits = outputs.next_token_logits
            else:
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # ----------------------------------------------------------------- #
            #                   pre-process distribution 
            # ----------------------------------------------------------------- #
            # logits_processor
            next_token_scores = next_token_logits  # cuda:0

            # ----------------------------------------------------------------- #
            #                        token selection
            # ----------------------------------------------------------------- #
            if isinstance(next_token_scores, torch.Tensor):
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

            elif isinstance(next_token_scores, infinicore.Tensor):
                next_tokens = torch.argmax(infinicore.convert_infini_to_torch_tensor(next_token_scores), dim=-1)
                next_tokens = infinicore.convert_torch_to_infini_tensor(next_tokens)  # shape: [1,1]

            # ----------------------------------------------------------------- #
            #                        收集结果
            # ----------------------------------------------------------------- #
            # update generated ids, model inputs, and length for next step
            if isinstance(next_tokens, torch.Tensor):
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                unfinished_sequences = unfinished_sequences
                this_peer_finished = unfinished_sequences.max() == 0  # flag表示全是0了，表示所有batch都收集完毕了

                #
                next_token = next_tokens[0].cpu().item()  # 将 torch.Tensor 转为 python的int类型
                model_kwargs["next_token"] = next_token


            elif isinstance(next_tokens, infinicore.Tensor):
                next_tokens = infinicore.convert_infini_to_torch_tensor(next_tokens).cpu()
                next_token = next_tokens[0, 0].item()  # 将 torch.Tensor 转为 python的int类型
                output_tokens_list.append(next_token)

                #
                model_kwargs["next_token"] = next_token

            cur_len += 1
            cur_count += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        return input_ids, output_tokens_list

    def generate_wpc(self,
                     **kwargs,
                     ):

        kwargs.pop("attention_mask")

        model_kwargs = kwargs
        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        inputs_tensor = kwargs.pop("input_ids", None)
        input_ids = inputs_tensor

        import infinicore
        model_kwargs["input_ids_infini"] = infinicore.convert_torch_to_infini_tensor(inputs_tensor)

        dynamic_cache_kwargs = {"config": self.config}
        model_kwargs["past_key_values"] = (
            DynamicCache(**dynamic_cache_kwargs)
        )

        model_kwargs["use_cache"] = True

        result = self._sample(input_ids,
                              **model_kwargs)
        return result
