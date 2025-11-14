from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import infinicore

from ..cache_utils import Cache, DynamicCache


class GenerationMixin:
    def _get_initial_cache_position(self, seq_length, device, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        cache_position_list = list(range(0, seq_length))

        cache_position_infini = infinicore.convert_list_to_infini_tensor(
            cache_position_list,
            shape=[seq_length],
            infini_device=infinicore.device(device.type, device.index),
        )
        model_kwargs["cache_position_infini"] = cache_position_infini

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        past_key_values: Optional[Cache] = None,
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
        # -------------------------------------------------------------------- #
        #                 所需的: KV Cache
        # -------------------------------------------------------------------- #
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        # -------------------------------------------------------------------- #
        #                 所需的: cache_position
        # -------------------------------------------------------------------- #
        model_inputs["cache_position_infini"] = kwargs.get(
            "cache_position_infini", None
        )

        # -------------------------------------------------------------------- #
        #                 所需的: token的input_ids
        # -------------------------------------------------------------------- #
        if kwargs.get("next_token", None) is not None:
            next_token = kwargs.get("next_token", None)
            input_ids_infini = infinicore.convert_list_to_infini_tensor(
                [next_token], shape=[1, 1]
            )
            model_inputs["input_ids_infini"] = input_ids_infini

        # -------------------------------------------------------------------- #
        #                 所需的: 其他
        # -------------------------------------------------------------------- #
        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs

    def generate(self, **kwargs):
        kwargs.pop("attention_mask")
        max_new_tokens = kwargs.pop("max_new_tokens", 10)

        inputs_tensor = kwargs.pop("input_ids", None)
        model_kwargs = kwargs
        model_kwargs["use_cache"] = True

        # -------------------------------------------------------------------- #
        #                  输入的 token_ids 转为 cpu                             #
        # -------------------------------------------------------------------- #
        input_ids = inputs_tensor
        model_kwargs["input_ids_infini"] = inputs_tensor.cpu().to_infini()

        # -------------------------------------------------------------------- #
        #                       创建 cache                                      #
        # -------------------------------------------------------------------- #
        model_kwargs["past_key_values"] = DynamicCache(config=self.config)

        # -------------------------------------------------------------------- #
        #                       _sample函数                                     #
        # -------------------------------------------------------------------- #
        result = self._sample(input_ids, max_new_tokens=max_new_tokens, **model_kwargs)
        return result

    def _sample(
        self,
        input_ids,
        max_new_tokens=10,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.

            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        """

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        device = input_ids.device

        # -------------------------------------------------------------------------- #
        #                     初始化 cache_position
        # -------------------------------------------------------------------------- #
        model_kwargs = self._get_initial_cache_position(cur_len, device, model_kwargs)

        # model_forward = self.__call__
        cur_count = 0
        output_tokens_list = []

        # -------------------------------------------------------------------------- #
        #                     循环生成，结束条件是字符数量
        # -------------------------------------------------------------------------- #
        while cur_count < max_new_tokens:
            # -------------------------------------------------------------------------- #
            #                     prepare model inputs
            # -------------------------------------------------------------------------- #
            model_inputs = self.prepare_inputs_for_generation(
                **model_kwargs
            )  # input_ids: Tensor [[1,1128,...]]

            # -------------------------------------------------------------------------- #
            #                     计算一次
            # -------------------------------------------------------------------------- #
            # => CausalLMOutputWithPast
            outputs = self.forward(**model_inputs, return_dict=True)

            # -------------------------------------------------------------------------- #
            #                     更新下一次所需的，cache_position
            # -------------------------------------------------------------------------- #
            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            if model_kwargs.get("use_cache", True):
                cache_position_infini = model_kwargs["cache_position_infini"]
                last_pos_value = infinicore.get_index_value(cache_position_infini, [-1])
                model_kwargs["cache_position_infini"] = (
                    infinicore.convert_list_to_infini_tensor(
                        [last_pos_value + 1],
                        infini_device=infinicore.device(device.type, device.index),
                    )
                )
            else:
                raise KeyError("cache_position error")

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)

            # -------------------------------------------------------------------------- #
            #                     处理 推理的输出
            # -------------------------------------------------------------------------- #
            next_token_logits = outputs.next_token_logits

            # pre-process distribution： logits_processor
            next_token_scores = next_token_logits  # cuda:0

            # random_sample : token selection
            if isinstance(next_token_scores, infinicore.Tensor):
                import torch

                if False:
                    next_tokens = infinicore.nn.functional.random_sample(
                        next_token_scores.view([32000]),
                        1.0,
                        1.0,
                        1,
                        1.0,
                    )
                    next_tokens = next_tokens.view([1, 1, 1])

                if True:
                    next_tokens = torch.argmax(
                        next_token_scores.to_torch(),
                        dim=-1,
                    )
                    next_tokens = next_tokens.to_infini()  # shape: [1,1]

            # ----------------------------------------------------------------- #
            #                        收集结果
            # ----------------------------------------------------------------- #
            # update generated ids, model inputs, and length for next step
            next_tokens = next_tokens.to_torch().cpu()

            # 将 torch.Tensor 中的数据 转为 python的int类型
            next_token = next_tokens[0, 0].item()
            output_tokens_list.append(next_token)
            model_kwargs["next_token"] = next_token

            cur_len += 1
            cur_count += 1

            del outputs

        return input_ids, output_tokens_list
