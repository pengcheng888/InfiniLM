import torch
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from ..cache_utils import (
    Cache,
    DynamicCache
)
import infinicore

class GenerationMixin:
    def __init__(self):
        pass

    def _get_initial_cache_position(self, seq_length, device, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        cache_position_list = list(range(0, seq_length))

        cache_position_infini = infinicore.convert_list_to_infini_tensor(cache_position_list,
                                                                        shape=[seq_length],
                                                                        infini_device=infinicore.device(device.type, device.index))
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
        model_inputs["inputs_embeds"] = None


        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values


        model_inputs["cache_position_infini"] = kwargs.get("cache_position_infini", None)
        if kwargs.get("next_token", None) is not None:
            next_token = kwargs.get("next_token", None)
            input_ids_infini = infinicore.convert_list_to_infini_tensor([next_token], 
                                                                        shape=[1, 1]  )
            model_inputs["input_ids_infini"] = input_ids_infini

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        
        torch.cuda.synchronize()

   
        return model_inputs
    
    def generate_wpc(self,
                     **kwargs):


        kwargs.pop("attention_mask")
        
        model_kwargs = kwargs
        model_kwargs["use_cache"] = True

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        inputs_tensor = kwargs.pop("input_ids", None)
        input_ids = inputs_tensor

        import infinicore
        model_kwargs["input_ids_infini"] = infinicore.convert_torch_to_infini_tensor(inputs_tensor.cpu())
        model_kwargs["past_key_values"] = DynamicCache(config= self.config)

        result = self._sample(input_ids,
                              **model_kwargs)
        return result

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

            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        """


        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        device = input_ids.device
        
        model_kwargs = self._get_initial_cache_position(cur_len,device, model_kwargs)

        # model_forward = self.__call__

        max_new_tokens = 10
        cur_count = 0
        output_tokens_list = []
        while (cur_count < max_new_tokens) :
          
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(**model_kwargs)  # input_ids: Tensor [[1,1128,...]]
       
            outputs = self.forward(**model_inputs, return_dict=True)  # => CausalLMOutputWithPast
      
            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            if model_kwargs.get("use_cache", True):
                cache_position_infini = model_kwargs["cache_position_infini"]
                
                last_pos_value = infinicore.get_index_value(cache_position_infini, [-1])
                model_kwargs["cache_position_infini"] =  infinicore.convert_list_to_infini_tensor([last_pos_value + 1],
                                                                                 infini_device=infinicore.device(device.type, device.index))
            else:
                raise KeyError("cache_position error")

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
        
            next_token_logits = outputs.next_token_logits
   
            # ----------------------------------------------------------------- #
            #                   pre-process distribution 
            # ----------------------------------------------------------------- #
            # logits_processor
            next_token_scores = next_token_logits  # cuda:0

            # ----------------------------------------------------------------- #
            #                        token selection
            # ----------------------------------------------------------------- #
            if isinstance(next_token_scores, infinicore.Tensor):
                next_tokens = torch.argmax(infinicore.convert_infini_to_torch_tensor(next_token_scores), dim=-1)
                next_tokens = infinicore.convert_torch_to_infini_tensor(next_tokens)  # shape: [1,1]

            # ----------------------------------------------------------------- #
            #                        收集结果
            # ----------------------------------------------------------------- #
            # update generated ids, model inputs, and length for next step
            if isinstance(next_tokens, infinicore.Tensor):
                next_tokens = infinicore.convert_infini_to_torch_tensor(next_tokens).cpu()
                next_token = next_tokens[0, 0].item()  # 将 torch.Tensor 转为 python的int类型
                output_tokens_list.append(next_token)

                #
                model_kwargs["next_token"] = next_token

            cur_len += 1
            cur_count += 1

            del outputs

        return input_ids, output_tokens_list

