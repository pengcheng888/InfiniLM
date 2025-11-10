# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union

from ...cache_utils_wpc import Cache, DynamicCache


from ...modeling_outputs_wpc import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...utils import  logging
from .configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

import infinicore
import torch
from torch import nn

LlamaRMSNorm = infinicore.nn.RMSNorm


class LlamaMLP(infinicore.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = infinicore.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = infinicore.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = infinicore.nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = infinicore.nn.functional.silu

    def forward(self, x: infinicore.Tensor) -> infinicore.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaAttention(infinicore.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rope_infinicore = infinicore.nn.RoPE(config)

        self.q_proj = infinicore.nn.Linear(config.hidden_size,
                                           config.num_attention_heads * self.head_dim,
                                           bias=config.attention_bias)

        self.k_proj = infinicore.nn.Linear(config.hidden_size,
                                           config.num_key_value_heads * self.head_dim,
                                           bias=config.attention_bias)

        self.v_proj = infinicore.nn.Linear(config.hidden_size,
                                           config.num_key_value_heads * self.head_dim,
                                           bias=config.attention_bias)

        self.o_proj = infinicore.nn.Linear(config.num_attention_heads * self.head_dim,
                                           config.hidden_size,
                                           bias=config.attention_bias)

    def forward(self,
                hidden_states: infinicore.Tensor,
                past_key_values: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,
                ) -> infinicore.Tensor:
        hidden_states_shape = hidden_states.shape  # [bs, ntok, hidden_size]
        bs, ntok = hidden_states_shape[:-1]  # [bs, ntok]

        query_hidden_shape = (bs, ntok, self.num_attention_heads, self.head_dim)
        key_hidden_shape = (bs, ntok, self.num_key_value_heads, self.head_dim)
        value_hidden_shape = (bs, ntok, self.num_key_value_heads, self.head_dim)

        query_states_infinicore = self.q_proj(hidden_states).view(query_hidden_shape)  # => [bs, ntok,  num_attention_heads, head_dim]
        key_states_infinicore = self.k_proj(hidden_states).view(key_hidden_shape)  # => [bs, ntok,  num_key_value_heads, head_dim]
        value_states_infinicore = self.v_proj(hidden_states).view(value_hidden_shape)  # => [bs, ntok, nkvh, head_dim]


        cache_position_infini = kwargs.pop("cache_position_infini", None)
        if cache_position_infini:
            query_states = self.rope_infinicore.forward(query_states_infinicore, cache_position_infini)
            key_states = self.rope_infinicore.forward(key_states_infinicore, cache_position_infini)
        else:
            raise KeyError("cache_position_infini errot")
            exit(-1)

        query_states = infinicore.convert_infini_to_torch_tensor(query_states).permute((0, 2, 1, 3)).contiguous()
        key_states = infinicore.convert_infini_to_torch_tensor(key_states).permute((0, 2, 1, 3)).contiguous()
        value_states = infinicore.convert_infini_to_torch_tensor(value_states_infinicore.permute((0, 2, 1, 3)))

        # kv cache
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states,  # [bs, num_key_value_heads, ntok, head_dim]
                                                              value_states,  # [bs, num_key_value_heads, ntok, head_dim]
                                                              self.layer_idx, cache_kwargs)

        query_states_infini = infinicore.convert_torch_to_infini_tensor(query_states.contiguous())
        key_states_infini = infinicore.convert_torch_to_infini_tensor(key_states.contiguous())
        value_states_infini = infinicore.convert_torch_to_infini_tensor(value_states.contiguous())

        # att_val => [bs, ntok, num_attention_heads, head_dim]
        # [4, 40, 64]
        att_val = infinicore.attention_lm(query_states_infini,  # [bs, num_attention_heads, ntok, head_dim]
                                          key_states_infini,  # [bs, num_key_value_heads, all_tok, head_dim]
                                          value_states_infini  # [bs, num_key_value_heads, all_tok, head_dim]
                                          )

        # --
        attn_output = att_val.view((bs, self.num_attention_heads, ntok, self.head_dim)).permute((0, 2, 1, 3)).contiguous()  # => {bs, ntok, num_attention_heads, dh }

        # ([bs, ntok, num_attention_heads, head_dim])
        attn_output = attn_output.view(hidden_states_shape)  # ==> [bs, ntok, hidden_size]

        # o_proj
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(infinicore.nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: infinicore.Tensor,  # [bs, ntok, hidden_size]
                past_key_values: Optional[Cache] = None,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.LongTensor] = None,  # necessary, but kept here for BC
                **kwargs,
                ) -> infinicore.Tensor:
        residual = hidden_states
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(infinicore.nn.Module):  # LlamaPreTrainedModel  torch.nn.Module
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = infinicore.nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = infinicore.nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,  # tensor([[    1,  1128,   526,   366, 29892]]) # tensor([[0, 1, 2, 3, 4]])
            past_key_values: Optional[Cache] = None,  # StaticCache(layers=[StaticLayer])
            inputs_embeds: Optional[torch.FloatTensor] = None,  # None
            cache_position: Optional[torch.LongTensor] = None,  # tensor([0, 1, 2, 3, 4])
            use_cache: Optional[bool] = None,  # True
            **kwargs,  # {}
    ) -> BaseModelOutputWithPast:

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            # input_ids :     {1,5}       tensor([[    1,  1128,   526,   366, 29892]])
            # inputs_embeds : {1,5,2048}  tensor([[[...]]])
            # input_ids = input_ids.to(dtype=torch.int32)

            input_ids_infini = kwargs.pop("input_ids_infini", None)
            if input_ids_infini is None:
                input_ids_infini = infinicore.convert_torch_to_infini_tensor(input_ids.to(device="cpu"))
                inputs_embeds = self.embed_tokens(input_ids_infini)
            elif isinstance(input_ids_infini, infinicore.Tensor):
                inputs_embeds = self.embed_tokens(input_ids_infini)


        hidden_states = inputs_embeds
        ilayer = 0
        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            print("ilayer: ", ilayer)
            ilayer += 1

            hidden_states = decoder_layer(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = infinicore.convert_infini_to_torch_tensor(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values,
                                       last_hidden_state_last_token=infinicore.convert_torch_to_infini_tensor(hidden_states[:, [-1], :]),
                                       )


class LlamaPreTrainedModel(torch.nn.Module):
    config: LlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class GenerationWPC:
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
        cache_position_list = list(range(0,seq_length))
        cache_position_infini = infinicore.convert_list_to_infini_tensor(cache_position_list,shape=[seq_length])
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
            model_inputs["input_ids"] =  input_ids.clone(memory_format=torch.contiguous_format) # [[1234]]
  

        # --------------------------------------------------- #
        if True:
            model_inputs["cache_position_infini"] =  kwargs.get("cache_position_infini", None)
            if past_key_values is not None:
                if kwargs.get("next_token", None) is not None:
                    next_token = kwargs.get("next_token", None) 
                    input_ids_infini = infinicore.convert_list_to_infini_tensor([next_token], shape = [1,1])
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
    ) -> Union[ torch.LongTensor]:
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
        while (cur_count < max_new_tokens) and (not this_peer_finished ):
            
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs) # input_ids: Tensor [[1,1128,...]]
            
            #outputs = model_forward(**model_inputs, return_dict=True)
            outputs = self(**model_inputs, return_dict=True) # => CausalLMOutputWithPast

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            if model_kwargs.get("use_cache", True):
                origin_cache_position = model_kwargs["cache_position"]
                cache_position = model_kwargs["cache_position"][-1:] + 1
                model_kwargs["cache_position"] = cache_position
                
                # TODO
                cache_position_infini = infinicore.convert_torch_to_infini_tensor(origin_cache_position)
                last_pos_value = infinicore.get_index_value(cache_position_infini, [-1])
                cache_position_infini =  infinicore.convert_list_to_infini_tensor([last_pos_value+1])
                model_kwargs["cache_position_infini"] =  cache_position_infini
        
            else:
                raise KeyError("cache_position error")

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            if  outputs.next_token_logits is not None:
                next_token_logits =  outputs.next_token_logits
            else:
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)


            # ----------------------------------------------------------------- #
            #                   pre-process distribution 
            # ----------------------------------------------------------------- #
            # logits_processor
            next_token_scores = next_token_logits # cuda:0


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
                next_tokens = torch.argmax( infinicore.convert_infini_to_torch_tensor (next_token_scores), dim=-1)
                next_tokens = infinicore.convert_torch_to_infini_tensor( next_tokens ) # shape: [1,1]

   
            # ----------------------------------------------------------------- #
            #                        收集结果
            # ----------------------------------------------------------------- #
            # update generated ids, model inputs, and length for next step
            if isinstance(next_tokens, torch.Tensor):
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                unfinished_sequences = unfinished_sequences 
                this_peer_finished = unfinished_sequences.max() == 0 # flag表示全是0了，表示所有batch都收集完毕了

                #
                next_token = next_tokens[0].cpu().item() # 将 torch.Tensor 转为 python的int类型
                model_kwargs["next_token"] =  next_token
  

            elif isinstance(next_tokens, infinicore.Tensor):
                next_tokens = infinicore.convert_infini_to_torch_tensor(next_tokens).cpu()
                next_token = next_tokens[0,0].item() # 将 torch.Tensor 转为 python的int类型
                output_tokens_list.append(next_token)

                #
                model_kwargs["next_token"] =  next_token
            
            cur_len += 1
            cur_count +=1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
        


        return input_ids, output_tokens_list

    def generate_wpc(self,
                    **kwargs,
                    ):
        
        kwargs.pop("attention_mask")
        
        model_kwargs= kwargs
        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        inputs_tensor = kwargs.pop("input_ids",None)
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

class LlamaForCausalLM(LlamaPreTrainedModel,GenerationWPC):  # torch.nn.Module LlamaPreTrainedModel,
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = infinicore.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.LlamaForCausalLM_config = config


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None, # [[13274]]  cuda:0
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None, # [0,1,2,3,4] ...  [5]   cuda:0
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ) -> CausalLMOutputWithPast:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state  # [1,5,2048]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  # [0,None,None]

        if True and (outputs.last_hidden_state_last_token is not None):  # torch.Size([1, 2048])
            logits = self.lm_head(outputs.last_hidden_state_last_token)  # logits torch.Size([1, 1, 32000])
            return CausalLMOutputWithPast(
                logits=infinicore.convert_infini_to_torch_tensor(logits),
                next_token_logits=logits,
                past_key_values=outputs.past_key_values
            )
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values
            )

   


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
]
