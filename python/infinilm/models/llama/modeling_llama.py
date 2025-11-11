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
import time
t_all = 0.0

from ...cache_utils import Cache, DynamicCache
import torch

from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_llama import LlamaConfig

from ...generation.utils import GenerationMixin

from transformers.utils import logging

logger = logging.get_logger(__name__)

import infinicore



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
        down_proj = self.down_proj.forward(self.act_fn(self.gate_proj.forward(x)) * self.up_proj.forward(x))
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

        query_states_infinicore = self.q_proj.forward(hidden_states).view(query_hidden_shape)  # => [bs, ntok,  num_attention_heads, head_dim]
        key_states_infinicore = self.k_proj.forward(hidden_states).view(key_hidden_shape)  # => [bs, ntok,  num_key_value_heads, head_dim]
        value_states_infinicore = self.v_proj.forward(hidden_states).view(value_hidden_shape)  # => [bs, ntok, nkvh, head_dim]

        cache_position_infini = kwargs.pop("cache_position_infini", None)
        if cache_position_infini:
            query_states = self.rope_infinicore.forward(query_states_infinicore, cache_position_infini)
            key_states = self.rope_infinicore.forward(key_states_infinicore, cache_position_infini)
        else:
            raise KeyError("cache_position_infini errot")
            exit(-1)

        t1 = time.time()
 
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

        t2 = time.time()

        # global t_all
        # t_all+= (t2-t1)*1000
        # print("t_all: ",t_all)

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
        attn_output = self.o_proj.forward(attn_output)

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
        hidden_states = self.input_layernorm.forward(hidden_states)

        hidden_states = self.self_attn.forward(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
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

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            # input_ids :     {1,5}       tensor([[    1,  1128,   526,   366, 29892]])
            # inputs_embeds : {1,5,2048}  tensor([[[...]]])
            # input_ids = input_ids.to(dtype=torch.int32)

            input_ids_infini = kwargs.pop("input_ids_infini", None)
            inputs_embeds = self.embed_tokens.forward(input_ids_infini)

        hidden_states = inputs_embeds
        ilayer = 0
        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            print("ilayer: ", ilayer)
            ilayer += 1

            hidden_states = decoder_layer.forward(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm.forward(hidden_states)
        hidden_states = infinicore.convert_infini_to_torch_tensor(hidden_states)

        return BaseModelOutputWithPast(
                                       past_key_values=past_key_values,
                                       last_hidden_state_last_token=infinicore.convert_torch_to_infini_tensor(hidden_states[:, [-1], :]),
                                       )


class LlamaPreTrainedModel(infinicore.nn.Module):
    config: LlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):  # torch.nn.Module LlamaPreTrainedModel,
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = infinicore.nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,  # [[13274]]  cuda:0
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,  # [0,1,2,3,4] ...  [5]   cuda:0
            **kwargs,
    ) -> CausalLMOutputWithPast:

    

        outputs: BaseModelOutputWithPast = self.model.forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        logits = self.lm_head.forward(outputs.last_hidden_state_last_token)  # logits torch.Size([1, 1, 32000])
        return CausalLMOutputWithPast(
            logits=infinicore.convert_infini_to_torch_tensor(logits),
            next_token_logits=logits,
            past_key_values=outputs.past_key_values
        )
     

__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
]
