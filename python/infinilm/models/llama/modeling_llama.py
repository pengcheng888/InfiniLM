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

import json
import os
from typing import Optional, Union

from transformers.utils import logging

import infinicore

from ...cache_utils import Cache, DynamicCache
from ...generation.utils import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

LlamaRMSNorm = infinicore.nn.RMSNorm


class LlamaMLP(infinicore.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = infinicore.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = infinicore.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = infinicore.nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
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
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rope_infinicore = infinicore.nn.RoPE(
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            head_dim=getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads,
        )

        self.q_proj = infinicore.nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.k_proj = infinicore.nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.v_proj = infinicore.nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.o_proj = infinicore.nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: infinicore.Tensor,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> infinicore.Tensor:
        hidden_states_shape = hidden_states.shape  # [bs, ntok, hidden_size]
        bs, ntok = hidden_states_shape[:-1]  # [bs, ntok]

        query_hidden_shape = (bs, ntok, self.num_attention_heads, self.head_dim)
        key_hidden_shape = (bs, ntok, self.num_key_value_heads, self.head_dim)
        value_hidden_shape = (bs, ntok, self.num_key_value_heads, self.head_dim)

        # --------------------------------------------------------------------------------------- #
        #                           对 Q,K，V进行 project 加上 rope
        # --------------------------------------------------------------------------------------- #
        # => [bs, ntok,  num_attention_heads, head_dim]
        query_states_infinicore = self.q_proj(hidden_states).view(query_hidden_shape)

        # => [bs, ntok,  num_key_value_heads, head_dim]
        key_states_infinicore = self.k_proj(hidden_states).view(key_hidden_shape)
        # => [bs, ntok, nkvh, head_dim]
        value_states_infinicore = self.v_proj(hidden_states).view(value_hidden_shape)

        # --------------------------------------------------------------------------------------- #
        #                           对 Q和K， 加上 rope
        # --------------------------------------------------------------------------------------- #
        cache_position_infini = kwargs.pop("cache_position_infini", None)
        if not cache_position_infini:
            raise KeyError("cache_position_infini error")

        query_states = self.rope_infinicore(
            query_states_infinicore, cache_position_infini
        )
        key_states = self.rope_infinicore(key_states_infinicore, cache_position_infini)

        # --------------------------------------------------------------------------------------- #
        #                           kv cache
        # --------------------------------------------------------------------------------------- #
        query_states_infini = query_states.permute((0, 2, 1, 3)).contiguous()
        # kv cache
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {}
            key_states_infini, value_states_infini = past_key_values.update(
                key_states,  # [bs, num_key_value_heads, ntok, head_dim]
                value_states_infinicore,  # [bs, num_key_value_heads, ntok, head_dim]
                self.layer_idx,
                cache_kwargs,
            )

        # --------------------------------------------------------------------------------------- #
        #                           注意力计算
        # --------------------------------------------------------------------------------------- #
        # att_val => [bs,  num_attention_heads, ntok, head_dim]
        att_val = infinicore.nn.functional.scaled_dot_product_attention(
            query_states_infini,  # [bs, num_attention_heads, ntok, head_dim]
            key_states_infini,  # [bs, num_key_value_heads, all_tok, head_dim]
            value_states_infini,  # [bs, num_key_value_heads, all_tok, head_dim]
            is_causal=True,
            enable_gqa=True,
        )

        # => [bs, ntok, num_attention_heads, dh ]
        attn_output = att_val.permute((0, 2, 1, 3)).contiguous()

        # --------------------------------------------------------------------------------------- #
        #                           out project
        # --------------------------------------------------------------------------------------- #
        # ([bs, ntok, num_attention_heads, head_dim]) ==> [bs, ntok, hidden_size ]
        attn_output = attn_output.view(hidden_states_shape)

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
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: infinicore.Tensor,  # [bs, ntok, hidden_size]
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> infinicore.Tensor:
        residual = hidden_states
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(infinicore.nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = infinicore.nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        self.layers = infinicore.nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        past_key_values: Optional[Cache] = None,  # StaticCache(layers=[StaticLayer])
        inputs_embeds=None,  # None
        use_cache: Optional[bool] = None,  # True
        **kwargs,  # {}
    ) -> BaseModelOutputWithPast:
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            # input_ids :     {1,5}       tensor([[    1,  1128,   526,   366, 29892]])
            # inputs_embeds : {1,5,2048}  tensor([[[...]]])
            # input_ids = input_ids.to(dtype=int32)

            input_ids_infini = kwargs.pop("input_ids_infini", None)
            inputs_embeds = self.embed_tokens(input_ids_infini)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)  # 1,5,2048
        _, ntoken, _ = hidden_states.shape

        last_hidden_state_last_token = infinicore.narrow(
            hidden_states, 1, ntoken - 1, 1
        )
        return BaseModelOutputWithPast(
            past_key_values=past_key_values,
            last_hidden_state_last_token=last_hidden_state_last_token,
        )


class LlamaForCausalLM(infinicore.nn.Module, GenerationMixin):
    config: LlamaConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = infinicore.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def forward(
        self,
        past_key_values: Optional[Cache] = None,
        inputs_embeds=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        input_ids: Optional[ LongTensor ] = None,  # tensor([[    1,  1128,   526,   366, 29892]])
        cache_position: Optional[ LongTensor ] = None,  # [0,1,2,3,4] ...  [5]   cuda:0
        """
        outputs: BaseModelOutputWithPast = self.model(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        # logits Size([1, 1, 32000])
        logits = self.lm_head(outputs.last_hidden_state_last_token.contiguous())

        return CausalLMOutputWithPast(
            next_token_logits=logits,
            past_key_values=outputs.past_key_values,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, os.PathLike]],
        ignore_mismatched_sizes: bool = False,
        weights_only: bool = True,
        *model_args,
        **kwargs,
    ):
        def load_config_json(dir_path_: str):
            with open(os.path.join(dir_path_, "config.json"), "r") as f:
                config = json.load(f)
            return config

        config_dict = load_config_json(os.path.join(model_path))
        config = LlamaConfig(**config_dict)

        return LlamaForCausalLM(config)


__all__ = [
    "LlamaModel",
    "LlamaForCausalLM",
]
