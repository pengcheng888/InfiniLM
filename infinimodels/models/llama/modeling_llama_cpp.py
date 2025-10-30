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

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

import infinicore
import torch


class LlamaRMSNorm(infinicore.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

        self.weight_infini = None

    def forward(self, hidden_states: Union[infinicore.Tensor, torch.Tensor]
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(hidden_states, infinicore.Tensor):
            return self.forward_infini2infini(hidden_states)

        return self.forward_torch2torch(hidden_states)

    def forward_torch2torch(self, hidden_states: torch.Tensor):
        # print(" LlamaRMSNorm :: forward_torch2torch ")
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def forward_infini2infini(self, hidden_states: infinicore.Tensor):
        from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor
        print(" LlamaRMSNorm :: forward_infini2infini ")

        if self.weight_infini is None:
            self.weight_infini = torch_tensor_2_infini_tensor(self.weight)

        return infinicore.rms_norm(hidden_states, self.weight_infini, self.variance_epsilon)

    def extra_repr(self):
        return f"infinicore op: {tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def sin_cos_table(self, pos, dim, device, theta, dtype):
        dim = 64
        theta = 10000.0
        pos = torch.Tensor([0, 1, 2, 3, 4])
        assert dim % 2 == 0, "Embedding dimension must be even."
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        angles = torch.outer(pos.cpu(), freqs)

        return torch.sin(angles), torch.cos(angles)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


from enum import Enum, auto


class Algorithm(Enum):
    GPT_J = 0
    GPT_NEOX = 1


def rotary_embedding(ans, t, sin, cos, algo=Algorithm.GPT_NEOX):
    def _torch_rope(sin, cos, t1, t2):
        cos = cos.unsqueeze(1)  # [seq_len, 1, dh // 2]
        sin = sin.unsqueeze(1)  # [seq_len, 1, dh // 2]
        t_out_1 = t1 * cos - t2 * sin
        t_out_2 = t1 * sin + t2 * cos
        return t_out_1, t_out_2

    dh = t.shape[-1]
    dt = t.dtype
    assert dh % 2 == 0, "Embedding dimension must be even."

    if algo == Algorithm.GPT_J:
        t_even = t[..., 0::2]  # [seq_len, n_head, dh // 2]
        t_odd = t[..., 1::2]  # [seq_len, n_head, dh // 2]

        t_out_even, t_out_odd = _torch_rope(sin, cos, t_even, t_odd)

        ans[..., 0::2] = t_out_even.to(dt)
        ans[..., 1::2] = t_out_odd.to(dt)
    else:
        half_dim = dh // 2
        t_first = t[..., :half_dim]
        t_second = t[..., half_dim:]

        t_out_first, t_out_second = _torch_rope(sin[..., 0:half_dim], cos[..., 0:half_dim], t_first, t_second)

        ans[..., :half_dim] = t_out_first.to(dt)
        ans[..., half_dim:] = t_out_second.to(dt)


import infinicore


class RoPE_infinicore_v2():
    sin_table: Union[infinicore.Tensor, None] = None  # sin_table: (max_position_embeddings, head_dim//2)
    cos_table: Union[infinicore.Tensor, None] = None  # cos_table: (max_position_embeddings, head_dim//2)

    def __init__(self, config):
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        if self.get_sin_table() is None:
            sin_table, cos_table = self.create_sin_cos_table(self.max_position_embeddings, head_dim=self.head_dim, theta=self.rope_theta)

            from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor
            device_str = "cpu"
            RoPE_infinicore_v2.sin_table = create_infinicore_tensor(sin_table.to(dtype=torch.float16), device_str)
            RoPE_infinicore_v2.cos_table = create_infinicore_tensor(cos_table.to(dtype=torch.float16), device_str)

        print(id(RoPE_infinicore_v2.sin_table))
        print(id(self.get_sin_table()))

    def forward_torch(self, states: torch.Tensor,
                      position_ids: torch.Tensor,
                      algo=Algorithm.GPT_NEOX):

        print("RoPE_infinicore_v2:: forward")
        from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor, infini_tensor_2_torch_tensor
        bs, num_attention_heads, ntok, head_dim = states.shape

        states = states.transpose(1, 2).reshape(-1, num_attention_heads, head_dim).contiguous()

        if algo == Algorithm.GPT_J:
            raise ValueError("RoPE_infinicore not support")
        else:
            out = infinicore.rope(torch_tensor_2_infini_tensor(states),
                                  torch_tensor_2_infini_tensor(position_ids),
                                  self.get_sin_table(),
                                  self.get_cos_table(),
                                  )

        out_torch = infini_tensor_2_torch_tensor(out)
        out_torch = out_torch.reshape(-1, ntok, num_attention_heads, head_dim).transpose(1, 2).contiguous()
        return out_torch

    def forward_infini(self, states: infinicore.Tensor,
                       position_ids: torch.Tensor,
                       algo=Algorithm.GPT_NEOX):
        print("RoPE_infinicore_v2:: forward_infini ???")
        from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor, infini_tensor_2_torch_tensor
        bs, ntok, num_attention_heads, head_dim = states.shape

        states = states.view((bs * ntok, num_attention_heads, head_dim))

        if algo == Algorithm.GPT_J:
            raise ValueError("RoPE_infinicore not support")
        else:
            out = infinicore.rope(states,
                                  torch_tensor_2_infini_tensor(position_ids),
                                  self.get_sin_table(),
                                  self.get_cos_table(),
                                  )

        out_torch = out.view((bs, ntok, num_attention_heads, head_dim))
        return out_torch

    def forward(self, states: Union[infinicore.Tensor, torch.Tensor],
                position_ids: torch.Tensor,
                ):
        if isinstance(input, torch.Tensor):
            self.forward_torch(states, position_ids)

        return self.forward_infini(states, position_ids)

    def get_sin_table(self):
        return RoPE_infinicore_v2.sin_table

    def get_cos_table(self):
        return RoPE_infinicore_v2.cos_table

    def create_sin_cos_table(self, max_position, head_dim=64, theta=10000.0):
        assert head_dim % 2 == 0, "Embedding dimension must be even."
        pos = torch.arange(0, max_position)
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        angles = torch.outer(pos, freqs)
        return torch.sin(angles), torch.cos(angles)


class RoPE_infinicore():
    shared_count = 0

    def __init__(self, config=None):
        pass

    def forward_torch(self, x: torch.Tensor,
                      sin_table,
                      cos_table):
        print("----------> RoPE_infinicore:: forward_torch")
        rotary_embedding(x, x, sin_table, cos_table)
        return x

    def forward_infini_huai(self, x: torch.Tensor,
                            sin_table: torch.Tensor,
                            cos_table: torch.Tensor,
                            algo=Algorithm.GPT_NEOX):
        print("----------> RoPE_infinicore:: forward_infini")
        from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor, infini_tensor_2_torch_tensor
        bs, num_attention_heads, ntok, head_dim = x.shape
        #

        half_dim = head_dim // 2
        pos = ntok

        pos_ids = torch.arange(0, pos, dtype=torch.int32)
        if algo == Algorithm.GPT_J:
            raise ValueError("RoPE_infinicore not support")
        else:
            out = infinicore.rope(torch_tensor_2_infini_tensor(x),
                                  torch_tensor_2_infini_tensor(pos_ids),
                                  torch_tensor_2_infini_tensor(sin_table.reshape(-1, head_dim)[..., 0:half_dim].contiguous()),
                                  torch_tensor_2_infini_tensor(cos_table.reshape(-1, head_dim)[..., 0:half_dim].contiguous()),
                                  )

        out_torch = infini_tensor_2_torch_tensor(out)

        return out_torch

    def forward_torch_infini(self, x: torch.Tensor,
                             sin_table,
                             cos_table,
                             algo=Algorithm.GPT_NEOX):
        print("----------> RoPE_infinicore:: forward_infini")
        from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor, infini_tensor_2_torch_tensor
        bs, num_attention_heads, ntok, head_dim = x.shape

        x = x.transpose(1, 2).reshape(-1, num_attention_heads, head_dim).contiguous()
        half_dim = head_dim // 2
        pos = ntok

        pos_ids = torch.arange(0, pos, dtype=torch.int32)
        if algo == Algorithm.GPT_J:
            raise ValueError("RoPE_infinicore not support")
        else:
            out = infinicore.rope(torch_tensor_2_infini_tensor(x),
                                  torch_tensor_2_infini_tensor(pos_ids),
                                  torch_tensor_2_infini_tensor(sin_table.reshape(-1, head_dim)[..., 0:half_dim].contiguous()),
                                  torch_tensor_2_infini_tensor(cos_table.reshape(-1, head_dim)[..., 0:half_dim].contiguous()),
                                  )

        out_torch = infini_tensor_2_torch_tensor(out)

        out_torch = out_torch.reshape(-1, ntok, num_attention_heads, head_dim).transpose(1, 2).contiguous()

        # exit()
        return out_torch

    def forward(self, x: Union[infinicore.Tensor, torch.Tensor],
                sin_table,
                cos_table,
                ):
        if isinstance(x, torch.Tensor):
            # return self.forward_torch(x, sin_table, cos_table)
            return self.forward_torch_infini(x, sin_table, cos_table)

        return self.forward_infini(x, sin_table, cos_table)


import infinicore


class LlamaMLP(infinicore.nn.Module):
    def __init__(self, config):
        super().__init__()
        '''
        nn.Linear 使用的是  infinicore.nn.Linear
        nn.Silu 使用的是  infinicore.nn.Silu

        整体的的输入是 touch.tensor，输出是 torch.tensor 
        '''
        import infinicore

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = infinicore.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = infinicore.nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = infinicore.nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward_infini2infini(self, x: infinicore.Tensor) -> infinicore.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def forward_torch2torch(self, x: torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def forward(self, x: Union[infinicore.Tensor, torch.Tensor]):
        if isinstance(x, torch.Tensor):
            return self.forward_torch2torch(x)

        return self.forward_infini2infini(x)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


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
        self.is_causal = True

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rope_infinicore = RoPE_infinicore_v2(config)

        import infinicore
        self.q_proj = infinicore.nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = infinicore.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = infinicore.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = infinicore.nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward_torch2infini2torch(self,
                                   hidden_states: torch.Tensor,
                                   position_embeddings: tuple[torch.Tensor, torch.Tensor],
                                   attention_mask: Optional[torch.Tensor],
                                   past_key_values: Optional[Cache] = None,
                                   cache_position: Optional[torch.LongTensor] = None,
                                   **kwargs: Unpack[TransformersKwargs],
                                   ) -> tuple[Union[infinicore.Tensor, torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        if True:
            from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor
            device_str = "cpu"

            query_hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)
            key_hidden_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
            value_hidden_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

            hidden_states_infinicore = create_infinicore_tensor(hidden_states, device_str)

            query_states_infinicore = self.q_proj(hidden_states_infinicore).view(query_hidden_shape)
            key_states_infinicore = self.k_proj(hidden_states_infinicore).view(key_hidden_shape)
            value_states_infinicore = self.v_proj(hidden_states_infinicore).view(value_hidden_shape).permute((0, 2, 1, 3))

            query_states = self.rope_infinicore.forward(query_states_infinicore, cache_position)
            key_states = self.rope_infinicore.forward(key_states_infinicore, cache_position)

            query_states = infini_tensor_2_torch_tensor(query_states).permute((0, 2, 1, 3))
            key_states = infini_tensor_2_torch_tensor(key_states).permute((0, 2, 1, 3))
            value_states = infini_tensor_2_torch_tensor(value_states_infinicore)
        else:
            hidden_shape = (*input_shape, -1, self.head_dim)

            #  [bs, ntok, hidden_size] => [bs, ntok, num_attention_heads * head_dim] => [bs, ntok, -1, head_dim] => [bs, -1, ntok, head_dim]
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # => [bs, num_attention_heads, ntok, head_dim]
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # => [bs, num_key_value_heads, ntok, head_dim]
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # => [bs, num_key_value_heads, ntok, head_dim]

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Size([1, 5, 32, 64])
        # Size([bs, ntok, num_attention_heads, head_dim])
        attn_output, attn_weights = attention_interface(
            self,
            query_states,  # [bs, num_attention_heads, ntok, head_dim]
            key_states,  # [bs, num_key_value_heads, all_tok, head_dim]
            value_states,  # [bs, num_key_value_heads, all_tok, head_dim]
            attention_mask,  # [1, 1, ntok,all_tok]
            dropout=0.0,  #
            scaling=self.scaling,  # 缩放系数 0.125
            **kwargs,  # 'position_ids': tensor([[0, 1, 2, 3, 4]])
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        if True:
            from infinicore.nn.modules.linear import torch_tensor_2_infini_tensor
            device_str = "cpu"
            print('????????????????????????????DDDDDDDDDDDDDDDDDD')
            attn_output_infinicore = create_infinicore_tensor(attn_output, device_str)
            attn_output_infinicore = self.o_proj(attn_output_infinicore)
            attn_output = infini_tensor_2_torch_tensor(attn_output_infinicore)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output

    def forward_infini2infini(self,
                              hidden_states: infinicore.Tensor,
                              position_embeddings: tuple[torch.Tensor, torch.Tensor],
                              attention_mask: Optional[torch.Tensor],
                              past_key_values: Optional[Cache] = None,
                              cache_position: Optional[torch.LongTensor] = None,
                              **kwargs: Unpack[TransformersKwargs],
                              ) -> tuple[Union[infinicore.Tensor, torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor
        device_str = "cpu"

        query_hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_hidden_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        value_hidden_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        query_states_infinicore = self.q_proj(hidden_states).view(query_hidden_shape)
        key_states_infinicore = self.k_proj(hidden_states).view(key_hidden_shape)
        value_states_infinicore = self.v_proj(hidden_states).view(value_hidden_shape).permute((0, 2, 1, 3))

        query_states = self.rope_infinicore.forward(query_states_infinicore, cache_position)
        key_states = self.rope_infinicore.forward(key_states_infinicore, cache_position)

        query_states = infini_tensor_2_torch_tensor(query_states).permute((0, 2, 1, 3))
        key_states = infini_tensor_2_torch_tensor(key_states).permute((0, 2, 1, 3))
        value_states = infini_tensor_2_torch_tensor(value_states_infinicore)

        # kv cache
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attention
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Size([bs, ntok, num_attention_heads, head_dim])  Size([1, 5, 32, 64])
        attn_output, attn_weights = attention_interface(
            self,
            query_states,  # [bs, num_attention_heads, ntok, head_dim]
            key_states,  # [bs, num_key_value_heads, all_tok, head_dim]
            value_states,  # [bs, num_key_value_heads, all_tok, head_dim]
            attention_mask,  # [1, 1, ntok,all_tok]
            dropout=0.0,  #
            scaling=self.scaling,  # 缩放系数 0.125
            **kwargs,  # 'position_ids': tensor([[0, 1, 2, 3, 4]])
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # o_proj
        device_str = "cpu"
        attn_output_infinicore = create_infinicore_tensor(attn_output, device_str)
        attn_output = self.o_proj(attn_output_infinicore)

        return attn_output

    def forward(self,
                hidden_states: Union[infinicore.Tensor, torch.Tensor],
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_values: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs: Unpack[TransformersKwargs],
                ) -> tuple[Union[infinicore.Tensor, torch.Tensor]]:

        if isinstance(hidden_states, torch.Tensor):
            return self.forward_torch2infini2torch(hidden_states=hidden_states,
                                                   attention_mask=attention_mask,
                                                   past_key_values=past_key_values,
                                                   cache_position=cache_position,
                                                   position_embeddings=position_embeddings,
                                                   **kwargs)

        return self.forward_infini2infini(hidden_states=hidden_states,
                                          attention_mask=attention_mask,
                                          past_key_values=past_key_values,
                                          cache_position=cache_position,
                                          position_embeddings=position_embeddings,
                                          **kwargs)


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: Union[infinicore.Tensor, torch.Tensor],  # [bs, ntok, hidden_size]
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                **kwargs: Unpack[TransformersKwargs],
                ) -> Union[infinicore.Tensor, torch.Tensor]:
        if isinstance(hidden_states, torch.Tensor):
            return self.forward_torch2torch(hidden_states,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            past_key_values=past_key_values,
                                            use_cache=use_cache,
                                            cache_position=cache_position,
                                            position_embeddings=position_embeddings,
                                            **kwargs)

        return self.forward_infini2infini(hidden_states,
                                          attention_mask=attention_mask,
                                          position_ids=position_ids,
                                          past_key_values=past_key_values,
                                          use_cache=use_cache,
                                          cache_position=cache_position,
                                          position_embeddings=position_embeddings,
                                          **kwargs)

    def forward_torch2torch(self,
                            hidden_states: torch.Tensor,  # [bs, ntok, hidden_size]
                            attention_mask: Optional[torch.Tensor] = None,
                            position_ids: Optional[torch.LongTensor] = None,
                            past_key_values: Optional[Cache] = None,
                            use_cache: Optional[bool] = False,
                            cache_position: Optional[torch.LongTensor] = None,
                            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                            **kwargs: Unpack[TransformersKwargs],
                            ) -> torch.Tensor:
        residual = hidden_states

        if True:
            from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor
            device_str = "cpu"

            hidden_states_infinicore = create_infinicore_tensor(hidden_states, device_str)

            #
            hidden_states_infinicore = self.input_layernorm(hidden_states_infinicore)
            hidden_states_infinicore = self.self_attn(
                hidden_states=hidden_states_infinicore,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = infini_tensor_2_torch_tensor(hidden_states_infinicore, device_str)
            hidden_states = residual + hidden_states

        else:
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        if True:
            from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor
            device_str = "cpu"
            hidden_states_infinicore = create_infinicore_tensor(hidden_states, device_str)

            hidden_states_infinicore = self.post_attention_layernorm(hidden_states_infinicore)
            hidden_states_infinicore = self.mlp(hidden_states_infinicore)

            hidden_states_output_torch = infini_tensor_2_torch_tensor(hidden_states_infinicore, device_str)
            hidden_states = hidden_states_output_torch

            hidden_states = residual + hidden_states
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states

    def forward_infini2infini(self,
                              hidden_states: infinicore.Tensor,  # [bs, ntok, hidden_size]
                              attention_mask: Optional[torch.Tensor] = None,
                              position_ids: Optional[torch.LongTensor] = None,
                              past_key_values: Optional[Cache] = None,
                              use_cache: Optional[bool] = False,
                              cache_position: Optional[torch.LongTensor] = None,
                              position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                              **kwargs: Unpack[TransformersKwargs],
                              ) -> torch.Tensor:

        residual = hidden_states
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = infinicore.nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            # input_ids :     {1,5}       tensor([[    1,  1128,   526,   366, 29892]])
            # inputs_embeds : {1,5,2048}  tensor([[[...]]])
            input_ids = input_ids.to(dtype=torch.int32)

            if True:
                import infinicore
                from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor

                input_ids_infini = create_infinicore_tensor(input_ids, "cpu")

                inputs_embeds_infini = self.embed_tokens(input_ids_infini)
                inputs_embeds = infini_tensor_2_torch_tensor(inputs_embeds_infini)

            else:
                inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(config=self.config,
                                         input_embeds=inputs_embeds,
                                         attention_mask=attention_mask,
                                         cache_position=cache_position,
                                         past_key_values=past_key_values,
                                         position_ids=position_ids)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if True:
            from infinicore.nn.modules.linear import create_infinicore_tensor, infini_tensor_2_torch_tensor
            device_str = "cpu"
            hidden_states = create_infinicore_tensor(hidden_states, device_str)

            for decoder_layer in self.layers[:self.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = self.norm(hidden_states)
            hidden_states = infini_tensor_2_torch_tensor(hidden_states)

        else:
            for decoder_layer in self.layers[:self.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values)


class LlamaForCausalLM_cpp(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(loss=None,
                                      logits=logits, past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions, )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
