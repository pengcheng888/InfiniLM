# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import warnings

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from torch import nn
from ..cache_utils_wpc import (
    Cache,
    DynamicCache,
)


from ..utils import (
    logging,
)

ALL_STATIC_CACHE_IMPLEMENTATIONS = ("static")
from .configuration_utils import (
    GenerationConfig,
    GenerationMode,
)
from .continuous_batching import ContinuousMixin


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)

# Variable names used to hold the cache at generation time
ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]

class GenerationMixin(ContinuousMixin):
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in model classes.
    Inheriting from this class causes the model to have special generation-related behavior, such as loading a
    `GenerationConfig` at initialization time or ensuring `generate`-related tests are run in `transformers` CI.

    A model class should inherit from `GenerationMixin` to enable calling methods like `generate`, or when it
    has defined a custom `generate` method that relies on `GenerationMixin`, directly or indirectly, which
    approximately shares the same interface to public methods like `generate`. Three examples:
        - `LlamaForCausalLM` should inherit from `GenerationMixin` to enable calling `generate` and other public
            methods in the mixin;

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

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

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[str], dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)

        inputs = inputs_kwarg
        return inputs, input_name, model_kwargs

  


    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: dict
    ) -> tuple[GenerationConfig, dict]:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """

        if generation_config is None:

            generation_config = self.generation_config

        # `torch.export.export` usually raises an exception if it is called
        # with ``strict=True``. deepcopy can only be processed if ``strict=False``.
        generation_config = copy.deepcopy(generation_config)

        # Finally, apply any passed kwargs
        model_kwargs = generation_config.update(**kwargs)

        return generation_config, model_kwargs

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

    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, model_kwargs) -> Cache:
        """
        Sets a cache for `generate`, that will persist across calls. A new cache will only be initialized a
        new `generate` call requires a larger cache or uses a different batch size.

        Returns the resulting cache object.
        """
        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )
        offload_cache = "offloaded" in cache_implementation

        if hasattr(self, "_cache"):
            cache_to_check = self._cache.self_attention_cache if requires_cross_attention_cache else self._cache

        need_new_cache = (
            not hasattr(self, "_cache")
            or cache_to_check.offloading != offload_cache
            or cache_to_check.max_batch_size != batch_size
            or cache_to_check.max_cache_len < max_cache_len
        )

        if requires_cross_attention_cache and hasattr(self, "_cache"):
            need_new_cache = (
                need_new_cache
                or self._cache.cross_attention_cache.max_cache_len != model_kwargs["encoder_outputs"][0].shape[1]
            )

        if need_new_cache:
            self_attention_cache_kwargs = {
                "config": self.config.get_text_config(decoder=True),
                "max_cache_len": max_cache_len,
                "offloading": offload_cache,
            }
            self._cache = StaticCache(**self_attention_cache_kwargs)
        else:
            self._cache.reset()
        return self._cache

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        """
        Return `True` if current model can use a `DynamicCache` instance when initializing the `past_key_values`.
        This adds exception for some models like `Mamba` models which use their own caches
        and do not need to initialize the Cache in advance in order to save memory (because no back and forth
        `to_legacy_cache` and `from_legacy_cache` will be performed for mamba-based models).
        """
        # NOTE: remove xlnet/reformer when the models are deprecated, non-standard model architecture/cache name
        return not cls._is_stateful and all(
            special_model_name not in cls.__name__.lower()
            for special_model_name in [
                "reformer",
                "minimax",
                "xlnet",
                "lfm2",
            ]
        )

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
    ) -> bool:
        """
        Prepares the cache for generation (if applicable), given `generate`'s parameterization. If a cache is
        instantiated, writes it to `model_kwargs`, under the name expected by the model.
        """

        is_hybrid_cache = any(class_name in self.__class__.__name__.lower() for class_name in ["mamba", "falconh1"])
        cache_name = "past_key_values" if not is_hybrid_cache else "cache_params"

        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )

        # Quick escape route 2: if the user specifies no cache is to be used. (conflicting arguments are handled in
        # `generation_config.validate()`)
        if generation_config.use_cache is False:
            return

        # Quick escape route 3: model that only supports legacy caches or models that supply it in `prepare_inputs_for_generation` (mamba, zamba, ...)
        if not self._supports_default_dynamic_cache():
            if generation_config.cache_implementation is not None:
                warnings.warn(
                    "This model does not support `Cache` instances, it only supports the legacy cache format (tuple "
                    f"of tuples). `cache_implementation` (set to {generation_config.cache_implementation}) will be "
                    "ignored.",
                    UserWarning,
                )
            return

        # Otherwise we NEED to prepare a cache, based on `generation_config.cache_implementation`

        # TODO(joao): support static caches in assisted generation. assisted generation needs to roll back caches,
        # which is only supported in dynamic caches atm
        if assistant_model is not None and generation_config.cache_implementation is not None:
            logger.warning_once(
                "An assistant model is provided, using a dynamic cache instead of a cache of type="
                f"'{generation_config.cache_implementation}'."
            )
            generation_config.cache_implementation = None

        # Assisted decoding and contrastive search require cache rollback, which is incompatible with sliding layers.
        # To handle this, we skip passing the model config to DynamicCache (forcing a full-layer cache).
        # The "dynamic_full" option is a shortcut for generate() users to avoid sliding layers on their own.
        generation_mode = generation_config.get_generation_mode(assistant_model)
        if (
            generation_mode in (GenerationMode.ASSISTED_GENERATION, GenerationMode.CONTRASTIVE_SEARCH)
            or generation_config.cache_implementation == "dynamic_full"
        ):
            dynamic_cache_kwargs = {}
        else:
            dynamic_cache_kwargs = {"config": self.config}
        if generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in ALL_STATIC_CACHE_IMPLEMENTATIONS:

                model_kwargs[cache_name] = self._get_cache(
                    cache_implementation=generation_config.cache_implementation,
                    batch_size=max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size,
                    max_cache_len=max_cache_length,
                    model_kwargs=model_kwargs,
                )
            elif "dynamic" in generation_config.cache_implementation:
                model_kwargs[cache_name] = DynamicCache(**dynamic_cache_kwargs)

        # Use DynamicCache instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        else:
            model_kwargs[cache_name] = (
                DynamicCache(**dynamic_cache_kwargs)
            )


    def _prepare_special_tokens(
        self,
        generation_config: GenerationConfig,
        kwargs_has_attention_mask: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        decoder_start_token_tensor = _tensor_or_none(generation_config.decoder_start_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")


     
        if eos_token_tensor is not None and (
            torch.is_floating_point(eos_token_tensor) or (eos_token_tensor < 0).any()
        ):
            logger.warning(
                f"`eos_token_id` should consist of positive integers, but is {eos_token_tensor}. Your generation "
                "will not stop until the maximum length is reached. Depending on other flags, it may even crash."
            )

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._decoder_start_token_tensor = decoder_start_token_tensor


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
 

        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[Union[str, Callable]] = None,
        **kwargs,
    ) -> Union[torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
  

            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://huggingface.co/papers/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.


            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            use_model_defaults (`bool`, *optional*):
                When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
                generation configuration (`model.generation_config`), as opposed to the global defaults
                (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
                `True`.
            custom_generate (`str` or `Callable`, *optional*):
                One of the following:
                - `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at
                  `custom_generate/generate.py` in that repository instead of the standard `generate` method. The
                  repository fully replaces the generation logic, and the return type may differ.
                - `str` (local repository path): same as above but from a local path, `trust_remote_code` not required.
                - `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to
                  run the decoding loop.
                For more information, see [the docs](../../generation_strategies#custom-generation-methods).
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],

        """
        # input_ids  attention_mask cache_implementation max_new_tokens
        kwargs.pop("attention_mask")
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
   
        # 2. Set generation parameters if not already defined
        synced_gpus = False
        kwargs_has_attention_mask =False

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids = inputs_tensor
        import infinicore
        model_kwargs["input_ids_infini"] = infinicore.convert_torch_to_infini_tensor(inputs_tensor)
         
        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[1]
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length

   
        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]

        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(None)

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )


        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result



    def _sample(
        self,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        synced_gpus: bool,

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