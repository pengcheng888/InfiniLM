# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import functools

import inspect
import os
import re
import copy
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, Callable, Optional, TypeVar, Union, get_type_hints


import torch
from packaging import version
from torch import Tensor, nn


from .configuration_utils import PretrainedConfig

from .generation import  GenerationConfig

from .integrations.accelerate import find_tied_parameters
from .integrations.sdpa_attention import sdpa_attention_forward



from .utils import (
    SAFE_WEIGHTS_NAME,
    ContextManagers,
    is_accelerate_available,
    is_safetensors_available,
    is_torch_greater_or_equal,
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
)
from .utils.generic import _CAN_RECORD_REGISTRY, GeneralInterface
from .utils.import_utils import (
    ENV_VARS_TRUE_VALUES,
    is_sagemaker_mp_enabled,
)


if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import (
        check_tied_parameters_on_same_device,
        get_balanced_memory,
        get_max_memory
    )
if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)

XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()
SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")
_init_weights = True


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name

TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}


@contextmanager
def no_init_weights():
    """
    Context manager to globally disable weight initialization to speed up loading large models.
    """
    global _init_weights
    old_init_weights = _init_weights

    _init_weights = False

    def _skip_init(*args, **kwargs):
        pass

    # Save the original initialization functions
    for name, init_func in TORCH_INIT_FUNCTIONS.items():
        setattr(torch.nn.init, name, _skip_init)

    try:
        yield
    finally:
        _init_weights = old_init_weights
        # Restore the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)


def restore_default_dtype(func):
    """
    Decorator to restore the default torch dtype
    at the end of the function. Serves
    as a backup in case calling the function raises
    an error after the function has changed the default dtype but before it could restore it.
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        old_dtype = torch.get_default_dtype()
        try:
            return func(*args, **kwargs)
        finally:
            torch.set_default_dtype(old_dtype)

    return _wrapper


def get_torch_context_manager_or_global_device():
    """
    Test if a device context manager is currently in use, or if it is not the case, check if the default device
    is not "cpu". This is used to infer the correct device to load the model on, in case `device_map` is not provided.
    """
    device_in_context = torch.tensor([]).device
    # `get_default_device` was only introduced in torch>=2.3 - use cpu otherwise to align the behavior
    default_device = torch.get_default_device() if is_torch_greater_or_equal("2.3") else torch.device("cpu")
    # This case means no context manager was used -> we still check if the default that was potentially set is not cpu
    if device_in_context == default_device:
        if default_device != torch.device("cpu"):
            return default_device
        return None
    return device_in_context


def get_parameter_device(parameter: Union[nn.Module, "ModuleUtilsMixin"]):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> list[tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: Union[nn.Module, "ModuleUtilsMixin"]):
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            # Adding fix for https://github.com/pytorch/xla/issues/4152
            # Fixes issue where the model code passes a value that is out of range for XLA_USE_BF16=1
            # and XLA_DOWNCAST_BF16=1 so the conversion would cast it to -inf
            # NOTE: `is_torch_xla_available()` is checked last as it induces a graph break in torch dynamo
            if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                return torch.bfloat16
            if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                if t.dtype == torch.float:
                    return torch.bfloat16
                if t.dtype == torch.double:
                    return torch.float32
            return t.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    # For nn.DataParallel compatibility in PyTorch > 1.5
    def find_tensor_attributes(module: nn.Module) -> list[tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for gen_tuple in gen:
        last_tuple = gen_tuple
        if gen_tuple[1].is_floating_point():
            return gen_tuple[1].dtype

    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype

    # fallback to buffer dtype
    for t in parameter.buffers():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype
    return last_dtype


def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    return next(state_dict.values()).dtype


str_to_torch_dtype = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

if is_torch_greater_or_equal("2.3.0"):
    str_to_torch_dtype["U16"] = torch.uint16
    str_to_torch_dtype["U32"] = torch.uint32
    str_to_torch_dtype["U64"] = torch.uint64

print("---------------------")
def load_state_dict(
        checkpoint_file: Union[str, os.PathLike],
        is_quantized: bool = False,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        weights_only: bool = True,
):
    """
    Reads a `safetensor` or a `.bin` checkpoint file. We load the checkpoint on "cpu" by default.
    """
    # Use safetensors if possible
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()

            if metadata is not None and metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
                raise OSError(
                    f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                    "you save your model with the `save_pretrained` method."
                )
            state_dict = {}
            for k in f.keys():
                if map_location == "meta":
                    _slice = f.get_slice(k)
                    k_dtype = _slice.get_dtype()
                    if k_dtype in str_to_torch_dtype:
                        dtype = str_to_torch_dtype[k_dtype]
                    else:
                        raise ValueError(f"Cannot load safetensors of unknown dtype {k_dtype}")
                    state_dict[k] = torch.empty(size=_slice.get_shape(), dtype=dtype, device="meta")
                else:
                    state_dict[k] = f.get_tensor(k)
            return state_dict

def _get_tied_weight_keys(module: nn.Module, prefix=""):
    tied_weight_keys = []
    if getattr(module, "_tied_weights_keys", None) is not None:
        names = [f"{prefix}.{k}" if prefix else k for k in module._tied_weights_keys]
        tied_weight_keys.extend(names)
    if getattr(module, "_dynamic_tied_weights_keys", None) is not None:
        names = [f"{prefix}.{k}" if prefix else k for k in module._dynamic_tied_weights_keys]
        tied_weight_keys.extend(names)
    for name, submodule in module.named_children():
        local_prefix = f"{prefix}.{name}" if prefix else name
        tied_weight_keys.extend(_get_tied_weight_keys(submodule, prefix=local_prefix))
    return tied_weight_keys



def _infer_parameter_dtype(
        model: "PreTrainedModel",
        param_name: str,
        empty_param: torch.Tensor,
        keep_in_fp32_regex: Optional[re.Pattern] = None,

) -> Union[bool, Optional[torch.dtype]]:
    try:
        old_param = model.get_parameter_or_buffer(param_name)
    except Exception as e:

        raise e
    is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
    # We convert floating dtypes to the `dtype` passed except for float8_e4m3fn type. We also want to keep the buffers/params
    # in int/uint/bool and not cast them.
    casting_dtype = None
    is_param_float8_e4m3fn = is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn
    if empty_param.dtype.is_floating_point and not is_param_float8_e4m3fn:
        # First fp32 if part of the exception list
        if keep_in_fp32_regex is not None and keep_in_fp32_regex.search(param_name):
            casting_dtype = torch.float32
        else:
            casting_dtype = old_param.dtype
    return old_param is not None and old_param.is_contiguous(), casting_dtype


def _load_parameter_into_model(model: "PreTrainedModel", param_name: str, tensor: torch.Tensor):
    """Cast a single parameter `param_name` into the `model`, with value `tensor`."""
    module, param_type = get_module_from_name(model, param_name)
    # This will check potential shape mismatch if skipped before
    module.load_state_dict({param_type: tensor}, strict=False, assign=True)


@torch.no_grad()
def _load_state_dict_into_meta_model(
        model: "PreTrainedModel",
        state_dict: dict,
        shard_file: str,
        expected_keys: list[str],
        reverse_renaming_mapping: dict[str, str],
        device_map: Optional[dict] = None,
        disk_offload_folder: Optional[str] = None,
        disk_offload_index: Optional[dict] = None,
        cpu_offload_folder: Optional[str] = None,
        cpu_offload_index: Optional[dict] = None,

        is_safetensors: bool = False,
        keep_in_fp32_regex: Optional[re.Pattern] = None,
        unexpected_keys: Optional[list[str]] = None,  # passing `unexpected` for cleanup from quantization items

) -> tuple[Optional[dict], Optional[dict]]:
    """Load parameters from `meta_state_dict` into the model. The parameters of the `meta_state_dict` are on the meta
    device in order to easily infer the shapes and dtypes that they will have. Then proper parameters are then loaded
    from `shard_file`, which is the actual state dict file on disk.
    This function takes care of correctly casting dtypes, devices, and sharding tensors in case of tensor parallelism.
    """

    tensor_device = "cpu"
    if device_map is not None and device_map.get("", None) is not None:
        if device_map[""] not in ("cpu", torch.device("cpu")):
            tensor_device = device_map[""].index if isinstance(device_map[""], torch.device) else device_map[""]
    if device_map is not None:
        device_map_regex = "|".join([re.escape(k) for k in sorted(device_map.keys(), reverse=True)])

    is_meta_state_dict = shard_file.endswith(".safetensors") 
    file_pointer = None
    if is_meta_state_dict:
        file_pointer = safe_open(shard_file, framework="pt", device=tensor_device)

    for param_name, empty_param in state_dict.items():
        if param_name not in expected_keys:  # when loading from ckpt, we skip param if doesnt exist in modeling
            continue
        # we need to use serialized_param_name as file pointer is untouched
        if is_meta_state_dict:
            # This is the name of the parameter as it appears on disk file
            serialized_param_name = reverse_renaming_mapping[param_name]
            param = file_pointer.get_slice(serialized_param_name)
        else:
            param = empty_param.to(tensor_device)  # It is actually not empty!
        to_contiguous, casting_dtype = _infer_parameter_dtype(
            model,
            param_name,
            empty_param,
            None,

        )

        param = param[...]
        if casting_dtype is not None:
            param = param.to(casting_dtype)
        if to_contiguous:
            param = param.contiguous()

        if device_map is None:
            param_device = "cpu"
        else:
            module_layer = re.search(device_map_regex, param_name)
            if not module_layer:
                raise ValueError(f"{param_name} doesn't have any device set.")
            else:
                param_device = device_map[module_layer.group()]


        _load_parameter_into_model(model, param_name, param.to(param_device))

    if file_pointer is not None:
        file_pointer.__exit__(None, None, None)

    return disk_offload_index, cpu_offload_index


def load_shard_file(args):
    (
        shard_file,
        state_dict,
        device_map,
        key_renaming_mapping,
        weights_only,
        model_to_load,
        expected_keys,
        reverse_key_renaming_mapping,
        unexpected_keys,
    ) =args

    map_location = "cpu"
    if (
            shard_file.endswith(".safetensors")
    ):
        map_location = "meta"

    # If shard_file is "", we use the existing state_dict instead of loading it
    if shard_file != "":
        state_dict = load_state_dict(
            shard_file, is_quantized=False, map_location=map_location, weights_only=weights_only
        )

    # Fix the key names
    state_dict = {key_renaming_mapping[k]: v for k, v in state_dict.items() if k in key_renaming_mapping}

    error_msgs = []

    disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model(
        model_to_load,
        state_dict,
        shard_file,
        expected_keys,
        reverse_key_renaming_mapping,
        device_map=device_map,
        disk_offload_folder=None,
        disk_offload_index=None,
        cpu_offload_folder=None,
        cpu_offload_index=None,
        is_safetensors=False,
        keep_in_fp32_regex=None,
        unexpected_keys=unexpected_keys
    )

    return error_msgs, disk_offload_index, cpu_offload_index


def _get_resolved_checkpoint_files(
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
) -> tuple[Optional[list[str]], Optional[dict]]:
    """Get all the checkpoint filenames based on `pretrained_model_name_or_path`, and optional metadata if the checkpoints are sharded.
    """
    def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
        if variant is not None:
            path, name = weights_name.rsplit(".", 1)
            weights_name = f"{path}.{variant}.{name}"
        return weights_name
    
    if pretrained_model_name_or_path is None:
        return None, None

    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(
                os.path.join(pretrained_model_name_or_path, _add_variant(SAFE_WEIGHTS_NAME))
        ):
            # Load from a safetensors checkpoint
            archive_file = os.path.join(
                pretrained_model_name_or_path, _add_variant(SAFE_WEIGHTS_NAME)
            )
        else:
            raise OSError(f"{pretrained_model_name_or_path} not found")
    else:
        raise OSError(f"{pretrained_model_name_or_path} not found")

    logger.info(f"loading weights file {archive_file}")
    resolved_archive_file = archive_file

    checkpoint_files = [resolved_archive_file] if pretrained_model_name_or_path is not None else None
    return checkpoint_files, None


def _get_dtype(
        cls,
        dtype: Optional[Union[str, torch.dtype, dict]],
        checkpoint_files: Optional[list[str]],
        config: PretrainedConfig,
        sharded_metadata: Optional[dict],
        state_dict: Optional[dict],
        weights_only: bool,
) -> tuple[PretrainedConfig, Optional[torch.dtype], Optional[torch.dtype]]:
    """Find the correct `dtype` to use based on provided arguments. Also update the `config` based on the
    inferred dtype. We do the following:
    1. If dtype is not None, we use that dtype
    2. If dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
        weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
    we also may have config.dtype available, but we won't rely on it till v5
    """
    dtype_orig = None
    is_sharded = sharded_metadata is not None

    if dtype is not None:
        if isinstance(dtype, str):
            if dtype == "auto":
                if hasattr(config, "dtype") and config.dtype is not None:
                    dtype = config.dtype
                    logger.info(f"Will use dtype={dtype} as defined in model's config object")
                else:
                    if is_sharded and "dtype" in sharded_metadata:
                        dtype = sharded_metadata["dtype"]
                    elif state_dict is not None:
                        dtype = get_state_dict_dtype(state_dict)
                    else:
                        state_dict = load_state_dict(
                            checkpoint_files[0], map_location="meta", weights_only=weights_only
                        )
                        dtype = get_state_dict_dtype(state_dict)
                    logger.info(
                        "Since the `dtype` attribute can't be found in model's config object, "
                        "will use dtype={dtype} as derived from model's weights"
                    )
            elif hasattr(torch, dtype):
                dtype = getattr(torch, dtype)
                config.dtype = dtype
                for sub_config_key in config.sub_configs:
                    sub_config = getattr(config, sub_config_key)
                    sub_config.dtype = dtype
        elif isinstance(dtype, torch.dtype):
            config.dtype = dtype
            for sub_config_key in config.sub_configs:
                sub_config = getattr(config, sub_config_key)
                sub_config.dtype = dtype
        elif isinstance(dtype, dict):
            for key, curr_dtype in dtype.items():
                if hasattr(config, key):
                    value = getattr(config, key)
                    curr_dtype = curr_dtype if not isinstance(curr_dtype, str) else getattr(torch, curr_dtype)
                    value.dtype = curr_dtype
            # main torch dtype for modules that aren't part of any sub-config
            dtype = dtype.get("")
            dtype = dtype if not isinstance(dtype, str) else getattr(torch, dtype)
            config.dtype = dtype
            if dtype is None:
                dtype = torch.float32
        else:
            raise ValueError(
                f"`dtype` can be one of: `torch.dtype`, `'auto'`, a string of a valid `torch.dtype` or a `dict` with valid `dtype` "
                f"for each sub-config in composite configs, but received {dtype}"
            )

        dtype_orig = cls._set_default_dtype(dtype)
    else:
        # set fp32 as the default dtype for BC
        default_dtype = torch.get_default_dtype()
        config.dtype = default_dtype
        for key in config.sub_configs:
            value = getattr(config, key)
            value.dtype = default_dtype

    return config, dtype, dtype_orig


def _get_device_map(
        model: "PreTrainedModel",
        device_map: Optional[Union[dict, str]],
        max_memory: Optional[dict],
        dtype: Optional[torch.dtype],
        keep_in_fp32_regex: Optional[re.Pattern],
) -> dict:
    """Compute the final `device_map` to use if we passed a value in ['auto', 'balanced', 'balanced_low_0', 'sequential'].
    Otherwise, we check for any device inconsistencies in the device_map.
    """
    if isinstance(device_map, str):
        special_dtypes = {}

        if keep_in_fp32_regex is not None:
            special_dtypes.update(
                {name: torch.float32 for name, _ in model.named_parameters() if keep_in_fp32_regex.search(name)}
            )

        target_dtype = dtype



        no_split_modules = model._get_no_split_modules(device_map)
        device_map_kwargs = {"no_split_module_classes": no_split_modules}

        if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
            device_map_kwargs["special_dtypes"] = special_dtypes
        elif len(special_dtypes) > 0:
            logger.warning(
                "This model has some weights that should be kept in higher precision, you need to upgrade "
                "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
            )

        if device_map != "sequential":
            inferred_max_memory = get_balanced_memory(
                model,
                dtype=target_dtype,
                low_zero=(device_map == "balanced_low_0"),
                max_memory=max_memory,
                **device_map_kwargs,
            )
        else:
            inferred_max_memory = get_max_memory(max_memory)


        # `inferred_max_memory` contains non-reserved memory. There may be *unused* reserved memory in the GPU,
        # which we can use to allocate parameters.
        for device_name in inferred_max_memory:
            if isinstance(device_name, int):  # it's a GPU device
                if is_torch_xpu_available():
                    unused_memory = torch.xpu.memory_reserved(device_name) - torch.xpu.memory_allocated(device_name)
                else:
                    unused_memory = torch.cuda.memory_reserved(device_name) - torch.cuda.memory_allocated(device_name)
                inferred_max_memory[device_name] += unused_memory
            # respect the `max_memory` passed by the user
            if max_memory is not None and device_name in max_memory:
                inferred_max_memory[device_name] = min(inferred_max_memory[device_name], max_memory[device_name])
        device_map_kwargs["max_memory"] = inferred_max_memory

        device_map = infer_auto_device_map(model, dtype=target_dtype, **device_map_kwargs)



    elif device_map is not None:
        tied_params = find_tied_parameters(model)
        # check if we don't have tied param in different devices
        check_tied_parameters_on_same_device(tied_params, device_map)

    return device_map


def _find_missing_and_unexpected_keys(
        cls,
        model: "PreTrainedModel",
        original_checkpoint_keys: list[str],
        checkpoint_keys: list[str],
        device_map: dict,
) -> tuple[list[str], list[str]]:
    """Find missing keys (keys that are part of the model parameters but were NOT found in the loaded state dict keys) and unexpected keys
    (keys found in the loaded state dict keys, but that are NOT part of the model parameters)
    """

    prefix = model.base_model_prefix

    # Compute expected keys, i.e. keys that the FULL model (not model_to_load) expects
    expected_keys = list(model.state_dict().keys())

    # Adjust prefix of the keys to make them match loaded keys before removing them
    missing_keys = sorted(set(expected_keys) - set(checkpoint_keys))
    unexpected_keys = set(checkpoint_keys) - set(expected_keys)


    # Remove nonpersistent buffers from unexpected keys: they are not in the expected keys (model state dict), but
    # may be in the loaded keys. Note that removing all buffers does the job, as they were part of the expected keys anyway
    model_buffers = {n for n, _ in model.named_buffers()}
    unexpected_keys = sorted(unexpected_keys - model_buffers)

    # Old checkpoints may have keys for rotary_emb.inv_freq for each layer, however we moved this buffer to the main model
    # (so the buffer name has changed). Remove them in such a case
    has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer in model_buffers)
    if has_inv_freq_buffers:
        unexpected_keys = [k for k in unexpected_keys if "rotary_emb.inv_freq" not in k]

    tied_params = find_tied_parameters(model)
    for group in tied_params:
        missing_in_group = [k for k in missing_keys if k in group]
        if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
            missing_keys = [k for k in missing_keys if k not in missing_in_group]

    # Model-specific exceptions for missing and unexpected keys (e.g. if the modeling change over time, or any other reason...)
    if cls._keys_to_ignore_on_load_missing is not None:
        for pattern in cls._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pattern, k) is None]

    if cls._keys_to_ignore_on_load_unexpected is not None:
        for pattern in cls._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pattern, k) is None]

    return missing_keys, unexpected_keys


def _find_mismatched_keys(
        model: "PreTrainedModel",
        state_dict: Optional[dict],
        checkpoint_files: Optional[list[str]],
        ignore_mismatched_sizes: bool,
        keys_to_rename_mapping: dict[str, str],
        weights_only: bool,
) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Find potential shape mismatch between the different state dicts and the model parameters, but only if `ignore_mismatched_sizes`
    is True. Otherwise, return immediately and any shape mismatch that may exist will be raised later on. This avoids checking
    every parameter in advance, as shape mismatch are extremely rare in practice. If we want to ignore them however, we do
    need to check in advance as we need to know which parameters we need to move back from meta to cpu, and initialize
    correctly. Indeed, as our model initialization takes place at the module level, and not the weight level, in the
    case of a sharded checkpoint we cannot correctly initialize the weights according to `model._init_weights()` if we perform
    this check on each state dict at loading time (after the first loaded checkpoint, there are no way to initialize only the
    mismatched weights if any, without overwriting the previously loaded weights as well because all the module will be
    initialized, not only the weights that are mismatched).
    """

    # An error will be raised later on anyway if there is a mismatch - this avoids running the rest of this function
    # if there are no mismatch (which is almost always the case)
    if not ignore_mismatched_sizes:
        return [], []

    if state_dict is not None:
        checkpoint_files = [""]

    model_state_dict = model.state_dict()
    mismatched_keys = []
    mismatched_shapes = []
    for shard_file in checkpoint_files:
        # If shard_file is "", we use the existing state_dict instead of loading it
        if shard_file != "":
            state_dict = load_state_dict(
                shard_file, is_quantized=False, map_location="meta", weights_only=weights_only
            )

        # Fix the key names
        new_state_dict = {keys_to_rename_mapping[k]: v for k, v in state_dict.items() if k in keys_to_rename_mapping}

        for key, tensor in new_state_dict.items():
            if key in model_state_dict and tensor.shape != model_state_dict[key].shape:
                # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size differences.
                # Without matching with module type or parameter type it seems like a practical way to detect valid 4bit weights.
                if not (
                        False and tensor.shape[-1] == 1 and tensor.numel() * 2 == model_state_dict[key].numel()
                ):
                    mismatched_keys.append(key)
                    mismatched_shapes.append((tensor.shape, model_state_dict[key].shape))

    return mismatched_keys, mismatched_shapes



class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

# import infinicore
class PreTrainedModel(torch.nn.Module, ModuleUtilsMixin):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:
        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):
        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a PyTorch model,
          taking as arguments:
            - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
            - **config** ([`PreTrainedConfig`]) -- An instance of the configuration associated to the model.
            - **path** (`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
        - **can_record_outputs** (dict):"""

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None

    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None

    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    _is_stateful = False

    # SDPA support
    _supports_sdpa = False

    # This flag signal that the model can be used as an efficient backend in TGI and vLLM
    # In practice, it means that they support attention (mask) interface functions, fully pass the kwargs
    # through all modules up to the Attention layer, can slice logits with Tensor, and have a default TP plan
    _can_record_outputs = None


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # For BC we keep the original `config_class` definition in case
        # there is a `config_class` attribute (e.g. remote code models),
        # otherwise we derive it from the annotated `config` attribute.

        # defined in this particular subclass
        child_annotation = cls.__dict__.get("__annotations__", {}).get("config", None)
        child_attribute = cls.__dict__.get("config_class", None)

        # defined in the class (this subclass or any parent class)
        full_annotation = get_type_hints(cls).get("config", None)
        full_attribute = cls.config_class

        # priority (child class_config -> child annotation -> global class_config -> global annotation)
        if child_attribute is not None:
            cls.config_class = child_attribute
        elif child_annotation is not None:
            cls.config_class = child_annotation
        elif full_attribute is not None:
            cls.config_class = full_attribute
        elif full_annotation is not None:
            cls.config_class = full_annotation

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            # raise TypeError(
            #     f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
            #     "`PretrainedConfig`. To create a model from a pretrained model use "
            #     f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            # )
            print(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

        
        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

        self._no_split_modules = self._no_split_modules or []
        _CAN_RECORD_REGISTRY[str(self.__class__)] = self._can_record_outputs  # added for executorch support only
        a = ""


    @classmethod
    def _set_default_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """
        Change the default dtype and return the previous one. This is needed when wanting to instantiate the model
        under specific dtype.

        Args:
            dtype (`torch.dtype`):
                a floating dtype to set to.

        Returns:
            `torch.dtype`: the original `dtype` that can be used to restore `torch.set_default_dtype(dtype)` if it was
            modified. If it wasn't, returns `None`.

        Note `set_default_dtype` currently only works with floating-point types and asserts if for example,
        `torch.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.
        """
        if not dtype.is_floating_point:
            raise ValueError(
                f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
            )

        logger.info(f"Instantiating {cls.__name__} model under default dtype {dtype}.")
        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        return dtype_orig

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()` from the `GenerationMixin`.

        Under the hood, on classes where this function returns True, some generation-specific changes are triggered:
        for instance, the model instance will have a populated `generation_config` attribute.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Directly inherits `GenerationMixin` -> can generate
        if "GenerationMixin" in str(cls.__bases__):
            return True
        # The class inherits from a class that can generate (recursive check) -> can generate
        for base in cls.__bases__:
            if not hasattr(base, "can_generate"):
                continue
            if "PreTrainedModel" not in str(base) and base.can_generate():
                return True
        # Detects whether `prepare_inputs_for_generation` has been overwritten in the model. Prior to v4.45, this
        # was how we detected whether a model could generate.
        if hasattr(cls, "prepare_inputs_for_generation"):  # implicit: doesn't inherit `GenerationMixin`
            logger.warning(
                f"{cls.__name__} has generative capabilities, as `prepare_inputs_for_generation` is explicitly "
                "defined. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, "
                "`PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability "
                "to call `generate` and other related functions."
                "\n  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the "
                "model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes"
                "\n  - If you are the owner of the model architecture code, please modify your model class such that "
                "it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception)."
                "\n  - If you are not the owner of the model architecture class, please contact the model code owner "
                "to update it."
            )
        # Otherwise, can't generate
        return False

    def _sdpa_can_dispatch(self, is_init_check: bool = False) -> bool:
        """
        Check the availability of SDPA for a given model.

        Args:
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, and/or if the model uses
                BetterTransformer, which are only available later after __init__. This allows to raise proper exceptions early
                before instantiating the full models if we know that the model does not support the requested attention.
        """
        if not self._supports_sdpa:
            raise ValueError(
                f"{self.__class__.__name__} does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet."
                " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/28005. If you believe"
                ' this error is a bug, please open an issue in Transformers GitHub repository and load your model with the argument `attn_implementation="eager"` meanwhile. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
            )

        if (
                torch.version.hip is not None
                and torch.cuda.device_count() > 1
                and version.parse(torch.__version__) < version.parse("2.4.1")
        ):
            logger.warning_once(
                "Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends."
            )
            torch.backends.cuda.enable_flash_sdp(False)

        if not is_init_check:
            if getattr(self, "use_bettertransformer", False):
                raise ValueError(
                    "SDPA and BetterTransformer API are not compatible. Please make sure to disable BetterTransformers by doing model.reverse_bettertransformer()"
                )

        return True




    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        
        return super().cuda(*args, **kwargs)

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        # For BNB/GPTQ models, we prevent users from casting the model to another dtype to restrict unwanted behaviours.
        # the correct API should be to load the model with the desired dtype directly through `from_pretrained`.
        dtype_present_in_args = "dtype" in kwargs

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break



        return super().to(*args, **kwargs)



    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            ignore_mismatched_sizes: bool = False,
            weights_only: bool = True,
            **kwargs,
    ):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.


            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).

            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)), or `"flash_attention_3"` (using [Dao-AILab/flash-attention/hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

                Accept HF kernel references in the form:
                  <namespace>/<repo_name>[@<revision>][:<kernel_name>]

                - <namespace> and <repo_name> are any non-"/" and non-":" sequences.
                - "@<revision>" is optional (branch, tag, or commit-ish), e.g. "@main", "@v1.2.0", "@abc123".
                - ":<kernel_name>" is optional and selects a function inside the kernel repo.
                - Both options can appear together and in this order only: @revision first, then :kernel_name.
                - We intentionally allow a leading "<wrapper>|" prefix (e.g., "flash|...") because the code
                  strips it before loading; '|' is not excluded in the character classes here.

                Examples that match:
                  "org/model"
                  "org/model@main"
                  "org/model:custom_kernel"
                  "org/model@v1.2.3:custom_kernel"

            > Parameters for big model inference

            dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch_dtype` and load the model under a specific `dtype`. The different options
                are:

                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                  `dtype`, ignoring the model's `config.dtype` if one exists. If not specified
                  - the model will get loaded in `torch.float` (fp32).

                2. `"auto"` - A `dtype` or `torch_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

                3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

                <Tip>

                For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
                reach out to the authors and ask them to add this information to the model's card and to insert the
                `dtype` entry in `config.json` on the hub.

                </Tip>

            device_map (`str` or `dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).


            weights_only (`bool`, *optional*, defaults to `True`):
                Indicates whether unpickler should be restricted to loading only tensors, primitive types,
                dictionaries and any types added via torch.serialization.add_safe_globals().
                When set to False, we can load wrapper tensor subclass weights.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        """

        dtype = kwargs.pop("dtype", None)  # xx
        device_map = kwargs.pop("device_map", None)  # xx

        if device_map == "auto" and int(os.environ.get("WORLD_SIZE", "0")):
            logger.info(
                "You've set device_map=`auto` while triggering a distributed run with torchrun. This might lead to unexpected behavior. "
                "If your plan is to load the model on each device, you should set device_map={"
                ": PartialState().process_index} where PartialState comes from accelerate library"
            )

        # change device_map into a map if we passed an int, a str or a torch.device
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": False}

        # Load config
        config_path = pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            cache_dir=None,
            return_unused_kwargs=True,
            force_download=False,
            proxies=None,
            local_files_only=True,
            token=None,
            revision="main",
            subfolder="",
            gguf_file=None,
            _from_auto=False,
            _from_pipeline=None,
            **kwargs,
        )
        if "gguf_file" in model_kwargs:
            model_kwargs.pop("gguf_file")

        # Because some composite configs call super().__init__ before instantiating the sub-configs,
        # we need this call to correctly redispatch recursively if the kwarg is provided
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        checkpoint_files, _ = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path)

        # Find the correct dtype based on current state
        config, dtype, dtype_orig = _get_dtype(
            cls, dtype, checkpoint_files, config, None, None, weights_only
        )

        config.name_or_path = pretrained_model_name_or_path
        model_init_context =[]
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        with ContextManagers(model_init_context):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # Prepare the full device map
        if device_map is not None:
            device_map = _get_device_map(model, device_map, None, dtype, None)


        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = cls._load_pretrained_model(
            model,
            checkpoint_files,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            device_map=device_map,
            dtype=dtype,
            weights_only=weights_only,
        )

      

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # If it is a model with generation capabilities, attempt to load generation files (generation config,
        # custom generate function)

        if model.can_generate() and pretrained_model_name_or_path is not None:
            repo_loading_kwargs = {
                "cache_dir": None,
                "force_download": False,
                "proxies": None,
                "local_files_only": True,
                "token": None,
                "revision": "main",
                "subfolder": "",
                **kwargs,
            }
            # Load generation config
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    _from_auto=False,
                    _from_pipeline=None,
                    **repo_loading_kwargs,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass
            # Load custom generate function if `pretrained_model_name_or_path` defines it (and override `generate`)
            if hasattr(model, "load_custom_generate"):
                try:
                    custom_generate = model.load_custom_generate(
                        pretrained_model_name_or_path, trust_remote_code=None, **repo_loading_kwargs
                    )
                    model.generate = functools.partial(custom_generate, model=model)
                except OSError:  # there is no custom generate function
                    pass

        # Dispatch model with hooks on all devices if necessary (not needed with a tp_plan, so we skip it as it slightly
        # harm performances)
        if device_map is not None:
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": None,
                "offload_index": None,
                "offload_buffers": False,
            }
            if "skip_keys" in inspect.signature(dispatch_model).parameters:
                device_map_kwargs["skip_keys"] = model._skip_keys_device_placement

            dispatch_model(model, **device_map_kwargs)

        return model

    @staticmethod
    def _fix_state_dict_key_on_load(key: str) -> tuple[str, bool]:
        """Replace legacy parameter names with their modern equivalents. E.g. beta -> bias, gamma -> weight."""
        # Rename LayerNorm beta & gamma params for some early models ported from Tensorflow (e.g. Bert)
        # This rename is logged.
        if key.endswith("LayerNorm.beta"):
            return key.replace("LayerNorm.beta", "LayerNorm.bias"), True
        if key.endswith("LayerNorm.gamma"):
            return key.replace("LayerNorm.gamma", "LayerNorm.weight"), True

        # Rename weight norm parametrizations to match changes across torch versions.
        # Impacts a number of speech/wav2vec models. e.g. Hubert, Wav2Vec2, and others.
        # This rename is not logged.
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            if key.endswith("weight_g"):
                return key.replace("weight_g", "parametrizations.weight.original0"), True
            if key.endswith("weight_v"):
                return key.replace("weight_v", "parametrizations.weight.original1"), True
        else:
            if key.endswith("parametrizations.weight.original0"):
                return key.replace("parametrizations.weight.original0", "weight_g"), True
            if key.endswith("parametrizations.weight.original1"):
                return key.replace("parametrizations.weight.original1", "weight_v"), True

        return key, False

    def _get_key_renaming_mapping(
            self,
            checkpoint_keys: list[str],
    ):
        """
        Compute a mapping between the serialized keys on disk `checkpoint_keys`, and the keys that the model
        that we are loading expects. This is the single entry point for key renaming that will be used during
        loading.
        """
        prefix = self.base_model_prefix
        _prefix = f"{prefix}."

        renamed_keys = {}
        key_renaming_mapping = {}
        for key in checkpoint_keys:
            # Class specific rename
            new_key, has_changed = self._fix_state_dict_key_on_load(key)
            key_renaming_mapping[key] = new_key

            # track gamma/beta rename for logging
            if has_changed:
                if key.endswith("LayerNorm.gamma"):
                    renamed_keys["LayerNorm.gamma"] = (key, new_key)
                elif key.endswith("LayerNorm.beta"):
                    renamed_keys["LayerNorm.beta"] = (key, new_key)

        if renamed_keys:
            warning_msg = f"A pretrained model of type `{self.__class__.__name__}` "
            warning_msg += "contains parameters that have been renamed internally (a few are listed below but more are present in the model):\n"
            for old_key, new_key in renamed_keys.values():
                warning_msg += f"* `{old_key}` -> `{new_key}`\n"
            warning_msg += "If you are using a model from the Hub, consider submitting a PR to adjust these weights and help future users."
            logger.info_once(warning_msg)

        return key_renaming_mapping

    @classmethod
    def _load_pretrained_model(
            cls,
            model: "PreTrainedModel",
            checkpoint_files: Optional[list[str]],
            pretrained_model_name_or_path: Optional[str],
            ignore_mismatched_sizes: bool = False,
            device_map: Optional[dict] = None,
            dtype: Optional[torch.dtype] = None,
            weights_only: bool = True,
    ):
        
        # Get all the keys of the state dicts that we have to initialize the model
        original_checkpoint_keys = list(
            load_state_dict(checkpoint_files[0], map_location="meta", weights_only=weights_only).keys()
        )

        # Find the key names that the model expects from the serialized keys
        key_renaming_mapping = model._get_key_renaming_mapping(    original_checkpoint_keys  )
          
        checkpoint_keys = list(key_renaming_mapping.values())

        # Find missing and unexpected keys from the state dict
        missing_keys, unexpected_keys = _find_missing_and_unexpected_keys(
            cls,
            model,
            original_checkpoint_keys,
            checkpoint_keys,
            device_map,
        )
        # Find all the keys with shape mismatch (if we ignore the mismatch, the weights need to be newly initialized the
        # same way as missing keys)
        mismatched_keys, mismatched_shapes = _find_mismatched_keys(
            model,
            None,
            checkpoint_files,
            ignore_mismatched_sizes,
            key_renaming_mapping,
            weights_only,
        )

        # We need to update both the mapping and the list of checkpoint keys to remove the mismatched ones
        key_renaming_mapping = {k: v for k, v in key_renaming_mapping.items() if v not in mismatched_keys}
        checkpoint_keys = list(key_renaming_mapping.values())
   
        # Make sure we are able to load base models as well as derived models (specific task models, with heads)
        model_to_load = model

        # Get reverse key mapping
        reverse_key_renaming_mapping = {v: k for k, v in key_renaming_mapping.items()}

        # Compute expected model keys
        expected_keys = list(model_to_load.state_dict().keys())
   


        # Prepare and compatabilize arguments for serial and parallel shard loading
        args_list = [
            (
                shard_file,
                None,
                device_map,
                key_renaming_mapping,
                weights_only,
                model_to_load,
                expected_keys,
                reverse_key_renaming_mapping,
                unexpected_keys,
            )
            for shard_file in checkpoint_files
        ]

        error_msgs = []
        if len(args_list) > 1:
            args_list = logging.tqdm(args_list, desc="Loading checkpoint shards")

        for args in args_list:
            _error_msgs, disk_offload_index, cpu_offload_index = load_shard_file(args) # 在这个函数中，权重被复制
            error_msgs += _error_msgs

        # All potential warnings/infos
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = logger.warning if model.__class__.__name__ in archs else logger.info
            warner(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, (shape1, shape2) in zip(mismatched_keys, mismatched_shapes)
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, disk_offload_index, error_msgs


    def get_parameter_or_buffer(self, target: str):
        """
        Return the parameter or buffer given by `target` if it exists, otherwise throw an error. This combines
        `get_parameter()` and `get_buffer()` in a single handy function. If the target is an `_extra_state` attribute,
        it will return the extra state provided by the module. Note that it only work if `target` is a leaf of the model.
        """
        try:
            return self.get_parameter(target)
        except AttributeError:
            pass
        try:
            return self.get_buffer(target)
        except AttributeError:
            pass
        module, param_name = get_module_from_name(self, target)
        if (
                param_name == "_extra_state"
                and getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
                is not torch.nn.Module.get_extra_state
        ):
            return module.get_extra_state()

        raise AttributeError(f"`{target}` is neither a parameter, buffer, nor extra state.")


def expand_device_map(device_map, param_names):
    """
    Expand a device map to return the correspondence parameter name to device.
    """
    new_device_map = {}
    for module, device in device_map.items():
        new_device_map.update(
            {p: device for p in param_names if p == module or p.startswith(f"{module}.") or module == ""}
        )
    return new_device_map


def is_accelerator_device(device: Union[str, int, torch.device]) -> bool:
    """Check if the device is an accelerator. We need to function, as device_map can be "disk" as well, which is not
    a proper `torch.device`.
    """
    if device == "disk":
        return False
    else:
        return torch.device(device).type not in ["meta", "cpu"]


class AttentionInterface(GeneralInterface):
    """
    Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "sdpa": sdpa_attention_forward,
    }


# Global AttentionInterface shared by all models which do not need to overwrite any of the existing ones
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()