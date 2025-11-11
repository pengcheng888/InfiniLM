import os
import numbers
import infinilm as transformers_v2
import time
import torch
##----
from typing import Any, Callable, Optional, TypeVar, Union, get_type_hints
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file



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


def load_state_dict(
        checkpoint_file: Union[str, os.PathLike],
        map_location: Optional[Union[str, torch.device]] = "cpu",
        weights_only: bool = True,
):
    """
    Reads a `safetensor` or a `.bin` checkpoint file. We load the checkpoint on "cpu" by default.
    """
    # Use safetensors if possible
    if checkpoint_file.endswith(".safetensors"):
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


def get_config(Folder):
    config_dict = transformers_v2.LlamaConfig._get_config_dict(Folder)[0]
    config = transformers_v2.LlamaConfig(**config_dict)
    return config


def get_config_v2(Folder):
    def load_config_json(dir_path_: str):
        import json
        with open(os.path.join(dir_path_, "config.json"), "r") as f:
            config = json.load(f)
        return config

    config_dict = load_config_json(os.path.join(Folder))
    config = transformers_v2.LlamaConfig(**config_dict)
    return config


import infinicore
def torch_2_infini_ref(model_param:dict):
    print("model_param: ", id(model_param) )

    model_param_infini = {}
    for key,value in model_param.items():
        model_param_infini[key] = infinicore.experimental.torch_2_infini_tensor_ref(value)
    
    return model_param_infini


def func(Folder):
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #

    config = get_config_v2(Folder)

    model = transformers_v2.LlamaForCausalLM(config)

    path = os.path.join(Folder, "model.safetensors")
    model_param = load_state_dict(path)
    if True:
        for k,v in model_param.items():
            model_param[k] = v.to(device="cuda")


    print(model_param)
    model_param_infini = torch_2_infini_ref(model_param) 
    print(model_param_infini)

 
    model_device = "cuda"
    model.load_state_dict(model_param_infini) # cpu进入，里面 to cuda

    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(Folder)

    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    prompt = ["How are you,"]  # {'input_ids': tensor([[    1,  1128,   526,   366, 29892]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
    prompt = "How are you,"  # {'input_ids': tensor([[    1,  1128,   526,   366, 29892]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}

    # prompt = "山东最高的山是"  # {'input_ids': tensor([[    1,  1128,   526,   366, 29892]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
    # prompt = ["How are you,",
    #           "How old are you,"]  # {'input_ids': tensor([[1,1128,526,366, 29892,2],  [1, 1128, 2030, 526, 366, 29892]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])}

    # 'input_ids': tensor([[ 1, 1128, 526, 366, 29892]]
    input_ids = tokenizer(prompt,
                          padding=True,  # 自动填充到相同长度
                          truncation=True,  # 自动截断到最大长度
                          max_length=128,  # 设置最大长度
                          return_tensors="pt").to(model_device)

    with torch.no_grad():
        print('------> start')
        t1 = time.time()
        # outputs, output_tokens_list = model.generate(**input_ids, max_new_tokens=15)  # cache_implementation="static",

        outputs, output_tokens_list = model.generate_wpc(**input_ids, max_new_tokens=15)  # cache_implementation="static",

        t2 = time.time()
        print("time: ", (t2 - t1) * 1000)
        print(outputs, output_tokens_list)
        outputs = torch.tensor([output_tokens_list])
        for output in outputs:
            print(tokenizer.decode(output, skip_special_tokens=True))


if __name__ == '__main__':
    Folder = r'/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/Llama2-TinyLlama-1.1B-Chat-v1.0/'
    Folder = r'/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/TinyLlama-1.1B-Chat-v1.0-small/'
    # Folder = r'/home/ubuntu/models/TinyLlama-1.1B-Chat-v1.0-small/'
    func(Folder)
