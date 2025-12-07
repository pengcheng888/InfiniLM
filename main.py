# 在内存足够的机器上执行一次
import torch
from safetensors.torch import save_file
import os

if False:
    folder = r"/data-aisoft/mechdancer/models/FM9G_70B_SFT_MHA"
    print("111")
    state_dict = torch.load(
        os.path.join(folder, "pytorch_model.bin"), map_location="cpu"
    )
    # 转换为 safetensors 格式
    print("222")
    save_file(state_dict, os.path.join(folder, "model.safetensors"))


def check_tensors():
    # https://huggingface.co/docs/safetensors/index

    from safetensors import safe_open

    model_path = r"/data-aisoft/mechdancer/models/FM9G_70B_SFT_MHA/model.safetensors"
    model_path = r"/data-aisoft/mechdancer/models/9G7B_MHA/model.safetensors"

    tensors = {}
    with safe_open(model_path, framework="pt") as f:
        print(f.metadata())

        for key in f.keys():
            if key.startswith("model.layers.0"):
                print(key)

            # if k == "model.layers.0.mlp.gate_proj.weight":
            #     data = f.get_tensor(k)
            #     data0 = data[0:2816, :][0:1, :]
            #     data1 = data[2816:5632, :][0:1, :]

            #     print(data.shape)
            #     print(data0)
            #     print(data1)
            #     exit(-1)

            # tensors[k] = f.get_tensor(k)
            # tensor_part = f.get_slice(k)[:]
            # print(k, tensor_part.shape)
            pass


def modify_tensors():
    from safetensors.torch import load_file, save_file

    model_path = r"/data-aisoft/mechdancer/models/FM9G_70B_SFT_MHA/model.safetensors"

    model = load_file(model_path)

    keylist = []
    for key in model.keys():
        if (
            key.startswith("model.layers.1")
            or key.startswith("model.layers.2")
            or key.startswith("model.layers.3")
            or key.startswith("model.layers.4")
            or key.startswith("model.layers.5")
            or key.startswith("model.layers.6")
            or key.startswith("model.layers.7")
            or key.startswith("model.layers.8")
            or key.startswith("model.layers.9")
        ):
            keylist.append(key)
    # -----------------------------------------------------#
    for key in keylist:
        model.pop(key)

    print(model.keys())

    save_file(model, "layer_1_model.safetensors", {"format": "pt"})

    return model


check_tensors()

"""
model.layers.0.input_layernorm.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.self_attn.k_proj.bias
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_proj.bias
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.v_proj.bias
model.layers.0.self_attn.v_proj.weight
"""
"""
{'format': 'pt'}
lm_head.weight
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.self_attn.k_proj.bias
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_proj.bias
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.v_proj.bias
model.layers.0.self_attn.v_proj.weight
model.norm.weight

"""
