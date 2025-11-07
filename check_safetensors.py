from safetensors import safe_open
import os

from safetensors import safe_open
import os


def func1():
    from huggingface_hub import get_safetensors_metadata
    model_path = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/openai-community_gpt2/model.safetensors"
    folder = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/openai-community_gpt2/"
    repo_id = "openai-community/gpt2"
    metadata = get_safetensors_metadata(repo_id)
    print(metadata)
    print(metadata.files_metadata["model.safetensors"].metadata)

    # metadata = get_safetensors_metadata("bigscience/bloom")
    # metadata
    # len(metadata.files_metadata)
    #
    # get_safetensors_metadata("runwayml/stable-diffusion-v1-5")


def check_tensors():
    # https://huggingface.co/docs/safetensors/index

    from safetensors import safe_open
    model_path = r"/home/ubuntu/models/Qwen3-0.6B-small/model.safetensors"

    tensors = {}
    with safe_open(model_path, framework="pt") as f:
        print(f.metadata())

        for k in f.keys():
            print(k)
            # tensors[k] = f.get_tensor(k)
            # tensor_part = f.get_slice(k)[:]
            # print(k, tensor_part.shape)
            pass


def modify_tensors():
    from safetensors.torch import load_file, save_file
    model_path = r"/home/ubuntu/models/Qwen3-0.6B-small/model.safetensors"

    model = load_file(model_path)

    keylist = []
    for key in model.keys():

        if key.startswith('model.layers.1') or key.startswith('model.layers.2')or key.startswith('model.layers.3')or key.startswith('model.layers.4')or key.startswith('model.layers.5')or key.startswith('model.layers.6')or key.startswith('model.layers.7')or key.startswith('model.layers.8')or key.startswith('model.layers.9') :
            keylist.append(key)
    # -----------------------------------------------------#
    for key in keylist:
        model.pop(key)

    print(model.keys())

    save_file(model, "layer_0_model.safetensors", {"format": "pt"})

    return model


def modify_tensors_2():
    from safetensors.torch import load_file, save_file
    model_path = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/deepseek-ai_DeepSeek-R1/src/layer_MoE_model.safetensors"

    model = load_file(model_path)

    # newmodel = {key.replace('model.layers.3.', 'model.layers.0.') if key.startswith('model.layers.3.') else key: value for key, value in model.items()}

    # -----------------------------------------------------#
    # delKeyList = []
    # for key, value in model.items():
    #     if key.startswith('model.layers.0.mlp.experts.'):
    #         index = key.split("model.layers.0.mlp.experts.")[1].split(".")[0]
    #         index = int(index)
    #         if index > 30:
    #             delKeyList.append(key)

    # -----------------------------------------------------#
    # for key in delKeyList:
    #     model.pop(key)

    expert256 ={}
    for i in range(31, 128):
        print(i)
        key1 = f'model.layers.0.mlp.experts.{i}.up_proj.weight'
        key2 = f'model.layers.0.mlp.experts.{i}.gate_proj.weight_scale_inv'
        key3 = f'model.layers.0.mlp.experts.{i}.up_proj.weight_scale_inv'
        key4 = f'model.layers.0.mlp.experts.{i}.down_proj.weight_scale_inv'
        key5 = f'model.layers.0.mlp.experts.{i}.gate_proj.weight'
        key6 = f'model.layers.0.mlp.experts.{i}.down_proj.weight'
        expert256[key1] = model['model.layers.0.mlp.experts.0.up_proj.weight'].clone()
        expert256[key2] = model['model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv'].clone()
        expert256[key3] = model['model.layers.0.mlp.experts.0.up_proj.weight_scale_inv'].clone()
        expert256[key4] = model['model.layers.0.mlp.experts.0.down_proj.weight_scale_inv'].clone()
        expert256[key5] = model['model.layers.0.mlp.experts.0.gate_proj.weight'].clone()
        expert256[key6] = model['model.layers.0.mlp.experts.0.down_proj.weight'].clone()

    # -----------------------------------------------------#

    save_file(expert256, "model-00002-of-000003.safetensors", {"format": "pt"})

    return model


def merge():
    from safetensors.torch import load_file, save_file
    model_path = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/deepseek-ai_DeepSeek-R1/src/layer_3-2-newkey_model.safetensors"
    model1 = load_file(model_path)

    model_path = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/deepseek-ai_DeepSeek-R1/src/layer_3-newkey_model.safetensors"
    model2 = load_file(model_path)

    model1.update(model2)

    save_file(model1, "layer_3-newnew_model.safetensors", {"format": "pt"})

    return model1


# 使用示例
if __name__ == "__main__":
    #check_tensors()
    modify_tensors()
