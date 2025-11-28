import time

import torch
from transformers import AutoModelForCausalLM


def func(Folder):
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Folder)

    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        Folder,
        dtype=torch.bfloat16,
        device_map="cuda",
    )  # sdpa flash_attention_2

    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------ #
    prompt = "山东最高的山是？"
    prompt = "how"

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    import time

    print("start -> ")
    torch.cuda.synchronize()
    t1 = time.time()
    output = model.generate(**input_ids, max_new_tokens=100)
    torch.cuda.synchronize()
    t2 = time.time()
    print("time: ", (t2 - t1) * 1000 / len(output[0]))
    print("output[0] : ", len(output[0]))
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    Folder = r"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Llama/TinyLlama-1.1B-Chat-v1.0-small"
    Folder = r"/data/huggingface/TinyLlama-1.1B-Chat-v1.0/"
    # Folder = r"/home/ubuntu/Downloads/Llama-3.2-3B-Instruct/"
    func(Folder)
