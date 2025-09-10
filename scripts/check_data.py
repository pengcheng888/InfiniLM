import numpy as np
import torch


def read_binary_file1(filename, dtype=np.float16):
    """
    读取C++写入的二进制文件
    参数:
        filename: 文件名
        dtype: 数据类型 (默认为float32)
    """
    try:
        # 读取二进制数据到numpy数组
        data = np.fromfile(filename, dtype=dtype)
        print(f"成功读取 {len(data)} 个元素")
        return data
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None


def read_binary_file(filename, dtype=torch.float16):
    """
    读取C++写入的二进制文件
    参数:
        filename: 文件名
        dtype: 数据类型 (默认为float32)
    """
    try:
        # 读取二进制数据到numpy数组

        data = torch.from_file(filename, dtype=dtype)
        print(f"成功读取 {len(data)} 个元素")
        return data
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None


def read_data_torch(file_path, data_dtype=torch.bfloat16):
    # 读取bin文件中的数据
    with open(file_path, 'rb') as f:
        data_bytes = f.read()

    # 确定数据量并转换为numpy数组
    # 假设你的bin文件是纯BF16数据，每个BF16数据占2个字节
    dtype = np.uint16  # 先用uint16来存储原始字节数据
    num_values = len(data_bytes) // 2  # 计算数据点的数量

    # 将字节数据转换为numpy数组
    np_data = np.frombuffer(data_bytes, dtype=dtype, count=num_values)

    # 将numpy数组转换为PyTorch张量（仍视为uint16）
    torch_uint16 = torch.from_numpy(np_data).clone()  # 使用.clone()确保数据可写

    # 关键步骤：通过view方法将uint16数据重新解释为bfloat16
    # 这不会改变底层的比特位，只是改变解释方式
    bf16_tensor = torch_uint16.view(data_dtype)

    # 如果你知道张量的形状，可以重新调整形状
    # 例如，如果你知道它原本是一个3x4的矩阵：
    # desired_shape = (3, 4)
    # bf16_tensor = bf16_tensor.reshape(desired_shape)

    # print("读取到的BF16张量:")
    # print(bf16_tensor)
    # print(f"张量形状: {bf16_tensor.shape}")
    # print(f"张量数据类型: {bf16_tensor.dtype}")

    return bf16_tensor


def func1():
    # ---------------- c++ 数据 ------------------ #
    file_path = f"/home/ubuntu/Downloads/InfiniLM-qwen2moe-test/layer_0_out.bin"
    data_array = read_binary_file(file_path, dtype=torch.bfloat16)
    data_array = torch.Tensor(data_array)
    print("file_path:: ", data_array.shape, data_array)

    # ---------------- python 数据 ------------------ #
    file_path = f"/home/ubuntu/workspace_aisys/tensorRT_quantization-main/Qwen/Qwen1.5-MoE-A2.7B_small/src/python_out1_5.pt"
    data_python = torch.load(file_path).cpu().reshape(-1)
    print(data_python)

    # ---------------- 比较 数据 ------------------ #
    diff = (data_array - data_python).abs()
    print("diff sort: ", torch.sort(diff)[0][-10:])
    print("diff sort: ", torch.sort(diff)[1][-10:])
    print("\n")


if __name__ == '__main__':


    # ---------------- c++ 数据 ------------------ #
    file_path = f"one_device.bin"
    data_array = read_data_torch(file_path, data_dtype=torch.bfloat16)
    data_array = data_array.reshape(-1)

    print("file_path:: ", data_array.shape, data_array)

    # ---------------- c++ 数据 ------------------ #
    file_path = f"two_device.bin"
    data_python = read_data_torch(file_path, data_dtype=torch.bfloat16)
    data_python = data_python.reshape(-1)
    print("file_path:: ", data_python.shape, data_python)


    # ---------------- 比较 数据 ------------------ #
    diff = (data_array - data_python).abs()
    print(diff)

    print("diff sort: ", torch.sort(diff)[0][-10:])
    print("diff sort: ", torch.sort(diff)[1][-10:])
    print("\n")
