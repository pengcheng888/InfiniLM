import numpy as np
import torch


def read_binary_file(filename, dtype=np.float16):
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

if __name__ == '__main__':
    # ---------------- c++ 数据 ------------------ #
    file_path = f"0_o_ok.bin"
    data_array0 = read_binary_file(file_path, dtype=np.float16)
    data_array0 = torch.Tensor(data_array0)
    print("file_path:: ", data_array0.shape, data_array0)

    # ---------------- c++ 数据 ------------------ #
    file_path = f"0_o.bin"
    data_array = read_binary_file(file_path, dtype=np.float16)
    data_array = torch.Tensor(data_array)
    print("file_path:: ", data_array.shape, data_array)

    # ---------------- c++ 数据 ------------------ #
    file_path = f"0_temp.bin"
    data_array_attn_val_gemm = read_binary_file(file_path, dtype=np.float16)
    data_array_attn_val_gemm = torch.Tensor(data_array_attn_val_gemm)
    print("file_path:: ", data_array_attn_val_gemm.shape, data_array_attn_val_gemm)

    exit()
    # ---------------- c++ 数据 ------------------ #
    file_path = f"1_o.bin"
    data_array2 = read_binary_file(file_path, dtype=np.float16)
    data_array2 = torch.Tensor(data_array2)
    print("file_path:: ", data_array2.shape, data_array2)

    # ---------------- 比较 数据 ------------------ #
    diff = (data_array0 - data_array[:69632]).abs()
    print("diff sort: ", torch.sort(diff)[0][-10:] )
    print("diff sort: ", torch.sort(diff)[1][-10:] )
    print("\n")
