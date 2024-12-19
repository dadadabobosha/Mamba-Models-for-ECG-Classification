import os
import numpy as np


def read_npy_file(file_path):
    """
    读取单个 .npy 文件并打印其中的数字。

    Args:
        file_path (str): .npy 文件路径。
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    try:
        data = np.load(file_path)  # 读取 .npy 文件内容
        print(f"Data in {file_path}:\n{data}")
        print(f"Shape: {data.shape}, Data Type: {data.dtype}")
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")


def read_multiple_npy_files(folder_path):
    """
    批量读取文件夹中的 .npy 文件并打印每个文件的内容。

    Args:
        folder_path (str): 文件夹路径。
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    if not npy_files:
        print(f"No .npy files found in {folder_path}.")
        return

    for npy_file in npy_files:
        print(f"Reading file: {npy_file}")
        file_path = os.path.join(folder_path, npy_file)
        read_npy_file(file_path)


if __name__ == "__main__":
    # # 示例：读取单个文件
    # single_file_path = "example.npy"  # 替换为实际的 .npy 文件路径
    # read_npy_file(single_file_path)

    # 示例：读取文件夹中的多个 .npy 文件
    folder_path = "N"  # 替换为包含 .npy 文件的文件夹路径
    read_multiple_npy_files(folder_path)
