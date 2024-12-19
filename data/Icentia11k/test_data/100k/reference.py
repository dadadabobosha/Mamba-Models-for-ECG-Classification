import os
import pandas as pd


def create_combined_reference_csv(n_folder_path, afib_folder_path, output_csv_path):
    """
    根据N类和AFIB类文件夹中的文件生成一个CSV文件，包含文件名、标签和相对路径。

    参数:
    - n_folder_path: N类文件夹路径
    - afib_folder_path: AFIB类文件夹路径
    - output_csv_path: 输出CSV文件路径
    """
    # 创建空的列表来存储数据
    data = []

    # 处理N类文件夹
    for file_name in os.listdir(n_folder_path):
        if file_name.endswith('.npy'):
            relative_path = os.path.join("N", file_name)
            data.append([file_name, "N", relative_path])

    # 处理AFIB类文件夹
    for file_name in os.listdir(afib_folder_path):
        if file_name.endswith('.npy'):
            relative_path = os.path.join("AFIB", file_name)
            data.append([file_name, "AFIB", relative_path])

    # 创建DataFrame并保存为CSV文件
    df = pd.DataFrame(data, columns=['FileName', 'Label', 'RelativePath'])
    df.to_csv(output_csv_path, index=False)
    print(f"CSV文件已生成: {output_csv_path}")


# 主函数
if __name__ == "__main__":
    # 设置文件夹路径
    n_folder_path = "N"  # N类文件夹路径
    afib_folder_path = "AFIB"  # AFIB类文件夹路径

    # 设置输出CSV路径
    output_csv_path = "REFERENCE.csv"

    # 生成统一的CSV文件
    create_combined_reference_csv(n_folder_path, afib_folder_path, output_csv_path)
