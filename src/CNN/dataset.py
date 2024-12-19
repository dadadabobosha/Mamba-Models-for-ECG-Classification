import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple

from config import Config


# 引入 data_simple.py 中的 expand_to_12_leads 函数
def expand_to_12_leads(signal: torch.Tensor) -> torch.Tensor:
    return signal.expand(-1, 12)


class ECGDataset(Dataset):
    """
    读取ECG数据集，并根据文件路径加载信号，不进行扩展到12导联。
    """

    def __init__(self, file_paths: List[str], labels: List[str]) -> None:
        self.file_paths = file_paths
        self.labels = labels
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 从 .npy 文件中加载信号数据
        signal = np.load(self.file_paths[index])

        # 这里不扩展为12导联，保持单导联信号
        signal_tensor = torch.tensor(signal, dtype=torch.float).unsqueeze(0)  # Shape: [1, sequence_length]

        # 获取对应的标签
        label = self.y[index].long()
        return signal_tensor, label


def get_dataloader(phase: str, batch_size: int = 16, num_workers: int = 4) -> DataLoader:
    """
    创建ECG数据的DataLoader，支持病人不交叉分类。

    Args:
        phase (str): 训练或验证阶段 ('train' 或 'val')。
        batch_size (int): 每个batch的样本数。
        num_workers (int): DataLoader的工作线程数。

    Returns:
        DataLoader: 数据加载器。
    """
    # 读取CSV文件（假设路径是配置文件中给出的）
    df = pd.read_csv(Config.train_csv_path)
    df.columns = ['file_name', 'label']

    # 提取病人ID，假设病人ID是file_name的一部分
    df['patient_id'] = df['file_name'].apply(lambda x: x.split('_')[0])  # 提取病人ID

    # 只保留 'N' 和 'A' 标签的记录
    df = df[df['label'].isin(['N', 'AFIB'])]

    # 按病人ID划分训练集、验证集和测试集，确保病人不交叉
    unique_patients = df['patient_id'].unique()
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.3, random_state=Config.seed)
    val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=Config.seed)







    # 根据 phase 决定使用哪部分数据
    if phase == 'train':
        selected_patients = train_patients
    else:
        selected_patients = val_patients

    # 过滤出对应 phase 的数据
    df = df[df['patient_id'].isin(selected_patients)]

    # 创建文件路径列表和标签列表
    file_paths = [os.path.join(Config.data_dir, row['label'], row['file_name']) for _, row in df.iterrows()]
    labels = df['label'].apply(lambda x: 0 if x == 'N' else 1).tolist()  # 将 'N' 和 'A' 标签转化为0和1

    # 创建数据集
    dataset = ECGDataset(file_paths, labels)

    # 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dataloader


if __name__ == '__main__':
    train_dataloader = get_dataloader(phase='train', batch_size=16)
    val_dataloader = get_dataloader(phase='val', batch_size=16)

    # 检查数据加载器
    for signals, targets in train_dataloader:
        print(signals.shape, targets.shape)
        print(signals, targets)
        # break
