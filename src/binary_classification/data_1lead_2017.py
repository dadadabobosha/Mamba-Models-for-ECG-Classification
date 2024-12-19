import os
import scipy.io
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class EKGDataset(Dataset):
    """
    This class represents the CinC2017 dataset, handling .mat files for segmented ECG data.
    """

    def __init__(self, X: List[str], y: List[List[str]], path: str, normalize: bool = True) -> None:
        """
        Constructor of EKGDataset for .mat files.

        Args:
            X (List[str]): List of file names where the raw data is stored.
            y (List[List[str]]): The labels corresponding to each ECG recording.
            path (str): Path where the .mat files are stored.
            normalize (bool): Whether to normalize the signal to [-1, 1].
        """
        self._path = path
        self.X = X
        self.normalize = normalize

        # Use MultiLabelBinarizer for encoding labels
        self._encoder = MultiLabelBinarizer()
        self.y = torch.tensor(self._encoder.fit_transform(y), dtype=torch.float)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load signal data
        filepath = os.path.join(self._path, self.X[index])
        mat_data = scipy.io.loadmat(filepath)
        signal = mat_data['val'].squeeze()
        signal_tensor = torch.tensor(signal, dtype=torch.float).unsqueeze(-1)

        # Normalize signal to [-1, 1] if required
        if self.normalize:
            signal_tensor = 2 * (signal_tensor - signal_tensor.min()) / (signal_tensor.max() - signal_tensor.min()) - 1

        label = self.y[index]
        return signal_tensor, label

    def get_label(self, encoded: torch.Tensor) -> List[str]:
        return self._encoder.inverse_transform(encoded.unsqueeze(0))


def load_ekg_data(
    path: str,
    batch_size: int = 128,
    drop_last: bool = False,
    num_workers: int = 1,
    num_classes: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load EKG data, split into train, val, and test sets, and create dataloaders.
    Args:
        path (str): Path to the dataset.
        batch_size (int): Batch size for DataLoader.
        drop_last (bool): Whether to drop the last batch if incomplete.
        num_workers (int): Number of workers for data loading.
        num_classes (int): Number of classes for classification.
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for train, validation, and test sets.
    """
    # Load labels from REFERENCE.csv
    label_file = "REFERENCE_balanced.csv" if num_classes == 2 else "REFERENCE.csv"
    df = pd.read_csv(os.path.join(path, label_file), header=None)
    df.columns = ['file_name', 'label']
    df['file_path'] = df['file_name'].apply(lambda x: os.path.join(path, f"{x}.mat"))
    X = df['file_path'].to_numpy()
    y = df['label'].apply(lambda x: [x]).to_list()

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Create datasets
    train_dataset = EKGDataset(X_train, y_train, path)
    val_dataset = EKGDataset(X_val, y_val, path)
    test_dataset = EKGDataset(X_test, y_test, path)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    path = "C:\\data\\CinC2017\\training2017\\"
    train_loader, val_loader, test_loader = load_ekg_data(path, batch_size=256)

    # Check data loading
    for inputs, labels in train_loader:
        print(f"inputs.shape: {inputs.shape}")
        print(f"labels.shape: {labels.shape}")
        n_count = (labels[:, 0] == 1).sum().item()  # Count N labels
        afib_count = (labels[:, 1] == 1).sum().item()  # Count AFIB labels

        print(f"Number of N labels: {n_count}")
        print(f"Number of AFIB labels: {afib_count}")
        break
