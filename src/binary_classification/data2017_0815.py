# deep learning libraries
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

# plotting libraries
import matplotlib.pyplot as plt

# other libraries
import ast
import os
import scipy.io
import requests
import shutil
from typing import List, Dict

pd.set_option("future.no_silent_downcasting", True)


class EKGDataset(Dataset):
    """
    This class represents the CinC2017 dataset, a dataset for the PhysioNet/Computing in Cardiology
    Challenge 2017 focused on AF classification from a short single lead ECG recording.

    When using this dataset in your work, please cite the following:

    - The original publication of the dataset:
        Clifford, G. D., Liu, C. Y., Moody, B., Li-wei, H. L., Silva, I., Li, Q., ... &
        Mark, R. G. (2017). AF classification from a short single lead ECG recording:
        the PhysioNet/Computing in Cardiology Challenge 2017. In 2017 Computing in Cardiology (CinC)
        (pp. 1-4). IEEE.
    """

    def __init__(self, X: List[str], features: List[int], y: List[List[str]], path: str) -> None:
        """
        Constructor of EKGDataset.

        Args:
            X (List[str]): The input data. Each row corresponds to the filename where the raw
            data is stored.

            features (List[int]): Additional patient data (here set to zero as a placeholder).

            y (List[List[str]]): The labels corresponding to the input data. Each element in the
            list is a list of strings, where each string is a diagnostic superclass for the
            corresponding EKG recording. The labels are binarized using a MultiLabelBinarizer to
            create a binary matrix indicating the presence of each diagnostic superclass
            for each EKG recording.

            path (str): the path where the data is stored.
        """

        self._path = path
        self.X = X
        self.features = torch.tensor(features)

        # Create a MultiLabelBinarizer object
        self._encoder = MultiLabelBinarizer()

        # Fit the encoder to the labels and transform the labels to binary format
        self.y = torch.tensor(self._encoder.fit_transform(y), dtype=torch.float)

        # # 打印类别顺序
        # print(self._encoder.classes_)

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            int: The number of EKG recordings in the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method loads an item based on the index.

        Args:
            index (int): The index of the element in the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the EKG
            recording and its corresponding labels. The EKG recording is expanded to 12 leads
            to simulate PTB-XL format, and the labels are a 1D tensor of binary values
            indicating the presence of each diagnostic superclass for the EKG recording.
        """
        signal = self.load_raw_data(index)
        # Expand single lead to simulate multi-lead format (12 leads)
        signal_expanded = np.tile(signal, (1, 1)).T

        # Convert ADU to mV using the 1000/mV scaling factor
        # Assuming the signal is already in ADU, convert to mV:
        signal_expanded = signal_expanded / 1000  # Scale ADU to mV

        # # Scale the amplitude to range [0, 2]
        # signal_min = np.min(signal_expanded)
        # signal_max = np.max(signal_expanded)
        # signal_expanded = 2 * (signal_expanded - signal_min) / (signal_max - signal_min)

        return torch.tensor(signal_expanded, dtype=torch.float), self.features[index], self.y[index]

    def load_raw_data(self, index: int):
        """
        Load raw data from a specified index.

        Args:
            index (int): The index where the data is stored.

        Returns:
            np.ndarray: The loaded raw data.
        """
        # filepath = self._path + self.X[index]
        filepath = os.path.join(self._path, self.X[index])
        mat_data = scipy.io.loadmat(filepath)
        # Assume the ECG data is stored under the 'val' key (common in CinC2017 dataset)
        return mat_data['val'].squeeze()

    def get_label(self, encoded: torch.Tensor) -> List[str]:
        """
        Recovers the label from the encoded vector.

        Args:
            encoded (torch.Tensor): the encoded label

        Returns:
            torch.Tensor: original label
        """
        return self._encoder.inverse_transform(encoded.unsqueeze(0))


def load_ekg_data(
    path: str,
    sampling_rate: int = 300,  # Default sampling rate is 300 Hz for CinC2017
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    num_classes: int = 2
):
    """
    Load EKG data, split it into train, validation, and test sets, and return dataloaders
    for each set.

    Args:
        path (str): The path where the data is stored.
        sampling_rate (int, optional): The sampling rate of the data. Defaults to 300.
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        num_workers (int, optional): The number of worker processes for data loading.
        Defaults to 0.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for the train, validation,
        and test sets.
    """
    if not os.path.isdir(f"{path}"):
        raise FileNotFoundError(f"Data path {path} does not exist.")

    # Load labels from REFERENCE.csv
    if num_classes == 4:
        Y = pd.read_csv(os.path.join(path, "REFERENCE.csv"), header=None)
    elif num_classes == 2:
        Y = pd.read_csv(os.path.join(path, "REFERENCE_balanced.csv"), header=None)    #2 classes balanced
    X = Y[0].apply(lambda x: x + ".mat").to_numpy()  # Append '.mat' extension to filenames
    y = Y[1].apply(lambda x: [x]).to_list()  # Convert labels to list of lists

    # In this example, we do not have additional patient features, so we'll use a placeholder
    features = np.zeros((len(X), 1))

    # Split the dataset into train, validation, and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_temp, features_train, features_temp, y_train, y_temp = train_test_split(
        X, features, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, features_val, features_test, y_val, y_test = train_test_split(
        X_temp, features_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Standardize the features (though they are all zeros here, this is to keep the format consistent)
    features_scaler = StandardScaler()
    features_scaler.fit(features_train)

    features_train = features_scaler.transform(features_train)
    features_val = features_scaler.transform(features_val)
    features_test = features_scaler.transform(features_test)

    # Create datasets
    train_dataset = EKGDataset(X_train, features_train, y_train, path)
    val_dataset = EKGDataset(X_val, features_val, y_val, path)
    test_dataset = EKGDataset(X_test, features_test, y_test, path)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return train_dataloader, val_dataloader, test_dataloader


def plot_ekg(
    dataloader: DataLoader, sampling_rate: int = 300, num_plots: int = 1
) -> None:
    """
    Plot EKG signals from a dataloader.

    Args:
        dataloader (DataLoader): The dataloader containing the EKG signals and labels.
        sampling_rate (int, optional): The sampling rate of the EKG signals. Defaults to 300.
        num_plots (int, optional): The number of EKG signals to plot. Defaults to 5.
    """
    ekg_signals, _, labels = next(iter(dataloader))

    color_major = (1, 0, 0)
    color_minor = (1, 0.7, 0.7)
    color_line = (0, 0, 0.7)

    for i in range(num_plots):
        signal = ekg_signals[i].numpy()

        fig, axes = plt.subplots(signal.shape[1], 1, figsize=(10, 10), sharex=True)

        for c in np.arange(signal.shape[1]):
            axes[c].grid(
                True, which="both", color=color_major, linestyle="-", linewidth=0.5
            )
            axes[c].minorticks_on()
            axes[c].grid(which="minor", linestyle=":", linewidth=0.5, color=color_minor)
            axes[c].plot(signal[:, c], color=color_line)

            if c < signal.shape[1] - 1:
                axes[c].set_xticklabels([])
            else:
                axes[c].set_xticks(np.arange(0, len(signal[:, c]), step=sampling_rate))
                axes[c].set_xticklabels(
                    np.arange(0, len(signal[:, c]) / sampling_rate, step=1)
                )

        plt.subplots_adjust(hspace=0.5)
        fig.text(0.04, 0.5, "Amplitude", va="center", rotation="vertical")
        axes[0].set_title(
            f"EKG Signal {i+1}, Label: {dataloader.dataset.get_label(labels[i])}"
        )
        plt.xlabel("Time (seconds)")
        plt.tight_layout(pad=4, w_pad=1.0, h_pad=0.1)
        plt.show()


if __name__ == "__main__":


    path = "C:\\data\\training2017\\"
    # Load data
    train_loader, val_loader, test_loader = load_ekg_data(path, batch_size=256)

    # Example: Iterate through the dataloader to check if the data is loaded correctly
    for inputs, features, labels in train_loader:
        print(f"inputs.shape: {inputs.shape}, features.shape: {features.shape}")
        # Assuming labels are one-hot encoded, so we check the second dimension
        # [1, 0] for N and [0, 1] for AFIB
        n_count = (labels[:, 0] == 1).sum().item()  # Count how many have N
        afib_count = (labels[:, 1] == 1).sum().item()  # Count how many have AFIB

        print(f"Number of N labels: {n_count}")
        print(f"Number of AFIB labels: {afib_count}")


