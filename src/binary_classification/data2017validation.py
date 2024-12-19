import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import os
import scipy.io
from typing import List, Dict
import matplotlib.pyplot as plt

class EKGDataset(Dataset):
    def __init__(self, X: List[str], features: List[int], y: List[List[str]], path: str) -> None:
        self._path = path
        self.X = X
        self.features = torch.tensor(features)
        self._encoder = MultiLabelBinarizer()
        self.y = torch.tensor(self._encoder.fit_transform(y), dtype=torch.float)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal = self.load_raw_data(index)
        signal_expanded = np.tile(signal, (1, 1)).T
        signal_expanded = signal_expanded / 1000  # Scale ADU to mV

        return torch.tensor(signal_expanded, dtype=torch.float), self.features[index], self.y[index]

    def load_raw_data(self, index: int):
        # filepath = self._path + self.X[index]
        filepath = os.path.join(self._path, self.X[index])


        mat_data = scipy.io.loadmat(filepath)

        return mat_data['val'].squeeze()

    def get_label(self, encoded: torch.Tensor) -> List[str]:
        return self._encoder.inverse_transform(encoded.unsqueeze(0))

def load_test_data(
    path: str,
    sampling_rate: int = 300,
    batch_size: int = 128,
    num_workers: int = 0,
    num_classes: int = 4
) -> DataLoader:
    if not os.path.isdir(f"{path}"):
        raise FileNotFoundError(f"Data path {path} does not exist.")

    if num_classes == 4:
        Y = pd.read_csv(os.path.join(path, "REFERENCE.csv"), header=None)
    elif num_classes == 2:
        Y = pd.read_csv(os.path.join(path, "REFERENCE_filtered.csv"), header=None)  # 2 classes balanced
    X = Y[0].apply(lambda x: x + ".mat").to_numpy()
    y = Y[1].apply(lambda x: [x]).to_list()

    features = np.zeros((len(X), 1))

    features_scaler = StandardScaler()
    features_scaler.fit(features)

    features_test = features_scaler.transform(features)

    test_dataset = EKGDataset(X, features_test, y, path)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_dataloader

def plot_test_ekg(dataloader: DataLoader, sampling_rate: int = 300, num_plots: int = 1) -> None:
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
    test_loader = load_test_data("../train/data/", batch_size=32)
    plot_test_ekg(test_loader)
