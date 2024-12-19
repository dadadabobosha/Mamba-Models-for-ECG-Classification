import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from typing import List, Tuple


class EKGDataset(Dataset):
    """
    This class represents the Icentia11k dataset, handling .npy files for segmented ECG data.
    """

    def __init__(self, file_paths: List[str], labels: List[str], normalize: bool = True) -> None:
        """
        Constructor of EKGDataset.

        Args:
            file_paths (List[str]): List of file paths for the ECG segments.
            labels (List[str]): The labels corresponding to the ECG segments.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.normalize = normalize
        # Use LabelBinarizer for binary classification (N and AFIB)
        self._encoder = LabelBinarizer()

        # Transform the labels to binary format (0 or 1)
        binary_labels = self._encoder.fit_transform(self.labels)

        # Convert to one-hot format: [1, 0] for N and [0, 1] for AFIB
        self.y = torch.tensor(np.eye(2)[binary_labels.flatten()], dtype=torch.float)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = np.load(self.file_paths[index])
        signal_tensor = torch.tensor(signal, dtype=torch.float).unsqueeze(-1)  # Shape: [sequence_length, 1]

        if self.normalize:
            # Scale to [-1, 1]
            signal_tensor = 2 * (signal_tensor - signal_tensor.min()) / (signal_tensor.max() - signal_tensor.min()) - 1

        label = self.y[index].float()
        return signal_tensor, label

    def get_label(self, encoded: torch.Tensor) -> List[str]:
        return self._encoder.inverse_transform(encoded.unsqueeze(0).numpy())


def load_ekg_data(
    path: str,
    batch_size: int = 128,
    drop_last: bool = False,
    num_classes: int = 2,
    num_workers: int = 1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Load the reference CSV
    df = pd.read_csv(os.path.join(path, "REFERENCE.csv"), header=None)
    df.columns = ['file_name', 'label']

    # Extract patient IDs from file names (assuming patient ID is the first part of the file name)
    df['patient_id'] = df['file_name'].apply(lambda x: x.split('_')[0])  # Assuming 'file_name' contains 'patientID-segmentID'

    # Split patient IDs into train, val, and test setsimport os

    unique_patients = df['patient_id'].unique()
    # Get corresponding labels for each patient
    patient_labels = [df[df['patient_id'] == pid]['label'].iloc[0] for pid in unique_patients]  # 每个 patient 的标签

    train_patients, test_patients = train_test_split(unique_patients, test_size=0.3, random_state=32,stratify=patient_labels)
    # Get the labels corresponding to test_patients for stratification
    test_patient_labels = [df[df['patient_id'] == pid]['label'].iloc[0] for pid in test_patients]
    val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=32,stratify=test_patient_labels)

    # Create train, val, and test dataframes based on patient IDs
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]

    # Create full file paths for each record
    train_file_paths = [os.path.join(path, row['label'], row['file_name']) for _, row in train_df.iterrows()]
    val_file_paths = [os.path.join(path, row['label'], row['file_name']) for _, row in val_df.iterrows()]
    test_file_paths = [os.path.join(path, row['label'], row['file_name']) for _, row in test_df.iterrows()]

    # Extract labels
    train_labels = train_df['label'].tolist()
    val_labels = val_df['label'].tolist()
    test_labels = test_df['label'].tolist()

    # Compute the sample weights for training data based on the labels' frequencies
    label_counts = pd.Series(train_labels).value_counts()
    class_weights = 1. / label_counts
    sample_weights = [class_weights[label] for label in train_labels]

    # Create WeightedRandomSampler for training data
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=False)
    # print(f"Sampler:{list(sampler)}")
    # Create datasets
    # No need to generate placeholder features, simplify data handling
    train_dataset = EKGDataset(train_file_paths, train_labels, normalize=True)
    val_dataset = EKGDataset(val_file_paths, val_labels, normalize=True)
    test_dataset = EKGDataset(test_file_paths, test_labels, normalize=True)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle for balanced sampling
        shuffle=False,  # No need to shuffle training
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,  # No need to shuffle validation data
        num_workers=num_workers,
        drop_last=drop_last,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    path = "C:\\data\\Icentia11k\\"

    # Load data
    train_loader, val_loader, test_loader = load_ekg_data(path, batch_size=128)

    count = 0

    # Example: Iterate through the dataloader to check if the data is loaded correctly
    for inputs, labels in train_loader:
        print(f"inputs.shape: {inputs.shape}, labels.shape: {labels.shape}")

        # Assuming labels are one-hot encoded, so we check the second dimension
        # [1, 0] for N and [0, 1] for AFIB
        n_count = (labels[:, 1] == 1).sum().item()  # Count how many have N [1,0]
        afib_count = (labels[:, 0] == 1).sum().item()  # Count how many have AFIB [0,1]

        print(f"Number of N labels: {n_count}")
        print(f"Number of AFIB labels: {afib_count}")

        # print(f"labels:{labels}")
        # print(inputs)
        count = count + 1
        if count == 5:
            break

