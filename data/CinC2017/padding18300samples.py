"""
CinC 2017 ECG Signal Preprocessing Script
================================

This script preprocesses electrocardiogram (ECG) signals stored in `.mat` files.
The preprocessing includes cyclic padding and resampling to standardize the length
of the signals, ensuring uniformity for downstream machine learning or analysis tasks.

### Key Features:
1. **Cyclic Padding**:
   - Extends the length of an ECG signal to a specified target length by repeating
     the original signal cyclically.
   - If the signal is longer than the target length, it trims the signal instead.

2. **Resampling**:
   - After padding, the signal is resampled to a desired length using Fourier-based
     resampling to maintain consistency across all input signals.
   -The original sample rate can be preserved with 18300.

3. **File Processing**:
   - Reads `.mat` files containing ECG signals from the specified input directory.
   - Saves the preprocessed signals into a new output directory in `.mat` format.

### Usage:
- Update `input_directory` with the path to the folder containing the raw `.mat` files.
- Update `output_directory` with the path where the processed files will be saved.
- Adjust `pad_length` and `resample_length` to define the target signal length for padding
  and resampling, respectively.
"""


import os
import scipy.io
import numpy as np
import scipy.signal
from tqdm import tqdm

def cyclic_pad(signal, target_length=18300):
    """
    Cyclically pad the ECG signal to reach the specified target length.

    Args:
        signal (np.ndarray): The original ECG signal.
        target_length (int): The target length after padding.

    Returns:
        np.ndarray: The cyclically padded ECG signal.
    """
    # Determine the amount of padding needed
    pad_length = target_length - len(signal)
    if pad_length <= 0:
        return signal[:target_length]  # Trim if signal is longer than target length

    # Perform cyclic padding
    padded_signal = np.concatenate([signal, np.tile(signal, pad_length // len(signal) + 1)[:pad_length]])
    return padded_signal


def resample_ecg_signal(signal, target_length=6000):
    """
    Resample the ECG signal to a specified target length.

    Args:
        signal (np.ndarray): The padded ECG signal.
        target_length (int): The desired length after resampling.

    Returns:
        np.ndarray: The resampled ECG signal.
    """
    resampled_signal = scipy.signal.resample(signal, target_length)
    return resampled_signal


def process_and_save_ecg_files(input_dir, output_dir, pad_length=18300, resample_length=6000):
    """
    Process all ECG files in the input directory, pad them to the specified length,
    resample to a target length, and save them to the output directory.

    Args:
        input_dir (str): Directory containing the original .mat ECG files.
        output_dir (str): Directory to save the processed .mat files.
        pad_length (int): Length to pad the signal to before resampling.
        resample_length (int): Length to resample the signal to.
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_files_count = 0

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".mat"):
            try:
                # Load the original ECG signal
                filepath = os.path.join(input_dir, filename)
                mat_data = scipy.io.loadmat(filepath)

                if 'val' not in mat_data:
                    print(f"Skipping {filename}: 'val' key not found.")
                    continue

                signal = mat_data['val'].squeeze()

                # Apply cyclic padding to reach pad_length
                padded_signal = cyclic_pad(signal, target_length=pad_length)

                # Resample the signal to the target length
                resampled_signal = resample_ecg_signal(padded_signal, target_length=resample_length)

                # Generate an output filename
                output_filepath = os.path.join(output_dir, filename)

                # Save the resampled signal
                scipy.io.savemat(output_filepath, {'val': resampled_signal})

                processed_files_count += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"Processed and saved {processed_files_count} files out of {len(os.listdir(input_dir))} .mat files.")


if __name__ == "__main__":
    input_directory = "."  # Path to the original .mat files
    output_directory = "./resampled18300_ecg_data/"  # Path to save the processed .mat files
    pad_length = 18300  # Target length for padding
    resample_length = 18300  # Target length after resampling

    process_and_save_ecg_files(input_directory, output_directory, pad_length, resample_length)