"""
CinC 2017 ECG Signal Resampling Script (resample1000.py)
============================================

This script processes electrocardiogram (ECG) signals stored in `.mat` files by **directly resampling**
them to a fixed target length of 1000 samples. Unlike the `padding18300samples.py` script, which
uses cyclic padding to ensure consistent signal lengths while preserving temporal patterns, this
script applies uniform resampling regardless of the original signal length.

### Key Difference:
- **resample000.py**: Directly resamples signals of varying lengths to **1000 samples**.
  This method is theoretically less mature because it may unevenly preserve information across
  recordings of different lengths.
- **padding18300samples.py**: Pads signals cyclically to a target length (e.g., 18300) to minimize
  loss of temporal features before any further processing.

### Functionality:
1. **Resampling**:
   - Uses `scipy.signal.resample` to scale the signal length to exactly 1000 samples.
   - This can lead to loss of information, particularly for recordings with significant variance
     in original lengths.

2. **Batch Processing**:
   - Iterates through all `.mat` files in the specified input directory.
   - Resampled signals are saved in `.mat` format to the specified output directory.

### How It Works:
1. Reads ECG signals from `.mat` files located in the `input_directory`.
2. Resamples each signal to 1000 samples, regardless of its original length.
3. Saves the processed signals to the `output_directory`.

### Notes:
- Use with caution for signals of highly variable lengths, as information loss may occur during
  aggressive downsampling.
"""








import os
import scipy.io
import numpy as np
import scipy.signal
from tqdm import tqdm


def resample_ecg_signal(signal, target_length=1000):
    """
    Resample the ECG signal to a specified target length.

    Args:
        signal (np.ndarray): The original ECG signal.
        target_length (int): The desired length after resampling.

    Returns:
        np.ndarray: The resampled ECG signal.
    """
    # Use scipy's resample function to change the length of the signal
    resampled_signal = scipy.signal.resample(signal, target_length)
    return resampled_signal


def process_and_save_ecg_files(input_dir, output_dir, target_length=1000):
    """
    Process all ECG files in the input directory, resample them to the target length, and save them to the output directory.

    Args:
        input_dir (str): Directory containing the original .mat ECG files.
        output_dir (str): Directory to save the processed .mat files.
        target_length (int): The desired length after resampling.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    processed_files_count = 0

    # Loop through all files in the input directory
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

                # Resample the signal to the target length
                resampled_signal = resample_ecg_signal(signal, target_length=target_length)

                # Generate a unique output filename to avoid overwriting
                output_filepath = os.path.join(output_dir, filename)

                # Save the resampled signal in the same format as the original file
                scipy.io.savemat(output_filepath, {'val': resampled_signal})

                processed_files_count += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"Processed and saved {processed_files_count} files out of {len(os.listdir(input_dir))} .mat files.")


if __name__ == "__main__":
    input_directory = "."  # Path to the original .mat files
    output_directory = "./resample9000/"  # Path to save the resampled .mat files
    target_length = 9000  # Desired length after resampling

    process_and_save_ecg_files(input_directory, output_directory, target_length)
