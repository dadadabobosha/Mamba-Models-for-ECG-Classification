"""
This script processes ECG signals in the WFDB format and segments them based on annotations.
Each segment is either padded cyclically or divided into overlapping windows to achieve a
uniform target length. The processed segments are saved as `.npy` files for further analysis.

Main functionalities:
1. Cyclic padding: Short segments are cyclically padded to the target length.
2. Overlapping windows: Long segments are divided into overlapping windows of the target length.
3. Handles segments based on annotations and saves them with appropriate labels.

Usage:
- Update `input_directory` and `output_directory` paths as needed.
- Specify the desired `target_length` for the processed segments.
- This version removes the `max_samples` limit and processes all available segments.

Outputs:
- `.npy` files for each processed segment saved in the specified output directory.
"""

import os
import wfdb
import numpy as np
from tqdm import tqdm


def cyclic_padding(segment, target_length):
    """
    Apply cyclic padding to a segment to extend it to the target length.

    Args:
        segment (np.ndarray): The input signal segment.
        target_length (int): The desired length.

    Returns:
        np.ndarray: The cyclically padded segment.
    """
    segment_length = len(segment)
    if segment_length >= target_length:
        return segment[:target_length]

    padded_segment = np.zeros(target_length)
    repeats = target_length // segment_length
    remainder = target_length % segment_length

    padded_segment[:repeats * segment_length] = np.tile(segment, repeats)
    padded_segment[repeats * segment_length:] = segment[:remainder]

    return padded_segment


def process_and_save_ecg_files(input_dir, output_dir, target_length):
    """
    Process and save ECG files with cyclic padding or segmentation.

    Args:
        input_dir (str): Path to the input directory containing .dat files.
        output_dir (str): Path to the output directory.
        target_length (int): Target segment length.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    processed_files_count = 0
    total_files = len([f for f in os.listdir(input_dir) if f.endswith(".dat")])

    # Main file processing progress bar
    with tqdm(total=total_files, desc="Processing Files", unit="file") as pbar:
        for record_number in os.listdir(input_dir):
            if record_number.endswith(".dat"):
                try:
                    base_name = record_number[:10]
                    record_path = os.path.join(input_dir, base_name)

                    record = wfdb.rdrecord(record_path)
                    annotation = wfdb.rdann(record_path, 'atr')

                    signal = record.p_signal[:, 0]
                    segments = []
                    start_positions = []
                    labels = []

                    # Parse the annotations
                    for i, label in enumerate(annotation.aux_note):
                        if label.startswith('('):
                            start_positions.append(annotation.sample[i])
                            labels.append(label)
                        elif label == ')':
                            if start_positions:
                                start = start_positions.pop(0)
                                segment = signal[start:annotation.sample[i]]
                                segments.append((segment, labels.pop(0)))

                    # Process each segment
                    for idx, (segment, label) in enumerate(
                        tqdm(segments, desc=f"Processing Segments [{base_name}]", leave=False)
                    ):
                        segment_length = len(segment)
                        label = label.strip('()')  # Remove parentheses
                        label = ''.join(c for c in label if c.isalnum() or c == '-')  # Keep alphanumeric and dashes only

                        # Handle segments within [target_length/10, target_length]
                        if target_length // 10 <= segment_length <= target_length:
                            padded_segment = cyclic_padding(segment, target_length)
                            output_filename = f"{base_name}-{label}-seg{idx}.npy"
                            output_filepath = os.path.join(output_dir, output_filename)
                            np.save(output_filepath, padded_segment)

                        # Handle segments within [target_length, 10 * target_length]
                        elif target_length < segment_length <= 10 * target_length:
                            for start_idx in range(0, segment_length - target_length + 1, target_length):
                                window = segment[start_idx:start_idx + target_length]
                                output_filename = f"{base_name}-{label}-seg{idx}-{start_idx}.npy"
                                output_filepath = os.path.join(output_dir, output_filename)
                                np.save(output_filepath, window)

                    processed_files_count += 1

                except Exception as e:
                    print(f"Error processing file {record_number}: {e}")

                pbar.update(1)  # Update the outer progress bar

    print(f"Processed and saved {processed_files_count} records out of {total_files} .dat files.")


if __name__ == "__main__":
    target_length = 100000  # Set the desired segment length

    #here the lista and listb should be changed into the pure patient number from the CSV file
    lista = [11, 12, 13, 14]
    for i in lista:
        input_directory = f"old_p00/p000{i}"
        output_directory = f"overlap/100k/N/"
        process_and_save_ecg_files(input_directory, output_directory, target_length)

    listb = [60, 65, 75, 91]
    for i in listb:
        input_directory = f"old_p00/p000{i}"
        output_directory = f"overlap/100k/AFIB/"
        process_and_save_ecg_files(input_directory, output_directory, target_length)
