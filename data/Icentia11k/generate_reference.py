"""
This script generates a `REFERENCE.csv` file for each experimental data group of different segment lengths.
The generated CSV file serves as a reference for the dataloader to efficiently load ECG data during experiments.

Main functionalities:
1. Gathers `.npy` files from specified directories for the "N" (Normal) and "AFIB" (Atrial Fibrillation) categories.
2. If the total number of files exceeds the specified limit (`max_total`), it performs proportional sampling.
3. Shuffles the data and saves it in a `REFERENCE.csv` file with columns for file names and their corresponding labels.

Usage:
- Specify the input directories for "N" and "AFIB" data.
- Set the output file name (`output_file`).
- Adjust the `max_total` parameter to limit the total number of samples if needed.

Output:
- A `REFERENCE.csv` file containing file names and their associated labels ("N" or "AFIB").
"""

import os
import random
import pandas as pd

def create_reference_csv(n_dir: str, afib_dir: str, output_file: str, max_total: int = 40000):
    # Gather all `.npy` files from the directories
    n_files = [f for f in os.listdir(n_dir) if f.endswith('.npy')]
    afib_files = [f for f in os.listdir(afib_dir) if f.endswith('.npy')]

    # Combine all files with their labels into a list
    files_with_labels = [(f, 'N') for f in n_files] + [(f, 'AFIB') for f in afib_files]

    # If the total number of files exceeds the limit, perform proportional sampling
    total_files = len(n_files) + len(afib_files)

    if total_files > max_total:
        # Calculate the ratio of "N" and "AFIB" files
        n_ratio = len(n_files) / total_files
        afib_ratio = len(afib_files) / total_files

        # Determine the number of samples to draw for each category
        n_sample_count = int(max_total * n_ratio)
        afib_sample_count = int(max_total * afib_ratio)

        # Perform sampling based on the calculated counts
        n_files = random.sample(n_files, n_sample_count)
        afib_files = random.sample(afib_files, afib_sample_count)

        # Update the combined list with sampled files
        files_with_labels = [(f, 'N') for f in n_files] + [(f, 'AFIB') for f in afib_files]

    # Shuffle the list of files with labels
    random.shuffle(files_with_labels)

    # Create a DataFrame with file names and labels
    df = pd.DataFrame(files_with_labels, columns=['file_name', 'label'])

    # Save the DataFrame as a CSV file
    df.to_csv(output_file, index=False, header=False)
    print(f"REFERENCE.csv created at {output_file}")

if __name__ == "__main__":
    n_dir = "N"
    afib_dir = "AFIB"
    output_file = "REFERENCE.csv"
    create_reference_csv(n_dir, afib_dir, output_file)
