"""
ECG Segment Length Analysis and Label Classification Script
===========================================================

This script processes ECG data stored in WFDB format, classifies segment lengths into predefined
categories, and identifies patients with "pure" labels (patients with only one annotation type).
The script analyzes ECG signals from multiple folders (e.g., p00 to p05) and generates the following:

1. **Segment Length Classification**:
   - Segments are classified into length categories:
     "1k-5k", "5k-10k", "10k-25k", "25k-50k", "50k-100k", and ">100k".

2. **Pure Label Detection**:
   - Patients are identified as having "pure" annotations if all segments contain the same label (e.g., "N", "AFIB", "AFL").

3. **Data Outputs**:
   - CSV files for patients with pure labels (`pure_N.csv`, `pure_AFIB.csv`, `pure_AFL.csv`).
   - CSV files for segment classifications, organized by label and segment length.
   - Bar plots showing the number of segments in each length category for each label.

### Workflow:
1. Traverse through subdirectories containing ECG `.dat` and `.atr` files.
2. Read WFDB records and annotations, calculate segment lengths, and classify them.
3. Detect pure-labeled patients and save their information.
4. Generate summary CSV files and visualizations.

"""

import os
import wfdb
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Function to classify segment lengths into predefined categories
def classify_by_length(segment_length):
    if 1000 < segment_length <= 5000:
        return "1k-5k"
    elif 5000 < segment_length <= 10000:
        return "5k-10k"
    elif 10000 < segment_length <= 25000:
        return "10k-25k"
    elif 25000 < segment_length <= 50000:
        return "25k-50k"
    elif 50000 < segment_length <= 100000:
        return "50k-100k"
    else:
        return ">100k"

# Function to analyze ECG records, segment lengths, and detect pure-labeled patients
def analyze_records(input_dir, pure_n_records, pure_afib_records, pure_afl_records, segment_length_dict):
    overall_labels = defaultdict(int)

    # Get all subdirectories and sort them in order
    sub_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for sub_dir in tqdm(sub_dirs, desc=f"Processing {input_dir}", unit="folder"):
        sub_dir_path = os.path.join(input_dir, sub_dir)

        label_set = set()  # Store unique labels for the patient
        patient_label = None  # Store the label if the patient is pure

        # Process each record in the subdirectory
        for record_number in sorted(os.listdir(sub_dir_path)):  # Sort to ensure numeric order
            if record_number.endswith(".dat"):
                try:
                    base_name = record_number[:10]
                    record_path = os.path.join(sub_dir_path, base_name)

                    # Check if both .dat and .atr files exist
                    if not os.path.exists(f"{record_path}.dat") or not os.path.exists(f"{record_path}.atr"):
                        print(f"Warning: Skipping incomplete record {record_number}")
                        continue

                    # Read the record and annotations
                    record = wfdb.rdrecord(record_path)
                    annotation = wfdb.rdann(record_path, 'atr')

                    # Iterate through all annotation labels
                    for i, label in enumerate(annotation.aux_note):
                        if label.startswith('(') and label != '(VF':  # Filter rhythm start labels, exclude '(VF'
                            cleaned_label = label.strip('()')  # Remove parentheses
                            label_set.add(cleaned_label)

                            # Determine segment start and end sample numbers
                            start_sample = annotation.sample[i]
                            end_sample = annotation.sample[i + 1] if i + 1 < len(annotation.sample) else len(record.p_signal)

                            segment_length = end_sample - start_sample
                            segment_class = classify_by_length(segment_length)

                            # Append the patient, record, and segment length to the corresponding category
                            segment_length_dict[cleaned_label][segment_class].append((sub_dir, segment_length))

                except Exception as e:
                    print(f"Error processing file {record_number}: {e}")

        # Determine if the patient has only one unique label (pure label)
        if len(label_set) == 1:
            patient_label = list(label_set)[0]

        # Classify the patient based on the single label
        if patient_label == "N":
            pure_n_records.append([sub_dir, "N"])
        elif patient_label == "AFIB":
            pure_afib_records.append([sub_dir, "AFIB"])
        elif patient_label == "AFL":
            pure_afl_records.append([sub_dir, "AFL"])

    return pure_n_records, pure_afib_records, pure_afl_records

# Function to plot bar charts showing segment counts for each label
def plot_segment_counts(segment_length_dict):
    labels = ['N', 'AFIB', 'AFL']
    length_classes = ["1k-5k", "5k-10k", "10k-25k", "25k-50k", "50k-100k", ">100k"]

    # Generate plots for each label
    for label in labels:
        counts = [len(segment_length_dict[label][length_class]) for length_class in length_classes]

        plt.figure(figsize=(10, 6))
        plt.bar(length_classes, counts, color='blue')
        plt.xlabel('Segment Length Class')
        plt.ylabel('Number of Segments')
        plt.title(f'Number of Segments for {label}')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot in multiple formats
        plt.savefig(f'{label}_segment_counts.png')
        plt.savefig(f'{label}_segment_counts.pdf')
        plt.savefig(f'{label}_segment_counts.svg')
        plt.close()  # Close the plot to free memory

        print(f"Saved {label}_segment_counts in .png, .pdf, and .svg formats.")

# Main function to process all folders and generate outputs
def process_all_folders(base_dir, pure_n_file, pure_afib_file, pure_afl_file):
    pure_n_records = []
    pure_afib_records = []
    pure_afl_records = []

    # Initialize dictionary for segment length classification
    segment_length_dict = {
        'N': defaultdict(list),
        'AFIB': defaultdict(list),
        'AFL': defaultdict(list),
    }

    # Process each folder (e.g., p00-p05)
    for folder in ['p00', 'p01', 'p02', 'p03', 'p04', 'p05']:
        input_directory = os.path.join(base_dir, folder)
        pure_n_records, pure_afib_records, pure_afl_records = analyze_records(
            input_directory, pure_n_records, pure_afib_records, pure_afl_records, segment_length_dict
        )

    # Save pure-labeled patient results to CSV files
    pd.DataFrame(pure_n_records, columns=["Patient", "Label"]).to_csv(pure_n_file, index=False)
    pd.DataFrame(pure_afib_records, columns=["Patient", "Label"]).to_csv(pure_afib_file, index=False)
    pd.DataFrame(pure_afl_records, columns=["Patient", "Label"]).to_csv(pure_afl_file, index=False)

    # Generate and save segment count bar charts
    plot_segment_counts(segment_length_dict)

    # Save segment classification results to CSV files
    for label in ['N', 'AFIB', 'AFL']:
        for length_class in segment_length_dict[label].keys():
            output_file = f"{label}_{length_class}.csv"
            df = pd.DataFrame(segment_length_dict[label][length_class], columns=["Patient", "Segment_Length"])
            df.to_csv(output_file, index=False)
            print(f"Saved {output_file} with {len(segment_length_dict[label][length_class])} records.")

    print(f"Finished processing all folders. Results saved in {pure_n_file}, {pure_afib_file}, {pure_afl_file}.")

if __name__ == "__main__":
    base_dir = "/work/scratch/js54mumy/icentia11k/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0"

    # Output files for pure labels
    pure_n_file = "pure_N.csv"
    pure_afib_file = "pure_AFIB.csv"
    pure_afl_file = "pure_AFL.csv"

    # Process all folders and segment lengths
    process_all_folders(base_dir, pure_n_file, pure_afib_file, pure_afl_file)
