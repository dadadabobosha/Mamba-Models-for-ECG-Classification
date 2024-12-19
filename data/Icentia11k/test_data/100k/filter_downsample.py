import os
import numpy as np
import scipy.signal as signal
from tqdm import tqdm  # 用于进度条


def lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def downsample(data, target_length):
    return signal.resample(data, target_length)


def process_and_save_npy_files(input_dir, output_dir, cutoff_freq=30, sample_rate=250, target_length=5000):
    os.makedirs(output_dir, exist_ok=True)

    for subfolder in ["N", "AFIB"]:  # Process both N and AFIB folders
        input_subfolder = os.path.join(input_dir, subfolder)
        output_subfolder = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        # 获取文件列表并创建进度条
        files = [f for f in os.listdir(input_subfolder) if f.endswith(".npy")]
        with tqdm(total=len(files), desc=f"Processing {subfolder}", unit="file") as pbar:
            for filename in files:
                try:
                    # Load the .npy file
                    file_path = os.path.join(input_subfolder, filename)
                    data = np.load(file_path)

                    # Apply low-pass filter
                    filtered_data = lowpass_filter(data, cutoff_freq, sample_rate)

                    # Downsample to target length
                    downsampled_data = downsample(filtered_data, target_length)

                    # Save the processed data
                    output_file_path = os.path.join(output_subfolder, filename)
                    np.save(output_file_path, downsampled_data)

                    # Update the progress bar
                    pbar.update(1)

                except Exception as e:
                    print(f"Error processing file {filename} in {subfolder}: {e}")
                    pbar.update(1)


if __name__ == "__main__":
    input_directory = "/work/scratch/js54mumy/icentia11k/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0/seg_npy4/100k/"
    output_directory = "/work/scratch/js54mumy/icentia11k/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0/seg_npy4/downsample5k/"

    process_and_save_npy_files(input_directory, output_directory)
