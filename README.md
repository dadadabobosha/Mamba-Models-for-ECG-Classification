# ECG Classification with Mamba Models: MambaBEAT and MAMCA

This repository focuses on evaluating the performance of two Mamba models—MambaBEAT and MAMCA—on ECG classification tasks using the CinC2017 and Icentia11k datasets. The classification is a binary task, specifically targeting the distinction between Normal and Atrial Fibrillation (AFIB) signals.

The Mamba models provide an efficient solution for handling long-sequence data, making them highly suitable for ECG classification. This project explores their effectiveness in real-world applications.


---
## 📜Model Sources

This project uses Mamba models sourced from their original repositories:
- MambaBEAT: https://github.com/SeroviICAI/Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT/tree/master
- MAMCA: https://github.com/ZhangYezhuo/MAMC/tree/main

We also introduced CNN model and Transformer model for the comparison of classification performance.
- CNN: https://github.com/mandrakedrink/ECG-Synthesis-and-Classification
- Transformer: https://github.com/emadeldeen24/ECGTransForm
Please refer to these repositories for the more implementation details and original design of the models.

---
## 📁Dataset

- CinC2017: https://physionet.org/content/challenge-2017/1.0.0/
- Icentia11k: https://physionet.org/content/icentia11k-continuous-ecg/1.0/

---
## 🛠️ Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project_name.git
   cd project_name
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Note: Install mamba-ssm library: The MAMCA model requires the mamba-ssm library. Follow the installation instructions at https://github.com/state-spaces/mamba.  This library is only supported on Linux systems.

---
## 📘Usage
1. Data Preparation:
    - Download the CinC2017 and Icentia11k datasets.
    - Preprocess the datasets using the scripts provided in the `data` directory.
2. Model Training:
    - To train the MambaBEAT model with CinC2017, run the following command:
      ```bash
      python3 -m src.train_binary.train_mambaBEAT_2017
      ```
    - To train the MAMCA model with CinC2017, run the following command:
      ```bash
        python3 -m src.train_binary.train_MAMCA_2017
        ```
    For the Icentia11k dataset, replace `2017` with `icentia11k` in the script names.
    - To train the CNN model, run the following command:
      ```bash
      python3 -m src.CNN.train
      ```
    - To train the Transformer model, run the following command:
      ```bash
      python3 -m src.train_binary.train_transformer
      ```
Note: Remember to adjust the paths in all files.

3. Model Evaluation:
  Two evaluation scripts are provided to evaluate the models on the testsets
