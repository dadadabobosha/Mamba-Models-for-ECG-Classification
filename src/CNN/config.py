import random
import numpy as np
import torch


class Config:
    csv_path = ''
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    attn_state_path = '../input/mitbih-with-synthetic/attn.pth'
    lstm_state_path = '../input/mitbih-with-synthetic/lstm.pth'
    cnn_state_path = '../input/mitbih-with-synthetic/cnn.pth'
    
    attn_logs = '../input/mitbih-with-synthetic/attn.csv'
    lstm_logs = '../input/mitbih-with-synthetic/lstm.csv'
    cnn_logs = '../input/mitbih-with-synthetic/cnn.csv'

    length = '1k'
    batch_size = 256
    data_dir = f"C:data/Icential11k_dataset/{length}"
    train_csv_path = f'{data_dir}/REFERENCE.csv'
    save_dir = 'C:ecg_classification/save_model/CNN/'

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
        
if __name__ == '__main__':        
    config = Config()
    seed_everything(config.seed)
