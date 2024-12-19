# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# other libraries
import time
import logging
from tqdm.auto import tqdm




# own modules
from src.utils.train_functions_1101 import train_step, val_step
from src.binary_classification.data_1lead_2017 import load_ekg_data
from src.utils.torchutils import set_seed, save_model
from src.modules.mamba_simple import MambaBEAT

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

class EarlyStoppingHandler:
    def __init__(self, patience: int, delta: float = 0.0, mode: str = 'max'):
        """
        Custom Early Stopping Handler

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change to qualify as an improvement.
            mode (str): 'max' or 'min'. Whether to maximize or minimize the monitored score.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score < self.best_score + self.delta) or \
                (self.mode == 'min' and score > self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"device:{device}")
# set all seeds and set number of threads
set_seed(32)
torch.set_num_threads(8)
length = '1k'
# static variables
current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
DATA_PATH = "C:data\\training2017\\resampled18300_ecg_data\\"
N_CLASSES: int = 2

# Configure logging to capture terminal output and write to a log file
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Function to count the number of trainable parameters in a model, mamba model only has several k parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # This is for loading a pre-trained model from the original MambaBEAT, because without this the training does not work.
# def load_model_or_weights(model_class, file_path, device):
#     loaded_obj = torch.load(file_path, map_location=device)
#
#     if isinstance(loaded_obj, dict):
#         model = model_class().to(device)  # Initialize model
#         model.load_state_dict(loaded_obj)
#         logger.info("Loaded model weights from state_dict.")
#     elif isinstance(loaded_obj, torch.nn.Module):
#         model = loaded_obj.to(device)
#         logger.info("Loaded entire model instance.")
#     else:
#         raise ValueError("Unsupported file format. Expected a state_dict or a model instance.")
#
#     return model

# Xavier Initialization Function
def xavier_initialize(model):
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv1d)):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

def main() -> None:
    """
    This function is the main program for the training.
    """
    # hyperparameters
    epochs: int = 60
    lr: float = 1e-2
    batch_size: int = 64
    step_size: int = 25
    gamma: float = 0.8

    # Mamba original Hyperparameters
    n_layers: int = 4
    latent_state_dim: int = 12
    expand: int = 2
    dt_rank: int = None
    kernel_size: int = 12
    conv_bias: bool = True
    bias: bool = False
    method: str = "zoh"
    dropout: float = 0.2



    # Log the start time
    start_time = time.time()
    # print(f"start_time:{start_time}")
    logger.info("Training started.")

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_ekg_data(DATA_PATH, batch_size=batch_size)

    # define name and writer
    name: str = "binary_MambaBEAT"
    patience = 50  # for early stopping
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")
    inputs: torch.Tensor = next(iter(train_data))[0]
    logger.info(f"inputs.shape: {inputs.shape}")

    # define model
    model: torch.nn.Module = (
        MambaBEAT(
            inputs.size(2),  # Updated to use the second dimension as input size, used to be 12, now it is 1
            N_CLASSES,
            n_layers,
            latent_state_dim,
            expand,
            dt_rank,
            kernel_size,
            conv_bias,
            bias,
            method,
            dropout,
        )
        .to(device)
        .float()
    )

    # # Load pre-trained weights (optional), This is for loading a pre-trained model from the original MambaBEAT, because without this the training does not work.
    # pretrained_path = "../benchmarks/binary_MambaBEAT.pth"
    # if os.path.exists(pretrained_path):
    #     logger.info(f"Loading pre-trained weights from {pretrained_path}")
    #     model = load_model_or_weights(MambaBEAT, pretrained_path, device).float()
    #     print(f"load pretrained model {pretrained_path}")
    #
    #     # Update classification layer if needed
    #     if model.linear.out_features != N_CLASSES:
    #         logger.info(f"Updating classification layer from {model.linear.out_features} to {N_CLASSES} classes.")
    #         model.linear = torch.nn.Linear(model.linear.in_features, N_CLASSES).to(device).float()

    total_params = count_params(model)
    print(f"Mamba Model parameters number: {total_params}")

    # Apply Xavier initialization
    xavier_initialize(model)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.BCELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # define an empty scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    # Initialize tracking for the best validation accuracy and training time
    best_val_accuracy = 0.0
    best_model_path = None  # To store the best model file path
    total_train_time = 0.0  # To accumulate the total training time across all epochs

    # To store the loss, accuracy, and F1 for plotting later
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_time = []

    # Initialize early stopping
    early_stopping = EarlyStoppingHandler(patience=patience, mode='max')

    try:
        for epoch in tqdm(range(epochs)):
            # Start timer for each epoch
            epoch_start_time = time.time()

            # Train step
            train_loss, train_accuracy, train_f1 = train_step(
                model, train_data, loss, optimizer, writer, epoch, device, logger
            )
            epoch_train_time = time.time() - epoch_start_time  # Calculate epoch train time
            train_time.append(epoch_train_time)  # Store the epoch train time
            total_train_time += epoch_train_time  # Accumulate the total training time

            print(f"Epoch {epoch} | Loss: {train_loss} | Accuracy: {train_accuracy} | F1: {train_f1} | Train Time: {epoch_train_time:.4f} seconds")
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_f1_scores.append(train_f1)

            # Validation step
            val_loss, val_accuracy, val_f1 = val_step(
                model, val_data, loss, scheduler, writer, epoch, device, logger
            )
            print(f"Epoch {epoch} | Val Loss: {val_loss} | Val Accuracy: {val_accuracy} | Val F1: {val_f1}")
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)

            # Check if the current model's validation accuracy is the best so far
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                # Delete the previous best model file if it exists
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                    logger.info(f"Removed previous best model: {best_model_path}")

                # Save the new best model
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = f"./models/best_model_{current_time}_{name}_{length}_{val_accuracy:.4f}.pth"
                save_model(model, best_model_path)
                logger.info(f"New best model saved with accuracy: {val_accuracy:.4f}")

            # Call EarlyStopping and pass in the validation accuracy to determine if training should stop
            early_stopping(val_accuracy)
            if early_stopping.early_stop:
                print(f"Stopping early at epoch {epoch} due to no improvement.")
                break

            torch.cuda.empty_cache()
        writer.close()

    except KeyboardInterrupt:
        pass

    # Log the end time and calculate total training time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Training completed. Total time: {total_time:.4f} seconds.")

    # Print total training time and accumulated train time
    print(f"Total training time (including validation): {total_time:.4f} seconds")
    print(f"Accumulated total training time (train_step only): {total_train_time:.4f} seconds")
    print(f"epoch{epochs}, lr:{lr}, batch_size:{batch_size}, lr:{lr}, step_size:{step_size}, gamma:{gamma}, n_layers:{n_layers}, latent_state_dim:{latent_state_dim}, expand:{expand}, dt_rank:{dt_rank}, kernel_size:{kernel_size}, conv_bias:{conv_bias}, bias:{bias}, method:{method}, dropout:{dropout}")
    # Plotting loss, accuracy, and F1
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, current_time)

    # create a DataFrame to store the training metrics
    df = pd.DataFrame({
        'train_time': train_time,  #
        'train_losses': train_losses,  #
        'train_accuracies': train_accuracies,  #
        'train_f1_scores': train_f1_scores,  #
        'val_losses': val_losses,  #
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores  #
    })

    # save the DataFrame to a CSV file
    df.to_csv(f'./models/output_{name}_{length}.csv', index=False, header=True)

    return None

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, current_time, name="MambaBEAT", N_CLASSES=2, lr=1e-2):
    """
    Plot the training and validation loss, accuracy, and F1-score on a single figure with three subplots.
    """
    plt.figure(figsize=(15, 5))

    # Subplot 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()

    # Subplot 2: Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()

    # Subplot 3: F1-Score
    plt.subplot(1, 3, 3)
    plt.plot(train_f1_scores, label="Train F1")
    plt.plot(val_f1_scores, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score")
    plt.title("F1-Score vs Epoch")
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"./models//seg_{current_time}_{name}_{length}.png")
    # plt.savefig(f"/work/home/js54mumy/Mamba/models/seg_{current_time}_{name}_{length}.pdf")
    # plt.savefig(f"/work/home/js54mumy/Mamba/models/seg_{current_time}_{name}_{length}.svg")
    plt.show()



if __name__ == "__main__":
    main()
