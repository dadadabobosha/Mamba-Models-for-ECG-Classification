import torch
torch.use_deterministic_algorithms(True, warn_only=False)
from torch.utils.tensorboard import SummaryWriter


# other libraries
import time
import logging
from tqdm.auto import tqdm


# own modules
from src.utils.train_functions_1101 import train_step, val_step
from src.binary_classification.data_1lead import load_ekg_data
from src.utils.torchutils import set_seed, save_model
from src.modules.transformer import ecgTransForm

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import warnings
warnings.filterwarnings("ignore", message="adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation")


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")
set_seed(32)
torch.set_num_threads(8)
length = '1k'
# static variables
current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
DATA_PATH = os.path.abspath(os.path.join(current_folder, "..", "..", "test_data", f"{length}")) # test data path in local
# DATA_PATH: str = f"/work/home/js54mumy/Mamba/dataset/Icential11k_dataset/{length}/"  # data path in Lichtenberg
N_CLASSES: int = 2

# Configure logging to capture terminal output and write to a log file
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class icentia11k_config:
    def __init__(self):
        self.num_classes = 2
        self.class_names = ['N', 'AFIB']
        self.sequence_len = 1000
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.dropout = 0.2
        self.mid_channels = 32
        self.final_out_channels = 128
        self.trans_dim = 128
        self.num_heads = 4




def main() -> None:
    epochs = 500
    lr = 1e-4
    batch_size = 128
    step_size = 25
    gamma = 0.8

    # Initialize config and model
    config = icentia11k_config()
    model = ecgTransForm(
        input_channels=config.input_channels,
        mid_channels=config.mid_channels,
        final_out_channels=config.final_out_channels,
        trans_dim=config.trans_dim,
        num_classes=config.num_classes,
        num_heads=config.num_heads,
        dropout=config.dropout
    ).to(device)

    total_params = count_params(model)
    print(f"Transformer Model parameters number: {total_params}")

    # Log the start time
    start_time = time.time()
    # print(f"start_time:{start_time}")
    logger.info("Training started.")

    # load data
    train_data, val_data, _ = load_ekg_data(DATA_PATH, batch_size=batch_size, num_classes=config.num_classes)

    # format data for Transformer, because the input shape is [batch_size, seq_len, input_dim], for Transformer, the input shape should be [batch_size, input_dim, seq_len]
    formatted_train_data = [
        (inputs.permute(0, 2, 1).to(device), targets.to(device))
        for inputs, targets in train_data
    ]
    formatted_val_data = [
        (inputs.permute(0, 2, 1).to(device), targets.to(device))
        for inputs, targets in val_data
    ]

    name: str = "binary_Transformer"
    patience = 50  # for early stopping
     # define loss and optimizer
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    writer: SummaryWriter = SummaryWriter(f"runs/{name}")
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
                model, formatted_train_data, loss, optimizer, writer, epoch, device, logger
            )
            epoch_train_time = time.time() - epoch_start_time  # Calculate epoch train time
            train_time.append(epoch_train_time)  # Store the epoch train time
            total_train_time += epoch_train_time  # Accumulate the total training time

            print(f"Epoch {epoch} | Loss: {train_loss} | Accuracy: {train_accuracy} | F1: {train_f1} | Train Time: {epoch_train_time:.4f} seconds", flush=True)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_f1_scores.append(train_f1)

            # Validation step
            val_loss, val_accuracy, val_f1 = val_step(
                model, formatted_val_data, loss, scheduler, writer, epoch, device, logger
            )
            print(f"Epoch {epoch} | Val Loss: {val_loss} | Val Accuracy: {val_accuracy} | Val F1: {val_f1}", flush=True)
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
            print(f"Epoch {epoch} | EarlyStopping Counter: {early_stopping.counter}")
            if early_stopping.early_stop:
                print(f"Stopping early at epoch {epoch} due to no improvement.")
                break

            torch.cuda.empty_cache()
        writer.close()

    except KeyboardInterrupt:
        pass

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Training completed. Total time: {total_time:.4f} seconds.")

    # Print total training time and accumulated train time
    print(f"Total training time (including validation): {total_time:.4f} seconds")
    print(f"Accumulated total training time (train_step only): {total_train_time:.4f} seconds")
    print(f"epoch{epochs}, lr:{lr}, batch_size:{batch_size}, lr:{lr}, step_size:{step_size}, gamma:{gamma}")

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

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, current_time, name="Transformer", num_classes=2, lr=1e-2):
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
