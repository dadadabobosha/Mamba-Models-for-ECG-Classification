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
from src.binary_classification.data2017_0815 import load_ekg_data
from src.utils.torchutils import set_seed, save_model
from src.modules.mamca_model import get_model  # 引入新的模型


import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import warnings

warnings.filterwarnings("ignore")


class EarlyStoppingHandler:
    def __init__(self, patience: int, delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "max" and score < self.best_score + self.delta) or (
            self.mode == "min" and score > self.best_score - self.delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                print(
                    f"Early stopping triggered after {self.patience} epochs with no improvement."
                )
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"device: {device}")
# set all seeds and set number of threads
set_seed(32)
torch.set_num_threads(8)
length = "CinC 2017_18300"  # CHANGE
# static variables
current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
DATA_PATH = os.path.abspath(os.path.join(current_folder, "..",  "data/training2017"))  # CHANGE
N_CLASSES: int = 2

# Configure logging to capture terminal output and write to a log file
logging.basicConfig(
    filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)
logger = logging.getLogger()


# Function to count the number of trainable parameters in a model
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    """
    Main training function.
    """
    # CHANGE: Adjusted hyperparameters
    epochs: int = 150
    lr: float = 1e-2  # Reduced learning rate
    batch_size: int = 16  # Reduced batch size
    step_size: int = 30  # More frequent LR adjustments (before 15)
    gamma: float = 0.9  # More aggressive LR decay (before 0.7)
    weight_decay: float = 1e-4  # Increased weight decay (before 1e-3)

    # Log the start time
    start_time = time.time()
    logger.info("Training started.")

    # Load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_ekg_data(DATA_PATH, batch_size=batch_size)

    # format data for mamca, because the input shape is [batch_size, seq_len, input_dim], for mamca, the input shape should be [batch_size, input_dim, seq_len]
    formatted_train_data = [
        (inputs.permute(0, 2, 1).to(device), targets.to(device))
        for inputs,_, targets in train_data
    ]
    formatted_val_data = [
        (inputs.permute(0, 2, 1).to(device), targets.to(device))
        for inputs,_, targets in val_data
    ]

    # Define name and writer
    name: str = "binary_MAMCA"
    patience = 150  # for early stopping
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")
    inputs: torch.Tensor = next(iter(train_data))[0]
    logger.info(f"inputs.shape: {inputs.shape}")

    # Define model
    input_length = inputs.size(1)  # sequence length
    input_channels = inputs.size(2)  # input channels
    model = get_model(input_length, num_classes=N_CLASSES, device=device)
    total_params = count_params(model)
    print(f"MAMCA Model parameters number: {total_params}")
    # print("Using fused_add_norm:", model.fused_add_norm)

    model = get_model(input_length, num_classes=N_CLASSES, device=device)

    # CHANGE: Changed loss function to include label smoothing
    # loss: torch.nn.Module = torch.nn.BCEWithLogitsLoss(label_smoothing=0.1) # Old version

    # CHANGE: for label smoothing effect, implement a variation of BCE loss
    class BCEWithLabelSmoothing(torch.nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        def forward(self, pred, target):
            smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
            return self.loss_fn(pred, smooth_target)

    # CHANGE: Use loss with 0.1 label smoothing
    loss: torch.nn.Module = BCEWithLabelSmoothing(smoothing=0.1)

    # CHANGE: Modified optimizer configuration
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )

    # CHANGE: Changed to ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=gamma, patience=5, verbose=True
    )

    # Initialize tracking for the best validation accuracy and training time
    best_val_accuracy = 0.0
    best_model_path = None
    total_train_time = 0.0

    # To store the loss, accuracy, and F1 for plotting later
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_time = []

    # Initialize early stopping
    early_stopping = EarlyStoppingHandler(patience=patience, mode="max")

    try:
        for epoch in tqdm(range(epochs)):
            epoch_start_time = time.time()

            # Train step
            train_loss, train_accuracy, train_f1 = train_step(
                # model, train_data, loss, optimizer, writer, epoch, device, logger
                model,
                formatted_train_data,
                loss,
                optimizer,
                writer,
                epoch,
                device,
                logger,
            )
            epoch_train_time = time.time() - epoch_start_time
            train_time.append(epoch_train_time)
            total_train_time += epoch_train_time

            print(
                f"Epoch {epoch} | Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f} | F1: {train_f1:.4f} | Train Time: {epoch_train_time:.4f} seconds"
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_f1_scores.append(train_f1)

            # Validation step
            val_loss, val_accuracy, val_f1 = val_step(
                model, formatted_val_data, loss, None, writer, epoch, device, logger
            )

            # CHANGE: Add scheduler step based on validation loss. Loss > Accuracy for more confident decisions
            scheduler.step(val_loss)
            # Alternative: scheduler.step(val_accuracy)

            # CHANGE: Add gradient clipping for stable learning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            print(
                f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f}"
            )
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

    # Log total training time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Training completed. Total time: {total_time:.4f} seconds.")
    print(f"Total training time (including validation): {total_time:.4f} seconds")
    print(f"Accumulated training time: {total_train_time:.4f} seconds")
    # print(
    #     f"epoch{epochs}, lr:{lr}, batch_size:{batch_size}, lr:{lr}, step_size:{step_size}, gamma:{gamma}, n_layers:{n_layers}, latent_state_dim:{latent_state_dim}, expand:{expand}, dt_rank:{dt_rank}, kernel_size:{kernel_size}, conv_bias:{conv_bias}, bias:{bias}, method:{method}, dropout:{dropout}")
    # Plotting loss, accuracy, and F1
    plot_metrics(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        train_f1_scores,
        val_f1_scores,
        current_time,
    )

    # create a DataFrame to store the training metrics
    df = pd.DataFrame(
        {
            "train_time": train_time,  #
            "train_losses": train_losses,  #
            "train_accuracies": train_accuracies,  #
            "train_f1_scores": train_f1_scores,  #
            "val_losses": val_losses,  #
            "val_accuracies": val_accuracies,
            "val_f1_scores": val_f1_scores,  #
        }
    )

    # save the DataFrame to a CSV file
    df.to_csv(f"./models/output_{name}_{length}.csv", index=False, header=True)

    return None


def plot_metrics(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    train_f1_scores,
    val_f1_scores,
    current_time,
    name="Mamca",
    N_CLASSES=2,
    lr=1e-2,
):
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
    plt.savefig(f"./models//seg_{current_time}_{name}_{length}.pdf")
    plt.savefig(f"./models//seg_{current_time}_{name}_{length}.svg")
    plt.show()


if __name__ == "__main__":
    main()
