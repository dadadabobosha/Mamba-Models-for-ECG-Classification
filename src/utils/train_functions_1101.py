# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from sklearn.metrics import accuracy_score, f1_score
from typing import Optional


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    logger=None,
) -> None:
    """
    This function trains the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
        logger: logger for logging training info (optional).
    """
    model.train()
    losses = []
    all_outputs = []
    all_targets = []

    for inputs, targets in train_data:
        inputs, targets = inputs.to(device), targets.to(device)
        model = model.to(device).float()

        # Forward pass
        outputs = model(inputs)

        # Store outputs and targets for later metric calculation
        all_outputs.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())

        # Compute loss
        loss_value = loss(outputs, targets)

        optimizer.zero_grad()
        loss_value.backward()

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()
        losses.append(loss_value.item())

    # Aggregate all predictions and calculate metrics
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Convert model outputs to prediction labels
    all_outputs_labels = np.argmax(all_outputs, axis=1)
    all_targets_labels = np.argmax(all_targets, axis=1)

    avg_loss = np.mean(losses)
    avg_accuracy = accuracy_score(all_targets_labels, all_outputs_labels)
    f1 = f1_score(all_targets_labels, all_outputs_labels)

    if logger:
        logger.info(
            f"Epoch {epoch} | Loss: {avg_loss} | Accuracy: {avg_accuracy} | F1: {f1}"
        )

    # Write metrics to TensorBoard
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", avg_accuracy, epoch)
    writer.add_scalar("train/f1_score", f1, epoch)

    return avg_loss, avg_accuracy, f1


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    logger=None,
) -> None:
    """
    This function evaluates the model on validation data.

    Args:
        model: model to evaluate.
        val_data: dataloader of validation data.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the validation.
        device: device for running operations.
        logger: logger for logging validation info (optional).
    """
    model.eval()
    losses = []
    all_outputs = []
    all_targets = []

    for inputs, targets in val_data:
        inputs, targets = inputs.to(device), targets.to(device)
        model = model.to(device).float()

        # Forward pass
        outputs = model(inputs)

        # Store outputs and targets for later metric calculation
        all_outputs.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())

        # Compute loss
        loss_value = loss(outputs, targets)
        losses.append(loss_value.item())

    # Aggregate all predictions and calculate metrics
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Convert model outputs to prediction labels
    all_outputs_labels = np.argmax(all_outputs, axis=1)
    all_targets_labels = np.argmax(all_targets, axis=1)

    avg_loss = np.mean(losses)
    avg_accuracy = accuracy_score(all_targets_labels, all_outputs_labels)
    f1 = f1_score(all_targets_labels, all_outputs_labels)

    if logger:
        logger.info(f"Epoch {epoch} | Loss: {avg_loss} | Accuracy: {avg_accuracy} | F1: {f1}")

    # Write metrics to TensorBoard
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", avg_accuracy, epoch)
    writer.add_scalar("val/f1_score", f1, epoch)

    if scheduler is not None:
        scheduler.step()

    return avg_loss, avg_accuracy, f1
