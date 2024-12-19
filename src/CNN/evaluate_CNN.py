import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from models import CNN  # Assuming CNN model is defined in models.py
from config import Config  # Assuming Config is defined in config.py
from dataset import get_dataloader  # Using validation dataloader logic from the Trainer
import time


# Load the model
def load_model(model_path, num_classes=2, hid_size=128):
    """
    Load a pre-trained CNN model.

    Args:
        model_path (str): Path to the saved model file.
        num_classes (int): Number of output classes for the model.
        hid_size (int): Hidden size of the CNN model.

    Returns:
        torch.nn.Module: Loaded CNN model.
    """
    model = CNN(num_classes=num_classes, hid_size=hid_size)
    model.load_state_dict(torch.load(model_path, map_location=Config.device))
    model.to(Config.device)
    model.eval()
    return model


# Evaluate the model and compute throughput
def evaluate_model(model, dataloader):
    """
    Evaluate the CNN model on the given dataset and calculate throughput.

    Args:
        model (torch.nn.Module): The CNN model to be evaluated.
        dataloader (DataLoader): Dataloader for the validation dataset.

    Returns:
        tuple: Accuracy, confusion matrix, F1 score, precision, recall, and throughput.
    """
    all_preds = []
    all_targets = []

    total_records = 0
    total_inference_time = 0.0

    with torch.no_grad():
        for data, target in dataloader:
            total_records += data.size(0)

            start_time = time.time()  # Record the start time
            data = data.to(Config.device)
            target = target.to(Config.device)

            output = model(data)
            preds = torch.argmax(output, dim=1)
            end_time = time.time()  # Record the end time

            total_inference_time += (end_time - start_time)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='binary')
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')

    # Calculate throughput
    throughput = total_records / total_inference_time if total_inference_time > 0 else 0.0

    return accuracy, cm, f1, precision, recall, throughput


# Display and plot the confusion matrix
def display_confusion_matrix(cm, labels):
    """
    Display the confusion matrix in a tabular format and plot a heatmap.

    Args:
        cm (ndarray): Confusion matrix.
        labels (list): List of class labels.
    """
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    # Specify the length of the dataset
    length = '1k'
    # Path to the pre-trained model
    model_path = rf'/work/home/js54mumy/CNN/ecg_classification/save_model/CNN/{length}/cnn_{length}.pth'

    # Load validation dataloader from the training script
    val_dataloader = get_dataloader(phase='val', batch_size=256)  # Set batch_size=256

    # Load the model
    model = load_model(model_path)

    # Evaluate the model
    accuracy, cm, f1, precision, recall, throughput = evaluate_model(model, val_dataloader)

    # Output the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Inference Throughput: {throughput:.2f} records/second")

    # Print and display the confusion matrix
    labels = ['N', 'A']
    display_confusion_matrix(cm, labels)
