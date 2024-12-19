import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from config import Config, seed_everything
from dataset import ECGDataset, get_dataloader
from models import CNN, RNNModel, RNNAttentionModel

from meter import Meter


class Trainer:
    def __init__(self, net, lr, batch_size, num_epochs):
        self.net = net.to(Config.device)
        self.num_epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.dataloaders = {
            phase: get_dataloader(phase, batch_size) for phase in self.phases
        }
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()
        self.epoch_times = []  # 用来记录每个 train epoch 的用时

    def _train_epoch(self, phase, show_confusion_matrix=False):
        print(f"{phase} mode | time: {time.strftime('%H:%M:%S')}")

        if phase == 'train':
            self.net.train()
        else:
            self.net.eval()

        meter = Meter(n_classes=2)
        meter.init_metrics()

        data_loader = self.dataloaders[phase]
        print(f"{phase}")
        data_loader = tqdm(data_loader, desc=f"{phase} Epoch Progress", unit="batch", disable=True)

        for i, (data, target) in enumerate(data_loader):
            data = data.to(Config.device)
            target = target.to(Config.device)

            with torch.set_grad_enabled(phase == 'train'):
                output = self.net(data)
                loss = self.criterion(output, target)
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            meter.update(output, target, loss.item())

        metrics = meter.get_metrics()
        df_logs = pd.DataFrame([metrics])
        confusion_matrix = meter.get_confusion_matrix()

        if phase == 'train':
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0, ignore_index=True)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0, ignore_index=True)

        print('{}: {:.4f}, {}: {:.4f}, {}: {:.4f}, {}: {:.4f}, {}: {:.4f}'
              .format('Loss', metrics['loss'], 'Accuracy', metrics['accuracy'],
                      'F1', metrics['f1'], 'Precision', metrics['precision'],
                      'Recall', metrics['recall']))

        if show_confusion_matrix:
            labels = ['N', 'A']
            self.display_confusion_matrix(confusion_matrix, labels)

        return metrics['loss']

    def display_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_metrics(self):
        min_len = min(len(self.train_df_logs), len(self.val_df_logs))
        self.train_df_logs = self.train_df_logs[:min_len]
        self.val_df_logs = self.val_df_logs[:min_len]

        epochs = range(1, min(len(self.train_df_logs), len(self.val_df_logs)) + 1)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_df_logs['loss'], label='Train Loss')
        plt.plot(epochs, self.val_df_logs['loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_df_logs['accuracy'], label='Train Accuracy')
        plt.plot(epochs, self.val_df_logs['accuracy'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.train_df_logs['f1'], label='Train F1')
        plt.plot(epochs, self.val_df_logs['f1'], label='Val F1')
        plt.title('F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{Config.save_dir}CNN_training_plot_{Config.length}.png")
        plt.savefig(f"{Config.save_dir}CNN_training_plot_{Config.length}.pdf")
        plt.savefig(f"{Config.save_dir}CNN_training_plot_{Config.length}.svg")
        plt.show()

    def run(self):
        total_train_time = 0  # 总训练时间
        epoch_progress = tqdm(range(self.num_epochs), desc="Total Epoch Progress", unit="epoch")
        for epoch in epoch_progress:
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # 仅对 train 阶段进行计时
            start_time = time.time()  # 记录 train epoch 开始时间
            self._train_epoch(phase='train')
            train_epoch_time = time.time() - start_time  # 计算 train epoch 用时
            self.epoch_times.append(train_epoch_time)
            total_train_time += train_epoch_time

            print(f"Train Epoch {epoch + 1} took {train_epoch_time:.2f} seconds.")

            # 验证阶段不记录时间
            with torch.no_grad():
                val_loss = self._train_epoch(phase='val')
                self.scheduler.step()

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('\nNew checkpoint\n')
                torch.save(self.net.state_dict(), f"save_model/cnn_{Config.length}.pth")

        # 打印总的 train 用时
        print(f"\nTotal training time (train phases only): {total_train_time:.2f} seconds.")

        with torch.no_grad():
            self._train_epoch(phase='val', show_confusion_matrix=True)

        self.plot_metrics()

    def save_logs(self):
        # 确保训练时间列表长度与训练日志相同
        if len(self.epoch_times) == len(self.train_df_logs):
            # 将训练时长作为新列加入 train_logs
            self.train_df_logs['time'] = self.epoch_times
        else:
            print(
                f"Warning: Mismatch in length of epoch_times ({len(self.epoch_times)}) and train_df_logs ({len(self.train_df_logs)})")

        # 合并 train_logs 和 val_logs
        train_logs = self.train_df_logs
        val_logs = self.val_df_logs

        if not train_logs.empty:
            train_logs.columns = ["train_" + colname for colname in train_logs.columns]
        if not val_logs.empty:
            val_logs.columns = ["val_" + colname for colname in val_logs.columns]

        logs = pd.concat([train_logs, val_logs], axis=1)
        logs.reset_index(drop=True, inplace=True)

        # 选择需要保存的列
        logs = logs.loc[:, [
                               'train_loss', 'val_loss',
                               'train_accuracy', 'val_accuracy',
                               'train_f1', 'val_f1',
                               'train_precision', 'val_precision',
                               'train_recall', 'val_recall',
                               'train_time'  # 新加入的训练时间列
                           ]]

        # 打印前几行，确保正确性
        print(logs.head())

        # 保存为 CSV 文件
        logs.to_csv(f'{Config.save_dir}cnn_{Config.length}.csv', index=False)


if __name__ == '__main__':
    config = Config()
    seed_everything(config.seed)

    model = CNN(num_classes=2, hid_size=128)

    trainer = Trainer(net=model, lr=1e-2, batch_size=256, num_epochs=2)
    trainer.run()

    # 保存日志，包括每个epoch的训练时长
    trainer.save_logs()
