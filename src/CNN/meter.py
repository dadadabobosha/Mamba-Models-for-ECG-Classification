import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


class Meter:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.init_metrics()
        self.confusion = torch.zeros((n_classes, n_classes))

    def update(self, y_pred, y_true, loss):
        y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
        y_true = y_true.detach().cpu().numpy()
        # print(f"Y_pred: {y_pred}")
        self.metrics['loss'].append(loss)
        self.metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        self.metrics['f1'].append(f1_score(y_true, y_pred, average='binary'))
        self.metrics['precision'].append(precision_score(y_true, y_pred, average='binary', zero_division=1))
        self.metrics['recall'].append(recall_score(y_true, y_pred, average='binary', zero_division=1))

        self._compute_cm(y_true, y_pred)

    def _compute_cm(self, y_true, y_pred):
        for t, p in zip(y_true, y_pred):
            self.confusion[t, p] += 1

    def init_metrics(self):
        self.metrics = {
            'loss': [],
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': []
        }

    def get_metrics(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}

    def get_confusion_matrix(self):
        return self.confusion