from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch
from torch import nn
import numpy as np

def compute_metrics(logits, labels, threshold=0.5):
    # apply sigmoid on predictions
    sigmoid = nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits.detach().cpu()))
    print(probs)
    # use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # compute metrics
    y_true = labels.detach().cpu()
    print(y_true)
    f1_micro_avg = f1_score(y_true, y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {'f1': f1_micro_avg,
                'roc_auc': roc_auc,
                'accuracy': accuracy}

    return metrics