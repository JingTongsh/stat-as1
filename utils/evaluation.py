import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from typing import Tuple, List, Dict


def prec_recall_to_iou(p: float, r: float) -> float:
    if p * r == 0:
        return 0
    else:
        return 1 / (1 / p + 1 / r - 1)


def prec_recall_and_iou(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    iou = prec_recall_to_iou(precision, recall)
    return precision, recall, iou


def evaluate_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    label_names: Dict[int, str] = None,
    num_classes: int = None,
) -> Dict[str, float]:
    """
    Evaluate accuracy, precision, recall and iou (intersection over union).
    :param pred: unnormalized score, shape (N, K)
    :param gt: ground truth label, shape (N)
    :param label_names: maps label id (int) to label name (str)
    :return: metrics
    """
    
    if label_names is None:
        label_names = {k: str(k) for k in range(1, num_classes + 1)}
    else:
        num_classes = len(label_names)

    metrics = {}

    pred = pred.argmax(axis=1) + 1
    print(pred)
    acc = (pred == gt).astype(int).sum() / pred.shape[0]
    metrics['acc'] = acc
    for k, v in label_names.items():
        k_pred = (pred == k).astype(int)
        k_true = (gt == k).astype(int)
        precision, recall, iou = prec_recall_and_iou(k_pred, k_true)
        metrics[f'precision_{k}_{v}'] = precision
        metrics[f'recall_{k}_{v}'] = recall
        metrics[f'iou_{k}_{v}'] = iou

    return metrics


def check_eval():
    """
    Show metrics to see whether they are are correct.
    """
    num = 16
    n_classes = 3
    label_names = {k: str(k) for k in range(n_classes)}

    pred = F.softmax(torch.rand(num, n_classes) - 0.5, dim=1).numpy()
    gt = np.random.randint(0, n_classes, (num,)) + 1

    print(pred)
    print(gt)
    metrics = evaluate_metrics(pred, gt, label_names=label_names)
    print(metrics)


if __name__ == '__main__':
    check_eval()
