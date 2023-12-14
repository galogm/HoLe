from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import f1_score as F1
from sklearn.metrics.cluster import contingency_matrix as ctg


def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = ctg(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix,
                          axis=0)) / np.sum(contingency_matrix)


def best_mapping(
    labels_true: list or np.array,
    labels_pred: list or np.array,
) -> Tuple[np.array, np.array]:
    """Get best mapping between labels_true and labels_pred.

    Args:
        labels_true (list or np.array): gnd labels.
        labels_pred (list or np.array): pred labels.

    Raises:
        ValueError: Labels must be in numpy format!

    Returns:
        Tuple[np.array,np.array]: best mapping.
    """
    if torch.is_tensor(labels_true) or torch.is_tensor(labels_pred):
        raise ValueError("Labels must be in numpy format!")
    D = max(labels_true.max(), labels_pred.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # pylint: disable=consider-using-enumerate
    for i in range(len(labels_pred)):
        w[labels_pred[i], labels_true[i]] += 1
    mapping = linear_assignment(w.max() - w)
    old_pred, new_pred = mapping
    label_map = dict(zip(old_pred, new_pred))
    labels_pred = [label_map[x] for x in labels_pred]
    return labels_true, labels_pred


def evaluation(
    labels_true: torch.Tensor or np.ndarray,
    labels_pred: torch.Tensor or np.ndarray,
) -> Tuple[float]:
    """Clustering evaluation.

    Args:
        labels_true (torch.Tensor or np.ndarray): Ground Truth Community.
        labels_pred (torch.Tensor or np.ndarray): Predicted Community.

    Returns:
        Tuple[float]: (ARI, NMI, ACC)

            - ARI: Adjusted Rand Index.

            - NMI: Normalized Mutual Informtion.

            - ACC: Accuracy.
    """
    if torch.is_tensor(labels_true):
        labels_true = labels_true.numpy().reshape(-1)
    if torch.is_tensor(labels_pred):
        labels_pred = labels_pred.numpy().reshape(-1)
    labels_true, labels_pred = best_mapping(labels_true, labels_pred)
    ARI_score = ARI(labels_true, labels_pred)
    NMI_score = NMI(labels_true, labels_pred)
    AMI_score = AMI(labels_true, labels_pred)
    ACC_score = ACC(labels_true, labels_pred)
    Micro_F1_score = F1(labels_true, labels_pred, average="micro")
    Macro_F1_score = F1(labels_true, labels_pred, average="macro")
    purity_score = purity(labels_true, labels_pred)
    return ARI_score, NMI_score, AMI_score, ACC_score, Micro_F1_score, Macro_F1_score, purity_score
