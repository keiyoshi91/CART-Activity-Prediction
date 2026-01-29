import numpy as np


def precision_top5_and_top10(
    pred_vals: np.ndarray, valid_vals: np.ndarray, topk_rate: int = 0.25
) -> tuple:
    topk = round(len(valid_vals) * topk_rate)
    sorted_idx = np.argsort(-pred_vals)
    ranking = np.argsort(-valid_vals[sorted_idx])
    precision_top5 = np.sum(ranking[:5] < topk) / 5
    precision_top10 = np.sum(ranking[:10] < topk) / 10
    return precision_top5, precision_top10


def recall_top5_and_top10(
    pred_vals: np.ndarray, valid_vals: np.ndarray, topk_rate: int = 0.25
) -> tuple:
    topk = round(len(valid_vals) * topk_rate)
    sorted_idx = np.argsort(-pred_vals)
    ranking = np.argsort(-valid_vals[sorted_idx])
    recall_top5 = np.sum(ranking[:5] < topk) / topk
    recall_top10 = np.sum(ranking[:10] < topk) / topk
    return recall_top5, recall_top10


def precision_top1to10(
    pred_vals: np.ndarray, valid_vals: np.ndarray, topk_rate: int = 0.25
) -> tuple:
    topk = round(len(valid_vals) * topk_rate)
    sorted_idx = np.argsort(-pred_vals)
    ranking = np.argsort(-valid_vals[sorted_idx])
    precisions = []
    for k in range(1, 11):
        precision_top = np.sum(ranking[:k] < topk) / k
        precisions.append(precision_top)
    return precisions


def recall_top1to10(
    pred_vals: np.ndarray, valid_vals: np.ndarray, topk_rate: int = 0.25
) -> tuple:
    topk = round(len(valid_vals) * topk_rate)
    sorted_idx = np.argsort(-pred_vals)
    ranking = np.argsort(-valid_vals[sorted_idx])
    recalls = []
    for k in range(1, 11):
        recall_top = np.sum(ranking[:k] < topk) / topk
        recalls.append(recall_top)
    return recalls
