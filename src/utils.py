import random
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_indices(n_items: int, batch_size: int, shuffle: bool = True) -> Iterable[np.ndarray]:
    indices = np.arange(n_items)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n_items, batch_size):
        yield indices[start : start + batch_size]


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "macro_f1": f1,
    }


def format_metrics_table(rows: List[Dict[str, float]]) -> str:
    headers = ["Method", "Accuracy", "Precision", "Recall", "Macro F1"]
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append(
            "| {method} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {macro_f1:.4f} |".format(
                **row
            )
        )
    return "\n".join([line, sep] + body)


def to_tensor(array, device="cpu", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)
