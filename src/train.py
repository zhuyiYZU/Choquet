from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import batch_indices, classification_metrics


def _validate_batch_shapes(model, logits: torch.Tensor, labels: torch.Tensor) -> None:
    num_classes = int(model.layer.num_classes)
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected logits [batch_size, {num_classes}], got {tuple(logits.shape)}."
        )
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Expected labels [batch_size], got {tuple(labels.shape)} for "
            f"logits {tuple(logits.shape)}."
        )
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(
            f"Labels must be in [0, {num_classes - 1}], got "
            f"min={int(labels.min())}, max={int(labels.max())}."
        )


def _run_epoch(model, df, optimizer, batch_size: int, device: str) -> float:
    model.layer.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for idx in batch_indices(len(df), batch_size, shuffle=True):
        batch = df.iloc[idx]
        labels = torch.tensor(batch["label"].values, dtype=torch.long, device=device)
        inputs = model.make_inputs(batch)
        logits = model.logits_from_inputs(inputs, use_pairwise=True, details=False)
        _validate_batch_shapes(model, logits, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_model(model, df, batch_size: int, device: str, use_pairwise: bool = True) -> Dict[str, float]:
    model.layer.eval()
    y_true, y_pred, losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    for idx in batch_indices(len(df), batch_size, shuffle=False):
        batch = df.iloc[idx]
        labels = torch.tensor(batch["label"].values, dtype=torch.long, device=device)
        inputs = model.make_inputs(batch)
        logits = model.logits_from_inputs(inputs, use_pairwise=use_pairwise, details=False)
        _validate_batch_shapes(model, logits, labels)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        y_true.extend(batch["label"].values.tolist())
        y_pred.extend(preds.tolist())
        losses.append(float(loss.detach().cpu()))
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_choquet_model(
    model,
    train_df,
    valid_df,
    model_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
) -> Tuple[object, Dict[str, float]]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.layer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_f1 = -1.0
    best_metrics = {}
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_df, optimizer, batch_size, device)
        valid_metrics = evaluate_model(model, valid_df, batch_size, device, use_pairwise=True)
        if valid_metrics["macro_f1"] > best_f1:
            best_f1 = valid_metrics["macro_f1"]
            best_metrics = valid_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
                f"valid_loss={valid_metrics['loss']:.4f} | "
                f"valid_acc={valid_metrics['accuracy']:.4f} | "
                f"valid_macro_f1={valid_metrics['macro_f1']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics
