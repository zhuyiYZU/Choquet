import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .utils import batch_indices, classification_metrics


def _compact_text(value: object, max_chars: int = 300) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def majority_voting(agent_probs: np.ndarray) -> np.ndarray:
    agent_preds = np.argmax(agent_probs, axis=-1)
    preds = []
    for row, probs in zip(agent_preds, agent_probs):
        counts = np.bincount(row, minlength=probs.shape[-1])
        if len(set(counts.tolist())) == 1:
            preds.append(int(np.argmax(probs.mean(axis=0))))
        else:
            preds.append(int(np.argmax(counts)))
    return np.array(preds)


def average_probability_voting(agent_probs: np.ndarray) -> np.ndarray:
    return np.argmax(agent_probs.mean(axis=1), axis=-1)


@torch.no_grad()
def learned_dynamic_voting(model, df, use_pairwise: bool) -> np.ndarray:
    preds, _ = model.predict(df, use_pairwise=use_pairwise, return_details=False)
    return preds


def compare_methods(model, test_df) -> List[Dict[str, float]]:
    inputs = model.make_inputs(test_df)
    y_true = test_df["label"].values
    methods = [
        ("Majority Voting", majority_voting(inputs["agent_probs_np"])),
        ("Average Probability Voting", average_probability_voting(inputs["agent_probs_np"])),
        ("Dynamic Single-Agent Weighting", learned_dynamic_voting(model, test_df, use_pairwise=False)),
        ("Choquet-inspired Pairwise Voting", learned_dynamic_voting(model, test_df, use_pairwise=True)),
    ]
    rows = []
    for name, preds in methods:
        metrics = classification_metrics(y_true, preds)
        rows.append({"method": name, **metrics})
    return rows


def _prediction_error_type(gold_label: object, pred_label: int, correct: bool) -> str:
    if correct:
        return "correct"
    try:
        gold_int = int(gold_label)
    except (TypeError, ValueError):
        return "incorrect"
    if gold_int == 0 and pred_label == 1:
        return "false_positive"
    if gold_int == 1 and pred_label == 0:
        return "false_negative"
    return "incorrect"


@torch.no_grad()
def export_test_predictions(model, test_df, output_path: Path, batch_size: int, device: str) -> Path:
    """Export current model predictions for the test split as an Excel-friendly CSV."""
    if "label" not in test_df.columns:
        raise KeyError("test_df must contain a 'label' column to export gold_label.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.layer.eval()

    probs_chunks = []
    pred_chunks = []
    safe_batch_size = max(1, int(batch_size))
    for idx in batch_indices(len(test_df), safe_batch_size, shuffle=False):
        batch = test_df.iloc[idx]
        inputs = model.make_inputs(batch)
        logits = model.logits_from_inputs(inputs, use_pairwise=True, details=False)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs_chunks.append(probs)
        pred_chunks.append(np.argmax(probs, axis=-1).astype(int))

    num_classes = int(getattr(model.layer, "num_classes", 2))
    if probs_chunks:
        probs_np = np.vstack(probs_chunks)
        preds = np.concatenate(pred_chunks)
    else:
        probs_np = np.empty((0, num_classes), dtype=float)
        preds = np.array([], dtype=int)

    result_df = test_df.reset_index(drop=True).copy()
    gold_values = result_df["label"].to_numpy()
    correct_values = preds == gold_values
    error_types = [
        _prediction_error_type(gold, int(pred), bool(correct))
        for gold, pred, correct in zip(gold_values, preds, correct_values)
    ]

    pred_data = {
        "pred_label": preds.astype(int),
        "pred_label_name": [f"class_{pred}" for pred in preds],
    }
    for class_idx in range(num_classes):
        pred_data[f"pred_prob_{class_idx}"] = np.round(probs_np[:, class_idx], 6)
    pred_data.update(
        {
            "gold_label": gold_values,
            "correct": correct_values.astype(bool),
            "error_type": error_types,
        }
    )
    pred_cols = pd.DataFrame(pred_data)
    export_df = pd.concat([pred_cols, result_df], axis=1)
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


@torch.no_grad()
def aggregate_details_by_task(model, df) -> Dict[str, Dict[str, np.ndarray]]:
    result = {}
    for task_name, task_df in df.groupby("task_name"):
        preds, probs, details, _ = model.predict(task_df, use_pairwise=True, return_details=True)
        result[task_name] = {
            "single_weights": details["single_weights"].cpu().numpy().mean(axis=0),
            "pair_weights": details["pair_weights"].cpu().numpy().mean(axis=0),
            "preds": preds,
            "probs": probs,
        }
    return result


def print_interpretability_summary(model, df, top_pairs: int = 5) -> None:
    summary = aggregate_details_by_task(model, df)
    pair_names = [
        f"{model.agent_names[i]} + {model.agent_names[j]}" for i, j in model.layer.pair_indices
    ]

    print("\n=== Task-level Average Single Weights ===")
    for task_name, values in summary.items():
        print(f"\nTask: {task_name}")
        for name, weight in zip(model.agent_names, values["single_weights"]):
            print(f"  {name:12s}: {weight:.4f}")

    print("\n=== Task-level Top Pairwise Weights ===")
    for task_name, values in summary.items():
        order = np.argsort(values["pair_weights"])[::-1][:top_pairs]
        print(f"\nTask: {task_name}")
        for idx in order:
            print(f"  {pair_names[idx]:28s}: {values['pair_weights'][idx]:.4f}")


def print_sample_decisions(model, df, n_samples: int = 4, seed: int = 42) -> None:
    show_df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    preds, probs, details, inputs = model.predict(show_df, use_pairwise=True, return_details=True)
    single = details["single_weights"].cpu().numpy()
    pair = details["pair_weights"].cpu().numpy()
    pair_names = [
        f"{model.agent_names[i]} + {model.agent_names[j]}" for i, j in model.layer.pair_indices
    ]

    print("\n=== Sample-level Decision Traces ===")
    for row_idx, row in show_df.iterrows():
        print("\n--- Sample ---")
        print(f"Task: {row['task_name']}")
        print(f"Text: {_compact_text(row['text'])}")
        print(f"Gold label: {row['label']} | Final prediction: {preds[row_idx]} | Final probs: {probs[row_idx].round(4)}")
        print("Agent predictions:")
        for agent_idx, name in enumerate(model.agent_names):
            agent_prob = inputs["agent_probs_np"][row_idx, agent_idx]
            agent_pred = int(np.argmax(agent_prob))
            conf = inputs["agent_conf_np"][row_idx, agent_idx]
            explanation = inputs["explanations"][agent_idx][row_idx]
            print(
                f"  {name:12s} pred={agent_pred} conf={conf:.3f} "
                f"prob={agent_prob.round(3)} | {explanation}"
            )
        print("Single weights:")
        for name, weight in zip(model.agent_names, single[row_idx]):
            print(f"  {name:12s}: {weight:.4f}")
        top = np.argsort(pair[row_idx])[::-1][:3]
        print("Top pairwise weights:")
        for idx in top:
            print(f"  {pair_names[idx]:28s}: {pair[row_idx, idx]:.4f}")


def export_readable_model_summary(model, df, output_path: Path) -> None:
    """Export a UTF-8 JSON companion file for the binary PyTorch checkpoint.

    The .pt file is intentionally binary and will look garbled in text editors.
    This JSON is the human-readable artifact to inspect in PyCharm.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_summary = aggregate_details_by_task(model, df)
    pair_names = [
        f"{model.agent_names[i]} + {model.agent_names[j]}" for i, j in model.layer.pair_indices
    ]

    state = {
        key: value.detach().cpu().numpy().round(6).tolist()
        for key, value in model.layer.state_dict().items()
    }
    readable = {
        "note": (
            "best_choquet_model.pt is a binary PyTorch checkpoint. "
            "Open this JSON file for a readable UTF-8 summary."
        ),
        "choquet_mode": getattr(model, "choquet_mode", getattr(model.layer, "mode", "")),
        "choquet_ablation": getattr(model, "choquet_ablation", getattr(model.layer, "ablation_mode", "full")),
        "agent_names": model.agent_names,
        "pair_names": pair_names,
        "trainable_parameters": state,
        "task_level_average_weights": {},
    }

    for task_name, values in task_summary.items():
        single_map = {
            agent: round(float(weight), 6)
            for agent, weight in zip(model.agent_names, values["single_weights"])
        }
        pair_map = {
            pair: round(float(weight), 6)
            for pair, weight in zip(pair_names, values["pair_weights"])
        }
        top_pairs = dict(
            sorted(pair_map.items(), key=lambda item: item[1], reverse=True)[:5]
        )
        readable["task_level_average_weights"][task_name] = {
            "single_weights": single_map,
            "pair_weights": pair_map,
            "top_pair_weights": top_pairs,
        }

    output_path.write_text(
        json.dumps(readable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
