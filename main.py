import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from config import (
    AGENT_BACKEND,
    BEST_MODEL_PATH,
    BATCH_SIZE,
    CHOQUET_ABLATION,
    CHOQUET_MODE,
    DATA_AUTOGENERATE_DEMO,
    DATA_PATH,
    DEVICE,
    DISABLED_AGENTS,
    EPOCHS,
    ENABLED_AGENTS,
    LEARNING_RATE,
    LLM_API_KEY_ENV,
    LLM_BASE_URL,
    LLM_CACHE_ENABLED,
    LLM_CACHE_PATH,
    LLM_MODEL,
    LLM_MODEL_CANDIDATES,
    LLM_PROVIDER,
    MODEL_DIR,
    MODEL_SUMMARY_PATH,
    RANDOM_SEED,
    RUN_SAMPLE_LIMIT,
    RUNS_DIR,
    TRAIN_RATIO,
    VALID_RATIO,
    WEIGHT_DECAY,
    has_llm_api_key,
)
from integrity_checker import (
    check_dataset_integrity,
    print_training_integrity_summary,
    raise_if_incomplete,
)
from src.dataset import infer_dataset_num_classes, load_dataset, split_dataframe
from src.evaluate import (
    compare_methods,
    export_readable_model_summary,
    export_test_predictions,
    print_interpretability_summary,
    print_sample_decisions,
)
from src.model import MultiAgentChoquetModel
from src.cache import LLMCache
from src.task_schemas import get_task_label_schema
from src.train import evaluate_model, train_choquet_model
from src.utils import format_metrics_table, set_seed


VALID_BACKENDS = {"rule", "llm", "hybrid"}
VALID_CHOQUET_MODES = {"inspired", "discrete_2additive"}
VALID_CHOQUET_ABLATIONS = {
    "full",
    "no_pairwise",
    "fixed_singleton",
    "fixed_capacity",
    "no_logit_calibration",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def _has_api_key() -> bool:
    return has_llm_api_key()

def _looks_like_inline_api_key(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(re.match(r"^(sk-|AIza|eyJ)[A-Za-z0-9_.-]{12,}$", text))


def _display_api_key_env() -> str:
    if _looks_like_inline_api_key(LLM_API_KEY_ENV):
        return "REDACTED_KEY_ENV_VALUE"
    return LLM_API_KEY_ENV


def _resolve_device() -> str:
    requested = (DEVICE or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested DEVICE={DEVICE!r}, but CUDA is not available. Falling back to CPU.")
        return "cpu"
    return requested


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _shorten_model_name(model_name: str) -> str:
    """Shorten model name for folder naming.

    Examples:
        deepseek-v4-flash -> ds-flash
        gpt-5.5 -> gpt-5.5
        gemini-3.5-flash -> gemini-3.5-flash
        claude-sonnet-4-6 -> claude-4-6
    """
    model_lower = model_name.lower().strip()
    if model_lower.startswith("deepseek"):
        # deepseek-v4-flash -> ds-flash, deepseek-v4-pro -> ds-pro
        parts = model_lower.split("-")
        if len(parts) >= 2:
            version = parts[1] if len(parts) >= 2 else ""
            variant = parts[-1] if len(parts) >= 3 else ""
            return f"ds-{variant}" if variant else f"ds-{version}"
    elif model_lower.startswith("claude"):
        # claude-sonnet-4-6 -> claude-4-6
        parts = model_lower.split("-")
        if len(parts) >= 3:
            return f"claude-{parts[-2]}-{parts[-1]}"
    # For other models, keep as is but remove extra hyphens
    return model_lower.replace("_", "-")


def _get_dataset_name(data_path: Path) -> str:
    """Extract dataset name from path.

    Examples:
        data/raw_data/clickbait/SCC.csv -> SCC
        data/toy_data.csv -> toy_data
    """
    stem = data_path.stem  # filename without extension
    return _safe_slug(stem)


def _create_run_dir():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in RUNS_DIR.iterdir():
        if path.is_dir():
            prefix = path.name.split("_", 1)[0]
            if prefix.isdigit():
                existing.append(int(prefix))
    run_index = max(existing, default=0) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # name = (
    #     f"{run_index:04d}_{timestamp}_{_safe_slug(AGENT_BACKEND)}_"
    #     f"{_safe_slug(CHOQUET_MODE)}_{_safe_slug(LLM_MODEL)}"
    # )
    # New naming format: {index}_{timestamp}_{dataset}_seed{seed}_{short_model}
    dataset_name = _get_dataset_name(DATA_PATH)
    short_model = _shorten_model_name(LLM_MODEL)
    name = f"{run_index:04d}_{timestamp}_{dataset_name}_seed{RANDOM_SEED}_{short_model}"

    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_run_json(run_dir, name: str, payload: dict) -> None:
    (run_dir / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class _TeeStream:
    """Write console output to both the original stream and a run log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())


def _start_console_capture(run_dir) -> None:
    """Mirror all following stdout/stderr output into run_dir/console_out.txt."""
    log_file = (run_dir / "console_out.txt").open("w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)


def _save_baseline_comparison(rows, run_dir, table_text: str | None = None) -> dict:
    """Persist structured baseline metrics for this run."""
    json_path = run_dir / "baseline_comparison.json"
    txt_path = run_dir / "baseline_comparison.txt"
    _write_run_json(run_dir, json_path.name, {"rows": rows})
    if table_text is not None:
        txt_path.write_text(table_text + "\n", encoding="utf-8")
    return {"json": str(json_path), "txt": str(txt_path)}


def _runtime_snapshot(effective_device: str, extra: dict | None = None) -> dict:
    data = {
        "random_seed": RANDOM_SEED,
        "data_path": str(DATA_PATH),
        "agent_backend_requested": AGENT_BACKEND,
        "enabled_agents": list(ENABLED_AGENTS),
        "disabled_agents": list(DISABLED_AGENTS),
        "choquet_mode": CHOQUET_MODE,
        "choquet_ablation": CHOQUET_ABLATION,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_model_candidates": LLM_MODEL_CANDIDATES,
        "llm_base_url": LLM_BASE_URL,
        "llm_api_key_env": _display_api_key_env(),
        "api_key_detected": _has_api_key(),
        "llm_cache_enabled": LLM_CACHE_ENABLED,
        "llm_cache_path": str(LLM_CACHE_PATH),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "device_requested": DEVICE,
        "device_effective": effective_device,
        "cuda_available": torch.cuda.is_available(),
        "run_sample_limit": RUN_SAMPLE_LIMIT,
    }
    if extra:
        data.update(extra)
    return data


def _print_runtime_config(run_dir, effective_device: str) -> None:
    print("=== Runtime Configuration ===")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"AGENT_BACKEND: {AGENT_BACKEND}")
    print(f"Enabled agents: {', '.join(ENABLED_AGENTS)}")
    if DISABLED_AGENTS:
        print(f"Disabled agents: {', '.join(DISABLED_AGENTS)}")
    print(f"CHOQUET_MODE: {CHOQUET_MODE}")
    print(f"CHOQUET_ABLATION: {CHOQUET_ABLATION}")
    print(f"DEVICE requested/effective: {DEVICE} / {effective_device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"LLM_MODEL: {LLM_MODEL}")
    print(f"LLM_MODEL_CANDIDATES: {', '.join(LLM_MODEL_CANDIDATES)}")
    print(f"LLM_BASE_URL: {LLM_BASE_URL}")
    print(f"LLM_API_KEY_ENV: {_display_api_key_env()}")
    print(f"API key detected: {_has_api_key()}")
    print(f"LLM_CACHE_ENABLED: {LLM_CACHE_ENABLED}")
    print(f"LLM cache path: {LLM_CACHE_PATH}")
    print(f"Run output dir: {run_dir}")


def _runtime_agent_backend() -> str:
    backend = AGENT_BACKEND.strip().lower()
    if backend not in VALID_BACKENDS:
        print(f"Unsupported AGENT_BACKEND={AGENT_BACKEND!r}. Use rule, llm, or hybrid.")
        print("Falling back to AGENT_BACKEND=rule for this run.")
        return "rule"

    needs_key = backend in {"llm", "hybrid"} and LLM_PROVIDER != "local"
    if needs_key and not _has_api_key():
        print(
            f"AGENT_BACKEND={backend} requires environment variable {_display_api_key_env()}, "
            "but it is not set."
        )
        if backend == "llm":
            print("LLM mode requires a working API key or a complete precomputed cache. Exiting before training.")
            return "missing_llm_key"
        print("Hybrid mode will use rule fallback for this run.")
        return "rule"
    return backend


def _validate_choquet_mode() -> str:
    mode = CHOQUET_MODE.strip().lower()
    if mode not in VALID_CHOQUET_MODES:
        print(f"Unsupported CHOQUET_MODE={CHOQUET_MODE!r}. Falling back to inspired.")
        return "inspired"
    return mode


def _validate_choquet_ablation() -> str:
    ablation = CHOQUET_ABLATION.strip().lower()
    if ablation not in VALID_CHOQUET_ABLATIONS:
        print(f"Unsupported CHOQUET_ABLATION={CHOQUET_ABLATION!r}. Falling back to full.")
        return "full"
    return ablation


def _ensure_cache_file() -> None:
    if LLM_CACHE_ENABLED and not LLM_CACHE_PATH.exists():
        LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LLM_CACHE_PATH.write_text("{}", encoding="utf-8")


def _smoke_test_llm_agents(model: MultiAgentChoquetModel, df) -> tuple[bool, list[str]]:
    print("\n=== LLM Agent Smoke Test ===")
    sample = df.iloc[0]
    text = str(sample["text"])
    task_name = sample.get("task_name")
    task_description = str(sample["task_description"])
    label_schema = get_task_label_schema(task_name, task_description)
    failures = []
    for agent in model.agents:
        try:
            output = agent.predict_one(text, task_description, label_schema)
        except Exception as exc:
            failures.append(f"{agent.name}: {exc}")
            print(f"{agent.name:12s} FAILED error={exc}")
            continue
        probs = output["probs"]
        conf = float(output["confidence"])
        explanation = str(output["explanation"])
        print(
            f"{agent.name:12s} probs={[round(float(x), 4) for x in probs]} "
            f"confidence={conf:.3f} explanation={explanation}"
        )
        if getattr(agent, "last_used_fallback", False):
            error = getattr(agent, "last_error", "unknown LLM failure")
            failures.append(f"{agent.name}: {error}")
        if len(probs) != model.layer.num_classes:
            failures.append(
                f"{agent.name}: expected {model.layer.num_classes} probabilities, "
                f"got {len(probs)}"
            )
    if failures:
        print("Smoke test detected LLM failures:")
        for item in failures:
            print(f"  - {item}")
        return False, failures
    print("Smoke test passed: all LLM agents returned parseable JSON.")
    return True, []


def _copy_latest_artifacts(run_model_path, run_summary_path) -> None:
    if run_model_path.exists():
        shutil.copy2(run_model_path, BEST_MODEL_PATH)
    if run_summary_path.exists():
        shutil.copy2(run_summary_path, MODEL_SUMMARY_PATH)


@torch.no_grad()
def _choquet_parameter_diagnostics(model) -> dict:
    layer = model.layer
    diagnostics = {
        "choquet_mode": getattr(layer, "mode", ""),
        "choquet_ablation": getattr(layer, "ablation_mode", "full"),
        "logit_scale": float(torch.clamp(layer.logit_scale.detach(), min=0.1, max=20.0).cpu()),
        "raw_logit_scale": float(layer.logit_scale.detach().cpu()),
        "logit_bias": layer.logit_bias.detach().cpu().numpy().round(8).tolist(),
    }
    if getattr(layer, "mode", "") != "discrete_2additive":
        return diagnostics

    single_m, pair_m = layer._mobius_parameters()
    raw_pair_m = layer.raw_pair_m.detach()
    diagnostics.update(
        {
            "single_m": single_m.detach().cpu().numpy().round(8).tolist(),
            "pair_m": pair_m.detach().cpu().numpy().round(8).tolist(),
            "max_abs_pair_m": float(pair_m.detach().abs().max().cpu()) if pair_m.numel() else 0.0,
            "mean_abs_pair_m": float(pair_m.detach().abs().mean().cpu()) if pair_m.numel() else 0.0,
            "raw_pair_m_min": float(raw_pair_m.min().cpu()) if raw_pair_m.numel() else 0.0,
            "raw_pair_m_max": float(raw_pair_m.max().cpu()) if raw_pair_m.numel() else 0.0,
            "raw_pair_m_mean": float(raw_pair_m.mean().cpu()) if raw_pair_m.numel() else 0.0,
            "discrete_pair_scale": float(layer.discrete_pair_scale),
        }
    )
    return diagnostics


def _save_choquet_parameter_diagnostics(model, run_dir) -> dict:
    diagnostics = _choquet_parameter_diagnostics(model)
    _write_run_json(run_dir, "choquet_parameter_diagnostics.json", diagnostics)
    print("\n=== Choquet Parameter Diagnostics ===")
    print(f"choquet_mode: {diagnostics.get('choquet_mode')}")
    print(f"choquet_ablation: {diagnostics.get('choquet_ablation')}")
    if diagnostics.get("choquet_mode") == "discrete_2additive":
        print(f"single_m: {diagnostics.get('single_m')}")
        print(f"max_abs_pair_m: {diagnostics.get('max_abs_pair_m'):.8f}")
        print(f"mean_abs_pair_m: {diagnostics.get('mean_abs_pair_m'):.8f}")
        print(
            "raw_pair_m min/max/mean: "
            f"{diagnostics.get('raw_pair_m_min'):.8f} / "
            f"{diagnostics.get('raw_pair_m_max'):.8f} / "
            f"{diagnostics.get('raw_pair_m_mean'):.8f}"
        )
        print(f"discrete_pair_scale: {diagnostics.get('discrete_pair_scale')}")
    print(f"logit_scale: {diagnostics.get('logit_scale'):.8f}")
    print(f"logit_bias: {diagnostics.get('logit_bias')}")
    print(f"Saved diagnostics to: {run_dir / 'choquet_parameter_diagnostics.json'}")
    return diagnostics


def _failed_cache_records() -> int:
    cache = LLMCache(LLM_CACHE_PATH)
    return sum(1 for item in cache._items.values() if isinstance(item, dict) and item.get("status") == "FAILED")


def _run_auto_retry_failed_cache() -> int:
    script = LLM_CACHE_PATH.parent.parent / "scripts" / "retry_failed_cache.py"
    print("\nAuto retry started...")
    result = subprocess.run([sys.executable, str(script)])
    print(f"Auto retry exit code: {result.returncode}")
    return int(result.returncode)


def _ensure_dataset_integrity_or_retry(df, model) -> dict:
    print("\n=== Dataset Integrity Check ===")
    integrity = check_dataset_integrity(df, model.agents, LLM_CACHE_PATH)
    print_training_integrity_summary(integrity)
    if int(integrity["missing_records"]) <= 0:
        return integrity

    failed_records = _failed_cache_records()
    print("\nDataset integrity failed.")
    print(f"FAILED cache records: {failed_records}")
    if failed_records <= 0:
        print("Training blocked.")
        raise_if_incomplete(integrity)

    _run_auto_retry_failed_cache()

    print("\n=== Dataset Integrity Recheck ===")
    integrity = check_dataset_integrity(df, model.agents, LLM_CACHE_PATH)
    print_training_integrity_summary(integrity)
    if int(integrity["missing_records"]) > 0:
        print("Training blocked.")
        raise_if_incomplete(integrity)
    return integrity


def main():
    set_seed(RANDOM_SEED)
    if len(ENABLED_AGENTS) < 2:
        print("At least two agents must be enabled for ensemble/Choquet aggregation.")
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_cache_file()
    effective_device = _resolve_device()
    choquet_mode = _validate_choquet_mode()
    choquet_ablation = _validate_choquet_ablation()
    run_dir = _create_run_dir()
    run_model_path = run_dir / "best_choquet_model.pt"
    run_summary_path = run_dir / "model_summary.json"
    _start_console_capture(run_dir)
    _print_runtime_config(run_dir, effective_device)
    _write_run_json(
        run_dir,
        "run_config.json",
        _runtime_snapshot(
            effective_device,
            {
                "choquet_mode_effective": choquet_mode,
                "choquet_ablation_effective": choquet_ablation,
            },
        ),
    )

    backend = _runtime_agent_backend()
    if backend == "missing_llm_key":
        _write_run_json(run_dir, "run_result.json", _runtime_snapshot(effective_device, {"status": "missing_llm_key"}))
        return

    df = load_dataset(DATA_PATH, allow_generate_demo=DATA_AUTOGENERATE_DEMO)
    if RUN_SAMPLE_LIMIT > 0 and RUN_SAMPLE_LIMIT < len(df):
        df = df.sample(n=RUN_SAMPLE_LIMIT, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"\nRUN_SAMPLE_LIMIT active: using {len(df)} rows for this run.")
    print(f"\nDataset: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(df.groupby(["task_name", "label"]).size().to_string())
    num_classes = infer_dataset_num_classes(df)
    print(f"Num classes: {num_classes}")

    train_df, valid_df, test_df = split_dataframe(
        df,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        seed=RANDOM_SEED,
    )
    print(f"\nSplit: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    model = MultiAgentChoquetModel(
        num_classes=num_classes,
        device=effective_device,
        agent_backend=backend,
        choquet_mode=choquet_mode,
        choquet_ablation=choquet_ablation,
    )
    smoke_ok = None
    smoke_failures = []

    if backend in {"llm", "hybrid"}:
        smoke_ok, smoke_failures = _smoke_test_llm_agents(model, df)
        if not smoke_ok and backend == "llm":
            print("LLM mode smoke test failed. Exiting before training.")
            print("First failure:", smoke_failures[0] if smoke_failures else "unknown")
            _write_run_json(
                run_dir,
                "run_result.json",
                _runtime_snapshot(
                    effective_device,
                    {
                        "status": "llm_smoke_failed",
                        "effective_backend": backend,
                        "smoke_failures": smoke_failures,
                    },
                ),
            )
            return
        if not smoke_ok and backend == "hybrid":
            print("Hybrid smoke test failed. Rebuilding with rule agents for this run.")
            backend = "rule"
            model = MultiAgentChoquetModel(
                num_classes=num_classes,
                device=effective_device,
                agent_backend=backend,
                choquet_mode=choquet_mode,
                choquet_ablation=choquet_ablation,
            )

    # Fit TF-IDF relevance on all available text/task descriptions. This is not
    # label leakage; it only builds a vocabulary for relevance estimation.
    model.fit_relevance(df)

    if backend in {"llm", "hybrid"}:
        print("\n=== Precomputing LLM Agent Output Cache ===")
        print("LLM calls happen here. Training epochs switch to cache-only reads afterward.")
        model.warm_agent_cache(df)
        missing_cache = model.missing_llm_cache_entries(df)
        if missing_cache:
            print("LLM cache precompute is incomplete. Missing entries:")
            for item in missing_cache:
                print(
                    f"  - row={item['row_index']} agent={item['agent']} "
                    f"model={item['model']} text={item['text_preview']!r}"
                )
        _ensure_dataset_integrity_or_retry(df, model)
        model.set_llm_cache_only(True)
        print(f"LLM cache saved to: {LLM_CACHE_PATH}")

    print(f"\n=== Training Choquet Layer ({choquet_mode}, ablation={choquet_ablation}) ===")
    try:
        model, best_valid = train_choquet_model(
            model=model,
            train_df=train_df,
            valid_df=valid_df,
            model_path=run_model_path,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            device=effective_device,
        )
    except Exception as exc:
        if backend == "llm":
            print("Training stopped, likely because an LLM cache entry is missing.")
            print("Run scripts/precompute_llm_outputs.py before llm-mode training.")
        raise exc
    print(
        "\nBest validation: "
        f"loss={best_valid['loss']:.4f}, acc={best_valid['accuracy']:.4f}, "
        f"macro_f1={best_valid['macro_f1']:.4f}"
    )
    print(f"Saved best model to: {run_model_path}")
    choquet_parameter_diagnostics = _save_choquet_parameter_diagnostics(model, run_dir)

    print("\n=== Test Metrics for Trained Model ===")
    test_metrics = evaluate_model(model, test_df, BATCH_SIZE, effective_device, use_pairwise=True)
    print(
        f"loss={test_metrics['loss']:.4f}, acc={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, recall={test_metrics['recall']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    print("\n=== Baseline Comparison ===")
    rows = compare_methods(model, test_df)
    baseline_table = format_metrics_table(rows)
    print(baseline_table)
    baseline_comparison_paths = _save_baseline_comparison(rows, run_dir, baseline_table)

    test_predictions_path = run_dir / "test_predictions.csv"
    export_test_predictions(
        model=model,
        test_df=test_df,
        output_path=test_predictions_path,
        batch_size=BATCH_SIZE,
        device=effective_device,
    )
    print(f"Test predictions saved to: {test_predictions_path}")

    print_interpretability_summary(model, test_df)
    print_sample_decisions(model, test_df, n_samples=5, seed=RANDOM_SEED)
    export_readable_model_summary(model, test_df, run_summary_path)
    _copy_latest_artifacts(run_model_path, run_summary_path)
    print(f"\nReadable model summary saved to: {run_summary_path}")
    print(f"Latest model copied to: {BEST_MODEL_PATH}")
    print(f"Latest summary copied to: {MODEL_SUMMARY_PATH}")

    monotonicity = model.layer.monotonicity_diagnostics()
    if choquet_mode == "discrete_2additive":
        print("\n=== Capacity Monotonicity Diagnostic ===")
        print(monotonicity)

    _write_run_json(
        run_dir,
        "run_result.json",
        _runtime_snapshot(
            effective_device,
            {
                "status": "completed",
                "effective_backend": backend,
                "choquet_mode_effective": choquet_mode,
                "choquet_ablation_effective": choquet_ablation,
                "smoke_ok": smoke_ok,
                "smoke_failures": smoke_failures,
                "best_validation": best_valid,
                "test_metrics": test_metrics,
                "choquet_parameter_diagnostics": choquet_parameter_diagnostics,
                "monotonicity_diagnostic": monotonicity,
                "run_model_path": str(run_model_path),
                "run_summary_path": str(run_summary_path),
                "test_predictions_path": str(test_predictions_path),
                "baseline_comparison_path": baseline_comparison_paths["json"],
                "baseline_comparison_txt_path": baseline_comparison_paths["txt"],
            },
        ),
    )
    print(f"Run metadata saved to: {run_dir / 'run_result.json'}")
    print("\nDone. Agents produced fixed outputs; only the selected Choquet aggregation layer was trained.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
