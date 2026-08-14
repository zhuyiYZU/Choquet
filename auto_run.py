from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

AGENT_CONFIGS = [
    ("Semantic", "ENABLE_SEMANTIC", "enable_semantic"),
    ("Emotion", "ENABLE_EMOTION", "enable_emotion"),
    ("Intention", "ENABLE_INTENTION", "enable_intention"),
    ("Lexical", "ENABLE_LEXICAL", "enable_lexical"),
    ("Consistency", "ENABLE_CONSISTENCY", "enable_consistency"),
]

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


PROJECT_ROOT = Path(__file__).resolve().parent


def load_project_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ===================== API Configuration Presets =====================
# Uncomment ONE of the following API configurations:

# Option 1: DeepSeek API (current)
API_PRESET = "deepseek"
API_CONFIGS = {
    "deepseek": {
        "LLM_BASE_URL": "https://api.deepseek.com/v1",
        "LLM_API_KEY_VALUE": "",  # Read from .env file
        "LLM_MODEL": "deepseek-v4-flash",
    },


    # Add more presets as needed:
    # "custom": {
    #     "LLM_BASE_URL": "https://your-api.com/v1",
    #     "LLM_API_KEY_VALUE": "your-api-key-here",
    #     "LLM_MODEL": "your-model-name",
    # },
}
# ===========================================================

# ===================== Editable CONFIG =====================
CONFIG = {
    # API Preset: Change this to switch between different API configurations
    # Available presets: "deepseek"
    "API_PRESET": API_PRESET,

    # Choquet mode:
    # "inspired" = original Choquet-inspired pairwise aggregation
    # "discrete_2additive" = discrete 2-additive Choquet approximation
    "CHOQUET_MODE": "discrete_2additive",
    # "CHOQUET_ABLATION": "full"
    # "CHOQUET_ABLATION": "no_pairwise"
    # "CHOQUET_ABLATION": "fixed_singleton"
    # "CHOQUET_ABLATION": "fixed_capacity"
    # "CHOQUET_ABLATION": "no_logit_calibration"
    "CHOQUET_ABLATION": "full",

    # Agent backend:
    # "rule" = rule agents, no API key needed
    # "hybrid" = prefer LLM, fallback to rule on failure/cache miss
    # "llm" = LLM only, fail fast on errors/cache miss
    "AGENT_BACKEND": "llm",

    # Agent role switches for role/scale ablations. Defaults preserve the original 5-agent run.
    "ENABLE_SEMANTIC": "1",
    "ENABLE_EMOTION": "1",
    "ENABLE_INTENTION": "1",
    "ENABLE_LEXICAL": "1",
    "ENABLE_CONSISTENCY": "1",

    # Few-shot configuration:
    # "true" = enable few-shot examples for LLM agents


    # "false" = disable few-shot examples
    "FEWSHOT_ENABLED": "false",

    # Dataset and cache paths. Relative paths are resolved by config.py from PROJECT_ROOT.
    # "DATA_PATH": "data/raw_data/shortText/tmnnews1.csv",
    # "LLM_CACHE_PATH": "outputs/llm_cache_tmnnews_ds.json",
    # "DATA_PATH": "data/raw_data/clickbait/SCC.csv",
    # "LLM_CACHE_PATH": "outputs/llm_cache_scc_ds.json",
    # "DATA_PATH": "data/raw_data/implicit/ProsCons.csv",
    # "LLM_CACHE_PATH": "outputs/llm_cache_pros_ds.json",
    # "DATA_PATH": "data/raw_data/implicit/SBIC.csv",
    # "LLM_CACHE_PATH": "outputs/llm_cache_sbic_ds.json",
    # "DATA_PATH": "data/raw_data/shortText/snippets1.csv",
    # "LLM_CACHE_PATH": "outputs/llm_cache_snippets_ds.json",
    "DATA_PATH": "data/raw_data/fakenews/recovery.csv",
    "LLM_CACHE_PATH": "outputs/llm_cache_rec_ds.json",
    # "DATA_PATH": "data/raw_data/fakenews/politifact.csv",
    # "LLM_CACHE_PATH": "outputs/llm_cache_pol_ds.json",
    # Fast test settings:
    # "8" means use 8 samples; "" or None means full dataset.
    "RUN_SAMPLE_LIMIT": "",

    # "2" means quick test; "" or None uses config.py defaults.
    "RANDOM_SEED": "88",
    # Optional seed sweep. Leave both empty to run only RANDOM_SEED.
    # Example: "78" to "85" runs seeds 78,79,80,81,82,83,84,85.
    "RANDOM_SEED_START": "",
    "RANDOM_SEED_END": "",
    "EPOCHS": "35",
    "BATCH_SIZE": "32",

    # LLM gateway settings - will be overridden by API_PRESET below
    "LLM_PROVIDER": "openai_compatible",
    "LLM_MODEL": "deepseek-v4-flash",  # Default, will be overridden by preset
    "LLM_API_KEY_ENV": "LLM_API_KEY",

    # Optional local private key. Prefer PROJECT_ROOT/.env instead of this field.
    # Leave as "" or None to read LLM_API_KEY from .env or the system environment.
    "LLM_API_KEY_VALUE": "",

    # LLM base URL - will be overridden by API_PRESET below
    "LLM_BASE_URL": "https://api.deepseek.com/v1",

    # Optional workflow switches:
    "RUN_TEST_GATEWAY": False,
    "RUN_PRECOMPUTE_LLM": True,
    "RUN_TEST_FEWSHOT": False,  # Test few-shot functionality
    "RUN_MAIN": True,

    # Stop remaining steps if one subprocess fails.
    "STOP_ON_ERROR": True,
}

# Preset examples:
# 1) Quick inspired test:
#    CHOQUET_MODE="inspired", AGENT_BACKEND="rule", RUN_SAMPLE_LIMIT="8", EPOCHS="2", BATCH_SIZE="2"
# 2) Quick discrete_2additive test:
#    CHOQUET_MODE="discrete_2additive", AGENT_BACKEND="rule", RUN_SAMPLE_LIMIT="8", EPOCHS="2", BATCH_SIZE="2"
# 3) Full rule training:
#    CHOQUET_MODE="inspired", AGENT_BACKEND="rule", RUN_SAMPLE_LIMIT="", EPOCHS="", BATCH_SIZE=""
# 4) LLM hybrid small-sample validation:
#    CHOQUET_MODE="inspired", AGENT_BACKEND="hybrid", RUN_SAMPLE_LIMIT="8", EPOCHS="2", BATCH_SIZE="2",
#    LLM_API_KEY_ENV="LLM_API_KEY", LLM_API_KEY_VALUE="" or your local key,
#    RUN_TEST_GATEWAY=True, RUN_PRECOMPUTE_LLM=True, RUN_MAIN=True
# ===========================================================

ENV_KEYS = [
    "API_PRESET",
    "CHOQUET_MODE",
    "CHOQUET_ABLATION",
    "AGENT_BACKEND",
    "ENABLE_SEMANTIC",
    "ENABLE_EMOTION",
    "ENABLE_INTENTION",
    "ENABLE_LEXICAL",
    "ENABLE_CONSISTENCY",
    "DATA_PATH",
    "LLM_CACHE_PATH",
    "RUN_SAMPLE_LIMIT",
    "RANDOM_SEED",
    "EPOCHS",
    "BATCH_SIZE",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY_ENV",
    "LLM_BASE_URL",
    "FEWSHOT_ENABLED",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Choquet experiment with command-line overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-preset", choices=sorted(API_CONFIGS.keys()), help="API preset name.")
    parser.add_argument("--choquet-mode", choices=["inspired", "discrete_2additive"], help="Choquet aggregation mode.")
    parser.add_argument(
        "--choquet-ablation",
        choices=["full", "no_pairwise", "fixed_singleton", "fixed_capacity", "no_logit_calibration"],
        help="Ablation mode for discrete_2additive Choquet aggregation.",
    )
    parser.add_argument("--agent-backend", choices=["rule", "hybrid", "llm"], help="Agent backend.")
    for agent_name, _, attr_name in AGENT_CONFIGS:
        parser.add_argument(
            f"--enable-{agent_name.lower()}",
            dest=attr_name,
            nargs="?",
            const="true",
            help=f"Enable the {agent_name} agent: true/false, 1/0, yes/no, on/off.",
        )
    parser.add_argument("--fewshot-enabled", choices=["true", "false", "1", "0", "yes", "no"], help="Enable few-shot examples.")
    parser.add_argument("--data-path", help="Dataset CSV path. Relative paths are resolved from project root.")
    parser.add_argument("--llm-cache-path", "--cache-path", dest="llm_cache_path", help="LLM cache JSON path.")
    parser.add_argument("--run-sample-limit", type=int, help="Limit samples for a run. Use 0 for full dataset.")
    parser.add_argument("--random-seed", type=int, help="Single random seed.")
    parser.add_argument("--random-seed-start", type=int, help="First seed for a sweep.")
    parser.add_argument("--random-seed-end", type=int, help="Last seed for a sweep.")
    parser.add_argument("--epochs", type=int, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--llm-provider", help="LLM provider, for example openai_compatible or local.")
    parser.add_argument("--llm-model", help="LLM model name.")
    parser.add_argument("--llm-base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--llm-api-key-env", help="Environment variable name that stores the API key.")
    parser.add_argument("--llm-api-key", help="Inline API key for this process only. Prefer .env when possible.")
    parser.add_argument("--run-test-gateway", action="store_true", help="Run gateway smoke test.")
    parser.add_argument("--skip-test-gateway", action="store_true", help="Skip gateway smoke test.")
    parser.add_argument("--run-test-fewshot", action="store_true", help="Run few-shot test.")
    parser.add_argument("--skip-test-fewshot", action="store_true", help="Skip few-shot test.")
    parser.add_argument("--run-precompute-llm", action="store_true", help="Run LLM precompute step.")
    parser.add_argument("--skip-precompute-llm", action="store_true", help="Skip LLM precompute step.")
    parser.add_argument("--run-main", action="store_true", help="Run main.py training/evaluation step.")
    parser.add_argument("--skip-main", action="store_true", help="Skip main.py step.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop remaining steps after a failure.")
    parser.add_argument("--no-stop-on-error", action="store_true", help="Continue remaining steps after a failure.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration without running subprocesses.")
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> None:
    mapping = {
        "API_PRESET": args.api_preset,
        "CHOQUET_MODE": args.choquet_mode,
        "CHOQUET_ABLATION": args.choquet_ablation,
        "AGENT_BACKEND": args.agent_backend,
        "ENABLE_SEMANTIC": args.enable_semantic,
        "ENABLE_EMOTION": args.enable_emotion,
        "ENABLE_INTENTION": args.enable_intention,
        "ENABLE_LEXICAL": args.enable_lexical,
        "ENABLE_CONSISTENCY": args.enable_consistency,
        "FEWSHOT_ENABLED": args.fewshot_enabled,
        "DATA_PATH": args.data_path,
        "LLM_CACHE_PATH": args.llm_cache_path,
        "RUN_SAMPLE_LIMIT": args.run_sample_limit,
        "RANDOM_SEED": args.random_seed,
        "RANDOM_SEED_START": args.random_seed_start,
        "RANDOM_SEED_END": args.random_seed_end,
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "LLM_PROVIDER": args.llm_provider,
        "LLM_MODEL": args.llm_model,
        "LLM_BASE_URL": args.llm_base_url,
        "LLM_API_KEY_ENV": args.llm_api_key_env,
        "LLM_API_KEY_VALUE": args.llm_api_key,
    }
    for key, value in mapping.items():
        if value is not None:
            CONFIG[key] = str(value)

    if args.run_test_gateway:
        CONFIG["RUN_TEST_GATEWAY"] = True
    if args.skip_test_gateway:
        CONFIG["RUN_TEST_GATEWAY"] = False
    if args.run_test_fewshot:
        CONFIG["RUN_TEST_FEWSHOT"] = True
    if args.skip_test_fewshot:
        CONFIG["RUN_TEST_FEWSHOT"] = False
    if args.run_precompute_llm:
        CONFIG["RUN_PRECOMPUTE_LLM"] = True
    if args.skip_precompute_llm:
        CONFIG["RUN_PRECOMPUTE_LLM"] = False
    if args.run_main:
        CONFIG["RUN_MAIN"] = True
    if args.skip_main:
        CONFIG["RUN_MAIN"] = False
    if args.stop_on_error:
        CONFIG["STOP_ON_ERROR"] = True
    if args.no_stop_on_error:
        CONFIG["STOP_ON_ERROR"] = False


def set_or_clear_env(key: str, value: object) -> None:
    if value is None or str(value).strip() == "":
        os.environ.pop(key, None)
    else:
        os.environ[key] = str(value)


def looks_like_inline_api_key(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(re.match(r"^(sk-|AIza|eyJ)[A-Za-z0-9_.-]{12,}$", text))


def apply_api_preset() -> None:
    api_preset = CONFIG.get("API_PRESET", "")
    if api_preset and api_preset in API_CONFIGS:
        preset = API_CONFIGS[api_preset]
        print(f"Using API preset: {api_preset}")
        for key, value in preset.items():
            CONFIG[key] = value
    elif api_preset:
        print(f"WARNING: Unknown API preset '{api_preset}', using default CONFIG values")


def apply_config() -> tuple[bool, str]:
    for key in ENV_KEYS:
        set_or_clear_env(key, CONFIG.get(key))

    api_key_env = str(CONFIG.get("LLM_API_KEY_ENV") or "LLM_API_KEY").strip()
    api_key_value = CONFIG.get("LLM_API_KEY_VALUE")

    if looks_like_inline_api_key(api_key_env) and not api_key_value:
        print("ERROR: LLM_API_KEY_ENV appears to contain a real API key.")
        print('Set LLM_API_KEY_ENV to an env var name like "LLM_API_KEY" and put the key in PROJECT_ROOT/.env or the system environment.')
        return False, "invalid_key_env"

    if api_key_value is not None and str(api_key_value).strip() != "":
        os.environ[api_key_env] = str(api_key_value).strip()
        return True, "CONFIG"
    if os.environ.get(api_key_env):
        return True, "environment"
    return True, "missing"


def api_key_status() -> tuple[bool, str]:
    api_key_env = os.environ.get("LLM_API_KEY_ENV", "LLM_API_KEY")
    config_value = CONFIG.get("LLM_API_KEY_VALUE")
    if config_value is not None and str(config_value).strip() != "":
        return True, "CONFIG"
    if os.environ.get(api_key_env):
        return True, "environment"
    return False, "missing"


def seed_values() -> list[int]:
    start = str(CONFIG.get("RANDOM_SEED_START") or "").strip()
    end = str(CONFIG.get("RANDOM_SEED_END") or "").strip()
    if start or end:
        if not start or not end:
            raise ValueError("RANDOM_SEED_START and RANDOM_SEED_END must be set together.")
        start_seed = int(start)
        end_seed = int(end)
        step = 1 if end_seed >= start_seed else -1
        return list(range(start_seed, end_seed + step, step))
    return [int(str(CONFIG.get("RANDOM_SEED") or os.environ.get("RANDOM_SEED") or "78").strip())]


def enabled_agent_names() -> list[str]:
    return [
        agent_name
        for agent_name, env_key, _ in AGENT_CONFIGS
        if parse_bool(os.environ.get(env_key, CONFIG.get(env_key, "true")))
    ]


def validate_enabled_agents() -> tuple[list[str], list[str]]:
    enabled = enabled_agent_names()
    disabled = [agent_name for agent_name, _, _ in AGENT_CONFIGS if agent_name not in enabled]
    if len(enabled) < 2:
        raise ValueError("At least two agents must be enabled for ensemble/Choquet aggregation.")
    return enabled, disabled


def print_enabled_agents() -> None:
    enabled, disabled = validate_enabled_agents()
    print(f"Enabled agents: {', '.join(enabled)}")
    if disabled:
        print(f"Disabled agents: {', '.join(disabled)}")


def print_config(project_root: Path) -> None:
    detected, source = api_key_status()
    print("=== auto_run.py configuration ===")
    print(f"Project root     : {project_root}")
    for key in [
        "API_PRESET",
        "AGENT_BACKEND",
        "CHOQUET_MODE",
        "CHOQUET_ABLATION",
        "FEWSHOT_ENABLED",
        "DATA_PATH",
        "LLM_CACHE_PATH",
        "RUN_SAMPLE_LIMIT",
        "RANDOM_SEED",
        "EPOCHS",
        "BATCH_SIZE",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY_ENV",
    ]:
        print(f"{key:16s}: {os.environ.get(key, '')}")
    print(f"API key detected : {detected}")
    print(f"API key source   : {source}")
    print_enabled_agents()
    print("=================================")


def run_step(title: str, args: Iterable[str], project_root: Path) -> int:
    print(f"\n=== {title} ===")
    command = [sys.executable, *args]
    print("Command:", " ".join(command))
    result = subprocess.run(command, cwd=project_root, env=os.environ.copy())
    print(f"{title} exit code: {result.returncode}")
    return result.returncode


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT
    os.chdir(project_root)
    load_project_env()
    apply_cli_overrides(args)
    apply_api_preset()
    apply_cli_overrides(args)
    config_ok, _ = apply_config()
    if not config_ok:
        return 2
    try:
        validate_enabled_agents()
        seeds = seed_values()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.dry_run:
        print(f"Seed runs        : {', '.join(str(seed) for seed in seeds)}")
        if seeds:
            os.environ["RANDOM_SEED"] = str(seeds[0])
        print_config(project_root)
        return 0

    steps: list[tuple[str, list[str]]] = []
    if CONFIG.get("RUN_TEST_GATEWAY"):
        steps.append(("Gateway smoke test", ["scripts/test_llm_gateway.py"]))
    if CONFIG.get("RUN_TEST_FEWSHOT"):
        steps.append(("Test few-shot functionality", ["scripts/test_fewshot.py"]))
    if CONFIG.get("RUN_PRECOMPUTE_LLM"):
        steps.append(("Precompute LLM outputs", ["scripts/precompute_llm_outputs.py"]))
    if CONFIG.get("RUN_MAIN"):
        steps.append(("Run main.py", ["main.py"]))

    if not steps:
        print("No steps selected. Enable RUN_TEST_GATEWAY, RUN_PRECOMPUTE_LLM, or RUN_MAIN in CONFIG.")
        return 0

    overall_success = True
    seed_results: list[tuple[int, bool]] = []
    print(f"Seed runs        : {', '.join(str(seed) for seed in seeds)}")
    for run_index, seed in enumerate(seeds, 1):
        os.environ["RANDOM_SEED"] = str(seed)
        print()
        print("=" * 80)
        print(f"Seed run {run_index}/{len(seeds)} | RANDOM_SEED={seed}")
        print("=" * 80)
        print_config(project_root)

        seed_success = True
        for title, args in steps:
            code = run_step(title, args, project_root)
            if code != 0:
                seed_success = False
                overall_success = False
                print(f"Step failed: seed={seed}, step={title}, exit code={code}")
                if CONFIG.get("STOP_ON_ERROR", True):
                    break
        seed_results.append((seed, seed_success))
        if not seed_success and CONFIG.get("STOP_ON_ERROR", True):
            break

    print("\n=== auto_run.py summary ===")
    print(f"Success          : {overall_success}")
    print(f"Choquet mode     : {os.environ.get('CHOQUET_MODE', '')}")
    print(f"Choquet ablation : {os.environ.get('CHOQUET_ABLATION', '')}")
    print_enabled_agents()
    print("Seed results     :")
    for seed, ok in seed_results:
        print(f"  seed={seed}: {'OK' if ok else 'FAILED'}")
    print(f"Run outputs dir  : {project_root / 'outputs' / 'runs'}")
    print(f"Latest model     : {project_root / 'outputs' / 'best_choquet_model.pt'}")
    print(f"Latest summary   : {project_root / 'outputs' / 'model_summary.json'}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
