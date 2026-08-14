import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEMO_DATA_PATH = PROJECT_ROOT / "data" / "toy_data.csv"


def _load_project_env() -> None:
    """Load PROJECT_ROOT/.env without overriding variables already set by the shell."""
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


_load_project_env()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


DATA_PATH = _resolve_path(os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "raw_data" / "fakenews" / "politifact.csv")))
DATA_AUTOGENERATE_DEMO = os.getenv("DATA_AUTOGENERATE_DEMO", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
MODEL_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = MODEL_DIR / "runs"
BEST_MODEL_PATH = MODEL_DIR / "best_choquet_model.pt"
MODEL_SUMMARY_PATH = MODEL_DIR / "model_summary.json"
LLM_CACHE_PATH = _resolve_path(os.getenv("LLM_CACHE_PATH", str(MODEL_DIR / "llm_cache_pol_ds.json")))
CHOQUET_MODE = os.getenv("CHOQUET_MODE", "inspired").strip().lower()
CHOQUET_ABLATION = os.getenv("CHOQUET_ABLATION", "full").strip().lower()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "78"))
NUM_CLASSES = 2
ALL_AGENT_NAMES = [
    "Semantic",
    "Emotion",
    "Intention",
    "Lexical",
    "Consistency",
]
AGENT_ENABLE_ENV = {
    "Semantic": "ENABLE_SEMANTIC",
    "Emotion": "ENABLE_EMOTION",
    "Intention": "ENABLE_INTENTION",
    "Lexical": "ENABLE_LEXICAL",
    "Consistency": "ENABLE_CONSISTENCY",
}


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value for {name}: {value!r}")


ENABLED_AGENTS = [
    agent_name
    for agent_name in ALL_AGENT_NAMES
    if _env_bool(AGENT_ENABLE_ENV[agent_name], True)
]
DISABLED_AGENTS = [
    agent_name for agent_name in ALL_AGENT_NAMES if agent_name not in ENABLED_AGENTS
]
AGENT_NAMES = ENABLED_AGENTS

# Default to rule agents so the project runs without any API key. LLM settings
# point at an OpenAI-compatible gateway and can be overridden by env vars.
AGENT_BACKEND = os.getenv("AGENT_BACKEND", "rule").strip().lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_MODEL_CANDIDATES = [
    item.strip()
    for item in os.getenv(
        "LLM_MODEL_CANDIDATES",
        "deepseek-v4-pro",
    ).split(",")
    if item.strip()
]
LLM_API_KEY_ENV = os.getenv("LLM_API_KEY_ENV", "LLM_API_KEY")
LEGACY_LLM_API_KEY_ENVS = ()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").strip()
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Few-shot examples configuration
FEWSHOT_ENABLED = os.getenv("FEWSHOT_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
FEWSHOT_DIR = PROJECT_ROOT / "data" / "fewshot_examples"

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "35"))
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
DEVICE = os.getenv("DEVICE", "auto").strip().lower()
RUN_SAMPLE_LIMIT = int(os.getenv("RUN_SAMPLE_LIMIT", "0"))


def get_llm_api_key() -> str:
    """Resolve the API key from the configured env name, then stable fallbacks."""
    candidates = [LLM_API_KEY_ENV]
    if "LLM_API_KEY" not in candidates:
        candidates.append("LLM_API_KEY")
    candidates.extend(name for name in LEGACY_LLM_API_KEY_ENVS if name not in candidates)
    for name in candidates:
        value = os.getenv(name)
        if value:
            return value
    return ""


def has_llm_api_key() -> bool:
    return bool(get_llm_api_key())
