# API Configuration Guide

This project does not require an API key when `AGENT_BACKEND=rule`.

For `hybrid` or `llm` mode, keep secrets outside git. Use a local `.env` file or a shell environment variable:

```powershell
$env:LLM_API_KEY = "your-api-key"
$env:LLM_BASE_URL = "https://api.deepseek.com/v1"
$env:LLM_MODEL = "deepseek-v4-flash"
python auto_run.py
```

Safe `auto_run.py` preset pattern:

```python
API_CONFIGS = {
    "deepseek": {
        "LLM_BASE_URL": "https://api.deepseek.com/v1",
        "LLM_API_KEY_VALUE": "",
        "LLM_MODEL": "deepseek-v4-flash",
    },
    "custom": {
        "LLM_BASE_URL": "https://your-openai-compatible-endpoint.example/v1",
        "LLM_API_KEY_VALUE": "",
        "LLM_MODEL": "your-model-name",
    },
}
```

Do not commit real values for `LLM_API_KEY_VALUE`, `.env`, LLM caches, run outputs, IDE settings, or compiled Python files.
