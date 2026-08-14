# Choquet Multi-Agent Voting Demo

这个项目演示一个二分类多 Agent 投票系统：5 个固定 Agent 先给出各自的概率、置信度和解释，`ChoquetInspiredVotingLayer` 再学习单 Agent 权重和两两交互权重。Agent 层支持三种后端：规则型、真实 LLM API、以及 LLM 失败后回退规则型的 hybrid 模式。

## 默认规则 Agent

默认配置是：

```python
AGENT_BACKEND = "rule"
```

直接运行即可，不需要任何 API key：

```bash
pip install -r requirements.txt
python main.py
```

默认 rule 模式只使用本地规则 Agent，离线可跑。

## Gemini 中转站 LLM Agent

LLM 调用走 OpenAI-compatible `chat/completions` HTTP 接口，不使用 Google Gemini 官方 SDK，也不使用 DashScope/Qwen SDK。

默认 LLM 配置指向：

```text
Base URL: https://your-openai-compatible-endpoint.example/v1
Endpoint: https://your-openai-compatible-endpoint.example/v1/chat/completions
Model: gemini-3.1-flash-lite
API key env: LLM_API_KEY
Provider: openai_compatible
```

PowerShell 示例：

```powershell
$env:AGENT_BACKEND = "hybrid"
$env:LLM_PROVIDER = "openai_compatible"
$env:LLM_MODEL = "gemini-3.1-flash-lite"
$env:LLM_API_KEY_ENV = "LLM_API_KEY"
$env:LLM_API_KEY = "your-api-key"
$env:LLM_BASE_URL = "https://your-openai-compatible-endpoint.example/v1"
```

测试中转站：

```powershell
python scripts/test_llm_gateway.py
```

运行项目：

```powershell
python main.py
```

## Backend 行为

- `rule`：只使用规则 Agent，不需要 key。
- `hybrid`：优先使用 LLM；没有 key 或 smoke test 失败时，清晰提示并回退规则 Agent。
- `llm`：必须有可用 key；缺 key 或 smoke test 失败会退出，不进入训练。

启动时会打印：`AGENT_BACKEND`、`LLM_PROVIDER`、`LLM_MODEL`、`LLM_BASE_URL`、`LLM_API_KEY_ENV`、是否检测到 key、缓存开关和缓存路径。不会打印 API key 内容。

## Agent 输出格式

规则 Agent 和 LLM Agent 都会转换为统一格式：

```json
{
  "probs": [0.3, 0.7],
  "confidence": 0.7,
  "explanation": "简短解释"
}
```

LLM 原始输出必须是严格 JSON：

```json
{
  "class_0_probability": 0.0,
  "class_1_probability": 1.0,
  "confidence": 0.0,
  "explanation": "不超过50字的中文解释"
}
```

代码会自动处理：

- `class_0 = 否 / 非标题党 / 非点击诱导`
- `class_1 = 是 / 标题党 / 点击诱导`
- 两个概率归一化
- `confidence` 限制到 `[0, 1]`
- 剥离 ```json 代码块
- 从多余文本中提取第一个 JSON 对象
- 解析失败时 fallback

## 5 个 LLM Agent 角色

- `SemanticAgent`：语义上下文专家
- `EmotionAgent`：情绪态度专家
- `IntentionAgent`：点击诱导/操纵意图专家
- `LexicalAgent`：词汇表层模式专家
- `ConsistencyAgent`：一致性/矛盾/转折专家

每个 Agent 只从自己的专家视角判断，不综合其他角度。

### LLM Agent 输入策略

LLM Agent 在 `src/agents.py` 的 `generate_input_text()` 阶段按 agent 类型选择输入来源；规则 Agent 行为保持不变，仍使用归一化后的 `df["text"]`。

| LLM Agent | 输入来源 |
| --- | --- |
| `LLMLexicalAgent` | 只使用 `title` |
| `LLMIntentionAgent` | 只使用 `title` |
| `LLMEmotionAgent` | 只使用 `title`，不再拼接 `content` |
| `LLMSemanticAgent` | 使用 `title + content` |
| `LLMConsistencyAgent` | 使用 `title + content` |

## 缓存

LLM 输出缓存到：

```text
outputs/llm_cache.json
```

缓存 key 由 agent 专属输入文本、`task_description`、`agent_name`、`model_name` 生成。训练多个 epoch 时不会重复调用 API，后续训练会直接读取缓存，降低耗时和成本。

## 训练的是什么

项目不会训练 Gemini/LLM。LLM 只负责生成 5 个 Agent 的概率、置信度和解释。

唯一可训练模块仍然是 `ChoquetInspiredVotingLayer`。训练逻辑继续使用 `CrossEntropyLoss` 和 `AdamW`，`task_relevance`、`sample_relevance` 仍由现有 TF-IDF 方法提供。

完整流水线保持为：`src/dataset.py` 加载 `toy_data.csv` 或用户数据集并归一化字段，`split_dataframe()` 划分 train/valid/test，`src/agents.py` 运行规则 Agent 或 LLM Agent 预测，`src/choquet_layer.py` 计算 single-agent contribution 与 pairwise interaction 或 `discrete_2additive` 聚合，`src/train.py` 优化 `ChoquetInspiredVotingLayer` 参数，`src/evaluate.py` 输出 Accuracy/F1/Precision/Recall 与 Baseline Comparison，最后归档到 `outputs/runs/<run_dir>`。

最终输出包括：

- `outputs/best_choquet_model.pt`
- `outputs/model_summary.json`
- `outputs/llm_cache.json`

## 每次运行的结果归档

`main.py` 每次启动都会创建一个新的运行目录：

```text
outputs/runs/0001_YYYYMMDD_HHMMSS_backend_model/
```

目录名开头是递增序号，后面是时间戳、backend 和模型名，便于区分先后顺序。每个目录会保存：

- `best_choquet_model.pt`：本轮训练得到的最佳 Choquet layer checkpoint
- `model_summary.json`：本轮可读模型摘要
- `run_config.json`：本轮运行配置，不包含 API key
- `run_result.json`：本轮运行状态、验证指标、测试指标和产物路径

同时，最新一轮成功训练的模型和摘要仍会复制到：

- `outputs/best_choquet_model.pt`
- `outputs/model_summary.json`

## 快速小样本验证

首次全量 LLM 预热会调用 `样本数 × 5` 次 API，可能较慢。可以先用小样本验证链路：

```powershell
$env:AGENT_BACKEND = "hybrid"
$env:RUN_SAMPLE_LIMIT = "8"
$env:EPOCHS = "2"
$env:BATCH_SIZE = "2"
python main.py
```

正式全量运行时清除这些调试变量即可：

```powershell
Remove-Item Env:RUN_SAMPLE_LIMIT -ErrorAction SilentlyContinue
Remove-Item Env:EPOCHS -ErrorAction SilentlyContinue
Remove-Item Env:BATCH_SIZE -ErrorAction SilentlyContinue
python main.py
```
## CPU / GPU

- 默认 `DEVICE=auto`，程序会自动检测 `torch.cuda.is_available()`。
- 没有 GPU 时会使用 CPU，项目可以完整运行。
- 当前真正训练的是 Choquet aggregation layer，参数量很小，CPU 足够。
- 规则 agent 不需要 GPU。
- 远程 LLM API agent 不使用本地 GPU。
- 只有未来接入本地大模型或本地 Transformer encoder 时，才可能需要 GPU。

## LLM 预计算与缓存

LLM agent 只负责生成 5 个 agent 的概率、置信度和解释。训练 Choquet layer 时不应在每个 epoch 重复请求 LLM。

推荐流程：

```powershell
python scripts/precompute_llm_outputs.py
python main.py
```

缓存文件：

```text
outputs/llm_cache.json
```

缓存 key 由 agent 专属输入文本、`task_description`、`agent_name`、`model_name` 生成。对于 1000 条样本和 5 个 agent，理论 LLM 调用次数应约为 `1000 * 5 = 5000` 次，和 epoch 数无关。如果每个 epoch 都调用 LLM，成本和耗时会乘以 epoch 数，并且远程模型输出可能引入额外不稳定性。

## Choquet 模式

通过环境变量切换：

```powershell
$env:CHOQUET_MODE = "inspired"              # 默认
$env:CHOQUET_MODE = "discrete_2additive"    # 对照实验
```

### inspired

默认模式，保留原有行为。公式近似为：

```text
score = sum_i w_i p_i + pair_scale * mean_{i<j}(w_ij * p_i * p_j)
```

其中 `w_i` 来自 task relevance、sample relevance 和 confidence 的 softmax 动态权重，`w_ij` 来自 pair relevance/agreement 的 sigmoid 动态权重。这个模式是 `Choquet-inspired pairwise aggregation`，不是严格离散 Choquet integral。

### discrete_2additive

新增可选模式，使用有限集合上的离散 Choquet 排序差分结构：

```text
C_mu(f) = sum_i [f_sigma(i) - f_sigma(i-1)] * mu({sigma(i), ..., sigma(K)})
```

其中 capacity 用 2-additive Mobius 近似：

```text
mu(S) = sum_{i in S} m_i + sum_{i<j, i,j in S} m_ij
```

实现中 `single_m = softmax(raw_single_m)`，`pair_m = pair_scale * tanh(raw_pair_m)`，并用 `mu(S) / mu(N)` 做安全归一化。

注意：当前 `discrete_2additive` 实现了排序差分形式，但没有完整强制 capacity 单调性约束，因此应称为 `2-additive Choquet approximation`，不要称为数学上完全严格的 Choquet integral。代码提供 `monotonicity_diagnostics()` 用于检查是否存在 `A subset B` 但 `mu(A) > mu(B)` 的违反情况。

公式验证：

```powershell
python scripts/test_choquet_formula.py
```
## Auto Run with Python

You can run the project without manually typing PowerShell environment variables each time:

```powershell
python auto_run.py
```

Edit the `CONFIG` block at the top of `auto_run.py` to switch modes:

- `CHOQUET_MODE`: `inspired` or `discrete_2additive`
- `AGENT_BACKEND`: `rule`, `hybrid`, or `llm`
- `RUN_SAMPLE_LIMIT`: `"8"` for a quick small-sample test, or `""` for the full dataset
- `EPOCHS`: `"2"` for a quick test, or `""` to use the default training epochs from `config.py`
- `BATCH_SIZE`: `"2"` for a quick test, or `""` to use the default batch size from `config.py`

`auto_run.py` does not store the real API key. If you use `hybrid` or `llm`, set `LLM_API_KEY` in your system environment or current shell before running it. The default `rule` backend does not need a key.

Common presets are documented as comments inside `auto_run.py`, including quick `inspired`, quick `discrete_2additive`, full rule training, and LLM hybrid small-sample validation.

## Auto Run Local API Key

`auto_run.py` supports two safe ways to provide the LLM gateway key.

Mode A, recommended: use a system or shell environment variable:

```powershell
$env:LLM_API_KEY = "your-api-key"
python auto_run.py
```

Mode B, convenient local use: fill the key in the editable `CONFIG` block of `auto_run.py`:

```python
"LLM_API_KEY_ENV": "LLM_API_KEY",
"LLM_API_KEY_VALUE": "",
```

Notes:

- `LLM_API_KEY_ENV` should remain an environment variable name such as `LLM_API_KEY`; do not put the real key there.
- `auto_run.py` never prints `LLM_API_KEY_VALUE`; it only prints `API key detected: True/False` and `API key source: CONFIG/environment/missing`.
- Do not upload or commit an `auto_run.py` that contains a real key. Clear `LLM_API_KEY_VALUE` before sharing or pushing to GitHub.
- If you prefer a separate private file later, add it to `.gitignore` and keep secrets out of committed files.
