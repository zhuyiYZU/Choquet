from typing import Dict

import numpy as np
import torch

from .agents import build_agents, run_agents
from .choquet_layer import ChoquetInspiredVotingLayer
from .embeddings import AGENT_DESCRIPTIONS, TfidfRelevanceEstimator
from .task_schemas import get_task_label_schema
from .utils import to_tensor

try:
    from config import CHOQUET_ABLATION
except ImportError:  # pragma: no cover - package-only import fallback
    CHOQUET_ABLATION = "full"

try:
    from config import ALL_AGENT_NAMES
except ImportError:  # pragma: no cover - package-only import fallback
    ALL_AGENT_NAMES = ["Semantic", "Emotion", "Intention", "Lexical", "Consistency"]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress display
    tqdm = None


class MultiAgentChoquetModel:
    """Thin orchestration wrapper around fixed agents + trainable aggregator.

    Agents can be rule-based or LLM-backed. They are inference-only; the only
    trainable module remains the selected Choquet aggregation layer mode.
    """

    def __init__(
        self,
        num_classes: int = 2,
        device: str = "cpu",
        agent_backend: str = None,
        choquet_mode: str = "inspired",
        choquet_ablation: str | None = None,
    ):
        self.device = device
        self.choquet_mode = choquet_mode
        self.choquet_ablation = choquet_ablation or CHOQUET_ABLATION
        self.agents = build_agents(agent_backend)
        self.agent_names = [a.name for a in self.agents]
        description_by_name = dict(zip(ALL_AGENT_NAMES, AGENT_DESCRIPTIONS))
        self.agent_descriptions = [description_by_name[name] for name in self.agent_names]
        self.relevance = TfidfRelevanceEstimator(self.agent_descriptions)
        self.layer = ChoquetInspiredVotingLayer(
            len(self.agents),
            num_classes,
            mode=choquet_mode,
            ablation_mode=self.choquet_ablation,
        ).to(device)

    def fit_relevance(self, df):
        self.relevance.fit(df["task_description"].tolist(), df["text"].tolist())
        return self

    def set_llm_cache_only(self, cache_only: bool = True) -> None:
        for agent in self.agents:
            setter = getattr(agent, "set_cache_only", None)
            if callable(setter):
                setter(cache_only)

    def warm_agent_cache(self, df, stop_on_full_miss: bool = False) -> dict:
        texts = df["text"].tolist()
        task_descriptions = df["task_description"].tolist()
        records = df.to_dict("records")
        todo = []
        existing_success = 0

        for agent in self.agents:
            cache = getattr(agent, "cache", None)
            model_name = getattr(agent, "model_name", None)
            input_builder = getattr(agent, "generate_input_text", None)
            for row_idx, (text, task_description, record) in enumerate(zip(texts, task_descriptions, records)):
                cache_text = input_builder(text, record) if callable(input_builder) else text
                if cache is not None and model_name is not None:
                    schema = get_task_label_schema(record.get("task_name"), task_description)
                    cached = cache.get(
                        cache_text, task_description, agent.name, model_name,
                        task_name=record.get("task_name"), label_schema=schema,
                    )
                    if cached is not None:
                        existing_success += 1
                        continue
                label_schema = get_task_label_schema(record.get("task_name"), task_description)
                todo.append(
                    (
                        agent,
                        row_idx,
                        text,
                        cache_text,
                        task_description,
                        label_schema,
                        record.get("task_name"),
                    )
                )

        total_expected = len(texts) * len(self.agents)
        self.last_cache_warm_plan = {
            "total": total_expected,
            "existing_success": existing_success,
            "need_requests": len(todo),
        }
        print(
            f"LLM cache warm plan: total={total_expected}, "
            f"existing_success={existing_success}, need_requests={len(todo)}"
        )

        if stop_on_full_miss and total_expected > 0:
            cache_path = None
            for agent in self.agents:
                cache = getattr(agent, "cache", None)
                path = getattr(cache, "path", None)
                if path is not None:
                    cache_path = path
                    break
            cache_exists = bool(cache_path and cache_path.exists())
            cache_size = cache_path.stat().st_size if cache_exists else 0
            if (
                cache_exists
                and cache_size > 1024 * 1024
                and existing_success == 0
                and len(todo) == total_expected
            ):
                raise RuntimeError(
                    "LLM cache full miss protection triggered: cache file exists "
                    f"and is larger than 1MB ({cache_path}, {cache_size} bytes), "
                    "but existing_success=0 and need_requests=total. This usually "
                    "means cache key rules changed, or task_description, "
                    "label_schema, model_name, or input_text no longer matches. "
                    "Run scripts/migrate_cache.py or verify the active dataset/cache "
                    "configuration before allowing new LLM requests."
                )

        iterator = todo
        progress = None
        if tqdm is not None and todo:
            progress = tqdm(todo, desc="Precomputing LLM cache", unit="req")
            iterator = progress

        for (
            agent,
            row_idx,
            text,
            cache_text,
            task_description,
            label_schema,
            task_name,
        ) in iterator:
            if progress is not None:
                progress.set_postfix(row=row_idx, agent=agent.name)
            try:
                prepared = getattr(agent, "_predict_one_prepared", None)
                if callable(prepared):
                    prepared(
                        cache_text, text, task_description, label_schema,
                        task_name,
                    )
                else:
                    agent.predict_one(text, task_description, label_schema)
            except Exception as exc:
                print(f"Cache warm failed: row={row_idx} agent={agent.name} error={exc}")

        if progress is None and todo:
            print(f"LLM cache warm completed requests: {len(todo)}")
        return self.last_cache_warm_plan

    def missing_llm_cache_entries(self, df, limit: int = 20):
        """Return a small list of missing LLM cache entries for diagnostics."""
        missing = []
        texts = df["text"].tolist()
        task_descriptions = df["task_description"].tolist()
        records = df.to_dict("records")
        for agent in self.agents:
            cache = getattr(agent, "cache", None)
            model_name = getattr(agent, "model_name", None)
            if cache is None or model_name is None:
                continue
            input_builder = getattr(agent, "generate_input_text", None)
            for row_idx, (text, task_description, record) in enumerate(zip(texts, task_descriptions, records)):
                cache_text = input_builder(text, record) if callable(input_builder) else text
                schema = get_task_label_schema(record.get("task_name"), task_description)
                cached = cache.get(
                    cache_text, task_description, agent.name, model_name,
                    task_name=record.get("task_name"), label_schema=schema,
                )
                if cached is None:
                    missing.append(
                        {
                            "row_index": row_idx,
                            "agent": agent.name,
                            "model": model_name,
                            "text_preview": str(cache_text)[:120],
                        }
                    )
                    if len(missing) >= limit:
                        return missing
        return missing

    def make_inputs(self, df_or_texts, task_descriptions=None) -> Dict[str, object]:
        if hasattr(df_or_texts, "__getitem__") and task_descriptions is None:
            texts = df_or_texts["text"].tolist()
            task_descriptions = df_or_texts["task_description"].tolist()
            records = df_or_texts.to_dict("records")
        else:
            texts = list(df_or_texts)
            task_descriptions = list(task_descriptions)
            records = None

        agent_probs, agent_conf, explanations = run_agents(
            self.agents,
            texts,
            task_descriptions,
            records=records,
        )
        task_rel = self.relevance.task_relevance(task_descriptions)
        sample_rel = self.relevance.sample_relevance(texts)
        return {
            "texts": texts,
            "task_descriptions": task_descriptions,
            "agent_probs_np": agent_probs,
            "agent_conf_np": agent_conf,
            "task_rel_np": task_rel,
            "sample_rel_np": sample_rel,
            "explanations": explanations,
            "agent_probs": to_tensor(agent_probs, self.device),
            "agent_confidences": to_tensor(agent_conf, self.device),
            "task_relevance": to_tensor(task_rel, self.device),
            "sample_relevance": to_tensor(sample_rel, self.device),
        }

    def logits_from_inputs(self, inputs: Dict[str, object], use_pairwise: bool = True, details: bool = False):
        return self.layer(
            inputs["agent_probs"],
            inputs["agent_confidences"],
            inputs["task_relevance"],
            inputs["sample_relevance"],
            use_pairwise=use_pairwise,
            return_details=details,
        )

    @torch.no_grad()
    def predict(self, df, use_pairwise: bool = True, return_details: bool = False):
        self.layer.eval()
        inputs = self.make_inputs(df)
        output = self.logits_from_inputs(inputs, use_pairwise=use_pairwise, details=return_details)
        if return_details:
            logits, details = output
        else:
            logits, details = output, None
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        if return_details:
            return preds, probs.cpu().numpy(), details, inputs
        return preds, probs.cpu().numpy()

    def state_dict(self):
        return self.layer.state_dict()

    def load_state_dict(self, state_dict):
        self.layer.load_state_dict(state_dict)
