import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from config import AGENT_NAMES, LLM_CACHE_PATH
from src.cache import LLMCache
from src.task_schemas import get_task_label_schema


REQUIRED_AGENTS = tuple(AGENT_NAMES)


def sample_id(text: str, task_description: str) -> str:
    raw = json.dumps(
        {"text": text, "task_description": task_description},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path = LLM_CACHE_PATH) -> Dict[str, Dict[str, object]]:
    cache = LLMCache(cache_path)
    return cache._items


def is_success_entry(entry: Optional[Dict[str, object]]) -> bool:
    if not isinstance(entry, dict):
        return False
    return entry.get("status") == "SUCCESS" and isinstance(entry.get("output"), dict)


def cache_integrity_details(
    cache_path: Path = LLM_CACHE_PATH,
    required_agents: Iterable[str] = REQUIRED_AGENTS,
) -> Dict[str, Dict[str, object]]:
    required = tuple(required_agents)
    grouped = defaultdict(set)
    for entry in load_cache(cache_path).values():
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", ""))
        task_description = str(entry.get("task_description", ""))
        model_name = str(entry.get("model_name", ""))
        agent_name = str(entry.get("agent_name", ""))
        if is_success_entry(entry):
            grouped[(text, task_description, model_name)].add(agent_name)

    details = {}
    for (text, task_description, model_name), agents in grouped.items():
        missing = [agent for agent in required if agent not in agents]
        sid = json.dumps(
            {"text": text, "task_description": task_description, "model_name": model_name},
            ensure_ascii=False,
            sort_keys=True,
        )
        details[sid] = {
            "text": text,
            "task_description": task_description,
            "model_name": model_name,
            "success_agents": len([agent for agent in required if agent in agents]),
            "missing_agents": missing,
        }
    return details


def check_cache_integrity(
    cache_path: Path = LLM_CACHE_PATH,
    required_agents: Iterable[str] = REQUIRED_AGENTS,
) -> Dict[str, object]:
    """Check only samples that already exist in cache, independent of any dataset."""
    required = tuple(required_agents)
    grouped = defaultdict(
        lambda: {
            "success_agents": set(),
            "observed_agents": set(),
            "text": "",
            "task_description": "",
            "model_name": "",
        }
    )
    success_records = 0
    failed_records = 0
    for entry in load_cache(cache_path).values():
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", ""))
        task_description = str(entry.get("task_description", ""))
        model_name = str(entry.get("model_name", ""))
        agent_name = str(entry.get("agent_name", ""))
        if not text and not task_description:
            continue
        group = grouped[(text, task_description, model_name)]
        group["text"] = text
        group["task_description"] = task_description
        group["model_name"] = model_name
        if agent_name:
            group["observed_agents"].add(agent_name)
        if is_success_entry(entry):
            success_records += 1
            group["success_agents"].add(agent_name)
        elif entry.get("status") == "FAILED":
            failed_records += 1

    details = {}
    for (text, task_description, model_name), group in grouped.items():
        success_agents = group["success_agents"]
        missing = [agent for agent in required if agent not in success_agents]
        sid = json.dumps(
            {"text": text, "task_description": task_description, "model_name": model_name},
            ensure_ascii=False,
            sort_keys=True,
        )
        details[sid] = {
            "text": text,
            "task_description": task_description,
            "model_name": model_name,
            "success_agents": len([agent for agent in required if agent in success_agents]),
            "observed_agents": sorted(group["observed_agents"]),
            "missing_agents": missing,
        }

    cache_samples = len(details)
    incomplete = {sid: data for sid, data in details.items() if data["missing_agents"]}
    return {
        "cache_samples": cache_samples,
        "complete_samples": cache_samples - len(incomplete),
        "incomplete_samples": len(incomplete),
        "success_records": success_records,
        "failed_records": failed_records,
        "details": details,
    }


def check_dataset_integrity(
    df: pd.DataFrame,
    agents,
    cache_path: Path = LLM_CACHE_PATH,
    required_agents: Iterable[str] = REQUIRED_AGENTS,
) -> Dict[str, object]:
    required = tuple(required_agents)
    cache = LLMCache(cache_path)
    details = {}
    success_records = 0
    primary_hits = 0
    fallback_hits = 0
    invalid_records = 0

    records = df.to_dict("records")
    for row_index, record in enumerate(records):
        text = str(record.get("text", ""))
        task_description = str(record.get("task_description", ""))
        present = set()
        missing = []
        for agent_name in required:
            agent = next((agent for agent in agents if agent.name == agent_name), None)
            if agent is None:
                missing.append(agent_name)
                continue
            model_name = getattr(agent, "model_name", "unknown")
            input_builder = getattr(agent, "generate_input_text", None)
            cache_text = input_builder(text, record) if callable(input_builder) else text
            task_name = record.get("task_name")
            label_schema = get_task_label_schema(task_name, task_description)
            primary_key = cache.make_key(
                cache_text,
                task_description,
                agent_name,
                model_name,
                task_name=task_name,
                label_schema=label_schema,
            )
            primary_entry = cache._items.get(primary_key)
            primary_output = cache._valid_success_output(primary_entry)
            cached = cache.get(
                cache_text,
                task_description,
                agent_name,
                model_name,
                task_name=task_name,
                label_schema=label_schema,
            )
            if cached is not None:
                present.add(agent_name)
                success_records += 1
                if primary_output is not None:
                    primary_hits += 1
                else:
                    fallback_hits += 1
            else:
                if primary_entry is not None:
                    invalid_records += 1
                missing.append(agent_name)
        sid = sample_id(text, task_description)
        details[sid] = {
            "row_index": row_index,
            "text": text,
            "task_description": task_description,
            "success_agents": len(present),
            "missing_agents": missing,
        }

    total_samples = len(details)
    expected_records = total_samples * len(required)
    incomplete = {sid: data for sid, data in details.items() if data["missing_agents"]}
    return {
        "total_samples": total_samples,
        "dataset_samples": total_samples,
        "expected_records": expected_records,
        "success_records": success_records,
        "missing_records": expected_records - success_records,
        "primary_hits": primary_hits,
        "fallback_hits": fallback_hits,
        "invalid_records": invalid_records,
        "complete_samples": total_samples - len(incomplete),
        "incomplete_samples": len(incomplete),
        "details": details,
    }


def failed_error_distribution(cache_path: Path = LLM_CACHE_PATH) -> Counter:
    counter = Counter()
    for entry in load_cache(cache_path).values():
        if isinstance(entry, dict) and entry.get("status") == "FAILED":
            counter[str(entry.get("error_type", "Unknown"))] += 1
    return counter


def missing_agent_distribution(details: Dict[str, Dict[str, object]]) -> Counter:
    counter = Counter()
    for data in details.values():
        for agent in data.get("missing_agents", []):
            counter[str(agent)] += 1
    return counter


def print_training_integrity_summary(result: Dict[str, object]) -> None:
    print(f"Dataset Samples: {result['dataset_samples']}")
    print(f"Expected Records: {result['expected_records']}")
    print(f"SUCCESS Records: {result['success_records']}")
    print(f"Missing Records: {result['missing_records']}")
    if "primary_hits" in result:
        print(f"Primary Hits: {result['primary_hits']}")
    if "fallback_hits" in result:
        print(f"Fallback Hits: {result['fallback_hits']}")
    if "invalid_records" in result:
        print(f"Invalid Records: {result['invalid_records']}")
    print(f"Complete Samples : {result['complete_samples']}")
    print(f"Incomplete Samples : {result['incomplete_samples']}")


def raise_if_incomplete(result: Dict[str, object]) -> None:
    incomplete = int(result["incomplete_samples"])
    if incomplete <= 0:
        return
    raise RuntimeError(
        "\nERROR\n\n"
        "Dataset integrity check failed.\n\n"
        f"{incomplete} samples are missing agent outputs.\n\n"
        "Run:\n\n"
        "python scripts/retry_failed_cache.py\n\n"
        "before training."
    )
