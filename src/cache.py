import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_error(error_message: object) -> str:
    text = str(error_message or "").strip()
    lower = text.lower()
    if not text:
        return "Unknown"
    if "IncompleteRead" in text:
        return "IncompleteRead"
    if "SSL" in text or "_ssl" in lower or "asn1" in lower or "not_enough_data" in lower:
        return "SSL_ERROR"
    if "HTTP 429" in text or "HTTP429" in text:
        return "HTTP429"
    if "ConnectionError" in text or "Network error" in text:
        return "ConnectionError"
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(?:\(|:|$)", text)
    return match.group(1) if match else "LLMError"


def normalize_input_text(text: object) -> str:
    """Return a cache-fingerprint-safe text representation.

    Fingerprints are intentionally less strict than primary keys so cache rows
    survive harmless whitespace/newline changes, but they still keep model,
    agent, task, and class-count boundaries.
    """
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\s+", " ", normalized)


class LLMCache:
    """Persistent JSON cache for LLM agent outputs.

    Key fields intentionally include the model name so changing providers/models
    does not silently reuse old judgments.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: Dict[str, Dict[str, object]] = {}
        self._fingerprint_index: Optional[Dict[str, Dict[str, object]]] = None
        if self.path.exists():
            self._load()

    @staticmethod
    def make_key(
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
        task_name: object = None,
        label_schema: Optional[Dict[str, str]] = None,
    ) -> str:
        schema = label_schema or {}
        raw = json.dumps(
            {
                "text": text,
                "task_description": task_description,
                "agent_name": agent_name,
                "model_name": model_name,
                "task_name": str(task_name or ""),
                "num_classes": len(schema) if schema else None,
                "label_schema": schema,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def make_legacy_key(
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
    ) -> str:
        """Key format used before task/schema metadata became part of the key."""
        raw = json.dumps(
            {
                "text": text,
                "task_description": task_description,
                "agent_name": agent_name,
                "model_name": model_name,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def make_fingerprint(
        text: str,
        agent_name: str,
        model_name: str,
        task_name: object = None,
        num_classes: object = None,
    ) -> str:
        raw = json.dumps(
            {
                "normalized_input_text": normalize_input_text(text),
                "agent_name": agent_name,
                "model_name": model_name,
                "task_name": str(task_name or ""),
                "num_classes": int(num_classes) if num_classes not in (None, "") else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        try:
            self._items = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._items = {}
        changed = self.migrate_legacy_items(flush=False)
        if changed:
            self._flush()

    def _flush(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    @staticmethod
    def _valid_success_output(item: object) -> Optional[Dict[str, object]]:
        if not isinstance(item, dict) or item.get("status") != "SUCCESS":
            return None
        output = item.get("output", {})
        if not isinstance(output, dict):
            return None
        try:
            probs = np.array(output["probs"], dtype=np.float32)
            confidence = float(output["confidence"])
            explanation = str(output["explanation"])
        except (KeyError, TypeError, ValueError):
            return None
        if probs.ndim != 1 or probs.size == 0:
            return None
        return {
            "probs": probs,
            "confidence": confidence,
            "explanation": explanation,
        }

    @staticmethod
    def _serializable_output(output: Dict[str, object]) -> Dict[str, object]:
        probs = output["probs"]
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        return {
            "probs": probs,
            "confidence": float(output["confidence"]),
            "explanation": str(output["explanation"]),
        }

    def _build_fingerprint_index(self) -> Dict[str, Dict[str, object]]:
        if self._fingerprint_index is not None:
            return self._fingerprint_index

        index: Dict[str, Dict[str, object]] = {}
        for item in self._items.values():
            output = self._valid_success_output(item)
            if output is None:
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            agent_name = item.get("agent_name")
            model_name = item.get("model_name")
            if text is None or not agent_name or not model_name:
                continue
            fingerprint = self.make_fingerprint(
                text,
                str(agent_name),
                str(model_name),
                task_name=item.get("task_name"),
                num_classes=item.get("num_classes"),
            )
            index.setdefault(fingerprint, item)
        self._fingerprint_index = index
        return index

    def _write_primary_success(
        self,
        key: str,
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
        item: Dict[str, object],
        task_name: object = None,
        label_schema: Optional[Dict[str, str]] = None,
        flush: bool = True,
    ) -> None:
        output = self._valid_success_output(item)
        if output is None:
            return
        self._items[key] = {
            "status": "SUCCESS",
            "text": text,
            "task_description": task_description,
            "agent_name": agent_name,
            "model_name": model_name,
            "task_name": str(task_name or ""),
            "num_classes": len(label_schema or {}),
            "label_schema": label_schema or {},
            "raw_text": str(item.get("raw_text", "")),
            "output": self._serializable_output(output),
            "migrated_from": item.get("migrated_from", "legacy_or_fingerprint"),
            "migrated_at": utc_now_iso(),
        }
        self._fingerprint_index = None
        if flush:
            self._flush()

    def get(
        self,
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
        task_name: object = None,
        label_schema: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, object]]:
        key = self.make_key(
            text, task_description, agent_name, model_name, task_name, label_schema
        )
        item = self._items.get(key)
        output = self._valid_success_output(item)
        if output is not None:
            return output

        legacy_key = self.make_legacy_key(text, task_description, agent_name, model_name)
        legacy_item = self._items.get(legacy_key)
        output = self._valid_success_output(legacy_item)
        if output is not None:
            return output

        fingerprint = self.make_fingerprint(
            text,
            agent_name,
            model_name,
            task_name=task_name,
            num_classes=len(label_schema or {}),
        )
        fingerprint_item = self._build_fingerprint_index().get(fingerprint)
        output = self._valid_success_output(fingerprint_item)
        if output is not None:
            return output

        return None

    def set(
        self,
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
        output: Dict[str, object],
        raw_text: str,
        task_name: object = None,
        label_schema: Optional[Dict[str, str]] = None,
    ) -> None:
        key = self.make_key(
            text, task_description, agent_name, model_name, task_name, label_schema
        )
        probs = output["probs"]
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        self._items[key] = {
            "status": "SUCCESS",
            "text": text,
            "task_description": task_description,
            "agent_name": agent_name,
            "model_name": model_name,
            "task_name": str(task_name or ""),
            "num_classes": len(label_schema or {}),
            "label_schema": label_schema or {},
            "raw_text": raw_text,
            "output": {
                "probs": probs,
                "confidence": float(output["confidence"]),
                "explanation": str(output["explanation"]),
            },
        }
        self._fingerprint_index = None
        self._flush()

    def set_failed(
        self,
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
        error_message: object,
        task_name: object = None,
        label_schema: Optional[Dict[str, str]] = None,
    ) -> None:
        key = self.make_key(
            text, task_description, agent_name, model_name, task_name, label_schema
        )
        existing = self._items.get(key, {})
        retry_count = int(existing.get("retry_count", 0) or 0) + 1
        self._items[key] = {
            "status": "FAILED",
            "text": text,
            "task_description": task_description,
            "agent_name": agent_name,
            "model_name": model_name,
            "task_name": str(task_name or ""),
            "num_classes": len(label_schema or {}),
            "label_schema": label_schema or {},
            "error_type": classify_error(error_message),
            "error_message": str(error_message),
            "retry_count": retry_count,
            "last_retry": utc_now_iso(),
        }
        self._fingerprint_index = None
        self._flush()

    def get_entry(
        self,
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
        task_name: object = None,
        label_schema: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, object]]:
        key = self.make_key(
            text, task_description, agent_name, model_name, task_name, label_schema
        )
        return self._items.get(key)

    def migrate_legacy_items(self, flush: bool = True) -> bool:
        changed = False
        for item in self._items.values():
            if not isinstance(item, dict) or item.get("status"):
                continue
            raw_text = str(item.get("raw_text", ""))
            output = item.get("output", {})
            probs = output.get("probs") if isinstance(output, dict) else None
            is_neutral_placeholder = (
                isinstance(probs, list)
                and len(probs) == 2
                and abs(float(probs[0]) - 0.5) < 1e-6
                and abs(float(probs[1]) - 0.5) < 1e-6
                and float(output.get("confidence", 1.0)) <= 0.31
            )
            if raw_text.startswith("FALLBACK_NEUTRAL_AFTER_LLM_ERROR:") or is_neutral_placeholder:
                error_message = raw_text.split(":", 1)[1].strip() if ":" in raw_text else raw_text
                item.pop("output", None)
                item["status"] = "FAILED"
                item["error_type"] = classify_error(error_message)
                item["error_message"] = error_message
                item["retry_count"] = int(item.get("retry_count", 0) or 0)
                item["last_retry"] = item.get("last_retry") or utc_now_iso()
            else:
                item["status"] = "SUCCESS"
            changed = True
        if changed and flush:
            self._flush()
        return changed

    def warm(self, agents, df) -> None:
        """Generate all agent outputs once before training begins."""
        texts = df["text"].tolist()
        task_descriptions = df["task_description"].tolist()
        records = df.to_dict("records")
        for agent in agents:
            agent.predict_batch(texts, task_descriptions, records=records)
