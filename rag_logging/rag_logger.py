"""
RAGAS-style logger — appends one JSON line per query to logs/rag_logs.jsonl.
"""

from __future__ import annotations

import sys
import os
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Thread-safe write lock
_lock = threading.Lock()


def _ensure_log_dir() -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log_query(result: dict[str, Any]) -> None:
    """
    Append a RAGAS-style log entry to ``logs/rag_logs.jsonl``.

    Parameters
    ----------
    result:
        The pipeline output dict (from ``RAGPipeline.run``).
    """
    _ensure_log_dir()

    entry = {
        "question": result.get("question", ""),
        "answer": result.get("answer"),
        "claims": result.get("claims", []),
        "contexts": result.get("contexts", []),
        "sources": result.get("sources", []),
        "status": result.get("status", ""),
        "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }

    line = json.dumps(entry, ensure_ascii=False)

    with _lock:
        with open(config.RAG_LOGS_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def log_eval_result(entry: dict[str, Any]) -> None:
    """
    Append one evaluation result line to ``logs/eval_results.jsonl``.

    Parameters
    ----------
    entry:
        A dict with keys: id, question, ground_truth, pred_answer,
        status, contexts, sources.
    """
    _ensure_log_dir()
    line = json.dumps(entry, ensure_ascii=False)

    with _lock:
        with open(config.EVAL_RESULTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
