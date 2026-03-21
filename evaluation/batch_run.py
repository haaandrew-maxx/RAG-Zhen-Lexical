"""
Batch evaluation runner.

Reads a ground truth CSV, runs the RAG pipeline for each row, and writes
results to logs/eval_results.jsonl.

Usage:
    python evaluation/batch_run.py
    python evaluation/batch_run.py --input EPD-questions-ground_truths.csv
    python evaluation/batch_run.py --input EPD-questions-ground_truths.csv --limit 20 --shuffle
    python evaluation/batch_run.py --score-after-run
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Any

# Add project root to path so all internal imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

config.validate()

from agent.pipeline import RAGPipeline
from rag_logging.rag_logger import log_eval_result

try:
    from tqdm import tqdm as _tqdm_cls

    _TQDM = True
except ImportError:
    _tqdm_cls = None  # type: ignore[assignment]
    _TQDM = False


QUESTION_ALIASES = ["question", "questions", "query", "pregunta", "q"]
GT_ALIASES = [
    "ground_truth",
    "ground_truths",
    "ground truth",
    "groundtruth",
    "groundtruths",
    "answer",
    "respuesta",
    "expected",
    "reference",
    "reference_answer",
    "gold",
    "gold_answer",
    "gt",
]


# ── CSV loading ────────────────────────────────────────────────────────────────


def _detect_delimiter(path: str) -> str:
    """Sniff the CSV delimiter (comma or semicolon)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        return dialect.delimiter
    except csv.Error:
        return ","


def discover_groundtruth_file(project_root: Path) -> Path | None:
    """Find a likely ground truth CSV in the project root."""
    csv_files = sorted(project_root.glob("*.csv"))
    matches = [
        path
        for path in csv_files
        if "ground" in path.name.lower() and "truth" in path.name.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        return None

    exact_name_order = [
        "epd-questions-ground_truths.csv",
        "groundtruth.csv",
        "ground_truth.csv",
    ]
    ranked: list[Path] = []
    for name in exact_name_order:
        ranked.extend(path for path in matches if path.name.lower() == name)

    if ranked:
        return ranked[0]

    return None


def load_groundtruth(path: str) -> list[dict[str, Any]]:
    """
    Load a ground truth CSV with robust delimiter and column-name handling.

    Accepted question column names  (case-insensitive, tried in order):
        question, questions, query, pregunta, q
    Accepted ground_truth column names:
        ground_truth, ground_truths, ground truth, groundtruth, groundtruths,
        answer, respuesta, expected, reference, reference_answer, gold,
        gold_answer, gt
    Returns list of dicts with normalised keys: id, question, ground_truth.
    """
    delimiter = _detect_delimiter(path)
    rows: list[dict[str, Any]] = []

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        # Build a lower-cased alias map from actual header names
        header_map: dict[str, str] = {}
        if reader.fieldnames:
            for col in reader.fieldnames:
                header_map[col.strip().lower()] = col

        # Resolve which actual column to use for question / ground_truth
        question_col = next(
            (header_map[a] for a in QUESTION_ALIASES if a in header_map), None
        )
        gt_col = next((header_map[a] for a in GT_ALIASES if a in header_map), None)
        id_col = header_map.get("id")

        if question_col is None:
            available = list(header_map.keys())
            raise ValueError(
                f"Could not find a question column in {path}. "
                f"Available columns: {available}. "
                f"Recognised names: {QUESTION_ALIASES}"
            )

        for row_index, row in enumerate(reader, start=1):
            row_id = row.get(id_col, "").strip() if id_col else ""
            rows.append(
                {
                    "id": row_id or str(row_index),
                    "question": row.get(question_col, "").strip()
                    if question_col
                    else "",
                    "ground_truth": row.get(gt_col, "").strip() if gt_col else "",
                }
            )
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────


def run_batch(
    input_path: str,
    limit: int | None = None,
    shuffle: bool = False,
) -> None:
    rows = load_groundtruth(input_path)

    if shuffle:
        random.shuffle(rows)

    if limit is not None:
        rows = rows[:limit]

    # Truncate the output file so we never mix old and new runs
    import config as _cfg

    _cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    open(_cfg.EVAL_RESULTS_PATH, "w", encoding="utf-8").close()

    pipeline = RAGPipeline()
    iterator: Any = rows

    if _TQDM and _tqdm_cls is not None:
        iterator = _tqdm_cls(rows, desc="Evaluating", unit="question")

    for row in iterator:
        question = row["question"]
        if not question:
            continue

        try:
            result = pipeline.run(question)
        except Exception as exc:  # noqa: BLE001
            result = {
                "status": "ERROR",
                "answer": f"Pipeline error: {exc}",
                "claims": [],
                "contexts": [],
                "sources": [],
            }

        entry = {
            "id": row["id"],
            "question": question,
            "ground_truth": row["ground_truth"],
            "pred_answer": result.get("answer") or "",
            "status": result.get("status", ""),
            "contexts": result.get("contexts", []),
            "sources": result.get("sources", []),
        }
        log_eval_result(entry)

    print(f"[batch_run] Done. Results written to {config.EVAL_RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch evaluation against a ground truth CSV"
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to the ground truth CSV. If omitted, the runner will try to "
            "auto-detect a file such as EPD-questions-ground_truths.csv in the "
            "project root."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to evaluate",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows before applying --limit",
    )
    parser.add_argument(
        "--score-after-run",
        action="store_true",
        help="Run evaluation/evaluate_accuracy.py immediately after batch generation",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else discover_groundtruth_file(config.PROJECT_ROOT)

    if input_path is None:
        print(
            "[batch_run] ERROR: no ground truth CSV was provided and auto-detection "
            "did not find a unique match in the project root.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not input_path.exists():
        print(f"[batch_run] ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[batch_run] Using ground truth file: {input_path}")
    run_batch(str(input_path), limit=args.limit, shuffle=args.shuffle)

    if args.score_after_run:
        from evaluation.evaluate_accuracy import evaluate

        evaluate(str(config.EVAL_RESULTS_PATH))


if __name__ == "__main__":
    main()
