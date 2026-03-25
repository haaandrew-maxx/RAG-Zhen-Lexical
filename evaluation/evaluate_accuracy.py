"""
Answer accuracy evaluator — powered by RAGAS AnswerAccuracy + DeepSeek.

Reads logs/eval_results.jsonl, scores each (question, pred_answer, ground_truth)
triple with RAGAS AnswerAccuracy (LLM-based), then writes logs/accuracy_report.csv.

Scoring:
    binary = 1 if score.value >= 0.5 else 0

Usage:
    python evaluation/evaluate_accuracy.py
    python evaluation/evaluate_accuracy.py --input logs/eval_results.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

config.validate()

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy


# ── LLM setup (DeepSeek via OpenAI-compat API) ────────────────────────────────


def _build_scorer() -> AnswerAccuracy:
    client = AsyncOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    llm = llm_factory(config.DEEPSEEK_MODEL, client=client)
    return AnswerAccuracy(llm=llm)


# ── Loading ────────────────────────────────────────────────────────────────────


def load_eval_results(path: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


# ── Report ─────────────────────────────────────────────────────────────────────


def evaluate(input_path: str) -> None:
    if not Path(input_path).exists():
        print(
            f"[evaluate_accuracy] ERROR: file not found: {input_path}", file=sys.stderr
        )
        sys.exit(1)

    entries = load_eval_results(input_path)
    if not entries:
        print("[evaluate_accuracy] No results found.", file=sys.stderr)
        sys.exit(1)

    scorer = _build_scorer()

    rows: list[dict[str, Any]] = []
    for r in entries:
        question = r.get("question") or ""
        pred = r.get("pred_answer") or ""
        gt = r.get("ground_truth") or ""

        if not pred or not gt:
            raw_score = 0.0
        else:
            print(f"\nQuestion: {question}")
            score_obj = scorer.score(
                user_input=question,
                response=pred,
                reference=gt,
            )
            raw_score = score_obj.value
            print(f" Raw score: {raw_score:.4f}")

        binary = 1 if raw_score >= 0.5 else 0
        print(f" Binary score: {binary} ({'✓ Correct' if binary == 1 else '✗ Incorrect'})")

        rows.append(
            {
                "id": r.get("id", ""),
                "score": round(raw_score, 4),
                "binary": binary,
                "question": question,
                "ground_truth": gt,
                "pred_answer": pred,
                "status": r.get("status", ""),
            }
        )

    total = len(rows)
    avg_score = sum(r["score"] for r in rows) / total if total > 0 else 0.0
    correct = sum(r["binary"] for r in rows)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n==============================")
    print("RAG Accuracy Evaluation Result")
    print("==============================")
    print(f"Total evaluated: {total}")
    print(f"Average score:   {avg_score:.4f}")
    print(f"Correct:         {correct} / {total}")
    print(f"Accuracy rate:   {correct / total:.2%}" if total > 0 else "Accuracy rate: N/A")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "score",
        "binary",
        "question",
        "ground_truth",
        "pred_answer",
        "status",
    ]

    with open(config.ACCURACY_REPORT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport saved to {config.ACCURACY_REPORT_PATH}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute answer accuracy using RAGAS AnswerAccuracy + DeepSeek"
    )
    parser.add_argument(
        "--input",
        default=str(config.EVAL_RESULTS_PATH),
        help="Path to eval_results.jsonl (default: logs/eval_results.jsonl)",
    )
    args = parser.parse_args()
    evaluate(args.input)


if __name__ == "__main__":
    main()
