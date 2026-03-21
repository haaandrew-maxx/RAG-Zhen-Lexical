"""
main.py — CLI entry point for the Lexical RAG Agent.

Usage:
    python main.py "your question here"
    python main.py "your question" --pretty
    python main.py "your question" --no-log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where the script is invoked
sys.path.insert(0, str(Path(__file__).parent))

import config

# Validate required env vars before anything else
try:
    config.validate()
except EnvironmentError as e:
    print(f"[main] Configuration error: {e}", file=sys.stderr)
    sys.exit(1)

from agent.pipeline import RAGPipeline
from rag_logging.rag_logger import log_query


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lexical Retrieval + Evidence-Gated Generative QA Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What is the maximum loan amount for first-time homebuyers?"
  python main.py "How do I submit a reimbursement request?" --pretty
  python main.py "What are the vacation leave rules?" --no-log
""",
    )
    parser.add_argument(
        "question",
        help="The question to answer",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output (indent=2)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write to logs/rag_logs.jsonl",
    )
    args = parser.parse_args()

    question = args.question.strip()
    if not question:
        print("[main] ERROR: question cannot be empty.", file=sys.stderr)
        sys.exit(1)

    pipeline = RAGPipeline()

    try:
        result = pipeline.run(question)
    except Exception as exc:  # noqa: BLE001
        error_result = {
            "status": "ERROR",
            "question": question,
            "answer": f"An unexpected error occurred: {exc}",
            "claims": [],
            "contexts": [],
            "sources": [],
            "timestamp": "",
        }
        print(
            json.dumps(
                error_result, ensure_ascii=False, indent=2 if args.pretty else None
            )
        )
        sys.exit(1)

    # Log unless suppressed
    if not args.no_log:
        log_query(result)

    # Print to stdout
    indent = 2 if args.pretty else None
    print(json.dumps(result, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    main()
