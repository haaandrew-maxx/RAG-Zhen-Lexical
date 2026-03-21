"""
ingest.py — pre-process documents for lexical search.

Extracts text from all PDF files in documents/ and writes sibling .txt files.
Run this once before using the agent, and again whenever new PDFs are added.

Usage:
    python ingest.py                    # extract only new/updated PDFs
    python ingest.py --force            # re-extract all PDFs
    python ingest.py --dir /other/path  # use a different documents directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from utils.pdf_extractor import extract_all_pdfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from PDF documents for lexical search"
    )
    parser.add_argument(
        "--dir",
        default=str(config.DOCUMENTS_DIR),
        help=f"Documents directory (default: {config.DOCUMENTS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract all PDFs even if .txt files already exist",
    )
    args = parser.parse_args()

    documents_dir = Path(args.dir)
    if not documents_dir.exists():
        print(f"[ingest] ERROR: directory not found: {documents_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] Extracting PDFs from: {documents_dir}")
    if args.force:
        print("[ingest] --force: re-extracting all PDFs")

    results = extract_all_pdfs(documents_dir, force=args.force, verbose=True)

    extracted = sum(1 for s in results.values() if s == "extracted")
    skipped = sum(1 for s in results.values() if s == "skipped")
    errors = sum(1 for s in results.values() if s.startswith("error"))

    print(
        f"\n[ingest] Done. Extracted: {extracted}  Skipped: {skipped}  Errors: {errors}"
    )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
