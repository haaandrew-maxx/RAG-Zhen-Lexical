"""
PDF text extractor using pdfplumber.

For each PDF in the documents directory, extracts all text and writes a
sibling .txt file with the same stem. The .txt file is what rg/grep will
search against.

Tables are converted to pipe-separated rows so that cell values remain on
the same line and are searchable (e.g. "GWP | A1 | 12.3 kg CO2-eq").
Non-table text is extracted normally and interleaved in reading order.

If a .txt file is already up-to-date (mtime >= PDF mtime) it is skipped.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _table_to_lines(table: list[list[str | None]]) -> list[str]:
    """
    Convert a pdfplumber table (list of rows, each row a list of cell strings)
    into pipe-separated text lines.

    Empty/None cells are replaced with "-" so column alignment is preserved
    and ripgrep can still match a value next to its row/column header.
    """
    lines: list[str] = []
    for row in table:
        cells = [str(c).strip() if c is not None else "-" for c in row]
        # Skip fully-empty rows
        if all(c in ("", "-") for c in cells):
            continue
        lines.append(" | ".join(cells))
    return lines


def extract_pdf_to_text(pdf_path: Path, txt_path: Path) -> int:
    """
    Extract all text from *pdf_path* and write to *txt_path*.

    Tables are rendered as pipe-separated rows so cell values are co-located
    with their headers and searchable by ripgrep.

    Returns the number of pages processed.
    Raises on unrecoverable errors.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required: pip install pdfplumber")

    parts: list[str] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_parts: list[str] = []

            # Extract tables first, collect their bounding boxes so we can
            # exclude those regions from the plain-text extraction.
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            for table_obj in tables:
                rows = table_obj.extract()
                if rows:
                    table_lines = _table_to_lines(rows)
                    if table_lines:
                        page_parts.append("\n".join(table_lines))

            # Extract non-table text by cropping out table areas.
            remaining = page
            for bbox in table_bboxes:
                try:
                    remaining = remaining.outside_bbox(bbox)
                except Exception:
                    pass

            try:
                text = remaining.extract_text() or ""
            except Exception:
                text = ""

            if text.strip():
                page_parts.insert(0, text)  # text before tables in reading order

            if page_parts:
                parts.append(f"[Page {page_num}]\n" + "\n\n".join(page_parts))

    full_text = "\n\n".join(parts)
    txt_path.write_text(full_text, encoding="utf-8")
    return len(parts)


def extract_all_pdfs(
    documents_dir: Path,
    *,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Walk *documents_dir* recursively and extract text for every PDF that
    does not yet have an up-to-date .txt counterpart.

    Parameters
    ----------
    documents_dir:
        Root directory to scan.
    force:
        If True, re-extract even if the .txt file is already current.
    verbose:
        Print progress to stdout.

    Returns
    -------
    dict mapping pdf_path -> status  ("extracted" | "skipped" | "error: ...")
    """
    results: dict[str, str] = {}

    pdf_files = sorted(documents_dir.rglob("*.pdf"))
    if not pdf_files:
        if verbose:
            print(f"[ingest] No PDF files found in {documents_dir}")
        return results

    for pdf_path in pdf_files:
        txt_path = pdf_path.with_suffix(".txt")

        # Skip if already extracted and not stale
        if not force and txt_path.exists():
            if txt_path.stat().st_mtime >= pdf_path.stat().st_mtime:
                results[str(pdf_path)] = "skipped"
                if verbose:
                    print(f"  [skip]    {pdf_path.name}")
                continue

        try:
            pages = extract_pdf_to_text(pdf_path, txt_path)
            results[str(pdf_path)] = "extracted"
            if verbose:
                print(f"  [ok]      {pdf_path.name}  ({pages} pages → {txt_path.name})")
        except Exception as exc:
            results[str(pdf_path)] = f"error: {exc}"
            if verbose:
                print(f"  [ERROR]   {pdf_path.name}: {exc}", file=sys.stderr)

    return results
