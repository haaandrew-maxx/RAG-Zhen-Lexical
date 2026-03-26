"""
Lexical search engine — NO embeddings, NO vector DB.

Steps:
  1. Filename search  (find -iname) — matches against .txt extracted files
  2. Content search   (rg / grep)   — searches only .txt files
  3. Rank hits by keyword coverage + hit count per file

NOTE: PDF files are NOT searched directly. Run `python ingest.py` first to
extract text from PDFs into sibling .txt files. The search engine then
operates exclusively on those .txt files.
"""

from __future__ import annotations

import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.shell_tools import (
    ShellHit,
    find_files_by_name,
    read_file_lines,
    search_content_rg,
)
from utils.text_utils import count_keyword_coverage, safe_regex


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class RankedHit:
    """A search hit enriched with a relevance score."""

    file_path: str
    line_number: int
    snippet: str
    score: float = 0.0
    source: str = "content"  # "content" | "filename"


# ── Search Engine ──────────────────────────────────────────────────────────────


class SearchEngine:
    """
    Performs pure lexical document search against ``config.DOCUMENTS_DIR``.

    Parameters
    ----------
    documents_dir:
        Override the default documents directory.
    max_hits_per_file:
        Maximum hits to keep per file before ranking.
    """

    def __init__(
        self,
        documents_dir: str | Path | None = None,
        max_hits_per_file: int | None = None,
    ) -> None:
        self.documents_dir = Path(documents_dir or config.DOCUMENTS_DIR)
        self.max_hits_per_file = max_hits_per_file or config.MAX_HITS_PER_FILE

    # ── Public ────────────────────────────────────────────────────────────────

    def search(
        self,
        search_terms: list[str],
        regex_patterns: list[str],
        *,
        max_results: int | None = None,
    ) -> list[RankedHit]:
        """
        Run filename + content search, then rank and return deduplicated hits.

        Parameters
        ----------
        search_terms:
            Short keyword strings for filename search and scoring.
        regex_patterns:
            Regex patterns for content search.
        max_results:
            Cap on total returned hits (default: MAX_EVIDENCE_WINDOWS * 3).

        Returns
        -------
        list[RankedHit]
            Sorted descending by relevance score.
        """
        if not self.documents_dir.exists():
            return []

        max_results = max_results or (config.MAX_EVIDENCE_WINDOWS * 3)

        # Sanitize patterns
        safe_patterns = [safe_regex(p) for p in regex_patterns if p.strip()]

        # Step 1: filename search (with targeted in-file content search)
        filename_hits = self._filename_search(search_terms, safe_patterns)

        # Step 2: content search
        content_hits = self._content_search(safe_patterns)

        # Step 3: merge and rank
        ranked = self._rank(
            filename_hits + content_hits,
            search_terms,
            safe_patterns,
        )

        return ranked[:max_results]

    # ── Private ───────────────────────────────────────────────────────────────

    def _filename_search(
        self,
        search_terms: list[str],
        safe_patterns: list[str] | None = None,
    ) -> list[RankedHit]:
        """
        Find files whose names match the search terms.

        For each matched file, attempt a targeted content search using the
        provided *safe_patterns* so that the returned hits point at the
        relevant line inside the file rather than always line 0 (the TOC).
        Falls back to a line-0 placeholder hit only when no content match
        is found inside the file.
        """
        hits: list[RankedHit] = []
        # Prioritise terms for filename search.
        #
        # Tier ordering (processed first = gets filename-source bonus first):
        #   Tier 0: compound terms — contain BOTH a 4+ digit numeric ID AND
        #           at least one letter word (4+ chars).  These are the most
        #           specific: e.g. "1005808 Panel táctil" identifies a single file.
        #   Tier 1: pure numeric ID terms (no letter words) or all-caps codes.
        #           These match all files sharing the same ID prefix.
        #   Tier 2: everything else.
        #
        # Processing compound terms first lets us record which numeric IDs have
        # already been "specifically resolved" so that later generic-ID terms
        # skip adding ancillary files (e.g. spare-parts lists, maintenance plans)
        # when the question clearly targets one document sub-type.

        import re as _stem_re

        def _term_priority(t: str) -> tuple[int, int]:
            has_num  = bool(_stem_re.search(r"\d{4,}", t))
            has_word = bool(_stem_re.search(r"[A-Za-z\u00C0-\u024F]{4,}", t))
            is_allcaps = bool(_stem_re.match(r"^[A-Z]{2,}", t.strip()))
            if has_num and has_word:
                tier = 0   # compound: most specific
            elif has_num or is_allcaps:
                tier = 1   # generic ID / code
            else:
                tier = 2
            return (tier, -len(t))

        terms_to_try = sorted(search_terms, key=_term_priority)[:8]
        seen_files: set[str] = set()
        # Numeric IDs whose document type is already resolved by a compound term.
        # Generic-ID-only terms will skip files from these IDs.
        covered_ids: set[str] = set()

        for term in terms_to_try:
            if not term.strip():
                continue

            numeric_match = _stem_re.search(r"\d{4,}", term)
            if numeric_match:
                numeric_id = numeric_match.group(0)
                stems = [numeric_id]
                # Extract non-numeric words (4+ chars) — used for document-type filtering.
                # E.g. "1005808 Panel táctil MTPE" → type_words = ["Panel", "táctil", "MTPE"]
                type_words = [
                    w for w in _stem_re.findall(r"[A-Za-z\u00C0-\u024F]{4,}", term)
                    if not w.isdigit()
                ]
            elif _stem_re.search(r"[ _+/]", term):
                numeric_id = None
                type_words = []
                words = _stem_re.findall(r"[A-Za-z\u00C0-\u024F]{5,}", term)
                stems = list(dict.fromkeys(words[:3])) or [term.split()[0]]
            else:
                numeric_id = None
                type_words = []
                stems = [term]

            # --- Skip generic ID terms whose ID was already resolved by a compound term ---
            if numeric_id and not type_words and numeric_id in covered_ids:
                continue

            # --- Find candidate files ---
            paths: list[str] = []
            for stem in stems:
                paths.extend(find_files_by_name(self.documents_dir, stem, extension=".txt"))

            # --- For compound terms: filter to strongly-matching files only ---
            # A "strong" match has BOTH the numeric ID AND at least one type word
            # in the filename (e.g. "táctil" in "1005808 02 Panel táctil.txt").
            # Files matching only the ID are dropped from filename search; they
            # will be picked up by the global content search if their content
            # matches. This prevents spare-parts lists / maintenance plans from
            # consuming evidence windows when the question is about a specific
            # document type.
            if numeric_id and type_words:
                strong = [
                    fp for fp in paths
                    if any(w.lower() in Path(fp).stem.lower() for w in type_words)
                ]
                if strong:
                    paths = strong
                    covered_ids.add(numeric_id)   # mark this ID as resolved

            for fp in paths:
                if fp not in seen_files:
                    seen_files.add(fp)
                    content_hits = self._content_hits_in_file(fp, safe_patterns or [])
                    if content_hits:
                        for h in content_hits:
                            h.source = "filename"
                            hits.append(h)
                    else:
                        hits.append(
                            RankedHit(
                                file_path=fp,
                                line_number=0,
                                snippet=f"[filename match: {Path(fp).name}]",
                                source="filename",
                            )
                        )
        return hits

    def _content_hits_in_file(
        self,
        file_path: str,
        safe_patterns: list[str],
    ) -> list[RankedHit]:
        """Run content search restricted to a single file."""
        if not safe_patterns:
            return []
        raw_hits = search_content_rg(
            Path(file_path).parent,
            safe_patterns,
            max_results=self.max_hits_per_file,
            glob_pattern=Path(file_path).name,
        )
        return [
            RankedHit(
                file_path=h.file_path,
                line_number=h.line_number,
                snippet=h.snippet,
                source="content",
            )
            for h in raw_hits
        ]

    def _content_search(self, safe_patterns: list[str]) -> list[RankedHit]:
        if not safe_patterns:
            return []

        raw_hits: list[ShellHit] = search_content_rg(
            self.documents_dir,
            safe_patterns,
            max_results=500,
            # Only search .txt files — avoids binary PDF noise
            glob_pattern="*.txt",
            # Cap per-file hits so that alphabetically-early files with hundreds
            # of matches (e.g. Plano eléctrico) cannot crowd out later relevant
            # files (e.g. the correct DGUV inspection report).
            per_file_limit=self.max_hits_per_file * 3,
        )

        return [
            RankedHit(
                file_path=h.file_path,
                line_number=h.line_number,
                snippet=h.snippet,
                source="content",
            )
            for h in raw_hits
        ]

    def _rank(
        self,
        hits: list[RankedHit],
        search_terms: list[str],
        safe_patterns: list[str],
    ) -> list[RankedHit]:
        """
        Score each hit by keyword coverage and aggregate per file.

        Strategy:
        - Score each hit by keyword coverage + file density + source bonus.
        - Filename-matched hits get a large bonus so they rank above
          content-only hits from unrelated files.
        - Density is normalised by file size so that large multi-document
          index files (e.g. 424 KB spare-parts lists) cannot crowd out
          smaller focused files (e.g. 12 KB DGUV inspection reports).
        - Limit hits per file to avoid one file dominating.

        Returns deduplicated hits sorted by score (desc).
        """
        import os as _os
        import math as _math

        hits_per_file: dict[str, int] = defaultdict(int)
        for h in hits:
            hits_per_file[h.file_path] += 1

        # File-size penalty reference: files larger than this get a log2 penalty
        # on their density score.  Files at or below this size get no penalty.
        _REF_BYTES = 50_000  # 50 KB

        file_size_cache: dict[str, int] = {}

        def _size_penalty(fp: str) -> float:
            """Return a multiplier in (0, 1] that demotes large files."""
            if fp not in file_size_cache:
                try:
                    file_size_cache[fp] = _os.path.getsize(fp)
                except OSError:
                    file_size_cache[fp] = 0
            sz = file_size_cache[fp]
            if sz <= _REF_BYTES:
                return 1.0
            log_penalty = _math.log2(sz / _REF_BYTES)
            return 1.0 / (1.0 + log_penalty)

        scored: list[RankedHit] = []
        seen: set[tuple[str, int]] = set()

        for h in hits:
            key = (h.file_path, h.line_number)
            if key in seen:
                continue
            seen.add(key)

            # Keyword coverage of the snippet — capped at 2.0 to prevent
            # header lines in parts lists from dominating.
            coverage = min(count_keyword_coverage(h.snippet, search_terms), 2.0)
            # Normalised hit density (more hits = higher but diminishing return)
            density = min(hits_per_file[h.file_path], 20) / 20.0
            # Apply file-size penalty: large files get reduced density contribution
            density_adj = density * _size_penalty(h.file_path)
            # Filename hits get a large bonus so they always rank above
            # content-only hits from unrelated files.
            source_bonus = 3.0 if h.source == "filename" else 0.0
            h.score = coverage + density_adj + source_bonus
            scored.append(h)

        scored.sort(key=lambda x: x.score, reverse=True)

        # Limit hits per file to max_hits_per_file to avoid one file dominating
        file_count: dict[str, int] = defaultdict(int)
        result: list[RankedHit] = []
        for h in scored:
            if file_count[h.file_path] < self.max_hits_per_file:
                result.append(h)
                file_count[h.file_path] += 1

        return result
