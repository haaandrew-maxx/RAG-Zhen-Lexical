"""
Evidence reader — given ranked search hits, expand each hit into a context window
of ±N lines and return structured evidence windows.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.shell_tools import read_file_lines
from retrieval.search_engine import RankedHit


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class EvidenceWindow:
    """A context window extracted around a search hit."""

    context_id: int
    file_path: str
    location: dict  # {"start_line": int, "end_line": int}
    context: str  # multi-line text of the window


# ── Evidence Reader ────────────────────────────────────────────────────────────


class EvidenceReader:
    """
    Expands search hits into context windows.

    Parameters
    ----------
    context_lines:
        Number of lines to include before and after the hit line.
    max_windows:
        Maximum number of evidence windows to return.
    """

    def __init__(
        self,
        context_lines: int | None = None,
        max_windows: int | None = None,
    ) -> None:
        self.context_lines = context_lines or config.EVIDENCE_CONTEXT_LINES
        self.max_windows = max_windows or config.MAX_EVIDENCE_WINDOWS

    # ── Public ────────────────────────────────────────────────────────────────

    def read(self, hits: list[RankedHit]) -> list[EvidenceWindow]:
        """
        Build evidence windows for the top hits.

        Merges overlapping windows within the same file to avoid duplication.
        Assigns sequential context_id values starting at 0.

        Parameters
        ----------
        hits:
            Ranked search hits (already sorted by relevance).

        Returns
        -------
        list[EvidenceWindow]
            At most ``max_windows`` non-overlapping context windows.
        """
        # Collect raw windows per file
        # file_path -> list of (start_line, end_line)
        file_windows: dict[str, list[tuple[int, int]]] = {}

        for hit in hits:
            if hit.line_number == 0:
                # Filename-only hit: read the first N lines of the file
                start = 1
                end = self.context_lines * 2
            else:
                start = max(1, hit.line_number - self.context_lines)
                end = hit.line_number + self.context_lines

            if hit.file_path not in file_windows:
                file_windows[hit.file_path] = []
            file_windows[hit.file_path].append((start, end))

        # Merge overlapping ranges and materialise windows
        windows: list[EvidenceWindow] = []
        context_id = 0

        for file_path, ranges in file_windows.items():
            merged = self._merge_ranges(ranges)
            for start, end in merged:
                lines = read_file_lines(file_path, start, end)
                if not lines:
                    continue
                context_text = "\n".join(lines)
                windows.append(
                    EvidenceWindow(
                        context_id=context_id,
                        file_path=file_path,
                        location={
                            "start_line": start,
                            "end_line": start + len(lines) - 1,
                        },
                        context=context_text,
                    )
                )
                context_id += 1
                if len(windows) >= self.max_windows:
                    return windows

        return windows

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge overlapping or adjacent [start, end] line ranges."""
        if not ranges:
            return []
        sorted_ranges = sorted(ranges)
        merged: list[tuple[int, int]] = [sorted_ranges[0]]
        for start, end in sorted_ranges[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged
