"""
Shell utilities for running find / rg / grep.
All document-search commands must go through this module.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class ShellHit:
    """A single match returned by a content or filename search."""

    file_path: str
    line_number: int  # 0 for filename-only hits
    snippet: str


# ── Internal helpers ───────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run *cmd* and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"


def _rg_available() -> bool:
    return shutil.which("rg") is not None


# ── Public API ─────────────────────────────────────────────────────────────────


def find_files_by_name(
    directory: str | Path,
    pattern: str,
    *,
    max_results: int = 50,
    extension: str | None = None,
) -> list[str]:
    """
    Return file paths whose names match *pattern* (case-insensitive).
    Uses ``find -iname``.

    Parameters
    ----------
    directory:
        Root directory to search.
    pattern:
        Substring to match against filenames (case-insensitive).
    max_results:
        Maximum number of paths to return.
    extension:
        If provided (e.g. ``".txt"``), only files with this extension are
        returned. This prevents matching raw binary PDFs when we only want
        the extracted text counterparts.
    """
    directory = str(directory)
    # Build the -iname glob
    iname_glob = f"*{pattern}*"
    if extension:
        # Restrict to the given extension by appending it; this handles the
        # common case where the stem contains the pattern.
        # We also accept files whose name ends with extension regardless.
        cmd = [
            "find",
            directory,
            "-type",
            "f",
            "(",
            "-iname",
            iname_glob,
            "-o",
            "-iname",
            f"*{extension}",
            ")",
            "-iname",
            f"*{extension}",
        ]
        # Simpler: just find by extension and filter in Python
        cmd = ["find", directory, "-type", "f", "-iname", f"*{extension}"]
        _, stdout, _ = _run(cmd)
        paths = [
            line.strip()
            for line in stdout.splitlines()
            if line.strip() and pattern.lower() in Path(line.strip()).stem.lower()
        ]
    else:
        cmd = ["find", directory, "-type", "f", "-iname", iname_glob]
        _, stdout, _ = _run(cmd)
        paths = [line.strip() for line in stdout.splitlines() if line.strip()]

    return paths[:max_results]


def search_content_rg(
    directory: str | Path,
    patterns: list[str],
    *,
    max_results: int = 200,
    context_lines: int = 0,
    glob_pattern: str | None = None,
    per_file_limit: int | None = None,
) -> list[ShellHit]:
    """
    Search file contents using ripgrep (preferred) or grep fallback.

    Each pattern is run as a **separate** rg invocation and the results are
    merged and deduplicated.  Running patterns separately avoids a Rust regex
    engine edge-case where combining non-ASCII patterns via ``|`` inside
    ``(?:...)`` groups with bounded quantifiers (e.g. ``{0,15}``) silently
    drops some matches.

    Parameters
    ----------
    directory:
        Root directory to search recursively.
    patterns:
        List of regex patterns.  Each pattern is searched independently.
    max_results:
        Cap on the number of returned hits (across all patterns).
    context_lines:
        Number of context lines before/after each match (ripgrep -C flag).
    glob_pattern:
        If provided, only files matching this glob are searched
        (e.g. ``"*.txt"`` to skip binary PDFs).
    """
    if not patterns:
        return []

    if _rg_available():
        # Run each pattern separately to avoid rg regex-engine quirks with
        # combined non-ASCII patterns.
        seen: set[tuple[str, int]] = set()
        hits: list[ShellHit] = []
        per_pattern_limit = max(max_results, 50)  # generous per-pattern budget

        for pattern in patterns:
            if not pattern.strip():
                continue
            new_hits = _rg_search(
                str(directory),
                pattern,  # single pattern — no wrapping needed
                context_lines,
                per_pattern_limit,
                glob_pattern=glob_pattern,
                per_file_limit=per_file_limit,
            )
            for h in new_hits:
                key = (h.file_path, h.line_number)
                if key not in seen:
                    seen.add(key)
                    hits.append(h)
            if len(hits) >= max_results * 3:
                break  # safety cap before deduplication

        return hits[:max_results]
    else:
        combined_pattern = "|".join(f"(?:{p})" for p in patterns)
        return _grep_search(
            str(directory),
            combined_pattern,
            max_results,
            include_glob=glob_pattern,
        )


def _rg_search(
    directory: str,
    combined_pattern: str,
    context_lines: int,
    max_results: int,
    *,
    glob_pattern: str | None = None,
    per_file_limit: int | None = None,
) -> list[ShellHit]:
    cmd = [
        "rg",
        "--line-number",  # include line numbers
        "--smart-case",  # case-insensitive when query is lowercase
        "--no-heading",  # one result per line
        "--with-filename",  # always show filename
    ]
    if glob_pattern:
        cmd += ["--glob", glob_pattern]
    if context_lines > 0:
        cmd += ["-C", str(context_lines)]
    if per_file_limit is not None:
        cmd += ["--max-count", str(per_file_limit)]
    cmd += ["-e", combined_pattern, directory]

    _, stdout, _ = _run(cmd)
    return _parse_rg_output(stdout, max_results)


def _parse_rg_output(stdout: str, max_results: int) -> list[ShellHit]:
    hits: list[ShellHit] = []
    for raw_line in stdout.splitlines():
        raw_line = raw_line.strip()
        if not raw_line or raw_line.startswith("--"):
            # "--" is the context separator in rg output
            continue
        # rg no-heading format: filepath:lineno:content
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            continue
        file_path, line_num_str, snippet = parts[0], parts[1], parts[2]
        try:
            line_number = int(line_num_str)
        except ValueError:
            continue
        hits.append(
            ShellHit(file_path=file_path, line_number=line_number, snippet=snippet)
        )
        if len(hits) >= max_results:
            break
    return hits


def _grep_search(
    directory: str,
    combined_pattern: str,
    max_results: int,
    *,
    include_glob: str | None = None,
) -> list[ShellHit]:
    include_flag = f"--include={include_glob}" if include_glob else "--include=*"
    cmd = [
        "grep",
        "-rE",
        include_flag,
        "-n",  # line numbers
        "-i",  # case-insensitive
        combined_pattern,
        directory,
    ]
    _, stdout, _ = _run(cmd)
    return _parse_grep_output(stdout, max_results)


def _parse_grep_output(stdout: str, max_results: int) -> list[ShellHit]:
    hits: list[ShellHit] = []
    for raw_line in stdout.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            continue
        file_path, line_num_str, snippet = parts[0], parts[1], parts[2]
        try:
            line_number = int(line_num_str)
        except ValueError:
            continue
        hits.append(
            ShellHit(file_path=file_path, line_number=line_number, snippet=snippet)
        )
        if len(hits) >= max_results:
            break
    return hits


def read_file_lines(
    file_path: str | Path,
    start_line: int,
    end_line: int,
) -> list[str]:
    """
    Read lines [start_line, end_line] (1-indexed, inclusive) from a file.
    Returns an empty list if the file cannot be read.
    """
    file_path = Path(file_path)
    try:
        all_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    # Convert to 0-indexed slice
    s = max(0, start_line - 1)
    e = min(len(all_lines), end_line)
    return all_lines[s:e]
