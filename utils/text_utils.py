"""
Text utility functions used across the project.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, max_chars: int = 500, suffix: str = "…") -> str:
    """Truncate *text* to at most *max_chars* characters."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def count_keyword_coverage(text: str, keywords: list[str]) -> int:
    """
    Return the number of distinct keywords (case-insensitive) found in *text*.
    """
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def safe_regex(pattern: str) -> str:
    """
    Return *pattern* if it compiles; otherwise return a regex-escaped literal.
    """
    try:
        re.compile(pattern)
        return pattern
    except re.error:
        return re.escape(pattern)


def extract_json_block(text: str) -> str:
    """
    Try to extract the first JSON object or array from *text*.
    Useful when the LLM leaks prose before/after the JSON.
    Returns the raw JSON string, or *text* unchanged if no block is found.
    """
    # Try to find ```json ... ``` fenced block first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Fall back: find the first { ... } or [ ... ] spanning multiple chars
    brace_match = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
    if brace_match:
        return brace_match.group(1).strip()

    return text


def is_substring(needle: str, haystack: str) -> bool:
    """Return True if *needle* appears verbatim in *haystack*."""
    return needle in haystack
