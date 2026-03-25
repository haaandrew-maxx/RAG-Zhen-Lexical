"""
JSON utility helpers: robust parsing with fallback extraction.
"""

from __future__ import annotations

import json
from typing import Any

from utils.text_utils import extract_json_block


def parse_json_strict(text: str) -> Any:
    """
    Parse *text* as JSON.
    Raises ``json.JSONDecodeError`` on failure.
    """
    return json.loads(text)


def parse_json_lenient(text: str) -> Any:
    """
    Try to parse JSON from *text*, extracting the first JSON block if needed.
    Raises ``json.JSONDecodeError`` if nothing can be parsed.
    """
    # First attempt: parse as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second attempt: extract embedded JSON block
    extracted = extract_json_block(text)
    return json.loads(extracted)


def to_json_str(obj: Any, *, indent: int | None = None) -> str:
    """Serialize *obj* to a JSON string."""
    return json.dumps(obj, ensure_ascii=False, indent=indent)


def safe_get(obj: Any, *keys: str | int, default: Any = None) -> Any:
    """
    Navigate nested dicts/lists with a key path.
    Returns *default* if any key is missing or the type is wrong.
    """
    current = obj
    for key in keys:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return default
    return current
