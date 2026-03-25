"""
DeepSeek LLM client using OpenAI-compatible SDK.
All credentials and model names are loaded from config.py.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

import sys
import os

# Allow running from anywhere by inserting the project root on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def _build_client() -> OpenAI:
    return OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )


# Module-level singleton – created lazily to allow tests to patch config first
_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def chat_completion(
    messages: list[ChatCompletionMessageParam],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    """
    Send a chat completion request and return the assistant message content.

    Parameters
    ----------
    messages:
        Standard OpenAI-format message list.
    model:
        Override the default model from config.
    temperature:
        Override the default temperature from config.
    max_tokens:
        Override the default max_tokens from config.
    **kwargs:
        Forwarded verbatim to the OpenAI client.

    Returns
    -------
    str
        The raw text of the first choice's message content.
    """
    response = get_client().chat.completions.create(
        model=model or config.DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
        **kwargs,
    )
    return response.choices[0].message.content or ""
