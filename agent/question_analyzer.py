"""
Question Analyzer — uses the DeepSeek LLM to extract search terms, regex patterns,
and question type from a natural-language query.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.client import chat_completion
from llm.prompts import QUESTION_ANALYZER_PROMPT
from utils.json_utils import parse_json_lenient
from utils.text_utils import safe_regex


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class QuestionAnalysis:
    """Structured output of the question analyzer."""

    question_type: str  # factual | rule | procedure | table | other
    search_terms: list[str] = field(default_factory=list)
    regex_patterns: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    raw_question: str = ""


# ── Analyzer ───────────────────────────────────────────────────────────────────


class QuestionAnalyzer:
    """
    Calls the LLM to decompose the user question into structured search directives.
    Falls back to a simple keyword-based analysis if the LLM call fails.
    """

    def analyze(self, question: str) -> QuestionAnalysis:
        """
        Analyze *question* and return a ``QuestionAnalysis``.

        Parameters
        ----------
        question:
            The raw user question string.

        Returns
        -------
        QuestionAnalysis
        """
        try:
            return self._llm_analyze(question)
        except Exception as exc:  # noqa: BLE001
            # Fallback: basic keyword extraction without LLM
            return self._fallback_analyze(question)

    # ── Private ───────────────────────────────────────────────────────────────

    def _llm_analyze(self, question: str) -> QuestionAnalysis:
        messages = [
            {"role": "system", "content": QUESTION_ANALYZER_PROMPT},
            {"role": "user", "content": question},
        ]
        raw = chat_completion(messages)  # type: ignore[arg-type]
        data = parse_json_lenient(raw)

        question_type = str(data.get("question_type", "other")).lower()
        search_terms = [str(t) for t in data.get("search_terms", [])]
        raw_patterns = [str(p) for p in data.get("regex_patterns", [])]
        sub_questions = [str(q) for q in data.get("sub_questions", [])]

        # Sanitise regex patterns
        regex_patterns = [safe_regex(p) for p in raw_patterns if p.strip()]

        # Guarantee at least some search terms from the question itself
        if not search_terms:
            search_terms = self._simple_keywords(question)

        return QuestionAnalysis(
            question_type=question_type,
            search_terms=search_terms,
            regex_patterns=regex_patterns,
            sub_questions=sub_questions,
            raw_question=question,
        )

    @staticmethod
    def _fallback_analyze(question: str) -> QuestionAnalysis:
        """Simple non-LLM fallback: split question into keywords (multilingual)."""
        import re

        # Stop words: English + Spanish
        stop = {
            # English
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "what", "which", "who",
            "where", "when", "how", "why", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "up", "about", "into", "through", "during",
            "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
            "her", "its", "our", "their", "and", "or", "but", "not", "no",
            "if", "then", "that", "this", "these", "those",
            # Spanish
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            "que", "qué", "cual", "cuál", "cuales", "cuáles",
            "como", "cómo", "cuando", "cuándo", "donde", "dónde",
            "quien", "quién", "quienes",
            "por", "para", "con", "sin", "sobre", "entre", "desde", "hasta",
            "durante", "ante", "bajo", "tras", "según", "segun",
            "del", "al",
            "ser", "estar", "tener", "hacer", "puede", "debe", "hay",
            "era", "fue", "son", "han", "sido", "tiene", "tiene",
            "sus", "esta", "este", "estos", "estas", "ese", "esa",
            "esos", "esas", "aquel", "aquella", "aquellos", "aquellas",
            "yo", "nos", "ellos", "ellas", "les", "más", "mas",
            "muy", "tan", "también", "tambien", "si", "ni",
            "pero", "sea", "hay", "les", "les",
        }
        # Unicode-aware tokeniser: matches runs of letters (incl. accented chars)
        tokens = re.findall(r"[^\W\d_]{3,}", question.lower(), re.UNICODE)
        keywords = [t for t in tokens if t not in stop]

        # Build simple patterns
        patterns = keywords[:4]

        return QuestionAnalysis(
            question_type="other",
            search_terms=keywords[:8],
            regex_patterns=patterns,
            sub_questions=[],
            raw_question=question,
        )

    @staticmethod
    def _simple_keywords(question: str) -> list[str]:
        import re

        return re.findall(r"[^\W\d_]{3,}", question.lower(), re.UNICODE)[:8]


