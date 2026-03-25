"""
Answer Generator — produces a generative, evidence-gated answer from retrieved contexts.

Responsibilities:
  1. Call the LLM with evidence windows to produce JSON (answer + claims).
  2. Validate every quote against the evidence contexts.
     Matching is whitespace-normalised: the LLM may collapse multi-line text into
     a single line, so we compare collapsed versions and recover the actual
     verbatim span from the original context.
  3. Attempt one automatic repair pass if validation fails.
  4. Return a structured result with status.
"""

from __future__ import annotations

import re
import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.client import chat_completion
from llm.prompts import ANSWER_GENERATOR_PROMPT, REPAIR_PROMPT
from retrieval.evidence_reader import EvidenceWindow
from utils.json_utils import parse_json_lenient, to_json_str


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class EvidenceItem:
    context_id: int
    quote: str  # verbatim substring recovered from context


@dataclass
class Claim:
    text: str
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass
class GeneratedAnswer:
    answer: str
    claims: list[Claim]
    status: str  # "OK" | "PARTIALLY_SUPPORTED" | "NOT_FOUND_IN_DOCS"
    validation_errors: list[str] = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _collapse(text: str) -> str:
    """Collapse all whitespace runs (spaces, tabs, newlines) into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _find_quote_in_context(quote: str, context: str) -> str | None:
    """
    Try to find *quote* in *context* using two strategies:

    1. Exact verbatim substring match.
    2. Whitespace-normalised match: collapse both to single spaces, find the
       position, then recover the actual span in the original context.

    Returns the verbatim quote string as it appears in *context*, or None.
    """
    # Strategy 1: exact match
    if quote in context:
        return quote

    # Strategy 2: whitespace-normalised match
    q_collapsed = _collapse(quote)
    c_collapsed = _collapse(context)

    if not q_collapsed or q_collapsed not in c_collapsed:
        return None

    # Recover the original span:
    # Find where in c_collapsed the match starts, then map back to original context.
    idx = c_collapsed.find(q_collapsed)
    if idx == -1:
        return None

    # Build a character mapping: collapsed_pos -> original_pos
    # Walk through context building collapsed version, tracking positions.
    orig_positions: list[
        int
    ] = []  # orig_positions[i] = original index for collapsed[i]
    prev_was_space = False
    for orig_i, ch in enumerate(context):
        if re.match(r"\s", ch):
            if not prev_was_space:
                orig_positions.append(orig_i)
                prev_was_space = True
        else:
            orig_positions.append(orig_i)
            prev_was_space = False

    # collapsed string starts with no leading space (strip was applied)
    # Find leading stripped offset in original
    stripped_start = len(context) - len(context.lstrip())

    if idx >= len(orig_positions) or idx + len(q_collapsed) - 1 >= len(orig_positions):
        # Fallback: return the collapsed quote itself if it's a substring of collapsed context
        return q_collapsed if q_collapsed in c_collapsed else None

    orig_start = orig_positions[idx]
    orig_end_collapsed_idx = idx + len(q_collapsed) - 1
    orig_end = orig_positions[orig_end_collapsed_idx]

    # Extend orig_end to include the full last word/character
    recovered = context[orig_start : orig_end + 1]
    return recovered if recovered.strip() else None


# ── Answer Generator ───────────────────────────────────────────────────────────


class AnswerGenerator:
    """
    Generates a cited answer from evidence windows.
    """

    def generate(
        self,
        question: str,
        evidence_windows: list[EvidenceWindow],
    ) -> GeneratedAnswer:
        if not evidence_windows:
            return GeneratedAnswer(
                answer="The requested information was not found in the available documents.",
                claims=[],
                status="NOT_FOUND_IN_DOCS",
            )

        context_map = {w.context_id: w.context for w in evidence_windows}
        contexts_text = self._format_contexts(evidence_windows)

        # ── First LLM call ────────────────────────────────────────────────────
        raw = self._call_llm(question, contexts_text)

        parsed, parse_errors = self._safe_parse(raw)
        if parsed is None:
            parsed, repair_errors = self._repair(
                question, contexts_text, raw, parse_errors
            )
            if parsed is None:
                return GeneratedAnswer(
                    answer="The requested information was not found in the available documents.",
                    claims=[],
                    status="NOT_FOUND_IN_DOCS",
                    validation_errors=parse_errors + repair_errors,
                )

        # ── Quote validation ──────────────────────────────────────────────────
        claims, validation_errors = self._validate_claims(
            parsed.get("claims", []), context_map
        )

        if validation_errors:
            # One repair attempt for invalid quotes
            repaired_parsed, _ = self._repair(
                question, contexts_text, to_json_str(parsed), validation_errors
            )
            if repaired_parsed is not None:
                repaired_claims, _ = self._validate_claims(
                    repaired_parsed.get("claims", []), context_map
                )
                # Only use repair if it produced more valid claims
                if len(repaired_claims) >= len(claims):
                    claims = repaired_claims
                    parsed = repaired_parsed

        answer_text = str(parsed.get("answer", "")).strip()

        if not claims:
            return GeneratedAnswer(
                answer=answer_text
                or "The requested information was not found in the available documents.",
                claims=[],
                status="NOT_FOUND_IN_DOCS",
                validation_errors=validation_errors,
            )

        all_orig_claims = parsed.get("claims", [])
        removed = len(all_orig_claims) - len(claims)
        status = "OK" if removed == 0 else "PARTIALLY_SUPPORTED"

        return GeneratedAnswer(
            answer=answer_text,
            claims=claims,
            status=status,
            validation_errors=validation_errors if removed > 0 else [],
        )

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_contexts(
        windows: list[EvidenceWindow],
        max_chars: int = 120_000,
    ) -> str:
        """
        Format evidence windows into a single string, enforcing a character budget.

        DeepSeek-chat context window is ~128 K tokens. We reserve around 30 K
        tokens for the system prompt, question, and completion, leaving
        ~100 K tokens for evidence. At ~4 chars/token for mixed-language
        technical text that is roughly 120 000 chars — the default here.

        Windows are included in order. If a single window would exceed the
        remaining budget it is truncated to fit (with a truncation marker),
        rather than skipped entirely.
        """
        parts = []
        total = 0
        separator = "\n\n---\n\n"
        sep_len = len(separator)

        for w in windows:
            header = (
                f"[Context ID: {w.context_id}] "
                f"(file: {w.file_path}, "
                f"lines {w.location['start_line']}-{w.location['end_line']})\n"
            )
            # Account for separators between parts
            overhead = sep_len if parts else 0
            remaining = max_chars - total - overhead - len(header)
            if remaining <= 0:
                break
            body = w.context
            if len(body) > remaining:
                body = body[:remaining] + "\n[...truncated]"
            chunk = header + body
            parts.append(chunk)
            total += overhead + len(chunk)

        return separator.join(parts)

    @staticmethod
    def _call_llm(question: str, contexts_text: str) -> str:
        user_message = f"QUESTION:\n{question}\n\nEVIDENCE CONTEXTS:\n{contexts_text}"
        messages = [
            {"role": "system", "content": ANSWER_GENERATOR_PROMPT},
            {"role": "user", "content": user_message},
        ]
        return chat_completion(messages)  # type: ignore[arg-type]

    @staticmethod
    def _safe_parse(raw: str) -> tuple[dict | None, list[str]]:
        try:
            data = parse_json_lenient(raw)
            if not isinstance(data, dict):
                return None, [
                    f"Top-level JSON is not an object, got {type(data).__name__}"
                ]
            return data, []
        except Exception as exc:
            return None, [f"JSON parse error: {exc}"]

    def _repair(
        self,
        question: str,
        contexts_text: str,
        invalid_response: str,
        errors: list[str],
    ) -> tuple[dict | None, list[str]]:
        # Truncate heavy fields to keep repair call within model context limits.
        # Repair already has the invalid response on top of contexts, so we use
        # a smaller budget for contexts here (≈ half of the main call budget).
        MAX_REPAIR_CONTEXTS = 60_000
        MAX_REPAIR_RESPONSE = 8_000
        truncated_contexts = (
            contexts_text[:MAX_REPAIR_CONTEXTS] + "\n[...truncated for repair]"
            if len(contexts_text) > MAX_REPAIR_CONTEXTS
            else contexts_text
        )
        truncated_response = (
            invalid_response[:MAX_REPAIR_RESPONSE] + "\n[...truncated]"
            if len(invalid_response) > MAX_REPAIR_RESPONSE
            else invalid_response
        )
        # Build repair prompt WITHOUT .format() to avoid KeyError on JSON braces
        errors_text = "\n".join(errors)
        prompt = (
            REPAIR_PROMPT.replace("{question}", question, 1)
            .replace("{contexts}", truncated_contexts, 1)
            .replace("{invalid_response}", truncated_response, 1)
            .replace("{errors}", errors_text, 1)
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = chat_completion(messages)  # type: ignore[arg-type]
            data, parse_errors = self._safe_parse(raw)
            return data, parse_errors
        except Exception as exc:
            return None, [f"Repair call failed: {exc}"]

    @staticmethod
    def _validate_claims(
        raw_claims: list,
        context_map: dict[int, str],
    ) -> tuple[list[Claim], list[str]]:
        """
        Validate each claim's evidence quotes against the context map.

        Uses whitespace-normalised matching so LLM-collapsed newlines don't
        cause false negatives. Returns (valid_claims, error_list).
        """
        valid_claims: list[Claim] = []
        errors: list[str] = []

        for i, raw_claim in enumerate(raw_claims):
            if not isinstance(raw_claim, dict):
                errors.append(f"Claim {i} is not a dict")
                continue

            claim_text = str(raw_claim.get("text", "")).strip()
            raw_evidence = raw_claim.get("evidence", [])

            if not isinstance(raw_evidence, list):
                errors.append(f"Claim {i} 'evidence' is not a list")
                continue

            valid_evidence: list[EvidenceItem] = []
            for j, ev in enumerate(raw_evidence):
                if not isinstance(ev, dict):
                    errors.append(f"Claim {i} evidence {j} is not a dict")
                    continue

                ctx_id = ev.get("context_id")
                quote = str(ev.get("quote", "")).strip()

                if ctx_id is None or not isinstance(ctx_id, int):
                    errors.append(
                        f"Claim {i} evidence {j}: context_id missing or not int (got {ctx_id!r})"
                    )
                    continue

                if ctx_id not in context_map:
                    errors.append(
                        f"Claim {i} evidence {j}: context_id {ctx_id} not in evidence windows"
                    )
                    continue

                if not quote:
                    errors.append(f"Claim {i} evidence {j}: quote is empty")
                    continue

                # Whitespace-tolerant match; recovers actual verbatim span
                recovered = _find_quote_in_context(quote, context_map[ctx_id])
                if recovered is None:
                    errors.append(
                        f"Claim {i} evidence {j}: quote not found in context {ctx_id} "
                        f"(even after whitespace normalisation). "
                        f"Quote: {quote[:80]!r}"
                    )
                    continue

                valid_evidence.append(EvidenceItem(context_id=ctx_id, quote=recovered))

            if valid_evidence:
                valid_claims.append(Claim(text=claim_text, evidence=valid_evidence))
            elif claim_text:
                errors.append(
                    f"Claim {i} ({claim_text[:50]!r}) has no valid evidence; dropped"
                )

        return valid_claims, errors
