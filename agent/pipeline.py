"""
RAG Pipeline — orchestrates the full question-answering flow:
  1. Analyze the question (QuestionAnalyzer)
  2. Search for relevant documents (SearchEngine)
  3. Read evidence windows (EvidenceReader)
  4. Generate a cited answer (AnswerGenerator)
  5. Return a machine-parseable output dict
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.question_analyzer import QuestionAnalyzer
from agent.answer_generator import AnswerGenerator, Claim, EvidenceItem
from retrieval.search_engine import SearchEngine
from retrieval.evidence_reader import EvidenceReader, EvidenceWindow


# ── Pipeline ───────────────────────────────────────────────────────────────────


class RAGPipeline:
    """
    End-to-end Lexical Retrieval + Evidence-Gated Generative QA pipeline.
    """

    def __init__(self) -> None:
        self.question_analyzer = QuestionAnalyzer()
        self.search_engine = SearchEngine()
        self.evidence_reader = EvidenceReader()
        self.answer_generator = AnswerGenerator()

    def run(self, question: str) -> dict[str, Any]:
        """
        Run the full pipeline for *question*.

        Returns
        -------
        dict
            {
              "status": "OK|NOT_FOUND_IN_DOCS|PARTIALLY_SUPPORTED",
              "question": str,
              "answer": str | None,
              "claims": [...],
              "contexts": [str, ...],
              "sources": [{"file_path": str, "location": {...}}, ...],
              "timestamp": ISO-8601 string
            }
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # ── Step 1: analyze question ──────────────────────────────────────────
        analysis = self.question_analyzer.analyze(question)

        # ── Step 2: search ────────────────────────────────────────────────────
        hits = self.search_engine.search(
            search_terms=analysis.search_terms,
            regex_patterns=analysis.regex_patterns,
        )

        # ── Step 3: read evidence windows ─────────────────────────────────────
        evidence_windows = self.evidence_reader.read(hits)

        # ── Step 4: generate answer ───────────────────────────────────────────
        generated = self.answer_generator.generate(question, evidence_windows)

        # ── Step 5: assemble output ───────────────────────────────────────────
        return self._assemble(
            question=question,
            generated=generated,
            evidence_windows=evidence_windows,
            timestamp=timestamp,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _assemble(
        question: str,
        generated: Any,
        evidence_windows: list[EvidenceWindow],
        timestamp: str,
    ) -> dict[str, Any]:
        # Serialise claims
        claims_out = []
        for claim in generated.claims:
            claims_out.append(
                {
                    "text": claim.text,
                    "evidence": [
                        {"context_id": ev.context_id, "quote": ev.quote}
                        for ev in claim.evidence
                    ],
                }
            )

        # contexts: list of context strings (one per evidence window)
        contexts_out = [w.context for w in evidence_windows]

        # sources: unique file/location entries
        seen_sources: set[str] = set()
        sources_out = []
        for w in evidence_windows:
            key = f"{w.file_path}:{w.location['start_line']}"
            if key not in seen_sources:
                seen_sources.add(key)
                sources_out.append({"file_path": w.file_path, "location": w.location})

        return {
            "status": generated.status,
            "question": question,
            "answer": generated.answer if generated.answer else None,
            "claims": claims_out,
            "contexts": contexts_out,
            "sources": sources_out,
            "timestamp": timestamp,
        }
