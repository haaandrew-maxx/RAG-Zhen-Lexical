"""
Central configuration module.
Loads all environment variables from .env via python-dotenv.
All other modules must import from here — no scattered os.getenv calls.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the project root (the directory containing config.py)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Load .env from project root
_env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

# ── DeepSeek / LLM ────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ── Paths ──────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR: Path = PROJECT_ROOT / "documents"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
RAG_LOGS_PATH: Path = LOGS_DIR / "rag_logs.jsonl"
EVAL_RESULTS_PATH: Path = LOGS_DIR / "eval_results.jsonl"
ACCURACY_REPORT_PATH: Path = LOGS_DIR / "accuracy_report.csv"

# ── Retrieval settings ─────────────────────────────────────────────────────────
EVIDENCE_CONTEXT_LINES: int = int(os.environ.get("EVIDENCE_CONTEXT_LINES", "40"))
MAX_HITS_PER_FILE: int = int(os.environ.get("MAX_HITS_PER_FILE", "5"))
MAX_EVIDENCE_WINDOWS: int = int(os.environ.get("MAX_EVIDENCE_WINDOWS", "12"))

# ── LLM generation settings ────────────────────────────────────────────────────
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "4096"))


# ── Validation ─────────────────────────────────────────────────────────────────
def validate() -> None:
    """Raise an error early if required config is missing."""
    if not DEEPSEEK_API_KEY:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )


# Ensure log directory exists at import time
LOGS_DIR.mkdir(parents=True, exist_ok=True)
