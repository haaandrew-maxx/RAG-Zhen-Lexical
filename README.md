# Lexical Retrieval + Evidence-Gated Generative QA Agent

A Python agent that answers questions grounded exclusively in local documents using **pure lexical search** (no embeddings, no vector DB) and **DeepSeek** for reasoning. Every claim in the answer is backed by a verbatim quote from the retrieved contexts.

---

## Project Structure

```
fact_check_agent/
  README.md
  requirements.txt
  .gitignore
  .env.example
  config.py
  main.py

  documents/              # place your .txt / .md / .pdf / .csv docs here
  logs/
    rag_logs.jsonl
    eval_results.jsonl
    accuracy_report.csv

  llm/
    client.py             # DeepSeek OpenAI-compatible client
    prompts.py            # system prompts for analyzer, generator, repair

  agent/
    pipeline.py           # orchestrates the full RAG flow
    question_analyzer.py  # LLM-assisted query decomposition
    answer_generator.py   # evidence-gated generation + validation

  retrieval/
    search_engine.py      # lexical search: find + rg/grep
    evidence_reader.py    # context window builder

  logging/
    rag_logger.py         # RAGAS-style JSONL logging

  evaluation/
    batch_run.py          # run pipeline against Groundtruth.csv
    evaluate_accuracy.py  # score predictions with rapidfuzz

  utils/
    shell_tools.py        # subprocess wrappers for find/rg/grep
    text_utils.py         # text helpers
    json_utils.py         # robust JSON parsing
```

---

## Setup

### 1. Install dependencies

```bash
cd fact_check_agent
pip install -r requirements.txt
```

### 2. Create the `.env` file

```bash
cp .env.example .env
```

Edit `.env` and fill in your DeepSeek API key:

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

> Use `deepseek-reasoner` for chain-of-thought reasoning on complex questions.

### 3. Add documents

Place any `.txt`, `.md`, `.csv`, or other text-based files in the `documents/` directory. The agent searches these files using `rg` (ripgrep) or `grep` — no pre-processing required.

---

## Running a Single Query

```bash
python main.py "What is the maximum loan amount for first-time homebuyers?"
```

Pretty-print output:

```bash
python main.py "What are the reimbursement rules?" --pretty
```

Skip logging:

```bash
python main.py "What is the capital of France?" --no-log
```

### Output format

```json
{
  "status": "OK",
  "question": "What is the maximum loan amount?",
  "answer": "The maximum loan amount is $500,000 for first-time homebuyers.",
  "claims": [
    {
      "text": "The maximum loan amount is $500,000 for first-time homebuyers.",
      "evidence": [
        {
          "context_id": 0,
          "quote": "maximum loan amount is $500,000 for first-time homebuyers"
        }
      ]
    }
  ],
  "contexts": ["...full context window text..."],
  "sources": [
    {"file_path": "documents/loan_policy.txt", "location": {"start_line": 14, "end_line": 44}}
  ],
  "timestamp": "2025-01-01T12:00:00+00:00"
}
```

**Status values:**
- `OK` — at least one claim with verified evidence
- `PARTIALLY_SUPPORTED` — some claims dropped, at least one valid claim remains
- `NOT_FOUND_IN_DOCS` — no verifiable evidence found

---

## Batch Evaluation

### Prepare a ground truth CSV

The evaluator accepts comma- or semicolon-delimited CSV files. It can read the
EPD dataset format directly, including files such as `EPD-questions-ground_truths.csv`.

Minimum supported columns:

```csv
id,question,ground_truth
1,"What is the maximum loan amount?","$500,000"
2,"What are the leave entitlements?","20 days per year"
```

Also accepted:

- Question aliases: `question`, `questions`, `query`, `pregunta`, `q`
- Ground-truth aliases: `ground_truth`, `ground_truths`, `ground truth`, `groundtruth`, `groundtruths`, `answer`, `expected`, `reference`, `gold`, `gt`

If `id` is missing, the runner assigns row numbers automatically.

### Run batch evaluation

```bash
python evaluation/batch_run.py
```

If a single matching CSV exists in the project root, it will be auto-detected.
You can also pass the file explicitly:

```bash
python evaluation/batch_run.py --input EPD-questions-ground_truths.csv
```

Options:

```bash
# Limit to first 10 questions
python evaluation/batch_run.py --input EPD-questions-ground_truths.csv --limit 10

# Shuffle before limiting
python evaluation/batch_run.py --input EPD-questions-ground_truths.csv --limit 10 --shuffle

# Generate predictions and score them in one command
python evaluation/batch_run.py --input EPD-questions-ground_truths.csv --score-after-run
```

Each batch run overwrites `logs/eval_results.jsonl` so old and new runs are not mixed.

---

## Computing Accuracy

```bash
python evaluation/evaluate_accuracy.py
```

Or specify a custom input file:

```bash
python evaluation/evaluate_accuracy.py --input logs/eval_results.jsonl
```

### Output

```
Total:         50
Average score: 0.7312
Correct:       38 / 50

Report saved to logs/accuracy_report.csv
```

The accuracy report CSV (`logs/accuracy_report.csv`) contains columns:

```
id, score, binary, question, ground_truth, pred_answer, status
```

**Scoring rule:** RAGAS `AnswerAccuracy` scores each
`(question, pred_answer, ground_truth)` triple with the configured DeepSeek model.
**Correct** if `score >= 0.5`.

---

## Configuration Reference

All settings are in `config.py`, loaded from `.env`:

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | **Required.** Your DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API base URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name |
| `EVIDENCE_CONTEXT_LINES` | `30` | Lines before/after each hit |
| `MAX_HITS_PER_FILE` | `5` | Max hits per file before ranking |
| `MAX_EVIDENCE_WINDOWS` | `10` | Max context windows passed to LLM |
| `LLM_TEMPERATURE` | `0` | LLM sampling temperature |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per LLM response |

---

## Architecture

```
Question
   │
   ▼
QuestionAnalyzer (LLM)
   │  question_type, search_terms, regex_patterns
   ▼
SearchEngine (find + rg/grep)
   │  RankedHit list
   ▼
EvidenceReader
   │  EvidenceWindow list (context strings with line numbers)
   ▼
AnswerGenerator (LLM)
   │  raw JSON: {answer, claims[{text, evidence[{context_id, quote}]}]}
   ▼
Quote Validator (programmatic)
   │  verifies every quote is a verbatim substring of its context
   │  auto-repair pass if validation fails
   ▼
Pipeline Output: {status, answer, claims, contexts, sources, timestamp}
   │
   ├──► stdout (JSON)
   └──► logs/rag_logs.jsonl
```

---

## Retrieval Details

- **Filename search:** `find documents/ -type f -iname "*<term>*"`
- **Content search:** `rg --line-number --smart-case --no-heading -e "<pattern>"` (falls back to `grep -rE -n -i` if `rg` is not installed)
- **Ranking:** hits are scored by keyword coverage + file hit density; deduplicated with per-file limits
- **Context windows:** ±`EVIDENCE_CONTEXT_LINES` lines around each hit; overlapping windows are merged

---

## Notes

- The agent **never** uses semantic search, embeddings, or vector databases.
- All evidence quotes are validated programmatically — if a quote cannot be found verbatim in the source context, the claim is removed or a repair is attempted.
- Logs are append-only JSONL files, safe for concurrent use.
