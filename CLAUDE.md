# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Daily Paper Collector v2 — an automated arXiv paper collection and analysis tool. Fetches daily papers from configurable arXiv categories, performs two-stage intelligent matching (embedding + LLM re-ranking) against user interests, generates Markdown reports, and delivers them via email. Includes a Streamlit GUI for browsing, interest management, and on-demand paper summarization.

**Status:** Design phase. See `daily-paper-collector-document.md` (architecture, schema, interfaces) and `tech-stack.md` (dependency rationale) for the full specification.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run pipeline (scheduler mode)
python -m src.main --mode scheduler

# Run pipeline (single manual run)
python -m src.main --mode run

# Launch Streamlit GUI
streamlit run gui/app.py

# CI/CD entry point (GitHub Actions)
python scripts/run_pipeline.py

# Tests
pytest
pytest --cov
pytest -v

# Lint & format
ruff check .
ruff format .
```

## Architecture

**Pipeline data flow:** Scheduler → ArXiv Fetcher → SQLite Store → Embedder → Embedding Matcher (top-N) → LLM Re-ranker (top-K) → Report Generator → Email Sender

**Two-stage matching:**

1. Coarse: `sentence-transformers` (`all-MiniLM-L6-v2`) computes cosine similarity between paper abstracts and user interest embeddings → top-N candidates (~50)
2. Fine: LLM scores each candidate for relevance (1-10 scale) with explanations → top-K results (~10)

**Interest sources (unified via embeddings):** keywords, user's past papers (arXiv IDs), manually added reference papers

**LLM provider abstraction:** `LLMProvider` ABC in `src/llm/base.py` with three implementations:

- `OpenAIProvider` — `openai` SDK
- `ClaudeProvider` — `anthropic` SDK
- `ClaudeCodeProvider` — `claude` CLI via subprocess (zero-cost via subscription)

**Streamlit GUI:** 5-page app (`gui/app.py`) — Dashboard, Papers (browse + summarize), Interests (CRUD + auto re-embed), Reports (historical), Settings

**Paper summarizer** (`src/summarizer/paper_summarizer.py`): Fetches full text via ar5iv HTML, parses with BeautifulSoup, LLM generates brief or detailed summaries. GUI-only (not in daily pipeline). Summaries cached in DB.

## Key Source Layout

```
src/
  main.py              # CLI entry point (--mode scheduler|run)
  pipeline.py          # DailyPipeline orchestrator
  fetcher/             # ArXiv API fetching
  store/               # SQLite CRUD (papers, interests, matches, summaries, reports)
  matcher/             # embedder.py (sentence-transformers), ranker.py (LLM re-rank)
  interest/            # Interest manager (keywords, past papers, references)
  report/              # Markdown report generation (general + specific)
  email/               # SMTP sender with Markdown→HTML→CSS-inline pipeline
  summarizer/          # ar5iv HTML parsing + LLM summarization
  llm/                 # base.py ABC, openai_provider.py, claude_provider.py, claude_code_provider.py
  scheduler/           # APScheduler CronTrigger wrapper
gui/
  app.py               # Streamlit main entry
  pages/               # dashboard, papers, interests, reports, settings
  components/          # paper_card, report_viewer
config/config.yaml     # All runtime configuration (categories, thresholds, LLM, email, scheduler)
```

## Database

SQLite at `data/papers.db`. Five tables: `papers` (with embedding BLOB), `interests` (with embedding BLOB), `matches` (embedding_score + llm_score + llm_reason), `summaries` (brief/detailed, cached), `daily_reports` (Markdown content). Full schema in design doc Section 5.

## Tech Constraints

- Python >= 3.11
- Embeddings are local (no API cost); vectors stored as serialized numpy arrays in SQLite BLOBs
- `numpy` cosine similarity for vector search (sufficient for <100k papers; `faiss-cpu` optional beyond that)
- Email pipeline: Markdown → `markdown` lib → HTML → `premailer` CSS inlining → `smtplib`
- Config in YAML (`config/config.yaml`); secrets in `.env` (API keys, SMTP credentials)
- Async interfaces for LLM calls (`async def complete`, `async def complete_json`)

## Environment Variables

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
EMAIL_USERNAME=...
EMAIL_PASSWORD=...
```

# Important notes：

* You must carefully and comprehensively read memory-bank/@architecture.md before writing any code.
* You must carefully and comprehensively read memory-bank/@daily-paper-collector-document.md before writing any code.
* You must update memory-bank/@architecture.md after you finish any milestone functions.
