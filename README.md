# Daily Paper Collector

Automated arXiv paper collection and analysis tool. Fetches daily papers from configurable arXiv categories, performs two-stage intelligent matching (embedding + LLM re-ranking) against user interests, generates Markdown reports, and delivers them via email.

## Features

- **Configurable arXiv categories** — cs.AI, cs.CL, cs.LG, cs.CV, etc.
- **Two-stage matching** — embedding-based coarse filtering → LLM re-ranking for precision
- **Multiple LLM providers** — OpenAI, Claude API, or Claude Code CLI (zero-cost with subscription)
- **Email delivery** — daily Markdown reports converted to styled HTML emails
- **Streamlit GUI** — browse papers, manage interests, read reports, on-demand summarization
- **Scheduler** — APScheduler with cron-based daily runs

## Requirements

- Python >= 3.11
- ~80 MB disk space for the sentence-transformers model (downloaded on first run)

## Setup

### 1. Create conda environment

```bash
git clone https://github.com/Zehong-Wang/daily-paper-collector.git
cd daily-paper-collector

conda create -n daily-paper-collector python=3.11 -y
conda activate daily-paper-collector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .            # installs the project in editable mode
```

### 3. Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` — only set the variables you need:

| Variable              | Required when              |
| --------------------- | -------------------------- |
| `OPENAI_API_KEY`    | LLM provider is `openai` |
| `ANTHROPIC_API_KEY` | LLM provider is `claude` |
| `EMAIL_USERNAME`    | Email delivery is enabled  |
| `EMAIL_PASSWORD`    | Email delivery is enabled  |

The default LLM provider is **`claude_code`**, which calls the `claude` CLI and requires no API key (uses your existing Claude Code subscription). If you use this default, no API keys are needed.

### 4. Edit configuration

All runtime settings live in `config/config.yaml`:

| Section       | Key settings                                                                                        |
| ------------- | --------------------------------------------------------------------------------------------------- |
| `arxiv`     | Categories to fetch (`cs.AI`, `cs.CL`, `cs.LG`, `cs.CV`), max results per category          |
| `matching`  | Embedding model, top-N coarse candidates (50), top-K final results (10), similarity threshold (0.3) |
| `llm`       | Provider (`openai` / `claude` / `claude_code`), model, timeout, retries, concurrency          |
| `email`     | SMTP host/port, sender, recipients, subject prefix, enabled toggle                                  |
| `scheduler` | Cron expression (default:`0 8 * * *` — daily at 8:00 AM)                                         |
| `database`  | SQLite path (default:`data/papers.db`, auto-created)                                              |

### 5. Add your research interests

Before the first run, add at least one interest so the matcher can find relevant papers. You can do this through the Streamlit GUI (Interests page) or by inserting directly into the SQLite database. Three interest types are supported:

- **keyword** — a research topic (e.g., "transformer architecture", "reinforcement learning")
- **paper** — an arXiv ID of one of your past papers (abstract auto-fetched for embedding)
- **reference_paper** — an arXiv ID of a paper you find relevant

Without interests, the pipeline still runs but only produces a general report (no personalized recommendations).

## Usage

### Run pipeline once

```bash
python -m src.main --mode run
```

### Run with scheduler (daily cron)

```bash
python -m src.main --mode scheduler
```

### Launch Streamlit GUI

```bash
streamlit run gui/app.py
```

### CI/CD (GitHub Actions)

```bash
python scripts/run_pipeline.py
```

## Development

### Run tests

```bash
pytest
pytest --cov
pytest -v
```

### Lint and format

```bash
ruff check .
ruff format .
```

## Project Structure

```
src/
  main.py              # CLI entry point (--mode scheduler|run)
  pipeline.py          # DailyPipeline orchestrator
  config.py            # Config loading, env helpers, logging setup
  fetcher/             # ArXiv API fetching
  store/               # SQLite CRUD (papers, interests, matches, summaries, reports)
  matcher/             # embedder.py (sentence-transformers), ranker.py (LLM re-rank)
  interest/            # Interest manager (keywords, past papers, references)
  report/              # Markdown report generation (general + specific)
  email/               # SMTP sender (Markdown → HTML → CSS-inline)
  summarizer/          # ar5iv HTML parsing + LLM summarization
  llm/                 # LLM provider abstraction (OpenAI, Claude API, Claude Code CLI)
  scheduler/           # APScheduler cron wrapper
gui/
  app.py               # Streamlit main entry
  pages/               # dashboard, papers, interests, reports, settings
  components/          # paper_card, report_viewer
scripts/
  run_pipeline.py      # CI/CD entry point (GitHub Actions)
templates/
  email_template.md    # Email format reference
config/config.yaml     # Runtime configuration
data/papers.db         # SQLite database (auto-created)
tests/                 # Unit, integration, and GUI tests
```

## License

MIT
