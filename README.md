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

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd daily-paper-collector

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

Only the keys for your chosen LLM provider are required. Email credentials are only needed if email delivery is enabled.

### 4. Edit configuration

All runtime settings live in `config/config.yaml`:

- **arXiv categories** — which categories to fetch
- **Matching thresholds** — embedding top-N, LLM top-K, similarity cutoff
- **LLM provider** — `openai`, `claude`, or `claude_code`
- **Email** — SMTP host, recipients, subject prefix
- **Scheduler** — cron expression (default: daily at 8:00 AM)

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
  fetcher/             # ArXiv API fetching
  store/               # SQLite CRUD (papers, interests, matches, summaries, reports)
  matcher/             # embedder.py, ranker.py
  interest/            # Interest manager (keywords, past papers, references)
  report/              # Markdown report generation
  email/               # SMTP sender (Markdown → HTML → CSS-inline)
  summarizer/          # ar5iv HTML parsing + LLM summarization
  llm/                 # LLM provider abstraction (OpenAI, Claude, Claude Code)
  scheduler/           # APScheduler cron wrapper
gui/
  app.py               # Streamlit main entry
  pages/               # dashboard, papers, interests, reports, settings
  components/          # paper_card, report_viewer
config/config.yaml     # Runtime configuration
data/papers.db         # SQLite database (auto-created)
```

## License

MIT
