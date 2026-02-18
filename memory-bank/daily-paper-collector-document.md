# Daily Paper Collector - Design Document

## 1. Project Overview

Daily Paper Collector is an automated arXiv paper collection and analysis tool. It fetches new papers daily from user-specified arXiv categories, performs interest-based intelligent matching and ranking, generates structured reports, and delivers them via email. A local Streamlit GUI is provided for browsing papers, managing interest profiles, and generating in-depth paper summaries.

### Core Goals

- Automatically fetch new arXiv papers daily
- Two-stage intelligent matching based on user interests (keywords / past papers / specified papers)
- Generate General Report (global trends) + Specific Report (personalized recommendations) + related paper listings
- Email delivery with Markdown rendered as HTML
- Local GUI for paper browsing, interest management, and paper summarization (brief / detailed modes)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     APScheduler                         │
│                  (Daily Trigger)                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                   Pipeline Runner                        │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ ArXiv       │  │ Embedding    │  │ LLM Re-rank    │  │
│  │ Fetcher     │──▶ Matcher      │──▶ & Scorer       │  │
│  └─────────────┘  └──────────────┘  └───────┬────────┘  │
│                                             │            │
│  ┌─────────────┐  ┌──────────────┐          │            │
│  │ Report      │◀─│ Interest     │◀─────────┘            │
│  │ Generator   │  │ Manager      │                       │
│  └──────┬──────┘  └──────────────┘                       │
│         │                                                │
│  ┌──────▼──────┐                                         │
│  │ Email       │                                         │
│  │ Sender      │                                         │
│  └─────────────┘                                         │
└──────────────────────────────────────────────────────────┘

┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│ SQLite DB   │    │ LLM Provider │    │ Streamlit GUI  │
│ (papers,    │    │ (OpenAI /    │    │ (browse,       │
│  embeddings,│    │  Claude API /│    │  config,       │
│  summaries) │    │  Claude CLI) │    │  summarize)    │
└─────────────┘    └──────────────┘    └────────────────┘
```

---

## 3. Core Components

### 3.1 ArXiv Fetcher (`src/fetcher/arxiv_fetcher.py`)

- Fetches daily new papers from user-configured arXiv categories using the `arxiv` Python library
- Supported categories configurable via YAML (e.g., `cs.AI`, `cs.CL`, `cs.LG`)
- Extracted fields: `arxiv_id`, `title`, `authors`, `abstract`, `categories`, `published_date`, `pdf_url`, `ar5iv_url`
- Deduplication: skips papers already in the database based on `arxiv_id`

### 3.2 Paper Store (`src/store/database.py`)

- SQLite database storing paper metadata, embedding vectors, user interests, and generated summaries
- Provides CRUD operations and query interfaces
- Embedding vectors stored as BLOBs (serialized numpy arrays)

### 3.3 Embedding Matcher (`src/matcher/embedder.py`)

- Uses a local `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`)
- Computes embedding vectors for paper abstracts and user interests
- First-stage coarse filtering via cosine similarity, returning top-N candidates

### 3.4 LLM Re-ranker (`src/matcher/ranker.py`)

- Receives coarse-filtered candidate paper list
- Calls LLM to score each paper's relevance to user interests (1-10 scale)
- Returns top-K final results with LLM-generated relevance explanations

### 3.5 Interest Manager (`src/interest/manager.py`)

- Manages three sources of user interests:
  - **Keywords**: defined directly in YAML config or via GUI
  - **Past papers**: user's own published papers (provided as arXiv IDs or titles)
  - **Reference papers**: manually added papers of interest
- Computes unified embeddings across all interest sources for matching
- Supports dynamic editing through GUI with automatic embedding recomputation

### 3.6 Report Generator (`src/report/generator.py`)

- **General Report**:
  - Daily paper count statistics (distribution by category)
  - Trending topics / keyword trends
  - Highlight papers (selected by LLM from all daily papers)
- **Specific Report**:
  - Top-K papers matched to user interests
  - LLM-generated personalized analysis: why each paper is relevant to the user
  - Brief recommendation note for each paper
- Output format: Markdown

### 3.7 Email Sender (`src/email/sender.py`)

- Sends emails via SMTP (e.g., Gmail)
- Renders Markdown reports to HTML (using the `markdown` library)
- Email content structure (three sections):
  1. General Report
  2. Specific Report
  3. Related Papers List (each with title, authors, abstract excerpt, arXiv link)

### 3.8 Paper Summarizer (`src/summarizer/paper_summarizer.py`)

- Fetches full paper text via ar5iv HTML version
- Parses HTML content using `requests` + `BeautifulSoup`
- Two summarization modes:
  - **Brief Summary**: 1-2 paragraphs covering core contributions and methodology
  - **Detailed Summary**: structured summary (motivation, method, experiments, conclusions, limitations)
- Triggered only from the GUI; not part of the daily automated pipeline

### 3.9 LLM Provider (`src/llm/`)

Unified LLM abstraction layer:

```python
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> str: ...

class OpenAIProvider(LLMProvider): ...      # openai Python SDK
class ClaudeProvider(LLMProvider): ...      # anthropic Python SDK
class ClaudeCodeProvider(LLMProvider): ...  # subprocess calling claude CLI
```

- `ClaudeCodeProvider` invokes the `claude` CLI via `subprocess`, leveraging the user's existing Claude Code subscription
- Provider selection configured in YAML

### 3.10 Scheduler (`src/scheduler/scheduler.py`)

- Uses APScheduler's `CronTrigger` for daily scheduled execution
- Configurable run time (default: 8:00 AM daily)
- GitHub Actions interface reserved: provides a standalone `run_pipeline.py` entry point callable from CI/CD

### 3.11 Streamlit GUI (`gui/app.py`)

- Multi-page application, launched via `streamlit run gui/app.py`
- Page details in Section 9

---

## 4. Data Flow

```
1. [Scheduler] triggers daily job
       │
2. [ArXiv Fetcher] fetches new papers from configured categories
       │
3. [Paper Store] saves to SQLite (with deduplication)
       │
4. [Embedder] computes embeddings for new paper abstracts
       │
5. [Embedding Matcher] calculates similarity against user interest embeddings → top-N candidates
       │
6. [LLM Re-ranker] re-ranks candidates with relevance scoring → top-K results
       │
7. [Report Generator]
       ├── General Report (based on all daily papers)
       └── Specific Report (based on top-K matched papers)
       │
8. [Email Sender] assembles three sections → Markdown → HTML → SMTP delivery
       │
9. [Paper Store] persists report records
```

---

## 5. Database Schema

```sql
CREATE TABLE papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,           -- JSON array
    abstract TEXT NOT NULL,
    categories TEXT NOT NULL,        -- JSON array
    published_date DATE NOT NULL,
    pdf_url TEXT,
    ar5iv_url TEXT,
    embedding BLOB,                  -- serialized numpy array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE interests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,              -- 'keyword' | 'paper' | 'reference_paper'
    value TEXT NOT NULL,             -- keyword text or arxiv_id
    description TEXT,                -- optional description
    embedding BLOB,                  -- serialized numpy array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER REFERENCES papers(id),
    run_date DATE NOT NULL,
    embedding_score REAL,            -- cosine similarity
    llm_score REAL,                  -- LLM relevance score (1-10)
    llm_reason TEXT,                 -- LLM relevance explanation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER REFERENCES papers(id),
    summary_type TEXT NOT NULL,      -- 'brief' | 'detailed'
    content TEXT NOT NULL,
    llm_provider TEXT,               -- LLM provider used
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE daily_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date DATE NOT NULL,
    general_report TEXT,             -- Markdown content
    specific_report TEXT,            -- Markdown content
    paper_count INTEGER,
    matched_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 6. Configuration (`config/config.yaml`)

```yaml
# arXiv settings
arxiv:
  categories:
    - cs.AI
    - cs.CL
    - cs.LG
    - cs.CV
  max_results_per_category: 200

# Matching settings
matching:
  embedding_model: "all-MiniLM-L6-v2"
  embedding_top_n: 50        # coarse filtering candidate count
  llm_top_k: 10              # final recommendation count
  similarity_threshold: 0.3  # minimum similarity threshold

# LLM settings
llm:
  provider: "openai"         # openai | claude | claude_code
  openai:
    model: "gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"
  claude:
    model: "claude-sonnet-4-5-20250929"
    api_key_env: "ANTHROPIC_API_KEY"
  claude_code:
    cli_path: "claude"       # path to claude CLI
    model: "sonnet"

# Email settings
email:
  enabled: true
  smtp:
    host: "smtp.gmail.com"
    port: 587
    username_env: "EMAIL_USERNAME"
    password_env: "EMAIL_PASSWORD"
  from: "your-email@gmail.com"
  to:
    - "recipient@example.com"
  subject_prefix: "[Daily Papers]"

# Scheduler settings
scheduler:
  enabled: true
  cron: "0 8 * * *"          # daily at 8:00 AM

# Database
database:
  path: "data/papers.db"

# GUI
gui:
  port: 8501
```

---

## 7. Directory Structure

```
daily-paper-collector-v2/
├── config/
│   └── config.yaml                # main configuration
├── src/
│   ├── __init__.py
│   ├── main.py                    # entry point: start scheduler or manual trigger
│   ├── pipeline.py                # daily pipeline orchestration
│   ├── fetcher/
│   │   ├── __init__.py
│   │   └── arxiv_fetcher.py
│   ├── store/
│   │   ├── __init__.py
│   │   └── database.py
│   ├── matcher/
│   │   ├── __init__.py
│   │   ├── embedder.py            # embedding computation
│   │   └── ranker.py              # LLM re-ranking
│   ├── interest/
│   │   ├── __init__.py
│   │   └── manager.py
│   ├── report/
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── email/
│   │   ├── __init__.py
│   │   └── sender.py
│   ├── summarizer/
│   │   ├── __init__.py
│   │   └── paper_summarizer.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                # LLMProvider ABC
│   │   ├── openai_provider.py
│   │   ├── claude_provider.py
│   │   └── claude_code_provider.py
│   └── scheduler/
│       ├── __init__.py
│       └── scheduler.py
├── gui/
│   ├── app.py                     # Streamlit main entry
│   ├── pages/
│   │   ├── dashboard.py           # home / daily overview
│   │   ├── papers.py              # paper browsing & search
│   │   ├── interests.py           # interest management
│   │   ├── reports.py             # historical report viewer
│   │   └── settings.py            # configuration management
│   └── components/
│       ├── paper_card.py          # paper card component
│       └── report_viewer.py       # report rendering component
├── templates/
│   └── email_template.md          # email Markdown template
├── data/                          # runtime data (gitignored)
│   └── papers.db
├── scripts/
│   └── run_pipeline.py            # standalone script for GitHub Actions
├── tests/
├── requirements.txt
├── pyproject.toml
├── .env.example                   # environment variable template
└── daily-paper-collector-document.md
```

---

## 8. Key Interfaces

### 8.1 Pipeline Runner

```python
# src/pipeline.py
class DailyPipeline:
    def __init__(self, config: dict):
        self.fetcher = ArxivFetcher(config)
        self.store = PaperStore(config)
        self.embedder = Embedder(config)
        self.ranker = LLMRanker(config)
        self.interest_mgr = InterestManager(config)
        self.report_gen = ReportGenerator(config)
        self.email_sender = EmailSender(config)

    async def run(self) -> None:
        # 1. Fetch
        papers = await self.fetcher.fetch_today()
        # 2. Store & embed
        new_papers = self.store.save_papers(papers)
        self.embedder.compute_embeddings(new_papers)
        # 3. Match
        interests = self.interest_mgr.get_all_interests()
        candidates = self.embedder.find_similar(interests, top_n=50)
        ranked = await self.ranker.rerank(candidates, interests, top_k=10)
        # 4. Report
        general = await self.report_gen.generate_general(new_papers)
        specific = await self.report_gen.generate_specific(ranked, interests)
        # 5. Email
        await self.email_sender.send(general, specific, ranked)
        # 6. Save report
        self.store.save_report(general, specific, len(new_papers), len(ranked))
```

### 8.2 LLM Provider Interface

```python
# src/llm/base.py
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> str: ...

    @abstractmethod
    async def complete_json(self, prompt: str, system: str = "") -> dict: ...
```

### 8.3 CLI Entry Points

```bash
# Start scheduler (runs as long-lived process)
python -m src.main --mode scheduler

# Manually trigger a single pipeline run
python -m src.main --mode run

# Launch GUI
streamlit run gui/app.py

# Standalone entry point for GitHub Actions
python scripts/run_pipeline.py
```

---

## 9. Streamlit GUI Pages

### 9.1 Dashboard (Home)
- Today's paper count overview
- Latest General Report preview
- Latest Specific Report preview
- Quick action: manually trigger pipeline

### 9.2 Papers (Browse & Search)
- Browse papers by date
- Search by title / author / keyword
- Each paper displays: title, authors, abstract, categories, arXiv link
- Paper summarization:
  - "Brief Summary" button: calls LLM to generate 1-2 paragraph overview
  - "Detailed Summary" button: calls LLM to generate structured deep summary
  - Previously generated summaries are cached in DB to avoid redundant LLM calls

### 9.3 Interests (Interest Management)
- View all current interest items (keywords / paper references)
- Add / edit / delete interest items
- Import arXiv paper IDs as interest anchors
- Automatic embedding recomputation on edit

### 9.4 Reports (Historical Reports)
- Browse historical General / Specific Reports by date
- View past matching results and scores

### 9.5 Settings (Configuration)
- View / edit YAML configuration
- Test email delivery
- Select LLM provider
- Manage arXiv category subscriptions

---

## 10. Email Format

Emails are rendered from Markdown to HTML with the following structure:

```markdown
# Daily Paper Report - 2025-01-15

## General Report
### Today's Overview
- **156** new papers collected
- cs.AI: 45 | cs.CL: 38 | cs.LG: 52 | cs.CV: 21

### Trending Topics
- Significant increase in multi-modal reasoning papers (12 papers)
- ...

### Highlight Papers
1. **Paper Title** - Authors - one-line recommendation

---

## Specific Report (Based on Your Interests)
Top papers matching your research interests today:

1. **Paper Title** (Relevance: 9.2/10)
   - Why it matters to you: ...
2. ...

---

## Related Papers

### 1. Paper Title
- **Authors**: Author1, Author2, ...
- **Categories**: cs.AI, cs.CL
- **Abstract**: First 200 characters of abstract...
- [arXiv](https://arxiv.org/abs/xxxx.xxxxx)

### 2. ...
```

---

## 11. Key Dependencies

```
arxiv                    # arXiv API client
sentence-transformers    # local embedding model
numpy                    # vector computation
openai                   # OpenAI API SDK
anthropic                # Claude API SDK
streamlit                # GUI framework
apscheduler              # task scheduling
markdown                 # Markdown to HTML conversion
beautifulsoup4           # ar5iv HTML parsing
requests                 # HTTP requests
pyyaml                   # YAML config parsing
python-dotenv            # environment variable management
```

---

## 12. Future Extensions

- GitHub Actions integration: run pipeline in the cloud via `scripts/run_pipeline.py`
- Multi-user support
- Paper bookmarking / read tracking
- Automatic interest learning from user browsing behavior
- RSS feed output
