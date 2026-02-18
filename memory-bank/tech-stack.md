# Daily Paper Collector - Tech Stack

## Reference Projects

The tech stack is informed by analysis of 14 existing arXiv paper collection/recommendation projects on GitHub:

| Project | Stars | Key Tech | Matching | LLM |
|---------|-------|----------|----------|-----|
| [arxiv-sanity-preserver](https://github.com/karpathy/arxiv-sanity-preserver) | 5.6k | Flask, SQLite+MongoDB | TF-IDF + SVM | None |
| [zotero-arxiv-daily](https://github.com/TideDra/zotero-arxiv-daily) | 4.6k | sentence-transformers | Embedding similarity | Qwen2.5-3B (local) |
| [daily-arXiv-ai-enhanced](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced) | 2.4k | GitHub Pages | Category filter | DeepSeek |
| [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) | 1.5k | Flask, SQLite | TF-IDF + SVM | None |
| [gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant) | 542 | arXiv RSS, JSON files | Author + GPT-4 scoring | GPT-4 |
| [ArxivDigest](https://github.com/AutoLLM/ArxivDigest) | 404 | Gradio | GPT-3.5 scoring | GPT-3.5 |
| [ArxivDigest-extra](https://github.com/linhkid/ArxivDigest-extra) | 46 | Gradio, multi-model | Two-stage LLM filter | GPT/Gemini/Claude/Ollama |
| [Paper-Recommendation-System](https://github.com/mcpeixoto/Paper-Recommendation-System) | 23 | Streamlit, FAISS | FAISS + sentence-transformers | None |

**Key patterns observed:**
- Python is universal (100% of projects)
- `arxiv` Python library is the standard fetching method
- Modern projects use embedding + LLM hybrid matching (our approach)
- SQLite is proven for single-user persistence (arxiv-sanity-lite served 25k+ papers)
- GitHub Actions is the dominant scheduler for zero-infra setups

---

## Final Tech Stack

### Python >= 3.11

Selected for performance improvements (faster startup, exception groups, `tomllib` in stdlib) and broad library compatibility.

---

### 1. ArXiv Fetching: `arxiv` (v2.1+)

```
pip install arxiv
```

| Option | Pros | Cons |
|--------|------|------|
| **`arxiv` library** | Mature, well-documented, used by majority of reference projects, clean API | Occasional rate-limit issues |
| arXiv RSS + `feedparser` | Lighter, simpler | Less metadata, harder to paginate |
| Direct OAI-PMH API | Most complete metadata | Complex XML parsing, verbose |
| Web scraping | Access to listing pages | Fragile, violates ToS |

**Decision:** `arxiv` library. It is the de facto standard used by arxiv-sanity-lite, zotero-arxiv-daily, ArxivDigest, and others. Provides clean access to all needed fields (title, authors, abstract, categories, dates, URLs). Built-in pagination and rate limiting.

**Rate limiting strategy:** arXiv API allows 1 request per 3 seconds. The `arxiv` library handles this internally. For fetching ~200 papers per category across 4 categories, expect ~5 minutes total fetch time.

---

### 2. Embedding: `sentence-transformers` + `all-MiniLM-L6-v2`

```
pip install sentence-transformers
```

| Model | Dims | Size | Speed | MTEB Avg | Notes |
|-------|------|------|-------|----------|-------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast | 56.3 | Best speed/quality tradeoff |
| `all-mpnet-base-v2` | 768 | 420MB | Medium | 57.8 | Marginally better quality |
| `nomic-embed-text-v1.5` | 768 | 548MB | Medium | 62.3 | Best quality, Matryoshka support |
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | Fast | 62.2 | Strong alternative |

**Decision:** Default to `all-MiniLM-L6-v2` for its speed and small footprint (80MB, 384 dims). This is the same model family used by zotero-arxiv-daily (4.6k stars). Make the model configurable in YAML so users can switch to `nomic-embed-text-v1.5` or `bge-small-en-v1.5` for better quality if they have more compute.

**Why not API-based embeddings:** Local models avoid per-request costs, work offline, and have zero latency overhead. For a daily batch of ~200-800 papers, `all-MiniLM-L6-v2` completes in under 30 seconds on CPU.

---

### 3. Vector Search: `numpy` (with FAISS optional)

```
pip install numpy
```

| Option | Pros | Cons |
|--------|------|------|
| **`numpy` cosine similarity** | Zero additional deps, simple, sufficient for <100k vectors | No indexing, linear scan |
| `faiss-cpu` | Optimized ANN search, scales to millions | Extra dependency, overkill for <10k papers |
| `sqlite-vec` | Integrated with SQLite | Young project (v0.1.x), limited ecosystem |
| `chromadb` | Full vector DB | Heavy dependency for a simple use case |

**Decision:** `numpy` for cosine similarity as the primary method. At our scale (hundreds to low thousands of papers per day, ~10k total after months), a brute-force cosine similarity over 384-dim vectors is <10ms. This matches the KISS principle followed by most reference projects. Add `faiss-cpu` as an optional dependency behind a config flag for users who accumulate 100k+ papers.

```python
# Core similarity computation - this is all we need
import numpy as np

def cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    query_norm = query / np.linalg.norm(query)
    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    return corpus_norm @ query_norm
```

---

### 4. LLM Providers: `openai` + `anthropic` + `subprocess` (Claude CLI)

```
pip install openai anthropic
```

| Provider | SDK | Use Case | Cost |
|----------|-----|----------|------|
| OpenAI (`gpt-4o-mini`) | `openai` >= 1.0 | Default: cheap, fast, good at scoring/summarization | ~$0.15/1M input tokens |
| Claude (`claude-sonnet-4-5-20250929`) | `anthropic` >= 0.30 | High-quality summarization and analysis | ~$3/1M input tokens |
| Claude Code CLI | `subprocess` | Zero marginal cost via existing subscription | Included in subscription |

**Decision:** Support all three via a unified `LLMProvider` abstract base class. Each provider implemented as a separate module. The `ClaudeCodeProvider` calls the `claude` CLI via subprocess with `--print` flag for non-interactive output, leveraging the user's existing Claude Code subscription at zero additional cost.

**Recommended defaults:**
- Re-ranking: `gpt-4o-mini` (fast, cheap, good at structured scoring)
- Report generation: `claude-sonnet-4-5-20250929` or `claude` CLI (better at long-form writing)
- Paper summarization: `claude` CLI (leverages subscription, no API cost)

---

### 5. Database: `SQLite` via Python `sqlite3` stdlib

No additional installation needed - `sqlite3` is in Python's standard library.

| Option | Pros | Cons |
|--------|------|------|
| **SQLite** | Zero setup, single file, ACID, proven (arxiv-sanity-lite) | No concurrent writers |
| PostgreSQL | Concurrent access, pgvector extension | Requires running a server |
| JSON files | Simplest possible | No querying, no ACID, fragile |
| TinyDB | Simple Python-native | No SQL, limited querying |

**Decision:** SQLite. It is the proven choice in this domain (used by both arxiv-sanity projects). A single-user local tool has no concurrency concerns. Embedding vectors stored as BLOBs (serialized numpy arrays). The entire database is a single portable file.

**Schema highlights:**
- `papers` table with embedding BLOB column
- `interests` table with embedding BLOB column
- `matches` table linking papers to daily runs with scores
- `summaries` table caching LLM-generated summaries
- `daily_reports` table storing Markdown reports

---

### 6. GUI: `Streamlit` (>= 1.40)

```
pip install streamlit
```

| Option | Pros | Cons |
|--------|------|------|
| **Streamlit** | Rich components, native Python, multi-page apps, active ecosystem | Stateless reruns, limited interactivity |
| Gradio | Quick ML demos, used by ArxivDigest | Fixed layouts, less suitable for dashboards |
| NiceGUI | More interactive, FastAPI-based | Smaller community |
| Panel/HoloViz | Data-focused, flexible | Steeper learning curve |
| Flask + Jinja2 | Full control | Much more code to write |

**Decision:** Streamlit. It provides the best balance of development speed and capability for a dashboard-style application. Native multi-page app support (via `st.navigation`) maps perfectly to our 5-page design (Dashboard, Papers, Interests, Reports, Settings). Used by Paper-Recommendation-System in the reference projects. The stateless rerun model is acceptable for our read-heavy use case.

**Key Streamlit features we'll use:**
- `st.navigation` / multi-page apps for page routing
- `st.dataframe` / `st.data_editor` for paper tables
- `st.expander` for paper detail views
- `st.form` for interest management
- `st.markdown` for report rendering
- `st.spinner` for LLM call feedback
- `st.cache_data` / `st.cache_resource` for DB connection and model caching

---

### 7. Scheduling: `APScheduler` (>= 3.10)

```
pip install apscheduler
```

| Option | Pros | Cons |
|--------|------|------|
| **APScheduler** | In-process, cron syntax, lightweight, well-maintained | Process must stay alive |
| `schedule` | Simpler API | No cron syntax, less flexible |
| Celery | Production-grade, distributed | Massive overkill, needs Redis/RabbitMQ |
| System cron / launchd | No process needed | OS-specific, harder to configure |

**Decision:** APScheduler with `CronTrigger` for the local scheduler. It runs in-process alongside the application, supports standard cron expressions, and requires no external services. The `BlockingScheduler` runs standalone; for GUI integration, `BackgroundScheduler` runs in a separate thread.

**GitHub Actions interface:** A standalone `scripts/run_pipeline.py` entry point is provided for CI/CD integration. This script imports and runs the same `DailyPipeline` directly, making GitHub Actions a drop-in scheduler replacement with zero code changes.

---

### 8. Email: `smtplib` (stdlib) + `markdown` + `premailer`

```
pip install markdown premailer
```

| Component | Library | Purpose |
|-----------|---------|---------|
| SMTP sending | `smtplib` + `email.mime` (stdlib) | Send emails via Gmail/SMTP |
| Markdown rendering | `markdown` (>= 3.5) | Convert Markdown reports to HTML |
| CSS inlining | `premailer` (>= 3.10) | Inline CSS for email client compatibility |

**Decision:** Use Python's built-in `smtplib` for sending (no third-party email service dependency). The `markdown` library converts reports to HTML, and `premailer` inlines all CSS styles since most email clients strip `<style>` tags. This is a lightweight, proven stack.

**Email rendering pipeline:**
```
Markdown report → markdown.markdown() → HTML → premailer.transform() → Inlined HTML → smtplib → SMTP
```

---

### 9. HTML Parsing: `requests` + `beautifulsoup4`

```
pip install requests beautifulsoup4 lxml
```

| Purpose | Library |
|---------|---------|
| HTTP client | `requests` (>= 2.31) |
| HTML parsing | `beautifulsoup4` (>= 4.12) with `lxml` parser |

**Decision:** For fetching and parsing ar5iv HTML versions of papers, `requests` + `BeautifulSoup` with the `lxml` parser is the standard Python approach. `lxml` is significantly faster than the default `html.parser` and handles malformed HTML gracefully.

---

### 10. Configuration & Environment: `PyYAML` + `python-dotenv`

```
pip install pyyaml python-dotenv
```

| Component | Library | Purpose |
|-----------|---------|---------|
| Config file | `PyYAML` (>= 6.0) | Parse `config/config.yaml` |
| Secrets | `python-dotenv` (>= 1.0) | Load `.env` file for API keys, SMTP credentials |

**Decision:** YAML for human-readable configuration (arXiv categories, matching thresholds, LLM settings). `.env` file for secrets (API keys, email passwords). This separation follows the 12-factor app principle of keeping secrets out of config files. `python-dotenv` auto-loads `.env` into `os.environ`.

---

### 11. Testing: `pytest` + `pytest-asyncio`

```
pip install pytest pytest-asyncio pytest-cov
```

**Decision:** `pytest` is the de facto Python testing standard. `pytest-asyncio` for testing async LLM provider calls. `pytest-cov` for coverage reporting.

---

### 12. Project Management: `pyproject.toml` + `pip`

**Decision:** Use `pyproject.toml` as the single project metadata file (PEP 621). Manage dependencies with `pip` and `requirements.txt` for simplicity. No need for Poetry or PDM for a single-user tool.

---

## Dependency Summary

### Core (required)

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| `arxiv` | >= 2.1 | arXiv API client | Tiny |
| `sentence-transformers` | >= 3.0 | Local embedding models | ~50MB (+ model download) |
| `numpy` | >= 1.26 | Vector math | ~30MB |
| `openai` | >= 1.0 | OpenAI API SDK | Tiny |
| `anthropic` | >= 0.30 | Claude API SDK | Tiny |
| `streamlit` | >= 1.40 | GUI framework | ~80MB |
| `apscheduler` | >= 3.10 | Task scheduling | Tiny |
| `markdown` | >= 3.5 | Markdown to HTML | Tiny |
| `premailer` | >= 3.10 | CSS inlining for email | Tiny |
| `beautifulsoup4` | >= 4.12 | HTML parsing | Tiny |
| `lxml` | >= 5.0 | Fast HTML/XML parser | ~10MB |
| `requests` | >= 2.31 | HTTP client | Tiny |
| `pyyaml` | >= 6.0 | YAML config parsing | Tiny |
| `python-dotenv` | >= 1.0 | .env file loading | Tiny |

### Optional

| Package | Version | Purpose | When to install |
|---------|---------|---------|-----------------|
| `faiss-cpu` | >= 1.8 | ANN vector search | If paper DB exceeds 100k entries |
| `torch` | >= 2.2 | GPU acceleration for embeddings | If GPU available |

### Dev

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 8.0 | Testing framework |
| `pytest-asyncio` | >= 0.23 | Async test support |
| `pytest-cov` | >= 5.0 | Coverage reporting |
| `ruff` | >= 0.4 | Linting + formatting |

---

## `requirements.txt`

```
# Core
arxiv>=2.1
sentence-transformers>=3.0
numpy>=1.26
openai>=1.0
anthropic>=0.30
streamlit>=1.40
apscheduler>=3.10
markdown>=3.5
premailer>=3.10
beautifulsoup4>=4.12
lxml>=5.0
requests>=2.31
pyyaml>=6.0
python-dotenv>=1.0

# Optional (uncomment if needed)
# faiss-cpu>=1.8
```

---

## Runtime Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python | 3.11 | 3.12+ |
| RAM | 2 GB | 4 GB |
| Disk | 500 MB (model + DB) | 2 GB |
| GPU | Not required | Optional (speeds up embedding) |
| Network | Required (arXiv API, LLM APIs) | - |
| OS | macOS / Linux | macOS (primary dev target) |
