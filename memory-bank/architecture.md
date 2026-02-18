# Architecture — Daily Paper Collector v2

## High-Level Pipeline

```
Scheduler → ArXiv Fetcher → SQLite Store → Embedder → Embedding Matcher (top-N)
  → LLM Re-ranker (top-K) → Report Generator → Email Sender
```

Streamlit GUI runs separately, reading from the same SQLite DB and invoking the Summarizer on demand.

---

## Implemented Files

### `src/config.py` — Configuration & Utilities
- `get_project_root()` — Walks up from `__file__` to find `pyproject.toml`. All path resolution is relative to this root.
- `load_config(path=None)` — Loads `config/config.yaml`, resolves `database.path` to an absolute path.
- `get_env(key)` — Reads `os.environ` with a clear `ValueError` on missing keys.
- `setup_logging(level)` — Configures root logger. Called once in `main.py` and `gui/app.py`.

### `config/config.yaml` — Runtime Configuration
Sections: `arxiv` (categories, max_results), `matching` (model, top_n, top_k, threshold), `llm` (provider + sub-configs for openai/claude/claude_code), `email` (SMTP settings), `scheduler` (cron), `database` (path), `gui` (port).

### `src/llm/base.py` — LLMProvider ABC
Abstract base class defining the LLM contract:
- `async def complete(prompt, system="") -> str` — Free-form text completion.
- `async def complete_json(prompt, system="") -> dict` — Structured JSON output. Implementations must parse and validate JSON.

Used by: `LLMRanker`, `ReportGenerator`, `PaperSummarizer`.

### `src/llm/__init__.py` — Factory Function
`create_llm_provider(config) -> LLMProvider` — Reads `config["llm"]["provider"]` (`"openai"` | `"claude"` | `"claude_code"`) and returns the corresponding implementation. Uses lazy imports to avoid loading unused SDKs.

### `src/llm/openai_provider.py` — OpenAIProvider
- Wraps `openai.AsyncOpenAI` client.
- Reads API key from env var specified in config (`config["llm"]["openai"]["api_key_env"]`).
- `complete()` builds a messages array with optional system message.
- `complete_json()` uses OpenAI's native `response_format={"type": "json_object"}` for reliable JSON output.

### `src/llm/claude_provider.py` — ClaudeProvider
- Wraps `anthropic.AsyncAnthropic` client.
- Reads API key from env var specified in config.
- `complete()` passes system message as a top-level `system` kwarg (Anthropic API style).
- `complete_json()` appends "Respond with valid JSON only." to system, then strips markdown code fences (```` ```json ... ``` ````) before parsing. Claude models sometimes wrap JSON in code blocks even when instructed not to.

### `src/llm/claude_code_provider.py` — ClaudeCodeProvider
- Calls `claude` CLI via `asyncio.create_subprocess_exec` with `--print --model <model>`.
- Zero API cost — uses existing Claude Code subscription.
- System message is prepended to the user prompt (CLI has no separate system parameter).
- `complete_json()` strips markdown code fences before parsing, same as ClaudeProvider.
- Raises `RuntimeError` on non-zero exit code with stderr details.

### `src/fetcher/arxiv_fetcher.py` — ArxivFetcher
- Fetches daily papers from user-configured arXiv categories using the `arxiv` Python library.
- `__init__(config)` — reads `config["arxiv"]["categories"]` (list of category strings) and `config["arxiv"]["max_results_per_category"]`.
- `fetch_today(cutoff_days=2)` — async entry point. Iterates over categories, calls `_fetch_category` via `asyncio.to_thread` to avoid blocking the event loop. Deduplicates results across categories. Default `cutoff_days=2` accounts for timezone/indexing delays.
- `_fetch_category(category, cutoff_date)` — queries `arxiv.Search(query="cat:{category}", sort_by=SubmittedDate)`. The arXiv API sorts but does **not** filter by date, so filtering is done in Python (`published.date() >= cutoff_date`). Strips version suffix from arxiv_id via regex (`2501.12345v2` → `2501.12345`). Strips newlines from title and abstract. Constructs ar5iv URL from arxiv_id.
- `_deduplicate(papers)` — removes duplicates by arxiv_id, keeping first occurrence. Needed because a paper can appear in multiple categories.
- Returns list of dicts with keys: `arxiv_id`, `title`, `authors`, `abstract`, `categories`, `published_date`, `pdf_url`, `ar5iv_url`.

Used by: `DailyPipeline` (Phase 10), `InterestManager._fetch_abstract_from_arxiv` uses the same `arxiv` library directly (Phase 5).

---

## Not Yet Implemented

| File | Phase | Purpose |
|------|-------|---------|
| `src/store/database.py` | 1 | SQLite PaperStore — 5 tables (papers, interests, matches, summaries, daily_reports) |
| `src/matcher/embedder.py` | 4 | sentence-transformers embedding + cosine similarity matching |
| `src/interest/manager.py` | 5 | Interest CRUD with auto-fetch abstracts + embedding recomputation |
| `src/matcher/ranker.py` | 6 | LLM re-ranking with concurrent scoring (asyncio.gather + Semaphore) |
| `src/report/generator.py` | 7 | Markdown report generation (general + specific) |
| `src/email/sender.py` | 8 | SMTP email with Markdown → HTML → CSS-inline pipeline |
| `src/summarizer/paper_summarizer.py` | 9 | ar5iv HTML parsing + LLM summarization (GUI-only) |
| `src/pipeline.py` | 10 | DailyPipeline orchestrator |
| `src/main.py` | 11 | CLI entry point (--mode scheduler\|run) |
| `src/scheduler/scheduler.py` | 11 | APScheduler CronTrigger wrapper |
| `gui/` | 12 | Streamlit 5-page app |

---

## Key Design Decisions

| Decision | Approach | Rationale |
|----------|----------|-----------|
| Async LLM interface | All providers use `async def` | Enables concurrent re-ranking (asyncio.gather) |
| Factory pattern | `create_llm_provider()` with lazy imports | Avoids loading unused SDKs (openai vs anthropic) |
| JSON fence stripping | Regex strip of ````json ... ```` | Claude models wrap JSON in code blocks despite instructions |
| Claude CLI integration | subprocess via `asyncio.create_subprocess_exec` | Zero cost, leverages existing subscription |
| System message in CLI | Prepended to prompt | CLI has no separate system parameter |
| ArXiv date filtering | Python-side filter after fetch | arXiv API sorts by SubmittedDate but doesn't filter by date |
| ArXiv async wrapping | `asyncio.to_thread` around sync `arxiv.Client` | Avoids blocking the event loop; arxiv lib is synchronous |
| ArXiv ID normalization | Regex strip `v\d+$` suffix | Ensures consistent IDs for deduplication and DB storage |
