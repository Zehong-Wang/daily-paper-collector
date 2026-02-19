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

### `src/matcher/embedder.py` — Embedder
Local embedding computation and cosine similarity matching using `sentence-transformers`.

- `__init__(config)` — reads `config["matching"]["embedding_model"]` (default `all-MiniLM-L6-v2`, 384 dims). Model is **lazy-loaded** via `@property` — the ~80MB model downloads/loads only on first call to `embed_text` or `embed_texts`, not at init time.
- `embed_text(text) -> np.ndarray` — Embeds a single string. Returns a normalized 1D array of shape `(384,)`.
- `embed_texts(texts) -> np.ndarray` — Batch embedding. Returns a normalized 2D array of shape `(N, 384)`. Uses `show_progress_bar=False` for clean logs.
- `serialize_embedding(embedding) -> bytes` — Converts `np.float32` array to raw bytes for SQLite BLOB storage. Static method.
- `deserialize_embedding(blob, dim=384) -> np.ndarray` — Converts bytes back to numpy array. Handles both exact-dim and arbitrary-length blobs. Static method.
- `compute_embeddings(papers, store)` — Batch-embeds paper abstracts via `embed_texts()`, serializes each, and calls `store.update_paper_embedding(id, blob)`. Accepts any object with that method (duck-typed).
- `compute_interest_embeddings(interests, store)` — Embeds each interest individually. For interests with a description, embeds `"{value}: {description}"`; otherwise just `"{value}"`. Calls `store.update_interest_embedding(id, blob)`.
- `find_similar(interests, papers, top_n, threshold=0.3) -> list[dict]` — Two-stage cosine similarity matching:
  1. Deserializes all interest and paper embeddings from bytes blobs into numpy matrices.
  2. Computes similarity matrix via `papers_matrix @ interests_matrix.T` → shape `(N, M)`.
  3. Takes MAX score per paper across all interests (a paper matching any one interest well is sufficient).
  4. Filters by threshold, sorts descending, returns top-N papers with `embedding_score` field added.

Used by: `DailyPipeline` (Phase 10) for coarse filtering, `InterestManager` (Phase 5) for embedding computation.

### `src/store/database.py` — PaperStore (SQLite CRUD Layer)
Central persistence layer. All data flows through this class — papers, interests, matches, summaries, and reports.

- `__init__(db_path)` — Stores path, calls `_init_db()` to create schema. Uses `logging.getLogger(__name__)`.
- `_init_db()` — Creates all 5 tables via `CREATE TABLE IF NOT EXISTS` (idempotent). Enables `PRAGMA journal_mode=WAL` (concurrent reads during writes) and `PRAGMA foreign_keys=ON`.
- `_get_conn() -> sqlite3.Connection` — Creates a new connection each call with `row_factory = sqlite3.Row` for dict-like access. Every public method opens and closes its own connection (no shared connection state — safe for multi-threaded access from Streamlit).
- `_row_to_paper(row) -> dict` — Converts `sqlite3.Row` to dict, deserializing `authors` and `categories` from JSON strings back to Python lists.

**Paper methods:**
- `save_papers(papers) -> list[dict]` — Bulk insert via `INSERT OR IGNORE`. Deduplication by `arxiv_id` UNIQUE constraint. Returns only newly inserted papers (checks `cursor.rowcount > 0`), each augmented with their assigned `id`.
- `get_paper_by_arxiv_id(arxiv_id) -> dict | None` — Single paper lookup.
- `get_papers_by_date(date) -> list[dict]` — All papers for a given `published_date`.
- `search_papers(query, limit=50) -> list[dict]` — `LIKE '%query%'` on title and abstract.
- `update_paper_embedding(paper_id, embedding_bytes)` — Updates BLOB column.
- `get_papers_without_embeddings() -> list[dict]` — Papers with `embedding IS NULL`.
- `get_papers_with_embeddings() -> list[dict]` — Papers with `embedding IS NOT NULL`, includes the BLOB.
- `get_papers_by_date_with_embeddings(date) -> list[dict]` — Combines date filter + embedding filter. Used by the pipeline to match only today's papers.

**Interest methods:**
- `save_interest(type, value, description=None) -> int` — Insert, return new ID.
- `get_all_interests() -> list[dict]`, `get_interest_by_id(id) -> dict | None`
- `update_interest(id, value=None, description=None)` — Partial update: only sets non-None fields.
- `delete_interest(id)`, `update_interest_embedding(id, embedding_bytes)`
- `get_interests_with_embeddings() -> list[dict]` — Interests with computed embeddings.

**Match methods:**
- `save_match(paper_id, run_date, embedding_score, llm_score=None, llm_reason=None) -> int`
- `get_matches_by_date(run_date) -> list[dict]` — JOINs with papers table to include title, arxiv_id, abstract, authors, categories, pdf_url. Ordered by `llm_score DESC` with NULLS LAST (via CASE expression), then `embedding_score DESC`.

**Summary methods:**
- `save_summary(paper_id, summary_type, content, llm_provider=None) -> int`
- `get_summary(paper_id, summary_type) -> dict | None` — Cache lookup for paper summarization.

**Report methods:**
- `save_report(run_date, general_report, specific_report, paper_count, matched_count) -> int`
- `get_report_by_date(run_date) -> dict | None`
- `get_all_report_dates() -> list[str]` — Sorted descending.

Used by: Every component that touches persistent data — `DailyPipeline`, `InterestManager`, `Embedder.compute_embeddings`, `PaperSummarizer`, all GUI pages.

### `src/matcher/ranker.py` — LLMRanker
Second stage of the two-stage matching pipeline: LLM-based re-ranking of embedding-filtered candidates.

- `__init__(llm, config)` — Takes an `LLMProvider` instance and reads `config["matching"]["llm_top_k"]` for the default number of results to return. Uses `logging.getLogger(__name__)`.
- `rerank(candidates, interests, top_k=None, max_concurrent=5) -> list[dict]` — Concurrent re-ranking entry point. Formats interests into a text block, then scores all candidates in parallel using `asyncio.gather` with an `asyncio.Semaphore(max_concurrent)` to limit concurrency. Sorts results by `llm_score` descending and returns the top-K. The `top_k` parameter overrides the config default if provided.
- `_score_paper(paper, interests_text) -> dict` — Prompts the LLM via `complete_json()` to score a single paper's relevance (1-10 scale) against the formatted interests text. Returns `{"llm_score": float, "llm_reason": str}`. On any exception (invalid JSON, API timeout, etc.), returns `{"llm_score": 0, "llm_reason": "Scoring failed"}` — never crashes the batch.
- `_format_interests(interests) -> str` — Converts a list of interest dicts into a readable bullet list for the LLM prompt. Each entry formatted as `"- {type}: {value} ({description})"`. Returns `"No interests specified."` for empty lists.

Used by: `DailyPipeline` (Phase 10) as the fine-grained second stage after `Embedder.find_similar()`.

### `src/report/generator.py` — ReportGenerator
Markdown report generation for both general (all daily papers) and specific (interest-matched papers) reports.

- `__init__(llm)` — Takes an `LLMProvider` instance. Uses `logging.getLogger(__name__)`.
- `generate_general(papers, run_date) -> str` — Builds a complete Markdown general report with three sections:
  1. **Today's Overview** (`_build_overview`) — Pure Python; uses `collections.Counter` on each paper's primary category (first element of `categories` list). Formats as `"cs.AI: 3 | cs.CL: 4 | ..."`.
  2. **Trending Topics** (`_build_trending_topics`) — Sends all paper titles to the LLM to identify 3-5 emerging research topics. Catches LLM exceptions gracefully.
  3. **Highlight Papers** (`_build_highlight_papers`) — Sends paper titles + first 150 chars of abstract + first 3 authors to the LLM to select 3-5 noteworthy papers. Catches LLM exceptions gracefully.
- `generate_specific(ranked_papers, interests, run_date) -> str` — Formats pre-scored data from the ranker into Markdown. **Does NOT call the LLM.** Two sections:
  1. Numbered list of ranked papers with `llm_score/10` and `llm_reason`.
  2. "Related Papers" section with authors, categories, abstract preview (first 200 chars), and arXiv link for each paper.
  Handles edge cases: empty results, string-type authors/categories (not just lists).

Used by: `DailyPipeline` (Phase 10) for generating both report types after matching.

---

## Not Yet Implemented

| File | Phase | Purpose |
|------|-------|---------|
| `src/interest/manager.py` | 5 | Interest CRUD with auto-fetch abstracts + embedding recomputation |
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
| Lazy model loading | `@property` with `_model = None` | Avoids loading ~80MB model until first embedding call |
| Normalized embeddings | `normalize_embeddings=True` in encode | Dot product = cosine similarity, no need for separate normalization |
| MAX interest scoring | `similarity_matrix.max(axis=1)` | A paper matching any single interest well is sufficient for recommendation |
| Duck-typed store parameter | `compute_embeddings(papers, store)` | Decouples embedder from concrete PaperStore; testable with MagicMock |
| Connection-per-call | `_get_conn()` creates new connection each method call | Safe for multi-threaded Streamlit; no shared mutable connection state |
| JSON serialization for lists | `authors`/`categories` stored as JSON strings | SQLite has no array type; JSON round-trips cleanly via `json.dumps`/`json.loads` |
| NULLS LAST ordering | `CASE WHEN llm_score IS NULL THEN 1 ELSE 0 END` | Papers with LLM scores ranked above unscored ones in match results |
| INSERT OR IGNORE dedup | `save_papers` uses `arxiv_id UNIQUE` constraint | Relies on DB-level uniqueness; no pre-query needed for duplicate detection |
| Concurrent LLM scoring | `asyncio.gather` + `Semaphore(max_concurrent)` | 5-10x faster than sequential for 50 candidates; semaphore prevents API rate-limit issues |
| Graceful scoring failure | `_score_paper` catches all exceptions, returns score 0 | One failed LLM call should not abort scoring of remaining candidates |
| Overview without LLM | `_build_overview` uses `Counter` in pure Python | Category counts are deterministic; no LLM cost for simple aggregation |
| Specific report is LLM-free | `generate_specific` only formats pre-scored data | Scores and reasons already computed by `LLMRanker`; no redundant LLM calls |
| Graceful LLM failure in reports | `_build_trending_topics`/`_build_highlight_papers` catch all exceptions | Report generation succeeds even if LLM is unavailable; error message replaces section content |
| Primary category = first element | `categories[0]` used for overview breakdown | arXiv lists primary category first; consistent with how papers are categorized |
