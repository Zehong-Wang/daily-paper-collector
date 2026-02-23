# Architecture — Daily Paper Collector

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
Sections: `arxiv` (categories, max_results_per_category, cutoff_days, page_size), `matching` (model, top_n, top_k, threshold), `llm` (provider [default: `claude_code`] + sub-configs for openai/claude/claude_code [with timeout, max_retries, max_concurrent]), `report` (chinese: true/false — enables Chinese report/summary generation), `email` (SMTP settings), `scheduler` (cron), `database` (path), `gui` (port).

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
- **Default LLM provider** — zero API cost via existing Claude Code subscription.
- Calls `claude` CLI via `asyncio.create_subprocess_exec` with flags: `--print`, `--model <model>`, `--output-format json`, `--no-session-persistence`.
- `__init__(config)` — Reads `cli_path` (default `"claude"`), `model` (default `"sonnet"`), `timeout` (default 120s), `max_retries` (default 3). Validates CLI availability via `shutil.which(cli_path)` — raises `RuntimeError` if CLI not found.
- `complete(prompt, system)` — Calls `_run_cli()` (with retry), parses the JSON envelope from `--output-format json` to extract `envelope["result"]`. Falls back to raw stdout if envelope parsing fails. System prompt passed via `--system-prompt` CLI flag (not concatenated with user prompt).
- `complete_json(prompt, system)` — Calls `complete()`, strips markdown code fences as fallback, parses result as JSON. Raises `ValueError` on invalid JSON (not retried).
- `_run_cli(prompt, system)` — Retry loop: up to `max_retries` attempts with exponential backoff (1s, 2s, 4s) on `RuntimeError`. Does not retry `ValueError`.
- `_execute_subprocess(prompt, system)` — Builds CLI command, runs subprocess with `asyncio.wait_for(timeout=self.timeout)`. On timeout: kills process, raises `RuntimeError`. On non-zero exit: raises `RuntimeError` with stderr.
- Config options in `config["llm"]["claude_code"]`: `cli_path`, `model`, `timeout`, `max_retries`, `max_concurrent`.

### `src/fetcher/arxiv_fetcher.py` — ArxivFetcher
- Fetches daily papers from user-configured arXiv categories using the `arxiv` Python library.
- `__init__(config)` — reads `config["arxiv"]["categories"]`, `max_results_per_category` (default 500, safety cap), `cutoff_days` (default 1), and `page_size` (default 500). Creates a shared `arxiv.Client(page_size=..., delay_seconds=3.0, num_retries=3)` once at init (reused across categories).
- `fetch_today(cutoff_days=None)` — async entry point. When `cutoff_days` is `None`, uses config default. Computes `start_date = today - cutoff_days` and `end_date = today`, passes both to `_fetch_category` via `loop.run_in_executor` to avoid blocking the event loop. Deduplicates results across categories.
- `_build_date_query(category, start_date, end_date)` — builds an arXiv API query with **server-side date filtering**: `"cat:{category} AND submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]"`. Start is 00:00 UTC of start_date, end is 23:59 UTC of end_date.
- `_fetch_category(category, start_date, end_date)` — uses `_build_date_query` for server-side filtering via `arxiv.Search`, then applies a **client-side safety net** (`published.date() >= start_date`). Uses the shared `self.client` for auto-paginating iteration. Strips version suffix from arxiv_id, strips newlines from title/abstract, constructs ar5iv URL. **Error handling:** wraps API calls in `try/except Exception` — on failure, logs the error and returns an empty list for that category.
- `_deduplicate(papers)` — removes duplicates by arxiv_id, keeping first occurrence.
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
- `_init_db()` — Creates all 5 tables via `CREATE TABLE IF NOT EXISTS` (idempotent). Enables `PRAGMA journal_mode=WAL` (concurrent reads during writes) and `PRAGMA foreign_keys=ON`. Runs `_migrate_add_column()` to add `general_report_zh` and `specific_report_zh` columns to existing `daily_reports` tables (backward-compatible migration).
- `_migrate_add_column(conn, table, column, col_type)` — Static method. Checks `PRAGMA table_info()` for existing columns; only runs `ALTER TABLE ADD COLUMN` if the column doesn't exist. Used for safe schema migration on existing databases.
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
- `get_papers_by_date_with_embeddings(date) -> list[dict]` — Combines date filter + embedding filter.
- `get_papers_by_ids_with_embeddings(paper_ids) -> list[int]` — Returns papers with given IDs that have embeddings. Used by the pipeline to match only newly inserted papers (avoids overlap across daily runs).

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
- `save_report(run_date, general_report, specific_report, paper_count, matched_count, general_report_zh=None, specific_report_zh=None, report_type="daily") -> int` — Persists both English and optional Chinese reports. `report_type` can be `"daily"`, `"3day"`, or `"1week"`. Multi-day reports use `run_date` in range-label format `"start~end"` (e.g., `"2026-02-20~2026-02-22"`).
- `get_report_by_date(run_date) -> dict | None` — Returns dict including `general_report_zh` and `specific_report_zh` fields.
- `get_all_report_dates() -> list[str]` — Sorted descending. Returns distinct run_dates.
- `get_all_report_entries() -> list[dict]` — Returns all report entries with metadata (`id`, `run_date`, `report_type`, `paper_count`, `matched_count`, `created_at`) sorted by `created_at DESC, id DESC`. Used by the Reports page for the type-aware report selector.
- `get_report_by_id(report_id) -> dict | None` — Returns a single report by primary key. Needed to distinguish between daily and multi-day reports with different `run_date` formats.

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
- `generate_general(papers, run_date, date_label=None) -> str` — Builds a complete Markdown general report. When `date_label` is provided (for multi-day reports, e.g. `"2026-02-20 ~ 2026-02-22"`), uses it in the header and switches to period-aware wording ("Period Overview" / "papers in this period" instead of "Today's Overview" / "new papers collected"). Three sections:
  1. **Today's Overview** (`_build_overview`) — Pure Python; uses `collections.Counter` on each paper's primary category (first element of `categories` list). Formats top-10 categories as a Markdown table with an "Others" row summarizing remaining categories.
  2. **Trending Topics** (`_build_trending_topics`) — Sends all paper titles to the LLM to identify 3-5 emerging research topics. LLM prompt explicitly forbids headings to prevent visual hierarchy conflicts. Catches LLM exceptions gracefully.
  3. **Highlight Papers** (`_build_highlight_papers`) — Sends paper titles + first 150 chars of abstract + first 3 authors to the LLM to select 3-5 noteworthy papers. LLM prompt explicitly forbids headings. Catches LLM exceptions gracefully.
- `generate_specific(ranked_papers, interests, run_date, date_label=None) -> str` — Generates a theme-based synthesis report using the LLM, followed by comprehensive paper details. When `date_label` is provided, uses "in this period" instead of "today" in intro text. Two sections:
  1. **Theme-based synthesis** (`_build_theme_synthesis`) — For >= 5 papers, sends paper titles, full abstracts, scores, and reasons to the LLM to group into 3-6 thematic clusters with `###` headings and flowing narrative paragraphs. On LLM failure, falls back to `_build_fallback_list` (numbered list with scores and arXiv links). For < 5 papers, uses `_build_simple_summary` (bullet list, no LLM call).
  2. **Paper Details** (`_build_paper_details`) — Comprehensive details for each paper: score, categories, **full author list** (no truncation), **full abstract** (no truncation), relevance reason (`llm_reason`), and arXiv link.
  Handles edge cases: empty results, string-type authors/categories (not just lists), LLM failures.
- `generate_general_zh(papers, run_date, date_label=None) -> str` — Chinese version of `generate_general`. When `date_label` is provided, uses 周期概览/本期共收录 instead of 今日概览/今日共收录. Three sections: 今日概览 (`_build_overview_zh`), 热门研究方向 (`_build_trending_topics_zh`), 亮点论文 (`_build_highlight_papers_zh`). All LLM prompts use Chinese system messages and instruct the model to respond in Chinese.
- `generate_specific_zh(ranked_papers, interests, run_date, date_label=None) -> str` — Chinese version of `generate_specific`. When `date_label` is provided, uses 本期 instead of 今日. Theme synthesis via `_build_theme_synthesis_zh` (LLM groups papers into Chinese thematic clusters), fallback via `_build_fallback_list_zh`, simple summary via `_build_simple_summary_zh`. Paper details via `_build_paper_details_zh` with Chinese labels (评分, 分类, 作者, 摘要, 推荐理由).

Used by: `DailyPipeline` (Phase 10) for generating both report types after matching. Chinese methods called conditionally when `config["report"]["chinese"]` is enabled.

### `src/email/sender.py` — EmailSender
SMTP email delivery with a Markdown → HTML → CSS-inline rendering pipeline.

- `__init__(config)` — Reads `config["email"]` sub-dict. Extracts SMTP settings (host, port) from `config["email"]["smtp"]`. Reads username and password from environment variables specified by `smtp.username_env` and `smtp.password_env`. Stores `from_address`, `to_addresses` (list), and `subject_prefix`. Uses `logging.getLogger(__name__)`.
- `render_markdown_to_html(md_content) -> str` — Three-step rendering pipeline:
  1. Converts Markdown to HTML via `markdown.markdown()` with `tables` and `fenced_code` extensions.
  2. Wraps the HTML body in a styled HTML template with a `.wrapper` div (720px max-width) and comprehensive CSS: light gray background, blue-accented h2 headings with left border and background, green-accented blockquotes for LLM reasons, styled tables with alternating row colors, proper list spacing, and responsive viewport meta tag.
  3. Inlines all CSS via `premailer.transform()` — required because most email clients strip `<style>` tags.
- `send(general_report, specific_report, ranked_papers, run_date, general_zh=None, specific_zh=None, subject_override=None)` — Async entry point. Combines English and optional Chinese Markdown reports via `_combine_reports()`, renders to HTML via `render_markdown_to_html()`, builds a MIME message via `_build_email()`, then sends via `_send_smtp()` wrapped in `asyncio.to_thread()` to avoid blocking the event loop. When `subject_override` is provided, it replaces the default `"{subject_prefix} {run_date}"` subject line.
- `send_sync(general_report, specific_report, run_date, general_zh=None, specific_zh=None, subject_override=None)` — Synchronous variant used by the GUI. Same combine/render/send flow without async wrapping. `subject_override` allows multi-day reports to set subjects like `"[Daily Papers] 3-Day Report - 2026-02-20~2026-02-22"`.
- `_combine_reports(general, specific, general_zh=None, specific_zh=None) -> str` — Merges English and Chinese reports into a single Markdown document with `---` separators. Chinese sections appended after English sections when present.
- `_build_email(html_content, subject) -> MIMEMultipart` — Constructs a `MIMEMultipart("alternative")` message with Subject, From, To headers and a single `MIMEText("...", "html")` attachment.
- `_send_smtp(msg)` — Synchronous SMTP sending using `smtplib.SMTP` context manager: `starttls()` → `login()` → `send_message()`. Called from a thread via `asyncio.to_thread`. **Error handling (Phase 14):** wraps SMTP operations in `try/except smtplib.SMTPException` — logs the error and re-raises so the pipeline's existing try/except can handle it gracefully.

Used by: `DailyPipeline` (Phase 10) for delivering the daily email after report generation.

### `src/summarizer/paper_summarizer.py` — PaperSummarizer
GUI-only component for on-demand paper summarization. Not part of the daily automated pipeline.

- `__init__(llm, store)` — Takes an `LLMProvider` instance and a `PaperStore` instance. Uses `logging.getLogger(__name__)`.
- `fetch_paper_text(ar5iv_url) -> str` — Fetches ar5iv HTML page via `requests.get(url, timeout=30)`. Parses with `BeautifulSoup(html, "lxml")`. Content extraction priority: `<article>` tag → class `ltx_document` → class `ltx_page_main` → `<body>` → entire soup. Extracts text from all `<p>`, `<h2>`, `<h3>` tags within the found container. Joins with double newlines. Includes heuristics to reject ar5iv/arXiv navigation-shell pages (help/search/citation UI without paper sections). Truncates to 15,000 characters (LLM context limit safety). Raises `RuntimeError` on HTTP errors, shell-content extraction, or empty extraction.
- `fetch_pdf_text(pdf_url) -> str` — Fallback path when ar5iv extraction fails. Downloads PDF and extracts text via `pypdf.PdfReader`, concatenates page text, truncates to 15,000 characters. Raises `RuntimeError` on fetch/parse/extraction failure.
- `summarize(paper_id, mode="brief") -> str` — Async entry point. Cache-first: checks `store.get_summary(paper_id, mode)` and returns cached content immediately if found. Retrieves paper by integer ID via `_get_paper_by_id()`. Full-text fallback chain: `ar5iv` HTML → `pdf_url` extraction → abstract fallback. Builds a mode-specific prompt:
  - `"brief"`: asks for 1-2 paragraphs covering core contributions and methodology.
  - `"detailed"`: asks for structured sections (Motivation, Method, Experiments, Conclusions, Limitations).
  - `"brief_zh"`: Chinese 1-2 paragraph summary. System prompt: "你是一位科学论文摘要专家。请提供清晰、准确的中文摘要。"
  - `"detailed_zh"`: Chinese structured summary with sections: 研究动机, 研究方法, 实验结果, 主要结论, 局限性.
  Calls `llm.complete(prompt, system=...)` with mode-appropriate system prompt. Saves the result to cache via `store.save_summary()` with the LLM provider class name. Raises `ValueError` if `paper_id` is not found in the database.
- `_get_paper_by_id(paper_id) -> dict | None` — Queries papers table by integer `id` (PaperStore only exposes `get_paper_by_arxiv_id`). Accesses `store._get_conn()` directly, deserializes `authors` and `categories` from JSON. Returns `None` if not found.

Used by: Streamlit GUI Papers page (Phase 12) for on-demand brief/detailed/brief_zh/detailed_zh summaries. Summaries are cached in the `summaries` table (keyed by `paper_id` + `summary_type`) to avoid redundant LLM calls.

### `src/interest/manager.py` — InterestManager
Manages the three types of user interests (keyword, paper, reference_paper) with automatic embedding computation.

- `__init__(store, embedder)` — Takes a `PaperStore` and an `Embedder` instance. Uses `logging.getLogger(__name__)`.
- `add_keyword(keyword, description=None) -> int` — Saves a keyword interest to the store, computes its embedding (using `"{keyword}: {description}"` if description provided, else just `"{keyword}"`), stores the embedding blob, returns the new ID.
- `add_paper(arxiv_id, description=None) -> int` — Saves a past-paper interest. If no description provided, auto-fetches the abstract via a 3-tier lookup: (1) check DB via `store.get_paper_by_arxiv_id`, (2) fetch from arXiv API via `_fetch_abstract_from_arxiv`, (3) fall back to using the arxiv_id string itself (logs a warning). Computes and stores the embedding.
- `add_reference_paper(arxiv_id, description=None) -> int` — Same auto-fetch logic as `add_paper` but saves with `type="reference_paper"`.
- `_fetch_abstract_from_arxiv(arxiv_id) -> str | None` — Uses `arxiv.Search(id_list=[arxiv_id])` to fetch a single paper's metadata. Returns the abstract with newlines stripped, or `None` on any failure. Catches all exceptions to avoid crashing the caller.
- `update_interest(id, value=None, description=None)` — Updates the interest fields in the store, then recomputes and stores a new embedding from the updated text.
- `remove_interest(id)` — Deletes via `store.delete_interest`.
- `get_all_interests()`, `get_interests_with_embeddings()` — Delegate to store.
- `recompute_all_embeddings()` — Iterates all interests, recomputes embeddings. Useful after switching embedding models.

Used by: `DailyPipeline` (Phase 10) to get interests with embeddings for matching. Streamlit GUI Interests page (Phase 12) for CRUD operations.

### `src/pipeline.py` — DailyPipeline
Central orchestrator that wires all components together and executes the daily paper collection pipeline.

- `__init__(config)` — Instantiates all 8 components from a single config dict:
  - `PaperStore(config["database"]["path"])` — persistence
  - `ArxivFetcher(config)` — paper fetching
  - `Embedder(config)` — embedding computation + similarity matching
  - `create_llm_provider(config)` — LLM for re-ranking + report generation
  - `LLMRanker(llm, config)` — second-stage re-ranking
  - `InterestManager(store, embedder)` — interest management
  - `ReportGenerator(llm)` — Markdown report generation
  - `EmailSender(config)` — SMTP email delivery
  - Also reads `max_concurrent` from the active LLM provider's config (defaults to 5; `claude_code` uses 2).
  - Reads `self.chinese_enabled` from `config["report"]["chinese"]` (defaults to `False`). When enabled, generates Chinese reports alongside English reports.
- `run() -> dict` — Async method executing the 12-step pipeline:
  1. **Fetch** — `fetcher.fetch_today()` retrieves papers from arXiv
  2. **Save** — `store.save_papers()` persists with deduplication, returns only new papers
  3. **Embed** — `embedder.compute_embeddings(new_papers, store)` computes abstract embeddings
  4. **Check interests** — `interest_mgr.get_interests_with_embeddings()`; if empty, generates general report only and returns early
  5. **Match** — `store.get_papers_by_ids_with_embeddings(new_paper_ids)` gets only newly inserted papers with embeddings (eliminates cross-run overlap), then `embedder.find_similar()` computes top-N candidates
  6. **Re-rank** — `ranker.rerank(candidates, interests, max_concurrent=self.max_concurrent)` scores via LLM for top-K
  7. **Save matches** — `store.save_match()` per ranked paper
  8. **General report** — `report_gen.generate_general(new_papers, run_date)`
  9. **Specific report** — `report_gen.generate_specific(ranked, interests, run_date)`
  9b. **Chinese reports** (if `chinese_enabled`) — `report_gen.generate_general_zh()` and `report_gen.generate_specific_zh()` called after English reports
  10. **Email** — `email_sender.send()` if `config["email"]["enabled"]`; passes Chinese reports when available; failures caught and logged
  11. **Save report** — `store.save_report()` persists both English and Chinese reports
  12. **Return** — `{"date", "papers_fetched", "new_papers", "matches", "email_sent"}`

- `run_range_report(start_date, end_date, report_type) -> dict` — Generates a consolidated report for papers in the given date range. Unlike `run()`, does NOT fetch new papers from arXiv or compute embeddings. Re-runs the matching pipeline (embedding similarity + LLM re-ranking) on all papers already in the DB for the date range.
  - Uses `run_date_label = f"{start_date}~{end_date}"` for DB storage (both matches and report).
  - Uses `date_label = f"{start_date} ~ {end_date}"` for display in report headers.
  - Generates both EN and ZH reports (when `chinese_enabled`), passing `date_label` to all report methods.
  - Does NOT send email — the GUI handles that separately via the "Send Report via Email" button.
  - Returns `{"date_range", "report_type", "papers_count", "matches"}`.
  - Early returns with zero counts if no papers found in range.
  - Generates general-only report if no interests configured.

Used by: `src/main.py` (Phase 11) in both `--mode run` and `--mode scheduler` modes. `scripts/run_pipeline.py` for CI/CD. `gui/views/dashboard.py` for multi-day report generation.

### `src/scheduler/scheduler.py` — PipelineScheduler
Wraps APScheduler's `BlockingScheduler` to run the daily pipeline on a cron schedule.

- `__init__(config)` — Stores the full config dict. Creates a `BlockingScheduler` instance. Uses `logging.getLogger(__name__)`.
- `start()` — Parses the 5-field cron string from `config["scheduler"]["cron"]` (format: `"M H day month day_of_week"`), creates a `CronTrigger` with the parsed fields, adds `_run_pipeline` as a job, then calls `scheduler.start()` which blocks forever. Logs the cron expression on start.
- `_run_pipeline()` — Lazily imports `DailyPipeline` (to avoid circular imports at module load time), instantiates it with the stored config, and runs it synchronously via `asyncio.run(pipeline.run())`. Logs the result.

Used by: `src/main.py` in `--mode scheduler`.

### `src/main.py` — CLI Entry Point
The primary command-line interface for running the paper collector.

- `main()` — Uses `argparse` with two arguments:
  - `--mode` (`scheduler`|`run`, default `run`) — `scheduler` starts the cron-based scheduler (blocks forever), `run` executes the pipeline once and exits.
  - `--config` (optional) — Path to a custom config YAML file; defaults to `config/config.yaml` relative to project root.
- Calls `setup_logging()` and `load_config(args.config)` at startup (lazy imports from `src.config` inside the function body).
- In `scheduler` mode: lazily imports `PipelineScheduler`, instantiates, and calls `start()`.
- In `run` mode: lazily imports `DailyPipeline`, instantiates, and runs via `asyncio.run(pipeline.run())`.
- All imports are local to `main()` to avoid triggering heavy module loads (e.g., sentence-transformers, openai, anthropic) at import time.

Used by: `python -m src.main --mode run` or `python -m src.main --mode scheduler`.

### `scripts/run_pipeline.py` — CI/CD Entry Point
Standalone script for GitHub Actions and other CI/CD environments.

- Adds the project root to `sys.path` so `src` and `scripts` packages are importable regardless of working directory.
- `main()` — Calls `setup_logging()`, `load_config()`, creates a `DailyPipeline`, runs it via `asyncio.run()`, and logs a warning if `result["new_papers"] == 0`.
- Uses top-level imports (unlike `src/main.py`) since this script is always invoked directly, not imported as a library.

Used by: `python scripts/run_pipeline.py` from GitHub Actions workflows.

### `gui/app.py` — Streamlit Main Entry Point
Main Streamlit entry with sidebar radio navigation across 5 pages (Dashboard, Papers, Interests, Reports, Settings).

- `get_config()` — `@st.cache_resource` cached config loader. Delegates to `load_config()`.
- `get_store()` — `@st.cache_resource` cached `PaperStore` instance. Avoids recreating DB connections on every Streamlit rerun.
- `get_embedder()` — `@st.cache_resource` cached `Embedder` instance. Avoids reloading the ~80MB sentence-transformer model on every rerun.
- `main()` — Sets page config, renders sidebar radio, dynamically imports the selected page module, and calls `render(get_store())`.
- `sys.path.insert(0, ...)` ensures project root is importable regardless of working directory.

Used by: `streamlit run gui/app.py`. Page modules import `get_embedder` from here.

### `gui/pages/dashboard.py` — Dashboard Page
Three metrics row (Papers Today, Matches Today, total Reports) via `st.metric`. Report previews (first 1000 chars) from today's report in General/Specific tabs. "Run Pipeline Now" button creates a `DailyPipeline` and runs via `asyncio.run()` with `st.spinner` feedback. **Multi-Day Reports section**: Two buttons — "Generate 3-Day Report" and "Generate 1-Week Report" — each calls `pipeline.run_range_report()` with the appropriate date range and report type. Uses `_run_range_report()` helper with `st.spinner` and success/warning feedback.

Used by: `gui/app.py` when page == "Dashboard".

### `gui/pages/papers.py` — Papers Browse & Summarize Page
Date selector (`st.date_input`) and search box (`st.text_input`). Search calls `store.search_papers()`; date mode calls `store.get_papers_by_date()`. Papers displayed in a compact `st.dataframe` table (Title, Authors truncated to 3, Primary Category, Date, arXiv link). Row selection (`on_select="rerun"`, `selection_mode="single-row"`) shows a detail panel below the table with full authors, all categories, full abstract, arXiv link, and four summary buttons: Brief Summary, Detailed Summary, 中文简要总结 (`brief_zh`), 中文详细总结 (`detailed_zh`). All four modes trigger `PaperSummarizer` on demand and are cached independently in the `summaries` table.

Used by: `gui/app.py` when page == "Papers".

### `gui/pages/interests.py` — Interest CRUD Page
Lists current interests with type/value/description, embedding status (Y/N), and delete button per row. "Add New Interest" form with type selector (`keyword`|`paper`|`reference_paper`), value input, optional description. On submit, creates `InterestManager(store, get_embedder())` and calls the appropriate `add_*` method.

Used by: `gui/app.py` when page == "Interests".

### `gui/pages/reports.py` → `gui/views/reports.py` — Reports Viewer Page
**Type-aware report selector** from `store.get_all_report_entries()`. The selectbox shows labels like `"2026-02-22 (Daily) - 156 papers, 12 matches"` or `"2026-02-20~2026-02-22 (3-Day) - 450 papers, 25 matches"`. Uses `store.get_report_by_id()` for report lookup (instead of date-based lookup) to correctly handle multiple report types. Dynamically builds tab list: always English tabs (General Report, Specific Report), plus Chinese tabs (综合报告, 个性化推荐) when Chinese report content exists for the selected report.

**General Report tab**: Renders the full general report markdown (overview, trending topics, highlights).

**Specific Report tab** — Two blocks:
1. **Theme synthesis narrative**: The stored specific report is split at the `---` divider (via `_split_specific_report()`). The synthesis portion (before the divider) is rendered as markdown.
2. **Matched papers table**: Individual matched papers from `store.get_matches_by_date(run_date)` shown in a compact `st.dataframe` table (via `_render_matches_table()`). For multi-day reports, `run_date` is the range label (e.g., `"2026-02-20~2026-02-22"`), which correctly retrieves matches saved under that label. Columns: #, Title, Score ("{N}/10"), Primary Category, Relevance (truncated to 80 chars), arXiv link. Row selection shows full details below (via `_render_match_detail()`) including all scores, full authors, full abstract, full relevance reason, and arXiv link.

**综合报告 (Chinese) tab**: Renders `general_report_zh` markdown when available.

**个性化推荐 (Chinese) tab**: Renders `specific_report_zh` synthesis + matched papers table (same as English Specific Report tab).

**Send Report via Email** button passes both English and Chinese reports to `EmailSender.send_sync()`. For multi-day reports, generates a `subject_override` like `"[Daily Papers] 3-Day Report - 2026-02-20~2026-02-22"`.

Used by: `gui/app.py` when page == "Reports".

### `gui/pages/settings.py` — Settings Page
Read-only config display (`yaml.dump` in `st.code`). Editable sections: ArXiv categories (`st.text_area`), LLM provider (`st.selectbox`), email enabled (`st.checkbox`). Save button writes to `config/config.yaml`. Test email button creates an `EmailSender` and sends a test message.

Used by: `gui/app.py` when page == "Settings".

### `gui/components/paper_card.py` — Reusable Paper Card Component
Renders a single paper in an `st.expander` with title, authors, categories, abstract, and arXiv link. Legacy component retained for potential reuse.

### `gui/components/table_helpers.py` — Table Formatting Utilities
Three helper functions for compact table display: `truncate_authors(authors, max_count=3)` truncates author lists with "et al.", `truncate_text(text, max_len=80)` truncates long text with "...", `get_primary_category(categories)` extracts the first category. Used by Papers page and Reports page.

### `gui/components/report_viewer.py` — Report Rendering Helper
Simple `render_report(report_markdown)` function that calls `st.markdown()`. Used by the Reports page.

### `tests/test_integration.py` — End-to-End Integration Tests
Verifies the full pipeline works when real components are wired together, with only external services mocked.

- **Test strategy**: Uses real `PaperStore` (temp SQLite DB), real `Embedder` (real sentence-transformers model producing real 384-dim embeddings), real `InterestManager`, real `LLMRanker`, real `ReportGenerator`, and real `EmailSender`. Only three things are mocked: `ArxivFetcher.fetch_today` (returns 10 synthetic ML papers), `LLMProvider` (concrete `MockLLMProvider` subclass with deterministic responses), and `smtplib.SMTP` (prevents real email sending).
- **`MockLLMProvider`**: A concrete `LLMProvider` subclass (not `MagicMock`) with call counters. `complete_json()` always returns `{"score": 7, "reason": "Relevant to user interests in machine learning"}`. `complete()` dispatches canned Markdown based on prompt keywords ("trending"/"emerging" → trending topics, "noteworthy"/"impactful" → highlight papers).
- **`_make_synthetic_papers(count)`**: Generates 10 ML-themed papers with carefully chosen titles and abstracts spanning transformers, reinforcement learning, vision, GNNs, federated learning, self-supervised learning, NAS, and code generation. Designed so embedding similarity tests can verify semantic relevance (e.g., transformer papers rank higher for transformer interests).
- **`TestEndToEndPipeline`** (6 tests): Full pipeline `run()` with DB state verification, no-interest path, duplicate handling, embedding relevance validation, email content integrity (MIME inspection), and interest-dependent ranking differences.
- **`TestComponentIntegration`** (4 tests): Targeted integration between pairs of components — store+embedder round-trip, interest manager+embedder embedding quality, ranker with real embedding-filtered candidates, report generator with real pipeline data.
- **Module-scoped `embedder` fixture**: Avoids reloading the ~80MB model per test. Same pattern as `test_embedder.py` and `test_interest_manager.py`.

Used by: `pytest tests/test_integration.py -v`. Part of the full test suite run.

### `tests/test_error_handling.py` — Error Handling & Hardening Tests (Phase 14)
Verifies that all external-service-facing components handle failures gracefully, and that project scaffolding files are correct.

- **TestArxivFetcherErrorHandling** (3 tests): Tests that `_fetch_category` catches `Exception` (including `ConnectionError`, `RuntimeError`) and returns an empty list for the failing category. If one category fails, papers from other categories are still returned. If all categories fail, returns an empty list without crashing.
- **TestLLMRankerErrorHandling** (2 tests): Tests that `_score_paper` catches `TimeoutError` and other exceptions, returning `llm_score=0` for the failed paper while other papers are scored normally. Tests that total failure of all LLM calls still returns all papers with score 0.
- **TestEmailSenderErrorHandling** (3 tests): Tests that `_send_smtp` catches `smtplib.SMTPException` subclasses (`SMTPAuthenticationError`, `SMTPConnectError`, generic `SMTPException`), logs the error, and re-raises so the pipeline can handle it.
- **TestFileVerification** (8 tests): Asserts `.env.example` exists and contains all 4 required env var keys. Asserts `.gitignore` contains `.env` and `data/`. Asserts `templates/email_template.md` exists with all 7 placeholder tokens and all 3 report sections.

Uses concrete `LLMProvider` subclasses (`FixedScoreLLM`, `TimeoutOnFirstLLM`, `AlwaysFailLLM`) for deterministic async behavior. Email tests use `monkeypatch.setenv` for SMTP credentials.

Used by: `pytest tests/test_error_handling.py -v`. Part of the full test suite run.

### `templates/email_template.md` — Email Reference Template (Phase 14)
A reference Markdown template showing the expected email report structure with placeholder tokens (`{date}`, `{total_count}`, `{category_breakdown}`, `{trending_topics}`, `{highlight_papers}`, `{specific_content}`, `{related_papers}`). Not used programmatically — the actual email content is built in code by `ReportGenerator.generate_general()` and `ReportGenerator.generate_specific()`. Serves as documentation for the email format.

---

## Key Design Decisions

| Decision | Approach | Rationale |
|----------|----------|-----------|
| Async LLM interface | All providers use `async def` | Enables concurrent re-ranking (asyncio.gather) |
| Factory pattern | `create_llm_provider()` with lazy imports | Avoids loading unused SDKs (openai vs anthropic) |
| JSON fence stripping | Regex strip of ````json ... ```` | Claude models wrap JSON in code blocks despite instructions |
| Claude CLI integration | subprocess via `asyncio.create_subprocess_exec` with `--output-format json` envelope | Zero cost, leverages existing subscription; structured envelope enables reliable output parsing |
| System message in CLI | `--system-prompt` CLI flag | Preserves semantic separation between system instructions and user content |
| CLI timeout | `asyncio.wait_for(timeout=self.timeout)` with process kill | Prevents hung CLI from blocking the pipeline indefinitely |
| CLI retry | Exponential backoff (1s, 2s, 4s) on `RuntimeError`, no retry on `ValueError` | Handles transient network/rate-limit failures; content errors are not retryable |
| CLI availability check | `shutil.which(cli_path)` in `__init__` | Fail-fast with clear error instead of cryptic subprocess failures at runtime |
| Session persistence | `--no-session-persistence` flag | Prevents session file accumulation during automated pipeline runs |
| Default provider | `claude_code` in `config.yaml` | Zero marginal LLM cost; other providers remain available via config switch |
| Provider-specific concurrency | `max_concurrent` in provider config, read by pipeline | Claude Code subscription has lower rate limits than API providers; default 2 for `claude_code`, 5 for others |
| ArXiv date filtering | Server-side `submittedDate` query + client-side safety net | Server-side filter scopes API results to the date range, avoiding the `max_results` truncation problem; client-side `published_date >= start_date` as belt-and-suspenders |
| ArXiv Client reuse | Single `arxiv.Client` created in `__init__`, reused across categories | Reduces object churn; library is designed for client reuse |
| ArXiv async wrapping | `loop.run_in_executor` around sync `arxiv.Client` | Avoids blocking the event loop; arxiv lib is synchronous; compatible with Python 3.8+ |
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
| Theme-based synthesis | `generate_specific` calls LLM for >= 5 papers to group into thematic clusters | Provides a cohesive narrative instead of paper-by-paper listing; helps readers understand the research landscape |
| Theme synthesis threshold | `_build_theme_synthesis` skips LLM for < 5 papers | Too few papers to form meaningful clusters; simple bullet list is clearer |
| Full authors/abstracts in Paper Details | No truncation in `_build_paper_details` | Comprehensive details for each paper; truncation was too limiting for research use |
| Graceful LLM failure in reports | `_build_trending_topics`/`_build_highlight_papers`/`_build_theme_synthesis` catch all exceptions | Report generation succeeds even if LLM is unavailable; fallback list replaces synthesis content |
| Primary category = first element | `categories[0]` used for overview breakdown | arXiv lists primary category first; consistent with how papers are categorized |
| Email CSS inlining | `premailer.transform()` after wrapping in HTML template | Most email clients strip `<style>` tags; inline styles ensure consistent rendering |
| SMTP in thread | `asyncio.to_thread(self._send_smtp, msg)` | `smtplib` is synchronous; wrapping in a thread avoids blocking the async event loop |
| Env-based credentials | Username/password read from env vars at init time via `os.environ.get` | Keeps secrets out of config files; follows 12-factor app principle |
| ar5iv content extraction priority | `<article>` → `ltx_document` → `ltx_page_main` → body | ar5iv pages use different structures; cascade handles all observed layouts |
| Text truncation at 15K chars | `full_text[:15000]` after joining paragraphs | Prevents exceeding LLM context limits; 15K chars ≈ 4K tokens, leaves room for prompt overhead |
| Cache-first summarization | Check `store.get_summary()` before calling LLM | Avoids redundant LLM calls for previously summarized papers; significant cost savings |
| Abstract fallback on fetch failure | Catch `RuntimeError` from `fetch_paper_text`, use abstract | Ensures summarization always works even if ar5iv is unavailable or returns errors |
| `_get_paper_by_id` via store's `_get_conn` | Direct SQL query through `store._get_conn()` | PaperStore only has `get_paper_by_arxiv_id`; avoids modifying the store API for a single consumer |
| 3-tier abstract auto-fetch | DB → arXiv API → fallback to ID | Better embedding quality for paper interests without burdening the user; each tier logged for observability |
| Match only new papers | `get_papers_by_ids_with_embeddings(new_paper_ids)` | Matches only papers newly inserted in the current run; eliminates overlap between consecutive daily reports. Previous approach used a date-range window which caused the same papers to appear in multiple reports. |
| No-interest early return | Skip matching, still generate general report | Useful for first-time users who haven't configured interests yet |
| Email failure resilience | `try/except` around `email_sender.send()` | Email failure should not prevent report saving; the run is still useful |
| Pipeline top-level imports | All imports at module level in `pipeline.py` | Components are always needed; avoids lazy-import complexity in the orchestrator |
| CLI lazy imports | Local imports inside `main()` in `src/main.py` | Avoids loading heavy dependencies (sentence-transformers, openai, anthropic) at import time; only loads what the chosen mode needs |
| BlockingScheduler | APScheduler `BlockingScheduler` for standalone scheduler mode | Simplest scheduler for a single-process daemon; no background thread complexity |
| Cron string parsing | Split 5-field string into `CronTrigger` kwargs | Standard cron format familiar to users; configurable in YAML |
| Lazy DailyPipeline import in scheduler | `_run_pipeline()` imports `DailyPipeline` inside the method | Avoids circular imports; creates a fresh pipeline instance per run for clean state |
| Separate CI/CD entry point | `scripts/run_pipeline.py` with `sys.path` manipulation | GitHub Actions can invoke it directly without `pip install -e .`; decoupled from the CLI's argparse |
| `@st.cache_resource` for config/store/embedder | Three cached singletons in `gui/app.py` | Avoids reloading config, recreating DB connections, and reloading the ~80MB model on every Streamlit rerun |
| Dynamic page imports | `from gui.pages.X import render` inside `if` branches | Only loads the selected page's module; avoids importing all pages (and their dependencies) upfront |
| `main()` guard in `gui/app.py` | `if __name__ == "__main__": main()` | Allows page modules to `from gui.app import get_embedder` without triggering the app's `main()` execution |
| GUI summarization on demand | Brief/Detailed buttons in Papers page | Summaries are expensive (LLM calls); only generated when user explicitly requests them |
| Cache-resource global scope | `autouse` fixture clears `st.cache_resource` in tests | Streamlit caches are process-global; without clearing, cached config/store from one test leaks into the next |
| AppTest timeout 30s | `_RUN_TIMEOUT = 30` for `at.run()` | First AppTest run imports sentence_transformers (→ torch), taking ~10s; default 3s is insufficient |
| Integration test real embeddings | Only mock external services (arXiv API, LLM APIs, SMTP) | Validates that the embedding pipeline produces semantically meaningful vectors end-to-end; would fail with random embeddings |
| Concrete MockLLMProvider in integration | Subclass of `LLMProvider` ABC, not `MagicMock` | Cleaner async behavior; call counters enable verifying LLM usage patterns; keyword dispatch enables testing both ranker and report generator |
| Synthetic papers for integration | 10 ML papers spanning transformers, RL, GNN, etc. | Enables semantic relevance assertions (transformer papers rank higher for transformer interests) without depending on real arXiv data |
| Three-level mocking in integration | `ArxivFetcher` (class), `create_llm_provider` (factory), `smtplib.SMTP` (stdlib) | Minimal mocking surface; real DB, real embeddings, real report formatting all exercised |
| Per-category error isolation | `_fetch_category` catches `Exception`, returns `[]` | One failing arXiv category (network, API error) doesn't block papers from other categories |
| SMTP error log-and-reraise | `_send_smtp` catches `SMTPException`, logs, re-raises | Pipeline's existing `try/except` around `email_sender.send()` handles it; error is observable in logs |
| Email template as reference doc | `templates/email_template.md` not used programmatically | Documents expected email format for developers; actual content built by `ReportGenerator` in code |
| Chinese report toggle | `config["report"]["chinese"]` boolean, defaults to `False` | Opt-in feature; avoids doubling LLM costs for users who don't need Chinese reports |
| Chinese summary as separate modes | `brief_zh`/`detailed_zh` stored as distinct `summary_type` values | Leverages existing `summaries` table schema; no schema changes needed; cached independently from English summaries |
| Chinese report DB columns | `general_report_zh`/`specific_report_zh` added to `daily_reports` | Stored alongside English reports for easy retrieval; nullable columns for backward compatibility |
| Schema migration via PRAGMA | `_migrate_add_column()` checks `PRAGMA table_info()` before `ALTER TABLE` | Safe idempotent migration; existing databases gain new columns without data loss; no migration framework dependency |
| Combined email for Chinese | `_combine_reports()` merges EN + ZH with `---` separators | Single email with both languages; avoids sending duplicate emails; users see all content in one place |
| Dynamic GUI tabs for Chinese | Tabs added only when `general_report_zh`/`specific_report_zh` exist | No UI clutter when Chinese reports are not generated; graceful degradation |
| Range label as run_date | Multi-day reports use `"start~end"` format (e.g., `"2026-02-20~2026-02-22"`) as `run_date` | Natural collision avoidance with daily reports; `matches` table's `UNIQUE(paper_id, run_date)` works transparently; `get_matches_by_date()` retrieves range matches correctly |
| report_type column with default | `TEXT DEFAULT 'daily'` migration | Backward compatible — existing daily reports automatically get `report_type = 'daily'`; no data migration needed |
| ID-based report lookup | Reports page uses `get_report_by_id()` instead of `get_report_by_date()` | Correctly handles multiple reports (daily + range) that may share overlapping dates; unambiguous selection |
| date_label parameter pattern | Optional `date_label` parameter on report generator methods | Backward compatible — `None` default preserves existing behavior; callers opt into period-aware wording |
| No email from range pipeline | `run_range_report()` does not send email | GUI handles email separately via "Send Report via Email" button; gives users control over when emails are sent |
| subject_override for email | Optional `subject_override` parameter on `send_sync()` and `send()` | Allows multi-day reports to customize subject lines without changing the default daily format |
| ORDER BY tiebreaker | `get_all_report_entries()` uses `ORDER BY created_at DESC, id DESC` | SQLite `CURRENT_TIMESTAMP` has second-level precision; `id DESC` tiebreaker ensures deterministic ordering for reports created in the same second |
