# Progress Tracker

## Completed Phases

### Phase 0: Project Scaffolding (Done)
- **Step 0.1** — Directory structure, `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`
- **Step 0.2** — `src/config.py` (get_project_root, load_config, get_env, setup_logging) + `config/config.yaml` + `tests/test_config.py`

### Phase 2: LLM Provider Abstraction (Done)
- **Step 2.1** — `LLMProvider` ABC in `src/llm/base.py` with `complete()` and `complete_json()` abstract methods. Factory function `create_llm_provider()` in `src/llm/__init__.py`.
- **Step 2.2** — `OpenAIProvider` in `src/llm/openai_provider.py`. Uses `openai.AsyncOpenAI`, supports `response_format=json_object` for structured output.
- **Step 2.3** — `ClaudeProvider` in `src/llm/claude_provider.py`. Uses `anthropic.AsyncAnthropic`, auto-strips markdown code fences from JSON responses.
- **Step 2.4** — `ClaudeCodeProvider` in `src/llm/claude_code_provider.py`. Calls `claude` CLI via `asyncio.create_subprocess_exec`, zero API cost via subscription.
- **Tests** — 4 test files (`test_llm_base.py`, `test_llm_openai.py`, `test_llm_claude.py`, `test_llm_claude_code.py`). All tests pass. All mocked — no real API calls.

### Phase 3: ArXiv Fetcher (Done)
- **Step 3.1** — `ArxivFetcher` in `src/fetcher/arxiv_fetcher.py`. Fetches papers from configurable arXiv categories via the `arxiv` Python library. Python-side date filtering (arXiv API sorts but doesn't filter by date). Version suffix stripping from arxiv_id (e.g., `v2`). Cross-category deduplication. Uses `asyncio.to_thread` to avoid blocking the event loop.
- **Tests** — `tests/test_fetcher.py` with 10 tests covering: date filtering, cutoff_days parameter, cross-category deduplication, version stripping, field extraction, and edge cases. All mocked — no real arXiv API calls.

### Phase 4: Embedding System (Done)
- **Step 4.1** — `Embedder` class in `src/matcher/embedder.py`. Lazy-loads `sentence-transformers` model (`all-MiniLM-L6-v2`, 384 dims). Methods: `embed_text()` (single string → 1D array), `embed_texts()` (batch → 2D array), `serialize_embedding()` / `deserialize_embedding()` (numpy ↔ bytes for SQLite BLOBs), `compute_embeddings()` (batch-embeds paper abstracts via store), `compute_interest_embeddings()` (embeds interest text with optional description via store).
- **Step 4.2** — `find_similar()` method: matrix cosine similarity (`papers @ interests.T`), takes MAX score per paper across all interests, filters by configurable threshold, returns top-N sorted descending with `embedding_score` field.
- **Tests** — `tests/test_embedder.py` with 17 tests covering: embedding shape/normalization, serialize/deserialize round-trip, compute methods with mocked store, lazy model loading, `find_similar` (top-N selection, descending sort, threshold filtering, empty inputs, field preservation, MAX-across-interests). Tests use the real model (module-scoped fixture to avoid reloading).

### Phase 1: Database Layer (Done)
- **Step 1.1** — `PaperStore` class in `src/store/database.py`. Schema initialization for 5 tables (papers, interests, matches, summaries, daily_reports). WAL mode enabled, foreign keys enforced. Idempotent `_init_db()` with `CREATE TABLE IF NOT EXISTS`. Each method opens and closes its own connection via `_get_conn()` (row_factory = sqlite3.Row for dict-like access).
- **Step 1.2** — Paper CRUD: `save_papers` (INSERT OR IGNORE for deduplication, returns only newly inserted papers with assigned IDs), `get_paper_by_arxiv_id`, `get_papers_by_date`, `search_papers` (LIKE on title/abstract), `update_paper_embedding`, `get_papers_without_embeddings`, `get_papers_with_embeddings`, `get_papers_by_date_with_embeddings`. Authors and categories stored as JSON strings, deserialized on read via `_row_to_paper()`.
- **Step 1.3** — Interest CRUD: `save_interest`, `get_all_interests`, `get_interest_by_id`, `update_interest` (partial updates — only non-None fields), `delete_interest`, `update_interest_embedding`, `get_interests_with_embeddings`.
- **Step 1.4** — Match CRUD: `save_match`, `get_matches_by_date` (JOIN with papers table, ORDER BY llm_score DESC with NULLS LAST, then embedding_score DESC). Summary CRUD: `save_summary`, `get_summary` (by paper_id + summary_type). Report CRUD: `save_report`, `get_report_by_date`, `get_all_report_dates` (sorted descending).
- **Tests** — `tests/test_store.py` with 30 tests across 6 test classes: TestSchemaInit (3 tests), TestPaperCRUD (14 tests), TestInterestCRUD (10 tests), TestMatchCRUD (3 tests), TestSummaryCRUD (3 tests), TestReportCRUD (4 tests). All use `tmp_path` fixture for isolated temp DBs. No mocks needed — tests the real SQLite layer.

### Phase 6: LLM Re-ranker (Done)
- **Step 6.1** — `LLMRanker` class in `src/matcher/ranker.py`. Concurrent LLM re-ranking of embedding-filtered candidates. Uses `asyncio.gather` with `asyncio.Semaphore(max_concurrent)` to limit parallel LLM calls (default 5). Each paper scored by the LLM on a 1-10 scale with a reason. Graceful error handling: scoring failures return `llm_score=0` without crashing the batch. `_format_interests()` builds a readable text block from interest dicts for the LLM prompt.
- **Tests** — `tests/test_ranker.py` with 13 tests across 4 test classes: TestLLMRankerBasic (7 tests — top_k truncation, LLM field presence, original field preservation, call count per candidate, top_k override, descending sort, empty candidates), TestLLMRankerFailure (2 tests — invalid JSON returns score 0, partial failure doesn't crash), TestLLMRankerConcurrency (2 tests — semaphore respects max_concurrent, default parallelism works), TestFormatInterests (3 tests — keywords with description, mixed types, empty list). All mocked — no real LLM API calls.

### Phase 7: Report Generator (Done)
- **Step 7.1** — `ReportGenerator.generate_general()` in `src/report/generator.py`. Produces a Markdown general report with three sections: (1) "Today's Overview" — total paper count + per-category breakdown computed in pure Python via `collections.Counter` on primary category, (2) "Trending Topics" — LLM identifies 3-5 emerging topics from paper titles, (3) "Highlight Papers" — LLM selects 3-5 noteworthy papers with reasons. Graceful error handling: LLM failures produce an error message instead of crashing.
- **Step 7.2** — `ReportGenerator.generate_specific()`. Formats pre-scored data from the ranker into Markdown — no LLM calls. Numbered list with `llm_score/10` and `llm_reason` for each paper, followed by a "Related Papers" section with authors, categories, abstract preview (first 200 chars), and arXiv links. Handles empty results, string-type authors/categories gracefully.
- **Tests** — `tests/test_report_generator.py` with 22 tests across 3 test classes: TestGenerateGeneral (9 tests — header with date, total count, category breakdown, trending/highlight sections, LLM call count, empty papers, section headers), TestGenerateSpecific (9 tests — header, paper titles, LLM scores, reasons, related papers, arXiv links, no LLM calls, empty results, authors/categories in related), TestEdgeCases (4 tests — string authors, string categories, LLM failure handling, single-category papers). All mocked — no real LLM API calls.

### Phase 8: Email Sender (Done)
- **Step 8.1** — `EmailSender` class in `src/email/sender.py`. SMTP email delivery with Markdown → HTML → CSS-inline pipeline. Reads SMTP config (host, port) from `config["email"]["smtp"]` and credentials from environment variables. `render_markdown_to_html()` converts Markdown to HTML via the `markdown` library (with `tables` + `fenced_code` extensions), wraps in an HTML template with a `<style>` block (body font, headings, tables, links, code), then inlines CSS via `premailer.transform()`. `send()` combines general + specific reports, renders to HTML, builds a `MIMEMultipart` message, and sends via `smtplib.SMTP` (starttls + login + send_message) wrapped in `asyncio.to_thread()` to avoid blocking the event loop.
- **Tests** — `tests/test_email_sender.py` with 22 tests across 4 test classes: TestInit (6 tests — SMTP settings, env credentials, addresses, subject prefix, custom prefix, multiple recipients), TestRenderMarkdownToHtml (7 tests — heading, bold, inline CSS, table, link, horizontal rule, bullet list), TestBuildEmail (6 tests — MIME type, subject/from/to headers, multiple recipients, HTML payload), TestSend (5 tests — SMTP starttls/login/send_message calls, correct subject, report content merging, no real email sent, custom host/port). All mocked — no real SMTP connections.

### Phase 9: Paper Summarizer (Done)
- **Step 9.1** — `PaperSummarizer` class in `src/summarizer/paper_summarizer.py`. Fetches full paper text from ar5iv HTML pages via `requests` + `BeautifulSoup` with `lxml` parser. Looks for `<article>` tag first, falls back to `ltx_document` → `ltx_page_main` → body. Extracts text from `<p>`, `<h2>`, `<h3>` tags, truncates to 15,000 characters for LLM context safety. Two summarization modes: "brief" (1-2 paragraphs on contributions + methodology) and "detailed" (structured: Motivation, Method, Experiments, Conclusions, Limitations). Cache-first: checks `store.get_summary()` before calling LLM. Falls back to abstract if ar5iv fetch fails. Caches results via `store.save_summary()` with LLM provider name. Includes `_get_paper_by_id()` helper for integer-id lookup (PaperStore only has `get_paper_by_arxiv_id`).
- **Tests** — `tests/test_summarizer.py` with 19 tests across 3 test classes: TestFetchPaperText (8 tests — `<article>` extraction, heading extraction, `ltx_document`/`ltx_page_main` fallbacks, HTTP error, connection error, 15K truncation, empty tag skipping), TestSummarize (9 tests — cache hit skips LLM, brief/detailed prompt construction, cache persistence with provider name, abstract fallback on fetch failure, nonexistent paper ValueError, title in prompt, system prompt, separate brief/detailed caching), TestGetPaperById (2 tests — found/not found). All use real `PaperStore` with temp DBs and concrete `MockLLMProvider` — no real API calls or HTTP requests.

### Phase 5: Interest Manager (Done)
- **Step 5.1** — `InterestManager` class in `src/interest/manager.py`. CRUD operations for three interest types (keyword, paper, reference_paper) with automatic embedding computation on add/update. Three-tier auto-fetch for paper abstracts: (1) check DB via `store.get_paper_by_arxiv_id`, (2) fetch from arXiv API via `arxiv.Search(id_list=...)`, (3) fall back to using arxiv_id as text with a warning. `recompute_all_embeddings()` for model changes. All methods delegate storage to `PaperStore` and embedding to `Embedder`.
- **Tests** — `tests/test_interest_manager.py` with 16 tests across 6 test classes: TestAddKeyword (4 tests — returns int ID, creates interest in store, computes 384-dim embedding, description support), TestAddPaper (5 tests — with description, computes embedding, auto-fetch from DB, auto-fetch from arXiv via mock, fallback to ID), TestAddReferencePaper (2 tests — with description, auto-fetch from DB), TestUpdateAndRemove (2 tests — update changes value+embedding, remove deletes), TestRecomputeAll (1 test), TestFetchAbstractFromArxiv (3 tests — success, no results, exception). Uses real `PaperStore` + real `Embedder` (module-scoped fixture); arXiv API calls mocked.

### Phase 10: Pipeline Orchestrator (Done)
- **Step 10.1** — `DailyPipeline` class in `src/pipeline.py`. Wires all 8 components together: `PaperStore`, `ArxivFetcher`, `Embedder`, `LLMProvider` (via factory), `LLMRanker`, `InterestManager`, `ReportGenerator`, `EmailSender`. The `run()` method executes the 12-step daily pipeline: fetch → save → embed → check interests → match today's papers → re-rank → save matches → generate general report → generate specific report → send email (if enabled) → save report → return summary dict. Two code paths: (a) with interests — full matching + ranking + both reports, (b) without interests — skip matching, generate general report only. Email failures are caught and logged without crashing the pipeline.
- **Tests** — `tests/test_pipeline.py` with 8 tests across 2 test classes: TestDailyPipelineFullRun (7 tests — full happy path verifying all component calls, no-interests skips matching, email disabled skips sending, email failure doesn't crash, save_match called per ranked paper, save_report called once with correct args, no-interests still saves general report), TestDailyPipelineInit (1 test — verifies all components instantiated). All components mocked — no real API calls, no real DB (except PaperStore which uses tmp_path).

### Phase 11: Scheduler and CLI Entry Points (Done)
- **Step 11.1** — `PipelineScheduler` class in `src/scheduler/scheduler.py`. Wraps APScheduler's `BlockingScheduler` with `CronTrigger`. Parses standard 5-field cron strings from `config["scheduler"]["cron"]`. `start()` adds the pipeline job and blocks. `_run_pipeline()` creates a `DailyPipeline` instance and runs it via `asyncio.run()`. Includes logging for scheduler start and pipeline completion.
- **Step 11.2** — CLI entry point in `src/main.py` and CI/CD script in `scripts/run_pipeline.py`. `src/main.py` uses `argparse` with `--mode` (`scheduler`|`run`, default `run`) and `--config` (optional path override). Calls `setup_logging()` and `load_config()` at startup, then dispatches to `PipelineScheduler.start()` or `asyncio.run(pipeline.run())`. `scripts/run_pipeline.py` is a standalone entry point for GitHub Actions that adds project root to `sys.path`, loads config, runs the pipeline, and logs a warning if no new papers were fetched.
- **Tests** — `tests/test_scheduler.py` with 9 tests across 3 test classes: TestPipelineSchedulerInit (2 tests — config storage, BlockingScheduler creation), TestPipelineSchedulerStart (5 tests — cron trigger creation, field parsing for various cron expressions, scheduler.start called), TestPipelineSchedulerRunPipeline (2 tests — pipeline creation+execution, config passing). `tests/test_main.py` with 7 tests across 3 test classes: TestMainRunMode (3 tests — pipeline execution, custom config path, default mode is run), TestMainSchedulerMode (2 tests — scheduler start, config passing), TestRunPipelineScript (2 tests — execution, no-papers warning). All mocked — no real scheduling, no real pipeline execution.

### Phase 12: Streamlit GUI (Done)
- **Step 12.1** — `gui/app.py` main Streamlit entry with sidebar radio navigation across 5 pages. Three `@st.cache_resource` functions (`get_config`, `get_store`, `get_embedder`) avoid reloading config, recreating DB connections, and reloading the ~80MB sentence-transformer model on every Streamlit rerun. `sys.path.insert` ensures project root is importable. `setup_logging()` called at module top level. `main()` guarded by `if __name__ == "__main__":` so `gui.app` can be imported by page modules (e.g., `interests.py` imports `get_embedder`) without triggering app execution.
- **Step 12.2** — `gui/pages/dashboard.py`. Three metrics row (Papers Today, Matches Today, total Reports count) via `st.metric`. General and specific report previews (first 1000 chars) from today's report. "Run Pipeline Now" button that creates a `DailyPipeline` instance and runs via `asyncio.run()` with `st.spinner` feedback and `st.rerun()` on completion.
- **Step 12.3** — `gui/pages/papers.py` + `gui/components/paper_card.py`. Date selector (`st.date_input`), search box (`st.text_input`). Search calls `store.search_papers()`; date mode calls `store.get_papers_by_date()`. Papers displayed in `st.expander` with authors, categories, abstract, arXiv link. Brief/Detailed summary buttons trigger `PaperSummarizer` on demand (cache-first via store). `paper_card.py` is a reusable component for rendering a single paper card.
- **Step 12.4** — `gui/pages/interests.py`. Lists current interests with type/value/description, embedding status (Y/N), and delete button per row. Delete triggers `store.delete_interest()` + `st.rerun()`. "Add New Interest" form with type selector (`keyword`|`paper`|`reference_paper`), value input, optional description. On submit, creates `InterestManager(store, get_embedder())` and calls the appropriate `add_*` method. The `get_embedder()` import from `gui.app` uses the cached Embedder instance.
- **Step 12.5** — `gui/pages/reports.py` + `gui/components/report_viewer.py`. Date dropdown from `store.get_all_report_dates()`. General/Specific reports displayed in `st.tabs`. Match results for the selected date shown in expanders with LLM score, embedding score, reason, and abstract preview. `report_viewer.py` provides a simple `render_report()` helper.
- **Step 12.6** — `gui/pages/settings.py`. Read-only config display (`yaml.dump` in `st.code`). Editable sections: ArXiv categories (`st.text_area`, one per line), LLM provider (`st.selectbox`), email enabled (`st.checkbox`). Save button writes to `config/config.yaml` via `get_project_root()`. Test email button creates an `EmailSender` and sends a test message.
- **Step 12.7** — `tests/test_gui.py` with 13 tests across 5 test classes using `AppTest.from_file("gui/app.py")`:
  - TestDashboardPage (3 tests — renders without errors, shows zero metrics on empty DB, has pipeline button)
  - TestPapersPage (3 tests — empty DB shows "0 papers", has date input + search box, displays papers when data exists)
  - TestInterestsPage (3 tests — empty DB shows "No interests", has form elements, displays interests when data exists)
  - TestReportsPage (2 tests — empty DB shows "No reports", displays reports when data exists)
  - TestSettingsPage (2 tests — renders without errors, has text_area + selectbox + checkbox controls)
  - Tests patch `src.config.load_config` to inject a test config with a temp DB path. An `autouse` fixture clears `st.cache_resource` before each test to prevent cache leakage between tests. `_RUN_TIMEOUT=30` handles the slow first-run import of sentence_transformers/torch.

### Phase 13: Integration Testing (Done)
- **Step 13.1** — `tests/test_integration.py` with 10 tests across 2 test classes. End-to-end integration tests using real components (PaperStore, Embedder, InterestManager, LLMRanker, ReportGenerator, EmailSender) with mocked external services (ArxivFetcher, LLM APIs, SMTP). Uses a concrete `MockLLMProvider` subclass (not `MagicMock`) that returns deterministic responses: `{"score": 7, "reason": "..."}` for `complete_json()` and keyword-dispatched canned Markdown for `complete()`.
  - **TestEndToEndPipeline** (6 tests):
    - `test_full_pipeline_with_real_components` — Main E2E test: creates temp DB, adds 2 keyword interests with real 384-dim embeddings, mocks ArxivFetcher (10 synthetic ML papers), mocks LLM (deterministic scores), mocks SMTP. Runs full `DailyPipeline.run()`. Verifies: 10 papers in DB with embeddings, matches saved with correct scores, report saved with both general/specific sections, SMTP starttls/login/send_message called, LLM used for both scoring and report generation.
    - `test_pipeline_with_no_interests` — No-interest early return path: general report generated and saved, specific report empty, matches = 0.
    - `test_pipeline_handles_duplicate_papers` — Runs pipeline twice with same papers; second run correctly reports 0 new papers, DB still has only 5.
    - `test_embedding_similarity_relevance` — Validates real embedding quality: transformer papers ranked higher than unrelated papers for a "transformer architectures" interest (at least 2 of 3 transformer papers in top 5).
    - `test_email_content_integrity` — Captures MIME message via `send_message` side_effect. Verifies subject, from/to headers, HTML payload with inlined CSS.
    - `test_interest_embedding_affects_matching` — Proves different interests (RL vs GNN) produce different top-3 rankings; RL papers ranked high for RL interest, GNN paper ranked high for GNN interest.
  - **TestComponentIntegration** (4 tests):
    - `test_store_embedder_round_trip` — Save → embed → retrieve cycle with real DB and normalized 384-dim embeddings.
    - `test_interest_manager_and_embedder_integration` — Different interest topics produce different embeddings (cosine similarity < 0.95).
    - `test_ranker_with_real_candidates` — LLMRanker scores real embedding-filtered candidates; verifies top_k truncation, field preservation, LLM call count.
    - `test_report_generator_with_real_data` — ReportGenerator produces valid Markdown with correct date headers, category counts, and formatted scores.
- **Step 13.2** — Full test suite execution: all 10 integration tests pass. Lint (`ruff check`) and format (`ruff format --check`) report zero issues.

### Phase 14: Polish and Hardening (Done)
- **Step 14.1** — Error handling added to critical locations:
  - `ArxivFetcher._fetch_category` (`src/fetcher/arxiv_fetcher.py`): wrapped arXiv API calls (`arxiv.Search`, `client.results`) in `try/except Exception`. On failure, logs the error and returns an empty list for that category — one failing category doesn't crash the whole fetch. Other categories still return their papers.
  - `LLMRanker._score_paper` (`src/matcher/ranker.py`): already had `try/except Exception` returning `llm_score=0` on failure. No changes needed (implemented correctly in Phase 6).
  - `EmailSender._send_smtp` (`src/email/sender.py`): wrapped SMTP operations (`starttls`, `login`, `send_message`) in `try/except smtplib.SMTPException`. Logs the error and re-raises so the pipeline's existing try/except can handle it. `SMTPAuthenticationError`, `SMTPConnectError`, and generic `SMTPException` are all caught.
  - `PaperSummarizer.fetch_paper_text` (`src/summarizer/paper_summarizer.py`): already caught `requests.RequestException` and raised `RuntimeError`. No changes needed (implemented correctly in Phase 9).
- **Step 14.2** — Template and file verification:
  - Created `templates/email_template.md` — reference Markdown template with all placeholders (`{date}`, `{total_count}`, `{category_breakdown}`, `{trending_topics}`, `{highlight_papers}`, `{specific_content}`, `{related_papers}`). Not used programmatically — the actual template is built in code by `ReportGenerator`.
  - Verified `.env.example` — contains all 4 required keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `EMAIL_USERNAME`, `EMAIL_PASSWORD`.
  - Verified `.gitignore` — excludes `.env` and `data/`.
- **Tests** — `tests/test_error_handling.py` with 16 tests across 4 test classes:
  - TestArxivFetcherErrorHandling (3 tests — ConnectionError on one category returns papers from others, all categories fail returns empty, generic Exception handled)
  - TestLLMRankerErrorHandling (2 tests — TimeoutError on one paper gives score 0 with others scored normally, all scoring failures returns all with zero)
  - TestEmailSenderErrorHandling (3 tests — SMTPAuthenticationError re-raised, SMTPConnectError re-raised, generic SMTPException re-raised)
  - TestFileVerification (8 tests — `.env.example` exists + has 4 keys, `.gitignore` has `.env` + `data/`, `templates/email_template.md` exists + has 7 placeholders + has 3 sections)
- **Full test suite**: 271 tests pass (255 existing + 16 new). Phase 14 files pass `ruff check` and `ruff format --check` with zero issues.

---

## Post-Phase Features

### Feature 1: Harden ClaudeCodeProvider as Primary LLM Provider (Done)

Based on `memory-bank/feature-implementation-1.md` (FR-1 through FR-8).

**Goal:** Make `ClaudeCodeProvider` production-ready and set it as the default LLM provider for zero marginal cost.

**Changes:**
- **`src/llm/claude_code_provider.py`** — Major rewrite (51 → 98 lines):
  - FR-1: System prompt via `--system-prompt` CLI flag (replaces prompt concatenation)
  - FR-2: Structured JSON output via `--output-format json` envelope parsing
  - FR-3: Configurable subprocess timeout (default 120s) via `asyncio.wait_for`; kills process on timeout
  - FR-4: Retry with exponential backoff (1s, 2s, 4s), up to `max_retries` (default 3) on `RuntimeError`; no retry on `ValueError`
  - FR-6: CLI availability check via `shutil.which()` in `__init__`; raises `RuntimeError` if not found
  - FR-8: `--no-session-persistence` flag to prevent session file accumulation
  - Method structure: `complete()` → `_run_cli()` (retry) → `_execute_subprocess()` (timeout + subprocess)
- **`config/config.yaml`** — FR-7: Default provider switched from `"openai"` to `"claude_code"`. Added 3 new settings: `timeout: 120`, `max_retries: 3`, `max_concurrent: 2`.
- **`src/pipeline.py`** — FR-5: Reads `max_concurrent` from the active LLM provider's config section (defaults to 5 for API providers, 2 for `claude_code`). Passes it to `ranker.rerank(max_concurrent=self.max_concurrent)`.
- **`tests/test_llm_claude_code.py`** — Expanded from 8 to 24 tests across 5 test classes:
  - TestClaudeCodeProviderInit (4 tests — defaults, custom values, CLI not found, CLI found)
  - TestClaudeCodeComplete (10 tests — envelope parsing, `--system-prompt` flag, `--output-format json`, `--no-session-persistence`, stdin prompt, nonzero exit, timeout kill, raw stdout fallback, CLI args order)
  - TestClaudeCodeRetry (5 tests — transient failure recovery, retry exhaustion, no retry on JSON parse error, backoff delays, single retry config)
  - TestClaudeCodeCompleteJSON (5 tests — envelope JSON parsing, markdown fence fallback, invalid JSON, system instruction injection, default system)
- **`memory-bank/architecture.md`** — Updated ClaudeCodeProvider section, pipeline section, config description, and Key Design Decisions table.

**Test results:** 24/24 new tests pass, all existing runnable tests unaffected. `ruff check` and `ruff format` clean.

### Feature 2: Theme-Based Synthesis Report (Done)

**Goal:** Replace paper-by-paper specific report with a theme-based synthesis narrative, and enhance Paper Details with comprehensive information.

**Changes:**
- **`config/config.yaml`** — Changed `llm_top_k` from 10 to 20 (more papers included in final recommendations).
- **`src/report/generator.py`** — Major rewrite of `generate_specific()` and addition of 4 new private methods:
  - `_build_theme_synthesis(ranked_papers, interests)` (async) — For >= 5 papers, sends paper titles, full abstracts, scores, and reasons to the LLM to group into 3-6 thematic clusters with `###` headings and flowing narrative paragraphs. For < 5 papers, delegates to `_build_simple_summary()`.
  - `_build_simple_summary(ranked_papers)` — Simple bullet list for < 5 papers (no LLM call).
  - `_build_fallback_list(ranked_papers)` — Numbered fallback list when LLM synthesis fails.
  - `_build_paper_details(ranked_papers)` — Comprehensive details: full authors (no truncation), full abstract (no truncation), relevance reason, score, categories, arXiv link.
  - Added `from __future__ import annotations` for Python 3.8 compatibility.
- **`templates/email_template.md`** — Updated placeholders: `{specific_content}` → `{theme_synthesis}`, `{related_papers}` → `{paper_details}`. Section name `## Related Papers` → `## Paper Details`.
- **`tests/test_report_generator.py`** — 29 tests (was 22):
  - Fixed pre-existing `_make_ranked_papers` bug (append inside loop).
  - Fixed pre-existing category breakdown assertions (table format).
  - Updated `MockLLMProvider` with synthesis keyword dispatch.
  - Replaced `test_no_llm_calls` with `test_no_llm_calls_for_few_papers` + `test_llm_called_for_synthesis`.
  - Added 6 new tests: theme synthesis present, full authors no truncation, full abstract no truncation, llm_reason in paper details, simple summary for few papers, LLM failure fallback.
- **`tests/test_integration.py`** — Added synthesis keyword dispatch to `MockLLMProvider.complete()`.
- **`tests/test_error_handling.py`** — Updated template placeholder and section assertions.
- **`memory-bank/architecture.md`** — Updated ReportGenerator section and Key Design Decisions table.

**Test results:** 29/29 report generator tests pass, 10/10 integration tests pass, 16/16 error handling tests pass. `ruff check` and `ruff format` clean on modified files.

### Feature 2b: GUI Reports Page — Two-Block Specific Report (Done)

**Goal:** Update the Streamlit Reports page to display the specific report in two blocks: (1) theme synthesis narrative, (2) expandable paper-wise cards with comprehensive details.

**Changes:**
- **`gui/views/reports.py`** — Rewrote the Specific Report tab:
  - Added `_split_specific_report(specific)` helper: splits the stored specific report markdown at the `---` divider to separate synthesis from paper details.
  - Added `_render_paper_cards(matches)` helper: renders expandable `st.expander` cards with full details (score, embedding score, categories, full authors, full abstract, relevance reason, arXiv link).
  - Block 1: Theme synthesis narrative rendered as markdown.
  - Block 2: Expandable paper cards from `store.get_matches_by_date()`.
  - Removed the standalone "Match Results" section that was at the bottom of the page (now integrated into the Specific Report tab).
- **`memory-bank/architecture.md`** — Updated Reports page section to describe two-block layout.

---

## All Phases Complete

The implementation plan (Phases 0–14) is fully implemented, plus Feature 1, Feature 2, and Feature 2b. The project is feature-complete with:
- 297 tests across 16 test files
- Full error handling at all external service boundaries
- `.env.example`, `.gitignore`, and email template verified
- `claude_code` as default LLM provider (zero marginal cost)
- Theme-based synthesis report with comprehensive Paper Details
- GUI Reports page with two-block specific report (synthesis + expandable cards)

## Notes for Future Developers
- Phase 2 was implemented before Phase 1 because it only depends on Phase 0 (no DB dependency).
- Phase 4 was implemented before Phase 1. The `compute_embeddings` and `compute_interest_embeddings` methods accept a `store` parameter (duck-typed), so they work without a concrete `PaperStore` — tests use `MagicMock`.
- Phase 1 was implemented 5th (after 0, 2, 3, 4) because earlier phases duck-typed the store dependency. Now that PaperStore exists, Phase 5 (InterestManager) can use the real store in its tests.
- All LLM providers use async interfaces. Tests mock the underlying SDK clients — no API keys needed to run tests.
- `ClaudeCodeProvider` sends the system message prepended to the user prompt via stdin (CLI doesn't have a separate system parameter).
- `ClaudeProvider` and `ClaudeCodeProvider` both strip markdown code fences (```json ... ```) from JSON responses since these models may wrap JSON in code blocks.
- The factory function uses lazy imports to avoid loading unused SDK dependencies.
- Embedder tests use a module-scoped fixture (`scope="module"`) so the ~80MB model is loaded only once across all test functions. The `find_similar` tests use synthetic normalized vectors via `_make_normalized_vector()` helper — no model loading needed for those.
- PaperSummarizer tests use real `PaperStore` (not mocks) because `_get_paper_by_id()` accesses `store._get_conn()` directly. A concrete `MockLLMProvider` class (with `call_count`, `last_prompt`, `last_system` tracking) is used instead of `MagicMock` for cleaner async behavior. HTTP requests are patched at the module level (`src.summarizer.paper_summarizer.requests.get`).
- PaperStore tests use `tmp_path` for isolated databases — no cleanup needed, no interference between tests. Each test class covers one CRUD domain (papers, interests, matches, summaries, reports).
- LLMRanker tests use concrete mock `LLMProvider` subclasses (not `MagicMock`) for cleaner async behavior. `MockLLMProviderConcurrency` uses an `asyncio.Lock` counter + `asyncio.sleep(0.05)` to verify the semaphore limits parallel execution. Tests use `max_concurrent=1` when deterministic ordering matters (e.g., the descending sort test with a `VaryingScoreLLM`).
- ReportGenerator tests also use concrete mock `LLMProvider` subclasses. The mock dispatches canned responses based on prompt keywords ("trending"/"emerging" vs "noteworthy"/"impactful" vs "thematic clusters"/"group these papers"). `generate_specific` with >= 5 papers calls LLM once for theme synthesis; with < 5 papers makes zero LLM calls.
- `src/main.py` uses local imports inside `main()` (lazy imports for `load_config`, `setup_logging`, `DailyPipeline`, `PipelineScheduler`). Tests must patch at the definition site (`src.config.setup_logging`, `src.config.load_config`, `src.pipeline.DailyPipeline`, `src.scheduler.scheduler.PipelineScheduler`) rather than at `src.main.*`, because local imports don't create module-level attributes that `unittest.mock.patch` can find.
- `scripts/run_pipeline.py` uses top-level imports, so its tests patch at `scripts.run_pipeline.*` directly.
- Integration tests use a module-scoped `embedder` fixture (same pattern as `test_embedder.py` and `test_interest_manager.py`) to avoid reloading the ~80MB model per test. The `MockLLMProvider` is a concrete subclass (not `MagicMock`) with call counters and keyword-dispatched responses — same pattern used by `test_summarizer.py` and `test_ranker.py`.
- Integration tests mock at three levels: `src.pipeline.ArxivFetcher` (class-level patch so `DailyPipeline.__init__` gets a mock), `src.pipeline.create_llm_provider` (returns `MockLLMProvider` instance), and `smtplib.SMTP` (prevents real SMTP connections). The `Embedder` is NOT mocked — real embeddings are used to verify semantic relevance.
- The `test_embedding_similarity_relevance` and `test_interest_embedding_affects_matching` tests validate that the embedding model actually produces semantically meaningful vectors — they would fail if embeddings were random. This is the only place in the test suite where embedding quality is verified end-to-end.
- Email env vars are set via `os.environ.setdefault()` (not `monkeypatch`) so they don't interfere with existing env. `EMAIL_USERNAME` and `EMAIL_PASSWORD` are only needed for tests that exercise `EmailSender.__init__`.
