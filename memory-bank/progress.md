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

## Next Up

### Phase 8: Email Sender (`src/email/sender.py`)
- Step 8.1: SMTP email with Markdown → HTML → CSS-inline pipeline

## Notes for Future Developers
- Phase 2 was implemented before Phase 1 because it only depends on Phase 0 (no DB dependency).
- Phase 4 was implemented before Phase 1. The `compute_embeddings` and `compute_interest_embeddings` methods accept a `store` parameter (duck-typed), so they work without a concrete `PaperStore` — tests use `MagicMock`.
- Phase 1 was implemented 5th (after 0, 2, 3, 4) because earlier phases duck-typed the store dependency. Now that PaperStore exists, Phase 5 (InterestManager) can use the real store in its tests.
- All LLM providers use async interfaces. Tests mock the underlying SDK clients — no API keys needed to run tests.
- `ClaudeCodeProvider` sends the system message prepended to the user prompt via stdin (CLI doesn't have a separate system parameter).
- `ClaudeProvider` and `ClaudeCodeProvider` both strip markdown code fences (```json ... ```) from JSON responses since these models may wrap JSON in code blocks.
- The factory function uses lazy imports to avoid loading unused SDK dependencies.
- Embedder tests use a module-scoped fixture (`scope="module"`) so the ~80MB model is loaded only once across all test functions. The `find_similar` tests use synthetic normalized vectors via `_make_normalized_vector()` helper — no model loading needed for those.
- PaperStore tests use `tmp_path` for isolated databases — no cleanup needed, no interference between tests. Each test class covers one CRUD domain (papers, interests, matches, summaries, reports).
- LLMRanker tests use concrete mock `LLMProvider` subclasses (not `MagicMock`) for cleaner async behavior. `MockLLMProviderConcurrency` uses an `asyncio.Lock` counter + `asyncio.sleep(0.05)` to verify the semaphore limits parallel execution. Tests use `max_concurrent=1` when deterministic ordering matters (e.g., the descending sort test with a `VaryingScoreLLM`).
- ReportGenerator tests also use concrete mock `LLMProvider` subclasses. The mock dispatches canned responses based on prompt keywords ("trending"/"emerging" vs "noteworthy"/"impactful"). `generate_specific` is tested to confirm it makes zero LLM calls — it only formats pre-scored data.
