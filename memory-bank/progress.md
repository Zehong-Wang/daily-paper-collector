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

## Next Up

### Phase 1: Database Layer (`src/store/database.py`)
- Steps 1.1–1.4: PaperStore class with schema init, Paper CRUD, Interest CRUD, Match/Summary/Report CRUD

### Phase 5: Interest Manager (`src/interest/manager.py`)
- Step 5.1: InterestManager with auto-fetch abstracts from DB/arXiv + embedding recomputation

## Notes for Future Developers
- Phase 2 was implemented before Phase 1 because it only depends on Phase 0 (no DB dependency).
- Phase 4 was implemented before Phase 1. The `compute_embeddings` and `compute_interest_embeddings` methods accept a `store` parameter (duck-typed), so they work without a concrete `PaperStore` — tests use `MagicMock`.
- All LLM providers use async interfaces. Tests mock the underlying SDK clients — no API keys needed to run tests.
- `ClaudeCodeProvider` sends the system message prepended to the user prompt via stdin (CLI doesn't have a separate system parameter).
- `ClaudeProvider` and `ClaudeCodeProvider` both strip markdown code fences (```json ... ```) from JSON responses since these models may wrap JSON in code blocks.
- The factory function uses lazy imports to avoid loading unused SDK dependencies.
- Embedder tests use a module-scoped fixture (`scope="module"`) so the ~80MB model is loaded only once across all test functions. The `find_similar` tests use synthetic normalized vectors via `_make_normalized_vector()` helper — no model loading needed for those.
