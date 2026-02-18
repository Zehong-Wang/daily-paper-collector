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

## Next Up

### Phase 1: Database Layer (`src/store/database.py`)
- Steps 1.1–1.4: PaperStore class with schema init, Paper CRUD, Interest CRUD, Match/Summary/Report CRUD

### Phase 3: ArXiv Fetcher
- Step 3.1: ArxivFetcher with Python-side date filtering

## Notes for Future Developers
- Phase 2 was implemented before Phase 1 because it only depends on Phase 0 (no DB dependency).
- All LLM providers use async interfaces. Tests mock the underlying SDK clients — no API keys needed to run tests.
- `ClaudeCodeProvider` sends the system message prepended to the user prompt via stdin (CLI doesn't have a separate system parameter).
- `ClaudeProvider` and `ClaudeCodeProvider` both strip markdown code fences (```json ... ```) from JSON responses since these models may wrap JSON in code blocks.
- The factory function uses lazy imports to avoid loading unused SDK dependencies.
