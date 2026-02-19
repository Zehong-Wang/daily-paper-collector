# Feature PRD: Claude Code Subscription as Primary LLM Provider

**Status:** Proposed
**Author:** Auto-generated
**Date:** 2026-02-19
**Priority:** High

---

## 1. Problem Statement

The Daily Paper Collector relies on LLM calls in two critical pipeline stages: **LLM re-ranking** (scoring ~50 candidate papers concurrently) and **report generation** (trending topics + highlight papers). The current default provider is OpenAI (`gpt-4o-mini`), which incurs per-token API costs. The Claude API provider also requires a paid API key.

The project already includes a `ClaudeCodeProvider` that calls the `claude` CLI via subprocess, leveraging the Claude Code subscription (flat monthly fee, zero marginal cost per call). However, the current implementation is minimal — it lacks timeout handling, retry logic, proper system prompt separation, and structured output support. This makes it unsuitable as a reliable default for daily automated runs.

**Goal:** Harden `ClaudeCodeProvider` into a production-ready provider and make it the default, so the pipeline runs at zero marginal LLM cost.

---

## 2. Feature Requirements

### FR-1: Proper System Prompt Handling

**Current behavior:** System and user prompts are concatenated into a single string:
```python
full_prompt = f"{system}\n\n{prompt}" if system else prompt
```

**Required behavior:** Use the `--system-prompt` CLI flag to pass the system prompt as a dedicated parameter, preserving the semantic separation between system instructions and user content.

**Implementation:**
```python
cmd = [self.cli_path, "--print", "--model", self.model]
if system:
    cmd.extend(["--system-prompt", system])
```

**File:** `src/llm/claude_code_provider.py` — `complete()` method

---

### FR-2: Structured JSON Output via CLI Envelope

**Current behavior:** `complete_json()` appends "Respond with valid JSON only" to the system prompt and uses regex to strip markdown code fences before parsing.

**Required behavior:** Use `--output-format json` CLI flag, which returns a structured envelope:
```json
{
  "type": "result",
  "subtype": "success",
  "cost_usd": 0.003,
  "is_error": false,
  "result": "... the model's text response ...",
  "session_id": "...",
  "num_turns": 1
}
```

Extract `result` from the envelope for `complete()`, and parse the `result` field as JSON for `complete_json()`. This eliminates reliance on regex stripping.

**Implementation:**
- Always use `--output-format json` in the CLI command.
- In `complete()`: parse the envelope, return `envelope["result"]`.
- In `complete_json()`: parse `envelope["result"]` as JSON (still keep code-fence stripping as a fallback).

**File:** `src/llm/claude_code_provider.py` — both `complete()` and `complete_json()`

---

### FR-3: Configurable Subprocess Timeout

**Current behavior:** No timeout on `process.communicate()`. A hung CLI process blocks the pipeline indefinitely.

**Required behavior:** Add a configurable timeout (default: 120 seconds). If the subprocess exceeds the timeout, kill it and raise a descriptive error.

**Implementation:**
```python
try:
    stdout, stderr = await asyncio.wait_for(
        process.communicate(input=full_prompt.encode("utf-8")),
        timeout=self.timeout,
    )
except asyncio.TimeoutError:
    process.kill()
    raise RuntimeError(f"Claude CLI timed out after {self.timeout}s")
```

**Config addition:**
```yaml
claude_code:
  cli_path: "claude"
  model: "sonnet"
  timeout: 120          # seconds per CLI call
```

**File:** `src/llm/claude_code_provider.py` — `complete()` method

---

### FR-4: Retry with Exponential Backoff

**Current behavior:** No retry. A single CLI failure (transient network issue, rate limit) causes `RuntimeError` or `ValueError` immediately.

**Required behavior:** Retry failed calls up to N times (default: 3) with exponential backoff (1s, 2s, 4s). Only retry on transient errors (non-zero exit code). Do not retry on JSON parse failures (`ValueError`).

**Implementation:**
```python
async def _run_cli(self, prompt: str, system: str = "") -> str:
    """Execute CLI with retry logic. Returns raw stdout text."""
    for attempt in range(self.max_retries):
        try:
            return await self._execute_subprocess(prompt, system)
        except RuntimeError:
            if attempt == self.max_retries - 1:
                raise
            wait = 2 ** attempt
            self.logger.warning(f"CLI attempt {attempt+1} failed, retrying in {wait}s...")
            await asyncio.sleep(wait)
```

**Config addition:**
```yaml
claude_code:
  max_retries: 3        # retry attempts for transient failures
```

**File:** `src/llm/claude_code_provider.py`

---

### FR-5: Reduced Default Concurrency for Claude Code

**Current behavior:** `LLMRanker.rerank()` defaults to `max_concurrent=5`. This works for API providers with high rate limits, but may overwhelm the Claude Code subscription rate limit.

**Required behavior:** Add a `max_concurrent` config option under `claude_code` (default: 2). The pipeline should pass this to the ranker when using `claude_code` provider.

**Option A — Config-driven (recommended):**
```yaml
claude_code:
  max_concurrent: 2     # concurrent CLI calls for re-ranking
```

The `DailyPipeline` reads `config["llm"]["claude_code"]["max_concurrent"]` and passes it to `ranker.rerank(max_concurrent=...)`.

**Option B — Provider-aware ranker:**
The ranker checks the provider type and adjusts concurrency automatically. (Less explicit, not recommended.)

**Files:**
- `config/config.yaml` — add `max_concurrent` to `claude_code` section
- `src/pipeline.py` — pass `max_concurrent` to `ranker.rerank()`

---

### FR-6: CLI Availability Check on Init

**Current behavior:** No validation that the `claude` CLI exists. Failures only surface at runtime when the first `complete()` call is made.

**Required behavior:** On `ClaudeCodeProvider.__init__()`, run `claude --version` (or `which claude`) to verify the CLI is available. Log the CLI version on success. Raise a clear `RuntimeError` if the CLI is not found.

**Implementation:**
```python
import shutil

def __init__(self, config: dict):
    self.cli_path = config.get("cli_path", "claude")
    if not shutil.which(self.cli_path):
        raise RuntimeError(
            f"Claude CLI not found at '{self.cli_path}'. "
            "Install it or set 'cli_path' in config."
        )
    self.model = config.get("model", "sonnet")
```

**File:** `src/llm/claude_code_provider.py` — `__init__()`

---

### FR-7: Switch Default Provider to `claude_code`

**Current behavior:** `config.yaml` sets `provider: "openai"` as the default.

**Required behavior:** Change the default to `claude_code` so out-of-the-box usage incurs no API costs.

**Change:**
```yaml
llm:
  provider: "claude_code"    # changed from "openai"
```

**File:** `config/config.yaml`

---

### FR-8: Disable Session Persistence

**Current behavior:** Each CLI call may create session files on disk, accumulating over time.

**Required behavior:** Pass `--no-session-persistence` flag to prevent session file accumulation during automated pipeline runs.

**Implementation:** Add the flag to the command array:
```python
cmd = [self.cli_path, "--print", "--model", self.model,
       "--output-format", "json", "--no-session-persistence"]
```

**File:** `src/llm/claude_code_provider.py` — `complete()` method

---

## 3. Configuration Changes

### Updated `config.yaml` Schema

```yaml
llm:
  provider: "claude_code"              # default provider (was "openai")
  openai:
    model: "gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"
  claude:
    model: "claude-sonnet-4-5-20250929"
    api_key_env: "ANTHROPIC_API_KEY"
  claude_code:
    cli_path: "claude"                 # path to claude CLI binary
    model: "sonnet"                    # claude model shortname
    timeout: 120                       # NEW: seconds per CLI call
    max_retries: 3                     # NEW: retry attempts for transient failures
    max_concurrent: 2                  # NEW: concurrent calls for re-ranking
```

---

## 4. Code Changes Summary

| File | Change | Scope |
|------|--------|-------|
| `src/llm/claude_code_provider.py` | Major rewrite: add `--system-prompt`, `--output-format json`, `--no-session-persistence`, timeout, retry, CLI check | ~100 lines (from 51) |
| `config/config.yaml` | Switch default provider, add timeout/retry/concurrency settings | 3 new lines, 1 changed |
| `src/pipeline.py` | Pass `max_concurrent` from config to `ranker.rerank()` | ~3 lines changed |
| `tests/test_llm_claude_code.py` | Expand tests for new flags, timeout, retry, envelope parsing, CLI check | ~25 tests (from ~8) |

---

## 5. Testing Strategy

### Unit Tests (`tests/test_llm_claude_code.py`)

1. **CLI command construction**: Verify `--system-prompt`, `--output-format json`, `--no-session-persistence` flags appear in command
2. **System prompt**: Verify `--system-prompt` is omitted when `system=""` and present when `system="..."`
3. **Envelope parsing**: Mock stdout with JSON envelope, verify `complete()` returns `envelope["result"]`
4. **JSON extraction from envelope**: Mock `envelope["result"]` containing valid JSON string, verify `complete_json()` parses it correctly
5. **Timeout**: Mock `asyncio.wait_for` to raise `TimeoutError`, verify `RuntimeError` with descriptive message
6. **Retry on transient failure**: Mock subprocess to fail twice then succeed, verify 3 attempts made with backoff
7. **Retry exhaustion**: Mock subprocess to fail all attempts, verify final `RuntimeError` raised
8. **No retry on JSON parse error**: Mock valid subprocess output with invalid JSON, verify `ValueError` raised immediately (no retry)
9. **CLI availability check**: Mock `shutil.which()` returning `None`, verify `RuntimeError` on init
10. **CLI availability success**: Mock `shutil.which()` returning path, verify init completes

### Integration Tests

1. **End-to-end with real CLI** (manual/CI): Run a simple `complete("Say hello")` call and verify non-empty string response
2. **Pipeline dry run**: Run `DailyPipeline` with `claude_code` provider against a small set of test papers

---

## 6. Rollback Plan

All other providers (`openai`, `claude`) remain fully functional. To rollback:

1. Change `config.yaml`: `provider: "openai"` (one-line change)
2. No code changes needed — the factory pattern dispatches based on config

The `ClaudeCodeProvider` enhancements are purely additive and backward-compatible. The `--output-format json` envelope parsing gracefully handles the case where the CLI returns plain text (fallback to raw stdout).

---

## 7. Dependencies

- **No new Python packages** — uses only `asyncio`, `json`, `shutil`, `re`, `logging` (all stdlib)
- **External dependency**: `claude` CLI must be installed and authenticated on the host machine
- Claude Code subscription (Max or Pro plan) must be active

---

## 8. Out of Scope

- Migrating existing stored data (summaries tagged with provider name) — no schema changes needed
- Adding new LLM provider types (e.g., Gemini, local models)
- Changing the LLMProvider ABC interface — the two-method contract (`complete`, `complete_json`) remains unchanged
- GUI changes — the Streamlit GUI already uses `create_llm_provider(config)` and will automatically pick up the new default
