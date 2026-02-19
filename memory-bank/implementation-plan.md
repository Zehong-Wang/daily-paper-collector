# Implementation Plan — Daily Paper Collector

This plan builds the project bottom-up: foundational utilities first, then individual components, then the orchestration layer, and finally the GUI. Every step includes concrete testing criteria.

---

## Phase 0: Project Scaffolding

### Step 0.1 — Create directory structure and package files

Create every directory and `__init__.py` listed in the design doc:

```
daily-paper-collector/
├── config/
├── src/
│   ├── __init__.py
│   ├── fetcher/__init__.py
│   ├── store/__init__.py
│   ├── matcher/__init__.py
│   ├── interest/__init__.py
│   ├── report/__init__.py
│   ├── email/__init__.py
│   ├── summarizer/__init__.py
│   ├── llm/__init__.py
│   └── scheduler/__init__.py
├── gui/
│   ├── pages/
│   └── components/
├── templates/
├── data/
├── scripts/
└── tests/
```

Create `pyproject.toml` with the following content:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "daily-paper-collector"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

Create `requirements.txt` with the exact contents from the tech-stack doc (core + dev dependencies). Create `.env.example` with placeholder keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

Add a `.gitignore` that ignores `data/`, `.env`, `__pycache__/`, `*.egg-info/`, `.ruff_cache/`.

**Test:**
- Run `pip install -e .` — must succeed with zero errors.
- Run `python -c "import src"` — must succeed.
- Run `pytest` — must exit 0 (no tests collected is OK).
- Run `ruff check .` — must report no issues.

---

### Step 0.2 — Create the YAML config file, config loader, path resolution, and logging

Create `config/config.yaml` with the exact YAML from design doc Section 6 (arxiv categories, matching settings, llm settings, email settings, scheduler, database path, GUI port).

Create `src/config.py` with the following functions:

```python
from pathlib import Path
import logging
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def get_project_root() -> Path:
    """Walk up from this file's directory to find the project root
    (identified by the presence of pyproject.toml).
    Returns the absolute Path to the project root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")

def load_config(path: str = None) -> dict:
    """Load and return the YAML config dict.
    If path is None, defaults to config/config.yaml relative to project root.
    Resolves the database path to an absolute path relative to project root."""
    root = get_project_root()
    if path is None:
        config_path = root / "config" / "config.yaml"
    else:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = root / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Resolve database path relative to project root
    db_path = config.get("database", {}).get("path", "data/papers.db")
    if not Path(db_path).is_absolute():
        config["database"]["path"] = str(root / db_path)
    return config

def get_env(key: str) -> str:
    """Read from os.environ. Raises ValueError if the key is missing."""
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value

def setup_logging(level: str = "INFO"):
    """Configure project-wide logging. Call once at startup."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
```

All subsequent components should use `self.logger = logging.getLogger(__name__)` in their `__init__` from the start. `setup_logging()` is called in `src/main.py` and `gui/app.py` at startup.

**Test:** `tests/test_config.py`
- Write a test that `get_project_root()` returns a path containing `pyproject.toml`.
- Write a test that creates a temp YAML file with known values, calls `load_config(temp_path)`, and asserts the returned dict contains the expected keys and values.
- Write a test that verifies the database path is absolute after loading config.
- Write a test that verifies `get_env("NONEXISTENT_KEY_ABC")` raises `ValueError`.
- Write a test that sets `os.environ["TEST_KEY_XYZ"] = "hello"`, calls `get_env("TEST_KEY_XYZ")`, and asserts the result is `"hello"`. Clean up the env var after.
- Write a test that `setup_logging()` can be called without error and sets the root logger level.

---

## Phase 1: Database Layer (`src/store/database.py`)

### Step 1.1 — Create PaperStore class with schema initialization

Create `src/store/database.py` containing class `PaperStore`:

```python
class PaperStore:
    def __init__(self, db_path: str):
        # Store the path, call self._init_db()
        ...

    def _init_db(self):
        # Connect to SQLite, execute CREATE TABLE IF NOT EXISTS for all 5 tables
        # (papers, interests, matches, summaries, daily_reports).
        # Use the exact schema from design doc Section 5.
        # Enable WAL mode: PRAGMA journal_mode=WAL
        # Enable foreign keys: PRAGMA foreign_keys=ON
        ...

    def _get_conn(self) -> sqlite3.Connection:
        # Return a new connection (sqlite3.connect) each time.
        # Set row_factory = sqlite3.Row for dict-like access.
        ...
```

**Test:** `tests/test_store.py`
- Use `tmp_path` fixture to create a temp DB. Instantiate `PaperStore(str(tmp_path / "test.db"))`.
- Assert the DB file exists on disk.
- Query `sqlite_master` to verify all 5 tables (`papers`, `interests`, `matches`, `summaries`, `daily_reports`) exist.
- Instantiate `PaperStore` a second time with the same path — verify it does not error (idempotent).

---

### Step 1.2 — Paper CRUD methods

Add these methods to `PaperStore`:

```python
def save_papers(self, papers: list[dict]) -> list[dict]:
    """Insert papers, skip duplicates by arxiv_id. Return only newly inserted papers.
    Each dict has keys: arxiv_id, title, authors (list), abstract, categories (list),
    published_date (date), pdf_url, ar5iv_url.
    authors and categories are JSON-serialized before storage."""

def get_paper_by_arxiv_id(self, arxiv_id: str) -> dict | None:
    """Return a single paper dict or None. Deserialize authors/categories from JSON."""

def get_papers_by_date(self, date: str) -> list[dict]:
    """Return all papers with published_date == date. date format: 'YYYY-MM-DD'."""

def search_papers(self, query: str, limit: int = 50) -> list[dict]:
    """Search papers where title or abstract LIKE '%query%'. Return up to limit results."""

def update_paper_embedding(self, paper_id: int, embedding: bytes):
    """Update the embedding BLOB for a given paper id."""

def get_papers_without_embeddings(self) -> list[dict]:
    """Return all papers where embedding IS NULL."""

def get_papers_with_embeddings(self) -> list[dict]:
    """Return all papers where embedding IS NOT NULL. Include the embedding bytes."""

def get_papers_by_date_with_embeddings(self, date: str) -> list[dict]:
    """Return papers for a given date (published_date == date) that have
    embeddings computed (embedding IS NOT NULL). Include the embedding bytes.
    Used by the pipeline to match only today's papers against interests."""
```

For `save_papers`, use `INSERT OR IGNORE` to handle duplicates. After insert, select back the rows by arxiv_id to get their assigned `id` values. Return only papers that were newly inserted (compare count before and after, or check `cursor.rowcount`).

**Test:** `tests/test_store.py` (extend)
- Insert 3 papers via `save_papers`. Assert return list has length 3.
- Insert the same 3 papers again. Assert return list has length 0 (all duplicates skipped).
- Call `get_paper_by_arxiv_id` with a known id — assert title matches.
- Call `get_paper_by_arxiv_id` with a non-existent id — assert returns `None`.
- Call `get_papers_by_date` with the insertion date — assert returns 3 papers.
- Call `search_papers("some keyword from the test abstract")` — assert at least 1 result.
- Call `update_paper_embedding(id, b"fake_blob")` then `get_papers_with_embeddings()` — assert 1 paper returned with correct blob.
- Call `get_papers_without_embeddings()` — assert 2 papers returned (the ones not updated).
- Call `get_papers_by_date_with_embeddings(today)` — assert 1 paper returned (the one with embedding and matching date). Insert a paper with a different date and embedding — assert it is NOT returned for today's date.

---

### Step 1.3 — Interest CRUD methods

Add these methods to `PaperStore`:

```python
def save_interest(self, type: str, value: str, description: str = None) -> int:
    """Insert an interest. type is 'keyword'|'paper'|'reference_paper'. Return the new id."""

def get_all_interests(self) -> list[dict]:
    """Return all interests."""

def get_interest_by_id(self, interest_id: int) -> dict | None:
    """Return a single interest or None."""

def update_interest(self, interest_id: int, value: str = None, description: str = None):
    """Update fields of an interest. Only update provided (non-None) fields."""

def delete_interest(self, interest_id: int):
    """Delete an interest by id."""

def update_interest_embedding(self, interest_id: int, embedding: bytes):
    """Update the embedding BLOB for a given interest id."""

def get_interests_with_embeddings(self) -> list[dict]:
    """Return interests where embedding IS NOT NULL."""
```

**Test:** `tests/test_store.py` (extend)
- Insert 2 interests (one keyword, one paper). Assert IDs returned are > 0.
- `get_all_interests()` returns 2 items.
- `get_interest_by_id` with valid id returns correct dict.
- `update_interest(id, value="new value")` then `get_interest_by_id` shows updated value.
- `delete_interest(id)` then `get_all_interests()` returns 1 item.
- `update_interest_embedding(id, b"blob")` then `get_interests_with_embeddings()` returns 1 item.

---

### Step 1.4 — Match, Summary, and Report CRUD methods

Add these methods to `PaperStore`:

```python
# --- Matches ---
def save_match(self, paper_id: int, run_date: str, embedding_score: float,
               llm_score: float = None, llm_reason: str = None) -> int:
    """Insert a match record. Return the new id."""

def get_matches_by_date(self, run_date: str) -> list[dict]:
    """Return all matches for a given run_date, joined with paper info (title, arxiv_id, abstract).
    Order by llm_score DESC (nulls last), then embedding_score DESC."""

# --- Summaries ---
def save_summary(self, paper_id: int, summary_type: str, content: str,
                 llm_provider: str = None) -> int:
    """Insert a summary. summary_type is 'brief'|'detailed'. Return new id."""

def get_summary(self, paper_id: int, summary_type: str) -> dict | None:
    """Return cached summary for a paper+type, or None if not cached."""

# --- Reports ---
def save_report(self, run_date: str, general_report: str, specific_report: str,
                paper_count: int, matched_count: int) -> int:
    """Insert a daily report record. Return new id."""

def get_report_by_date(self, run_date: str) -> dict | None:
    """Return report for a date, or None."""

def get_all_report_dates(self) -> list[str]:
    """Return all run_dates that have reports, sorted descending."""
```

For `get_matches_by_date`, use a JOIN: `SELECT m.*, p.title, p.arxiv_id, p.abstract, p.authors, p.categories, p.pdf_url FROM matches m JOIN papers p ON m.paper_id = p.id WHERE m.run_date = ? ORDER BY ...`.

**Test:** `tests/test_store.py` (extend)
- Insert a paper, then save a match for it. `get_matches_by_date` returns 1 result with paper info joined.
- Save a summary for a paper. `get_summary(paper_id, "brief")` returns it. `get_summary(paper_id, "detailed")` returns `None`.
- Save a report. `get_report_by_date` returns it. `get_all_report_dates` returns 1 date string.

---

## Phase 2: LLM Provider Abstraction (`src/llm/`)

### Step 2.1 — Define the LLMProvider ABC

Create `src/llm/base.py`:

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the text response."""
        ...

    @abstractmethod
    async def complete_json(self, prompt: str, system: str = "") -> dict:
        """Send a prompt and parse the response as JSON.
        Implementations should instruct the model to return valid JSON
        and parse the result with json.loads().
        Raise ValueError if the response is not valid JSON."""
        ...
```

Create a factory function in `src/llm/__init__.py`:

```python
def create_llm_provider(config: dict) -> LLMProvider:
    """Create an LLM provider based on config['llm']['provider'].
    Valid values: 'openai', 'claude', 'claude_code'.
    Raises ValueError for unknown provider."""
```

This function reads `config["llm"]["provider"]` and instantiates the corresponding class, passing the relevant sub-config (e.g., `config["llm"]["openai"]`).

**Test:** `tests/test_llm_base.py`
- Verify that `LLMProvider` cannot be instantiated directly (it's abstract).
- Create a mock subclass that implements both methods (just returns hardcoded strings). Instantiate it and call both methods — assert correct return values.
- Test the factory: pass a config with `provider: "openai"` — assert it returns an `OpenAIProvider` instance (after Step 2.2). Pass `provider: "unknown"` — assert `ValueError`.

---

### Step 2.2 — Implement OpenAIProvider

Create `src/llm/openai_provider.py`:

```python
class OpenAIProvider(LLMProvider):
    def __init__(self, config: dict):
        # config is the config["llm"]["openai"] sub-dict.
        # Read the API key from os.environ using config["api_key_env"].
        # Create an openai.AsyncOpenAI client.
        # Store config["model"] (e.g., "gpt-4o-mini").

    async def complete(self, prompt: str, system: str = "") -> str:
        # Build messages list: [{"role":"system","content":system}, {"role":"user","content":prompt}]
        # If system is empty, omit the system message.
        # Call self.client.chat.completions.create(model=self.model, messages=messages)
        # Return response.choices[0].message.content

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        # Same as complete, but append "Respond with valid JSON only." to system.
        # Set response_format={"type": "json_object"} in the API call.
        # Parse result with json.loads(). Raise ValueError on failure.
```

**Test:** `tests/test_llm_openai.py`
- Use `unittest.mock.patch` to mock `openai.AsyncOpenAI`. Instantiate `OpenAIProvider` with a fake config. Call `complete("test prompt")`. Assert the mocked client's `chat.completions.create` was called with the correct model and messages. Assert the return value matches the mocked response.
- Test `complete_json` returns a parsed dict when the mocked response is valid JSON.
- Test `complete_json` raises `ValueError` when the mocked response is `"not json"`.

---

### Step 2.3 — Implement ClaudeProvider

Create `src/llm/claude_provider.py`:

```python
class ClaudeProvider(LLMProvider):
    def __init__(self, config: dict):
        # config is config["llm"]["claude"] sub-dict.
        # Read API key from os.environ using config["api_key_env"].
        # Create an anthropic.AsyncAnthropic client.
        # Store config["model"].

    async def complete(self, prompt: str, system: str = "") -> str:
        # Call self.client.messages.create(
        #     model=self.model,
        #     max_tokens=4096,
        #     system=system if system else anthropic.NOT_GIVEN,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # Return response.content[0].text

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        # Append "Respond with valid JSON only." to system.
        # Call complete() with the modified system.
        # Extract JSON from the response (strip markdown code fences if present).
        # Parse with json.loads(). Raise ValueError on failure.
```

**Test:** `tests/test_llm_claude.py`
- Mirror the same mocking approach as Step 2.2 but for `anthropic.AsyncAnthropic`.
- Test `complete` calls `messages.create` correctly.
- Test `complete_json` parses valid JSON. Test it raises `ValueError` for invalid JSON.

---

### Step 2.4 — Implement ClaudeCodeProvider

Create `src/llm/claude_code_provider.py`:

```python
import asyncio
import subprocess
import json

class ClaudeCodeProvider(LLMProvider):
    def __init__(self, config: dict):
        # config is config["llm"]["claude_code"] sub-dict.
        # Store config["cli_path"] (default "claude").
        # Store config.get("model", "sonnet").

    async def complete(self, prompt: str, system: str = "") -> str:
        # Build command: [self.cli_path, "--print", "--model", self.model]
        # If system is non-empty, prepend it to prompt: f"{system}\n\n{prompt}"
        # Run via asyncio.create_subprocess_exec with stdin=PIPE, stdout=PIPE, stderr=PIPE
        # Write prompt to stdin, await process.communicate()
        # If returncode != 0, raise RuntimeError with stderr.
        # Return stdout decoded as UTF-8, stripped.

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        # Append "Respond with valid JSON only. No markdown formatting." to system.
        # Call complete() with modified system.
        # Strip markdown code fences if present (regex: ```json\n...\n```).
        # json.loads(). Raise ValueError on failure.
```

**Test:** `tests/test_llm_claude_code.py`
- Use `unittest.mock.patch("asyncio.create_subprocess_exec")` to mock the subprocess.
- Test `complete` passes correct args and returns stdout.
- Test `complete_json` parses valid JSON from mocked stdout.
- Test `complete` raises `RuntimeError` when returncode is non-zero.

---

## Phase 3: ArXiv Fetcher (`src/fetcher/arxiv_fetcher.py`)

### Step 3.1 — Implement ArxivFetcher

Create `src/fetcher/arxiv_fetcher.py`:

```python
import arxiv
from datetime import date, timedelta

class ArxivFetcher:
    def __init__(self, config: dict):
        # Store config["arxiv"]["categories"] (list of strings).
        # Store config["arxiv"]["max_results_per_category"] (int).

    async def fetch_today(self, cutoff_days: int = 2) -> list[dict]:
        """Fetch papers from all configured categories.
        For each category, query arXiv sorted by SubmittedDate, then filter in Python
        to only keep papers published within the last `cutoff_days` days
        (default 2, to account for timezone and indexing delays).
        Return a list of dicts with keys: arxiv_id, title, authors, abstract,
        categories, published_date, pdf_url, ar5iv_url.
        Deduplicate across categories by arxiv_id (a paper can appear in multiple categories).
        The cutoff_days parameter exists for testability."""

    def _fetch_category(self, category: str, cutoff_date: date) -> list[dict]:
        """Fetch papers for a single category.
        Use arxiv.Client() and arxiv.Search(query=f"cat:{category}",
            max_results=self.max_results, sort_by=arxiv.SortCriterion.SubmittedDate).

        **Important:** The arXiv Search API sorts by SubmittedDate but does NOT filter by date.
        After fetching results, filter in Python to only keep papers where
        `result.published.date() >= cutoff_date`.
        Log the count before and after date filtering for observability.

        For each result that passes the date filter, extract:
          - arxiv_id: result.entry_id.split('/')[-1]  (e.g., "2501.12345v1" → strip version → "2501.12345")
          - title: result.title (strip newlines)
          - authors: [a.name for a in result.authors]
          - abstract: result.summary (strip newlines)
          - categories: result.categories
          - published_date: result.published.date().isoformat()
          - pdf_url: result.pdf_url
          - ar5iv_url: f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        """

    def _deduplicate(self, papers: list[dict]) -> list[dict]:
        """Remove duplicates by arxiv_id, keeping the first occurrence."""
```

Note: The `arxiv` library's `Client.results()` is synchronous. Wrap it with `asyncio.to_thread` inside `fetch_today` to avoid blocking the event loop. Alternatively, since the fetcher is the only I/O-heavy sync part, you can call `_fetch_category` synchronously and wrap the entire `fetch_today` in `asyncio.to_thread`.

**Test:** `tests/test_fetcher.py`
- Use `unittest.mock.patch("arxiv.Client")` to mock the arxiv client. Make the mock return a list of 5 fake `arxiv.Result`-like objects (use `MagicMock` with the needed attributes: `entry_id`, `title`, `authors`, `summary`, `categories`, `published`, `pdf_url`). Set 3 with `published` = today, and 2 with `published` = 10 days ago.
- Call `fetch_today()`, assert only 3 papers returned (the 2 old ones filtered out).
- Test deduplication: mock returns papers where two have the same `arxiv_id`. Assert result has only unique IDs.
- Test the `arxiv_id` version stripping: mock an `entry_id` like `"http://arxiv.org/abs/2501.12345v2"` and assert extracted `arxiv_id` is `"2501.12345"`.
- Test the `cutoff_days` parameter: call `fetch_today(cutoff_days=15)` with the same mocked data — assert all 5 papers returned (the 10-day-old ones now within cutoff).

---

## Phase 4: Embedding System (`src/matcher/embedder.py`)

### Step 4.1 — Implement Embedder class

Create `src/matcher/embedder.py`:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, config: dict):
        # Store model_name from config["matching"]["embedding_model"].
        # Lazy-load model: set self._model = None, load on first use.

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string. Return a 1D numpy array (384 dims for MiniLM)."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Return a 2D numpy array of shape (N, dims)."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize a numpy array to bytes for SQLite BLOB storage."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes, dim: int = 384) -> np.ndarray:
        """Deserialize bytes back to a numpy array."""
        return np.frombuffer(blob, dtype=np.float32).reshape(-1) if len(blob) == dim * 4 \
            else np.frombuffer(blob, dtype=np.float32)

    def compute_embeddings(self, papers: list[dict], store: "PaperStore"):
        """Compute embeddings for papers that don't have them yet.
        For each paper: embed the abstract, serialize, call store.update_paper_embedding(id, blob)."""

    def compute_interest_embeddings(self, interests: list[dict], store: "PaperStore"):
        """Compute embeddings for interests that don't have them yet.
        For keywords: embed the keyword text (+ description if available).
        For paper/reference_paper types: embed the value (which is a paper title or arXiv ID description).
        Serialize and call store.update_interest_embedding(id, blob)."""
```

**Test:** `tests/test_embedder.py`
- Create an `Embedder` with the default model config.
- Call `embed_text("machine learning")` — assert result is a 1D `np.ndarray` with shape `(384,)`.
- Call `embed_texts(["hello", "world"])` — assert result shape is `(2, 384)`.
- Test round-trip: `serialize_embedding` → `deserialize_embedding` → assert `np.allclose` with original.
- Test that embeddings are normalized: assert `np.linalg.norm(embed_text("test"))` is approximately 1.0.

---

### Step 4.2 — Implement cosine similarity matching

Add to `src/matcher/embedder.py`:

```python
def find_similar(self, interests: list[dict], papers: list[dict],
                 top_n: int, threshold: float = 0.3) -> list[dict]:
    """
    Compute cosine similarity between each paper and all interest embeddings.
    For each paper, use the MAX similarity across all interests as its score.
    Return top_n papers (above threshold) sorted by score descending.

    Each returned dict includes all original paper fields plus 'embedding_score'.

    interests: list of dicts with 'embedding' key (bytes blob).
    papers: list of dicts with 'embedding' key (bytes blob).
    """
    # 1. Deserialize all interest embeddings into a 2D array (M, dims).
    # 2. Deserialize all paper embeddings into a 2D array (N, dims).
    # 3. Compute similarity matrix: papers_matrix @ interests_matrix.T → (N, M)
    # 4. For each paper (row), take max across all interests → (N,) scores.
    # 5. Filter by threshold, sort descending, return top_n.
```

**Test:** `tests/test_embedder.py` (extend)
- Create 3 synthetic interest embeddings (random normalized vectors) and 5 paper embeddings.
- Serialize them into bytes blobs.
- Call `find_similar` with `top_n=2, threshold=0.0`.
- Assert result has exactly 2 items, each with an `embedding_score` key.
- Assert results are sorted descending by `embedding_score`.
- Test threshold filtering: set threshold very high (e.g., 0.99) — assert empty result.

---

## Phase 5: Interest Manager (`src/interest/manager.py`)

### Step 5.1 — Implement InterestManager

Create `src/interest/manager.py`:

```python
class InterestManager:
    def __init__(self, store: "PaperStore", embedder: "Embedder"):
        self.store = store
        self.embedder = embedder

    def add_keyword(self, keyword: str, description: str = None) -> int:
        """Add a keyword interest. Compute and store its embedding. Return the new id."""
        interest_id = self.store.save_interest("keyword", keyword, description)
        text = f"{keyword}: {description}" if description else keyword
        embedding = self.embedder.embed_text(text)
        self.store.update_interest_embedding(
            interest_id, self.embedder.serialize_embedding(embedding)
        )
        return interest_id

    def add_paper(self, arxiv_id: str, description: str = None) -> int:
        """Add a past-paper interest.
        If no description is provided, auto-fetch the paper's abstract:
          1. Check DB first via store.get_paper_by_arxiv_id(arxiv_id)
          2. If not in DB, fetch from arXiv API via _fetch_abstract_from_arxiv(arxiv_id)
          3. If all lookups fail, fall back to using arxiv_id as text (log a warning)
        Embed the resolved text and store the embedding."""
        if not description:
            paper = self.store.get_paper_by_arxiv_id(arxiv_id)
            if paper:
                description = paper["abstract"]
                self.logger.info(f"Found abstract for {arxiv_id} in DB")
            else:
                description = self._fetch_abstract_from_arxiv(arxiv_id)
                if description:
                    self.logger.info(f"Fetched abstract for {arxiv_id} from arXiv")
                else:
                    self.logger.warning(
                        f"Could not fetch abstract for {arxiv_id}, using ID as fallback"
                    )

        interest_id = self.store.save_interest("paper", arxiv_id, description)
        text = description if description else arxiv_id
        embedding = self.embedder.embed_text(text)
        self.store.update_interest_embedding(
            interest_id, self.embedder.serialize_embedding(embedding)
        )
        return interest_id

    def add_reference_paper(self, arxiv_id: str, description: str = None) -> int:
        """Add a reference paper interest. Same auto-fetch logic as add_paper
        but with type='reference_paper'."""
        # Same auto-fetch logic as add_paper but with type "reference_paper".

    def _fetch_abstract_from_arxiv(self, arxiv_id: str) -> str | None:
        """Fetch a single paper's abstract from arXiv by ID.
        Uses arxiv.Search(id_list=[arxiv_id]) to fetch the paper metadata.
        Returns the abstract text, or None if the fetch fails or paper not found."""
        try:
            import arxiv
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(arxiv.Client().results(search))
            if results:
                return results[0].summary.replace("\n", " ").strip()
        except Exception as e:
            self.logger.warning(f"Failed to fetch abstract from arXiv for {arxiv_id}: {e}")
        return None

    def remove_interest(self, interest_id: int):
        """Delete an interest by ID."""
        self.store.delete_interest(interest_id)

    def update_interest(self, interest_id: int, value: str = None, description: str = None):
        """Update an interest and recompute its embedding."""
        self.store.update_interest(interest_id, value=value, description=description)
        updated = self.store.get_interest_by_id(interest_id)
        text = f"{updated['value']}: {updated['description']}" if updated.get("description") \
            else updated["value"]
        embedding = self.embedder.embed_text(text)
        self.store.update_interest_embedding(
            interest_id, self.embedder.serialize_embedding(embedding)
        )

    def get_all_interests(self) -> list[dict]:
        """Return all interests from the store."""
        return self.store.get_all_interests()

    def get_interests_with_embeddings(self) -> list[dict]:
        """Return interests that have computed embeddings."""
        return self.store.get_interests_with_embeddings()

    def recompute_all_embeddings(self):
        """Recompute embeddings for all interests. Useful after model change."""
        for interest in self.store.get_all_interests():
            text = f"{interest['value']}: {interest['description']}" \
                if interest.get("description") else interest["value"]
            embedding = self.embedder.embed_text(text)
            self.store.update_interest_embedding(
                interest["id"], self.embedder.serialize_embedding(embedding)
            )
```

**Test:** `tests/test_interest_manager.py`
- Create a temp DB, a real `PaperStore`, and a real `Embedder` instance.
- Call `add_keyword("transformer architectures")` — assert returns an int ID.
- Call `get_all_interests()` — assert 1 interest exists with type "keyword" and value "transformer architectures".
- Call `get_interests_with_embeddings()` — assert 1 item returned with non-null embedding.
- Deserialize the embedding — assert shape is `(384,)`.
- Call `update_interest(id, value="attention mechanisms")` — verify value changed and embedding changed (not equal to original bytes).
- Call `remove_interest(id)` — verify `get_all_interests()` returns empty list.
- Call `add_paper("2501.12345", "My paper about reinforcement learning")` — verify interest type is "paper" and embedding is computed.
- Test auto-fetch from DB: insert a paper with arxiv_id "2501.99999" and an abstract into the store first. Call `add_paper("2501.99999")` with no description — verify the interest's description is the paper's abstract.
- Test auto-fetch from arXiv: mock `_fetch_abstract_from_arxiv` to return "Fetched abstract". Call `add_paper("9999.99999")` with no description and no paper in DB — verify the interest uses the fetched abstract.
- Test fallback: mock `_fetch_abstract_from_arxiv` to return None. Call `add_paper("0000.00000")` with no description and no paper in DB — verify a warning is logged and the embedding still gets computed (using the arxiv_id as fallback text).

---

## Phase 6: LLM Re-ranker (`src/matcher/ranker.py`)

### Step 6.1 — Implement LLMRanker

Create `src/matcher/ranker.py`:

```python
class LLMRanker:
    def __init__(self, llm: "LLMProvider", config: dict):
        self.llm = llm
        self.top_k = config["matching"]["llm_top_k"]

    async def rerank(self, candidates: list[dict], interests: list[dict],
                     top_k: int = None, max_concurrent: int = 5) -> list[dict]:
        """
        Re-rank candidate papers using the LLM with concurrent scoring.
        For each candidate paper, call the LLM to score relevance (1-10) against
        the user's interests. Uses asyncio.gather with a semaphore to limit
        concurrent LLM calls (default: 5 concurrent).

        candidates: list of paper dicts (from embedding matcher, with 'embedding_score').
        interests: list of interest dicts (with 'value' and 'description').
        top_k: override for self.top_k.
        max_concurrent: maximum number of concurrent LLM calls (default 5).

        Returns top_k papers sorted by llm_score descending.
        Each returned dict includes original paper fields + 'llm_score' + 'llm_reason'.
        """
        k = top_k or self.top_k
        # Format interests as a readable text block.
        interests_text = self._format_interests(interests)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_limit(paper):
            async with semaphore:
                score_data = await self._score_paper(paper, interests_text)
                return {**paper, **score_data}

        results = await asyncio.gather(
            *[score_with_limit(paper) for paper in candidates]
        )

        # Sort by llm_score descending, return top_k.
        results = sorted(results, key=lambda x: x.get("llm_score", 0), reverse=True)
        return results[:k]

    async def _score_paper(self, paper: dict, interests_text: str) -> dict:
        """Ask the LLM to score a single paper.
        Returns {"llm_score": float, "llm_reason": str}.

        Prompt the LLM with the paper title + abstract + interest list.
        Instruct it to return JSON: {"score": <1-10>, "reason": "<1-2 sentence explanation>"}
        Use self.llm.complete_json().
        If parsing fails, return {"llm_score": 0, "llm_reason": "Scoring failed"}.
        """

    def _format_interests(self, interests: list[dict]) -> str:
        """Format interests into a readable text block for the LLM prompt.
        e.g., '- keyword: transformer architectures\n- paper: reinforcement learning'"""
```

**Test:** `tests/test_ranker.py`
- Create a mock `LLMProvider` that returns `'{"score": 8.5, "reason": "Highly relevant"}'` for `complete_json`. Track the number of calls.
- Create `LLMRanker` with the mock provider and config `{"matching": {"llm_top_k": 2}}`.
- Call `rerank` with 5 candidate papers and 2 interests.
- Assert result has length 2 (top_k).
- Assert each result has `llm_score` == 8.5 and `llm_reason` == "Highly relevant".
- Assert mock LLM was called exactly 5 times (once per candidate, all concurrently).
- Test failure case: mock returns invalid JSON. Assert `llm_score` == 0 and `llm_reason` contains "failed".
- Test concurrency: create a mock LLM that tracks max simultaneous calls (using an asyncio counter). Call `rerank` with `max_concurrent=3` and 10 candidates. Assert max simultaneous calls never exceeded 3.

---

## Phase 7: Report Generator (`src/report/generator.py`)

### Step 7.1 — Implement ReportGenerator for general report

Create `src/report/generator.py`:

```python
class ReportGenerator:
    def __init__(self, llm: "LLMProvider"):
        self.llm = llm

    async def generate_general(self, papers: list[dict], run_date: str) -> str:
        """Generate a Markdown general report.
        Contents:
        1. Header: "# Daily Paper Report - {run_date}"
        2. "## General Report"
        3. "### Today's Overview" — total count, count per category.
           Compute category distribution from papers (each paper has 'categories' list;
           count primary category = first in list).
        4. "### Trending Topics" — call LLM with all paper titles to identify 3-5 trends.
        5. "### Highlight Papers" — call LLM to select 3-5 noteworthy papers from the list.
           Format: numbered list with title, authors, one-line description.

        Return the complete Markdown string.
        """
```

Implementation details:
- For the overview section, compute counts purely in Python (no LLM needed). Build a `Counter` from each paper's primary category.
- For trending topics, send the LLM a list of all paper titles and ask it to identify 3-5 emerging topics with brief descriptions.
- For highlight papers, send the LLM a list of paper titles + one-line abstracts (first 150 chars of abstract) and ask it to pick 3-5 noteworthy ones with reasons.

**Test:** `tests/test_report_generator.py`
- Create a mock LLM that returns canned Markdown strings.
- Call `generate_general` with 10 fake papers (3 in cs.AI, 4 in cs.CL, 3 in cs.LG).
- Assert returned Markdown contains the header with the run date.
- Assert the overview section includes correct counts (e.g., "cs.AI: 3").
- Assert "Trending Topics" and "Highlight Papers" sections are present.

---

### Step 7.2 — Implement specific report generation

Add to `ReportGenerator`:

```python
async def generate_specific(self, ranked_papers: list[dict],
                            interests: list[dict], run_date: str) -> str:
    """Generate a Markdown specific report.
    Contents:
    1. "## Specific Report (Based on Your Interests)"
    2. Numbered list of ranked papers, each with:
       - Title (bold)
       - Relevance score: "{llm_score}/10"
       - "Why it matters to you: {llm_reason}"
    3. "## Related Papers" section: for each ranked paper, include
       - Title, Authors, Categories, Abstract (first 200 chars), arXiv link.

    Return the Markdown string.
    """
```

This method should NOT call the LLM — it formats the already-scored data from the ranker.

**Test:** `tests/test_report_generator.py` (extend)
- Call `generate_specific` with 3 pre-scored papers (with `llm_score` and `llm_reason` fields) and 2 interests.
- Assert each paper's title appears in the output.
- Assert each paper's `llm_score` appears (e.g., "8.5/10").
- Assert the "Related Papers" section contains arXiv links.

---

## Phase 8: Email Sender (`src/email/sender.py`)

### Step 8.1 — Implement EmailSender

Create `src/email/sender.py`:

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import markdown
import premailer

class EmailSender:
    def __init__(self, config: dict):
        # Store config["email"] sub-dict.
        # Extract SMTP settings: host, port, username (from env), password (from env).
        # Store from_address, to_addresses (list), subject_prefix.

    def render_markdown_to_html(self, md_content: str) -> str:
        """Convert Markdown to HTML and inline CSS.
        1. markdown.markdown(md_content, extensions=["tables", "fenced_code"])
        2. Wrap in a basic HTML template with a <style> block for readability
           (body font, heading sizes, table borders, link colors).
        3. premailer.transform(html) to inline CSS.
        Return the final HTML string."""

    async def send(self, general_report: str, specific_report: str,
                   ranked_papers: list[dict], run_date: str):
        """Assemble and send the daily email.
        1. Combine general_report + specific_report into one Markdown string.
        2. Convert to HTML via render_markdown_to_html().
        3. Create MIMEMultipart message with subject: "{subject_prefix} {run_date}".
        4. Attach HTML as MIMEText("...", "html").
        5. Connect to SMTP, starttls, login, send.
        Use asyncio.to_thread() to wrap the synchronous smtplib calls."""

    def _build_email(self, html_content: str, subject: str) -> MIMEMultipart:
        """Build the MIME message object."""
```

**Test:** `tests/test_email_sender.py`
- Test `render_markdown_to_html`: pass simple Markdown (`"# Hello\n\n**bold**"`), assert result contains `<h1>Hello</h1>` and `<strong>bold</strong>`. Assert result contains `style=` (inline CSS from premailer).
- Test `_build_email`: assert the MIMEMultipart message has correct subject, from, to, and HTML payload.
- Test `send` with `unittest.mock.patch("smtplib.SMTP")`: assert `starttls()`, `login()`, and `send_message()` are called. Assert no actual email is sent.

---

## Phase 9: Paper Summarizer (`src/summarizer/paper_summarizer.py`)

### Step 9.1 — Implement HTML fetching and text extraction

Create `src/summarizer/paper_summarizer.py`:

```python
import requests
from bs4 import BeautifulSoup

class PaperSummarizer:
    def __init__(self, llm: "LLMProvider", store: "PaperStore"):
        self.llm = llm
        self.store = store

    def fetch_paper_text(self, ar5iv_url: str) -> str:
        """Fetch the ar5iv HTML page and extract the paper's main text.
        1. requests.get(ar5iv_url, timeout=30).
        2. Parse with BeautifulSoup(html, "lxml").
        3. Find the main content: look for <article> tag, or fall back to
           class "ltx_document" or "ltx_page_main".
        4. Extract text from all <p>, <h2>, <h3> tags within the main content.
        5. Join paragraphs with double newlines.
        6. Truncate to 15000 characters (LLM context limit safety).
        Return the extracted text. Raise RuntimeError if fetch fails."""

    async def summarize(self, paper_id: int, mode: str = "brief") -> str:
        """Generate a summary for a paper.
        1. Check cache: self.store.get_summary(paper_id, mode). If exists, return it.
        2. Get paper info from store.
        3. Fetch full text via fetch_paper_text(paper["ar5iv_url"]).
           If fetch fails, fall back to using just the abstract.
        4. Build prompt based on mode:
           - "brief": "Summarize this paper in 1-2 paragraphs covering core contributions
             and methodology."
           - "detailed": "Provide a structured summary with sections: Motivation, Method,
             Experiments, Conclusions, Limitations."
        5. Call self.llm.complete(prompt, system="You are a scientific paper summarizer.")
        6. Save to cache: self.store.save_summary(paper_id, mode, result, llm_provider_name).
        7. Return the summary text."""
```

**Test:** `tests/test_summarizer.py`
- Test `fetch_paper_text`: use `unittest.mock.patch("requests.get")` with a mock response containing a simple HTML document:
  ```html
  <html><body><article><p>Introduction text.</p><p>Method text.</p></article></body></html>
  ```
  Assert returned text contains "Introduction text." and "Method text.".
- Test `fetch_paper_text` raises `RuntimeError` when response status is 404.
- Test `summarize` with cache hit: mock `store.get_summary` to return a cached string. Assert LLM is NOT called.
- Test `summarize` without cache: mock `store.get_summary` returning None, mock LLM returning "Summary text", mock `fetch_paper_text`. Assert `store.save_summary` is called with the result.

---

## Phase 10: Pipeline Orchestrator (`src/pipeline.py`)

### Step 10.1 — Implement DailyPipeline

Create `src/pipeline.py`:

```python
import logging
from datetime import date

class DailyPipeline:
    def __init__(self, config: dict):
        # Initialize all components:
        from src.store.database import PaperStore
        from src.fetcher.arxiv_fetcher import ArxivFetcher
        from src.matcher.embedder import Embedder
        from src.matcher.ranker import LLMRanker
        from src.interest.manager import InterestManager
        from src.report.generator import ReportGenerator
        from src.email.sender import EmailSender
        from src.llm import create_llm_provider

        self.config = config
        self.store = PaperStore(config["database"]["path"])
        self.fetcher = ArxivFetcher(config)
        self.embedder = Embedder(config)
        self.llm = create_llm_provider(config)
        self.ranker = LLMRanker(self.llm, config)
        self.interest_mgr = InterestManager(self.store, self.embedder)
        self.report_gen = ReportGenerator(self.llm)
        self.email_sender = EmailSender(config)
        self.logger = logging.getLogger(__name__)

    async def run(self) -> dict:
        """Execute the full daily pipeline. Return a summary dict.
        Steps:
        1. Fetch papers from arXiv.
        2. Save to DB (deduplication handled by store).
        3. Compute embeddings for new papers.
        4. Get all interests with embeddings.
        5. If no interests exist, log a warning and skip matching.
        6. Run embedding similarity matching → top-N candidates.
        7. Run LLM re-ranking → top-K results.
        8. Save match records to DB.
        9. Generate general report.
        10. Generate specific report.
        11. Send email (if email.enabled is true in config).
        12. Save report to DB.
        13. Return summary: {"date": ..., "papers_fetched": ..., "new_papers": ...,
            "matches": ..., "email_sent": ...}
        """
        run_date = date.today().isoformat()
        self.logger.info(f"Starting daily pipeline for {run_date}")

        # Step 1: Fetch
        self.logger.info("Fetching papers from arXiv...")
        papers = await self.fetcher.fetch_today()
        self.logger.info(f"Fetched {len(papers)} papers")

        # Step 2: Save
        new_papers = self.store.save_papers(papers)
        self.logger.info(f"Saved {len(new_papers)} new papers ({len(papers) - len(new_papers)} duplicates)")

        # Step 3: Embed
        self.logger.info("Computing embeddings for new papers...")
        self.embedder.compute_embeddings(new_papers, self.store)

        # Step 4: Interests
        interests = self.interest_mgr.get_interests_with_embeddings()
        if not interests:
            self.logger.warning("No interests configured. Skipping matching.")
            # Still generate general report.
            general = await self.report_gen.generate_general(new_papers, run_date)
            self.store.save_report(run_date, general, "", len(new_papers), 0)
            return {"date": run_date, "papers_fetched": len(papers),
                    "new_papers": len(new_papers), "matches": 0, "email_sent": False}

        # Step 5-6: Match (only today's papers, not the entire historical DB)
        todays_papers = self.store.get_papers_by_date_with_embeddings(run_date)
        self.logger.info(f"Found {len(todays_papers)} papers with embeddings for {run_date}")
        top_n = self.config["matching"]["embedding_top_n"]
        threshold = self.config["matching"]["similarity_threshold"]
        candidates = self.embedder.find_similar(interests, todays_papers, top_n, threshold)
        self.logger.info(f"Embedding matcher found {len(candidates)} candidates")

        # Step 7: Re-rank
        ranked = await self.ranker.rerank(candidates, interests)
        self.logger.info(f"LLM re-ranker selected {len(ranked)} papers")

        # Step 8: Save matches
        for paper in ranked:
            self.store.save_match(
                paper["id"], run_date,
                paper.get("embedding_score", 0),
                paper.get("llm_score", 0),
                paper.get("llm_reason", "")
            )

        # Step 9-10: Reports
        general = await self.report_gen.generate_general(new_papers, run_date)
        specific = await self.report_gen.generate_specific(ranked, interests, run_date)

        # Step 11: Email
        email_sent = False
        if self.config.get("email", {}).get("enabled", False):
            try:
                await self.email_sender.send(general, specific, ranked, run_date)
                email_sent = True
                self.logger.info("Email sent successfully")
            except Exception as e:
                self.logger.error(f"Email sending failed: {e}")

        # Step 12: Save report
        self.store.save_report(run_date, general, specific, len(new_papers), len(ranked))

        return {"date": run_date, "papers_fetched": len(papers),
                "new_papers": len(new_papers), "matches": len(ranked),
                "email_sent": email_sent}
```

**Test:** `tests/test_pipeline.py`
- Mock every component (fetcher, store, embedder, ranker, report_gen, email_sender, interest_mgr) using `unittest.mock.patch`.
- Mock fetcher to return 5 papers. Mock store.save_papers to return 3 (2 duplicates).
- Mock interest_mgr to return 2 interests with embeddings.
- Mock store.get_papers_by_date_with_embeddings to return today's papers with embeddings.
- Mock embedder.find_similar to return 4 candidates.
- Mock ranker.rerank to return 2 ranked results.
- Mock report_gen methods to return canned Markdown.
- Call `pipeline.run()`. Assert:
  - `fetcher.fetch_today` was called once.
  - `store.save_papers` was called with the 5 papers.
  - `embedder.compute_embeddings` was called with the 3 new papers.
  - `ranker.rerank` was called with 4 candidates and 2 interests.
  - `store.save_match` was called twice (once per ranked paper).
  - `store.save_report` was called once.
  - Returned dict has correct counts.

- Test the "no interests" path: mock interest_mgr to return empty list. Assert ranking is skipped and general report is still generated.

---

## Phase 11: Scheduler and CLI Entry Points

### Step 11.1 — Implement Scheduler

Create `src/scheduler/scheduler.py`:

```python
import asyncio
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

class PipelineScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.scheduler = BlockingScheduler()

    def start(self):
        """Start the scheduler with the configured cron expression.
        Parse cron string from config["scheduler"]["cron"] (format: "M H * * *").
        Split into fields: minute, hour, day, month, day_of_week.
        Add job using CronTrigger.
        Call self.scheduler.start() (blocks forever)."""
        cron = self.config["scheduler"]["cron"]
        parts = cron.split()
        trigger = CronTrigger(
            minute=parts[0], hour=parts[1], day=parts[2],
            month=parts[3], day_of_week=parts[4]
        )
        self.scheduler.add_job(self._run_pipeline, trigger)
        self.scheduler.start()

    def _run_pipeline(self):
        """Create and run the DailyPipeline inside an asyncio event loop."""
        from src.pipeline import DailyPipeline
        pipeline = DailyPipeline(self.config)
        result = asyncio.run(pipeline.run())
        print(f"Pipeline completed: {result}")
```

**Test:** `tests/test_scheduler.py`
- Instantiate `PipelineScheduler` with a config containing `cron: "30 9 * * *"`.
- Use `unittest.mock.patch.object` on the scheduler's `BlockingScheduler` to prevent actual blocking.
- Call `start()`. Assert `add_job` was called with a `CronTrigger`.
- Verify the trigger's fields match: minute=30, hour=9.

---

### Step 11.2 — Implement CLI entry point

Create `src/main.py`:

```python
import argparse
import asyncio
import logging

def main():
    parser = argparse.ArgumentParser(description="Daily Paper Collector")
    parser.add_argument("--mode", choices=["scheduler", "run"], default="run",
                        help="'scheduler' starts the cron scheduler; 'run' executes once.")
    parser.add_argument("--config", default=None,
                        help="Path to config file (default: config/config.yaml relative to project root).")
    args = parser.parse_args()

    from src.config import load_config, setup_logging
    setup_logging()
    config = load_config(args.config)

    if args.mode == "scheduler":
        from src.scheduler.scheduler import PipelineScheduler
        scheduler = PipelineScheduler(config)
        scheduler.start()
    elif args.mode == "run":
        from src.pipeline import DailyPipeline
        pipeline = DailyPipeline(config)
        result = asyncio.run(pipeline.run())
        print(f"Pipeline completed: {result}")

if __name__ == "__main__":
    main()
```

Create `scripts/run_pipeline.py`:

```python
"""Standalone entry point for GitHub Actions / CI/CD."""
import asyncio
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.pipeline import DailyPipeline

def main():
    config = load_config()
    pipeline = DailyPipeline(config)
    result = asyncio.run(pipeline.run())
    print(f"Pipeline completed: {result}")
    if result["new_papers"] == 0:
        print("Warning: No new papers fetched.")

if __name__ == "__main__":
    main()
```

**Test:** `tests/test_main.py`
- Use `unittest.mock.patch("sys.argv", ["main", "--mode", "run", "--config", "config/config.yaml"])`.
- Mock `DailyPipeline` and `asyncio.run`.
- Call `main()`. Assert `DailyPipeline` was instantiated and `run()` was called.
- Test with `--mode scheduler`: mock `PipelineScheduler`. Assert `start()` was called.

---

## Phase 12: Streamlit GUI

### Step 12.1 — Create main app entry with navigation

Create `gui/app.py`:

```python
import streamlit as st
import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, setup_logging
from src.store.database import PaperStore
from src.matcher.embedder import Embedder

setup_logging()

@st.cache_resource
def get_config() -> dict:
    """Cached config dict. Cleared on app restart."""
    return load_config()

@st.cache_resource
def get_store() -> PaperStore:
    """Cached PaperStore instance. Avoids recreating DB connections on every rerun."""
    config = get_config()
    return PaperStore(config["database"]["path"])

@st.cache_resource
def get_embedder() -> Embedder:
    """Cached Embedder instance. Avoids reloading the ~80MB sentence-transformer model
    on every Streamlit rerun."""
    config = get_config()
    return Embedder(config)

def main():
    st.set_page_config(page_title="Daily Paper Collector", layout="wide")

    pages = {
        "Dashboard": "gui/pages/dashboard.py",
        "Papers": "gui/pages/papers.py",
        "Interests": "gui/pages/interests.py",
        "Reports": "gui/pages/reports.py",
        "Settings": "gui/pages/settings.py",
    }

    # Use st.navigation or st.sidebar radio for page selection.
    # Each page is a separate .py file with a render() function.
    page = st.sidebar.radio("Navigation", list(pages.keys()))

    if page == "Dashboard":
        from gui.pages.dashboard import render
    elif page == "Papers":
        from gui.pages.papers import render
    elif page == "Interests":
        from gui.pages.interests import render
    elif page == "Reports":
        from gui.pages.reports import render
    elif page == "Settings":
        from gui.pages.settings import render

    render(get_store())

if __name__ == "__main__":
    main()
```

**Test:**
- Manual test: run `streamlit run gui/app.py`. Verify the sidebar appears with 5 navigation options. Clicking each should load without errors (pages will be stubs at first).

---

### Step 12.2 — Dashboard page

Create `gui/pages/dashboard.py`:

```python
import streamlit as st
from datetime import date

def render(store):
    st.title("Dashboard")

    today = date.today().isoformat()

    # Row 1: Metrics
    col1, col2, col3 = st.columns(3)
    papers_today = store.get_papers_by_date(today)
    report = store.get_report_by_date(today)
    matches = store.get_matches_by_date(today)

    col1.metric("Papers Today", len(papers_today))
    col2.metric("Matches Today", len(matches))
    col3.metric("Reports", len(store.get_all_report_dates()))

    # Row 2: Latest General Report preview
    if report and report.get("general_report"):
        st.subheader("Latest General Report")
        st.markdown(report["general_report"][:1000] + "...")

    # Row 3: Latest Specific Report preview
    if report and report.get("specific_report"):
        st.subheader("Latest Specific Report")
        st.markdown(report["specific_report"][:1000] + "...")

    # Manual trigger button
    if st.button("Run Pipeline Now"):
        with st.spinner("Running pipeline..."):
            import asyncio
            from src.pipeline import DailyPipeline
            from src.config import load_config
            config = load_config()
            pipeline = DailyPipeline(config)
            result = asyncio.run(pipeline.run())
            st.success(f"Pipeline completed: {result}")
            st.rerun()
```

**Test:**
- Manual: launch the GUI, verify Dashboard shows metrics (all zeros for empty DB).
- Verify "Run Pipeline Now" button appears and clicking it triggers the pipeline (may need mocked arXiv API for a dry run; alternatively, just verify it doesn't crash).

---

### Step 12.3 — Papers page with browsing, search, and summarization

Create `gui/pages/papers.py`:

```python
import streamlit as st
from datetime import date, timedelta

def render(store):
    st.title("Papers")

    # Date selector
    selected_date = st.date_input("Select date", value=date.today())
    date_str = selected_date.isoformat()

    # Search box
    search_query = st.text_input("Search papers (title or abstract)")

    if search_query:
        papers = store.search_papers(search_query)
        st.info(f"Found {len(papers)} results for '{search_query}'")
    else:
        papers = store.get_papers_by_date(date_str)
        st.info(f"{len(papers)} papers on {date_str}")

    # Display papers
    for paper in papers:
        with st.expander(f"**{paper['title']}**"):
            st.write(f"**Authors:** {paper['authors']}")
            st.write(f"**Categories:** {paper['categories']}")
            st.write(f"**Published:** {paper['published_date']}")
            st.markdown(paper["abstract"])
            st.markdown(f"[arXiv]({paper['pdf_url']})")

            # Summarize buttons
            col1, col2 = st.columns(2)
            if col1.button("Brief Summary", key=f"brief_{paper['id']}"):
                _show_summary(store, paper, "brief")
            if col2.button("Detailed Summary", key=f"detailed_{paper['id']}"):
                _show_summary(store, paper, "detailed")

def _show_summary(store, paper, mode):
    """Check cache or generate summary."""
    cached = store.get_summary(paper["id"], mode)
    if cached:
        st.markdown(cached["content"])
    else:
        with st.spinner(f"Generating {mode} summary..."):
            import asyncio
            from src.config import load_config
            from src.llm import create_llm_provider
            from src.summarizer.paper_summarizer import PaperSummarizer
            config = load_config()
            llm = create_llm_provider(config)
            summarizer = PaperSummarizer(llm, store)
            summary = asyncio.run(summarizer.summarize(paper["id"], mode))
            st.markdown(summary)
```

Create `gui/components/paper_card.py`:

```python
import streamlit as st

def render_paper_card(paper: dict):
    """Render a single paper card inside an expander."""
    with st.expander(f"**{paper['title']}**"):
        st.write(f"**Authors:** {paper['authors']}")
        st.write(f"**Categories:** {paper['categories']}")
        st.markdown(paper["abstract"][:300] + "...")
        st.markdown(f"[arXiv]({paper.get('pdf_url', '#')})")
```

**Test:**
- Manual: launch GUI, navigate to Papers. Verify date selector and search box appear. With empty DB, verify "0 papers" message. After running the pipeline (or inserting test data), verify papers display with expanders. Test the search box with a keyword.

---

### Step 12.4 — Interests management page

Create `gui/pages/interests.py`:

```python
import streamlit as st

def render(store):
    st.title("Interest Management")

    # Show existing interests
    interests = store.get_all_interests()
    if interests:
        st.subheader(f"Current Interests ({len(interests)})")
        for interest in interests:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"**[{interest['type']}]** {interest['value']}"
                      + (f" — {interest['description']}" if interest.get('description') else ""))
            has_emb = "✓" if interest.get("embedding") else "✗"
            col2.write(f"Embedding: {has_emb}")
            if col3.button("Delete", key=f"del_{interest['id']}"):
                store.delete_interest(interest["id"])
                st.rerun()
    else:
        st.info("No interests configured yet. Add some below.")

    # Add new interest form
    st.subheader("Add New Interest")
    with st.form("add_interest"):
        interest_type = st.selectbox("Type", ["keyword", "paper", "reference_paper"])
        value = st.text_input("Value (keyword text or arXiv ID)")
        description = st.text_input("Description (optional)")
        submitted = st.form_submit_button("Add Interest")

        if submitted and value:
            from gui.app import get_embedder
            from src.interest.manager import InterestManager

            embedder = get_embedder()  # Uses cached instance (no model reload)
            mgr = InterestManager(store, embedder)

            if interest_type == "keyword":
                mgr.add_keyword(value, description or None)
            elif interest_type == "paper":
                mgr.add_paper(value, description or None)
            else:
                mgr.add_reference_paper(value, description or None)
            st.success(f"Added {interest_type}: {value}")
            st.rerun()
```

**Test:**
- Manual: launch GUI, navigate to Interests. Add a keyword interest "transformer". Verify it appears in the list with an embedding checkmark. Delete it and verify removal. Add a paper interest with an arXiv ID.

---

### Step 12.5 — Reports page

Create `gui/pages/reports.py`:

```python
import streamlit as st

def render(store):
    st.title("Reports")

    report_dates = store.get_all_report_dates()
    if not report_dates:
        st.info("No reports generated yet. Run the pipeline first.")
        return

    selected_date = st.selectbox("Select report date", report_dates)
    report = store.get_report_by_date(selected_date)

    if report:
        tab1, tab2 = st.tabs(["General Report", "Specific Report"])
        with tab1:
            if report.get("general_report"):
                st.markdown(report["general_report"])
            else:
                st.info("No general report for this date.")
        with tab2:
            if report.get("specific_report"):
                st.markdown(report["specific_report"])
            else:
                st.info("No specific report for this date.")

        # Show matches for this date
        st.subheader("Match Results")
        matches = store.get_matches_by_date(selected_date)
        if matches:
            for m in matches:
                with st.expander(f"**{m['title']}** (LLM: {m.get('llm_score', 'N/A')}/10, "
                                f"Embedding: {m.get('embedding_score', 0):.3f})"):
                    st.write(f"**Reason:** {m.get('llm_reason', 'N/A')}")
                    st.markdown(m.get("abstract", "")[:300] + "...")
        else:
            st.info("No matches for this date.")
```

Create `gui/components/report_viewer.py`:

```python
import streamlit as st

def render_report(report_markdown: str):
    """Render a Markdown report in Streamlit."""
    st.markdown(report_markdown)
```

**Test:**
- Manual: launch GUI, navigate to Reports. With empty DB, verify info message. After running the pipeline, verify date dropdown appears and reports render correctly in tabs.

---

### Step 12.6 — Settings page

Create `gui/pages/settings.py`:

```python
import streamlit as st
import yaml

def render(store):
    st.title("Settings")

    from src.config import load_config

    config = load_config()

    # Display current config (read-only for safety)
    st.subheader("Current Configuration")
    st.code(yaml.dump(config, default_flow_style=False), language="yaml")

    # Editable sections
    st.subheader("ArXiv Categories")
    categories = st.text_area("Categories (one per line)",
                              value="\n".join(config.get("arxiv", {}).get("categories", [])))

    st.subheader("LLM Provider")
    provider = st.selectbox("Provider",
                           ["openai", "claude", "claude_code"],
                           index=["openai", "claude", "claude_code"].index(
                               config.get("llm", {}).get("provider", "openai")))

    st.subheader("Email")
    email_enabled = st.checkbox("Enable email",
                                value=config.get("email", {}).get("enabled", False))

    if st.button("Save Settings"):
        config["arxiv"]["categories"] = [c.strip() for c in categories.split("\n") if c.strip()]
        config["llm"]["provider"] = provider
        config["email"]["enabled"] = email_enabled
        with open("config/config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        st.success("Settings saved!")

    # Email test
    st.subheader("Test Email")
    if st.button("Send Test Email"):
        with st.spinner("Sending test email..."):
            try:
                import asyncio
                from src.email.sender import EmailSender
                sender = EmailSender(config)
                asyncio.run(sender.send(
                    "# Test Email\n\nThis is a test.",
                    "## No specific report", [], "test"
                ))
                st.success("Test email sent!")
            except Exception as e:
                st.error(f"Failed: {e}")
```

**Test:**
- Manual: launch GUI, navigate to Settings. Verify current config displays. Change a category, save, reload page — verify change persists in config file. Test the email button (expect failure without valid SMTP credentials, but verify it doesn't crash the app).

---

### Step 12.7 — Automated GUI tests with Streamlit AppTest

Create `tests/test_gui.py`:

```python
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock

def test_dashboard_renders():
    """Test that the dashboard page renders without errors on an empty DB."""
    at = AppTest.from_file("gui/app.py")
    at.run()
    assert not at.exception

def test_papers_page_empty_db():
    """Test that papers page handles empty database gracefully."""
    at = AppTest.from_file("gui/app.py")
    at.run()
    # Navigate to Papers page
    at.sidebar.radio[0].set_value("Papers")
    at.run()
    assert not at.exception
    # Should display "0 papers" message
    assert any("0 papers" in str(el) for el in at.info)

def test_interests_page_add_keyword():
    """Test adding a keyword interest via the GUI form."""
    at = AppTest.from_file("gui/app.py")
    at.run()
    at.sidebar.radio[0].set_value("Interests")
    at.run()
    # Verify form elements exist
    assert len(at.selectbox) >= 1  # Type selector
    assert len(at.text_input) >= 1  # Value input
    assert not at.exception

def test_reports_page_empty():
    """Test that reports page handles no reports gracefully."""
    at = AppTest.from_file("gui/app.py")
    at.run()
    at.sidebar.radio[0].set_value("Reports")
    at.run()
    assert not at.exception
    assert any("No reports" in str(el) for el in at.info)

def test_settings_page_renders():
    """Test that settings page renders config without errors."""
    at = AppTest.from_file("gui/app.py")
    at.run()
    at.sidebar.radio[0].set_value("Settings")
    at.run()
    assert not at.exception
```

Use `unittest.mock.patch` to mock PaperStore and Embedder for isolated GUI tests that don't need a real DB or model. These tests verify that pages render without exceptions and display appropriate empty-state messages.

**Test command:** `pytest tests/test_gui.py -v`

---

## Phase 13: Integration Testing

### Step 13.1 — End-to-end pipeline test with mocked external services

Create `tests/test_integration.py`:

```python
"""End-to-end integration test with mocked external services (arXiv API, LLM APIs, SMTP)."""
```

This test:

1. Creates a temp directory with a temp DB and temp config file.
2. Adds 2 keyword interests to the DB using `InterestManager` (with real `Embedder` for real embeddings).
3. Mocks `ArxivFetcher.fetch_today` to return 10 synthetic papers (with titles/abstracts relevant to the interests).
4. Mocks the `LLMProvider.complete` and `complete_json` to return deterministic responses:
   - For scoring: `{"score": 7, "reason": "Test reason"}`
   - For report generation: canned Markdown strings.
5. Mocks `smtplib.SMTP` to prevent actual email sending.
6. Runs `DailyPipeline.run()`.
7. Asserts:
   - 10 papers are in the DB.
   - All 10 have embeddings.
   - Matches table has entries.
   - A report record exists for today's date.
   - `smtplib.SMTP.send_message` was called once.
   - The returned dict has `new_papers == 10` and `matches > 0`.

**Test command:** `pytest tests/test_integration.py -v`

---

### Step 13.2 — Full test suite execution and coverage

Run the complete test suite:

```bash
pytest --cov=src --cov-report=term-missing -v
```

**Acceptance criteria:**
- All tests pass.
- Coverage for `src/store/database.py` >= 90%.
- Coverage for `src/matcher/embedder.py` >= 85%.
- Coverage for `src/pipeline.py` >= 80%.
- Overall `src/` coverage >= 75%.
- `ruff check .` reports zero issues.
- `ruff format --check .` reports zero issues.

---

## Phase 14: Polish and Hardening

> **Note:** Logging setup has been moved to Phase 0.2 (`src/config.py` → `setup_logging()`).
> All components should include `self.logger = logging.getLogger(__name__)` in their `__init__`
> from the start (not as a polish step). The logging points listed below should already be present
> in their respective components:
> - `ArxivFetcher`: log category being fetched, number of papers found per category, count before/after date filtering.
> - `Embedder`: log number of embeddings computed.
> - `LLMRanker`: log number of candidates received, number of results returned.
> - `ReportGenerator`: log report generation start/complete.
> - `EmailSender`: log email recipient and send success/failure.
> - `PipelineScheduler`: log next scheduled run time on start.
> - `InterestManager`: log abstract auto-fetch source (DB/arXiv/fallback).

### Step 14.1 — Add error handling and retries

Add try/except blocks in these critical locations:

1. `ArxivFetcher._fetch_category`: catch `Exception`, log error, return empty list for that category (don't fail the whole run).
2. `LLMRanker._score_paper`: catch `Exception`, return score 0 (don't skip the paper).
3. `EmailSender.send`: catch `smtplib.SMTPException`, log error, re-raise (pipeline.py already catches this).
4. `PaperSummarizer.fetch_paper_text`: catch `requests.RequestException`, raise `RuntimeError` (already handled by summarize method which falls back to abstract).

**Test:** `tests/test_error_handling.py`
- Mock arXiv to raise `ConnectionError` for one category. Run fetcher — assert it returns papers from other categories, doesn't crash.
- Mock LLM to raise `TimeoutError` for one paper in ranker. Run rerank — assert that paper gets score 0 and others are scored normally.
- Mock SMTP to raise `SMTPAuthenticationError`. Assert `EmailSender.send` raises an exception (not silently swallowed).

---

### Step 14.2 — Create `.env.example` and email template

Create `templates/email_template.md` (a reference, not used programmatically — the actual template is built in code):

```markdown
# Daily Paper Report - {date}

## General Report
### Today's Overview
- **{total_count}** new papers collected
- {category_breakdown}

### Trending Topics
{trending_topics}

### Highlight Papers
{highlight_papers}

---

## Specific Report (Based on Your Interests)
{specific_content}

---

## Related Papers
{related_papers}
```

Verify `.env.example` has all required keys with placeholder values. Verify `.gitignore` excludes `.env` and `data/`.

**Test:**
- Assert `.env.example` file exists and contains `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `EMAIL_USERNAME`, `EMAIL_PASSWORD`.
- Assert `.gitignore` contains `.env` and `data/`.

---

## Summary of Implementation Order

| Phase | Component | Files | Depends On |
|-------|-----------|-------|------------|
| 0 | Scaffolding + Config + Logging | `pyproject.toml`, `requirements.txt`, `config/`, `src/config.py` (incl. `get_project_root`, `setup_logging`) | — |
| 1 | Database Store | `src/store/database.py` (incl. `get_papers_by_date_with_embeddings`) | Phase 0 |
| 2 | LLM Providers | `src/llm/base.py`, `*_provider.py` | Phase 0 |
| 3 | ArXiv Fetcher | `src/fetcher/arxiv_fetcher.py` (with Python-side date filtering) | Phase 0 |
| 4 | Embedder | `src/matcher/embedder.py` | Phase 1 |
| 5 | Interest Manager | `src/interest/manager.py` (with auto-fetch abstract from DB/arXiv) | Phase 1, 3, 4 |
| 6 | LLM Re-ranker | `src/matcher/ranker.py` (concurrent scoring via asyncio.gather + Semaphore) | Phase 2 |
| 7 | Report Generator | `src/report/generator.py` | Phase 2 |
| 8 | Email Sender | `src/email/sender.py` | Phase 0 |
| 9 | Paper Summarizer | `src/summarizer/paper_summarizer.py` | Phase 1, 2 |
| 10 | Pipeline | `src/pipeline.py` (matches today's papers only) | Phase 1–9 |
| 11 | Scheduler + CLI | `src/scheduler/`, `src/main.py`, `scripts/` | Phase 10 |
| 12 | Streamlit GUI | `gui/` (with `@st.cache_resource`, AppTest automated tests) | Phase 1–9 |
| 13 | Integration Tests | `tests/test_integration.py` | Phase 0–12 |
| 14 | Polish | Error handling, templates (logging already in Phase 0) | Phase 0–13 |

## Key Design Decisions

| Decision | Chosen Approach | Rationale |
|----------|----------------|-----------|
| Matching scope | Today's papers only | Simpler, faster, focused on daily discovery |
| Paper interest embedding | Auto-fetch abstract from DB → arXiv → fallback to ID | Better embedding quality without user burden |
| LLM re-ranking concurrency | `asyncio.gather` + `Semaphore(5)` | 5-10x faster than sequential for 50 candidates |
| Path resolution | Project-root relative via `get_project_root()` | Works regardless of CWD |
| ArXiv date filtering | Python-side filter after fetch | arXiv API sorts but doesn't filter by date |
| Streamlit caching | `@st.cache_resource` for Store, Embedder, Config | Avoids reloading 80MB model on every rerun |
| GUI testing | Streamlit AppTest automated tests | Catches rendering regressions automatically |
| Logging | Phase 0 setup, used from the start | Consistent observability throughout development |
