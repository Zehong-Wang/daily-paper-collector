# Feature PRD: Table-Based Paper Layout in Streamlit GUI

**Status:** Implemented
**Author:** Auto-generated
**Date:** 2026-02-21
**Priority:** Medium

---

## 1. Problem Statement

The Streamlit GUI presents papers as vertically stacked `st.expander` cards on both the **Papers** page (all daily papers) and the **Reports** page (matched papers in the Specific Report tab). Each card requires a click to expand, making it hard to scan and compare papers at a glance. When dozens of papers are listed, the card-based layout becomes unwieldy.

**Goal:** Replace card-based paper layouts with compact `st.dataframe` tables. Users can scan key metadata (title, score, categories, relevance) in a single view, with row selection to drill into full details below.

---

## 2. Feature Requirements

### FR-1: Papers Page — Table View

**Previous behavior:** Papers displayed as expandable `st.expander` cards, each containing authors, categories, published date, abstract, arXiv link, and summary buttons.

**New behavior:** Papers displayed in a `st.dataframe` table:

| Column | Content |
|--------|---------|
| Title | Paper title (text) |
| Authors | First 3 authors + "et al." if truncated |
| Category | Primary category (first in list) |
| Date | Published date |
| arXiv | Link to arXiv page (LinkColumn, display text "Open") |

Row selection (`on_select="rerun"`, `selection_mode="single-row"`) shows full paper details and summary buttons below the table.

**File:** `gui/views/papers.py`

---

### FR-2: Reports Page — Matched Papers Table

**Previous behavior:** `_render_paper_cards()` rendered expandable cards with score, embedding score, categories, full authors, full abstract, relevance reason, and arXiv link.

**New behavior:** `_render_matches_table()` renders a `st.dataframe` table:

| Column | Content |
|--------|---------|
| # | Rank number (1-based) |
| Title | Paper title (text) |
| Score | LLM score as "{N}/10" |
| Category | Primary category (first in list) |
| Relevance | Truncated `llm_reason` (first 80 chars + "...") |
| arXiv | Link to arXiv page (LinkColumn, display text "Open") |

Row selection shows full paper details (all scores, full authors, full abstract, full relevance reason) below the table. Block 1 (theme synthesis narrative) is unchanged.

**File:** `gui/views/reports.py`

---

### FR-3: Shared Table Helper Utilities

Three utility functions extracted to avoid duplication:

```python
def truncate_authors(authors: list | str, max_count: int = 3) -> str:
    """Truncate author list to max_count names + 'et al.'"""

def truncate_text(text: str, max_len: int = 80) -> str:
    """Truncate text to max_len characters + '...'"""

def get_primary_category(categories: list | str) -> str:
    """Return the first category from the list."""
```

**File:** `gui/components/table_helpers.py` (new)

---

### FR-4: Detail Panel Below Table

When a user selects a row:
- **Papers page:** Full authors, all categories, full abstract, arXiv link, Brief/Detailed Summary buttons
- **Reports page:** Score + embedding score, full authors, all categories, full abstract, full relevance reason, arXiv link

When no row is selected, shows hint: "Select a row to see details."

**Files:** `gui/views/papers.py`, `gui/views/reports.py`

---

## 3. Code Changes Summary

| File | Change | Scope |
|------|--------|-------|
| `gui/components/table_helpers.py` | New file: `truncate_authors`, `truncate_text`, `get_primary_category` | ~25 lines (new) |
| `gui/views/papers.py` | Replaced expander loop with `st.dataframe` table + `_render_detail_panel()` | Full rewrite |
| `gui/views/reports.py` | Replaced `_render_paper_cards()` with `_render_matches_table()` + `_render_match_detail()` | ~60 lines changed |
| `tests/test_table_helpers.py` | Unit tests for helper functions | ~40 lines (new) |

---

## 4. Testing Strategy

### Unit Tests (`tests/test_table_helpers.py`)

- `truncate_authors`: list <=3 -> full, >3 -> first 3 + "et al.", string passthrough, empty list
- `truncate_text`: short text unchanged, long text truncated + "...", empty string
- `get_primary_category`: list -> first item, string -> passthrough, empty list -> "unknown"

### Manual GUI Testing

1. Papers page: table renders with correct columns, sorting works, row selection shows detail panel with summary buttons
2. Reports page: synthesis narrative unchanged, matched papers table renders, row selection shows full details
3. Both pages: arXiv "Open" links work correctly

---

## 5. Dependencies

- **No new Python packages** -- `pandas` is already a Streamlit dependency
- **Streamlit >= 1.40** -- already in `requirements.txt`
- `st.dataframe` `on_select` parameter -- available since Streamlit 1.35

---

## 6. Rollback Plan

Changes are confined to 2 GUI view files and 1 new helper file. To rollback:
1. Revert `gui/views/papers.py` and `gui/views/reports.py`
2. Delete `gui/components/table_helpers.py` and `tests/test_table_helpers.py`

No database, pipeline, report generation, or email changes involved.

---

## 7. Out of Scope

- Email report format (uses Markdown -> HTML pipeline, unrelated to GUI)
- Report generation logic (`src/report/generator.py`)
- Paper summarizer logic (existing integration preserved in detail panel)
- GUI Dashboard, Interests, or Settings pages
