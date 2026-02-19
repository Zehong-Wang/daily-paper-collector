"""Automated GUI tests using Streamlit AppTest framework (Step 12.7).

These tests verify that all GUI pages render without exceptions and display
appropriate empty-state messages. Uses a temp SQLite DB via patched config.
Embedder is safe because it lazy-loads the model (no download at init).

Key testing considerations:
- @st.cache_resource caches are GLOBAL in the Streamlit runtime and persist
  across AppTest instances within the same pytest process. We must clear them
  before each test via an autouse fixture.
- The first AppTest.run() is slow (~10s) because gui/app.py imports
  sentence_transformers (which imports torch). We use timeout=30.
"""

import streamlit as st
import pytest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.store.database import PaperStore

# Generous timeout for AppTest.run(). The first run imports sentence_transformers
# (which imports torch), taking ~10s. Subsequent runs are fast (<1s) because
# the modules are already cached in sys.modules.
_RUN_TIMEOUT = 30


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Clear all @st.cache_resource caches before each test.

    Without this, get_config()/get_store()/get_embedder() cached by one test
    leak into subsequent tests, causing them to use the wrong temp DB path.
    """
    st.cache_resource.clear()


def _make_test_config(db_path: str) -> dict:
    """Build a minimal config dict pointing to the given temp DB path."""
    return {
        "arxiv": {"categories": ["cs.AI", "cs.CL"], "max_results_per_category": 10},
        "matching": {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_top_n": 50,
            "llm_top_k": 10,
            "similarity_threshold": 0.3,
        },
        "llm": {
            "provider": "openai",
            "openai": {"model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
            "claude": {"model": "claude-sonnet-4-5-20250929", "api_key_env": "ANTHROPIC_API_KEY"},
            "claude_code": {"cli_path": "claude", "model": "sonnet"},
        },
        "email": {
            "enabled": False,
            "smtp": {
                "host": "smtp.gmail.com",
                "port": 587,
                "username_env": "EMAIL_USERNAME",
                "password_env": "EMAIL_PASSWORD",
            },
            "from": "test@test.com",
            "to": ["test@test.com"],
            "subject_prefix": "[Test]",
        },
        "scheduler": {"enabled": False, "cron": "0 8 * * *"},
        "database": {"path": db_path},
        "gui": {"port": 8501},
    }


@pytest.fixture
def temp_db_path(tmp_path):
    """Return a temp DB path with initialized schema."""
    db_path = str(tmp_path / "test_gui.db")
    PaperStore(db_path)  # Initialize schema
    return db_path


def _run_app_on_page(db_path: str, page: str = "Dashboard"):
    """Create and run an AppTest instance navigated to the given page.

    Patches src.config.load_config so the app uses a temp DB.
    Returns the AppTest instance after running.
    """
    config = _make_test_config(db_path)

    with patch("src.config.load_config", return_value=config):
        at = AppTest.from_file("gui/app.py")
        at.run(timeout=_RUN_TIMEOUT)
        if page != "Dashboard":
            at.sidebar.radio[0].set_value(page)
            at.run(timeout=_RUN_TIMEOUT)
    return at


# --- Dashboard tests ---


class TestDashboardPage:
    def test_dashboard_renders(self, temp_db_path):
        """Dashboard page renders without errors on an empty DB."""
        at = _run_app_on_page(temp_db_path)
        assert not at.exception

    def test_dashboard_shows_zero_metrics(self, temp_db_path):
        """Dashboard shows zero metrics for empty DB."""
        at = _run_app_on_page(temp_db_path)
        assert not at.exception
        metrics = at.metric
        assert len(metrics) == 3
        assert metrics[0].value == "0"
        assert metrics[1].value == "0"

    def test_dashboard_has_pipeline_button(self, temp_db_path):
        """Dashboard has a 'Run Pipeline Now' button."""
        at = _run_app_on_page(temp_db_path)
        assert not at.exception
        assert len(at.button) >= 1
        assert any("Run Pipeline Now" in str(b.label) for b in at.button)


# --- Papers tests ---


class TestPapersPage:
    def test_papers_page_empty_db(self, temp_db_path):
        """Papers page handles empty database gracefully."""
        at = _run_app_on_page(temp_db_path, "Papers")
        assert not at.exception
        assert any("0 papers" in str(el.value) for el in at.info)

    def test_papers_page_has_search_box(self, temp_db_path):
        """Papers page has date input and search box."""
        at = _run_app_on_page(temp_db_path, "Papers")
        assert not at.exception
        assert len(at.date_input) >= 1
        assert len(at.text_input) >= 1

    def test_papers_page_with_data(self, temp_db_path):
        """Papers page displays papers when data exists."""
        store = PaperStore(temp_db_path)
        from datetime import date

        today = date.today().isoformat()
        store.save_papers(
            [
                {
                    "arxiv_id": "2501.00001",
                    "title": "Test Paper on Transformers",
                    "authors": ["Author A", "Author B"],
                    "abstract": "This is a test abstract about transformers.",
                    "categories": ["cs.AI"],
                    "published_date": today,
                    "pdf_url": "https://arxiv.org/pdf/2501.00001",
                    "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.00001",
                }
            ]
        )

        at = _run_app_on_page(temp_db_path, "Papers")
        assert not at.exception
        assert any("1 papers" in str(el.value) for el in at.info)


# --- Interests tests ---


class TestInterestsPage:
    def test_interests_page_empty(self, temp_db_path):
        """Interests page shows 'No interests' for empty DB."""
        at = _run_app_on_page(temp_db_path, "Interests")
        assert not at.exception
        assert any("No interests" in str(el.value) for el in at.info)

    def test_interests_page_form_elements(self, temp_db_path):
        """Interests page has type selector and value input."""
        at = _run_app_on_page(temp_db_path, "Interests")
        assert not at.exception
        assert len(at.selectbox) >= 1  # Type selector
        assert len(at.text_input) >= 1  # Value input

    def test_interests_page_with_data(self, temp_db_path):
        """Interests page displays existing interests."""
        store = PaperStore(temp_db_path)
        store.save_interest("keyword", "transformers", "Neural network architectures")
        at = _run_app_on_page(temp_db_path, "Interests")
        assert not at.exception
        # Should NOT show "No interests" message
        assert not any("No interests" in str(el.value) for el in at.info)


# --- Reports tests ---


class TestReportsPage:
    def test_reports_page_empty(self, temp_db_path):
        """Reports page handles no reports gracefully."""
        at = _run_app_on_page(temp_db_path, "Reports")
        assert not at.exception
        assert any("No reports" in str(el.value) for el in at.info)

    def test_reports_page_with_data(self, temp_db_path):
        """Reports page displays report when data exists."""
        store = PaperStore(temp_db_path)
        store.save_report(
            "2025-01-15",
            "# General Report\n\nTest content",
            "# Specific Report\n\nTest specific",
            10,
            3,
        )
        at = _run_app_on_page(temp_db_path, "Reports")
        assert not at.exception
        # Should NOT show "No reports" message
        assert not any("No reports" in str(el.value) for el in at.info)
        # Should have tabs for General and Specific reports
        assert len(at.tabs) >= 1


# --- Settings tests ---


class TestSettingsPage:
    def test_settings_page_renders(self, temp_db_path):
        """Settings page renders config without errors."""
        at = _run_app_on_page(temp_db_path, "Settings")
        assert not at.exception

    def test_settings_page_has_controls(self, temp_db_path):
        """Settings page has editable controls."""
        at = _run_app_on_page(temp_db_path, "Settings")
        assert not at.exception
        # Should have text_area for categories, selectbox for provider, checkbox for email
        assert len(at.text_area) >= 1
        assert len(at.selectbox) >= 1
        assert len(at.checkbox) >= 1
