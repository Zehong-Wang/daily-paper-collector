"""Tests for Phase 14: Error handling and hardening."""

import asyncio
import smtplib
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.config import get_project_root
from src.email.sender import EmailSender
from src.fetcher.arxiv_fetcher import ArxivFetcher
from src.llm.base import LLMProvider
from src.matcher.ranker import LLMRanker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_result(
    arxiv_id: str,
    title: str,
    abstract: str,
    categories: list[str],
    published: date,
):
    """Create a mock arxiv.Result-like object."""
    result = MagicMock()
    result.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
    result.title = title
    result.summary = abstract
    result.categories = categories
    result.published = datetime.combine(published, datetime.min.time(), tzinfo=timezone.utc)
    result.pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"
    result.authors = [MagicMock(name="Author One")]
    result.authors[0].name = "Author One"
    return result


def _make_candidate(paper_id: int, title: str, abstract: str = "Some abstract") -> dict:
    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "arxiv_id": f"2501.{paper_id:05d}",
        "authors": ["Author A"],
        "categories": ["cs.AI"],
        "pdf_url": f"https://arxiv.org/pdf/2501.{paper_id:05d}",
        "embedding_score": 0.5 + paper_id * 0.01,
    }


def _make_email_config():
    return {
        "email": {
            "enabled": True,
            "smtp": {
                "host": "smtp.gmail.com",
                "port": 587,
                "username_env": "EMAIL_USERNAME",
                "password_env": "EMAIL_PASSWORD",
            },
            "from": "sender@test.com",
            "to": ["recipient@test.com"],
            "subject_prefix": "[Daily Papers]",
        }
    }


# ---------------------------------------------------------------------------
# Mock LLM Providers
# ---------------------------------------------------------------------------


class FixedScoreLLM(LLMProvider):
    """Returns a fixed score for every paper."""

    def __init__(self, score: float = 8.5, reason: str = "Relevant"):
        self.call_count = 0

    async def complete(self, prompt: str, system: str = "") -> str:
        return "Mock response"

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        self.call_count += 1
        return {"score": 8.5, "reason": "Relevant"}


class TimeoutOnFirstLLM(LLMProvider):
    """Raises TimeoutError on the first call, then returns normal scores."""

    def __init__(self):
        self.call_count = 0

    async def complete(self, prompt: str, system: str = "") -> str:
        return "Mock response"

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        self.call_count += 1
        if self.call_count == 1:
            raise TimeoutError("LLM request timed out")
        return {"score": 7.0, "reason": "Normal score"}


# ---------------------------------------------------------------------------
# TestArxivFetcherErrorHandling
# ---------------------------------------------------------------------------


class TestArxivFetcherErrorHandling:
    """Test that ArxivFetcher handles errors gracefully per category."""

    @pytest.fixture
    def config(self):
        return {
            "arxiv": {
                "categories": ["cs.AI", "cs.CL"],
                "max_results_per_category": 100,
                "cutoff_days": 1,
                "page_size": 100,
            }
        }

    @pytest.fixture
    def fetcher(self, config):
        with patch("src.fetcher.arxiv_fetcher.arxiv.Client"):
            f = ArxivFetcher(config)
        f.client = MagicMock()
        return f

    def test_rss_error_one_category_returns_papers_from_others(self, fetcher):
        """If one category's RSS fails, fetcher still returns papers from other categories."""
        today = date.today().isoformat()

        # Build a mock feed that succeeds for cs.CL
        good_feed = MagicMock()
        good_feed.bozo = False
        entry1 = MagicMock()
        entry1.get = lambda k, d="": {
            "id": "oai:arXiv.org:2501.00001v1",
            "summary": "Abstract: Paper A abstract.",
            "title": "Paper A",
            "author": "Author A",
            "tags": [{"term": "cs.CL"}],
        }.get(k, d)
        entry1.arxiv_announce_type = "new"
        entry2 = MagicMock()
        entry2.get = lambda k, d="": {
            "id": "oai:arXiv.org:2501.00002v1",
            "summary": "Abstract: Paper B abstract.",
            "title": "Paper B",
            "author": "Author B",
            "tags": [{"term": "cs.CL"}],
        }.get(k, d)
        entry2.arxiv_announce_type = "new"
        good_feed.entries = [entry1, entry2]

        call_count = 0

        def mock_parse(url):
            nonlocal call_count
            call_count += 1
            if "cs.AI" in url:
                raise ConnectionError("Network unreachable")
            return good_feed

        with patch("src.fetcher.arxiv_fetcher.feedparser.parse", side_effect=mock_parse):
            papers = asyncio.run(fetcher.fetch_today())

        # Should have papers from cs.CL, not crash due to cs.AI failure
        assert len(papers) == 2
        assert papers[0]["arxiv_id"] == "2501.00001"
        assert papers[1]["arxiv_id"] == "2501.00002"

    def test_all_categories_fail_returns_empty(self, fetcher):
        """If all categories fail via RSS and REST fallback also empty, returns empty list."""
        with (
            patch("src.fetcher.arxiv_fetcher.feedparser.parse", side_effect=ConnectionError("fail")),
            patch.object(fetcher, "_fetch_via_rest_api", return_value=[]),
        ):
            papers = asyncio.run(fetcher.fetch_today())

        assert papers == []

    def test_generic_exception_handled(self, fetcher):
        """Any Exception type is caught in RSS fetching."""
        with (
            patch("src.fetcher.arxiv_fetcher.feedparser.parse", side_effect=RuntimeError("Unexpected")),
            patch.object(fetcher, "_fetch_via_rest_api", return_value=[]),
        ):
            papers = asyncio.run(fetcher.fetch_today())

        assert papers == []


# ---------------------------------------------------------------------------
# TestLLMRankerErrorHandling
# ---------------------------------------------------------------------------


class TestLLMRankerErrorHandling:
    """Test that LLMRanker handles scoring failures gracefully."""

    def test_timeout_on_one_paper_gives_score_zero(self):
        """If LLM raises TimeoutError for one paper, that paper gets score 0, others scored normally."""
        llm = TimeoutOnFirstLLM()
        ranker = LLMRanker(llm, {"matching": {"llm_top_k": 5}})

        candidates = [
            _make_candidate(1, "Paper One"),
            _make_candidate(2, "Paper Two"),
            _make_candidate(3, "Paper Three"),
        ]
        interests = [{"type": "keyword", "value": "machine learning"}]

        # Use max_concurrent=1 to ensure deterministic ordering of calls
        results = asyncio.run(ranker.rerank(candidates, interests, max_concurrent=1))

        assert len(results) == 3
        # First candidate should have score 0 (TimeoutError)
        failed = [r for r in results if r["llm_score"] == 0]
        assert len(failed) == 1
        assert "failed" in failed[0]["llm_reason"].lower()

        # Other two should have normal scores
        scored = [r for r in results if r["llm_score"] > 0]
        assert len(scored) == 2
        for r in scored:
            assert r["llm_score"] == 7.0

    def test_all_scoring_failures_returns_all_with_zero(self):
        """If all LLM calls fail, all papers get score 0 (no crash)."""

        class AlwaysFailLLM(LLMProvider):
            async def complete(self, prompt, system=""):
                return ""

            async def complete_json(self, prompt, system=""):
                raise Exception("Total failure")

        ranker = LLMRanker(AlwaysFailLLM(), {"matching": {"llm_top_k": 5}})
        candidates = [_make_candidate(i, f"Paper {i}") for i in range(3)]
        interests = [{"type": "keyword", "value": "test"}]

        results = asyncio.run(ranker.rerank(candidates, interests))

        assert len(results) == 3
        for r in results:
            assert r["llm_score"] == 0
            assert "failed" in r["llm_reason"].lower()


# ---------------------------------------------------------------------------
# TestEmailSenderErrorHandling
# ---------------------------------------------------------------------------


class TestEmailSenderErrorHandling:
    """Test that EmailSender handles SMTP errors correctly."""

    @pytest.fixture
    def sender(self, monkeypatch):
        monkeypatch.setenv("EMAIL_USERNAME", "testuser")
        monkeypatch.setenv("EMAIL_PASSWORD", "testpass")
        return EmailSender(_make_email_config())

    def test_smtp_auth_error_raises(self, sender):
        """SMTPAuthenticationError is logged and re-raised."""
        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")

        with patch("src.email.sender.smtplib.SMTP", return_value=mock_server):
            with pytest.raises(smtplib.SMTPAuthenticationError):
                asyncio.run(sender.send("# General", "## Specific", [], "2025-01-15"))

    def test_smtp_connection_error_raises(self, sender):
        """SMTPConnectError is logged and re-raised."""
        with patch(
            "src.email.sender.smtplib.SMTP",
            side_effect=smtplib.SMTPConnectError(421, b"Service unavailable"),
        ):
            with pytest.raises(smtplib.SMTPConnectError):
                asyncio.run(sender.send("# General", "## Specific", [], "2025-01-15"))

    def test_smtp_generic_error_raises(self, sender):
        """Generic SMTPException is logged and re-raised."""
        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)
        mock_server.send_message.side_effect = smtplib.SMTPException("Unexpected SMTP error")

        with patch("src.email.sender.smtplib.SMTP", return_value=mock_server):
            with pytest.raises(smtplib.SMTPException):
                asyncio.run(sender.send("# General", "## Specific", [], "2025-01-15"))


# ---------------------------------------------------------------------------
# TestFileVerification (Step 14.2)
# ---------------------------------------------------------------------------


class TestFileVerification:
    """Verify .env.example, .gitignore, and email template exist with correct content."""

    @pytest.fixture
    def project_root(self):
        return get_project_root()

    def test_env_example_exists(self, project_root):
        env_file = project_root / ".env.example"
        assert env_file.exists(), ".env.example file does not exist"

    def test_env_example_contains_required_keys(self, project_root):
        content = (project_root / ".env.example").read_text()
        assert "OPENAI_API_KEY" in content
        assert "ANTHROPIC_API_KEY" in content
        assert "EMAIL_USERNAME" in content
        assert "EMAIL_PASSWORD" in content

    def test_gitignore_exists(self, project_root):
        gitignore = project_root / ".gitignore"
        assert gitignore.exists(), ".gitignore file does not exist"

    def test_gitignore_contains_env(self, project_root):
        content = (project_root / ".gitignore").read_text()
        assert ".env" in content

    def test_gitignore_contains_data(self, project_root):
        content = (project_root / ".gitignore").read_text()
        assert "data/" in content

    def test_email_template_exists(self, project_root):
        template = project_root / "templates" / "email_template.md"
        assert template.exists(), "templates/email_template.md does not exist"

    def test_email_template_has_placeholders(self, project_root):
        content = (project_root / "templates" / "email_template.md").read_text()
        assert "{date}" in content
        assert "{total_count}" in content
        assert "{category_breakdown}" in content
        assert "{trending_topics}" in content
        assert "{highlight_papers}" in content
        assert "{theme_synthesis}" in content
        assert "{paper_details}" in content

    def test_email_template_has_sections(self, project_root):
        content = (project_root / "templates" / "email_template.md").read_text()
        assert "## General Report" in content
        assert "## Specific Report" in content
        assert "## Paper Details" in content
