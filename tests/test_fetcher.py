from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.fetcher.arxiv_fetcher import ArxivFetcher


@pytest.fixture
def config():
    return {
        "arxiv": {
            "categories": ["cs.AI", "cs.CL"],
            "max_results_per_category": 100,
        }
    }


@pytest.fixture
def fetcher(config):
    return ArxivFetcher(config)


def _make_mock_result(
    arxiv_id: str,
    title: str,
    abstract: str,
    categories: list[str],
    published: date,
    authors: list[str] | None = None,
):
    """Create a mock arxiv.Result-like object."""
    result = MagicMock()
    result.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
    result.title = title
    result.summary = abstract
    result.categories = categories
    result.published = datetime.combine(published, datetime.min.time(), tzinfo=timezone.utc)
    result.pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"

    if authors is None:
        authors = ["Author One", "Author Two"]
    result.authors = [MagicMock(name=a) for a in authors]
    for mock_author, name in zip(result.authors, authors):
        mock_author.name = name

    return result


class TestFetchToday:
    @pytest.mark.asyncio
    async def test_filters_old_papers(self, fetcher):
        """Papers older than cutoff_days are filtered out."""
        today = date.today()
        old_date = today - timedelta(days=10)

        mock_results = [
            _make_mock_result("2501.00001", "Recent Paper 1", "Abstract 1", ["cs.AI"], today),
            _make_mock_result("2501.00002", "Recent Paper 2", "Abstract 2", ["cs.AI"], today),
            _make_mock_result("2501.00003", "Recent Paper 3", "Abstract 3", ["cs.AI"], today),
            _make_mock_result("2501.00004", "Old Paper 1", "Old abstract 1", ["cs.AI"], old_date),
            _make_mock_result("2501.00005", "Old Paper 2", "Old abstract 2", ["cs.AI"], old_date),
        ]

        mock_client = MagicMock()
        mock_client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Client", return_value=mock_client):
            papers = await fetcher.fetch_today()

        # 3 recent papers per category * 2 categories, but deduplicated = 3
        assert len(papers) == 3
        for paper in papers:
            assert "Old" not in paper["title"]

    @pytest.mark.asyncio
    async def test_cutoff_days_parameter(self, fetcher):
        """With a larger cutoff_days, older papers should be included."""
        today = date.today()
        old_date = today - timedelta(days=10)

        mock_results = [
            _make_mock_result("2501.00001", "Recent Paper", "Abstract", ["cs.AI"], today),
            _make_mock_result("2501.00002", "Old Paper", "Old abstract", ["cs.AI"], old_date),
        ]

        mock_client = MagicMock()
        mock_client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Client", return_value=mock_client):
            papers = await fetcher.fetch_today(cutoff_days=15)

        # Both papers are within 15-day cutoff, but deduped across 2 categories = 2
        assert len(papers) == 2

    @pytest.mark.asyncio
    async def test_deduplication_across_categories(self, fetcher):
        """Papers appearing in multiple categories are deduplicated."""
        today = date.today()

        # Same paper appears in both categories
        mock_results = [
            _make_mock_result(
                "2501.00001",
                "Shared Paper",
                "Shared abstract",
                ["cs.AI", "cs.CL"],
                today,
            ),
            _make_mock_result("2501.00002", "Unique Paper", "Unique abstract", ["cs.AI"], today),
        ]

        mock_client = MagicMock()
        mock_client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Client", return_value=mock_client):
            papers = await fetcher.fetch_today()

        # 2 unique papers, even though fetched from 2 categories
        arxiv_ids = [p["arxiv_id"] for p in papers]
        assert len(arxiv_ids) == len(set(arxiv_ids))
        assert len(papers) == 2


class TestArxivIdVersionStripping:
    @pytest.mark.asyncio
    async def test_version_stripped_from_id(self, fetcher):
        """Version suffix (e.g., v2) should be stripped from arxiv_id."""
        today = date.today()

        mock_results = [
            _make_mock_result(
                "2501.12345v2",
                "Versioned Paper",
                "Abstract",
                ["cs.AI"],
                today,
            ),
        ]

        mock_client = MagicMock()
        mock_client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Client", return_value=mock_client):
            papers = await fetcher.fetch_today()

        assert papers[0]["arxiv_id"] == "2501.12345"

    @pytest.mark.asyncio
    async def test_id_without_version_unchanged(self, fetcher):
        """IDs without version suffix remain unchanged."""
        today = date.today()

        mock_results = [
            _make_mock_result(
                "2501.12345",
                "No Version Paper",
                "Abstract",
                ["cs.AI"],
                today,
            ),
        ]

        mock_client = MagicMock()
        mock_client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Client", return_value=mock_client):
            papers = await fetcher.fetch_today()

        assert papers[0]["arxiv_id"] == "2501.12345"


class TestPaperFieldExtraction:
    @pytest.mark.asyncio
    async def test_paper_dict_fields(self, fetcher):
        """Verify all expected fields are present and correctly extracted."""
        today = date.today()

        mock_results = [
            _make_mock_result(
                "2501.99999v1",
                "Test\nPaper\nTitle",
                "Test\nabstract\ncontent",
                ["cs.AI", "cs.LG"],
                today,
                authors=["Alice Smith", "Bob Jones"],
            ),
        ]

        mock_client = MagicMock()
        mock_client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Client", return_value=mock_client):
            papers = await fetcher.fetch_today()

        paper = papers[0]
        assert paper["arxiv_id"] == "2501.99999"
        assert paper["title"] == "Test Paper Title"  # newlines stripped
        assert paper["abstract"] == "Test abstract content"  # newlines stripped
        assert paper["authors"] == ["Alice Smith", "Bob Jones"]
        assert paper["categories"] == ["cs.AI", "cs.LG"]
        assert paper["published_date"] == today.isoformat()
        assert "pdf" in paper["pdf_url"]
        assert paper["ar5iv_url"] == "https://ar5iv.labs.arxiv.org/html/2501.99999"


class TestDeduplicate:
    def test_deduplicate_keeps_first(self, fetcher):
        """_deduplicate keeps the first occurrence of each arxiv_id."""
        papers = [
            {"arxiv_id": "001", "title": "First"},
            {"arxiv_id": "002", "title": "Second"},
            {"arxiv_id": "001", "title": "Duplicate of First"},
            {"arxiv_id": "003", "title": "Third"},
        ]
        result = fetcher._deduplicate(papers)
        assert len(result) == 3
        assert result[0]["title"] == "First"
        assert result[1]["title"] == "Second"
        assert result[2]["title"] == "Third"

    def test_deduplicate_empty(self, fetcher):
        """_deduplicate handles empty list."""
        assert fetcher._deduplicate([]) == []

    def test_deduplicate_no_duplicates(self, fetcher):
        """_deduplicate returns all items when there are no duplicates."""
        papers = [
            {"arxiv_id": "001", "title": "A"},
            {"arxiv_id": "002", "title": "B"},
        ]
        result = fetcher._deduplicate(papers)
        assert len(result) == 2


class TestFetchCategory:
    def test_fetch_category_date_filter(self, fetcher):
        """_fetch_category filters out papers before cutoff_date."""
        today = date.today()
        cutoff = today - timedelta(days=2)
        old_date = today - timedelta(days=5)

        mock_results = [
            _make_mock_result("001", "Recent", "Abstract", ["cs.AI"], today),
            _make_mock_result("002", "Old", "Abstract", ["cs.AI"], old_date),
        ]

        mock_client = MagicMock()
        mock_client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Client", return_value=mock_client):
            with patch("src.fetcher.arxiv_fetcher.arxiv.Search"):
                papers = fetcher._fetch_category("cs.AI", cutoff)

        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "001"

    def test_fetch_category_all_filtered(self, fetcher):
        """_fetch_category returns empty list if all papers are too old."""
        cutoff = date.today()

        mock_results = [
            _make_mock_result(
                "001", "Old", "Abstract", ["cs.AI"], date.today() - timedelta(days=10)
            ),
        ]

        mock_client = MagicMock()
        mock_client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Client", return_value=mock_client):
            with patch("src.fetcher.arxiv_fetcher.arxiv.Search"):
                papers = fetcher._fetch_category("cs.AI", cutoff)

        assert papers == []
