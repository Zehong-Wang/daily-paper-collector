from __future__ import annotations

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
            "cutoff_days": 1,
            "page_size": 100,
        }
    }


@pytest.fixture
def fetcher(config):
    with patch("src.fetcher.arxiv_fetcher.arxiv.Client"):
        f = ArxivFetcher(config)
    f.client = MagicMock()
    return f


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


def _make_rss_papers(papers_data: list[dict]) -> list[dict]:
    """Create paper dicts as returned by _fetch_category_rss."""
    today = date.today().isoformat()
    result = []
    for p in papers_data:
        result.append(
            {
                "arxiv_id": p.get("arxiv_id", "2501.00001"),
                "title": p.get("title", "Test Paper"),
                "authors": p.get("authors", ["Author One", "Author Two"]),
                "abstract": p.get("abstract", "Test abstract"),
                "categories": p.get("categories", ["cs.AI"]),
                "published_date": p.get("published_date", today),
                "pdf_url": f"https://arxiv.org/pdf/{p.get('arxiv_id', '2501.00001')}.pdf",
                "ar5iv_url": f"https://ar5iv.labs.arxiv.org/html/{p.get('arxiv_id', '2501.00001')}",
            }
        )
    return result


class TestFetchToday:
    @pytest.mark.asyncio
    async def test_rss_returns_papers(self, fetcher):
        """fetch_today uses RSS and returns papers."""
        papers_ai = _make_rss_papers([
            {"arxiv_id": "2501.00001", "title": "Paper A"},
            {"arxiv_id": "2501.00002", "title": "Paper B"},
        ])
        papers_cl = _make_rss_papers([
            {"arxiv_id": "2501.00003", "title": "Paper C"},
        ])

        def mock_rss(category):
            if category == "cs.AI":
                return papers_ai
            return papers_cl

        with patch.object(fetcher, "_fetch_category_rss", side_effect=mock_rss):
            papers = await fetcher.fetch_today()

        assert len(papers) == 3

    @pytest.mark.asyncio
    async def test_deduplication_across_categories(self, fetcher):
        """Papers appearing in multiple categories are deduplicated."""
        papers_ai = _make_rss_papers([
            {"arxiv_id": "2501.00001", "title": "Shared Paper"},
            {"arxiv_id": "2501.00002", "title": "Unique Paper"},
        ])
        papers_cl = _make_rss_papers([
            {"arxiv_id": "2501.00001", "title": "Shared Paper"},
        ])

        def mock_rss(category):
            if category == "cs.AI":
                return papers_ai
            return papers_cl

        with patch.object(fetcher, "_fetch_category_rss", side_effect=mock_rss):
            papers = await fetcher.fetch_today()

        arxiv_ids = [p["arxiv_id"] for p in papers]
        assert len(arxiv_ids) == len(set(arxiv_ids))
        assert len(papers) == 2

    @pytest.mark.asyncio
    async def test_rss_empty_falls_back_to_rest(self, fetcher):
        """When RSS returns empty, falls back to REST API."""
        rest_papers = _make_rss_papers([
            {"arxiv_id": "2501.00001", "title": "REST Paper"},
        ])

        with (
            patch.object(fetcher, "_fetch_category_rss", return_value=[]),
            patch.object(fetcher, "_fetch_via_rest_api", return_value=rest_papers) as mock_rest,
        ):
            papers = await fetcher.fetch_today()

        mock_rest.assert_awaited_once()
        assert len(papers) == 1
        assert papers[0]["title"] == "REST Paper"

    @pytest.mark.asyncio
    async def test_rss_nonempty_skips_rest(self, fetcher):
        """When RSS returns papers, REST API is NOT called."""
        rss_papers = _make_rss_papers([
            {"arxiv_id": "2501.00001", "title": "RSS Paper"},
        ])

        with (
            patch.object(fetcher, "_fetch_category_rss", return_value=rss_papers),
            patch.object(fetcher, "_fetch_via_rest_api") as mock_rest,
        ):
            papers = await fetcher.fetch_today()

        mock_rest.assert_not_awaited()
        assert len(papers) == 1


class TestRSSFetching:
    def test_extract_abstract_with_prefix(self, fetcher):
        """Extracts abstract text after 'Abstract:' prefix."""
        summary = "arXiv:2602.17676v1 Announce Type: new\nAbstract: The rapid deployment of LLMs."
        result = fetcher._extract_abstract_from_rss(summary)
        assert result == "The rapid deployment of LLMs."

    def test_extract_abstract_with_html(self, fetcher):
        """Strips HTML tags from the summary."""
        summary = "<p>arXiv:2602.17676v1 Announce Type: new\nAbstract: Some text here.</p>"
        result = fetcher._extract_abstract_from_rss(summary)
        assert result == "Some text here."

    def test_extract_abstract_no_prefix(self, fetcher):
        """Returns full cleaned text when no 'Abstract:' prefix found."""
        summary = "Just a plain summary text."
        result = fetcher._extract_abstract_from_rss(summary)
        assert result == "Just a plain summary text."

    def test_fetch_category_rss_error_returns_empty(self, fetcher):
        """RSS parse failure returns empty list."""
        with patch("src.fetcher.arxiv_fetcher.feedparser.parse", side_effect=Exception("Network error")):
            papers = fetcher._fetch_category_rss("cs.AI")
        assert papers == []

    def test_fetch_category_rss_skips_replace(self, fetcher):
        """RSS entries with announce_type 'replace' are skipped."""
        mock_feed = MagicMock()
        mock_feed.bozo = False
        new_entry = MagicMock()
        new_entry.get = lambda k, d="": {
            "id": "oai:arXiv.org:2501.00001v1",
            "summary": "arXiv:2501.00001v1 Announce Type: new\nAbstract: New paper.",
            "title": "New Paper",
            "author": "Alice, Bob",
            "tags": [{"term": "cs.AI"}],
        }.get(k, d)
        new_entry.arxiv_announce_type = "new"

        replace_entry = MagicMock()
        replace_entry.get = lambda k, d="": {
            "id": "oai:arXiv.org:2501.00002v2",
            "summary": "arXiv:2501.00002v2 Announce Type: replace\nAbstract: Updated paper.",
            "title": "Updated Paper",
            "author": "Charlie",
            "tags": [{"term": "cs.AI"}],
        }.get(k, d)
        replace_entry.arxiv_announce_type = "replace"

        mock_feed.entries = [new_entry, replace_entry]

        with patch("src.fetcher.arxiv_fetcher.feedparser.parse", return_value=mock_feed):
            papers = fetcher._fetch_category_rss("cs.AI")

        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "2501.00001"


class TestBuildDateQuery:
    def test_date_query_format(self, fetcher):
        """Verify the submittedDate query string format."""
        query = fetcher._build_date_query("cs.AI", date(2026, 2, 19), date(2026, 2, 20))
        assert query == "cat:cs.AI AND submittedDate:[202602190000 TO 202602202359]"

    def test_same_day_range(self, fetcher):
        """Start and end on the same day produces a valid query."""
        query = fetcher._build_date_query("cs.LG", date(2026, 2, 20), date(2026, 2, 20))
        assert query == "cat:cs.LG AND submittedDate:[202602200000 TO 202602202359]"

    def test_multi_day_range(self, fetcher):
        """Multi-day range produces correct boundaries."""
        query = fetcher._build_date_query("cs.CL", date(2026, 2, 15), date(2026, 2, 20))
        assert "202602150000" in query
        assert "202602202359" in query
        assert query.startswith("cat:cs.CL AND submittedDate:")


class TestFetchCategoryRest:
    def test_fetch_category_rest_date_filter(self, fetcher):
        """_fetch_category_rest filters out papers before start_date."""
        today = date.today()
        start_date = today - timedelta(days=2)
        end_date = today
        old_date = today - timedelta(days=5)

        mock_results = [
            _make_mock_result("001", "Recent", "Abstract", ["cs.AI"], today),
            _make_mock_result("002", "Old", "Abstract", ["cs.AI"], old_date),
        ]

        fetcher.client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Search"):
            papers = fetcher._fetch_category_rest("cs.AI", start_date, end_date)

        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "001"

    def test_fetch_category_rest_all_filtered(self, fetcher):
        """_fetch_category_rest returns empty list if all papers are too old."""
        today = date.today()

        mock_results = [
            _make_mock_result(
                "001", "Old", "Abstract", ["cs.AI"], date.today() - timedelta(days=10)
            ),
        ]

        fetcher.client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Search"):
            papers = fetcher._fetch_category_rest("cs.AI", today, today)

        assert papers == []

    def test_fetch_category_rest_uses_date_query(self, fetcher):
        """_fetch_category_rest passes submittedDate query to arxiv.Search."""
        today = date.today()
        start_date = today - timedelta(days=1)
        end_date = today

        fetcher.client.results.return_value = []

        with patch("src.fetcher.arxiv_fetcher.arxiv.Search") as MockSearch:
            fetcher._fetch_category_rest("cs.AI", start_date, end_date)

            call_args = MockSearch.call_args
            query = call_args[1]["query"] if "query" in call_args[1] else call_args[0][0]
            assert "cat:cs.AI" in query
            assert "submittedDate:" in query

    def test_fetch_category_rest_error_returns_empty(self, fetcher):
        """REST API failure returns empty list."""
        today = date.today()
        fetcher.client.results.side_effect = ConnectionError("Network error")

        with patch("src.fetcher.arxiv_fetcher.arxiv.Search"):
            papers = fetcher._fetch_category_rest("cs.AI", today, today)

        assert papers == []


class TestArxivIdVersionStripping:
    def test_version_stripped_in_rest(self, fetcher):
        """Version suffix (e.g., v2) should be stripped in REST results."""
        today = date.today()
        mock_results = [
            _make_mock_result("2501.12345v2", "Versioned Paper", "Abstract", ["cs.AI"], today),
        ]
        fetcher.client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Search"):
            papers = fetcher._fetch_category_rest("cs.AI", today - timedelta(days=1), today)

        assert papers[0]["arxiv_id"] == "2501.12345"

    def test_version_stripped_in_rss(self, fetcher):
        """Version suffix should be stripped from RSS entry IDs."""
        mock_feed = MagicMock()
        mock_feed.bozo = False
        entry = MagicMock()
        entry.get = lambda k, d="": {
            "id": "oai:arXiv.org:2501.12345v2",
            "summary": "Abstract: Test paper.",
            "title": "Test Paper",
            "author": "Alice",
            "tags": [{"term": "cs.AI"}],
        }.get(k, d)
        entry.arxiv_announce_type = "new"
        mock_feed.entries = [entry]

        with patch("src.fetcher.arxiv_fetcher.feedparser.parse", return_value=mock_feed):
            papers = fetcher._fetch_category_rss("cs.AI")

        assert papers[0]["arxiv_id"] == "2501.12345"


class TestPaperFieldExtraction:
    def test_paper_dict_fields_from_rest(self, fetcher):
        """Verify all expected fields are present from REST API."""
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

        fetcher.client.results.return_value = mock_results

        with patch("src.fetcher.arxiv_fetcher.arxiv.Search"):
            papers = fetcher._fetch_category_rest("cs.AI", today - timedelta(days=1), today)

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
