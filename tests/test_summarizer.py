import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.summarizer.paper_summarizer import PaperSummarizer


# --- Helpers ---


def _make_mock_store(tmp_path):
    """Create a real PaperStore with a temp DB for tests that need _get_conn."""
    from src.store.database import PaperStore

    store = PaperStore(str(tmp_path / "test.db"))
    return store


def _insert_paper(
    store,
    paper_id=1,
    arxiv_id="2501.12345",
    title="Test Paper",
    abstract="This is a test abstract about machine learning.",
    ar5iv_url="https://ar5iv.labs.arxiv.org/html/2501.12345",
):
    """Insert a paper into the store and return its id."""
    papers = store.save_papers(
        [
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": ["Author One", "Author Two"],
                "abstract": abstract,
                "categories": ["cs.AI", "cs.LG"],
                "published_date": "2025-01-15",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "ar5iv_url": ar5iv_url,
            }
        ]
    )
    return papers[0]["id"]


class MockLLMProvider:
    """Mock LLM that returns a canned summary."""

    def __init__(self, response="This is a summary of the paper."):
        self.response = response
        self.call_count = 0
        self.last_prompt = None
        self.last_system = None

    async def complete(self, prompt, system=""):
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system = system
        return self.response

    async def complete_json(self, prompt, system=""):
        return {}


# --- Test fetch_paper_text ---


class TestFetchPaperText:
    def test_extracts_paragraphs_from_article_tag(self, tmp_path):
        """Test HTML extraction from <article> tag with <p> elements."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        html = (
            "<html><body><article>"
            "<p>Introduction text.</p>"
            "<p>Method text.</p>"
            "</article></body></html>"
        )

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            text = summarizer.fetch_paper_text("https://ar5iv.labs.arxiv.org/html/2501.12345")

        assert "Introduction text." in text
        assert "Method text." in text

    def test_extracts_headings(self, tmp_path):
        """Test that h2 and h3 tags are extracted."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        html = (
            "<html><body><article>"
            "<h2>Section Title</h2>"
            "<p>Content here.</p>"
            "<h3>Subsection</h3>"
            "<p>More content.</p>"
            "</article></body></html>"
        )

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            text = summarizer.fetch_paper_text("https://example.com")

        assert "Section Title" in text
        assert "Subsection" in text
        assert "Content here." in text
        assert "More content." in text

    def test_falls_back_to_ltx_document_class(self, tmp_path):
        """Test fallback to ltx_document class when no <article> tag."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        html = '<html><body><div class="ltx_document"><p>Document content.</p></div></body></html>'

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            text = summarizer.fetch_paper_text("https://example.com")

        assert "Document content." in text

    def test_falls_back_to_ltx_page_main_class(self, tmp_path):
        """Test fallback to ltx_page_main class."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        html = (
            '<html><body><div class="ltx_page_main"><p>Main page content.</p></div></body></html>'
        )

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            text = summarizer.fetch_paper_text("https://example.com")

        assert "Main page content." in text

    def test_raises_runtime_error_on_http_failure(self, tmp_path):
        """Test that RuntimeError is raised when HTTP request fails (e.g., 404)."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_response = MagicMock()
            import requests as req

            mock_response.raise_for_status.side_effect = req.HTTPError("404 Not Found")
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError, match="Failed to fetch paper"):
                summarizer.fetch_paper_text("https://example.com/nonexistent")

    def test_raises_runtime_error_on_connection_error(self, tmp_path):
        """Test that RuntimeError is raised on connection errors."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        import requests as req

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_get.side_effect = req.ConnectionError("Connection refused")

            with pytest.raises(RuntimeError, match="Failed to fetch paper"):
                summarizer.fetch_paper_text("https://example.com")

    def test_truncates_to_15000_characters(self, tmp_path):
        """Test that output is truncated to 15000 characters."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        # Create HTML with very long content
        long_paragraph = "A" * 20000
        html = f"<html><body><article><p>{long_paragraph}</p></article></body></html>"

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            text = summarizer.fetch_paper_text("https://example.com")

        assert len(text) == 15000

    def test_skips_empty_tags(self, tmp_path):
        """Test that empty tags are skipped."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        html = "<html><body><article><p></p><p>   </p><p>Real content.</p></article></body></html>"

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            text = summarizer.fetch_paper_text("https://example.com")

        assert text == "Real content."

    def test_raises_on_navigation_shell_content(self, tmp_path):
        """ar5iv shell-like pages should be rejected instead of treated as paper content."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        html = (
            "<html><body><article>"
            "<p>Help</p><p>Search</p><p>References & Citations</p><p>Export BibTeX</p>"
            "<p>Submission history</p><p>View PDF</p><p>bookmark</p><p>Add to lists</p>"
            "</article></body></html>"
        )

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError, match="navigation shell content"):
                summarizer.fetch_paper_text("https://ar5iv.labs.arxiv.org/html/2501.12345")


class TestFetchPdfText:
    def test_extracts_text_from_pdf(self, tmp_path):
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        class _FakePage:
            def __init__(self, text):
                self._text = text

            def extract_text(self):
                return self._text

        class _FakeReader:
            def __init__(self, _fp):
                self.pages = [_FakePage("Introduction"), _FakePage("Method and Results")]

        with patch("src.summarizer.paper_summarizer.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.content = b"%PDF-fake"
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            with patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=_FakeReader)}):
                text = summarizer.fetch_pdf_text("https://arxiv.org/pdf/2501.12345")

        assert "Introduction" in text
        assert "Method and Results" in text


# --- Test summarize ---


class TestSummarize:
    @pytest.mark.asyncio
    async def test_returns_cached_summary(self, tmp_path):
        """Test that cached summary is returned without calling LLM."""
        store = _make_mock_store(tmp_path)
        paper_id = _insert_paper(store)
        # Pre-cache a summary
        store.save_summary(paper_id, "brief", "Cached summary text.", "TestProvider")

        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        result = await summarizer.summarize(paper_id, "brief")

        assert result == "Cached summary text."
        assert llm.call_count == 0  # LLM should NOT be called

    @pytest.mark.asyncio
    async def test_generates_brief_summary(self, tmp_path):
        """Test generating a brief summary when no cache exists."""
        store = _make_mock_store(tmp_path)
        paper_id = _insert_paper(store)

        llm = MockLLMProvider("Brief summary of the paper.")
        summarizer = PaperSummarizer(llm, store)

        with patch.object(summarizer, "fetch_paper_text", return_value="Full paper text here."):
            result = await summarizer.summarize(paper_id, "brief")

        assert result == "Brief summary of the paper."
        assert llm.call_count == 1
        assert "1-2 paragraphs" in llm.last_prompt
        assert "core contributions" in llm.last_prompt

    @pytest.mark.asyncio
    async def test_generates_detailed_summary(self, tmp_path):
        """Test generating a detailed summary."""
        store = _make_mock_store(tmp_path)
        paper_id = _insert_paper(store)

        llm = MockLLMProvider("Detailed structured summary.")
        summarizer = PaperSummarizer(llm, store)

        with patch.object(summarizer, "fetch_paper_text", return_value="Full paper text."):
            result = await summarizer.summarize(paper_id, "detailed")

        assert result == "Detailed structured summary."
        assert llm.call_count == 1
        assert "Motivation" in llm.last_prompt
        assert "Method" in llm.last_prompt
        assert "Experiments" in llm.last_prompt
        assert "Conclusions" in llm.last_prompt
        assert "Limitations" in llm.last_prompt

    @pytest.mark.asyncio
    async def test_saves_summary_to_cache(self, tmp_path):
        """Test that generated summary is saved to the store cache."""
        store = _make_mock_store(tmp_path)
        paper_id = _insert_paper(store)

        llm = MockLLMProvider("New summary text.")
        summarizer = PaperSummarizer(llm, store)

        with patch.object(summarizer, "fetch_paper_text", return_value="Paper text."):
            await summarizer.summarize(paper_id, "brief")

        # Verify it was cached
        cached = store.get_summary(paper_id, "brief")
        assert cached is not None
        assert cached["content"] == "New summary text."
        assert cached["llm_provider"] == "MockLLMProvider"

    @pytest.mark.asyncio
    async def test_falls_back_to_abstract_on_fetch_failure(self, tmp_path):
        """Test fallback to abstract when fetch_paper_text fails."""
        store = _make_mock_store(tmp_path)
        abstract = "This is the abstract about deep learning."
        paper_id = _insert_paper(store, abstract=abstract)

        llm = MockLLMProvider("Summary from abstract.")
        summarizer = PaperSummarizer(llm, store)

        with patch.object(summarizer, "fetch_paper_text", side_effect=RuntimeError("Fetch failed")):
            result = await summarizer.summarize(paper_id, "brief")

        assert result == "Summary from abstract."
        assert abstract in llm.last_prompt

    @pytest.mark.asyncio
    async def test_falls_back_to_pdf_when_ar5iv_is_unusable(self, tmp_path):
        """When ar5iv extraction fails (e.g., page shell), summarizer should use PDF text."""
        store = _make_mock_store(tmp_path)
        paper_id = _insert_paper(store)

        llm = MockLLMProvider("Summary from PDF.")
        summarizer = PaperSummarizer(llm, store)

        with patch.object(
            summarizer, "fetch_paper_text", side_effect=RuntimeError("navigation shell content")
        ):
            with patch.object(summarizer, "fetch_pdf_text", return_value="Full text from PDF."):
                result = await summarizer.summarize(paper_id, "brief")

        assert result == "Summary from PDF."
        assert "Full text from PDF." in llm.last_prompt

    @pytest.mark.asyncio
    async def test_raises_value_error_for_nonexistent_paper(self, tmp_path):
        """Test that ValueError is raised for a paper not in the database."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        with pytest.raises(ValueError, match="Paper with id 9999 not found"):
            await summarizer.summarize(9999, "brief")

    @pytest.mark.asyncio
    async def test_includes_paper_title_in_prompt(self, tmp_path):
        """Test that the paper title is included in the LLM prompt."""
        store = _make_mock_store(tmp_path)
        paper_id = _insert_paper(store, title="Attention Is All You Need")

        llm = MockLLMProvider("Summary.")
        summarizer = PaperSummarizer(llm, store)

        with patch.object(summarizer, "fetch_paper_text", return_value="Paper text."):
            await summarizer.summarize(paper_id, "brief")

        assert "Attention Is All You Need" in llm.last_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_is_set(self, tmp_path):
        """Test that the LLM is called with an appropriate system prompt."""
        store = _make_mock_store(tmp_path)
        paper_id = _insert_paper(store)

        llm = MockLLMProvider("Summary.")
        summarizer = PaperSummarizer(llm, store)

        with patch.object(summarizer, "fetch_paper_text", return_value="Text."):
            await summarizer.summarize(paper_id, "brief")

        assert "scientific paper summarizer" in llm.last_system.lower()

    @pytest.mark.asyncio
    async def test_brief_and_detailed_cached_separately(self, tmp_path):
        """Test that brief and detailed summaries are cached independently."""
        store = _make_mock_store(tmp_path)
        paper_id = _insert_paper(store)

        llm = MockLLMProvider("Brief summary.")
        summarizer = PaperSummarizer(llm, store)

        with patch.object(summarizer, "fetch_paper_text", return_value="Text."):
            await summarizer.summarize(paper_id, "brief")

        llm.response = "Detailed summary."
        with patch.object(summarizer, "fetch_paper_text", return_value="Text."):
            await summarizer.summarize(paper_id, "detailed")

        brief_cached = store.get_summary(paper_id, "brief")
        detailed_cached = store.get_summary(paper_id, "detailed")

        assert brief_cached["content"] == "Brief summary."
        assert detailed_cached["content"] == "Detailed summary."
        assert llm.call_count == 2


# --- Test _get_paper_by_id ---


class TestGetPaperById:
    def test_returns_paper_by_integer_id(self, tmp_path):
        """Test that _get_paper_by_id returns the correct paper."""
        store = _make_mock_store(tmp_path)
        paper_id = _insert_paper(store, arxiv_id="2501.11111", title="Found Paper")

        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        paper = summarizer._get_paper_by_id(paper_id)

        assert paper is not None
        assert paper["title"] == "Found Paper"
        assert paper["arxiv_id"] == "2501.11111"
        assert isinstance(paper["authors"], list)
        assert isinstance(paper["categories"], list)

    def test_returns_none_for_nonexistent_id(self, tmp_path):
        """Test that _get_paper_by_id returns None for a missing id."""
        store = _make_mock_store(tmp_path)
        llm = MockLLMProvider()
        summarizer = PaperSummarizer(llm, store)

        paper = summarizer._get_paper_by_id(99999)
        assert paper is None
