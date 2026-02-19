"""Tests for ReportGenerator (Phase 7)."""

import pytest

from src.llm.base import LLMProvider
from src.report.generator import ReportGenerator


# ---------------------------------------------------------------------------
# Mock LLM Provider
# ---------------------------------------------------------------------------


class MockLLMProvider(LLMProvider):
    """Returns canned Markdown strings for trending topics and highlights."""

    TRENDING = (
        "- **Multi-modal reasoning**: 4 papers explore combining vision and language models.\n"
        "- **Efficient transformers**: 3 papers propose new attention mechanisms.\n"
        "- **Reinforcement learning from human feedback**: 2 papers advance RLHF techniques."
    )

    HIGHLIGHTS = (
        "1. **Attention Is Still All You Need** by Alice et al. "
        "— Proposes a novel sparse attention that is 3x faster.\n"
        "2. **Scaling Laws for Code** by Bob, Charlie "
        "— Establishes power-law scaling for code generation.\n"
        "3. **Better RLHF** by Dave "
        "— Introduces a reward-model-free approach to RLHF."
    )

    def __init__(self):
        self.calls: list[str] = []

    async def complete(self, prompt: str, system: str = "") -> str:
        self.calls.append(prompt)
        if "trending" in prompt.lower() or "emerging" in prompt.lower():
            return self.TRENDING
        if "noteworthy" in prompt.lower() or "impactful" in prompt.lower():
            return self.HIGHLIGHTS
        return "LLM response"

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        self.calls.append(prompt)
        return {"result": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_papers(n: int, categories_map: dict[str, int] | None = None) -> list[dict]:
    """Create n fake papers.

    If categories_map is provided, it maps category -> count.
    E.g. {"cs.AI": 3, "cs.CL": 4, "cs.LG": 3} creates 10 papers.
    Otherwise creates n papers all in cs.AI.
    """
    papers = []
    if categories_map:
        idx = 0
        for cat, count in categories_map.items():
            for j in range(count):
                papers.append(_make_paper(idx, primary_category=cat))
                idx += 1
    else:
        for i in range(n):
            papers.append(_make_paper(i))
    return papers


def _make_paper(idx: int, primary_category: str = "cs.AI") -> dict:
    return {
        "id": idx + 1,
        "arxiv_id": f"2501.{10000 + idx}",
        "title": f"Paper Title {idx}",
        "authors": [f"Author A{idx}", f"Author B{idx}"],
        "abstract": f"This paper investigates topic {idx} in depth. " * 10,
        "categories": [primary_category, "cs.LG"],
        "published_date": "2025-01-15",
        "pdf_url": f"https://arxiv.org/pdf/2501.{10000 + idx}",
        "ar5iv_url": f"https://ar5iv.labs.arxiv.org/html/2501.{10000 + idx}",
    }


def _make_ranked_papers(n: int) -> list[dict]:
    """Create n pre-scored papers (as returned by the LLM ranker)."""
    papers = []
    for i in range(n):
        paper = _make_paper(i)
        paper["llm_score"] = round(9.0 - i * 0.5, 1)
        paper["llm_reason"] = f"Reason for paper {i}"
        paper["embedding_score"] = round(0.9 - i * 0.05, 3)
    papers.append(paper)
    return papers


# ---------------------------------------------------------------------------
# Step 7.1 — General Report Tests
# ---------------------------------------------------------------------------


class TestGenerateGeneral:
    @pytest.fixture
    def llm(self):
        return MockLLMProvider()

    @pytest.fixture
    def gen(self, llm):
        return ReportGenerator(llm)

    @pytest.mark.asyncio
    async def test_header_contains_run_date(self, gen):
        papers = _make_papers(5)
        report = await gen.generate_general(papers, "2025-01-15")
        assert "# Daily Paper Report - 2025-01-15" in report

    @pytest.mark.asyncio
    async def test_overview_total_count(self, gen):
        papers = _make_papers(7)
        report = await gen.generate_general(papers, "2025-01-15")
        assert "**7** new papers collected" in report

    @pytest.mark.asyncio
    async def test_overview_category_breakdown(self, gen):
        papers = _make_papers(10, categories_map={"cs.AI": 3, "cs.CL": 4, "cs.LG": 3})
        report = await gen.generate_general(papers, "2025-01-15")
        assert "cs.AI: 3" in report
        assert "cs.CL: 4" in report
        assert "cs.LG: 3" in report

    @pytest.mark.asyncio
    async def test_trending_topics_section_present(self, gen):
        papers = _make_papers(10)
        report = await gen.generate_general(papers, "2025-01-15")
        assert "### Trending Topics" in report
        assert "Multi-modal reasoning" in report

    @pytest.mark.asyncio
    async def test_highlight_papers_section_present(self, gen):
        papers = _make_papers(10)
        report = await gen.generate_general(papers, "2025-01-15")
        assert "### Highlight Papers" in report
        assert "Attention Is Still All You Need" in report

    @pytest.mark.asyncio
    async def test_llm_called_twice(self, gen, llm):
        """LLM should be called once for trending, once for highlights."""
        papers = _make_papers(5)
        await gen.generate_general(papers, "2025-01-15")
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_empty_papers(self, gen):
        """Empty paper list should produce a valid report with zero count."""
        report = await gen.generate_general([], "2025-01-15")
        assert "**0** new papers collected" in report
        assert "No papers available" in report

    @pytest.mark.asyncio
    async def test_general_report_section_header(self, gen):
        papers = _make_papers(3)
        report = await gen.generate_general(papers, "2025-01-15")
        assert "## General Report" in report

    @pytest.mark.asyncio
    async def test_overview_section_header(self, gen):
        papers = _make_papers(3)
        report = await gen.generate_general(papers, "2025-01-15")
        assert "### Today's Overview" in report


# ---------------------------------------------------------------------------
# Step 7.2 — Specific Report Tests
# ---------------------------------------------------------------------------


class TestGenerateSpecific:
    @pytest.fixture
    def llm(self):
        return MockLLMProvider()

    @pytest.fixture
    def gen(self, llm):
        return ReportGenerator(llm)

    @pytest.mark.asyncio
    async def test_specific_report_header(self, gen):
        ranked = _make_ranked_papers(3)
        interests = [{"type": "keyword", "value": "transformers"}]
        report = await gen.generate_specific(ranked, interests, "2025-01-15")
        assert "## Specific Report (Based on Your Interests)" in report

    @pytest.mark.asyncio
    async def test_paper_titles_appear(self, gen):
        ranked = _make_ranked_papers(3)
        interests = [{"type": "keyword", "value": "transformers"}]
        report = await gen.generate_specific(ranked, interests, "2025-01-15")
        for paper in ranked:
            assert paper["title"] in report

    @pytest.mark.asyncio
    async def test_llm_scores_appear(self, gen):
        ranked = _make_ranked_papers(3)
        interests = [{"type": "keyword", "value": "transformers"}]
        report = await gen.generate_specific(ranked, interests, "2025-01-15")
        for paper in ranked:
            score_str = f"{paper['llm_score']}/10"
            assert score_str in report

    @pytest.mark.asyncio
    async def test_llm_reasons_appear(self, gen):
        ranked = _make_ranked_papers(3)
        interests = [{"type": "keyword", "value": "transformers"}]
        report = await gen.generate_specific(ranked, interests, "2025-01-15")
        for paper in ranked:
            assert paper["llm_reason"] in report

    @pytest.mark.asyncio
    async def test_related_papers_section(self, gen):
        ranked = _make_ranked_papers(3)
        interests = [{"type": "keyword", "value": "transformers"}]
        report = await gen.generate_specific(ranked, interests, "2025-01-15")
        assert "## Related Papers" in report

    @pytest.mark.asyncio
    async def test_arxiv_links_present(self, gen):
        ranked = _make_ranked_papers(3)
        interests = [{"type": "keyword", "value": "transformers"}]
        report = await gen.generate_specific(ranked, interests, "2025-01-15")
        for paper in ranked:
            assert f"https://arxiv.org/abs/{paper['arxiv_id']}" in report

    @pytest.mark.asyncio
    async def test_no_llm_calls(self, gen, llm):
        """generate_specific should NOT call the LLM."""
        ranked = _make_ranked_papers(3)
        interests = [{"type": "keyword", "value": "transformers"}]
        await gen.generate_specific(ranked, interests, "2025-01-15")
        assert len(llm.calls) == 0

    @pytest.mark.asyncio
    async def test_empty_ranked_papers(self, gen):
        """No matches should produce a valid report with a message."""
        interests = [{"type": "keyword", "value": "transformers"}]
        report = await gen.generate_specific([], interests, "2025-01-15")
        assert "No papers matched" in report

    @pytest.mark.asyncio
    async def test_authors_and_categories_in_related(self, gen):
        ranked = _make_ranked_papers(2)
        interests = [{"type": "keyword", "value": "transformers"}]
        report = await gen.generate_specific(ranked, interests, "2025-01-15")
        assert "**Authors**:" in report
        assert "**Categories**:" in report


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.fixture
    def llm(self):
        return MockLLMProvider()

    @pytest.fixture
    def gen(self, llm):
        return ReportGenerator(llm)

    @pytest.mark.asyncio
    async def test_paper_with_string_authors(self, gen):
        """Authors stored as a string (not a list) should not crash."""
        papers = [_make_paper(0)]
        papers[0]["authors"] = "Single Author"
        report = await gen.generate_specific(papers, [], "2025-01-15")
        assert "Single Author" in report

    @pytest.mark.asyncio
    async def test_paper_with_string_categories(self, gen):
        """Categories stored as a string should not crash."""
        papers = [_make_paper(0)]
        papers[0]["categories"] = "cs.AI"
        papers[0]["llm_score"] = 8.0
        papers[0]["llm_reason"] = "test"
        report = await gen.generate_specific(papers, [], "2025-01-15")
        assert "cs.AI" in report

    @pytest.mark.asyncio
    async def test_llm_failure_trending_topics(self, gen):
        """LLM failure should produce a graceful error message, not crash."""

        class FailingLLM(LLMProvider):
            async def complete(self, prompt, system=""):
                raise RuntimeError("LLM unavailable")

            async def complete_json(self, prompt, system=""):
                raise RuntimeError("LLM unavailable")

        gen_fail = ReportGenerator(FailingLLM())
        papers = _make_papers(3)
        report = await gen_fail.generate_general(papers, "2025-01-15")
        assert "Unable to generate" in report

    @pytest.mark.asyncio
    async def test_overview_with_single_category_paper(self, gen):
        """Paper with a single-element categories list should work."""
        papers = [_make_paper(0)]
        papers[0]["categories"] = ["cs.CV"]
        report = await gen.generate_general(papers, "2025-01-15")
        assert "cs.CV: 1" in report
