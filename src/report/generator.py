import logging
from collections import Counter

from src.llm.base import LLMProvider


class ReportGenerator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    async def generate_general(self, papers: list[dict], run_date: str) -> str:
        """Generate a Markdown general report.

        Contents:
        1. Header with run date
        2. Today's Overview — total count, count per primary category
        3. Trending Topics — LLM-identified trends from paper titles
        4. Highlight Papers — LLM-selected noteworthy papers
        """
        self.logger.info("Generating general report for %s", run_date)

        sections = []

        # Header
        sections.append(f"# Daily Paper Report - {run_date}\n")
        sections.append("## General Report\n")

        # Overview section (pure Python, no LLM)
        sections.append(self._build_overview(papers))

        # Trending Topics (LLM)
        trending = await self._build_trending_topics(papers)
        sections.append(trending)

        # Highlight Papers (LLM)
        highlights = await self._build_highlight_papers(papers)
        sections.append(highlights)

        report = "\n".join(sections)
        self.logger.info("General report generation complete")
        return report

    async def generate_specific(
        self, ranked_papers: list[dict], interests: list[dict], run_date: str
    ) -> str:
        """Generate a Markdown specific report from already-scored papers.

        This method does NOT call the LLM — it formats the pre-scored data
        from the ranker.

        Contents:
        1. Header
        2. Numbered list of ranked papers with scores and reasons
        3. Related Papers section with details and arXiv links
        """
        self.logger.info("Generating specific report for %s", run_date)

        sections = []

        sections.append("## Specific Report (Based on Your Interests)\n")

        if not ranked_papers:
            sections.append("No papers matched your interests today.\n")
            self.logger.info("Specific report generation complete (no matches)")
            return "\n".join(sections)

        sections.append(
            f"Top {len(ranked_papers)} papers matching your research interests today:\n"
        )

        # Numbered list with scores and reasons
        for i, paper in enumerate(ranked_papers, 1):
            score = paper.get("llm_score", 0)
            reason = paper.get("llm_reason", "N/A")
            title = paper.get("title", "Unknown")
            sections.append(f"{i}. **{title}** (Relevance: {score}/10)")
            sections.append(f"   - Why it matters to you: {reason}\n")

        # Related Papers section with full details
        sections.append("---\n")
        sections.append("## Related Papers\n")

        for i, paper in enumerate(ranked_papers, 1):
            title = paper.get("title", "Unknown")
            authors = paper.get("authors", [])
            if isinstance(authors, list):
                authors_str = ", ".join(authors)
            else:
                authors_str = str(authors)
            categories = paper.get("categories", [])
            if isinstance(categories, list):
                categories_str = ", ".join(categories)
            else:
                categories_str = str(categories)
            abstract = paper.get("abstract", "")
            abstract_preview = abstract[:200] + "..." if len(abstract) > 200 else abstract
            arxiv_id = paper.get("arxiv_id", "")

            sections.append(f"### {i}. {title}")
            sections.append(f"- **Authors**: {authors_str}")
            sections.append(f"- **Categories**: {categories_str}")
            sections.append(f"- **Abstract**: {abstract_preview}")
            sections.append(f"- [arXiv](https://arxiv.org/abs/{arxiv_id})\n")

        report = "\n".join(sections)
        self.logger.info("Specific report generation complete")
        return report

    def _build_overview(self, papers: list[dict]) -> str:
        """Build the overview section with paper counts per primary category."""
        total = len(papers)
        lines = []
        lines.append("### Today's Overview")
        lines.append(f"- **{total}** new papers collected")

        if papers:
            # Count by primary category (first in the categories list)
            category_counter = Counter()
            for paper in papers:
                categories = paper.get("categories", [])
                if isinstance(categories, list) and categories:
                    primary = categories[0]
                elif isinstance(categories, str):
                    primary = categories
                else:
                    primary = "unknown"
                category_counter[primary] += 1

            # Format as "cs.AI: 3 | cs.CL: 4 | ..."
            breakdown = " | ".join(
                f"{cat}: {count}" for cat, count in category_counter.most_common()
            )
            lines.append(f"- {breakdown}")

        lines.append("")
        return "\n".join(lines)

    async def _build_trending_topics(self, papers: list[dict]) -> str:
        """Ask the LLM to identify trending topics from paper titles."""
        if not papers:
            return "### Trending Topics\n\nNo papers available for trend analysis.\n"

        titles = [paper.get("title", "") for paper in papers]
        titles_text = "\n".join(f"- {t}" for t in titles)

        prompt = (
            f"Here are {len(titles)} paper titles published today on arXiv:\n\n"
            f"{titles_text}\n\n"
            "Based on these titles, identify 3-5 emerging or trending research topics. "
            "For each topic, provide a brief description (1-2 sentences) explaining "
            "the trend and how many papers relate to it. "
            "Format as a Markdown bullet list."
        )
        system = "You are a research trend analyst summarizing daily arXiv publications."

        try:
            response = await self.llm.complete(prompt, system=system)
        except Exception as e:
            self.logger.error("LLM call failed for trending topics: %s", e)
            response = "- Unable to generate trending topics due to an error."

        lines = ["### Trending Topics", "", response, ""]
        return "\n".join(lines)

    async def _build_highlight_papers(self, papers: list[dict]) -> str:
        """Ask the LLM to select noteworthy papers."""
        if not papers:
            return "### Highlight Papers\n\nNo papers available for highlighting.\n"

        # Send titles + first 150 chars of abstract
        entries = []
        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            snippet = abstract[:150] + "..." if len(abstract) > 150 else abstract
            authors = paper.get("authors", [])
            if isinstance(authors, list):
                authors_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_str += " et al."
            else:
                authors_str = str(authors)
            entries.append(f"- **{title}** by {authors_str}: {snippet}")

        entries_text = "\n".join(entries)

        prompt = (
            f"Here are {len(papers)} papers published today on arXiv:\n\n"
            f"{entries_text}\n\n"
            "Select 3-5 of the most noteworthy or impactful papers from this list. "
            "For each, provide: the paper title, the authors, and a one-line description "
            "of why it is noteworthy. Format as a numbered Markdown list."
        )
        system = "You are a research curator selecting the most important daily arXiv papers."

        try:
            response = await self.llm.complete(prompt, system=system)
        except Exception as e:
            self.logger.error("LLM call failed for highlight papers: %s", e)
            response = "1. Unable to generate highlights due to an error."

        lines = ["### Highlight Papers", "", response, ""]
        return "\n".join(lines)
