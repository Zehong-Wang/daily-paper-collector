from __future__ import annotations

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
        """Generate a Markdown specific report with theme-based synthesis.

        Uses the LLM to synthesize ranked papers into thematic clusters,
        followed by comprehensive paper details.

        Contents:
        1. Header
        2. Theme-based synthesis (LLM-generated narrative grouped by themes)
        3. Paper Details section with full authors, abstracts, and relevance reasons
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

        # Theme-based synthesis (LLM for >= 5 papers, simple summary otherwise)
        synthesis = await self._build_theme_synthesis(ranked_papers, interests)
        sections.append(synthesis)

        # Paper Details section with comprehensive info
        sections.append("\n---\n")
        sections.append(self._build_paper_details(ranked_papers))

        report = "\n".join(sections)
        self.logger.info("Specific report generation complete")
        return report

    async def _build_theme_synthesis(self, ranked_papers: list[dict], interests: list[dict]) -> str:
        """Build a theme-based synthesis of ranked papers using the LLM.

        For >= 5 papers, asks the LLM to group papers into thematic clusters
        with narrative paragraphs. For < 5 papers, returns a simple bullet
        list without LLM calls.
        """
        if len(ranked_papers) < 5:
            return self._build_simple_summary(ranked_papers)

        # Format interests for context
        interest_values = [i.get("value", "") for i in interests if i.get("value")]
        interests_csv = ", ".join(interest_values) if interest_values else "general research"

        # Build paper entries for the prompt
        paper_entries = []
        for i, paper in enumerate(ranked_papers, 1):
            title = paper.get("title", "Unknown")
            score = paper.get("llm_score", 0)
            reason = paper.get("llm_reason", "N/A")
            abstract = paper.get("abstract", "")
            paper_entries.append(
                f"Paper {i}: {title}\n"
                f"  Score: {score}/10\n"
                f"  Reason: {reason}\n"
                f"  Abstract: {abstract}"
            )

        papers_text = "\n\n".join(paper_entries)

        prompt = (
            f"You are given {len(ranked_papers)} research papers that matched "
            f"a user's interests: {interests_csv}.\n\n"
            f"Here are the papers:\n\n{papers_text}\n\n"
            "Group these papers into 3-6 thematic clusters based on their topics "
            "and methodologies. For each theme:\n"
            "1. Give the theme a bold descriptive name as a ### heading\n"
            "2. Write a flowing 2-4 sentence narrative naming specific papers "
            "in this cluster and how they relate to the user's interests\n"
            "3. Highlight connections or contrasts between papers\n\n"
            "Format as Markdown. IMPORTANT: Do NOT include any top-level headings "
            "(# or ##). Start directly with ### theme headings."
        )
        system = (
            "You are a research synthesis expert. Your job is to identify thematic "
            "patterns across papers and write concise, insightful narratives that "
            "help the reader understand the research landscape."
        )

        try:
            response = await self.llm.complete(prompt, system=system)
        except Exception as e:
            self.logger.error("LLM call failed for theme synthesis: %s", e)
            return self._build_fallback_list(ranked_papers)

        return response + "\n"

    def _build_simple_summary(self, ranked_papers: list[dict]) -> str:
        """Build a simple bullet-list summary for fewer than 5 papers (no LLM)."""
        lines = []
        for paper in ranked_papers:
            title = paper.get("title", "Unknown")
            score = paper.get("llm_score", 0)
            reason = paper.get("llm_reason", "N/A")
            lines.append(f"- **{title}** (Score: {score}/10): {reason}")
        lines.append("")
        return "\n".join(lines)

    def _build_fallback_list(self, ranked_papers: list[dict]) -> str:
        """Build a numbered fallback list when LLM synthesis fails."""
        lines = []
        for i, paper in enumerate(ranked_papers, 1):
            title = paper.get("title", "Unknown")
            score = paper.get("llm_score", 0)
            reason = paper.get("llm_reason", "N/A")
            arxiv_id = paper.get("arxiv_id", "")
            lines.append(
                f"{i}. **{title}** — Relevance: **{score}/10** "
                f"([arXiv](https://arxiv.org/abs/{arxiv_id}))  \n"
                f"   {reason}"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_paper_details(self, ranked_papers: list[dict]) -> str:
        """Build comprehensive paper details section."""
        lines = []
        lines.append("## Paper Details\n")

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
            arxiv_id = paper.get("arxiv_id", "")
            score = paper.get("llm_score", 0)
            reason = paper.get("llm_reason", "N/A")

            lines.append(f"### {i}. {title}\n")
            lines.append(f"**Score**: {score}/10 | **Categories**: {categories_str}\n")
            lines.append(f"**Authors**: {authors_str}\n")
            lines.append(f"**Abstract**: {abstract}\n")
            lines.append(f"**Why this paper is relevant**: {reason}\n")
            lines.append(f"[Read on arXiv →](https://arxiv.org/abs/{arxiv_id})\n")

        return "\n".join(lines)

    def _build_overview(self, papers: list[dict]) -> str:
        """Build the overview section with paper counts per primary category."""
        total = len(papers)
        lines = []
        lines.append("### Today's Overview\n")
        lines.append(f"**{total}** new papers collected\n")

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

            # Show top categories as a table
            lines.append("| Category | Papers |")
            lines.append("|----------|--------|")
            most_common = category_counter.most_common()
            top_categories = most_common[:10]
            rest = most_common[10:]
            for cat, count in top_categories:
                lines.append(f"| {cat} | {count} |")
            if rest:
                rest_total = sum(c for _, c in rest)
                lines.append(f"| Others ({len(rest)} categories) | {rest_total} |")

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
            "For each topic, provide a bold topic name followed by a brief description "
            "(1-2 sentences) explaining the trend and approximate paper count. "
            "Format as a Markdown bullet list. "
            "IMPORTANT: Do NOT include any headings (# or ##). "
            "Start directly with the bullet list."
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
            "For each, provide: the paper title (bold), the authors, and a one-line "
            "description of why it is noteworthy. Format as a numbered Markdown list. "
            "IMPORTANT: Do NOT include any headings (# or ##). "
            "Start directly with the numbered list."
        )
        system = "You are a research curator selecting the most important daily arXiv papers."

        try:
            response = await self.llm.complete(prompt, system=system)
        except Exception as e:
            self.logger.error("LLM call failed for highlight papers: %s", e)
            response = "1. Unable to generate highlights due to an error."

        lines = ["### Highlight Papers", "", response, ""]
        return "\n".join(lines)
