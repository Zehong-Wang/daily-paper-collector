from __future__ import annotations

import logging
from collections import Counter

from src.llm.base import LLMProvider


class ReportGenerator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    async def generate_general(
        self, papers: list[dict], run_date: str, date_label: str = None
    ) -> str:
        """Generate a Markdown general report.

        Contents:
        1. Header with run date (or date_label for multi-day reports)
        2. Today's Overview — total count, count per primary category
        3. Trending Topics — LLM-identified trends from paper titles
        4. Highlight Papers — LLM-selected noteworthy papers

        Args:
            date_label: Optional display label for multi-day reports (e.g. "2026-02-20 ~ 2026-02-22").
                        When provided, replaces run_date in the header and uses period-aware wording.
        """
        self.logger.info("Generating general report for %s", run_date)

        sections = []

        # Header
        display_date = date_label or run_date
        sections.append(f"# Daily Paper Report - {display_date}\n")
        sections.append("## General Report\n")

        # Overview section (pure Python, no LLM)
        sections.append(self._build_overview(papers, period=date_label is not None))

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
        self,
        ranked_papers: list[dict],
        interests: list[dict],
        run_date: str,
        date_label: str = None,
    ) -> str:
        """Generate a Markdown specific report with theme-based synthesis.

        Uses the LLM to synthesize ranked papers into thematic clusters,
        followed by comprehensive paper details.

        Contents:
        1. Header
        2. Theme-based synthesis (LLM-generated narrative grouped by themes)
        3. Paper Details section with full authors, abstracts, and relevance reasons

        Args:
            date_label: Optional display label for multi-day reports. When provided,
                        uses "in this period" instead of "today" in the intro text.
        """
        self.logger.info("Generating specific report for %s", run_date)

        sections = []

        sections.append("## Specific Report (Based on Your Interests)\n")

        period_text = "in this period" if date_label else "today"
        if not ranked_papers:
            sections.append(f"No papers matched your interests {period_text}.\n")
            self.logger.info("Specific report generation complete (no matches)")
            return "\n".join(sections)

        sections.append(
            f"Top {len(ranked_papers)} papers matching your research interests {period_text}:\n"
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

    # ---- Chinese report generation methods ----

    async def generate_general_zh(
        self, papers: list[dict], run_date: str, date_label: str = None
    ) -> str:
        """Generate a Chinese Markdown general report.

        Args:
            date_label: Optional display label for multi-day reports.
        """
        self.logger.info("Generating Chinese general report for %s", run_date)

        sections = []
        display_date = date_label or run_date
        sections.append(f"# 每日论文报告 - {display_date}\n")
        sections.append("## 综合报告\n")
        sections.append(self._build_overview_zh(papers, period=date_label is not None))

        trending = await self._build_trending_topics_zh(papers)
        sections.append(trending)

        highlights = await self._build_highlight_papers_zh(papers)
        sections.append(highlights)

        report = "\n".join(sections)
        self.logger.info("Chinese general report generation complete")
        return report

    async def generate_specific_zh(
        self,
        ranked_papers: list[dict],
        interests: list[dict],
        run_date: str,
        date_label: str = None,
    ) -> str:
        """Generate a Chinese Markdown specific report with theme-based synthesis.

        Args:
            date_label: Optional display label for multi-day reports.
        """
        self.logger.info("Generating Chinese specific report for %s", run_date)

        sections = []
        sections.append("## 个性化推荐报告（基于您的研究兴趣）\n")

        period_text = "本期" if date_label else "今日"
        if not ranked_papers:
            sections.append(f"{period_text}没有匹配您研究兴趣的论文。\n")
            self.logger.info("Chinese specific report generation complete (no matches)")
            return "\n".join(sections)

        sections.append(f"{period_text}共有 {len(ranked_papers)} 篇论文匹配您的研究兴趣：\n")

        synthesis = await self._build_theme_synthesis_zh(ranked_papers, interests)
        sections.append(synthesis)

        sections.append("\n---\n")
        sections.append(self._build_paper_details_zh(ranked_papers))

        report = "\n".join(sections)
        self.logger.info("Chinese specific report generation complete")
        return report

    async def _build_theme_synthesis_zh(
        self, ranked_papers: list[dict], interests: list[dict]
    ) -> str:
        """Build a Chinese theme-based synthesis using the LLM."""
        if len(ranked_papers) < 5:
            return self._build_simple_summary_zh(ranked_papers)

        interest_values = [i.get("value", "") for i in interests if i.get("value")]
        interests_csv = ", ".join(interest_values) if interest_values else "综合研究"

        paper_entries = []
        for i, paper in enumerate(ranked_papers, 1):
            title = paper.get("title", "Unknown")
            score = paper.get("llm_score", 0)
            reason = paper.get("llm_reason", "N/A")
            abstract = paper.get("abstract", "")
            paper_entries.append(
                f"论文 {i}: {title}\n  评分: {score}/10\n  推荐理由: {reason}\n  摘要: {abstract}"
            )

        papers_text = "\n\n".join(paper_entries)

        prompt = (
            f"以下是 {len(ranked_papers)} 篇与用户研究兴趣（{interests_csv}）"
            f"匹配的论文：\n\n{papers_text}\n\n"
            "请将这些论文按主题分成 3-6 个类别。对每个主题：\n"
            "1. 用 ### 标题给出主题名称（中文）\n"
            "2. 用 2-4 句话描述该主题下的论文及其与用户兴趣的关联\n"
            "3. 指出论文之间的联系或差异\n\n"
            "请用中文回答，格式为 Markdown。重要：不要使用 # 或 ## 级别的标题，"
            "直接从 ### 主题标题开始。"
        )
        system = (
            "你是一位研究综述专家。你的任务是识别论文之间的主题模式，"
            "撰写简洁有深度的中文叙述，帮助读者理解研究前沿。"
        )

        try:
            response = await self.llm.complete(prompt, system=system)
        except Exception as e:
            self.logger.error("LLM call failed for Chinese theme synthesis: %s", e)
            return self._build_fallback_list_zh(ranked_papers)

        return response + "\n"

    def _build_simple_summary_zh(self, ranked_papers: list[dict]) -> str:
        """Build a simple Chinese bullet-list summary for fewer than 5 papers."""
        lines = []
        for paper in ranked_papers:
            title = paper.get("title", "Unknown")
            score = paper.get("llm_score", 0)
            reason = paper.get("llm_reason", "N/A")
            lines.append(f"- **{title}**（评分: {score}/10）: {reason}")
        lines.append("")
        return "\n".join(lines)

    def _build_fallback_list_zh(self, ranked_papers: list[dict]) -> str:
        """Build a numbered Chinese fallback list when LLM synthesis fails."""
        lines = []
        for i, paper in enumerate(ranked_papers, 1):
            title = paper.get("title", "Unknown")
            score = paper.get("llm_score", 0)
            reason = paper.get("llm_reason", "N/A")
            arxiv_id = paper.get("arxiv_id", "")
            lines.append(
                f"{i}. **{title}** — 相关性: **{score}/10** "
                f"([arXiv](https://arxiv.org/abs/{arxiv_id}))  \n"
                f"   {reason}"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_paper_details_zh(self, ranked_papers: list[dict]) -> str:
        """Build comprehensive Chinese paper details section."""
        lines = []
        lines.append("## 论文详情\n")

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
            lines.append(f"**评分**: {score}/10 | **分类**: {categories_str}\n")
            lines.append(f"**作者**: {authors_str}\n")
            lines.append(f"**摘要**: {abstract}\n")
            lines.append(f"**推荐理由**: {reason}\n")
            lines.append(f"[在 arXiv 上阅读 →](https://arxiv.org/abs/{arxiv_id})\n")

        return "\n".join(lines)

    def _build_overview_zh(self, papers: list[dict], period: bool = False) -> str:
        """Build Chinese overview section with paper counts per primary category.

        Args:
            period: If True, use period-aware wording instead of "today".
        """
        total = len(papers)
        lines = []
        if period:
            lines.append("### 周期概览\n")
            lines.append(f"本期共收录 **{total}** 篇论文\n")
        else:
            lines.append("### 今日概览\n")
            lines.append(f"今日共收录 **{total}** 篇新论文\n")

        if papers:
            category_counter = Counter()
            for paper in papers:
                categories = paper.get("categories", [])
                if isinstance(categories, list) and categories:
                    primary = categories[0]
                elif isinstance(categories, str):
                    primary = categories
                else:
                    primary = "未知"
                category_counter[primary] += 1

            lines.append("| 分类 | 论文数 |")
            lines.append("|------|--------|")
            most_common = category_counter.most_common()
            top_categories = most_common[:10]
            rest = most_common[10:]
            for cat, count in top_categories:
                lines.append(f"| {cat} | {count} |")
            if rest:
                rest_total = sum(c for _, c in rest)
                lines.append(f"| 其他（{len(rest)} 个分类）| {rest_total} |")

        lines.append("")
        return "\n".join(lines)

    async def _build_trending_topics_zh(self, papers: list[dict]) -> str:
        """Ask the LLM to identify trending topics in Chinese."""
        if not papers:
            return "### 热门研究方向\n\n暂无论文可供趋势分析。\n"

        titles = [paper.get("title", "") for paper in papers]
        titles_text = "\n".join(f"- {t}" for t in titles)

        prompt = (
            f"以下是今天在 arXiv 上发表的 {len(titles)} 篇论文标题：\n\n"
            f"{titles_text}\n\n"
            "根据这些标题，识别 3-5 个新兴或热门的研究方向。"
            "对每个方向，请提供加粗的方向名称和 1-2 句描述（说明趋势和大致论文数量）。"
            "格式为 Markdown 无序列表。请用中文回答。"
            "重要：不要包含任何标题（# 或 ##），直接从列表开始。"
        )
        system = "你是一位研究趋势分析师，负责总结每日 arXiv 论文的研究动态。请用中文回答。"

        try:
            response = await self.llm.complete(prompt, system=system)
        except Exception as e:
            self.logger.error("LLM call failed for Chinese trending topics: %s", e)
            response = "- 由于错误，暂时无法生成热门研究方向。"

        lines = ["### 热门研究方向", "", response, ""]
        return "\n".join(lines)

    async def _build_highlight_papers_zh(self, papers: list[dict]) -> str:
        """Ask the LLM to select noteworthy papers and describe in Chinese."""
        if not papers:
            return "### 亮点论文\n\n暂无论文可供推荐。\n"

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
            f"以下是今天在 arXiv 上发表的 {len(papers)} 篇论文：\n\n"
            f"{entries_text}\n\n"
            "请从中选出 3-5 篇最值得关注的论文。"
            "对每篇论文，请提供：论文标题（加粗）、作者、以及一句话推荐理由。"
            "格式为编号的 Markdown 列表。请用中文回答。"
            "重要：不要包含任何标题（# 或 ##），直接从编号列表开始。"
        )
        system = "你是一位研究策展人，负责从每日 arXiv 论文中选出最重要的论文。请用中文回答。"

        try:
            response = await self.llm.complete(prompt, system=system)
        except Exception as e:
            self.logger.error("LLM call failed for Chinese highlight papers: %s", e)
            response = "1. 由于错误，暂时无法生成亮点论文推荐。"

        lines = ["### 亮点论文", "", response, ""]
        return "\n".join(lines)

    # ---- English report helper methods ----

    def _build_overview(self, papers: list[dict], period: bool = False) -> str:
        """Build the overview section with paper counts per primary category.

        Args:
            period: If True, use "Period Overview" wording instead of "Today's Overview".
        """
        total = len(papers)
        lines = []
        if period:
            lines.append("### Period Overview\n")
            lines.append(f"**{total}** papers in this period\n")
        else:
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
