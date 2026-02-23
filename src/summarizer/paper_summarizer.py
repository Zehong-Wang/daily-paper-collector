import io
import logging

import requests
from bs4 import BeautifulSoup


class PaperSummarizer:
    _MAX_TEXT_CHARS = 15000

    def __init__(self, llm, store):
        self.llm = llm
        self.store = store
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _truncate_text(text: str) -> str:
        """Limit text length to stay within LLM context budget."""
        return text[: PaperSummarizer._MAX_TEXT_CHARS]

    @staticmethod
    def _looks_like_navigation_shell(text: str) -> bool:
        """Detect non-paper UI shell text (help/search/citation nav) from ar5iv/arXiv pages."""
        lower = text.lower()
        nav_markers = [
            "help",
            "search",
            "references & citations",
            "export bibtex",
            "submission history",
            "view pdf",
            "arxivlabs",
            "bookmark",
            "add to lists",
        ]
        section_markers = [
            "abstract",
            "introduction",
            "method",
            "experiment",
            "conclusion",
            "result",
        ]
        nav_hits = sum(marker in lower for marker in nav_markers)
        has_sections = any(marker in lower for marker in section_markers)
        return nav_hits >= 4 and not has_sections

    def fetch_paper_text(self, ar5iv_url: str) -> str:
        """Fetch the ar5iv HTML page and extract the paper's main text.

        1. requests.get(ar5iv_url, timeout=30).
        2. Parse with BeautifulSoup(html, "lxml").
        3. Find the main content: look for <article> tag, or fall back to
           class "ltx_document" or "ltx_page_main".
        4. Extract text from all <p>, <h2>, <h3> tags within the main content.
        5. Join paragraphs with double newlines.
        6. Truncate to 15000 characters (LLM context limit safety).
        Return the extracted text. Raise RuntimeError if fetch fails.
        """
        try:
            response = requests.get(
                ar5iv_url,
                timeout=30,
                headers={"User-Agent": "daily-paper-collector/0.1 (+https://arxiv.org)"},
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch paper from {ar5iv_url}: {e}") from e

        soup = BeautifulSoup(response.text, "lxml")

        # Find main content container
        main_content = soup.find("article")
        if main_content is None:
            main_content = soup.find(class_="ltx_document")
        if main_content is None:
            main_content = soup.find(class_="ltx_page_main")
        if main_content is None:
            main_content = soup.body if soup.body else soup

        # Extract text from relevant tags
        paragraphs = []
        for tag in main_content.find_all(["p", "h2", "h3"]):
            text = tag.get_text(strip=True)
            if text:
                paragraphs.append(text)

        full_text = "\n\n".join(paragraphs)
        if not full_text:
            raise RuntimeError(f"No extractable text found in {ar5iv_url}")
        if self._looks_like_navigation_shell(full_text):
            raise RuntimeError(
                f"Extracted navigation shell content instead of paper body from {ar5iv_url}"
            )

        # Truncate to max characters
        full_text = self._truncate_text(full_text)

        self.logger.info(f"Extracted {len(full_text)} characters from {ar5iv_url}")
        return full_text

    def fetch_pdf_text(self, pdf_url: str) -> str:
        """Fetch the paper PDF and extract text as fallback when ar5iv HTML is unusable."""
        try:
            response = requests.get(
                pdf_url,
                timeout=60,
                headers={"User-Agent": "daily-paper-collector/0.1 (+https://arxiv.org)"},
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch PDF from {pdf_url}: {e}") from e

        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "PDF fallback requires 'pypdf'. Install dependencies from requirements.txt."
            ) from e

        try:
            reader = PdfReader(io.BytesIO(response.content))
        except Exception as e:  # pragma: no cover - parser-specific failures
            raise RuntimeError(f"Failed to parse PDF from {pdf_url}: {e}") from e

        chunks = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                chunks.append(page_text.strip())
            if sum(len(c) for c in chunks) >= self._MAX_TEXT_CHARS:
                break

        full_text = "\n\n".join(chunks).strip()
        if not full_text:
            raise RuntimeError(f"No extractable text found in PDF {pdf_url}")

        full_text = self._truncate_text(full_text)
        self.logger.info(f"Extracted {len(full_text)} characters from PDF {pdf_url}")
        return full_text

    async def summarize(self, paper_id: int, mode: str = "brief") -> str:
        """Generate a summary for a paper.

        1. Check cache: self.store.get_summary(paper_id, mode). If exists, return it.
        2. Get paper info from store.
        3. Fetch full text via fetch_paper_text(paper["ar5iv_url"]).
           If fetch fails, fall back to using just the abstract.
        4. Build prompt based on mode:
           - "brief": summarize in 1-2 paragraphs covering core contributions and methodology.
           - "detailed": structured summary with Motivation, Method, Experiments,
             Conclusions, Limitations.
        5. Call self.llm.complete(prompt, system=...).
        6. Save to cache: self.store.save_summary(paper_id, mode, result, llm_provider_name).
        7. Return the summary text.
        """
        # Check cache
        cached = self.store.get_summary(paper_id, mode)
        if cached:
            self.logger.info(f"Cache hit for paper {paper_id} ({mode} summary)")
            return cached["content"]

        # Get paper info by integer id
        paper = self._get_paper_by_id(paper_id)
        if paper is None:
            raise ValueError(f"Paper with id {paper_id} not found in database")

        # Fetch full text, fall back to abstract on failure
        paper_text = None
        if paper.get("ar5iv_url"):
            try:
                paper_text = self.fetch_paper_text(paper["ar5iv_url"])
                self.logger.info(f"Fetched ar5iv full text for paper {paper_id}")
            except RuntimeError as e:
                self.logger.warning(f"Failed ar5iv extraction for paper {paper_id}: {e}")

        if not paper_text and paper.get("pdf_url"):
            try:
                paper_text = self.fetch_pdf_text(paper["pdf_url"])
                self.logger.info(f"Fetched PDF full text for paper {paper_id}")
            except RuntimeError as e:
                self.logger.warning(f"Failed PDF extraction for paper {paper_id}: {e}")

        if not paper_text:
            paper_text = paper.get("abstract", "")
            self.logger.info(f"Using abstract for paper {paper_id}")

        # Build prompt based on mode and language
        if mode == "brief_zh":
            system = "你是一位科学论文摘要专家。请提供清晰、准确的中文摘要。"
            prompt = (
                f"请用中文对以下论文进行 1-2 段的简要总结，"
                f"涵盖核心贡献和方法论。\n\n"
                f"标题: {paper['title']}\n\n"
                f"论文内容:\n{paper_text}"
            )
        elif mode == "detailed_zh":
            system = "你是一位科学论文摘要专家。请提供清晰、准确的中文结构化摘要。"
            prompt = (
                f"请用中文对以下论文进行结构化总结，包含以下部分：\n"
                f"- **研究动机**: 为什么要做这项研究？\n"
                f"- **研究方法**: 采用了什么方法？\n"
                f"- **实验结果**: 进行了哪些实验，结果如何？\n"
                f"- **主要结论**: 主要的发现和结论是什么？\n"
                f"- **局限性**: 这项工作有哪些局限性？\n\n"
                f"标题: {paper['title']}\n\n"
                f"论文内容:\n{paper_text}"
            )
        elif mode == "brief":
            system = "You are a scientific paper summarizer. Provide clear, accurate summaries."
            prompt = (
                f"Summarize this paper in 1-2 paragraphs covering core contributions "
                f"and methodology.\n\n"
                f"Title: {paper['title']}\n\n"
                f"Paper content:\n{paper_text}"
            )
        else:  # detailed
            system = "You are a scientific paper summarizer. Provide clear, accurate summaries."
            prompt = (
                f"Provide a structured summary of this paper with the following sections:\n"
                f"- **Motivation**: Why was this work done?\n"
                f"- **Method**: What approach was used?\n"
                f"- **Experiments**: What experiments were conducted and what were the results?\n"
                f"- **Conclusions**: What are the main takeaways?\n"
                f"- **Limitations**: What are the limitations of this work?\n\n"
                f"Title: {paper['title']}\n\n"
                f"Paper content:\n{paper_text}"
            )

        # Call LLM
        summary = await self.llm.complete(prompt, system=system)

        # Save to cache
        llm_provider_name = type(self.llm).__name__
        self.store.save_summary(paper_id, mode, summary, llm_provider_name)
        self.logger.info(f"Generated and cached {mode} summary for paper {paper_id}")

        return summary

    def _get_paper_by_id(self, paper_id: int) -> dict | None:
        """Get a paper by its integer id. Uses direct SQL query via store's connection."""
        import json

        conn = self.store._get_conn()
        try:
            row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["authors"] = json.loads(d["authors"])
            d["categories"] = json.loads(d["categories"])
            return d
        finally:
            conn.close()
