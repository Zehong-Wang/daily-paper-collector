import logging

import requests
from bs4 import BeautifulSoup


class PaperSummarizer:
    def __init__(self, llm, store):
        self.llm = llm
        self.store = store
        self.logger = logging.getLogger(__name__)

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
            response = requests.get(ar5iv_url, timeout=30)
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

        # Truncate to 15000 characters
        if len(full_text) > 15000:
            full_text = full_text[:15000]

        self.logger.info(f"Extracted {len(full_text)} characters from {ar5iv_url}")
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
                self.logger.info(f"Fetched full text for paper {paper_id}")
            except RuntimeError as e:
                self.logger.warning(f"Failed to fetch full text, using abstract: {e}")

        if not paper_text:
            paper_text = paper.get("abstract", "")
            self.logger.info(f"Using abstract for paper {paper_id}")

        # Build prompt
        system = "You are a scientific paper summarizer. Provide clear, accurate summaries."

        if mode == "brief":
            prompt = (
                f"Summarize this paper in 1-2 paragraphs covering core contributions "
                f"and methodology.\n\n"
                f"Title: {paper['title']}\n\n"
                f"Paper content:\n{paper_text}"
            )
        else:  # detailed
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
