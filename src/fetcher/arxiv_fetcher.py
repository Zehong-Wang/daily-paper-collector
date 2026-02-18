import asyncio
import logging
import re
from datetime import date, timedelta

import arxiv


class ArxivFetcher:
    def __init__(self, config: dict):
        self.categories = config["arxiv"]["categories"]
        self.max_results = config["arxiv"]["max_results_per_category"]
        self.logger = logging.getLogger(__name__)

    async def fetch_today(self, cutoff_days: int = 2) -> list[dict]:
        """Fetch papers from all configured categories.

        For each category, query arXiv sorted by SubmittedDate, then filter in Python
        to only keep papers published within the last `cutoff_days` days
        (default 2, to account for timezone and indexing delays).

        Return a list of dicts with keys: arxiv_id, title, authors, abstract,
        categories, published_date, pdf_url, ar5iv_url.

        Deduplicate across categories by arxiv_id (a paper can appear in multiple categories).
        """
        cutoff_date = date.today() - timedelta(days=cutoff_days)
        self.logger.info(
            f"Fetching papers from {len(self.categories)} categories (cutoff_date={cutoff_date})"
        )

        all_papers = []
        for category in self.categories:
            papers = await asyncio.to_thread(self._fetch_category, category, cutoff_date)
            all_papers.extend(papers)

        deduplicated = self._deduplicate(all_papers)
        self.logger.info(
            f"Total: {len(all_papers)} papers fetched, {len(deduplicated)} after deduplication"
        )
        return deduplicated

    def _fetch_category(self, category: str, cutoff_date: date) -> list[dict]:
        """Fetch papers for a single category.

        Uses arxiv.Client() and arxiv.Search to fetch papers sorted by SubmittedDate.
        Filters in Python to only keep papers published on or after cutoff_date.
        """
        self.logger.info(f"Fetching category: {category} (max_results={self.max_results})")

        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        client = arxiv.Client()
        results = list(client.results(search))
        self.logger.info(f"  {category}: {len(results)} results from API")

        papers = []
        for result in results:
            published_date = result.published.date()
            if published_date < cutoff_date:
                continue

            raw_id = result.entry_id.split("/")[-1]
            arxiv_id = re.sub(r"v\d+$", "", raw_id)

            papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": result.title.replace("\n", " ").strip(),
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary.replace("\n", " ").strip(),
                    "categories": result.categories,
                    "published_date": published_date.isoformat(),
                    "pdf_url": result.pdf_url,
                    "ar5iv_url": f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
                }
            )

        self.logger.info(
            f"  {category}: {len(papers)} papers after date filtering (cutoff={cutoff_date})"
        )
        return papers

    def _deduplicate(self, papers: list[dict]) -> list[dict]:
        """Remove duplicates by arxiv_id, keeping the first occurrence."""
        seen = set()
        unique = []
        for paper in papers:
            if paper["arxiv_id"] not in seen:
                seen.add(paper["arxiv_id"])
                unique.append(paper)
        return unique
