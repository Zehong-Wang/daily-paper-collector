from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, timedelta
from typing import Optional

import arxiv


class ArxivFetcher:
    def __init__(self, config: dict):
        self.categories = config["arxiv"]["categories"]
        self.max_results = config["arxiv"].get("max_results_per_category", 500)
        self.default_cutoff_days = config["arxiv"].get("cutoff_days", 1)
        page_size = config["arxiv"].get("page_size", 500)
        self.client = arxiv.Client(
            page_size=page_size,
            delay_seconds=3.0,
            num_retries=3,
        )
        self.logger = logging.getLogger(__name__)

    async def fetch_today(self, cutoff_days: Optional[int] = None) -> list[dict]:
        """Fetch papers from all configured categories.

        Uses server-side date filtering via arXiv's submittedDate query syntax
        combined with client-side filtering as a safety net.

        Args:
            cutoff_days: Number of days back from today to include.
                         Defaults to config value (arxiv.cutoff_days, default 1).
                         Value of 1 means yesterday+today (arXiv listing cycle).
                         Value of 0 means today only (UTC).

        Return a list of dicts with keys: arxiv_id, title, authors, abstract,
        categories, published_date, pdf_url, ar5iv_url.

        Deduplicate across categories by arxiv_id (a paper can appear in multiple categories).
        """
        if cutoff_days is None:
            cutoff_days = self.default_cutoff_days

        today = date.today()
        start_date = today - timedelta(days=cutoff_days)
        end_date = today

        self.logger.info(
            "Fetching papers from %d categories (date range: %s to %s, cutoff_days=%d)",
            len(self.categories),
            start_date.isoformat(),
            end_date.isoformat(),
            cutoff_days,
        )

        all_papers = []
        loop = asyncio.get_event_loop()
        for category in self.categories:
            papers = await loop.run_in_executor(
                None, self._fetch_category, category, start_date, end_date
            )
            all_papers.extend(papers)

        deduplicated = self._deduplicate(all_papers)
        self.logger.info(
            "Total: %d papers fetched, %d after deduplication",
            len(all_papers),
            len(deduplicated),
        )
        return deduplicated

    def _build_date_query(self, category: str, start_date: date, end_date: date) -> str:
        """Build an arXiv API query string with server-side date filtering.

        Uses the submittedDate field with format YYYYMMDDHHMM (12 digits).
        Start is at 00:00 UTC of start_date, end is at 23:59 UTC of end_date.
        """
        start_str = start_date.strftime("%Y%m%d") + "0000"
        end_str = end_date.strftime("%Y%m%d") + "2359"
        return f"cat:{category} AND submittedDate:[{start_str} TO {end_str}]"

    def _fetch_category(self, category: str, start_date: date, end_date: date) -> list[dict]:
        """Fetch papers for a single category with server-side date filtering.

        Builds a query with submittedDate range to let the arXiv API filter
        server-side. Applies a client-side date filter as a safety net
        (published_date >= start_date).

        On failure, logs the error and returns an empty list (doesn't fail the whole run).
        """
        query = self._build_date_query(category, start_date, end_date)
        self.logger.info(
            "Fetching category: %s (query=%r, max_results=%s)",
            category,
            query,
            self.max_results,
        )

        try:
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            results = list(self.client.results(search))
        except Exception as e:
            self.logger.error("Failed to fetch category %s: %s", category, e)
            return []

        self.logger.info("  %s: %d results from API", category, len(results))

        papers = []
        for result in results:
            published_date = result.published.date()
            # Client-side safety net: skip papers outside our date range
            if published_date < start_date:
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
            "  %s: %d papers after client-side date filter (start=%s)",
            category,
            len(papers),
            start_date,
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
