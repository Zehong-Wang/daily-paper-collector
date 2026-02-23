from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, timedelta
from typing import Optional

import arxiv
import feedparser


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
        """Fetch today's announced papers from all configured categories via RSS.

        arXiv RSS feeds list papers by their announcement date (the date they
        appear on the "new" listings page), which differs from their submission
        date. This ensures we get the same papers visible on arxiv.org/list/.

        Falls back to the REST API if RSS returns no results (e.g., feed not
        yet updated or network issue).

        Args:
            cutoff_days: Ignored for RSS mode. Retained for backward
                         compatibility and used only in REST API fallback.

        Return a list of dicts with keys: arxiv_id, title, authors, abstract,
        categories, published_date, pdf_url, ar5iv_url.

        Deduplicate across categories by arxiv_id.
        """
        self.logger.info(
            "Fetching today's papers from %d categories via RSS",
            len(self.categories),
        )

        loop = asyncio.get_event_loop()
        all_papers = []
        for category in self.categories:
            papers = await loop.run_in_executor(
                None, self._fetch_category_rss, category
            )
            all_papers.extend(papers)

        deduplicated = self._deduplicate(all_papers)
        self.logger.info(
            "RSS: %d papers fetched, %d after deduplication",
            len(all_papers),
            len(deduplicated),
        )

        if deduplicated:
            return deduplicated

        # Fallback to REST API if RSS returned nothing
        self.logger.warning(
            "RSS returned 0 papers. Falling back to REST API with submittedDate filter."
        )
        return await self._fetch_via_rest_api(cutoff_days)

    def _fetch_category_rss(self, category: str) -> list[dict]:
        """Fetch new papers for a single category from the arXiv RSS feed.

        Only includes entries with announce_type 'new' or 'cross' (new
        submissions and cross-listed papers). Skips 'replace' entries
        (updated versions of existing papers).

        On failure, logs the error and returns an empty list.
        """
        feed_url = f"https://rss.arxiv.org/rss/{category}"
        self.logger.info("Fetching RSS: %s", feed_url)

        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            self.logger.error("Failed to parse RSS for %s: %s", category, e)
            return []

        if feed.bozo and not feed.entries:
            self.logger.error(
                "RSS feed error for %s: %s", category, feed.bozo_exception
            )
            return []

        self.logger.info("  %s: %d entries in RSS feed", category, len(feed.entries))

        papers = []
        for entry in feed.entries:
            # Only include new submissions and cross-listings
            announce_type = getattr(entry, "arxiv_announce_type", "new")
            if announce_type not in ("new", "cross"):
                continue

            # Extract arxiv_id from the entry id (format: oai:arXiv.org:2602.17676v1)
            raw_id = entry.get("id", "")
            if ":" in raw_id:
                raw_id = raw_id.split(":")[-1]
            arxiv_id = re.sub(r"v\d+$", "", raw_id)

            if not arxiv_id:
                continue

            # Extract abstract from summary (format: "arXiv:ID Announce Type: ...\nAbstract: ...")
            summary = entry.get("summary", "")
            abstract = self._extract_abstract_from_rss(summary)

            # Parse authors: RSS gives a single comma-separated string
            author_str = entry.get("author", "")
            authors = [a.strip() for a in author_str.split(",") if a.strip()]

            # Categories from tags
            categories = [
                tag["term"] for tag in entry.get("tags", []) if tag.get("term")
            ]

            # Use today's date as the published date (announcement date)
            published_date = date.today().isoformat()

            papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": entry.get("title", "").replace("\n", " ").strip(),
                    "authors": authors,
                    "abstract": abstract,
                    "categories": categories,
                    "published_date": published_date,
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "ar5iv_url": f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
                }
            )

        self.logger.info(
            "  %s: %d new/cross papers from RSS", category, len(papers)
        )
        return papers

    def _extract_abstract_from_rss(self, summary: str) -> str:
        """Extract the abstract text from an RSS summary field.

        RSS summaries have the format:
          arXiv:2602.17676v1 Announce Type: new
          Abstract: The rapid deployment of ...

        This strips the prefix and returns just the abstract text.
        Also handles cases where there's an HTML <p> wrapper.
        """
        # Remove HTML tags if present
        text = re.sub(r"<[^>]+>", "", summary).strip()

        # Try to find "Abstract:" prefix and extract what follows
        match = re.search(r"Abstract:\s*", text)
        if match:
            return text[match.end():].replace("\n", " ").strip()

        # Fallback: return the full text cleaned up
        return text.replace("\n", " ").strip()

    async def _fetch_via_rest_api(
        self, cutoff_days: Optional[int] = None
    ) -> list[dict]:
        """Fallback: fetch papers using the REST API with submittedDate filter.

        Note: submittedDate != announcement date. Papers submitted on one day
        may be announced days later. This is kept as a fallback only.
        """
        if cutoff_days is None:
            cutoff_days = self.default_cutoff_days

        today = date.today()
        start_date = today - timedelta(days=cutoff_days)
        end_date = today

        self.logger.info(
            "REST API fallback: %d categories (date range: %s to %s)",
            len(self.categories),
            start_date.isoformat(),
            end_date.isoformat(),
        )

        all_papers = []
        loop = asyncio.get_event_loop()
        for category in self.categories:
            papers = await loop.run_in_executor(
                None, self._fetch_category_rest, category, start_date, end_date
            )
            all_papers.extend(papers)

        deduplicated = self._deduplicate(all_papers)
        self.logger.info(
            "REST API: %d papers fetched, %d after deduplication",
            len(all_papers),
            len(deduplicated),
        )
        return deduplicated

    def _build_date_query(
        self, category: str, start_date: date, end_date: date
    ) -> str:
        """Build an arXiv API query string with server-side date filtering."""
        start_str = start_date.strftime("%Y%m%d") + "0000"
        end_str = end_date.strftime("%Y%m%d") + "2359"
        return f"cat:{category} AND submittedDate:[{start_str} TO {end_str}]"

    def _fetch_category_rest(
        self, category: str, start_date: date, end_date: date
    ) -> list[dict]:
        """Fetch papers for a single category using the REST API.

        On failure, logs the error and returns an empty list.
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
