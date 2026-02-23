import logging
from datetime import date

from src.store.database import PaperStore
from src.fetcher.arxiv_fetcher import ArxivFetcher
from src.matcher.embedder import Embedder
from src.matcher.ranker import LLMRanker
from src.interest.manager import InterestManager
from src.report.generator import ReportGenerator
from src.email.sender import EmailSender
from src.llm import create_llm_provider


class DailyPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.store = PaperStore(config["database"]["path"])
        self.fetcher = ArxivFetcher(config)
        self.embedder = Embedder(config)
        self.llm = create_llm_provider(config)
        self.ranker = LLMRanker(self.llm, config)
        self.interest_mgr = InterestManager(self.store, self.embedder)
        self.report_gen = ReportGenerator(self.llm)
        self.email_sender = EmailSender(config)
        self.logger = logging.getLogger(__name__)

        # Read max_concurrent from the active LLM provider's config
        provider_name = config.get("llm", {}).get("provider", "openai")
        provider_config = config.get("llm", {}).get(provider_name, {})
        self.max_concurrent = provider_config.get("max_concurrent", 5)

        # Chinese report generation toggle
        self.chinese_enabled = config.get("report", {}).get("chinese", False)

    async def run(self) -> dict:
        """Execute the full daily pipeline. Return a summary dict.

        Steps:
        1. Fetch papers from arXiv.
        2. Save to DB (deduplication handled by store).
        3. Compute embeddings for new papers.
        4. Get all interests with embeddings.
        5. If no interests exist, log a warning and skip matching.
        6. Run embedding similarity matching → top-N candidates.
        7. Run LLM re-ranking → top-K results.
        8. Save match records to DB.
        9. Generate general report.
        10. Generate specific report.
        11. Send email (if email.enabled is true in config).
        12. Save report to DB.
        13. Return summary dict.
        """
        run_date = date.today().isoformat()
        self.logger.info("Starting daily pipeline for %s", run_date)

        # Step 1: Fetch
        self.logger.info("Fetching papers from arXiv...")
        papers = await self.fetcher.fetch_today()
        self.logger.info("Fetched %d papers", len(papers))

        # Step 2: Save
        new_papers = self.store.save_papers(papers)
        self.logger.info(
            "Saved %d new papers (%d duplicates)", len(new_papers), len(papers) - len(new_papers)
        )

        # Step 3: Embed new papers that don't have embeddings yet
        self.logger.info("Computing embeddings for papers without embeddings...")
        papers_to_embed = self.store.get_papers_without_embeddings()
        self.embedder.compute_embeddings(papers_to_embed, self.store)

        # Step 4: Interests
        interests = self.interest_mgr.get_interests_with_embeddings()
        if not interests:
            self.logger.warning("No interests configured. Skipping matching.")
            # Still generate general report using all fetched papers
            general = await self.report_gen.generate_general(papers, run_date)
            general_zh = None
            if self.chinese_enabled:
                general_zh = await self.report_gen.generate_general_zh(papers, run_date)
            self.store.save_report(
                run_date,
                general,
                "",
                len(papers),
                0,
                general_report_zh=general_zh,
            )
            return {
                "date": run_date,
                "papers_fetched": len(papers),
                "new_papers": len(new_papers),
                "matches": 0,
                "email_sent": False,
            }

        # Step 5-6: Match only newly inserted papers (avoids overlap with previous runs)
        new_paper_ids = [p["id"] for p in new_papers]
        recent_papers = self.store.get_papers_by_ids_with_embeddings(new_paper_ids)
        self.logger.info(
            "Found %d new papers with embeddings for matching",
            len(recent_papers),
        )
        top_n = self.config["matching"]["embedding_top_n"]
        threshold = self.config["matching"]["similarity_threshold"]
        candidates = self.embedder.find_similar(interests, recent_papers, top_n, threshold)
        self.logger.info("Embedding matcher found %d candidates", len(candidates))

        # Step 7: Re-rank
        ranked = await self.ranker.rerank(candidates, interests, max_concurrent=self.max_concurrent)
        self.logger.info("LLM re-ranker selected %d papers", len(ranked))

        # Step 8: Save matches
        for paper in ranked:
            self.store.save_match(
                paper["id"],
                run_date,
                paper.get("embedding_score", 0),
                paper.get("llm_score", 0),
                paper.get("llm_reason", ""),
            )

        # Step 9-10: Reports (use all fetched papers, not just new ones)
        general = await self.report_gen.generate_general(papers, run_date)
        specific = await self.report_gen.generate_specific(ranked, interests, run_date)

        # Step 9b-10b: Chinese reports (if enabled)
        general_zh = None
        specific_zh = None
        if self.chinese_enabled:
            self.logger.info("Generating Chinese reports...")
            general_zh = await self.report_gen.generate_general_zh(papers, run_date)
            specific_zh = await self.report_gen.generate_specific_zh(ranked, interests, run_date)

        # Step 11: Email
        email_sent = False
        if self.config.get("email", {}).get("enabled", False):
            try:
                await self.email_sender.send(
                    general,
                    specific,
                    ranked,
                    run_date,
                    general_zh=general_zh,
                    specific_zh=specific_zh,
                )
                email_sent = True
                self.logger.info("Email sent successfully")
            except Exception as e:
                self.logger.error("Email sending failed: %s", e)

        # Step 12: Save report
        self.store.save_report(
            run_date,
            general,
            specific,
            len(papers),
            len(ranked),
            general_report_zh=general_zh,
            specific_report_zh=specific_zh,
        )

        return {
            "date": run_date,
            "papers_fetched": len(papers),
            "new_papers": len(new_papers),
            "matches": len(ranked),
            "email_sent": email_sent,
        }

    async def run_range_report(
        self, start_date: str, end_date: str, report_type: str
    ) -> dict:
        """Generate a report for papers in the given date range.

        Unlike run(), this does NOT fetch new papers from arXiv or compute
        embeddings. It re-runs the matching pipeline on all papers already
        in the DB for the date range.

        Args:
            start_date: ISO date string, e.g. "2026-02-20"
            end_date: ISO date string, e.g. "2026-02-22"
            report_type: "3day" or "1week"

        Returns:
            Summary dict with keys: date_range, report_type, papers_count, matches
        """
        date_label = f"{start_date} ~ {end_date}"
        run_date_label = f"{start_date}~{end_date}"
        self.logger.info("Starting %s report for %s", report_type, date_label)

        # Step 1: Get all papers in range that have embeddings
        papers_in_range = self.store.get_papers_in_date_range_with_embeddings(
            start_date, end_date
        )
        self.logger.info(
            "Found %d papers with embeddings in range %s",
            len(papers_in_range),
            date_label,
        )

        if not papers_in_range:
            self.logger.warning("No papers found in date range %s", date_label)
            return {
                "date_range": date_label,
                "report_type": report_type,
                "papers_count": 0,
                "matches": 0,
            }

        # Step 2: Get interests
        interests = self.interest_mgr.get_interests_with_embeddings()
        if not interests:
            self.logger.warning("No interests configured. Generating general report only.")
            general = await self.report_gen.generate_general(
                papers_in_range, run_date_label, date_label=date_label
            )
            general_zh = None
            if self.chinese_enabled:
                general_zh = await self.report_gen.generate_general_zh(
                    papers_in_range, run_date_label, date_label=date_label
                )
            self.store.save_report(
                run_date_label,
                general,
                "",
                len(papers_in_range),
                0,
                general_report_zh=general_zh,
                report_type=report_type,
            )
            return {
                "date_range": date_label,
                "report_type": report_type,
                "papers_count": len(papers_in_range),
                "matches": 0,
            }

        # Step 3: Embedding similarity matching
        top_n = self.config["matching"]["embedding_top_n"]
        threshold = self.config["matching"]["similarity_threshold"]
        candidates = self.embedder.find_similar(
            interests, papers_in_range, top_n, threshold
        )
        self.logger.info("Embedding matcher found %d candidates", len(candidates))

        # Step 4: LLM re-ranking
        ranked = await self.ranker.rerank(
            candidates, interests, max_concurrent=self.max_concurrent
        )
        self.logger.info("LLM re-ranker selected %d papers", len(ranked))

        # Step 5: Save matches (using range label as run_date)
        for paper in ranked:
            self.store.save_match(
                paper["id"],
                run_date_label,
                paper.get("embedding_score", 0),
                paper.get("llm_score", 0),
                paper.get("llm_reason", ""),
            )

        # Step 6: Generate reports
        general = await self.report_gen.generate_general(
            papers_in_range, run_date_label, date_label=date_label
        )
        specific = await self.report_gen.generate_specific(
            ranked, interests, run_date_label, date_label=date_label
        )

        general_zh = None
        specific_zh = None
        if self.chinese_enabled:
            self.logger.info("Generating Chinese reports for range %s...", date_label)
            general_zh = await self.report_gen.generate_general_zh(
                papers_in_range, run_date_label, date_label=date_label
            )
            specific_zh = await self.report_gen.generate_specific_zh(
                ranked, interests, run_date_label, date_label=date_label
            )

        # Step 7: Save report
        self.store.save_report(
            run_date_label,
            general,
            specific,
            len(papers_in_range),
            len(ranked),
            general_report_zh=general_zh,
            specific_report_zh=specific_zh,
            report_type=report_type,
        )

        return {
            "date_range": date_label,
            "report_type": report_type,
            "papers_count": len(papers_in_range),
            "matches": len(ranked),
        }
