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
            self.store.save_report(run_date, general, "", len(papers), 0)
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
            "Found %d new papers with embeddings for matching", len(recent_papers),
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

        # Step 11: Email
        email_sent = False
        if self.config.get("email", {}).get("enabled", False):
            try:
                await self.email_sender.send(general, specific, ranked, run_date)
                email_sent = True
                self.logger.info("Email sent successfully")
            except Exception as e:
                self.logger.error("Email sending failed: %s", e)

        # Step 12: Save report
        self.store.save_report(run_date, general, specific, len(papers), len(ranked))

        return {
            "date": run_date,
            "papers_fetched": len(papers),
            "new_papers": len(new_papers),
            "matches": len(ranked),
            "email_sent": email_sent,
        }
