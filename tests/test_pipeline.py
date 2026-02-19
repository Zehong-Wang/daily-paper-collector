import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline import DailyPipeline


def _make_config(tmp_path):
    """Build a minimal config dict for testing."""
    return {
        "database": {"path": str(tmp_path / "test.db")},
        "arxiv": {"categories": ["cs.AI"], "max_results_per_category": 100},
        "matching": {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_top_n": 50,
            "llm_top_k": 10,
            "similarity_threshold": 0.3,
        },
        "llm": {
            "provider": "openai",
            "openai": {"model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
        },
        "email": {
            "enabled": True,
            "smtp": {
                "host": "smtp.gmail.com",
                "port": 587,
                "username_env": "EMAIL_USERNAME",
                "password_env": "EMAIL_PASSWORD",
            },
            "from": "test@example.com",
            "to": ["recipient@example.com"],
            "subject_prefix": "[Test]",
        },
        "scheduler": {"cron": "0 8 * * *"},
    }


def _make_fake_papers(count=5):
    """Create a list of fake paper dicts."""
    today = date.today().isoformat()
    return [
        {
            "arxiv_id": f"2501.{10000 + i}",
            "title": f"Test Paper {i}",
            "authors": ["Author A", "Author B"],
            "abstract": f"This is the abstract for test paper {i} about machine learning.",
            "categories": ["cs.AI", "cs.LG"],
            "published_date": today,
            "pdf_url": f"https://arxiv.org/pdf/2501.{10000 + i}",
            "ar5iv_url": f"https://ar5iv.labs.arxiv.org/html/2501.{10000 + i}",
        }
        for i in range(count)
    ]


def _make_saved_papers(papers, start_id=1):
    """Add 'id' field to papers, simulating what save_papers returns."""
    return [{**p, "id": start_id + i} for i, p in enumerate(papers)]


def _make_candidates(papers, start_id=1):
    """Add 'id' and 'embedding_score' to papers, simulating embedding matcher output."""
    return [
        {**p, "id": start_id + i, "embedding_score": 0.8 - i * 0.1, "embedding": b"fake"}
        for i, p in enumerate(papers)
    ]


def _make_ranked(candidates):
    """Add 'llm_score' and 'llm_reason' to candidates, simulating ranker output."""
    return [
        {**c, "llm_score": 9.0 - i * 0.5, "llm_reason": f"Relevant because {i}"}
        for i, c in enumerate(candidates[:2])
    ]


def _make_interests():
    """Create fake interest dicts with embeddings."""
    return [
        {"id": 1, "type": "keyword", "value": "machine learning", "embedding": b"blob1"},
        {"id": 2, "type": "keyword", "value": "transformers", "embedding": b"blob2"},
    ]


@pytest.fixture
def config(tmp_path):
    return _make_config(tmp_path)


class TestDailyPipelineFullRun:
    """Test the full pipeline with all components mocked."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, config, tmp_path):
        """Test the full happy path: fetch → save → embed → match → rank → report → email."""
        fake_papers = _make_fake_papers(5)
        saved_papers = _make_saved_papers(fake_papers[:3])  # 2 duplicates
        interests = _make_interests()
        candidates = _make_candidates(fake_papers[:4])
        ranked = _make_ranked(candidates)

        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.Embedder") as MockEmbedder,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("src.pipeline.LLMRanker") as MockRanker,
            patch("src.pipeline.InterestManager") as MockInterestMgr,
            patch("src.pipeline.ReportGenerator") as MockReportGen,
            patch("src.pipeline.EmailSender") as MockEmailSender,
        ):
            # Configure fetcher mock
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_today = AsyncMock(return_value=fake_papers)

            # Configure embedder mock
            mock_embedder = MockEmbedder.return_value
            mock_embedder.compute_embeddings = MagicMock()
            mock_embedder.find_similar = MagicMock(return_value=candidates)

            # Configure LLM mock
            mock_llm = MagicMock()
            mock_create_llm.return_value = mock_llm

            # Configure ranker mock
            mock_ranker = MockRanker.return_value
            mock_ranker.rerank = AsyncMock(return_value=ranked)

            # Configure interest manager mock
            mock_interest_mgr = MockInterestMgr.return_value
            mock_interest_mgr.get_interests_with_embeddings = MagicMock(return_value=interests)

            # Configure report generator mock
            mock_report_gen = MockReportGen.return_value
            mock_report_gen.generate_general = AsyncMock(return_value="# General Report")
            mock_report_gen.generate_specific = AsyncMock(return_value="## Specific Report")

            # Configure email sender mock
            mock_email_sender = MockEmailSender.return_value
            mock_email_sender.send = AsyncMock()

            # Create pipeline — the constructor will use our mocked classes
            pipeline = DailyPipeline(config)

            # Mock the store's save_papers to return only new papers
            pipeline.store.save_papers = MagicMock(return_value=saved_papers)
            pipeline.store.get_papers_by_date_with_embeddings = MagicMock(
                return_value=candidates
            )

            result = await pipeline.run()

            # Verify fetcher was called
            mock_fetcher.fetch_today.assert_awaited_once()

            # Verify store.save_papers was called with the fetched papers
            pipeline.store.save_papers.assert_called_once_with(fake_papers)

            # Verify embeddings computed for new papers
            mock_embedder.compute_embeddings.assert_called_once_with(
                saved_papers, pipeline.store
            )

            # Verify ranker was called with candidates and interests
            mock_ranker.rerank.assert_awaited_once_with(candidates, interests)

            # Verify reports were generated
            mock_report_gen.generate_general.assert_awaited_once()
            mock_report_gen.generate_specific.assert_awaited_once()

            # Verify email was sent (email.enabled = True)
            mock_email_sender.send.assert_awaited_once()

            # Verify result dict
            assert result["papers_fetched"] == 5
            assert result["new_papers"] == 3
            assert result["matches"] == 2
            assert result["email_sent"] is True
            assert result["date"] == date.today().isoformat()

    @pytest.mark.asyncio
    async def test_no_interests_skips_matching(self, config, tmp_path):
        """When no interests exist, matching and re-ranking are skipped."""
        fake_papers = _make_fake_papers(5)
        saved_papers = _make_saved_papers(fake_papers)

        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.Embedder") as MockEmbedder,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("src.pipeline.LLMRanker") as MockRanker,
            patch("src.pipeline.InterestManager") as MockInterestMgr,
            patch("src.pipeline.ReportGenerator") as MockReportGen,
            patch("src.pipeline.EmailSender") as MockEmailSender,
        ):
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_today = AsyncMock(return_value=fake_papers)

            mock_embedder = MockEmbedder.return_value
            mock_embedder.compute_embeddings = MagicMock()
            mock_embedder.find_similar = MagicMock()

            mock_create_llm.return_value = MagicMock()

            mock_ranker = MockRanker.return_value
            mock_ranker.rerank = AsyncMock()

            mock_interest_mgr = MockInterestMgr.return_value
            mock_interest_mgr.get_interests_with_embeddings = MagicMock(return_value=[])

            mock_report_gen = MockReportGen.return_value
            mock_report_gen.generate_general = AsyncMock(return_value="# General Report")

            mock_email_sender = MockEmailSender.return_value

            pipeline = DailyPipeline(config)
            pipeline.store.save_papers = MagicMock(return_value=saved_papers)

            result = await pipeline.run()

            # Matching should be skipped
            mock_embedder.find_similar.assert_not_called()
            mock_ranker.rerank.assert_not_awaited()

            # General report should still be generated
            mock_report_gen.generate_general.assert_awaited_once()

            # No specific report generated
            mock_report_gen.generate_specific.assert_not_called()

            # Email not sent (matching skipped early)
            assert result["matches"] == 0
            assert result["email_sent"] is False

    @pytest.mark.asyncio
    async def test_email_disabled_skips_sending(self, config, tmp_path):
        """When email.enabled is false, email is not sent."""
        config["email"]["enabled"] = False
        fake_papers = _make_fake_papers(3)
        saved_papers = _make_saved_papers(fake_papers)
        interests = _make_interests()
        candidates = _make_candidates(fake_papers[:2])
        ranked = _make_ranked(candidates)

        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.Embedder") as MockEmbedder,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("src.pipeline.LLMRanker") as MockRanker,
            patch("src.pipeline.InterestManager") as MockInterestMgr,
            patch("src.pipeline.ReportGenerator") as MockReportGen,
            patch("src.pipeline.EmailSender") as MockEmailSender,
        ):
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_today = AsyncMock(return_value=fake_papers)

            mock_embedder = MockEmbedder.return_value
            mock_embedder.compute_embeddings = MagicMock()
            mock_embedder.find_similar = MagicMock(return_value=candidates)

            mock_create_llm.return_value = MagicMock()

            mock_ranker = MockRanker.return_value
            mock_ranker.rerank = AsyncMock(return_value=ranked)

            mock_interest_mgr = MockInterestMgr.return_value
            mock_interest_mgr.get_interests_with_embeddings = MagicMock(return_value=interests)

            mock_report_gen = MockReportGen.return_value
            mock_report_gen.generate_general = AsyncMock(return_value="# General")
            mock_report_gen.generate_specific = AsyncMock(return_value="## Specific")

            mock_email_sender = MockEmailSender.return_value
            mock_email_sender.send = AsyncMock()

            pipeline = DailyPipeline(config)
            pipeline.store.save_papers = MagicMock(return_value=saved_papers)
            pipeline.store.get_papers_by_date_with_embeddings = MagicMock(
                return_value=candidates
            )

            result = await pipeline.run()

            # Email should NOT be sent
            mock_email_sender.send.assert_not_awaited()
            assert result["email_sent"] is False
            # But matches should still be saved
            assert result["matches"] == 2

    @pytest.mark.asyncio
    async def test_email_failure_does_not_crash_pipeline(self, config, tmp_path):
        """When email sending fails, the pipeline continues and saves the report."""
        fake_papers = _make_fake_papers(3)
        saved_papers = _make_saved_papers(fake_papers)
        interests = _make_interests()
        candidates = _make_candidates(fake_papers[:2])
        ranked = _make_ranked(candidates)

        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.Embedder") as MockEmbedder,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("src.pipeline.LLMRanker") as MockRanker,
            patch("src.pipeline.InterestManager") as MockInterestMgr,
            patch("src.pipeline.ReportGenerator") as MockReportGen,
            patch("src.pipeline.EmailSender") as MockEmailSender,
        ):
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_today = AsyncMock(return_value=fake_papers)

            mock_embedder = MockEmbedder.return_value
            mock_embedder.compute_embeddings = MagicMock()
            mock_embedder.find_similar = MagicMock(return_value=candidates)

            mock_create_llm.return_value = MagicMock()

            mock_ranker = MockRanker.return_value
            mock_ranker.rerank = AsyncMock(return_value=ranked)

            mock_interest_mgr = MockInterestMgr.return_value
            mock_interest_mgr.get_interests_with_embeddings = MagicMock(return_value=interests)

            mock_report_gen = MockReportGen.return_value
            mock_report_gen.generate_general = AsyncMock(return_value="# General")
            mock_report_gen.generate_specific = AsyncMock(return_value="## Specific")

            # Email sender raises an exception
            mock_email_sender = MockEmailSender.return_value
            mock_email_sender.send = AsyncMock(side_effect=Exception("SMTP auth failed"))

            pipeline = DailyPipeline(config)
            pipeline.store.save_papers = MagicMock(return_value=saved_papers)
            pipeline.store.get_papers_by_date_with_embeddings = MagicMock(
                return_value=candidates
            )

            result = await pipeline.run()

            # Pipeline should not crash
            assert result["email_sent"] is False
            assert result["matches"] == 2

    @pytest.mark.asyncio
    async def test_save_match_called_per_ranked_paper(self, config, tmp_path):
        """Verify save_match is called once per ranked paper with correct args."""
        fake_papers = _make_fake_papers(3)
        saved_papers = _make_saved_papers(fake_papers)
        interests = _make_interests()
        candidates = _make_candidates(fake_papers[:3])
        ranked = _make_ranked(candidates)

        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.Embedder") as MockEmbedder,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("src.pipeline.LLMRanker") as MockRanker,
            patch("src.pipeline.InterestManager") as MockInterestMgr,
            patch("src.pipeline.ReportGenerator") as MockReportGen,
            patch("src.pipeline.EmailSender") as MockEmailSender,
        ):
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_today = AsyncMock(return_value=fake_papers)

            mock_embedder = MockEmbedder.return_value
            mock_embedder.compute_embeddings = MagicMock()
            mock_embedder.find_similar = MagicMock(return_value=candidates)

            mock_create_llm.return_value = MagicMock()

            mock_ranker = MockRanker.return_value
            mock_ranker.rerank = AsyncMock(return_value=ranked)

            mock_interest_mgr = MockInterestMgr.return_value
            mock_interest_mgr.get_interests_with_embeddings = MagicMock(return_value=interests)

            mock_report_gen = MockReportGen.return_value
            mock_report_gen.generate_general = AsyncMock(return_value="# G")
            mock_report_gen.generate_specific = AsyncMock(return_value="## S")

            mock_email_sender = MockEmailSender.return_value
            mock_email_sender.send = AsyncMock()

            pipeline = DailyPipeline(config)
            pipeline.store.save_papers = MagicMock(return_value=saved_papers)
            pipeline.store.get_papers_by_date_with_embeddings = MagicMock(
                return_value=candidates
            )
            pipeline.store.save_match = MagicMock(return_value=1)

            result = await pipeline.run()

            # save_match called once per ranked paper
            assert pipeline.store.save_match.call_count == len(ranked)

            # Verify first call args
            first_call_args = pipeline.store.save_match.call_args_list[0]
            assert first_call_args[0][0] == ranked[0]["id"]  # paper_id
            assert first_call_args[0][1] == date.today().isoformat()  # run_date

    @pytest.mark.asyncio
    async def test_save_report_called_once(self, config, tmp_path):
        """Verify save_report is called once with both reports."""
        fake_papers = _make_fake_papers(2)
        saved_papers = _make_saved_papers(fake_papers)
        interests = _make_interests()
        candidates = _make_candidates(fake_papers)
        ranked = _make_ranked(candidates)

        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.Embedder") as MockEmbedder,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("src.pipeline.LLMRanker") as MockRanker,
            patch("src.pipeline.InterestManager") as MockInterestMgr,
            patch("src.pipeline.ReportGenerator") as MockReportGen,
            patch("src.pipeline.EmailSender") as MockEmailSender,
        ):
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_today = AsyncMock(return_value=fake_papers)

            mock_embedder = MockEmbedder.return_value
            mock_embedder.compute_embeddings = MagicMock()
            mock_embedder.find_similar = MagicMock(return_value=candidates)

            mock_create_llm.return_value = MagicMock()

            mock_ranker = MockRanker.return_value
            mock_ranker.rerank = AsyncMock(return_value=ranked)

            mock_interest_mgr = MockInterestMgr.return_value
            mock_interest_mgr.get_interests_with_embeddings = MagicMock(return_value=interests)

            mock_report_gen = MockReportGen.return_value
            mock_report_gen.generate_general = AsyncMock(return_value="# General Report")
            mock_report_gen.generate_specific = AsyncMock(return_value="## Specific Report")

            mock_email_sender = MockEmailSender.return_value
            mock_email_sender.send = AsyncMock()

            pipeline = DailyPipeline(config)
            pipeline.store.save_papers = MagicMock(return_value=saved_papers)
            pipeline.store.get_papers_by_date_with_embeddings = MagicMock(
                return_value=candidates
            )
            pipeline.store.save_report = MagicMock(return_value=1)

            await pipeline.run()

            # save_report called once with both reports
            pipeline.store.save_report.assert_called_once_with(
                date.today().isoformat(),
                "# General Report",
                "## Specific Report",
                len(saved_papers),
                len(ranked),
            )

    @pytest.mark.asyncio
    async def test_no_interests_still_saves_general_report(self, config, tmp_path):
        """When no interests, a general report is still saved."""
        fake_papers = _make_fake_papers(3)
        saved_papers = _make_saved_papers(fake_papers)

        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.Embedder") as MockEmbedder,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("src.pipeline.LLMRanker") as MockRanker,
            patch("src.pipeline.InterestManager") as MockInterestMgr,
            patch("src.pipeline.ReportGenerator") as MockReportGen,
            patch("src.pipeline.EmailSender") as MockEmailSender,
        ):
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_today = AsyncMock(return_value=fake_papers)

            mock_embedder = MockEmbedder.return_value
            mock_embedder.compute_embeddings = MagicMock()

            mock_create_llm.return_value = MagicMock()
            MockRanker.return_value.rerank = AsyncMock()

            mock_interest_mgr = MockInterestMgr.return_value
            mock_interest_mgr.get_interests_with_embeddings = MagicMock(return_value=[])

            mock_report_gen = MockReportGen.return_value
            mock_report_gen.generate_general = AsyncMock(return_value="# General Only")

            MockEmailSender.return_value

            pipeline = DailyPipeline(config)
            pipeline.store.save_papers = MagicMock(return_value=saved_papers)
            pipeline.store.save_report = MagicMock(return_value=1)

            await pipeline.run()

            # General report generated and saved
            mock_report_gen.generate_general.assert_awaited_once()
            pipeline.store.save_report.assert_called_once_with(
                date.today().isoformat(),
                "# General Only",
                "",  # empty specific report
                len(saved_papers),
                0,  # no matches
            )


class TestDailyPipelineInit:
    """Test that the pipeline initializes all components correctly."""

    def test_init_creates_all_components(self, config, tmp_path):
        """Verify that __init__ creates instances of all required components."""
        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.Embedder") as MockEmbedder,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("src.pipeline.LLMRanker") as MockRanker,
            patch("src.pipeline.InterestManager") as MockInterestMgr,
            patch("src.pipeline.ReportGenerator") as MockReportGen,
            patch("src.pipeline.EmailSender") as MockEmailSender,
        ):
            mock_create_llm.return_value = MagicMock()
            pipeline = DailyPipeline(config)

            assert pipeline.store is not None
            MockFetcher.assert_called_once_with(config)
            MockEmbedder.assert_called_once_with(config)
            mock_create_llm.assert_called_once_with(config)
            MockRanker.assert_called_once()
            MockInterestMgr.assert_called_once()
            MockReportGen.assert_called_once()
            MockEmailSender.assert_called_once_with(config)
