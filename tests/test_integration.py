"""End-to-end integration test with mocked external services (arXiv API, LLM APIs, SMTP).

This test exercises the full pipeline using:
- Real PaperStore (temp SQLite DB)
- Real Embedder (real sentence-transformers model for real embeddings)
- Real InterestManager (real DB + real embeddings)
- Real ReportGenerator (with mocked LLM)
- Real EmailSender (with mocked SMTP)
- Real LLMRanker (with mocked LLM)
- Mocked ArxivFetcher.fetch_today (returns synthetic papers)
- Mocked LLMProvider (deterministic responses)
- Mocked smtplib.SMTP (prevents actual email sending)
"""

import os
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.llm.base import LLMProvider
from src.matcher.embedder import Embedder
from src.store.database import PaperStore
from src.interest.manager import InterestManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


# Module-scoped embedder to avoid reloading the ~80MB model per test
@pytest.fixture(scope="module")
def embedder():
    config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
    return Embedder(config)


class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM that returns canned responses."""

    def __init__(self):
        self.complete_calls = 0
        self.complete_json_calls = 0

    async def complete(self, prompt: str, system: str = "") -> str:
        self.complete_calls += 1
        if "trending" in prompt.lower() or "emerging" in prompt.lower():
            return (
                "1. **Neural Architecture Search**: Growing interest in automated "
                "architecture design\n"
                "2. **Multimodal Learning**: Combining vision and language models\n"
                "3. **Efficient Training**: Reducing computational costs of large models"
            )
        if "noteworthy" in prompt.lower() or "impactful" in prompt.lower():
            return (
                "1. **Transformer Attention Mechanisms in NLP** - Author A - "
                "Novel attention mechanism\n"
                "2. **Deep Reinforcement Learning for Robotics** - Author B - "
                "Breakthrough in sim-to-real transfer"
            )
        return "This is a test LLM response."

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        self.complete_json_calls += 1
        return {"score": 7, "reason": "Relevant to user interests in machine learning"}


def _make_test_config(db_path: str) -> dict:
    """Create a complete test configuration dict."""
    return {
        "database": {"path": db_path},
        "arxiv": {"categories": ["cs.AI", "cs.LG"], "max_results_per_category": 100},
        "matching": {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_top_n": 50,
            "llm_top_k": 5,
            "similarity_threshold": 0.1,  # Low threshold to ensure matches in test
        },
        "llm": {
            "provider": "openai",
            "openai": {"model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
        },
        "email": {
            "enabled": True,
            "smtp": {
                "host": "smtp.test.com",
                "port": 587,
                "username_env": "EMAIL_USERNAME",
                "password_env": "EMAIL_PASSWORD",
            },
            "from": "test@example.com",
            "to": ["recipient@example.com"],
            "subject_prefix": "[Test Papers]",
        },
        "scheduler": {"cron": "0 8 * * *"},
    }


def _make_synthetic_papers(count: int = 10) -> list[dict]:
    """Create synthetic papers with ML-related titles and abstracts.

    Papers are designed so that some are highly relevant to interests
    in "transformer architectures" and "reinforcement learning".
    """
    today = date.today().isoformat()
    papers = [
        {
            "arxiv_id": "2501.10001",
            "title": "Attention Is All You Need: Revisited",
            "authors": ["Alice Smith", "Bob Jones"],
            "abstract": (
                "We revisit the transformer architecture and propose new attention "
                "mechanisms that improve performance on natural language processing tasks. "
                "Our method achieves state-of-the-art results on machine translation."
            ),
            "categories": ["cs.AI", "cs.CL"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10001",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10001",
        },
        {
            "arxiv_id": "2501.10002",
            "title": "Deep Reinforcement Learning for Autonomous Navigation",
            "authors": ["Charlie Brown", "Diana Prince"],
            "abstract": (
                "This paper presents a deep reinforcement learning framework for "
                "autonomous vehicle navigation. We use proximal policy optimization "
                "with a transformer-based policy network for improved decision making."
            ),
            "categories": ["cs.AI", "cs.RO"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10002",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10002",
        },
        {
            "arxiv_id": "2501.10003",
            "title": "Efficient Vision Transformers for Image Classification",
            "authors": ["Eve White", "Frank Black"],
            "abstract": (
                "We propose an efficient vision transformer architecture that reduces "
                "computational cost by 60% while maintaining accuracy on ImageNet. "
                "Our approach uses a novel token pruning strategy."
            ),
            "categories": ["cs.CV", "cs.LG"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10003",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10003",
        },
        {
            "arxiv_id": "2501.10004",
            "title": "Multi-Agent Reinforcement Learning in Complex Environments",
            "authors": ["Grace Hopper", "Alan Turing"],
            "abstract": (
                "We study multi-agent reinforcement learning in complex cooperative "
                "and competitive environments. Our method uses communication protocols "
                "between agents to achieve improved coordination."
            ),
            "categories": ["cs.AI", "cs.MA"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10004",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10004",
        },
        {
            "arxiv_id": "2501.10005",
            "title": "Large Language Models for Code Generation",
            "authors": ["Hank Green", "Ivy Blue"],
            "abstract": (
                "We present a large language model fine-tuned for code generation "
                "tasks. The model uses a transformer decoder architecture and achieves "
                "state-of-the-art results on HumanEval and MBPP benchmarks."
            ),
            "categories": ["cs.CL", "cs.SE"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10005",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10005",
        },
        {
            "arxiv_id": "2501.10006",
            "title": "Graph Neural Networks for Molecular Property Prediction",
            "authors": ["Jack Red", "Kate Yellow"],
            "abstract": (
                "This work applies graph neural networks to predict molecular "
                "properties for drug discovery. We introduce a novel message passing "
                "scheme that captures long-range atomic interactions."
            ),
            "categories": ["cs.LG", "q-bio.BM"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10006",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10006",
        },
        {
            "arxiv_id": "2501.10007",
            "title": "Self-Supervised Learning with Contrastive Objectives",
            "authors": ["Leo Brown", "Mary White"],
            "abstract": (
                "We explore self-supervised learning methods using contrastive "
                "objectives for representation learning. Our approach combines "
                "data augmentation strategies with a momentum-based encoder."
            ),
            "categories": ["cs.LG", "cs.CV"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10007",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10007",
        },
        {
            "arxiv_id": "2501.10008",
            "title": "Federated Learning with Differential Privacy Guarantees",
            "authors": ["Nick Silver", "Olivia Gold"],
            "abstract": (
                "We propose a federated learning framework that provides differential "
                "privacy guarantees while maintaining model accuracy. Our method uses "
                "gradient clipping and noise injection techniques."
            ),
            "categories": ["cs.LG", "cs.CR"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10008",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10008",
        },
        {
            "arxiv_id": "2501.10009",
            "title": "Neural Architecture Search via Transformer-Based Controllers",
            "authors": ["Peter Green", "Quinn Blue"],
            "abstract": (
                "We present a neural architecture search method that uses a "
                "transformer-based controller to generate and evaluate candidate "
                "architectures. The approach finds efficient models for edge deployment."
            ),
            "categories": ["cs.AI", "cs.LG"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10009",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10009",
        },
        {
            "arxiv_id": "2501.10010",
            "title": "Reward Shaping in Deep Reinforcement Learning",
            "authors": ["Rachel Red", "Sam Orange"],
            "abstract": (
                "We investigate reward shaping techniques in deep reinforcement "
                "learning to accelerate training convergence. Our method uses "
                "potential-based reward functions derived from domain knowledge."
            ),
            "categories": ["cs.AI", "cs.LG"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.10010",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.10010",
        },
    ]
    return papers[:count]


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """End-to-end integration test with mocked external services."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_real_components(self, tmp_path, embedder):
        """Run the full pipeline with real DB, real embeddings, mocked external services.

        This test:
        1. Creates a temp DB and config.
        2. Adds 2 keyword interests via InterestManager (real embeddings).
        3. Mocks ArxivFetcher.fetch_today to return 10 synthetic papers.
        4. Mocks LLMProvider to return deterministic responses.
        5. Mocks smtplib.SMTP to prevent actual email sending.
        6. Runs DailyPipeline.run().
        7. Asserts DB state and pipeline result.
        """
        db_path = str(tmp_path / "integration_test.db")
        config = _make_test_config(db_path)
        today = date.today().isoformat()
        synthetic_papers = _make_synthetic_papers(10)
        mock_llm = MockLLMProvider()

        # Set up email env vars for EmailSender init
        os.environ.setdefault("EMAIL_USERNAME", "test@example.com")
        os.environ.setdefault("EMAIL_PASSWORD", "testpassword")

        # Step 1: Set up real store and add interests with real embeddings
        store = PaperStore(db_path)
        interest_mgr = InterestManager(store, embedder)
        interest_mgr.add_keyword(
            "transformer architectures",
            "Self-attention mechanisms and transformer models for NLP and vision",
        )
        interest_mgr.add_keyword(
            "reinforcement learning",
            "Deep RL, policy optimization, reward shaping, multi-agent systems",
        )

        # Verify interests were created with embeddings
        interests = store.get_interests_with_embeddings()
        assert len(interests) == 2
        for interest in interests:
            emb = Embedder.deserialize_embedding(interest["embedding"])
            assert emb.shape == (384,)

        # Step 2: Run the pipeline with mocked external services
        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("smtplib.SMTP") as MockSMTP,
        ):
            # Mock fetcher to return synthetic papers
            mock_fetcher_instance = MockFetcher.return_value
            mock_fetcher_instance.fetch_today = AsyncMock(return_value=synthetic_papers)

            # Mock LLM to return deterministic responses
            mock_create_llm.return_value = mock_llm

            # Mock SMTP to prevent real email sending
            mock_smtp_instance = MockSMTP.return_value.__enter__.return_value
            mock_smtp_instance.starttls = MagicMock()
            mock_smtp_instance.login = MagicMock()
            mock_smtp_instance.send_message = MagicMock()

            # Import here to use the patched classes
            from src.pipeline import DailyPipeline

            pipeline = DailyPipeline(config)
            result = await pipeline.run()

        # Step 3: Assertions on pipeline result
        assert result["date"] == today
        assert result["papers_fetched"] == 10
        assert result["new_papers"] == 10
        assert result["matches"] > 0
        assert result["email_sent"] is True

        # Step 4: Verify DB state — papers
        all_papers = store.get_papers_by_date(today)
        assert len(all_papers) == 10

        # All papers should have embeddings
        papers_with_emb = store.get_papers_with_embeddings()
        assert len(papers_with_emb) == 10
        for paper in papers_with_emb:
            emb = Embedder.deserialize_embedding(paper["embedding"])
            assert emb.shape == (384,)
            assert np.isclose(np.linalg.norm(emb), 1.0, atol=0.01)

        # Step 5: Verify matches were saved
        matches = store.get_matches_by_date(today)
        assert len(matches) > 0
        assert len(matches) == result["matches"]
        for match in matches:
            assert match["embedding_score"] > 0
            assert match["llm_score"] == 7  # Our mock always returns 7
            assert "Relevant to user interests" in match["llm_reason"]
            # Joined paper info should be present
            assert match["title"] is not None
            assert match["arxiv_id"] is not None

        # Step 6: Verify report was saved
        report = store.get_report_by_date(today)
        assert report is not None
        assert report["paper_count"] == 10
        assert report["matched_count"] == result["matches"]
        # General report should contain the date header
        assert today in report["general_report"]
        assert "General Report" in report["general_report"]
        # Specific report should contain scored papers
        assert "Specific Report" in report["specific_report"]
        assert "7.0/10" in report["specific_report"]  # Our mock LLM score

        # Step 7: Verify SMTP was called
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once()
        mock_smtp_instance.send_message.assert_called_once()

        # Step 8: Verify LLM was used for scoring and report generation
        assert mock_llm.complete_json_calls > 0  # Used by ranker
        assert mock_llm.complete_calls > 0  # Used by report generator

    @pytest.mark.asyncio
    async def test_pipeline_with_no_interests(self, tmp_path, embedder):
        """Pipeline should generate a general report even with no interests configured."""
        db_path = str(tmp_path / "no_interests_test.db")
        config = _make_test_config(db_path)
        config["email"]["enabled"] = False  # Disable email for simplicity
        today = date.today().isoformat()
        synthetic_papers = _make_synthetic_papers(5)
        mock_llm = MockLLMProvider()

        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
        ):
            mock_fetcher_instance = MockFetcher.return_value
            mock_fetcher_instance.fetch_today = AsyncMock(return_value=synthetic_papers)
            mock_create_llm.return_value = mock_llm

            from src.pipeline import DailyPipeline

            pipeline = DailyPipeline(config)
            result = await pipeline.run()

        assert result["papers_fetched"] == 5
        assert result["new_papers"] == 5
        assert result["matches"] == 0
        assert result["email_sent"] is False

        # General report should still be saved
        store = PaperStore(db_path)
        report = store.get_report_by_date(today)
        assert report is not None
        assert report["general_report"] != ""
        assert report["specific_report"] == ""
        assert report["paper_count"] == 5
        assert report["matched_count"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_duplicate_papers(self, tmp_path, embedder):
        """Running the pipeline twice with the same papers should handle duplicates."""
        db_path = str(tmp_path / "dedup_test.db")
        config = _make_test_config(db_path)
        config["email"]["enabled"] = False
        synthetic_papers = _make_synthetic_papers(5)
        mock_llm = MockLLMProvider()

        # Run pipeline twice
        for run_num in range(2):
            with (
                patch("src.pipeline.ArxivFetcher") as MockFetcher,
                patch("src.pipeline.create_llm_provider") as mock_create_llm,
            ):
                mock_fetcher_instance = MockFetcher.return_value
                mock_fetcher_instance.fetch_today = AsyncMock(return_value=synthetic_papers)
                mock_create_llm.return_value = mock_llm

                from src.pipeline import DailyPipeline

                pipeline = DailyPipeline(config)
                result = await pipeline.run()

            if run_num == 0:
                assert result["new_papers"] == 5
            else:
                # Second run: all papers are duplicates
                assert result["new_papers"] == 0

        # Only 5 papers in DB total (not 10)
        store = PaperStore(db_path)
        all_papers = store.get_papers_by_date(date.today().isoformat())
        assert len(all_papers) == 5

    @pytest.mark.asyncio
    async def test_embedding_similarity_relevance(self, tmp_path, embedder):
        """Verify that embedding similarity actually ranks relevant papers higher.

        Adds an interest in "transformer architectures" and checks that
        transformer-related papers get higher embedding scores than unrelated ones.
        """
        db_path = str(tmp_path / "relevance_test.db")
        store = PaperStore(db_path)
        interest_mgr = InterestManager(store, embedder)

        # Add a specific interest
        interest_mgr.add_keyword(
            "transformer architectures",
            "Self-attention mechanisms and transformer models",
        )

        # Insert papers and compute their embeddings
        papers = _make_synthetic_papers(10)
        saved = store.save_papers(papers)
        embedder.compute_embeddings(saved, store)

        # Get interests and papers with embeddings
        interests = store.get_interests_with_embeddings()
        papers_with_emb = store.get_papers_by_date_with_embeddings(date.today().isoformat())

        # Find similar papers
        results = embedder.find_similar(interests, papers_with_emb, top_n=10, threshold=0.0)

        # Transformer-related papers should score higher than e.g. federated learning
        transformer_scores = {}
        for r in results:
            transformer_scores[r["arxiv_id"]] = r["embedding_score"]

        # Papers about transformers should rank near the top
        # "2501.10001" = "Attention Is All You Need: Revisited"
        # "2501.10003" = "Efficient Vision Transformers"
        # "2501.10009" = "NAS via Transformer-Based Controllers"
        # "2501.10008" = "Federated Learning" (less related)
        transformer_ids = {"2501.10001", "2501.10003", "2501.10009"}
        top_5_ids = {r["arxiv_id"] for r in results[:5]}

        # At least 2 of the 3 transformer papers should be in the top 5
        overlap = transformer_ids & top_5_ids
        assert len(overlap) >= 2, (
            f"Expected at least 2 transformer papers in top 5, "
            f"got {overlap}. Top 5: {[(r['arxiv_id'], r['title'], r['embedding_score']) for r in results[:5]]}"
        )

    @pytest.mark.asyncio
    async def test_email_content_integrity(self, tmp_path, embedder):
        """Verify that the email sent contains both reports with proper HTML formatting."""
        db_path = str(tmp_path / "email_test.db")
        config = _make_test_config(db_path)
        today = date.today().isoformat()
        synthetic_papers = _make_synthetic_papers(5)
        mock_llm = MockLLMProvider()

        os.environ.setdefault("EMAIL_USERNAME", "test@example.com")
        os.environ.setdefault("EMAIL_PASSWORD", "testpassword")

        # Add an interest so we get both reports
        store = PaperStore(db_path)
        interest_mgr = InterestManager(store, embedder)
        interest_mgr.add_keyword("machine learning", "ML and deep learning research")

        sent_messages = []

        with (
            patch("src.pipeline.ArxivFetcher") as MockFetcher,
            patch("src.pipeline.create_llm_provider") as mock_create_llm,
            patch("smtplib.SMTP") as MockSMTP,
        ):
            mock_fetcher_instance = MockFetcher.return_value
            mock_fetcher_instance.fetch_today = AsyncMock(return_value=synthetic_papers)
            mock_create_llm.return_value = mock_llm

            mock_smtp_instance = MockSMTP.return_value.__enter__.return_value
            mock_smtp_instance.starttls = MagicMock()
            mock_smtp_instance.login = MagicMock()
            mock_smtp_instance.send_message = MagicMock(
                side_effect=lambda msg: sent_messages.append(msg)
            )

            from src.pipeline import DailyPipeline

            pipeline = DailyPipeline(config)
            result = await pipeline.run()

        assert result["email_sent"] is True
        assert len(sent_messages) == 1

        msg = sent_messages[0]
        assert f"[Test Papers] {today}" in msg["Subject"]
        assert msg["From"] == "test@example.com"
        assert "recipient@example.com" in msg["To"]

        # Extract HTML payload
        html_payload = msg.get_payload()[0].get_payload(decode=True).decode("utf-8")
        assert "<html" in html_payload.lower()
        assert "style=" in html_payload  # CSS should be inlined by premailer

    @pytest.mark.asyncio
    async def test_interest_embedding_affects_matching(self, tmp_path, embedder):
        """Different interests should produce different match results."""
        today = date.today().isoformat()
        synthetic_papers = _make_synthetic_papers(10)

        # Run 1: Interest in "reinforcement learning"
        db_path_1 = str(tmp_path / "rl_test.db")
        store_1 = PaperStore(db_path_1)
        mgr_1 = InterestManager(store_1, embedder)
        mgr_1.add_keyword("reinforcement learning", "Deep RL and policy optimization")

        saved_1 = store_1.save_papers(synthetic_papers)
        embedder.compute_embeddings(saved_1, store_1)
        interests_1 = store_1.get_interests_with_embeddings()
        papers_1 = store_1.get_papers_by_date_with_embeddings(today)
        results_1 = embedder.find_similar(interests_1, papers_1, top_n=5, threshold=0.0)

        # Run 2: Interest in "graph neural networks"
        db_path_2 = str(tmp_path / "gnn_test.db")
        store_2 = PaperStore(db_path_2)
        mgr_2 = InterestManager(store_2, embedder)
        mgr_2.add_keyword("graph neural networks", "GNN for molecular and graph data")

        saved_2 = store_2.save_papers(synthetic_papers)
        embedder.compute_embeddings(saved_2, store_2)
        interests_2 = store_2.get_interests_with_embeddings()
        papers_2 = store_2.get_papers_by_date_with_embeddings(today)
        results_2 = embedder.find_similar(interests_2, papers_2, top_n=5, threshold=0.0)

        # The top results should differ
        top_ids_1 = [r["arxiv_id"] for r in results_1[:3]]
        top_ids_2 = [r["arxiv_id"] for r in results_2[:3]]
        assert top_ids_1 != top_ids_2, (
            f"Different interests should produce different rankings, but both got: {top_ids_1}"
        )

        # RL interest should rank RL papers higher
        rl_paper_ids = {"2501.10002", "2501.10004", "2501.10010"}
        assert any(pid in rl_paper_ids for pid in top_ids_1), (
            f"RL interest should rank RL papers highly, but top 3: {top_ids_1}"
        )

        # GNN interest should rank the GNN paper higher
        gnn_paper_id = "2501.10006"
        assert gnn_paper_id in top_ids_2, (
            f"GNN interest should rank GNN paper highly, but top 3: {top_ids_2}"
        )


class TestComponentIntegration:
    """Test that individual components integrate correctly with each other."""

    def test_store_embedder_round_trip(self, tmp_path, embedder):
        """Verify papers saved → embedded → retrieved cycle works end-to-end."""
        db_path = str(tmp_path / "round_trip.db")
        store = PaperStore(db_path)

        papers = _make_synthetic_papers(3)
        saved = store.save_papers(papers)
        assert len(saved) == 3

        # Compute and store embeddings
        embedder.compute_embeddings(saved, store)

        # Retrieve with embeddings
        with_emb = store.get_papers_with_embeddings()
        assert len(with_emb) == 3

        for paper in with_emb:
            emb = Embedder.deserialize_embedding(paper["embedding"])
            assert emb.shape == (384,)
            assert np.isclose(np.linalg.norm(emb), 1.0, atol=0.01)

    def test_interest_manager_and_embedder_integration(self, tmp_path, embedder):
        """Verify InterestManager correctly computes and stores embeddings via Embedder."""
        db_path = str(tmp_path / "interest_emb.db")
        store = PaperStore(db_path)
        mgr = InterestManager(store, embedder)

        # Add different types of interests
        mgr.add_keyword("attention mechanisms")
        mgr.add_keyword("graph neural networks", "GNN for molecular prediction")

        interests = store.get_interests_with_embeddings()
        assert len(interests) == 2

        # Embeddings should be different for different topics
        emb1 = Embedder.deserialize_embedding(interests[0]["embedding"])
        emb2 = Embedder.deserialize_embedding(interests[1]["embedding"])
        similarity = float(np.dot(emb1, emb2))
        assert similarity < 0.95, (
            f"Different interest topics should have different embeddings, "
            f"but similarity is {similarity}"
        )

    @pytest.mark.asyncio
    async def test_ranker_with_real_candidates(self, tmp_path, embedder):
        """Verify LLMRanker correctly scores and ranks candidates from the embedder."""
        from src.matcher.ranker import LLMRanker

        db_path = str(tmp_path / "ranker_int.db")
        store = PaperStore(db_path)
        mgr = InterestManager(store, embedder)
        mgr.add_keyword("transformer architectures")

        papers = _make_synthetic_papers(10)
        saved = store.save_papers(papers)
        embedder.compute_embeddings(saved, store)

        interests = store.get_interests_with_embeddings()
        papers_with_emb = store.get_papers_by_date_with_embeddings(date.today().isoformat())
        candidates = embedder.find_similar(interests, papers_with_emb, top_n=5, threshold=0.0)
        assert len(candidates) == 5

        # Use mock LLM for re-ranking
        mock_llm = MockLLMProvider()
        config = {"matching": {"llm_top_k": 3}}
        ranker = LLMRanker(mock_llm, config)

        ranked = await ranker.rerank(candidates, interests)
        assert len(ranked) == 3
        for paper in ranked:
            assert "llm_score" in paper
            assert "llm_reason" in paper
            assert paper["llm_score"] == 7
            assert "embedding_score" in paper  # Preserved from candidates

        # Ranker should have called the LLM once per candidate
        assert mock_llm.complete_json_calls == 5

    @pytest.mark.asyncio
    async def test_report_generator_with_real_data(self, tmp_path, embedder):
        """Verify ReportGenerator produces valid Markdown from real pipeline data."""
        from src.report.generator import ReportGenerator

        mock_llm = MockLLMProvider()
        gen = ReportGenerator(mock_llm)

        papers = _make_synthetic_papers(10)
        today = date.today().isoformat()

        general = await gen.generate_general(papers, today)
        assert f"# Daily Paper Report - {today}" in general
        assert "General Report" in general
        assert "10" in general  # total count
        assert "cs.AI" in general  # category

        # Specific report with pre-scored data
        ranked = [
            {
                **papers[0],
                "id": 1,
                "llm_score": 8.5,
                "llm_reason": "Highly relevant to transformers",
                "embedding_score": 0.85,
            },
            {
                **papers[1],
                "id": 2,
                "llm_score": 7.0,
                "llm_reason": "Related to RL research",
                "embedding_score": 0.72,
            },
        ]
        interests = [
            {"type": "keyword", "value": "transformers", "description": "Attention models"},
        ]

        specific = await gen.generate_specific(ranked, interests, today)
        assert "Specific Report" in specific
        assert "8.5/10" in specific
        assert "Highly relevant to transformers" in specific
        assert papers[0]["title"] in specific
