import pytest
from unittest.mock import patch, MagicMock

from src.store.database import PaperStore
from src.matcher.embedder import Embedder
from src.interest.manager import InterestManager


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped embedder to avoid reloading the ~80MB model per test."""
    config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
    return Embedder(config)


@pytest.fixture
def store(tmp_path):
    """Fresh PaperStore per test."""
    return PaperStore(str(tmp_path / "test.db"))


@pytest.fixture
def manager(store, embedder):
    return InterestManager(store, embedder)


class TestAddKeyword:
    def test_add_keyword_returns_int_id(self, manager):
        interest_id = manager.add_keyword("transformer architectures")
        assert isinstance(interest_id, int)
        assert interest_id > 0

    def test_add_keyword_creates_interest_in_store(self, manager):
        manager.add_keyword("transformer architectures")
        interests = manager.get_all_interests()
        assert len(interests) == 1
        assert interests[0]["type"] == "keyword"
        assert interests[0]["value"] == "transformer architectures"

    def test_add_keyword_computes_embedding(self, manager, embedder):
        manager.add_keyword("transformer architectures")
        interests = manager.get_interests_with_embeddings()
        assert len(interests) == 1
        blob = interests[0]["embedding"]
        assert blob is not None
        arr = embedder.deserialize_embedding(blob)
        assert arr.shape == (384,)

    def test_add_keyword_with_description(self, manager):
        manager.add_keyword("attention", description="self-attention mechanisms in deep learning")
        interests = manager.get_all_interests()
        assert len(interests) == 1
        assert interests[0]["description"] == "self-attention mechanisms in deep learning"


class TestAddPaper:
    def test_add_paper_with_description(self, manager):
        interest_id = manager.add_paper("2501.12345", "My paper about reinforcement learning")
        assert isinstance(interest_id, int)
        interests = manager.get_all_interests()
        assert len(interests) == 1
        assert interests[0]["type"] == "paper"
        assert interests[0]["value"] == "2501.12345"

    def test_add_paper_computes_embedding(self, manager):
        manager.add_paper("2501.12345", "My paper about reinforcement learning")
        interests = manager.get_interests_with_embeddings()
        assert len(interests) == 1
        assert interests[0]["embedding"] is not None

    def test_add_paper_auto_fetch_from_db(self, store, manager):
        """When paper exists in DB, uses its abstract as the description."""
        store.save_papers(
            [
                {
                    "arxiv_id": "2501.99999",
                    "title": "Test Paper",
                    "authors": ["Author One"],
                    "abstract": "This paper explores novel reinforcement learning approaches.",
                    "categories": ["cs.AI"],
                    "published_date": "2025-01-15",
                    "pdf_url": "https://arxiv.org/pdf/2501.99999",
                    "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.99999",
                }
            ]
        )
        interest_id = manager.add_paper("2501.99999")
        interest = store.get_interest_by_id(interest_id)
        assert interest["description"] == "This paper explores novel reinforcement learning approaches."

    def test_add_paper_auto_fetch_from_arxiv(self, manager, store):
        """When paper is NOT in DB, fetches from arXiv API."""
        with patch.object(
            manager, "_fetch_abstract_from_arxiv", return_value="Fetched abstract from arXiv"
        ):
            interest_id = manager.add_paper("9999.99999")
            interest = store.get_interest_by_id(interest_id)
            assert interest["description"] == "Fetched abstract from arXiv"

    def test_add_paper_fallback_to_id(self, manager, store):
        """When all lookups fail, falls back to using arxiv_id as text."""
        with patch.object(manager, "_fetch_abstract_from_arxiv", return_value=None):
            interest_id = manager.add_paper("0000.00000")
            interest = store.get_interest_by_id(interest_id)
            # Description should be None (no abstract found)
            assert interest["description"] is None
            # But embedding should still be computed (using arxiv_id as text)
            interests_with_emb = manager.get_interests_with_embeddings()
            assert len(interests_with_emb) == 1


class TestAddReferencePaper:
    def test_add_reference_paper_with_description(self, manager):
        interest_id = manager.add_reference_paper("2501.11111", "A reference about NLP")
        interests = manager.get_all_interests()
        assert len(interests) == 1
        assert interests[0]["type"] == "reference_paper"
        assert interests[0]["value"] == "2501.11111"

    def test_add_reference_paper_auto_fetch_from_db(self, store, manager):
        """Same auto-fetch logic as add_paper but with type='reference_paper'."""
        store.save_papers(
            [
                {
                    "arxiv_id": "2501.88888",
                    "title": "Reference Paper",
                    "authors": ["Author Two"],
                    "abstract": "Abstract of the reference paper.",
                    "categories": ["cs.CL"],
                    "published_date": "2025-01-15",
                    "pdf_url": "https://arxiv.org/pdf/2501.88888",
                    "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.88888",
                }
            ]
        )
        interest_id = manager.add_reference_paper("2501.88888")
        interest = store.get_interest_by_id(interest_id)
        assert interest["type"] == "reference_paper"
        assert interest["description"] == "Abstract of the reference paper."


class TestUpdateAndRemove:
    def test_update_interest_changes_value_and_embedding(self, manager, embedder):
        interest_id = manager.add_keyword("transformer architectures")
        original_blob = manager.get_interests_with_embeddings()[0]["embedding"]

        manager.update_interest(interest_id, value="attention mechanisms")

        updated = manager.get_all_interests()[0]
        assert updated["value"] == "attention mechanisms"

        new_blob = manager.get_interests_with_embeddings()[0]["embedding"]
        # Embedding should have changed
        assert original_blob != new_blob

    def test_remove_interest(self, manager):
        interest_id = manager.add_keyword("test keyword")
        assert len(manager.get_all_interests()) == 1
        manager.remove_interest(interest_id)
        assert len(manager.get_all_interests()) == 0


class TestRecomputeAll:
    def test_recompute_all_embeddings(self, manager, embedder):
        manager.add_keyword("keyword one")
        manager.add_keyword("keyword two", description="second keyword")

        # Both should have embeddings
        interests = manager.get_interests_with_embeddings()
        assert len(interests) == 2

        # Recompute all
        manager.recompute_all_embeddings()

        # Still both have embeddings
        interests = manager.get_interests_with_embeddings()
        assert len(interests) == 2


class TestFetchAbstractFromArxiv:
    def test_fetch_abstract_success(self, manager):
        """Mock the arxiv library to return a result."""
        mock_result = MagicMock()
        mock_result.summary = "This is a test abstract\nwith newlines."

        with patch("src.interest.manager.arxiv.Search") as mock_search, patch(
            "src.interest.manager.arxiv.Client"
        ) as mock_client:
            mock_client.return_value.results.return_value = [mock_result]
            result = manager._fetch_abstract_from_arxiv("2501.12345")
            assert result == "This is a test abstract with newlines."

    def test_fetch_abstract_no_results(self, manager):
        """When arXiv returns no results."""
        with patch("src.interest.manager.arxiv.Search") as mock_search, patch(
            "src.interest.manager.arxiv.Client"
        ) as mock_client:
            mock_client.return_value.results.return_value = []
            result = manager._fetch_abstract_from_arxiv("0000.00000")
            assert result is None

    def test_fetch_abstract_exception(self, manager):
        """When arXiv fetch raises an exception."""
        with patch("src.interest.manager.arxiv.Search") as mock_search, patch(
            "src.interest.manager.arxiv.Client"
        ) as mock_client:
            mock_client.return_value.results.side_effect = ConnectionError("Network error")
            result = manager._fetch_abstract_from_arxiv("0000.00000")
            assert result is None
