"""Tests for src/matcher/embedder.py — Phase 4 (Steps 4.1 & 4.2)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.matcher.embedder import Embedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    """Create an Embedder with the default model. Module-scoped to avoid
    reloading the ~80MB model for every test."""
    config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
    return Embedder(config)


# ---------------------------------------------------------------------------
# Step 4.1 — Embedder class
# ---------------------------------------------------------------------------

class TestEmbedText:
    def test_embed_text_returns_1d_array(self, embedder):
        result = embedder.embed_text("machine learning")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (384,)

    def test_embed_text_is_normalized(self, embedder):
        result = embedder.embed_text("test")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5


class TestEmbedTexts:
    def test_embed_texts_returns_2d_array(self, embedder):
        result = embedder.embed_texts(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (2, 384)

    def test_embed_texts_all_normalized(self, embedder):
        result = embedder.embed_texts(["hello", "world", "test"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestSerializeDeserialize:
    def test_roundtrip(self, embedder):
        original = embedder.embed_text("roundtrip test")
        blob = Embedder.serialize_embedding(original)
        restored = Embedder.deserialize_embedding(blob)
        assert np.allclose(original, restored)

    def test_serialize_returns_bytes(self, embedder):
        embedding = embedder.embed_text("bytes test")
        blob = Embedder.serialize_embedding(embedding)
        assert isinstance(blob, bytes)
        assert len(blob) == 384 * 4  # float32 = 4 bytes each

    def test_deserialize_with_explicit_dim(self):
        arr = np.random.randn(384).astype(np.float32)
        blob = arr.tobytes()
        restored = Embedder.deserialize_embedding(blob, dim=384)
        assert restored.shape == (384,)
        assert np.allclose(arr, restored)

    def test_deserialize_with_different_dim(self):
        arr = np.random.randn(768).astype(np.float32)
        blob = arr.tobytes()
        restored = Embedder.deserialize_embedding(blob, dim=768)
        assert restored.shape == (768,)
        assert np.allclose(arr, restored)


class TestComputeEmbeddings:
    def test_compute_embeddings_calls_store(self, embedder):
        mock_store = MagicMock()
        papers = [
            {"id": 1, "abstract": "Deep learning for NLP"},
            {"id": 2, "abstract": "Reinforcement learning in robotics"},
        ]
        embedder.compute_embeddings(papers, mock_store)
        assert mock_store.update_paper_embedding.call_count == 2
        # Verify each call used the correct paper id
        call_args_list = mock_store.update_paper_embedding.call_args_list
        assert call_args_list[0][0][0] == 1
        assert call_args_list[1][0][0] == 2
        # Verify the blob is bytes of correct length
        blob = call_args_list[0][0][1]
        assert isinstance(blob, bytes)
        assert len(blob) == 384 * 4

    def test_compute_embeddings_empty_list(self, embedder):
        mock_store = MagicMock()
        embedder.compute_embeddings([], mock_store)
        mock_store.update_paper_embedding.assert_not_called()


class TestComputeInterestEmbeddings:
    def test_compute_interest_embeddings_keyword(self, embedder):
        mock_store = MagicMock()
        interests = [
            {"id": 10, "type": "keyword", "value": "transformers", "description": "attention models"},
        ]
        embedder.compute_interest_embeddings(interests, mock_store)
        assert mock_store.update_interest_embedding.call_count == 1
        call_args = mock_store.update_interest_embedding.call_args[0]
        assert call_args[0] == 10
        assert isinstance(call_args[1], bytes)

    def test_compute_interest_embeddings_no_description(self, embedder):
        mock_store = MagicMock()
        interests = [
            {"id": 20, "type": "keyword", "value": "graph neural networks", "description": None},
        ]
        embedder.compute_interest_embeddings(interests, mock_store)
        assert mock_store.update_interest_embedding.call_count == 1

    def test_compute_interest_embeddings_empty_list(self, embedder):
        mock_store = MagicMock()
        embedder.compute_interest_embeddings([], mock_store)
        mock_store.update_interest_embedding.assert_not_called()


class TestLazyLoading:
    def test_model_not_loaded_on_init(self):
        config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
        e = Embedder(config)
        assert e._model is None

    def test_model_loaded_on_first_use(self, embedder):
        # The module-scoped fixture already used the model, so it should be loaded
        assert embedder._model is not None


# ---------------------------------------------------------------------------
# Step 4.2 — Cosine similarity matching (find_similar)
# ---------------------------------------------------------------------------

def _make_normalized_vector(seed: int, dim: int = 384) -> np.ndarray:
    """Create a deterministic normalized random vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


class TestFindSimilar:
    def test_returns_top_n(self):
        config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
        e = Embedder(config)

        # Create 3 interest embeddings and 5 paper embeddings
        interests = []
        for i in range(3):
            vec = _make_normalized_vector(seed=i)
            interests.append({"id": i, "embedding": Embedder.serialize_embedding(vec)})

        papers = []
        for i in range(5):
            vec = _make_normalized_vector(seed=100 + i)
            papers.append({
                "id": i,
                "title": f"Paper {i}",
                "abstract": f"Abstract {i}",
                "embedding": Embedder.serialize_embedding(vec),
            })

        result = e.find_similar(interests, papers, top_n=2, threshold=0.0)
        assert len(result) == 2
        for item in result:
            assert "embedding_score" in item

    def test_sorted_descending(self):
        config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
        e = Embedder(config)

        interests = [
            {"id": 0, "embedding": Embedder.serialize_embedding(_make_normalized_vector(0))},
        ]

        papers = []
        for i in range(5):
            papers.append({
                "id": i,
                "title": f"Paper {i}",
                "embedding": Embedder.serialize_embedding(_make_normalized_vector(100 + i)),
            })

        result = e.find_similar(interests, papers, top_n=5, threshold=0.0)
        scores = [r["embedding_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_filtering(self):
        config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
        e = Embedder(config)

        interests = [
            {"id": 0, "embedding": Embedder.serialize_embedding(_make_normalized_vector(0))},
        ]
        papers = [
            {
                "id": 0,
                "title": "Paper 0",
                "embedding": Embedder.serialize_embedding(_make_normalized_vector(100)),
            },
        ]

        # With a very high threshold, nothing should pass
        result = e.find_similar(interests, papers, top_n=10, threshold=0.99)
        assert len(result) == 0

    def test_empty_interests(self):
        config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
        e = Embedder(config)
        papers = [
            {
                "id": 0,
                "title": "Paper 0",
                "embedding": Embedder.serialize_embedding(_make_normalized_vector(0)),
            },
        ]
        result = e.find_similar([], papers, top_n=5, threshold=0.0)
        assert result == []

    def test_empty_papers(self):
        config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
        e = Embedder(config)
        interests = [
            {"id": 0, "embedding": Embedder.serialize_embedding(_make_normalized_vector(0))},
        ]
        result = e.find_similar(interests, [], top_n=5, threshold=0.0)
        assert result == []

    def test_preserves_paper_fields(self):
        config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
        e = Embedder(config)

        interests = [
            {"id": 0, "embedding": Embedder.serialize_embedding(_make_normalized_vector(0))},
        ]
        papers = [
            {
                "id": 42,
                "title": "My Paper",
                "abstract": "An abstract",
                "arxiv_id": "2501.12345",
                "embedding": Embedder.serialize_embedding(_make_normalized_vector(1)),
            },
        ]

        result = e.find_similar(interests, papers, top_n=5, threshold=0.0)
        assert len(result) == 1
        assert result[0]["id"] == 42
        assert result[0]["title"] == "My Paper"
        assert result[0]["arxiv_id"] == "2501.12345"
        assert "embedding_score" in result[0]

    def test_max_similarity_across_interests(self):
        """Verify that the score is the MAX across all interests, not average."""
        config = {"matching": {"embedding_model": "all-MiniLM-L6-v2"}}
        e = Embedder(config)

        # Create a paper vector
        paper_vec = _make_normalized_vector(seed=42)

        # Create one interest that's identical to the paper (similarity = 1.0)
        # and one that's very different
        identical_interest = paper_vec.copy()
        different_interest = _make_normalized_vector(seed=999)

        interests = [
            {"id": 0, "embedding": Embedder.serialize_embedding(identical_interest)},
            {"id": 1, "embedding": Embedder.serialize_embedding(different_interest)},
        ]
        papers = [
            {
                "id": 0,
                "title": "Test",
                "embedding": Embedder.serialize_embedding(paper_vec),
            },
        ]

        result = e.find_similar(interests, papers, top_n=1, threshold=0.0)
        assert len(result) == 1
        # Max should be ~1.0 (identical vector)
        assert result[0]["embedding_score"] > 0.99
