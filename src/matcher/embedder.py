import logging

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, config: dict):
        self.model_name = config["matching"]["embedding_model"]
        self._model = None
        self.logger = logging.getLogger(__name__)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string. Return a 1D numpy array (384 dims for MiniLM)."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Return a 2D numpy array of shape (N, dims)."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize a numpy array to bytes for SQLite BLOB storage."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes, dim: int = 384) -> np.ndarray:
        """Deserialize bytes back to a numpy array."""
        if len(blob) == dim * 4:
            return np.frombuffer(blob, dtype=np.float32).reshape(-1)
        return np.frombuffer(blob, dtype=np.float32)

    def compute_embeddings(self, papers: list[dict], store):
        """Compute embeddings for papers that don't have them yet.
        For each paper: embed the abstract, serialize, call store.update_paper_embedding(id, blob).
        """
        if not papers:
            return
        texts = [p["abstract"] for p in papers]
        embeddings = self.embed_texts(texts)
        for paper, embedding in zip(papers, embeddings):
            blob = self.serialize_embedding(embedding)
            store.update_paper_embedding(paper["id"], blob)
        self.logger.info(f"Computed embeddings for {len(papers)} papers")

    def compute_interest_embeddings(self, interests: list[dict], store):
        """Compute embeddings for interests that don't have them yet.
        For keywords: embed the keyword text (+ description if available).
        For paper/reference_paper types: embed the value (which is a paper title or arXiv ID description).
        Serialize and call store.update_interest_embedding(id, blob).
        """
        if not interests:
            return
        for interest in interests:
            if interest.get("description"):
                text = f"{interest['value']}: {interest['description']}"
            else:
                text = interest["value"]
            embedding = self.embed_text(text)
            blob = self.serialize_embedding(embedding)
            store.update_interest_embedding(interest["id"], blob)
        self.logger.info(f"Computed embeddings for {len(interests)} interests")

    def find_similar(
        self,
        interests: list[dict],
        papers: list[dict],
        top_n: int,
        threshold: float = 0.3,
    ) -> list[dict]:
        """Compute cosine similarity between each paper and all interest embeddings.
        For each paper, use the MAX similarity across all interests as its score.
        Return top_n papers (above threshold) sorted by score descending.

        Each returned dict includes all original paper fields plus 'embedding_score'.

        interests: list of dicts with 'embedding' key (bytes blob).
        papers: list of dicts with 'embedding' key (bytes blob).
        """
        if not interests or not papers:
            return []

        # 1. Deserialize all interest embeddings into a 2D array (M, dims).
        interest_embeddings = np.array(
            [self.deserialize_embedding(i["embedding"]) for i in interests]
        )

        # 2. Deserialize all paper embeddings into a 2D array (N, dims).
        paper_embeddings = np.array(
            [self.deserialize_embedding(p["embedding"]) for p in papers]
        )

        # 3. Compute similarity matrix: papers_matrix @ interests_matrix.T → (N, M)
        similarity_matrix = paper_embeddings @ interest_embeddings.T

        # 4. For each paper (row), take max across all interests → (N,) scores.
        max_scores = similarity_matrix.max(axis=1)

        # 5. Filter by threshold, sort descending, return top_n.
        results = []
        for idx, score in enumerate(max_scores):
            if score >= threshold:
                paper_with_score = {**papers[idx], "embedding_score": float(score)}
                results.append(paper_with_score)

        results.sort(key=lambda x: x["embedding_score"], reverse=True)
        return results[:top_n]
