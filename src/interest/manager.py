import logging

import arxiv


class InterestManager:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder
        self.logger = logging.getLogger(__name__)

    def add_keyword(self, keyword: str, description: str = None) -> int:
        """Add a keyword interest. Compute and store its embedding. Return the new id."""
        interest_id = self.store.save_interest("keyword", keyword, description)
        text = f"{keyword}: {description}" if description else keyword
        embedding = self.embedder.embed_text(text)
        self.store.update_interest_embedding(
            interest_id, self.embedder.serialize_embedding(embedding)
        )
        return interest_id

    def add_paper(self, arxiv_id: str, description: str = None) -> int:
        """Add a past-paper interest.
        If no description is provided, auto-fetch the paper's abstract:
          1. Check DB first via store.get_paper_by_arxiv_id(arxiv_id)
          2. If not in DB, fetch from arXiv API via _fetch_abstract_from_arxiv(arxiv_id)
          3. If all lookups fail, fall back to using arxiv_id as text (log a warning)
        Embed the resolved text and store the embedding."""
        if not description:
            paper = self.store.get_paper_by_arxiv_id(arxiv_id)
            if paper:
                description = paper["abstract"]
                self.logger.info("Found abstract for %s in DB", arxiv_id)
            else:
                description = self._fetch_abstract_from_arxiv(arxiv_id)
                if description:
                    self.logger.info("Fetched abstract for %s from arXiv", arxiv_id)
                else:
                    self.logger.warning(
                        "Could not fetch abstract for %s, using ID as fallback", arxiv_id
                    )

        interest_id = self.store.save_interest("paper", arxiv_id, description)
        text = description if description else arxiv_id
        embedding = self.embedder.embed_text(text)
        self.store.update_interest_embedding(
            interest_id, self.embedder.serialize_embedding(embedding)
        )
        return interest_id

    def add_reference_paper(self, arxiv_id: str, description: str = None) -> int:
        """Add a reference paper interest. Same auto-fetch logic as add_paper
        but with type='reference_paper'."""
        if not description:
            paper = self.store.get_paper_by_arxiv_id(arxiv_id)
            if paper:
                description = paper["abstract"]
                self.logger.info("Found abstract for %s in DB", arxiv_id)
            else:
                description = self._fetch_abstract_from_arxiv(arxiv_id)
                if description:
                    self.logger.info("Fetched abstract for %s from arXiv", arxiv_id)
                else:
                    self.logger.warning(
                        "Could not fetch abstract for %s, using ID as fallback", arxiv_id
                    )

        interest_id = self.store.save_interest("reference_paper", arxiv_id, description)
        text = description if description else arxiv_id
        embedding = self.embedder.embed_text(text)
        self.store.update_interest_embedding(
            interest_id, self.embedder.serialize_embedding(embedding)
        )
        return interest_id

    def _fetch_abstract_from_arxiv(self, arxiv_id: str) -> str | None:
        """Fetch a single paper's abstract from arXiv by ID.
        Uses arxiv.Search(id_list=[arxiv_id]) to fetch the paper metadata.
        Returns the abstract text, or None if the fetch fails or paper not found."""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(arxiv.Client().results(search))
            if results:
                return results[0].summary.replace("\n", " ").strip()
        except Exception as e:
            self.logger.warning("Failed to fetch abstract from arXiv for %s: %s", arxiv_id, e)
        return None

    def remove_interest(self, interest_id: int):
        """Delete an interest by ID."""
        self.store.delete_interest(interest_id)

    def update_interest(self, interest_id: int, value: str = None, description: str = None):
        """Update an interest and recompute its embedding."""
        self.store.update_interest(interest_id, value=value, description=description)
        updated = self.store.get_interest_by_id(interest_id)
        text = (
            f"{updated['value']}: {updated['description']}"
            if updated.get("description")
            else updated["value"]
        )
        embedding = self.embedder.embed_text(text)
        self.store.update_interest_embedding(
            interest_id, self.embedder.serialize_embedding(embedding)
        )

    def get_all_interests(self) -> list[dict]:
        """Return all interests from the store."""
        return self.store.get_all_interests()

    def get_interests_with_embeddings(self) -> list[dict]:
        """Return interests that have computed embeddings."""
        return self.store.get_interests_with_embeddings()

    def recompute_all_embeddings(self):
        """Recompute embeddings for all interests. Useful after model change."""
        for interest in self.store.get_all_interests():
            text = (
                f"{interest['value']}: {interest['description']}"
                if interest.get("description")
                else interest["value"]
            )
            embedding = self.embedder.embed_text(text)
            self.store.update_interest_embedding(
                interest["id"], self.embedder.serialize_embedding(embedding)
            )
