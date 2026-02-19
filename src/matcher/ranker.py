import asyncio
import logging


class LLMRanker:
    def __init__(self, llm, config: dict):
        self.llm = llm
        self.top_k = config["matching"]["llm_top_k"]
        self.logger = logging.getLogger(__name__)

    async def rerank(
        self,
        candidates: list[dict],
        interests: list[dict],
        top_k: int = None,
        max_concurrent: int = 5,
    ) -> list[dict]:
        """Re-rank candidate papers using the LLM with concurrent scoring.

        candidates: list of paper dicts (from embedding matcher, with 'embedding_score').
        interests: list of interest dicts (with 'value' and 'description').
        top_k: override for self.top_k.
        max_concurrent: maximum number of concurrent LLM calls (default 5).

        Returns top_k papers sorted by llm_score descending.
        Each returned dict includes original paper fields + 'llm_score' + 'llm_reason'.
        """
        k = top_k or self.top_k
        interests_text = self._format_interests(interests)

        self.logger.info(
            f"Re-ranking {len(candidates)} candidates against {len(interests)} interests"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_limit(paper):
            async with semaphore:
                score_data = await self._score_paper(paper, interests_text)
                return {**paper, **score_data}

        results = await asyncio.gather(*[score_with_limit(paper) for paper in candidates])

        results = sorted(results, key=lambda x: x.get("llm_score", 0), reverse=True)
        self.logger.info(f"Re-ranking complete, returning top {k} of {len(results)} scored papers")
        return results[:k]

    async def _score_paper(self, paper: dict, interests_text: str) -> dict:
        """Ask the LLM to score a single paper.
        Returns {"llm_score": float, "llm_reason": str}.
        """
        prompt = (
            f"Rate the relevance of this paper to the user's research interests.\n\n"
            f"Paper Title: {paper.get('title', 'Unknown')}\n"
            f"Abstract: {paper.get('abstract', 'No abstract available')}\n\n"
            f"User's Research Interests:\n{interests_text}\n\n"
            f"Return a JSON object with:\n"
            f'- "score": a float from 1 to 10 (10 = extremely relevant)\n'
            f'- "reason": a 1-2 sentence explanation of the relevance'
        )
        system = "You are a research paper relevance scorer. Evaluate papers against user interests."

        try:
            result = await self.llm.complete_json(prompt, system=system)
            return {
                "llm_score": float(result.get("score", 0)),
                "llm_reason": str(result.get("reason", "")),
            }
        except Exception as e:
            self.logger.warning(f"Scoring failed for paper '{paper.get('title', '?')}': {e}")
            return {"llm_score": 0, "llm_reason": "Scoring failed"}

    def _format_interests(self, interests: list[dict]) -> str:
        """Format interests into a readable text block for the LLM prompt."""
        lines = []
        for interest in interests:
            entry = f"- {interest.get('type', 'keyword')}: {interest.get('value', '')}"
            if interest.get("description"):
                entry += f" ({interest['description']})"
            lines.append(entry)
        return "\n".join(lines) if lines else "No interests specified."
