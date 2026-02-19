"""Tests for LLM Re-ranker (src/matcher/ranker.py)."""

import asyncio

import pytest

from src.llm.base import LLMProvider
from src.matcher.ranker import LLMRanker


# --- Helpers ---


def _make_candidate(paper_id: int, title: str, abstract: str = "Some abstract") -> dict:
    """Create a fake candidate paper dict with an embedding_score."""
    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "arxiv_id": f"2501.{paper_id:05d}",
        "authors": ["Author A"],
        "categories": ["cs.AI"],
        "pdf_url": f"https://arxiv.org/pdf/2501.{paper_id:05d}",
        "embedding_score": 0.5 + paper_id * 0.01,
    }


def _make_interest(type_: str, value: str, description: str = None) -> dict:
    return {"type": type_, "value": value, "description": description}


RANKER_CONFIG = {"matching": {"llm_top_k": 2}}


# --- Mock LLM Providers ---


class MockLLMProvider(LLMProvider):
    """Returns a fixed score for every paper."""

    def __init__(self, score: float = 8.5, reason: str = "Highly relevant"):
        self.score = score
        self.reason = reason
        self.call_count = 0

    async def complete(self, prompt: str, system: str = "") -> str:
        return f'{{"score": {self.score}, "reason": "{self.reason}"}}'

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        self.call_count += 1
        return {"score": self.score, "reason": self.reason}


class MockLLMProviderInvalidJSON(LLMProvider):
    """Simulates a provider that raises ValueError (invalid JSON)."""

    async def complete(self, prompt: str, system: str = "") -> str:
        return "not json"

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        raise ValueError("Invalid JSON response")


class MockLLMProviderConcurrency(LLMProvider):
    """Tracks the maximum number of concurrent calls."""

    def __init__(self):
        self._current = 0
        self.max_concurrent = 0
        self._lock = asyncio.Lock()

    async def complete(self, prompt: str, system: str = "") -> str:
        return '{"score": 5, "reason": "ok"}'

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        async with self._lock:
            self._current += 1
            if self._current > self.max_concurrent:
                self.max_concurrent = self._current
        # Simulate LLM latency so concurrent calls overlap
        await asyncio.sleep(0.05)
        async with self._lock:
            self._current -= 1
        return {"score": 5, "reason": "ok"}


# --- Tests ---


class TestLLMRankerBasic:
    """Basic re-ranking behavior."""

    @pytest.mark.asyncio
    async def test_rerank_returns_top_k(self):
        llm = MockLLMProvider(score=8.5, reason="Highly relevant")
        ranker = LLMRanker(llm, RANKER_CONFIG)
        candidates = [_make_candidate(i, f"Paper {i}") for i in range(5)]
        interests = [_make_interest("keyword", "transformers")]

        results = await ranker.rerank(candidates, interests)

        assert len(results) == 2  # top_k = 2

    @pytest.mark.asyncio
    async def test_rerank_results_have_llm_fields(self):
        llm = MockLLMProvider(score=8.5, reason="Highly relevant")
        ranker = LLMRanker(llm, RANKER_CONFIG)
        candidates = [_make_candidate(i, f"Paper {i}") for i in range(5)]
        interests = [_make_interest("keyword", "transformers")]

        results = await ranker.rerank(candidates, interests)

        for result in results:
            assert result["llm_score"] == 8.5
            assert result["llm_reason"] == "Highly relevant"

    @pytest.mark.asyncio
    async def test_rerank_preserves_original_fields(self):
        llm = MockLLMProvider()
        ranker = LLMRanker(llm, RANKER_CONFIG)
        candidates = [_make_candidate(1, "My Paper", "My abstract")]
        interests = [_make_interest("keyword", "AI")]

        results = await ranker.rerank(candidates, interests)

        assert results[0]["title"] == "My Paper"
        assert results[0]["abstract"] == "My abstract"
        assert results[0]["arxiv_id"] == "2501.00001"
        assert "embedding_score" in results[0]

    @pytest.mark.asyncio
    async def test_rerank_calls_llm_once_per_candidate(self):
        llm = MockLLMProvider()
        ranker = LLMRanker(llm, RANKER_CONFIG)
        candidates = [_make_candidate(i, f"Paper {i}") for i in range(5)]
        interests = [_make_interest("keyword", "AI"), _make_interest("keyword", "NLP")]

        await ranker.rerank(candidates, interests)

        assert llm.call_count == 5

    @pytest.mark.asyncio
    async def test_rerank_top_k_override(self):
        llm = MockLLMProvider()
        ranker = LLMRanker(llm, RANKER_CONFIG)
        candidates = [_make_candidate(i, f"Paper {i}") for i in range(10)]
        interests = [_make_interest("keyword", "AI")]

        results = await ranker.rerank(candidates, interests, top_k=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_rerank_sorted_by_llm_score_descending(self):
        """When different papers get different scores, results are sorted descending."""
        scores = [3.0, 9.0, 6.0, 1.0, 7.0]

        class VaryingScoreLLM(LLMProvider):
            def __init__(self):
                self._idx = 0

            async def complete(self, prompt: str, system: str = "") -> str:
                return ""

            async def complete_json(self, prompt: str, system: str = "") -> dict:
                # Return scores in the order papers are submitted
                score = scores[self._idx]
                self._idx += 1
                return {"score": score, "reason": f"Score {score}"}

        llm = VaryingScoreLLM()
        config = {"matching": {"llm_top_k": 5}}
        ranker = LLMRanker(llm, config)
        candidates = [_make_candidate(i, f"Paper {i}") for i in range(5)]
        interests = [_make_interest("keyword", "AI")]

        results = await ranker.rerank(candidates, interests, max_concurrent=1)

        result_scores = [r["llm_score"] for r in results]
        assert result_scores == sorted(result_scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self):
        llm = MockLLMProvider()
        ranker = LLMRanker(llm, RANKER_CONFIG)

        results = await ranker.rerank([], [_make_interest("keyword", "AI")])

        assert results == []
        assert llm.call_count == 0


class TestLLMRankerFailure:
    """LLM failure handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_score_zero(self):
        llm = MockLLMProviderInvalidJSON()
        ranker = LLMRanker(llm, RANKER_CONFIG)
        candidates = [_make_candidate(1, "Paper 1")]
        interests = [_make_interest("keyword", "AI")]

        results = await ranker.rerank(candidates, interests)

        assert len(results) == 1
        assert results[0]["llm_score"] == 0
        assert "failed" in results[0]["llm_reason"].lower()

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_crash(self):
        """If scoring fails for some papers, others still get scored."""
        call_count = 0

        class PartialFailureLLM(LLMProvider):
            async def complete(self, prompt: str, system: str = "") -> str:
                return ""

            async def complete_json(self, prompt: str, system: str = "") -> dict:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("LLM API timeout")
                return {"score": 7.0, "reason": "Good match"}

        llm = PartialFailureLLM()
        config = {"matching": {"llm_top_k": 5}}
        ranker = LLMRanker(llm, config)
        candidates = [_make_candidate(i, f"Paper {i}") for i in range(3)]
        interests = [_make_interest("keyword", "AI")]

        results = await ranker.rerank(candidates, interests, max_concurrent=1)

        assert len(results) == 3
        scores = [r["llm_score"] for r in results]
        assert 0 in scores  # the failed one
        assert 7.0 in scores  # the successful ones


class TestLLMRankerConcurrency:
    """Verify the semaphore limits concurrent LLM calls."""

    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self):
        llm = MockLLMProviderConcurrency()
        config = {"matching": {"llm_top_k": 10}}
        ranker = LLMRanker(llm, config)
        candidates = [_make_candidate(i, f"Paper {i}") for i in range(10)]
        interests = [_make_interest("keyword", "AI")]

        await ranker.rerank(candidates, interests, max_concurrent=3)

        assert llm.max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_default_concurrency_allows_parallel(self):
        """With default max_concurrent=5, multiple calls should run in parallel."""
        llm = MockLLMProviderConcurrency()
        config = {"matching": {"llm_top_k": 10}}
        ranker = LLMRanker(llm, config)
        candidates = [_make_candidate(i, f"Paper {i}") for i in range(10)]
        interests = [_make_interest("keyword", "AI")]

        await ranker.rerank(candidates, interests)  # default max_concurrent=5

        # With 10 candidates and sleep(0.05), parallelism should kick in
        assert llm.max_concurrent > 1
        assert llm.max_concurrent <= 5


class TestFormatInterests:
    """Test the _format_interests helper."""

    def test_format_keywords(self):
        ranker = LLMRanker(MockLLMProvider(), RANKER_CONFIG)
        interests = [
            _make_interest("keyword", "transformers", "attention mechanisms"),
            _make_interest("keyword", "reinforcement learning"),
        ]

        result = ranker._format_interests(interests)

        assert "- keyword: transformers (attention mechanisms)" in result
        assert "- keyword: reinforcement learning" in result

    def test_format_mixed_types(self):
        ranker = LLMRanker(MockLLMProvider(), RANKER_CONFIG)
        interests = [
            _make_interest("keyword", "NLP"),
            _make_interest("paper", "2501.12345", "My paper about LLMs"),
        ]

        result = ranker._format_interests(interests)

        assert "- keyword: NLP" in result
        assert "- paper: 2501.12345 (My paper about LLMs)" in result

    def test_format_empty_interests(self):
        ranker = LLMRanker(MockLLMProvider(), RANKER_CONFIG)

        result = ranker._format_interests([])

        assert result == "No interests specified."
