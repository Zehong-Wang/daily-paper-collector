"""Tests for LLM provider ABC and factory function."""

import os

import pytest

from src.llm.base import LLMProvider


class TestLLMProviderABC:
    """Verify LLMProvider cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            LLMProvider()

    def test_mock_subclass_complete(self):
        """A concrete subclass implementing both methods should work."""

        class MockProvider(LLMProvider):
            async def complete(self, prompt: str, system: str = "") -> str:
                return f"response to: {prompt}"

            async def complete_json(self, prompt: str, system: str = "") -> dict:
                return {"result": prompt}

        provider = MockProvider()
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_mock_subclass_returns_correct_values(self):
        class MockProvider(LLMProvider):
            async def complete(self, prompt: str, system: str = "") -> str:
                return "hello"

            async def complete_json(self, prompt: str, system: str = "") -> dict:
                return {"key": "value"}

        provider = MockProvider()
        assert await provider.complete("test") == "hello"
        assert await provider.complete_json("test") == {"key": "value"}


class TestCreateLLMProviderFactory:
    """Test the factory function in src/llm/__init__.py."""

    def test_factory_openai(self):
        """Factory with provider='openai' returns OpenAIProvider."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        try:
            from src.llm import create_llm_provider
            from src.llm.openai_provider import OpenAIProvider

            config = {
                "llm": {
                    "provider": "openai",
                    "openai": {"model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
                }
            }
            provider = create_llm_provider(config)
            assert isinstance(provider, OpenAIProvider)
        finally:
            del os.environ["OPENAI_API_KEY"]

    def test_factory_claude(self):
        """Factory with provider='claude' returns ClaudeProvider."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        try:
            from src.llm import create_llm_provider
            from src.llm.claude_provider import ClaudeProvider

            config = {
                "llm": {
                    "provider": "claude",
                    "claude": {
                        "model": "claude-sonnet-4-5-20250929",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                }
            }
            provider = create_llm_provider(config)
            assert isinstance(provider, ClaudeProvider)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_factory_claude_code(self):
        """Factory with provider='claude_code' returns ClaudeCodeProvider."""
        from src.llm import create_llm_provider
        from src.llm.claude_code_provider import ClaudeCodeProvider

        config = {
            "llm": {
                "provider": "claude_code",
                "claude_code": {"cli_path": "claude", "model": "sonnet"},
            }
        }
        provider = create_llm_provider(config)
        assert isinstance(provider, ClaudeCodeProvider)

    def test_factory_unknown_provider_raises(self):
        """Factory raises ValueError for unknown provider."""
        from src.llm import create_llm_provider

        config = {"llm": {"provider": "unknown"}}
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(config)
