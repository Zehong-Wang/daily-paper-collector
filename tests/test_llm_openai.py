"""Tests for OpenAI LLM provider with mocked API calls."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.openai_provider import OpenAIProvider


@pytest.fixture
def openai_config():
    return {"model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"}


@pytest.fixture
def mock_env():
    os.environ["OPENAI_API_KEY"] = "sk-test-key"
    yield
    del os.environ["OPENAI_API_KEY"]


class TestOpenAIProviderInit:
    def test_init_success(self, mock_env, openai_config):
        provider = OpenAIProvider(openai_config)
        assert provider.model == "gpt-4o-mini"

    def test_init_missing_api_key(self, openai_config):
        # Ensure the key is not set
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIProvider(openai_config)


class TestOpenAIComplete:
    @pytest.mark.asyncio
    async def test_complete_calls_api_correctly(self, mock_env, openai_config):
        provider = OpenAIProvider(openai_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.complete("test prompt")

        assert result == "Test response"
        provider.client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test prompt"}],
        )

    @pytest.mark.asyncio
    async def test_complete_with_system_message(self, mock_env, openai_config):
        provider = OpenAIProvider(openai_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.complete("prompt", system="system msg")

        call_args = provider.client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "system msg"}
        assert messages[1] == {"role": "user", "content": "prompt"}


class TestOpenAICompleteJSON:
    @pytest.mark.asyncio
    async def test_complete_json_parses_valid_json(self, mock_env, openai_config):
        provider = OpenAIProvider(openai_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 8.5, "reason": "Relevant"}'

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.complete_json("score this paper")

        assert result == {"score": 8.5, "reason": "Relevant"}

        # Verify response_format was set
        call_args = provider.client.chat.completions.create.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_complete_json_raises_on_invalid_json(self, mock_env, openai_config):
        provider = OpenAIProvider(openai_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not json at all"

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="not valid JSON"):
            await provider.complete_json("test")

    @pytest.mark.asyncio
    async def test_complete_json_includes_json_instruction_in_system(self, mock_env, openai_config):
        provider = OpenAIProvider(openai_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"ok": true}'

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.complete_json("test", system="Be helpful")

        call_args = provider.client.chat.completions.create.call_args
        system_msg = call_args.kwargs["messages"][0]["content"]
        assert "Respond with valid JSON only" in system_msg
        assert "Be helpful" in system_msg
