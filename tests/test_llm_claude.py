"""Tests for Claude LLM provider with mocked API calls."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.claude_provider import ClaudeProvider


@pytest.fixture
def claude_config():
    return {"model": "claude-sonnet-4-5-20250929", "api_key_env": "ANTHROPIC_API_KEY"}


@pytest.fixture
def mock_env():
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
    yield
    del os.environ["ANTHROPIC_API_KEY"]


class TestClaudeProviderInit:
    def test_init_success(self, mock_env, claude_config):
        provider = ClaudeProvider(claude_config)
        assert provider.model == "claude-sonnet-4-5-20250929"

    def test_init_missing_api_key(self, claude_config):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            ClaudeProvider(claude_config)


class TestClaudeComplete:
    @pytest.mark.asyncio
    async def test_complete_calls_messages_create(self, mock_env, claude_config):
        provider = ClaudeProvider(claude_config)

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Claude response"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.complete("test prompt")

        assert result == "Claude response"
        provider.client.messages.create.assert_called_once()

        call_kwargs = provider.client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-5-20250929"
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]
        assert "system" not in call_kwargs

    @pytest.mark.asyncio
    async def test_complete_with_system_message(self, mock_env, claude_config):
        provider = ClaudeProvider(claude_config)

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Response"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        await provider.complete("prompt", system="system msg")

        call_kwargs = provider.client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "system msg"


class TestClaudeCompleteJSON:
    @pytest.mark.asyncio
    async def test_complete_json_parses_valid_json(self, mock_env, claude_config):
        provider = ClaudeProvider(claude_config)

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '{"score": 9, "reason": "Very relevant"}'

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.complete_json("score this paper")

        assert result == {"score": 9, "reason": "Very relevant"}

    @pytest.mark.asyncio
    async def test_complete_json_strips_markdown_fences(self, mock_env, claude_config):
        provider = ClaudeProvider(claude_config)

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '```json\n{"key": "value"}\n```'

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.complete_json("test")

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_complete_json_raises_on_invalid_json(self, mock_env, claude_config):
        provider = ClaudeProvider(claude_config)

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "not json"

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="not valid JSON"):
            await provider.complete_json("test")

    @pytest.mark.asyncio
    async def test_complete_json_adds_json_instruction(self, mock_env, claude_config):
        provider = ClaudeProvider(claude_config)

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '{"ok": true}'

        provider.client.messages.create = AsyncMock(return_value=mock_response)

        await provider.complete_json("test", system="Be concise")

        call_kwargs = provider.client.messages.create.call_args.kwargs
        assert "Respond with valid JSON only" in call_kwargs["system"]
        assert "Be concise" in call_kwargs["system"]
