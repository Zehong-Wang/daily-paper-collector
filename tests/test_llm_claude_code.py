"""Tests for ClaudeCode LLM provider with mocked subprocess."""

from unittest.mock import AsyncMock, patch

import pytest

from src.llm.claude_code_provider import ClaudeCodeProvider


@pytest.fixture
def claude_code_config():
    return {"cli_path": "claude", "model": "sonnet"}


class TestClaudeCodeProviderInit:
    def test_init_defaults(self, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)
        assert provider.cli_path == "claude"
        assert provider.model == "sonnet"

    def test_init_custom_values(self):
        config = {"cli_path": "/usr/local/bin/claude", "model": "opus"}
        provider = ClaudeCodeProvider(config)
        assert provider.cli_path == "/usr/local/bin/claude"
        assert provider.model == "opus"


class TestClaudeCodeComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_stdout(self, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            b"Claude CLI response",
            b"",
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            result = await provider.complete("test prompt")

        assert result == "Claude CLI response"

        # Verify correct CLI args
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args
        assert call_args[0] == ("claude", "--print", "--model", "sonnet")

    @pytest.mark.asyncio
    async def test_complete_with_system_prepends_to_prompt(self, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"response", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await provider.complete("prompt", system="system msg")

        # Verify the system message was prepended to prompt in stdin
        call_args = mock_process.communicate.call_args
        stdin_data = call_args.kwargs["input"]
        assert b"system msg" in stdin_data
        assert b"prompt" in stdin_data

    @pytest.mark.asyncio
    async def test_complete_raises_on_nonzero_exit(self, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"Error: something failed")
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                await provider.complete("test")


class TestClaudeCodeCompleteJSON:
    @pytest.mark.asyncio
    async def test_complete_json_parses_valid_json(self, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            b'{"score": 7, "reason": "Relevant"}',
            b"",
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await provider.complete_json("score this")

        assert result == {"score": 7, "reason": "Relevant"}

    @pytest.mark.asyncio
    async def test_complete_json_strips_markdown_fences(self, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            b'```json\n{"key": "value"}\n```',
            b"",
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await provider.complete_json("test")

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_complete_json_raises_on_invalid_json(self, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"not json", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(ValueError, match="not valid JSON"):
                await provider.complete_json("test")

    @pytest.mark.asyncio
    async def test_complete_json_adds_no_markdown_instruction(self, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b'{"ok": true}', b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await provider.complete_json("test", system="Be concise")

        # Verify system includes JSON + no markdown instructions
        stdin_data = mock_process.communicate.call_args.kwargs["input"].decode("utf-8")
        assert "Respond with valid JSON only" in stdin_data
        assert "No markdown formatting" in stdin_data
        assert "Be concise" in stdin_data
