"""Tests for ClaudeCode LLM provider with mocked subprocess."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from src.llm.claude_code_provider import ClaudeCodeProvider


def _make_envelope(result_text: str) -> bytes:
    """Build a JSON envelope like --output-format json produces."""
    envelope = {
        "type": "result",
        "subtype": "success",
        "cost_usd": 0.003,
        "is_error": False,
        "result": result_text,
        "session_id": "test-session",
        "num_turns": 1,
    }
    return json.dumps(envelope).encode("utf-8")


@pytest.fixture
def claude_code_config():
    return {"cli_path": "claude", "model": "sonnet", "timeout": 120, "max_retries": 3}


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestClaudeCodeProviderInit:
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    def test_init_defaults(self, _mock_which):
        provider = ClaudeCodeProvider({"cli_path": "claude"})
        assert provider.cli_path == "claude"
        assert provider.model == "sonnet"
        assert provider.timeout == 120
        assert provider.max_retries == 3

    @patch("shutil.which", return_value="/custom/path/claude")
    def test_init_custom_values(self, _mock_which):
        config = {
            "cli_path": "/custom/path/claude",
            "model": "opus",
            "timeout": 60,
            "max_retries": 5,
        }
        provider = ClaudeCodeProvider(config)
        assert provider.cli_path == "/custom/path/claude"
        assert provider.model == "opus"
        assert provider.timeout == 60
        assert provider.max_retries == 5

    @patch("shutil.which", return_value=None)
    def test_init_cli_not_found(self, _mock_which):
        with pytest.raises(RuntimeError, match="Claude CLI not found"):
            ClaudeCodeProvider({"cli_path": "nonexistent"})

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    def test_init_cli_found(self, mock_which):
        provider = ClaudeCodeProvider({"cli_path": "claude"})
        mock_which.assert_called_once_with("claude")
        assert provider.cli_path == "claude"


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestClaudeCodeComplete:
    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_returns_envelope_result(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            _make_envelope("Claude CLI response"),
            b"",
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await provider.complete("test prompt")

        assert result == "Claude CLI response"

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_with_system_uses_flag(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (_make_envelope("response"), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await provider.complete("prompt", system="system msg")

        call_args = mock_exec.call_args[0]
        assert "--system-prompt" in call_args
        idx = call_args.index("--system-prompt")
        assert call_args[idx + 1] == "system msg"

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_without_system_omits_flag(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (_make_envelope("response"), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await provider.complete("prompt")

        call_args = mock_exec.call_args[0]
        assert "--system-prompt" not in call_args

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_has_output_format_json(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (_make_envelope("ok"), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await provider.complete("prompt")

        call_args = mock_exec.call_args[0]
        assert "--output-format" in call_args
        idx = call_args.index("--output-format")
        assert call_args[idx + 1] == "json"

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_has_no_session_persistence(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (_make_envelope("ok"), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await provider.complete("prompt")

        call_args = mock_exec.call_args[0]
        assert "--no-session-persistence" in call_args

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_sends_prompt_via_stdin(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (_make_envelope("ok"), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await provider.complete("my test prompt")

        stdin_data = mock_process.communicate.call_args.kwargs["input"]
        assert stdin_data == b"my test prompt"

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_raises_on_nonzero_exit(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)
        provider.max_retries = 1  # Don't retry in this test

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"Error: something failed")
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                await provider.complete("test")

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_timeout_kills_process(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)
        provider.timeout = 1
        provider.max_retries = 1

        mock_process = AsyncMock()
        mock_process.kill = AsyncMock()

        async def mock_wait_for(coro, timeout):
            # Consume the coroutine to avoid RuntimeWarning
            coro.close()
            raise asyncio.TimeoutError()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("src.llm.claude_code_provider.asyncio.wait_for", side_effect=mock_wait_for):
                with pytest.raises(RuntimeError, match="timed out after 1s"):
                    await provider.complete("test")

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_fallback_to_raw_stdout(self, _mock_which, claude_code_config):
        """If stdout is not a valid JSON envelope, fall back to raw text."""
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"plain text response", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await provider.complete("test")

        assert result == "plain text response"

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_correct_cli_args_order(self, _mock_which, claude_code_config):
        """Verify the full CLI command structure."""
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (_make_envelope("ok"), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await provider.complete("prompt", system="sys")

        call_args = mock_exec.call_args[0]
        assert call_args[0] == "claude"
        assert call_args[1] == "--print"
        assert "--model" in call_args
        assert "--output-format" in call_args
        assert "--no-session-persistence" in call_args
        assert "--system-prompt" in call_args


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestClaudeCodeRetry:
    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_retry_on_transient_failure(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        call_count = 0

        async def mock_execute(prompt, system=""):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("CLI failed")
            return json.dumps({"type": "result", "result": "success"})

        with patch.object(provider, "_execute_subprocess", side_effect=mock_execute):
            with patch(
                "src.llm.claude_code_provider.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep:
                result = await provider.complete("test")

        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_retry_exhaustion(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        async def always_fail(prompt, system=""):
            raise RuntimeError("CLI failed")

        with patch.object(provider, "_execute_subprocess", side_effect=always_fail):
            with patch("src.llm.claude_code_provider.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match="CLI failed"):
                    await provider.complete("test")

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_no_retry_on_json_parse_error(self, _mock_which, claude_code_config):
        """ValueError from JSON parsing should not trigger retries."""
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            _make_envelope("not valid json {{{"),
            b"",
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(ValueError, match="not valid JSON"):
                await provider.complete_json("test")

        # complete() succeeded (no RuntimeError), only JSON parse failed — no retry
        assert mock_process.communicate.call_count == 1

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_retry_backoff_delays(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        call_count = 0

        async def fail_then_succeed(prompt, system=""):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("CLI failed")
            return json.dumps({"type": "result", "result": "ok"})

        with patch.object(provider, "_execute_subprocess", side_effect=fail_then_succeed):
            with patch(
                "src.llm.claude_code_provider.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep:
                await provider.complete("test")

        # First retry: 2^0 = 1s, second retry: 2^1 = 2s
        assert mock_sleep.call_args_list[0][0][0] == 1
        assert mock_sleep.call_args_list[1][0][0] == 2

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_single_retry_config(self, _mock_which):
        """With max_retries=1, no retries happen — failure raises immediately."""
        provider = ClaudeCodeProvider({"cli_path": "claude", "model": "sonnet", "max_retries": 1})

        async def always_fail(prompt, system=""):
            raise RuntimeError("CLI failed")

        with patch.object(provider, "_execute_subprocess", side_effect=always_fail):
            with pytest.raises(RuntimeError, match="CLI failed"):
                await provider.complete("test")


# ---------------------------------------------------------------------------
# complete_json()
# ---------------------------------------------------------------------------


class TestClaudeCodeCompleteJSON:
    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_json_parses_from_envelope(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            _make_envelope('{"score": 7, "reason": "Relevant"}'),
            b"",
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await provider.complete_json("score this")

        assert result == {"score": 7, "reason": "Relevant"}

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_json_strips_markdown_fences_fallback(
        self, _mock_which, claude_code_config
    ):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            _make_envelope('```json\n{"key": "value"}\n```'),
            b"",
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await provider.complete_json("test")

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_json_raises_on_invalid_json(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (_make_envelope("not json"), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(ValueError, match="not valid JSON"):
                await provider.complete_json("test")

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_json_adds_json_instruction_to_system(
        self, _mock_which, claude_code_config
    ):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            _make_envelope('{"ok": true}'),
            b"",
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await provider.complete_json("test", system="Be concise")

        call_args = mock_exec.call_args[0]
        idx = call_args.index("--system-prompt")
        system_value = call_args[idx + 1]
        assert "Respond with valid JSON only" in system_value
        assert "No markdown formatting" in system_value
        assert "Be concise" in system_value

    @pytest.mark.asyncio
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    async def test_complete_json_without_custom_system(self, _mock_which, claude_code_config):
        provider = ClaudeCodeProvider(claude_code_config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            _make_envelope('{"ok": true}'),
            b"",
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await provider.complete_json("test")

        call_args = mock_exec.call_args[0]
        idx = call_args.index("--system-prompt")
        system_value = call_args[idx + 1]
        assert system_value == "Respond with valid JSON only. No markdown formatting."
