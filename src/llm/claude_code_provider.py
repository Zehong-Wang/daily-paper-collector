import asyncio
import json
import logging
import os
import re
import shutil

from src.llm.base import LLMProvider


class ClaudeCodeProvider(LLMProvider):
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.cli_path = config.get("cli_path", "claude")
        self.model = config.get("model", "sonnet")
        self.timeout = config.get("timeout", 120)
        self.max_retries = config.get("max_retries", 3)
        self.permission_mode = config.get("permission_mode", "dontAsk")
        self.disable_tools = config.get("disable_tools", True)
        self.inherit_anthropic_api_key = config.get("inherit_anthropic_api_key", False)
        self._logged_env_override = False

        if not shutil.which(self.cli_path):
            raise RuntimeError(
                f"Claude CLI not found at '{self.cli_path}'. "
                "Install it or set 'cli_path' in config."
            )

    async def complete(self, prompt: str, system: str = "") -> str:
        raw = await self._run_cli(prompt, system)

        # Parse the JSON envelope from --output-format json
        try:
            envelope = json.loads(raw)
            if isinstance(envelope, dict):
                if envelope.get("is_error"):
                    raise RuntimeError(f"Claude CLI returned error: {envelope.get('result', raw)}")
                if "result" in envelope:
                    return str(envelope["result"]).strip()
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: return raw stdout if envelope parsing fails
        return raw.strip()

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        system_msg = (
            f"{system}\nRespond with valid JSON only. No markdown formatting."
            if system
            else "Respond with valid JSON only. No markdown formatting."
        )
        text = await self.complete(prompt, system=system_msg)

        # Strip markdown code fences if present (fallback for non-envelope output)
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        stripped = re.sub(r"\n?```\s*$", "", stripped)

        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {e}") from e

    async def _run_cli(self, prompt: str, system: str = "") -> str:
        """Execute CLI with retry logic. Returns raw stdout text."""
        for attempt in range(self.max_retries):
            try:
                return await self._execute_subprocess(prompt, system)
            except RuntimeError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2**attempt
                self.logger.warning(
                    "CLI attempt %d/%d failed: %s — retrying in %ds...",
                    attempt + 1, self.max_retries, e, wait,
                )
                await asyncio.sleep(wait)

    async def _execute_subprocess(self, prompt: str, system: str = "") -> str:
        """Run the Claude CLI subprocess with timeout."""
        cmd = [
            self.cli_path,
            "--print",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--no-session-persistence",
        ]
        if self.permission_mode:
            cmd.extend(["--permission-mode", self.permission_mode])
        if self.disable_tools:
            cmd.extend(["--tools", ""])
        if system:
            cmd.extend(["--system-prompt", system])

        child_env = os.environ.copy()
        if not self.inherit_anthropic_api_key:
            removed_key = child_env.pop("ANTHROPIC_API_KEY", None)
            if removed_key and not self._logged_env_override:
                self.logger.info(
                    "Ignoring ANTHROPIC_API_KEY for claude_code provider to use local CLI auth/session."
                )
                self._logged_env_override = True

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            kill_result = process.kill()
            if asyncio.iscoroutine(kill_result):
                await kill_result
            await process.wait()
            raise RuntimeError(f"Claude CLI timed out after {self.timeout}s")

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if process.returncode != 0:
            error_text = (
                self._extract_envelope_message(stderr_text)
                or self._extract_envelope_message(stdout_text)
            )
            if not error_text:
                error_text = "No error details in stdout/stderr. Run 'claude --print --debug' to diagnose."
            raise RuntimeError(
                f"Claude CLI exited with code {process.returncode}: {error_text}"
            )

        return stdout_text

    @staticmethod
    def _extract_envelope_message(text: str) -> str:
        """Extract a human-readable message from raw text or a JSON envelope."""
        if not text:
            return ""
        stripped = text.strip()
        try:
            envelope = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return stripped

        if not isinstance(envelope, dict):
            return stripped

        for key in ("result", "error", "message"):
            if envelope.get(key):
                return str(envelope[key]).strip()
        return stripped
