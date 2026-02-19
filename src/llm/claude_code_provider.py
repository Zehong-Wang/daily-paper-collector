import asyncio
import json
import logging
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
            if isinstance(envelope, dict) and "result" in envelope:
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
            except RuntimeError:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2**attempt
                self.logger.warning("CLI attempt %d failed, retrying in %ds...", attempt + 1, wait)
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
        if system:
            cmd.extend(["--system-prompt", system])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            raise RuntimeError(f"Claude CLI timed out after {self.timeout}s")

        if process.returncode != 0:
            raise RuntimeError(
                f"Claude CLI exited with code {process.returncode}: {stderr.decode('utf-8')}"
            )

        return stdout.decode("utf-8")
