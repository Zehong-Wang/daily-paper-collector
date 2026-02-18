import asyncio
import json
import logging
import re

from src.llm.base import LLMProvider


class ClaudeCodeProvider(LLMProvider):
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.cli_path = config.get("cli_path", "claude")
        self.model = config.get("model", "sonnet")

    async def complete(self, prompt: str, system: str = "") -> str:
        cmd = [self.cli_path, "--print", "--model", self.model]

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(input=full_prompt.encode("utf-8"))

        if process.returncode != 0:
            raise RuntimeError(
                f"Claude CLI exited with code {process.returncode}: {stderr.decode('utf-8')}"
            )

        return stdout.decode("utf-8").strip()

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        system_msg = (
            f"{system}\nRespond with valid JSON only. No markdown formatting."
            if system
            else "Respond with valid JSON only. No markdown formatting."
        )
        text = await self.complete(prompt, system=system_msg)

        # Strip markdown code fences if present
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        stripped = re.sub(r"\n?```\s*$", "", stripped)

        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {e}") from e
