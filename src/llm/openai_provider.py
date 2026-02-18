import json
import logging
import os

import openai

from src.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_env}' is not set")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4o-mini")

    async def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        system_msg = (
            f"{system}\nRespond with valid JSON only."
            if system
            else "Respond with valid JSON only."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {e}") from e
