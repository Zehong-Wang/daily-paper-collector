from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the text response."""
        ...

    @abstractmethod
    async def complete_json(self, prompt: str, system: str = "") -> dict:
        """Send a prompt and parse the response as JSON.
        Implementations should instruct the model to return valid JSON
        and parse the result with json.loads().
        Raise ValueError if the response is not valid JSON."""
        ...
