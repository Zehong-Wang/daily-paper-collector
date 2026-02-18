from src.llm.base import LLMProvider


def create_llm_provider(config: dict) -> LLMProvider:
    """Create an LLM provider based on config['llm']['provider'].
    Valid values: 'openai', 'claude', 'claude_code'.
    Raises ValueError for unknown provider."""
    provider = config.get("llm", {}).get("provider", "openai")

    if provider == "openai":
        from src.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(config["llm"]["openai"])
    elif provider == "claude":
        from src.llm.claude_provider import ClaudeProvider

        return ClaudeProvider(config["llm"]["claude"])
    elif provider == "claude_code":
        from src.llm.claude_code_provider import ClaudeCodeProvider

        return ClaudeCodeProvider(config["llm"]["claude_code"])
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'")
