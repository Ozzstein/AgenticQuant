"""LLM factory for the QuantAgentLab agent pipeline."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.utils.config import AgentConfig, get_config
from src.utils.exceptions import LLMError
from src.utils.logger import get_logger

_log = get_logger("agents.llm")


def get_llm(
    config: AgentConfig,
    *,
    mode: str = "standard",
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_seconds: float = 30,
    app_config=None,
) -> BaseChatModel:
    """Return a configured LangChain chat model for the requested mode.

    The desk's AgentCallBudget admits calls and records actual raw usage with
    explicit configured pricing. The legacy cost callback is deliberately absent:
    its unknown-price/unknown-usage zero fallback is not a valid cost measurement.

    Args:
        config: AgentConfig instance with provider/model settings.
        mode: LLM persona — "standard", "deep", or "quick".
        temperature: Override temperature; falls back to config.temperature when None.
        max_tokens: Override max output tokens; falls back to config.max_tokens when None.

    Returns:
        A BaseChatModel instance ready for inference.

    Raises:
        LLMError: If ``mode`` or ``config.llm_provider`` is unrecognised.
    """
    _mode_map = {
        "standard": config.llm_model,
        "deep": config.deep_think_model,
        "quick": config.quick_think_model,
    }
    if mode not in _mode_map:
        raise LLMError(f"Unknown LLM mode: {mode}")

    model_name = _mode_map[mode]
    temp = temperature if temperature is not None else config.temperature
    tokens = max_tokens if max_tokens is not None else config.max_tokens

    _log.debug(
        "Building LLM: provider={} model={} mode={} temp={}",
        config.llm_provider,
        model_name,
        mode,
        temp,
    )

    provider = config.llm_provider.lower()

    if provider == "anthropic":
        app_config = app_config or get_config()
        kwargs: dict = {
            "model": model_name,
            "temperature": temp,
            "max_tokens_to_sample": tokens,
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if app_config.anthropic_api_key:
            kwargs["api_key"] = app_config.anthropic_api_key
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        app_config = app_config or get_config()
        kwargs = {
            "model": model_name,
            "temperature": temp,
            "max_tokens": tokens,
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if app_config.openai_api_key:
            kwargs["api_key"] = app_config.openai_api_key
        return ChatOpenAI(**kwargs)

    if provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=temp,
            client_kwargs={"timeout": timeout_seconds},
        )

    raise LLMError(f"Unknown LLM provider: {config.llm_provider}")
