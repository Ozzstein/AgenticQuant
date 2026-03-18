"""LLM factory for the QuantAgentLab agent pipeline."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.utils.config import AgentConfig, get_config
from src.utils.exceptions import LLMError
from src.utils.logger import get_logger

_log = get_logger("agents.llm")


class _CostTrackingCallback(BaseCallbackHandler):
    """LangChain callback that records token usage in :class:`CostTracker`.

    Args:
        model_name: The LLM model identifier used to look up pricing.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Extract token counts from *response* and record them in CostTracker.

        Args:
            response: The completed LLM response containing token usage metadata.
            run_id: LangChain run identifier (unused).
            parent_run_id: Parent run identifier (unused).
            tags: Optional tags attached to this run (unused).
            **kwargs: Additional keyword arguments forwarded by LangChain.
        """
        try:
            from src.utils.cost_tracker import CostTracker

            usage = (response.llm_output or {}).get("usage", {})
            if not usage:
                _log.debug("CostTracker: response for {} has no usage metadata", self.model_name)
            # Anthropic surfaces token counts under different keys depending on version
            prompt_tokens = int(
                usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or 0
            )
            completion_tokens = int(
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or 0
            )
            CostTracker().record_call(self.model_name, prompt_tokens, completion_tokens)
        except Exception as exc:  # noqa: BLE001
            _log.warning("CostTracker callback failed (non-fatal): {}", exc)


def get_llm(
    config: AgentConfig,
    *,
    mode: str = "standard",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Return a configured LangChain chat model for the requested mode.

    A :class:`_CostTrackingCallback` is attached to every returned model so that
    token usage is transparently recorded in :class:`~src.utils.cost_tracker.CostTracker`
    after each invocation.  The callback is wrapped with try/except inside
    :meth:`_CostTrackingCallback.on_llm_end` so cost-tracking failures can never
    propagate to the caller.

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

    callbacks = [_CostTrackingCallback(model_name)]
    provider = config.llm_provider.lower()

    if provider == "anthropic":
        app_config = get_config()
        kwargs: dict = {
            "model": model_name,
            "temperature": temp,
            "max_tokens_to_sample": tokens,
            "callbacks": callbacks,
        }
        if app_config.anthropic_api_key:
            kwargs["api_key"] = app_config.anthropic_api_key
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        app_config = get_config()
        kwargs = {
            "model": model_name,
            "temperature": temp,
            "max_tokens": tokens,
            "callbacks": callbacks,
        }
        if app_config.openai_api_key:
            kwargs["api_key"] = app_config.openai_api_key
        return ChatOpenAI(**kwargs)

    if provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=temp,
            callbacks=callbacks,
        )

    raise LLMError(f"Unknown LLM provider: {config.llm_provider}")
