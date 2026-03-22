"""ResearchAnalyst — Opus-backed meta-analyst for research memo generation.

Synthesizes batch factor evaluation results into concise research memos
(200–400 words) that guide the next round of FactorProposer proposals.

Falls back to a static placeholder when anthropic_api_key is empty.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.schemas import EvalResult

# Import at module level so patch("src.core.research_analyst.ChatAnthropic") works
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None  # type: ignore[assignment,misc]

_MEMO_ROLLING_WINDOW = 5
_MEMO_MAX_TOKENS = 600


class ResearchAnalyst:
    """Synthesize factor evaluation results into research memos using Claude Opus.

    Args:
        config: Full application config. Falls back to get_full_config() if None.
    """

    def __init__(self, config: FullAppConfig | None = None) -> None:
        self.config = config or get_full_config()
        self._llm_available = bool(self.config.anthropic_api_key) and ChatAnthropic is not None
        self._llm: Any = None
        if self._llm_available:
            try:
                self._llm = ChatAnthropic(
                    model=self.config.agents.deep_think_model,
                    api_key=self.config.anthropic_api_key,
                    max_tokens=_MEMO_MAX_TOKENS,
                )
                logger.debug("ResearchAnalyst: Opus LLM initialised.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("ResearchAnalyst: LLM init failed ({}). Using fallback.", exc)
                self._llm_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_memo(
        self,
        batch_results: list[EvalResult],
        kb: dict,
        bt_results: dict | None = None,  # dict[str, BacktestValidationResult]
    ) -> str:
        """Synthesize batch evaluation results into a research memo.

        Args:
            batch_results: EvalResult objects from the latest evaluation batch.
            kb: Current knowledge base dict (read-only in this method).
            bt_results: Optional mapping of factor name to BacktestValidationResult.

        Returns:
            200–400 word research memo string.
        """
        if self._llm_available:
            try:
                return self._llm_write_memo(batch_results, kb, bt_results=bt_results)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ResearchAnalyst.write_memo LLM failed ({}). Using fallback.", exc)
        return self._fallback_memo(batch_results)

    def load_memo(self, kb: dict) -> str:
        """Load the most recent research memo from the KB.

        Args:
            kb: Knowledge base dict with optional 'research_memos' list.

        Returns:
            Most recent memo string, or empty string if none exists.
        """
        memos = kb.get("research_memos", [])
        return memos[-1] if memos else ""

    def save_memo(self, memo: str, kb: dict) -> None:
        """Append memo to KB, maintaining a rolling window of 5 memos.

        Args:
            memo: Memo string to append.
            kb: Knowledge base dict to update in-place.
        """
        if "research_memos" not in kb:
            kb["research_memos"] = []
        kb["research_memos"].append(memo)
        if len(kb["research_memos"]) > _MEMO_ROLLING_WINDOW:
            kb["research_memos"] = kb["research_memos"][-_MEMO_ROLLING_WINDOW:]

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _llm_write_memo(
        self,
        batch_results: list[EvalResult],
        kb: dict,
        bt_results: dict | None = None,
    ) -> str:
        """Call Opus to generate a research memo from batch results."""
        prior_memo = self.load_memo(kb)
        results_table = self._format_results_table(batch_results)
        kb_stats = self._format_kb_stats(kb)

        prompt = (
            f"Prior research memo:\n{prior_memo or 'No prior research.'}\n\n"
            f"Latest batch results:\n{results_table}\n\n"
            f"Knowledge base stats:\n{kb_stats}\n\n"
            "Write a concise 200-400 word research memo covering:\n"
            "1. What worked (factors that passed IC thresholds)\n"
            "2. What failed and why\n"
            "3. Observed patterns across categories\n"
            "4. Hypotheses and recommendations for the next batch"
        )

        response = self._llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a senior quantitative researcher reviewing alpha factor research. "
                    "Write a concise research memo."
                ),
            },
            {"role": "user", "content": prompt},
        ])
        return response.content if hasattr(response, "content") else str(response)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_memo(batch_results: list[EvalResult]) -> str:
        """Generate a static placeholder memo summarising the batch."""
        passed = [r for r in batch_results if r.passed]
        failed = [r for r in batch_results if not r.passed]
        return (
            f"[Fallback memo — no API key configured]\n"
            f"Batch summary: {len(batch_results)} factors evaluated. "
            f"{len(passed)} passed ({[r.factor_name for r in passed]}). "
            f"{len(failed)} failed ({[r.reason for r in failed]}). "
            "No LLM analysis available."
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results_table(results: list[EvalResult]) -> str:
        """Format a list of EvalResult objects as a plain-text table."""
        lines = ["name | stage1_IC | stage2_IC | ICIR | passed"]
        lines.append("-" * 60)
        for r in results:
            s2_ic = f"{r.stage2_ic:.4f}" if r.stage2_ic is not None else "—"
            s2_icir = f"{r.stage2_icir:.4f}" if r.stage2_icir is not None else "—"
            lines.append(
                f"{r.factor_name} | {r.stage1_ic:.4f} | {s2_ic} | {s2_icir} | {r.passed}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_kb_stats(kb: dict) -> str:
        """Summarise key counts from the knowledge base."""
        tested = len(kb.get("tested_factors", []))
        failed = len(kb.get("failed_factors", []))
        discoveries = len(kb.get("discoveries", []))
        return f"tested={tested}, failed={failed}, discoveries={discoveries}"
