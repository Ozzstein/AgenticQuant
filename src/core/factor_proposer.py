"""FactorProposer — LLM-backed proposal generation for all four RD-Agent domains.

Uses Claude Haiku (config.agents.quick_think_model) via langchain-anthropic.
Falls back to keyword/random logic when anthropic_api_key is empty or the LLM
call fails after tenacity retries.
"""
from __future__ import annotations

import random
import re
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.schemas import (
    FactorDefinition,
    FactorProposal,
    FactorProposalList,
    ModelConfigProposal,
    ModelConfigProposalList,
    Strategy,
)

# Import ChatAnthropic at module level so patch("src.core.factor_proposer.ChatAnthropic") works
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Keyword fallback templates (preserved from original RDAgentRunner)
# ---------------------------------------------------------------------------
_FACTOR_TEMPLATES: list[dict[str, str]] = [
    {"name": "momentum_20d", "expression": "close / close.shift(20) - 1", "category": "momentum"},
    {"name": "momentum_5d", "expression": "close / close.shift(5) - 1", "category": "momentum"},
    {"name": "momentum_60d", "expression": "close / close.shift(60) - 1", "category": "momentum"},
    {"name": "volume_ratio", "expression": "volume / volume.rolling(20).mean()", "category": "volume"},
    {"name": "rsi_14_proxy", "expression": "(close - close.rolling(14).min()) / (close.rolling(14).max() - close.rolling(14).min() + 1e-9)", "category": "mean_reversion"},
    {"name": "volatility_20d", "expression": "close.pct_change().rolling(20).std()", "category": "volatility"},
    {"name": "high_low_range", "expression": "(high - low) / close", "category": "volatility"},
    {"name": "close_to_high", "expression": "(high - close) / (high - low + 1e-9)", "category": "mean_reversion"},
    {"name": "obv_proxy", "expression": "(close.pct_change() * volume).rolling(20).sum()", "category": "volume"},
    {"name": "price_to_ma50", "expression": "close / close.rolling(50).mean() - 1", "category": "momentum"},
]

_MODEL_TYPES = ["LightGBM", "CatBoost", "XGBoost", "Linear"]

_STRATEGY_KEYWORD_MAP = {
    "momentum": {"factor_set": ["Alpha158", "momentum_20d"], "model": "LightGBM"},
    "reversion": {"factor_set": ["Alpha158", "rsi_14_proxy"], "model": "LightGBM"},
    "volatility": {"factor_set": ["Alpha158", "volatility_20d"], "model": "LightGBM"},
    "volume": {"factor_set": ["Alpha158", "volume_ratio"], "model": "LightGBM"},
}


class FactorProposer:
    """Generate factor/model/strategy/regime proposals using Claude Haiku.

    Falls back to keyword/random logic when no API key is configured or after
    tenacity exhausts retries on LLM failures.

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
                    model=self.config.agents.quick_think_model,
                    api_key=self.config.anthropic_api_key,
                )
                logger.debug("FactorProposer: Haiku LLM initialised.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("FactorProposer: LLM init failed ({}). Using fallback.", exc)
                self._llm_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_factors(
        self,
        n: int,
        memo: str,
        tested: list[str],
        description: str = "",
    ) -> list[FactorDefinition]:
        """Generate n candidate factor proposals.

        Args:
            n: Number of proposals to generate.
            memo: Latest research memo from ResearchAnalyst (empty string if none).
            tested: Names of already-tested factors to avoid duplicates.
            description: Optional NL context (e.g., paper text or copilot hint).

        Returns:
            List of FactorDefinition objects (name, expression, category populated).
        """
        if self._llm_available:
            try:
                proposals = self._llm_propose_factors(n, memo, tested, description)
                return [self._proposal_to_definition(p) for p in proposals]
            except Exception as exc:  # noqa: BLE001
                logger.warning("FactorProposer.propose_factors LLM failed ({}). Falling back.", exc)
        return self._keyword_propose_factors(n, tested)

    def propose_model_config(
        self,
        n: int,
        memo: str,
        tested: list[dict],
    ) -> list[ModelConfigProposal]:
        """Generate n model hyperparameter proposals.

        Args:
            n: Number of configs to generate.
            memo: Latest research memo.
            tested: Previously tested config dicts (to avoid duplicates).

        Returns:
            List of ModelConfigProposal objects.
        """
        if self._llm_available:
            try:
                return self._llm_propose_model_config(n, memo, tested)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FactorProposer.propose_model_config LLM failed ({}). Falling back.", exc)
        return self._random_model_configs(n)

    def propose_strategy(self, description: str, memo: str) -> Strategy:
        """Generate a Strategy from a natural-language description.

        Args:
            description: NL description of the desired strategy.
            memo: Latest research memo.

        Returns:
            A populated Strategy Pydantic model.
        """
        if self._llm_available:
            try:
                return self._llm_propose_strategy(description, memo)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FactorProposer.propose_strategy LLM failed ({}). Falling back.", exc)
        return self._keyword_strategy(description)

    def propose_regime_change(self, description: str, memo: str) -> dict:
        """Generate a regime configuration change from a natural-language description.

        Args:
            description: NL description of the desired regime adjustment.
            memo: Latest research memo.

        Returns:
            Dict with keys: regime, allocation_adjustments, rationale.
        """
        if self._llm_available:
            try:
                return self._llm_propose_regime(description, memo)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FactorProposer.propose_regime_change LLM failed ({}). Falling back.", exc)
        return self._keyword_regime(description)

    # ------------------------------------------------------------------
    # LLM paths (with tenacity retry)
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _llm_propose_factors(
        self,
        n: int,
        memo: str,
        tested: list[str],
        description: str,
    ) -> list[FactorProposal]:
        chain = self._llm.with_structured_output(FactorProposalList)
        tested_str = ", ".join(tested[:20]) if tested else "none"
        context = f"\nAdditional context: {description}" if description else ""
        user_msg = (
            f"Generate {n} novel alpha factor expressions for US equities.\n"
            f"Already tested (do not repeat): {tested_str}\n"
            f"Research memo:\n{memo or 'No prior research.'}"
            f"{context}\n\n"
            "Requirements:\n"
            "- Expressions must use only: close, high, low, open, volume (pandas Series)\n"
            "- Use standard pandas methods: .shift(), .rolling(), .mean(), .std(), .pct_change()\n"
            "- No external libraries\n"
            f"- Provide exactly {n} factors"
        )
        result: FactorProposalList = chain.invoke([
            {"role": "system", "content": "You are a quantitative researcher. Generate novel alpha factor expressions for US equities."},
            {"role": "user", "content": user_msg},
        ])
        # Deduplicate against tested list
        filtered = [p for p in result.factors if p.name not in tested]
        if len(filtered) < n:
            logger.warning(
                "FactorProposer: LLM returned {}/{} non-duplicate factors; supplementing with fallback.",
                len(filtered),
                n,
            )
        return filtered

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _llm_propose_model_config(
        self,
        n: int,
        memo: str,
        tested: list[dict],
    ) -> list[ModelConfigProposal]:
        chain = self._llm.with_structured_output(ModelConfigProposalList)
        tested_str = str(tested[:5]) if tested else "none"
        user_msg = (
            f"Propose {n} LightGBM/CatBoost/XGBoost/Linear model hyperparameter configurations "
            f"for a cross-sectional equity alpha model.\n"
            f"Already tested: {tested_str}\n"
            f"Research memo:\n{memo or 'No prior research.'}\n\n"
            f"For each config provide: model_type, n_estimators (50-2000), "
            f"learning_rate (0.001-0.3), max_depth (3-12), num_leaves (15-255), "
            f"and a 1-sentence rationale."
        )
        result: ModelConfigProposalList = chain.invoke([
            {"role": "system", "content": "You are a quantitative researcher optimizing ML model hyperparameters for equity prediction."},
            {"role": "user", "content": user_msg},
        ])
        return result.configs[:n]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _llm_propose_strategy(self, description: str, memo: str) -> Strategy:
        from langchain_core.messages import HumanMessage, SystemMessage
        chain = self._llm.with_structured_output(Strategy)
        result: Strategy = chain.invoke([
            SystemMessage(content="You are a quantitative portfolio strategist. Propose a complete trading strategy definition."),
            HumanMessage(content=(
                f"Description: {description}\n"
                f"Research memo:\n{memo or 'No prior research.'}\n\n"
                "Provide a Strategy with name, description, factor_set, model, universe, "
                "entry_rules, exit_rules, sizing_method, rebalance_frequency."
            )),
        ])
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _llm_propose_regime(self, description: str, memo: str) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = self._llm.invoke([
            SystemMessage(content="You are a macro regime strategist. Propose allocation adjustments as a JSON dict."),
            HumanMessage(content=(
                f"Description: {description}\n"
                f"Research memo:\n{memo or 'No prior research.'}\n\n"
                "Return JSON with keys: regime (risk_on/neutral/risk_off/crisis), "
                "allocation_adjustments (dict of asset class to pct), rationale (string)."
            )),
        ])
        import json
        content = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"regime": "neutral", "allocation_adjustments": {}, "rationale": content}

    # ------------------------------------------------------------------
    # Keyword / random fallbacks
    # ------------------------------------------------------------------

    def _keyword_propose_factors(self, n: int, tested: list[str]) -> list[FactorDefinition]:
        """Sample n factors from templates, excluding tested names."""
        tested_set = set(tested)
        available = [t for t in _FACTOR_TEMPLATES if t["name"] not in tested_set]
        if len(available) < n:
            for tmpl in _FACTOR_TEMPLATES:
                for suffix in range(1, 10):
                    candidate = f"{tmpl['name']}_v{suffix}"
                    if candidate not in tested_set:
                        augmented = dict(tmpl)
                        augmented["name"] = candidate
                        available.append(augmented)
                    if len(available) >= n:
                        break
                if len(available) >= n:
                    break

        random.shuffle(available)
        selected = available[:n]
        return [
            FactorDefinition(
                name=t["name"],
                expression=t["expression"],
                category=t["category"],
                source="rd_agent_fallback",
            )
            for t in selected
        ]

    def _random_model_configs(self, n: int) -> list[ModelConfigProposal]:
        results = []
        for _ in range(n):
            model_type = random.choice(_MODEL_TYPES)
            results.append(ModelConfigProposal(
                model_type=model_type,
                n_estimators=random.choice([100, 200, 500, 1000]),
                learning_rate=random.choice([0.01, 0.05, 0.1]),
                max_depth=random.choice([4, 6, 8, 10]),
                num_leaves=random.choice([31, 63, 127]),
                rationale=f"Random {model_type} config (fallback mode)",
            ))
        return results

    def _keyword_strategy(self, description: str) -> Strategy:
        desc_lower = description.lower()
        matched = None
        for keyword, spec in _STRATEGY_KEYWORD_MAP.items():
            if keyword in desc_lower:
                matched = spec
                break
        if matched is None:
            matched = {"factor_set": ["Alpha158"], "model": "LightGBM"}

        name_part = re.sub(r"[^a-z0-9]+", "_", desc_lower)[:30].strip("_")
        return Strategy(
            name=f"copilot_{name_part}",
            description=description,
            factor_set=matched["factor_set"],
            model=matched["model"],
            source="rd_agent_fallback",
        )

    def _keyword_regime(self, description: str) -> dict:
        desc_lower = description.lower()
        if "risk_on" in desc_lower or "bull" in desc_lower:
            regime = "risk_on"
        elif "risk_off" in desc_lower or "bear" in desc_lower:
            regime = "risk_off"
        elif "crisis" in desc_lower:
            regime = "crisis"
        else:
            regime = "neutral"
        return {
            "regime": regime,
            "allocation_adjustments": {},
            "rationale": "Keyword-matched regime from description (fallback mode)",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _proposal_to_definition(proposal: FactorProposal) -> FactorDefinition:
        return FactorDefinition(
            name=proposal.name,
            expression=proposal.expression,
            category=proposal.category,
            description=proposal.description,
            source="rd_agent_llm",
        )
