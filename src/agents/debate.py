"""DebateManager, debate_node, and debate_router for multi-round analyst debate."""

from __future__ import annotations

try:
    from src.agents.llm import get_llm
    from src.agents.state import TradingDeskState
except ImportError:
    import operator
    from typing import Annotated, TypedDict

    from src.utils.schemas import AnalysisResult

    def _merge_reports(a, b):
        return {**a, **b}

    class TradingDeskState(TypedDict):  # type: ignore[no-redef]
        ticker: str
        asset_class: str
        qlib_context: str
        market_context: str
        macro_regime: str
        macro_confidence: float
        agent_reports: Annotated[dict, _merge_reports]
        debate_round: int
        debate_transcript: Annotated[list, operator.add]
        risk_veto: bool
        risk_veto_reason: str
        result: AnalysisResult | None

    def get_llm(config, *, mode="standard", temperature=None, max_tokens=None):
        raise ImportError("llm.py not available")


from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.utils.config import AgentConfig, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DebateManager:
    """Manages the multi-round debate between analyst agents."""

    @staticmethod
    def should_debate(state: TradingDeskState, config: AgentConfig) -> bool:
        """Return True if a debate round should be triggered.

        Conditions (ALL must be true):
        - state["debate_round"] < config.max_debate_rounds
        - spread between highest and lowest agent confidence > config.debate_trigger_spread

        Args:
            state: Current trading desk state.
            config: Agent configuration with debate parameters.

        Returns:
            True if debate should proceed, False otherwise.
        """
        if state.get("debate_round", 0) >= config.max_debate_rounds:
            return False

        reports = state.get("agent_reports", {})
        # Exclude risk_manager from debate spread calculation
        analyst_reports = {k: v for k, v in reports.items() if k != "risk_manager"}
        if len(analyst_reports) < 2:
            return False

        confidences = [r.confidence for r in analyst_reports.values()]
        spread = max(confidences) - min(confidences)
        return spread > config.debate_trigger_spread

    @staticmethod
    def run_debate_round(state: TradingDeskState, llm: BaseChatModel) -> dict:
        """Run one debate round.

        Each analyst agent with below-median confidence gets to respond
        to the highest-confidence agent's reasoning.

        Args:
            state: Current trading desk state.
            llm: LLM to use for generating debate responses.

        Returns:
            Partial state dict with updated debate_transcript and debate_round.
        """
        reports = state.get("agent_reports", {})
        analyst_reports = {k: v for k, v in reports.items() if k != "risk_manager"}

        if not analyst_reports:
            return {"debate_round": state.get("debate_round", 0) + 1, "debate_transcript": []}

        round_num = state.get("debate_round", 0) + 1
        transcript_entries: list[str] = [f"\n=== Debate Round {round_num} ==="]

        # Find highest confidence agent
        best_agent = max(analyst_reports, key=lambda k: analyst_reports[k].confidence)
        best_report = analyst_reports[best_agent]

        # Other agents respond — only those below median confidence debate
        sorted_confs = sorted(r.confidence for r in analyst_reports.values())
        median_conf = sorted_confs[len(analyst_reports) // 2]

        for agent_name, report in analyst_reports.items():
            if agent_name == best_agent:
                continue
            if report.confidence >= median_conf:
                continue  # Only low-confidence agents debate

            prompt = (
                f"You are the {agent_name} analyst.\n"
                f"The {best_agent} analyst argued (confidence {best_report.confidence:.0f}%):\n"
                f"{best_report.reasoning[:500]}\n\n"
                f"Your current position ({report.decision}, confidence {report.confidence:.0f}%):\n"
                f"{report.reasoning[:300]}\n\n"
                "Do you want to revise your position? Provide updated reasoning in 2-3 sentences.\n"
                "If you maintain your position, explain why. Be concise."
            )

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                content = response.content if hasattr(response, "content") else str(response)
                transcript_entries.append(f"[{agent_name}]: {content[:300]}")
            except Exception as e:
                logger.warning(f"Debate round failed for {agent_name}: {e}")
                transcript_entries.append(f"[{agent_name}]: (no response - {str(e)[:50]})")

        return {
            "debate_round": round_num,
            "debate_transcript": transcript_entries,
        }


def debate_node(state: TradingDeskState, llm: BaseChatModel | None = None) -> dict:
    """Run one debate round between analyst agents.

    Args:
        state: Current trading desk state.
        llm: Optional LLM override. If None, creates from config.

    Returns:
        Partial state dict with debate_round incremented and debate_transcript appended.
    """
    if llm is None:
        config = get_config()
        llm = get_llm(config.agents)

    return DebateManager.run_debate_round(state, llm)


def debate_router(state: TradingDeskState) -> str:
    """Route to 'debate' or 'risk_manager' based on debate conditions.

    Args:
        state: Current trading desk state.

    Returns:
        "debate" if debate should continue, "risk_manager" otherwise.
    """
    config = get_config()
    if DebateManager.should_debate(state, config.agents):
        logger.info(f"Routing to debate (round {state.get('debate_round', 0) + 1})")
        return "debate"
    logger.info("Routing to risk_manager")
    return "risk_manager"
