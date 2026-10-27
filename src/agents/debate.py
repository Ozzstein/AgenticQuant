"""Configured challenge rounds with source-linked revised analyst reports."""

from src.agents.desk import structured_report
from src.utils.config import get_config


class DebateManager:
    @staticmethod
    def should_debate(state, config):
        # Unlike legacy confidence fields are never compared or averaged. Full
        # challenge rounds remain required even when initial recommendations agree.
        return state.get("debate_round", 0) < config.max_debate_rounds

    @staticmethod
    def run_debate_round(state, llm=None):
        reports = state.get("desk_reports", {})
        round_number = state.get("debate_round", 0) + 1
        updates, legacy, transcript = {}, {}, [f"=== Debate Round {round_number} ==="]
        context = {
            "reports": [r.model_dump(mode="json") for r in reports.values()],
            "citation_checks": [
                {"claim": c.claim.claim, "status": c.status, "reason": c.reason}
                for c in state.get("citation_checks", ())
            ],
            "prior_debate": state.get("debate_transcript", []),
        }
        for role, report in reports.items():
            if role in ("risk_manager", "strategist") or report.status == "not_applicable":
                continue
            result = structured_report(
                state,
                role,
                "Challenge the other analysts using source evidence. Address discrepancies and "
                "unsupported claims. Revise your own recommendation and reasoning when warranted; "
                "otherwise explain why the contrary evidence does not change it. Preserve exact citations.",
                llm=llm,
                kinds=(
                    "fundamental",
                    "technical",
                    "news",
                    "transcript",
                    "macro",
                    "memory",
                    "other",
                ),
                extra=context,
            )
            revised = result["desk_reports"][role]
            updates[role] = revised
            legacy.update(result["agent_reports"])
            transcript.append(f"[{role}] {revised.action.value}: {revised.reasoning}")
        return {
            "desk_reports": updates,
            "agent_reports": legacy,
            "debate_round": round_number,
            "debate_transcript": transcript,
        }


def debate_node(state, llm=None):
    return DebateManager.run_debate_round(state, llm)


def debate_router(state):
    config = state.get("app_config") or get_config()
    return "debate" if DebateManager.should_debate(state, config.agents) else "risk_manager"
