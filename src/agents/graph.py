"""Complete evidence-grounded trading desk; no stub nodes or inferred availability."""

from __future__ import annotations

import time
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.agents.budget import AgentCallBudget
from src.agents.debate import debate_node, debate_router
from src.agents.earnings_handler import earnings_node
from src.agents.fundamental import fundamental_node
from src.agents.macro_regime import macro_regime_node, market_context_node
from src.agents.memory import load_memory_node, save_memory_node
from src.agents.risk_manager import risk_manager_node
from src.agents.sentiment import sentiment_node
from src.agents.state import TradingDeskState
from src.agents.strategist import strategist_node
from src.agents.technical import technical_node
from src.agents.transcript_nlp import transcript_nlp_node
from src.decision.service import DecisionService
from src.decision.store import DecisionStore
from src.decision.workflows import EvidenceJudgments
from src.domain.decisions import AgentAction, AgentDecision, DecisionPolicy, DecisionStatus
from src.utils.config import get_config


def build_trading_desk_graph(
    config=None,
    *,
    decision_service=None,
    policy=None,
    llm_factory=None,
    observation_mode="cache",
    evidence_store=None,
):
    config = config or get_config()
    service = decision_service or DecisionService(
        DecisionStore(Path(config.output_dir) / "decisions.sqlite")
    )
    policy = policy or DecisionPolicy()

    def environment(state):
        return {
            **state,
            "app_config": config,
            "llm_factory": llm_factory,
            "workflow": EvidenceJudgments(
                service,
                state["context"],
                state["instrument"],
                policy=policy,
                mode=observation_mode,
                deadline=state.get("decision_deadline"),
            ),
        }

    def bind(node):
        def run(state):
            return node(environment(state))

        return run

    def prepare(state):
        # Explicit input contracts; never fabricate a security identity/current date.
        if state.get("context") is None or state.get("instrument") is None:
            raise ValueError("graph requires common.RunContext and common.Instrument")
        return {
            "decision_deadline": time.monotonic()
            + getattr(config.agents, "analysis_timeout_seconds", 180),
            "call_budget": AgentCallBudget(
                config.agents,
                state["context"].run_id + "/" + state["instrument"].instrument_id,
                deadline=time.monotonic() + getattr(config.agents, "analysis_timeout_seconds", 180),
            ),
            "new_exposure_blocked": bool(state.get("new_exposure_blocked")),
            "desk_reports": {},
            "decision_results": [],
            "debate_round": 0,
            "debate_transcript": [],
            "evidence_issues": (),
            "risk_veto": bool(state.get("risk_veto")),
        }

    def evidence_node(state):
        workflow = environment(state)["workflow"]
        sources = state.get("sources", ())
        results = []
        if state.get("memory_context"):
            from src.decision.questions import CONFLICT, RELEVANCE

            results.append(
                workflow.ask(
                    sources,
                    (RELEVANCE, CONFLICT),
                    {
                        "query": "Relevance of eligible prior analysis to this instrument and current evidence",
                        "prior_analysis": state["memory_context"],
                        "reference": state.get("qlib_context", ""),
                    },
                )
            )
        for source, result in workflow.retrieve(
            sources,
            query="Investment evidence for " + state["instrument"].symbol,
            reference=state.get("qlib_context", ""),
        ):
            results.append(result)
            if source.kind == "news":
                results.append(workflow.news(source))
            elif source.kind == "transcript":
                results.append(workflow.transcript(source))
        # Retrieval never suppresses contradictory or uncertain evidence.
        unavailable = not sources or (
            policy.activation == "evaluated"
            and any(
                r.status != DecisionStatus.OK
                or any(j.disposition != "accepted" for j in r.judgments)
                for r in results
            )
        )
        return {
            "decision_results": results,
            "new_exposure_blocked": bool(state.get("new_exposure_blocked")) or unavailable,
        }

    def verify_node(state):
        workflow = environment(state)["workflow"]
        checks = []
        issues = []
        results = []
        for role, report in state.get("desk_reports", {}).items():
            if report.status == "not_applicable":
                continue
            if report.status != "ok" or not report.claims:
                issues.append(role + ": unavailable or uncited report: " + report.reasoning)
            for claim in report.claims:
                check = workflow.verify(claim, state.get("sources", ()))
                checks.append(check)
                if check.result:
                    results.append(check.result)
                if check.status in ("invalid_reference", "quote_missing", "ineligible"):
                    issues.append(role + ": " + check.status)
                elif policy.activation == "evaluated" and (
                    check.status != "supported"
                    or not check.result
                    or check.result.judgments[0].disposition != "accepted"
                ):
                    issues.append(role + ": unresolved semantic support")
        discrepancy = workflow.debate(
            state.get("sources", ()), state.get("desk_reports", {}).values()
        )
        results.append(discrepancy)
        if policy.activation == "evaluated" and (
            discrepancy.status != DecisionStatus.OK
            or discrepancy.judgments[0].choice != "consistent"
            or discrepancy.judgments[0].disposition != "accepted"
        ):
            issues.append("unresolved debate discrepancy")
        return {
            "citation_checks": tuple(checks),
            "evidence_issues": tuple(issues),
            "decision_results": results,
        }

    def route(state):
        return debate_router(environment(state))

    def risk(state):
        return risk_manager_node(
            environment(
                {
                    **state,
                    "new_exposure_blocked": bool(state.get("new_exposure_blocked"))
                    or bool(state.get("evidence_issues")),
                }
            )
        )

    def persist(state):
        try:
            from src.agents.history import DeskHistoryStore

            DeskHistoryStore(
                Path(config.output_dir) / "desk-history.sqlite3"
            ).record_graph_completion(state, decision_store=service.store)
            return save_memory_node(state, config)
        except Exception as exc:
            # Preserve completed reasoning and qualitative vetoes if persistence
            # fails; the outer failure handler cannot recover this node's state.
            decision = state["agent_decision"]
            veto = decision.risk_veto or decision.action == AgentAction.VETO
            decision = decision.model_copy(
                update={
                    "action": AgentAction.VETO if veto else AgentAction.ABSTAIN,
                    "risk_veto": veto,
                    "new_exposure_blocked": True,
                    "reasoning": decision.reasoning
                    + f"\nPersistence unavailable ({type(exc).__name__})",
                    "risk_flags": decision.risk_flags + ("decision_persistence_unavailable",),
                }
            )
            output = {"agent_decision": decision, "new_exposure_blocked": True, "risk_veto": veto}
            if state.get("result") is not None:
                from src.agents.contracts import legacy_analysis

                output["result"] = legacy_analysis(decision)
            return output

    graph = StateGraph(TradingDeskState)
    nodes = {
        "prepare": prepare,
        "market_context": bind(market_context_node),
        "macro_regime": bind(macro_regime_node),
        "load_memory": lambda state: load_memory_node(
            state, config, evidence_store=evidence_store, decision_store=service.store
        ),
        "evidence": evidence_node,
        "fundamental": bind(fundamental_node),
        "sentiment": bind(sentiment_node),
        "technical": bind(technical_node),
        "earnings": bind(earnings_node),
        "transcript_nlp": bind(transcript_nlp_node),
        "verify_evidence": verify_node,
        "debate": bind(debate_node),
        "risk_manager": risk,
        "strategist": bind(strategist_node),
        "save_memory": persist,
    }
    for name, node in nodes.items():
        graph.add_node(name, node)
    chain = [START, "prepare", "market_context", "macro_regime", "load_memory", "evidence"]
    for left, right in zip(chain, chain[1:]):
        graph.add_edge(left, right)
    specialists = ["fundamental", "sentiment", "technical", "earnings", "transcript_nlp"]
    for specialist in specialists:
        graph.add_edge("evidence", specialist)
    # Explicit join barrier: verification cannot run on a partial-role subset.
    graph.add_edge(specialists, "verify_evidence")
    graph.add_conditional_edges(
        "verify_evidence", route, {"debate": "debate", "risk_manager": "risk_manager"}
    )
    graph.add_edge("debate", "verify_evidence")
    graph.add_edge("risk_manager", "strategist")
    graph.add_edge("strategist", "save_memory")
    graph.add_edge("save_memory", END)
    return graph.compile()


def analyze_instrument(
    instrument,
    context,
    *,
    sources,
    hmm_context=None,
    portfolio_context=None,
    qlib_context="",
    config=None,
    decision_service=None,
    policy=None,
    llm_factory=None,
    observation_mode="cache",
    evidence_store=None,
) -> AgentDecision:
    config = config or get_config()
    graph = build_trading_desk_graph(
        config,
        decision_service=decision_service,
        policy=policy,
        llm_factory=llm_factory,
        observation_mode=observation_mode,
        evidence_store=evidence_store,
    )
    initial = {
        "instrument": instrument,
        "context": context,
        "sources": tuple(sources),
        "hmm_context": hmm_context,
        "portfolio_context": portfolio_context,
        "ticker": instrument.symbol,
        "asset_class": instrument.asset_class.upper(),
        "qlib_context": qlib_context,
        "agent_reports": {},
        "desk_reports": {},
        "risk_veto": False,
    }
    try:
        final = graph.invoke(initial, {"recursion_limit": 30 + config.agents.max_debate_rounds * 3})
        decision = final.get("agent_decision")
        if decision is None:
            raise ValueError("no desk decision")
        return decision
    except Exception as exc:
        # A hard accounting veto remains visible even if graph execution fails.
        vetoes = portfolio_context.deterministic_vetoes if portfolio_context else ()
        veto = bool(vetoes) or bool(hmm_context and hmm_context.regime == "crisis")
        return AgentDecision(
            instrument=instrument,
            context=context,
            action=AgentAction.VETO if veto else AgentAction.ABSTAIN,
            risk_veto=veto,
            new_exposure_blocked=True,
            reasoning=f"Agent desk unavailable ({type(exc).__name__})",
            risk_flags=tuple(vetoes),
        )


def analyze_ticker(ticker, asset_class="EQUITY", qlib_context="", config=None, **kwargs):
    """Explicit legacy bridge; requires instrument/context/evidence supplied by caller."""
    from src.agents.contracts import legacy_analysis

    instrument = kwargs.pop("instrument", None)
    context = kwargs.pop("context", None)
    if instrument is None or context is None:
        raise ValueError(
            "analyze_ticker requires canonical instrument, context and sources; use analyze_instrument"
        )
    if instrument.symbol != ticker or instrument.asset_class.upper() != asset_class.upper():
        raise ValueError("ticker/asset class disagree with canonical instrument")
    return legacy_analysis(
        analyze_instrument(instrument, context, qlib_context=qlib_context, config=config, **kwargs)
    )
