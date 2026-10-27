"""Complete graph invariants with synthetic provider fixtures, no runtime stand-ins."""

# ruff: noqa: F811
import json
import sqlite3
import time
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from src.agents.budget import AgentBudgetError, AgentCallBudget
from src.agents.contracts import HMMContext, PortfolioReviewContext, legacy_confidence
from src.agents.graph import analyze_instrument, analyze_ticker, build_trading_desk_graph
from src.agents.memory import DeskMemoryStore
from src.decision.service import DecisionService, OperatingLimits
from src.decision.store import DecisionStore
from src.decision.transport import JevHTTPTransport
from src.decision.workflows import EvidenceJudgments
from src.domain.decisions import (
    AgentAction,
    AgentDecision,
    CitedClaim,
    ConfidenceSemantics,
    DecisionSource,
    SpecialistReport,
)
from src.utils.config import AppConfig
from tests.test_decision_contracts import (  # noqa: F401
    decision_context,
    decision_instrument,
    decision_question,
    decision_request,
    decision_source,
)


@pytest.fixture
def desk_config(tmp_path, decision_context):
    cfg = AppConfig(output_dir=str(tmp_path / "outputs"), log_dir=str(tmp_path / "logs"))
    prices = [
        SimpleNamespace(
            provider=cfg.agents.llm_provider,
            model=model,
            price_version="synthetic-test-rate",
            source_uri="fixture://price",
            effective_at=decision_context.as_of,
            input_usd_per_million=Decimal("0.01"),
            output_usd_per_million=Decimal("0.02"),
        )
        for model in (cfg.agents.llm_model, cfg.agents.deep_think_model)
    ]
    # Matches coordinator-owned shared config without mutating its launch schema.
    cfg.agents = cfg.agents.model_copy(
        update={
            "pricing": prices,
            "budget_store_path": str(tmp_path / "chat.sqlite"),
            "max_tokens_per_analysis": 1_000_000,
            "max_tokens_per_day": 2_000_000,
        }
    )
    cfg.memory.enabled = False
    return cfg


@pytest.fixture
def desk_sources(decision_source):
    return tuple(
        DecisionSource.from_document(
            decision_source.record.model_copy(update={"evidence_id": "fixture-" + kind}),
            decision_source.text,
            start=0,
            end=len(decision_source.text),
            kind=kind,
        )
        for kind in ("fundamental", "technical", "news", "transcript", "macro")
    )


@pytest.fixture
def desk_hmm(decision_context):
    return HMMContext(
        regime="risk_on",
        state_probability=0.8,
        model_version="fixture-hmm-v1",
        as_of=decision_context.as_of,
    )


@pytest.fixture
def desk_portfolio(decision_context):
    return PortfolioReviewContext(
        as_of=decision_context.as_of,
        snapshot_id="fixture-portfolio",
        portfolio_json='{"cash":900,"nav":1000,"positions":{"fixture:A":100}}',
        correlation_json='{"fixture:A":{"fixture:A":1}}',
    )


class GenerativeFixture:
    def __init__(self, role, calls, *, final="BUY", fail_role=None, bad_quote=False):
        self.role, self.calls, self.final, self.fail_role, self.bad_quote = (
            role,
            calls,
            final,
            fail_role,
            bad_quote,
        )

    def with_structured_output(self, schema, include_raw=False):
        assert schema is SpecialistReport and include_raw
        return self

    def invoke(self, messages):
        self.calls.append((self.role, messages))
        if self.role == self.fail_role:
            raise ConnectionError("fixture unavailable")
        payload = json.loads(messages[-1].content)
        source = payload["sources"][0]
        action = (
            self.final
            if self.role == "strategist"
            else ("HOLD" if self.role == "risk_manager" else "BUY")
        )
        report = SpecialistReport(
            role=self.role,
            action=action,
            status="ok",
            reasoning="Fixture reasoning " + self.role + ": demand evidence and risk addressed.",
            confidence=ConfidenceSemantics(analyst_conviction=0.6, evidence_completeness=0.5),
            claims=(
                CitedClaim(
                    claim="Demand declined",
                    evidence_id=source["evidence_id"],
                    passage_hash=source["passage_hash"],
                    quote="invented quote" if self.bad_quote else "Demand fell 10%.",
                ),
            ),
        )
        return {
            "parsed": report,
            "parsing_error": None,
            "raw": SimpleNamespace(
                usage_metadata={"input_tokens": 100, "output_tokens": 100},
                response_metadata={"model": "fixture-chat"},
            ),
        }


def jev_service(tmp_path):
    def handler(req):
        payload = json.loads(req.content)
        answers = {}
        for key, q in payload["questions"].items():
            criteria = q["criteria"]
            choice = {
                "claim_support": "supported",
                "debate_discrepancy": "consistent",
                "relevance": "direct",
                "conflict": "conflict",
            }.get(key, next(iter(criteria)))
            answers[key] = {
                "type": "choice",
                "choice": choice,
                "confidence": 1.0,
                "probabilities": {k: float(k == choice) for k in criteria},
            }
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 100},
            },
        )

    return DecisionService(
        DecisionStore(tmp_path / "jev.sqlite"),
        transport=JevHTTPTransport("fixture", transport=httpx.MockTransport(handler)),
        limits=OperatingLimits(min_interval_seconds=0, run_spend_usd=1, run_tokens=10_000_000),
    )


def run_desk(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
    *,
    final="BUY",
    fail_role=None,
    bad_quote=False,
):
    calls = []
    result = analyze_instrument(
        decision_instrument,
        decision_context,
        sources=desk_sources,
        hmm_context=desk_hmm,
        portfolio_context=desk_portfolio,
        config=desk_config,
        decision_service=jev_service(tmp_path),
        llm_factory=lambda role: GenerativeFixture(
            role, calls, final=final, fail_role=fail_role, bad_quote=bad_quote
        ),
    )
    return result, calls


def test_full_graph_all_roles_two_rounds_and_transcript(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
):
    result, calls = run_desk(
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        desk_sources,
        desk_hmm,
        desk_portfolio,
    )
    assert result.action == AgentAction.BUY, result
    assert {r.role for r in result.reports} == {
        "fundamental",
        "sentiment",
        "technical",
        "earnings",
        "transcript_nlp",
        "risk_manager",
        "strategist",
    }
    assert len(result.debate_transcript) == 12  # 5 challenges + header in each round
    assert len(calls) == 17
    assert result.judgment_observation_ids
    assert all(
        json.loads(messages[-1].content)["accepted_task_judgments"] == [] for _, messages in calls
    )
    assert result.reports[-1].derived_from  # strategist is never an independent vote
    with sqlite3.connect(desk_config.agents.budget_store_path) as db:
        assert db.execute('select count(*) from calls where status="ok"').fetchone()[0] == 17


@pytest.mark.parametrize("action", ["HOLD", "REDUCE", "EXIT", "ABSTAIN"])
def test_explicit_actions_survive_boundary(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
    action,
):
    result, _ = run_desk(
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        desk_sources,
        desk_hmm,
        desk_portfolio,
        final=action,
    )
    assert result.action.value == action


def test_veto_cannot_be_cleared_or_changed_into_exit(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
):
    veto = desk_portfolio.model_copy(update={"deterministic_vetoes": ("concentration breach",)})
    result, _ = run_desk(
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        desk_sources,
        desk_hmm,
        veto,
        final="EXIT",
    )
    assert result.action == AgentAction.VETO and result.risk_veto and result.new_exposure_blocked


@pytest.mark.parametrize(
    "fail_role", ["fundamental", "sentiment", "technical", "risk_manager", "strategist"]
)
def test_partial_role_or_provider_failure_cannot_approve(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
    fail_role,
):
    result, _ = run_desk(
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        desk_sources,
        desk_hmm,
        desk_portfolio,
        fail_role=fail_role,
    )
    assert result.action == AgentAction.ABSTAIN and result.new_exposure_blocked


def test_hmm_missing_is_not_neutral(
    tmp_path, desk_config, decision_context, decision_instrument, desk_sources, desk_portfolio
):
    result, _ = run_desk(
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        desk_sources,
        None,
        desk_portfolio,
    )
    assert result.action == AgentAction.ABSTAIN


def test_exact_quotes_checked_before_semantic_provider(
    tmp_path, decision_context, decision_instrument, decision_source
):
    class Forbidden:
        def evaluate(self, *args, **kwargs):
            pytest.fail("Invalid quote reached semantic provider")

    workflow = EvidenceJudgments(Forbidden(), decision_context, decision_instrument)
    claim = CitedClaim(
        claim="Guidance increased",
        evidence_id=decision_source.record.evidence_id,
        passage_hash=decision_source.passage_hash,
        quote="Guidance was raised.",
    )
    assert workflow.verify(claim, (decision_source,)).status == "quote_missing"
    wrong = claim.model_copy(update={"evidence_id": "nonexistent"})
    assert workflow.verify(wrong, (decision_source,)).status == "invalid_reference"


def test_invalid_citations_block_graph_buy(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
):
    result, _ = run_desk(
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        desk_sources,
        desk_hmm,
        desk_portfolio,
        bad_quote=True,
    )
    assert result.action == AgentAction.ABSTAIN
    assert any("quote_missing" in flag for flag in result.risk_flags)


def test_retrieval_keeps_material_conflicts(
    tmp_path, decision_context, decision_instrument, desk_sources
):
    workflow = EvidenceJudgments(jev_service(tmp_path), decision_context, decision_instrument)
    scored = workflow.retrieve(desk_sources, query="Demand", reference="Demand increased")
    assert len(scored) == len(desk_sources)
    assert all(r.judgments[1].choice == "conflict" for _, r in scored)


def test_legacy_bridge_does_not_rescale():
    migrated = legacy_confidence(73)
    assert migrated.legacy_confidence_0_100 == 73
    assert migrated.analyst_conviction is None and migrated.calibrated_task_accuracy is None


def test_legacy_ticker_only_call_rejected():
    with pytest.raises(ValueError, match="canonical instrument"):
        analyze_ticker("A")


def test_graph_nodes_are_complete(desk_config, tmp_path):
    graph = build_trading_desk_graph(
        desk_config, decision_service=jev_service(tmp_path)
    ).get_graph()
    assert {"transcript_nlp", "verify_evidence", "debate", "risk_manager", "strategist"} <= set(
        graph.nodes
    )


def test_history_observation_time_prevents_historical_leak(
    tmp_path, decision_context, decision_instrument
):
    store = DeskMemoryStore(tmp_path / "memory.sqlite")
    decision = AgentDecision(
        instrument=decision_instrument,
        context=decision_context,
        action="HOLD",
        reasoning="Fixture historical decision",
        new_exposure_blocked=False,
    )
    store.save(decision, available_at=decision_context.as_of + timedelta(days=1))
    assert store.retrieve(decision_instrument.instrument_id, as_of=decision_context.as_of) == []
    assert (
        len(
            store.retrieve(
                decision_instrument.instrument_id, as_of=decision_context.as_of + timedelta(days=2)
            )
        )
        == 1
    )


def test_unknown_price_blocks_chat_and_budget_survives_restart(desk_config, decision_context):
    cfg = desk_config.agents
    budget = AgentCallBudget(cfg, decision_context.run_id, deadline=time.monotonic() + 10)
    from langchain_core.messages import HumanMessage

    with pytest.raises(AgentBudgetError, match="pricing unavailable"):
        budget.reserve("unpriced-model", [HumanMessage("hello")], 100)
    capped = cfg.model_copy(update={"max_requests_per_analysis": 1})
    one = AgentCallBudget(capped, "new-run", deadline=time.monotonic() + 10)
    one.reserve(cfg.llm_model, [HumanMessage("hello")], 100)
    restarted = AgentCallBudget(capped, "new-run", deadline=time.monotonic() + 10)
    with pytest.raises(AgentBudgetError, match="budget exhausted"):
        restarted.reserve(cfg.llm_model, [HumanMessage("hello")], 100)


@pytest.fixture
def desk_state(
    desk_config, decision_context, decision_instrument, desk_sources, desk_hmm, desk_portfolio
):
    return {
        "app_config": desk_config,
        "context": decision_context,
        "instrument": decision_instrument,
        "ticker": decision_instrument.symbol,
        "asset_class": decision_instrument.asset_class.upper(),
        "sources": desk_sources,
        "hmm_context": desk_hmm,
        "portfolio_context": desk_portfolio,
        "macro_regime": desk_hmm.regime,
        "macro_confidence": desk_hmm.state_probability,
        "memory_context": "",
        "qlib_context": "",
        "decision_results": [],
        "desk_reports": {},
        "agent_reports": {},
        "debate_round": 0,
        "debate_transcript": [],
        "risk_veto": False,
        "risk_veto_reason": "",
        "new_exposure_blocked": False,
        "call_budget": AgentCallBudget(
            desk_config.agents, decision_context.run_id, deadline=time.monotonic() + 30
        ),
    }


@pytest.mark.parametrize("proposal", ["EXIT", "REDUCE", "HOLD"])
def test_failed_required_analysis_cannot_turn_into_liquidation_or_neutral(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
    proposal,
):
    result, _ = run_desk(
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        desk_sources,
        desk_hmm,
        desk_portfolio,
        fail_role="technical",
        final=proposal,
    )
    assert result.action == AgentAction.ABSTAIN


def test_debate_cannot_replace_missing_technical_sources_with_news(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
):
    sources = tuple(s for s in desk_sources if s.kind != "technical")
    result, calls = run_desk(
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        sources,
        desk_hmm,
        desk_portfolio,
    )
    assert result.action == AgentAction.ABSTAIN
    assert not any(role == "technical" for role, _ in calls)


def test_configured_role_disable_is_visible_abstention(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
):
    desk_config.agents.technical_enabled = False
    result, calls = run_desk(
        tmp_path,
        desk_config,
        decision_context,
        decision_instrument,
        desk_sources,
        desk_hmm,
        desk_portfolio,
    )
    assert result.action == AgentAction.ABSTAIN
    technical = next(r for r in result.reports if r.role == "technical")
    assert "disabled" in technical.reasoning and not any(role == "technical" for role, _ in calls)
