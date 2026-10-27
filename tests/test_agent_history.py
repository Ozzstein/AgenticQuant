"""Prospective history proofs use actual stores; providers are offline fixtures."""

# ruff: noqa: F811
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agents.consensus import consensus_recommendation
from src.agents.graph import analyze_instrument
from src.agents.history import DeskHistoryStore
from src.agents.memory import DeskMemoryStore
from src.domain.decisions import AgentAction, DecisionSource
from tests.test_agent_desk_invariants import (  # noqa: F401
    GenerativeFixture,
    decision_context,
    decision_instrument,
    decision_source,
    desk_config,
    desk_hmm,
    desk_portfolio,
    desk_sources,
    jev_service,
)


@pytest.fixture
def history_env(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
):
    EvidenceStore = pytest.importorskip("src.data.evidence.store").EvidenceStore
    sources = tuple(
        DecisionSource.from_document(
            s.record.model_copy(update={"requested_as_of": decision_context.as_of}),
            s.text,
            start=0,
            end=len(s.text),
            kind=s.kind,
        )
        for s in desk_sources
    )
    store = EvidenceStore(tmp_path / "sources.sqlite")
    for source in sources:
        store.put(
            source.record,
            source.text.encode(),
            kind=source.kind,
            logical_key=source.record.evidence_id,
            document=source.text,
        )
    service = jev_service(tmp_path)
    decision = analyze_instrument(
        decision_instrument,
        decision_context,
        sources=sources,
        hmm_context=desk_hmm,
        portfolio_context=desk_portfolio,
        config=desk_config,
        decision_service=service,
        llm_factory=lambda role: GenerativeFixture(role, []),
    )
    history = DeskHistoryStore(Path(desk_config.output_dir) / "desk-history.sqlite3")
    options = dict(
        evidence_store=store,
        decision_store=service.store,
        budget_store_path=desk_config.agents.budget_store_path,
    )
    return decision, history, options


def test_full_proof_exact_sources_calls_jev_and_consensus(history_env):
    decision, history, options = history_env
    assert decision.action == AgentAction.BUY
    receipt = history.export(decision.content_hash(), as_of=datetime.now(timezone.utc), **options)
    assert receipt.decision == decision
    assert (
        len(receipt.calls) == len(receipt.generations) == 17
    )  # Five roles, two challenges, CRO, strategist.
    assert receipt.started_at <= receipt.completed_at <= receipt.available_at
    assert receipt.available_at > decision.context.as_of
    assert all(c.actual_model == c.model == "fixture-chat" for c in receipt.calls)
    assert all(
        json.loads(c.admission_json)["usage_at_admission"]["run_id"]["requests"] > 0
        for c in receipt.calls
    )
    observations = json.loads(receipt.jev_observations_json)
    assert observations and any(o["reservations"] for o in observations)
    assert consensus_recommendation(receipt.decision).action == AgentAction.BUY
    assert all(g.request_json and g.source_hashes for g in receipt.generations)


def test_oldtext_generation_never_backdated_to_signal(history_env):
    decision, history, options = history_env
    with pytest.raises(ValueError, match="after signal_at"):
        history.export(decision.content_hash(), as_of=decision.context.as_of, **options)
    receipt = history.export(decision.content_hash(), as_of=datetime.now(timezone.utc), **options)
    with pytest.raises(ValueError, match="after signal_at"):
        history.export(
            decision.content_hash(),
            as_of=receipt.available_at - timedelta(microseconds=1),
            **options,
        )


def test_legacy_memory_cannot_certify_supplied_decision(history_env, tmp_path):
    decision, _, options = history_env
    DeskMemoryStore(tmp_path / "old.sqlite").save(decision, available_at=decision.context.as_of)
    history = DeskHistoryStore(tmp_path / "old.sqlite")
    with pytest.raises(ValueError, match="no authoritative generation"):
        history.export(decision.content_hash(), as_of=datetime.now(timezone.utc), **options)


@pytest.mark.parametrize("target", ["budget", "generation", "source", "jev", "history"])
def test_export_rejects_missing_or_corrupt_authority(history_env, tmp_path, target):
    decision, history, options = history_env
    if target == "budget":
        with sqlite3.connect(options["budget_store_path"]) as db:
            db.execute("UPDATE calls SET actual_model='forged'")
    elif target == "generation":
        with sqlite3.connect(options["budget_store_path"]) as db:
            db.execute("DROP TRIGGER agent_generations_update")
            db.execute("UPDATE agent_generations SET payload='{}'")
    elif target == "source":
        options["evidence_store"] = type(options["evidence_store"])(
            tmp_path / "empty-source.sqlite"
        )
    elif target == "jev":
        options["decision_store"] = type(options["decision_store"])(tmp_path / "empty-jev.sqlite")
    else:
        with sqlite3.connect(history.path) as db:
            db.execute("DROP TRIGGER certified_desk_update")
            db.execute("UPDATE certified_desk_history SET receipt_hash=?", ("0" * 64,))
    with pytest.raises((ValueError, KeyError)):
        history.export(decision.content_hash(), as_of=datetime.now(timezone.utc), **options)


def test_receipts_reject_mutation_and_wrong_cutoff(history_env):
    decision, history, options = history_env
    for path, table in (
        (history.path, "certified_desk_history"),
        (options["budget_store_path"], "agent_admissions"),
        (options["budget_store_path"], "agent_settlements"),
        (options["budget_store_path"], "agent_generations"),
        (options["decision_store"].path, "reservations"),
    ):
        with sqlite3.connect(path) as db, pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(f"DELETE FROM {table}")
    with pytest.raises(ValueError, match="timezone"):
        history.export(decision.content_hash(), as_of=datetime(2026, 1, 1), **options)
