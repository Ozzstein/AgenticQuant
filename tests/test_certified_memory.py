"""Model memory is a projection of real immutable history, not supplied prose."""

# ruff: noqa: F811
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from src.agents.graph import analyze_instrument
from src.agents.history import memory_projection
from src.agents.memory import load_memory_node
from tests.test_agent_history import (  # noqa: F401
    GenerativeFixture,
    decision_context,
    decision_instrument,
    decision_source,
    desk_config,
    desk_hmm,
    desk_portfolio,
    desk_sources,
    history_env,
)


def test_projection_exact_refs_generation_cutoff_and_instrument(history_env, desk_config):
    decision, history, authorities = history_env
    receipt = history.export(
        decision.content_hash(), as_of=datetime.now(timezone.utc), **authorities
    )
    assert (
        history.retrieve(
            decision.instrument.instrument_id, as_of=decision.context.as_of, **authorities
        )
        == ()
    )
    assert history.retrieve("unrelated", as_of=receipt.available_at, **authorities) == ()
    values = history.retrieve(
        decision.instrument.instrument_id, as_of=receipt.available_at, **authorities
    )
    text, refs = memory_projection(values)
    assert refs == ((decision.content_hash(), receipt.content_hash()),)
    assert len(text) <= 2000
    assert json.loads(text)[0]["receipt_hash"] == receipt.content_hash()
    desk_config.memory.enabled = True
    state = {
        "context": decision.context.model_copy(update={"as_of": receipt.available_at}),
        "instrument": decision.instrument,
    }
    loaded = load_memory_node(
        state,
        desk_config,
        evidence_store=authorities["evidence_store"],
        decision_store=authorities["decision_store"],
    )
    assert loaded["memory_context"] == text
    assert loaded["memory_receipts"] == refs
    assert loaded["memory_status"] == "available"
    missing = load_memory_node(state, desk_config)
    assert missing["memory_status"] == "unavailable" and missing["new_exposure_blocked"]
    assert missing["memory_context"] == ""


@pytest.mark.parametrize("tamper", ["instrument", "clock", "receipt_hash", "source", "budget"])
def test_changed_authority_never_becomes_model_memory(history_env, desk_config, tamper):
    decision, history, authorities = history_env
    desk_config.memory.enabled = True
    if tamper in ("instrument", "clock", "receipt_hash"):
        with sqlite3.connect(history.path) as db:
            db.execute("DROP TRIGGER certified_desk_update")
            if tamper == "receipt_hash":
                db.execute("UPDATE certified_desk_history SET receipt_hash=?", ("0" * 64,))
            else:
                payload = json.loads(
                    db.execute("SELECT payload FROM certified_desk_history").fetchone()[0]
                )
                if tamper == "instrument":
                    payload["decision"]["instrument"]["instrument_id"] = "wrong"
                else:
                    payload["available_at"] = "2999-01-01T00:00:00Z"
                db.execute("UPDATE certified_desk_history SET payload=?", (json.dumps(payload),))
    elif tamper == "budget":
        authorities["budget_store_path"] = str(history.path.parent / "absent-budget.sqlite")
        desk_config.agents.budget_store_path = authorities["budget_store_path"]
    else:
        # Actual absent source authority rather than a fake verifier.
        from src.data.evidence.store import EvidenceStore

        authorities["evidence_store"] = EvidenceStore(history.path.parent / "empty.sqlite")
    state = {
        "context": decision.context.model_copy(update={"as_of": datetime.now(timezone.utc)}),
        "instrument": decision.instrument,
    }
    loaded = load_memory_node(
        state,
        desk_config,
        evidence_store=authorities["evidence_store"],
        decision_store=authorities["decision_store"],
    )
    assert loaded["memory_context"] == "" and loaded["memory_status"] == "unavailable"
    assert loaded["new_exposure_blocked"]


def test_full_graph_projects_hashes_and_recursively_reverifies_prior_history(
    history_env, desk_config, desk_hmm, desk_portfolio
):
    decision, history, authorities = history_env
    receipt = history.export(
        decision.content_hash(), as_of=datetime.now(timezone.utc), **authorities
    )
    context = decision.context.model_copy(
        update={"as_of": datetime.now(timezone.utc), "run_id": "with-certified-memory"}
    )
    desk_config.memory.enabled = True
    desk_config.memory.store_path = history.path.parent / "display.jsonl"
    # Reuse the real transport fixture, while preserving authoritative observations.
    from tests.test_agent_desk_invariants import jev_service

    service = jev_service(history.path.parent)
    service.store = authorities["decision_store"]
    calls = []
    outcome = analyze_instrument(
        decision.instrument,
        context,
        sources=receipt.sources,
        hmm_context=desk_hmm.model_copy(update={"as_of": context.as_of}),
        portfolio_context=desk_portfolio.model_copy(update={"as_of": context.as_of}),
        config=desk_config,
        evidence_store=authorities["evidence_store"],
        decision_service=service,
        llm_factory=lambda role: GenerativeFixture(role, calls),
    )
    assert outcome.action.value == "BUY", outcome.model_dump_json()
    text, refs = memory_projection((receipt,))
    assert all(json.loads(messages[-1].content)["memory"] == text for _, messages in calls)
    exported = history.export(
        outcome.content_hash(), as_of=datetime.now(timezone.utc), **authorities
    )
    assert tuple(map(tuple, json.loads(exported.inputs_json)["memory_receipts"])) == refs
    assert json.loads(exported.inputs_json)["memory_context"] == text
    # An ancestor cannot disappear or be substituted on later replay.
    with sqlite3.connect(history.path) as db:
        db.execute("DROP TRIGGER certified_desk_delete")
        db.execute(
            "DELETE FROM certified_desk_history WHERE decision_hash=?", (decision.content_hash(),)
        )
    with pytest.raises(ValueError, match="no authoritative"):
        history.export(outcome.content_hash(), as_of=datetime.now(timezone.utc), **authorities)
