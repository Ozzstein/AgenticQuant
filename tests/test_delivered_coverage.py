"""Actual source stores and span budgets; synthetic documents are test-only."""

# ruff: noqa: F811
from datetime import timedelta
from types import SimpleNamespace

import pytest

from src.data.evidence.store import EvidenceStore
from src.decision.evidence import allocate_desk_sources, validate_delivered_coverage
from src.domain.decisions import digest
from src.operations.cycle import PaperCycleService
from tests.test_agent_desk_invariants import (  # noqa: F401
    GenerativeFixture,
    desk_config,
    desk_hmm,
    desk_portfolio,
    jev_service,
)
from tests.test_decision_contracts import decision_context, decision_instrument, decision_source  # noqa:F401


@pytest.fixture
def source_env(tmp_path, decision_context, decision_source, decision_instrument):
    store = EvidenceStore(tmp_path / "evidence.sqlite")

    def put(identity, kind, text, *, available=None, document=True, logical=None):
        record = decision_source.record.model_copy(
            update={
                "evidence_id": identity,
                "raw_hash": digest(text),
                "available_at": available or decision_context.as_of,
                "requested_as_of": decision_context.as_of,
            }
        )
        store.put(
            record,
            text.encode(),
            kind=kind,
            logical_key=logical or identity,
            document=text if document else None,
        )
        return identity

    return SimpleNamespace(
        store=store, put=put, context=decision_context, instrument=decision_instrument
    )


def serial(coverage):
    return {
        r: {k: v.model_dump(mode="json") for k, v in kinds.items()} for r, kinds in coverage.items()
    }


def test_long_document_cannot_starve_roles_and_partial_is_not_complete(source_env):
    x = source_env
    x.put("a-fund", "income_statement", "Long fundamental source. " * 1000)
    x.put("z-news", "news", "Material contrary news.")
    x.put("z-tech", "technical", "Actual technical evidence.")
    sources, coverage = allocate_desk_sources(x.store, x.instrument, x.context, max_chars=2400)
    assert {s.record.evidence_id for s in sources} == {"a-fund", "z-news", "z-tech"}
    assert sum(len(s.text) for s in sources) <= 2400
    assert coverage["fundamental"]["income_statement"].status == "partial"
    assert not coverage["fundamental"]["income_statement"].delivery_complete
    assert coverage["sentiment"]["news"].status == "available"
    validate_delivered_coverage(serial(coverage), sources, store=x.store, context=x.context)
    plan = SimpleNamespace(
        inputs=SimpleNamespace(context=x.context),
        desk_evidence_ids=("a-fund", "z-news", "z-tech"),
        max_source_chars=2400,
    )
    actual = PaperCycleService._desk_sources(SimpleNamespace(evidence=x.store), plan, x.instrument)
    assert actual == (sources, serial(coverage))


def test_actual_omissions_and_exclusions_are_exact_not_available(source_env):
    x = source_env
    x.put("fund", "filing", "Full filing.")
    x.put("news", "news", "Actual news.")
    x.put("future", "news", "Future news.", available=x.context.as_of + timedelta(days=1))
    x.put("outside", "news", "Excluded by frozen whitelist.")
    x.put("no-document", "technical", "Raw only.", document=False)
    sources, coverage = allocate_desk_sources(
        x.store,
        x.instrument,
        x.context,
        allowed_evidence_ids=("fund", "news", "future", "no-document"),
        max_passages=1,
    )
    news = coverage["sentiment"]["news"]
    assert news.status == "missing" and news.omitted_ids == ("news",) and not news.evidence_ids
    assert ("outside", "outside_frozen_whitelist") in news.excluded
    assert any(identity == "future" for identity, _ in news.excluded)
    technical = coverage["technical"]["technical"]
    assert technical.omitted_ids == ("no-document",)
    assert ("no-document", "complete_document_unavailable") in technical.excluded
    validate_delivered_coverage(serial(coverage), sources, store=x.store, context=x.context)


def test_contradictory_documents_survive_and_budget_truncation_is_visible(source_env):
    x = source_env
    x.put("bull", "news", "Revenue rose.", logical="event")
    x.put("bear", "news", "Revenue fell.", logical="event")
    sources, coverage = allocate_desk_sources(
        x.store, x.instrument, x.context, max_chars=8, max_passages=2
    )
    c = coverage["sentiment"]["news"]
    assert c.status == "conflicting" and c.store_status == "conflicting"
    assert c.evidence_ids == ("bear", "bull") and c.partial_ids == ("bear", "bull")
    assert not c.delivery_complete and sum(len(s.text) for s in sources) == 8
    validate_delivered_coverage(serial(coverage), sources, store=x.store, context=x.context)


def test_source_selection_determinism_and_exact_unicode_spans(source_env):
    x = source_env
    for identity in ("z", "a", "m"):
        x.put(identity, "transcript", "€😀 demanding Unicode " + identity)
    first = allocate_desk_sources(
        x.store,
        x.instrument,
        x.context,
        allowed_evidence_ids=("z", "m", "a"),
        max_chars=20,
        max_passages=3,
    )
    second = allocate_desk_sources(
        x.store,
        x.instrument,
        x.context,
        allowed_evidence_ids=("a", "z", "m"),
        max_chars=20,
        max_passages=3,
    )
    assert first == second and sum(len(s.text) for s in first[0]) == 20
    for source in first[0]:
        assert (
            x.store.document(source.record.evidence_id, as_of=x.context.as_of).text[
                source.start : source.end
            ]
            == source.text
        )


@pytest.mark.parametrize("change", ["complete", "omit", "span", "clock", "kind"])
def test_supplied_coverage_cannot_substitute_actual_delivery(source_env, change):
    x = source_env
    x.put("source", "news", "Long source evidence. " * 100)
    sources, coverage = allocate_desk_sources(x.store, x.instrument, x.context, max_chars=10)
    data = serial(coverage)
    value = data["sentiment"]["news"]
    if change == "complete":
        value["delivery_complete"] = True
        value["status"] = "available"
        value["partial_ids"] = []
    elif change == "omit":
        value["evidence_ids"] = []
        value["omitted_ids"] = ["source"]
    elif change == "span":
        value["delivered_spans"][0]["end"] += 1
    elif change == "clock":
        value["as_of"] = (x.context.as_of + timedelta(days=1)).isoformat()
    else:
        data["sentiment"] = {"filing": value}
    with pytest.raises(ValueError):
        validate_delivered_coverage(data, sources, store=x.store, context=x.context)


def test_graph_receives_verified_coverage_partial_confidence_and_omission_block(
    source_env, desk_config, desk_hmm, desk_portfolio
):
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from src.agents.graph import analyze_instrument
    from src.agents.history import DeskHistoryStore

    x = source_env
    for identity, kind, text in (
        ("fund", "income_statement", "Demand fell 10%. " * 100),
        ("news", "news", "Demand fell 10%."),
        ("technical", "technical", "Demand fell 10%."),
        ("transcript", "transcript", "Demand fell 10%."),
    ):
        x.put(identity, kind, text)
    for limit in (32, 3):
        sources, coverage = allocate_desk_sources(
            x.store, x.instrument, x.context, max_chars=500, max_passages=limit
        )
        calls = []
        context = x.context.model_copy(update={"run_id": "delivery-" + str(limit)})
        service = jev_service(x.store.path.parent / str(limit))
        result = analyze_instrument(
            x.instrument,
            context,
            sources=sources,
            source_coverage=serial(coverage),
            evidence_store=x.store,
            hmm_context=desk_hmm,
            portfolio_context=desk_portfolio,
            config=desk_config,
            decision_service=service,
            llm_factory=lambda role: GenerativeFixture(role, calls),
        )
        assert calls
        assert all(
            json.loads(messages[-1].content)["source_coverage"] == serial(coverage)
            for _, messages in calls
        )
        assert result.action.value == ("BUY" if limit == 32 else "ABSTAIN")
        assert all(r.confidence.evidence_completeness is None for r in result.reports)
        receipt = DeskHistoryStore(Path(desk_config.output_dir) / "desk-history.sqlite3").export(
            result.content_hash(),
            as_of=datetime.now(timezone.utc),
            evidence_store=x.store,
            decision_store=service.store,
            budget_store_path=desk_config.agents.budget_store_path,
        )
        assert json.loads(receipt.inputs_json)["source_coverage"] == serial(coverage)
