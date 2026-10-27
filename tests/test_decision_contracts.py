"""Synthetic invariant fixtures; these do not assert financial/provider accuracy."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.domain.common import EvidenceRecord, Instrument, RunContext
from src.domain.decisions import (
    AgentAction,
    AgentDecision,
    ConfidenceSemantics,
    Criterion,
    DecisionPolicy,
    DecisionQuestion,
    DecisionRequest,
    DecisionSource,
    SpecialistReport,
    digest,
)


@pytest.fixture
def decision_context():
    return RunContext(
        run_id="fixture-run",
        mode="backtest",
        as_of=datetime(2025, 1, 2, tzinfo=timezone.utc),
        config_hash="a" * 64,
        code_version="fixture-v1",
    )


@pytest.fixture
def decision_instrument():
    return Instrument(
        instrument_id="fixture:A", symbol="A", asset_class="equity", quote_currency="USD"
    )


@pytest.fixture
def decision_source(decision_context):
    text = "Management said: Demand fell 10%. Guidance was lowered."
    record = EvidenceRecord(
        evidence_id="fixture-doc",
        provider="test",
        provider_version="1",
        source_uri="fixture://doc",
        raw_hash=digest(text),
        ingested_at=decision_context.as_of,
        available_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        availability_basis="documented",
        quality="usable",
        instrument_ids=("fixture:A",),
    )
    return DecisionSource.from_document(record, text, start=0, end=len(text), kind="transcript")


@pytest.fixture
def decision_question():
    return DecisionQuestion(
        question_id="direction",
        version="1",
        kind="choice",
        instructions="Direction?",
        criteria=(
            Criterion(key="down", description="Decline"),
            Criterion(key="unknown", description="Missing"),
        ),
    )


@pytest.fixture
def decision_request(decision_context, decision_instrument, decision_source, decision_question):
    return DecisionRequest(
        context=decision_context,
        instrument=decision_instrument,
        sources=(decision_source,),
        questions=(decision_question,),
    )


def test_frozen_nested_source_and_question(decision_request):
    with pytest.raises(ValidationError):
        decision_request.sources[0].record.raw_hash = "b" * 64
    with pytest.raises(ValidationError):
        decision_request.questions[0].criteria[0].description = "different"


@pytest.mark.parametrize("model", ["jev-latest", "jev-preview", "jev-1.14", ""])
def test_reject_model_alias(model):
    with pytest.raises(ValidationError):
        DecisionPolicy(model=model)


def test_source_offsets_and_hash_rejected(decision_source):
    data = decision_source.model_dump()
    data["text"] += " changed"
    with pytest.raises(ValidationError):
        DecisionSource.model_validate(data)


@pytest.mark.parametrize(
    "change", ["state", "question", "ordering", "policy", "model", "asof", "source"]
)
def test_semantic_changes_invalidate_cache(decision_request, change):
    data = decision_request.model_dump(mode="json")
    if change == "state":
        data["state_json"] = '{"period":"different"}'
    if change == "question":
        data["questions"][0]["instructions"] += " revised"
    if change == "ordering":
        data["questions"][0]["criteria"].reverse()
    if change == "policy":
        data["policy"]["version"] = "2"
    if change == "model":
        data["policy"]["model"] = "jev-1.13.0"
    if change == "asof":
        data["context"]["as_of"] = "2025-01-03T00:00:00Z"
    if change == "source":
        data["sources"][0]["record"]["revision_id"] = "new-revision"
    assert DecisionRequest.model_validate(data).cache_key() != decision_request.cache_key()


def test_run_id_is_lineage_not_semantic_identity(decision_request):
    data = decision_request.model_dump()
    data["context"]["run_id"] = "other-run"
    other = DecisionRequest.model_validate(data)
    assert other.cache_key() == decision_request.cache_key()
    assert other.content_hash() != decision_request.content_hash()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_confidence_is_finite_and_bounded(value):
    with pytest.raises(ValidationError):
        ConfidenceSemantics(analyst_conviction=value)


def test_missing_report_cannot_be_hold():
    with pytest.raises(ValidationError):
        SpecialistReport(
            role="fundamental", action="HOLD", status="unavailable", reasoning="Missing"
        )


@pytest.mark.parametrize("action", [AgentAction.BUY, AgentAction.EXIT, AgentAction.HOLD])
def test_veto_cannot_clear_or_liquidate(decision_context, decision_instrument, action):
    with pytest.raises(ValidationError):
        AgentDecision(
            context=decision_context,
            instrument=decision_instrument,
            action=action,
            risk_veto=True,
            reasoning="veto",
            new_exposure_blocked=True,
        )


def test_activated_policy_requires_evaluation():
    with pytest.raises(ValidationError):
        DecisionPolicy(activation="evaluated")


def test_state_nan_rejected(decision_request):
    with pytest.raises(ValidationError):
        DecisionRequest(**{**decision_request.model_dump(), "state_json": '{"bad":NaN}'})
