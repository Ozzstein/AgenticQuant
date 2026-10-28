"""Evidence, research feature, evaluation and executable-entry contracts."""
# ruff: noqa: F401, F811

import json
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from src.agents.consensus import consensus_recommendation
from src.decision.__main__ import main
from src.decision.evaluation import EvaluationPlan, LabeledPrediction, evaluate_classification
from src.decision.evidence import load_desk_sources, source_from_passage
from src.decision.features import demand_weakness_feature
from src.decision.service import DecisionService
from src.decision.transport import JevHTTPTransport
from src.decision.workflows import EvidenceJudgments
from src.domain.decisions import (
    AgentDecision,
    DecisionRequest,
    ProbabilityValue,
    SpecialistReport,
)
from tests.test_decision_contracts import (
    decision_context,
    decision_instrument,
    decision_question,
    decision_request,
    decision_source,
)
from tests.test_decision_service import service_for


def test_exact_document_adapter(decision_source, decision_context):
    source = decision_source
    document = SimpleNamespace(
        evidence=source.record, text=source.text, document_hash=source.document_hash
    )

    class Store:
        def document(self, evidence_id, *, as_of):
            assert evidence_id == source.record.evidence_id and as_of == decision_context.as_of
            return document

    passage = SimpleNamespace(
        evidence=source.record,
        text=source.text,
        document_hash=source.document_hash,
        start=0,
        end=len(source.text),
    )
    assert (
        source_from_passage(Store(), passage, as_of=decision_context.as_of, kind="transcript")
        == source
    )
    passage.text = passage.text.replace("10%", "20%")
    with pytest.raises(ValueError, match="immutable"):
        source_from_passage(Store(), passage, as_of=decision_context.as_of, kind="transcript")


def test_role_bundle_adapter_deduplicates_and_preserves_coverage(
    tmp_path, decision_source, decision_context, decision_instrument
):
    from src.data.evidence.service import EvidenceService
    from src.data.evidence.store import EvidenceStore
    from src.domain.decisions import digest

    store = EvidenceStore(tmp_path / "source.sqlite")
    for index, text in enumerate((decision_source.text, "Contrary transcript source.")):
        record = decision_source.record.model_copy(
            update={
                "evidence_id": "transcript-" + str(index),
                "raw_hash": digest(text),
                "requested_as_of": decision_context.as_of,
            }
        )
        store.put(record, text.encode(), kind="transcript", logical_key="same-event", document=text)
    sources, coverage = load_desk_sources(
        EvidenceService(store), decision_instrument, decision_context
    )
    assert len(sources) == 2 and all(s.kind == "transcript" for s in sources)
    assert set(coverage) == {"fundamental", "sentiment", "technical", "risk", "strategist"}
    assert coverage["sentiment"]["transcript"].status == "conflicting"
    assert coverage["strategist"]["transcript"].status == "conflicting"


@pytest.mark.parametrize("available", [True, False])
def test_text_feature_missing_is_not_zero(
    tmp_path, decision_context, decision_instrument, decision_source, available
):
    seen = []

    def handler(request):
        data = json.loads(request.content)
        question = next(iter(data["questions"]))
        seen.append(question)
        if question == "demand_available":
            label = "present" if available else "unknown"
            answer = {
                "type": "choice",
                "choice": label,
                "confidence": 1.0,
                "probabilities": {"present": float(available), "unknown": float(not available)},
            }
        else:
            answer = {
                "type": "score",
                "score": 2.0,
                "confidence": 1.0,
                "legend": {
                    str(i): s for i, s in enumerate(data["questions"][question]["criteria"])
                },
                "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0},
            }
        return httpx.Response(
            200,
            json={
                "model": data["model"],
                "answers": {question: answer},
                "usage": {"input_tokens": 10, "output_tokens": 10},
            },
        )

    workflow = EvidenceJudgments(
        service_for(tmp_path, handler), decision_context, decision_instrument
    )
    feature = demand_weakness_feature(workflow, (decision_source,))
    assert feature.value == (pytest.approx(2 / 3) if available else None)
    assert len(seen) == (2 if available else 1)
    assert feature.observation_ids and feature.source_hashes and feature.question_hash
    assert feature.evidence_available_at == decision_source.record.available_at
    assert feature.generated_at > feature.evidence_available_at


def test_cli_missing_key_and_recorded_replay(tmp_path, decision_request, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    request_path = tmp_path / "request.json"
    request_path.write_text(decision_request.model_dump_json())
    path = tmp_path / "cli.sqlite"
    assert main([str(request_path), "--store", str(path)]) == 2
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "unavailable"
    assert first["raw_response"] is None
    assert (
        main(
            [
                str(request_path),
                "--store",
                str(path),
                "--mode",
                "replay",
                "--observation-id",
                first["observation_id"],
            ]
        )
        == 2
    )
    replay = json.loads(capsys.readouterr().out)
    assert replay["replayed"] and replay["reused_observation_id"] == first["observation_id"]


def test_cli_invalid_file_non_success(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"invalid":true}')
    assert main([str(path), "--store", str(tmp_path / "db")]) == 2
    assert "Invalid decision request" in capsys.readouterr().err


def test_heldout_evaluation_proper_scores_and_option_order(decision_context):
    plan = EvaluationPlan(
        task_id="fixture",
        question_hash="a" * 64,
        model="jev-1.14.0",
        policy_hash="b" * 64,
        dataset_hash="c" * 64,
        split_id="heldout",
        frozen_at=decision_context.as_of,
        min_macro_recall=0.8,
        max_abstention_rate=0.1,
        max_false_block_rate=0.1,
        max_missed_material_rate=0.1,
    )
    cases = tuple(
        LabeledPrediction(
            case_id=str(i),
            source_hash="d" * 64,
            gold="negative",
            prediction=prediction,
            probabilities=(
                ProbabilityValue(option="negative", probability=0.8),
                ProbabilityValue(option="unknown", probability=0.2),
            ),
            material=True,
            variant_group="option-order",
            stressor="reordered_options",
            observation_id=str(i),
            observed_at=decision_context.as_of + timedelta(seconds=1),
        )
        for i, prediction in enumerate(["negative", None])
    )
    report = evaluate_classification(plan, cases)
    assert report["brier_score"] == pytest.approx(0.08)
    assert report["abstention_rate"] == 0.5 and report["missed_material_rate"] == 0.5
    assert report["option_variant_agreement"] == 0 and not report["passed"]
    assert report["per_class"]["negative"]["recall"] == 0.5
    assert report["reliability"][0]["mean_probability"] == 0.8
    too_late = plan.model_copy(update={"frozen_at": cases[0].observed_at})
    with pytest.raises(ValueError, match="frozen before"):
        evaluate_classification(too_late, cases)


def test_consensus_requires_four_named_votes_and_records_derivation(
    decision_context, decision_instrument
):
    roles = ("fundamental", "sentiment", "technical", "risk_manager", "strategist")
    reports = tuple(
        SpecialistReport(
            role=role,
            action="BUY" if role != "risk_manager" else "HOLD",
            status="ok",
            reasoning="fixture vote",
            derived_from=roles[:4] if role == "strategist" else (),
        )
        for role in roles
    )
    decision = AgentDecision(
        context=decision_context,
        instrument=decision_instrument,
        action="BUY",
        reasoning="fixture synthesis",
        new_exposure_blocked=False,
        reports=reports,
    )
    result = consensus_recommendation(decision)
    assert result.action.value == "BUY" and len(result.agreeing_roles) == 4
    assert result.derived_votes == (("strategist", roles[:4]),)
    missing = decision.model_copy(update={"reports": reports[:3]})
    assert consensus_recommendation(missing).action.value == "ABSTAIN"
    veto = decision.model_copy(
        update={"risk_veto": True, "action": "VETO", "new_exposure_blocked": True}
    )
    assert consensus_recommendation(veto).action.value == "VETO"


@pytest.mark.parametrize("canonical", ["ABSTAIN", "HOLD"])
@pytest.mark.parametrize("votes", ["BUY", "REDUCE", "EXIT"])
def test_consensus_never_revives_blocked_or_different_canonical_action(
    decision_context, decision_instrument, canonical, votes
):
    roles = ("fundamental", "sentiment", "technical", "risk_manager", "strategist")
    reports = tuple(
        SpecialistReport(
            role=role,
            action=votes,
            status="ok",
            reasoning="fixture earlier vote",
            derived_from=roles[:4] if role == "strategist" else (),
        )
        for role in roles
    )
    decision = AgentDecision(
        context=decision_context,
        instrument=decision_instrument,
        action=canonical,
        reasoning="Canonical evidence gate or final synthesis",
        reports=reports,
    )
    assert consensus_recommendation(decision).action.value == "ABSTAIN"


def test_consensus_rejects_duplicate_or_unattributed_role_votes(
    decision_context, decision_instrument
):
    roles = ("fundamental", "sentiment", "technical", "risk_manager", "strategist")
    reports = tuple(
        SpecialistReport(
            role=role,
            action="HOLD",
            status="ok",
            reasoning="fixture vote",
        )
        for role in roles
    )
    decision = AgentDecision(
        context=decision_context,
        instrument=decision_instrument,
        action="HOLD",
        reasoning="fixture synthesis",
        reports=reports,
    )
    assert consensus_recommendation(decision).reason == "Strategist derivation is incomplete"
    attributed = reports[:-1] + (reports[-1].model_copy(update={"derived_from": roles[:4]}),)
    valid = decision.model_copy(update={"reports": attributed})
    assert consensus_recommendation(valid).action.value == "HOLD"
    duplicate = valid.model_copy(update={"reports": attributed + (attributed[0],)})
    assert consensus_recommendation(duplicate).action.value == "ABSTAIN"
    risk_veto = reports[3].model_copy(update={"action": "VETO"})
    gated = valid.model_copy(update={"reports": attributed[:3] + (risk_veto,) + attributed[4:]})
    assert consensus_recommendation(gated).action.value == "VETO"
