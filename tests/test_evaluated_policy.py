"""Synthetic labels/provider boundary test authority only, never financial acceptance.

All source/label bytes, provider observations, budgets, freezes and publications
use actual stores. No runtime canned cohort or accepted policy is installed.
"""

# ruff: noqa: F811
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from src.data.evidence.store import EvidenceStore
from src.decision.evaluation import (
    EvaluationCase,
    EvaluationPlan,
    EvaluationService,
    LabeledPrediction,
    evaluate_classification,
    evaluation_dataset_hash,
)
from src.decision.service import DecisionService, OperatingLimits, parse_response
from src.decision.store import DecisionStore
from src.decision.transport import JevHTTPTransport
from src.domain.decisions import (
    Criterion,
    DecisionPolicy,
    DecisionQuestion,
    DecisionSource,
    DecisionStatus,
    ProbabilityValue,
    TaskThreshold,
    canonical_json,
    digest,
)
from tests.test_decision_contracts import (  # noqa: F401
    decision_context,
    decision_instrument,
    decision_question,
    decision_request,
    decision_source,
)
from tests.test_decision_service import response_for


@pytest.fixture(params=["choice", "noul", "score"])
def evaluation_env(request, tmp_path, decision_request, decision_source):
    kind = request.param
    decision_source = decision_source.model_copy(
        update={
            "record": decision_source.record.model_copy(
                update={"requested_as_of": decision_request.context.as_of}
            )
        }
    )
    decision_request = decision_request.model_copy(update={"sources": (decision_source,)})
    labels = (
        ("down", "up")
        if kind == "choice"
        else (("true", "false") if kind == "noul" else ("0", "1"))
    )
    question = DecisionQuestion(
        question_id="direction",
        version="test-only-1",
        kind=kind,
        instructions="Synthetic evaluation protocol fixture",
        criteria=tuple(Criterion(key=k, description=k) for k in labels),
    )
    policy = DecisionPolicy(
        policy_id="test-protocol-only",
        thresholds=(TaskThreshold(question_hash=question.content_hash(), min_probability=0.5),),
    )
    evidence = EvidenceStore(tmp_path / "sources.sqlite")
    evidence.put(
        decision_source.record,
        decision_source.text.encode(),
        kind="transcript",
        logical_key="source",
        document=decision_source.text,
    )
    cases = []
    for index, gold in enumerate(labels):
        case_id = "case-" + str(index)
        context = decision_request.context.model_copy(update={"run_id": case_id})
        observed_request = decision_request.model_copy(
            update={
                "context": context,
                "questions": (question,),
                "policy": policy,
                "state_json": canonical_json({"case_id": case_id}),
            }
        )
        annotation = {
            "case_id": case_id,
            "gold": gold,
            "material": True,
            "gold_blocked": False,
            "variant_group": None,
            "stressor": None,
        }
        text = canonical_json(annotation)
        record = decision_source.record.model_copy(
            update={"evidence_id": "label-" + case_id, "raw_hash": digest(text)}
        )
        label_source = DecisionSource.from_document(
            record, text, start=0, end=len(text), kind="other"
        )
        evidence.put(
            record, text.encode(), kind="annotation", logical_key=record.evidence_id, document=text
        )
        cases.append(
            EvaluationCase(request=observed_request, label_source=label_source, **annotation)
        )
    cases = tuple(cases)
    plan = EvaluationPlan(
        task_id="test-" + kind,
        question_hash=question.content_hash(),
        model=policy.model,
        policy_hash=policy.evaluation_subject_hash(),
        dataset_hash=evaluation_dataset_hash(cases),
        split_id="test-only",
        frozen_at=datetime.now(timezone.utc),
        labels=labels,
        min_macro_recall=1.0,
        max_abstention_rate=0.0,
        max_false_block_rate=0.0,
        max_missed_material_rate=0.0,
        max_ordinal_mae=0.0 if kind == "score" else None,
    )
    seen = []

    def handler(req):
        payload = json.loads(req.content)
        seen.append(payload)
        # Synthetic protocol response only; the fixture follows its frozen case.
        index = next(
            (
                i
                for i, c in enumerate(cases)
                if c.case_id == payload["state"]["context"].get("case_id")
            ),
            0,
        )
        gold = labels[index]
        answer = {"type": kind}
        if kind == "noul":
            answer["noul"] = float(gold == "true")
        else:
            answer.update(confidence=1.0, probabilities={k: float(k == gold) for k in labels})
            if kind == "choice":
                answer["choice"] = gold
            else:
                answer.update(score=float(gold), legend={k: k for k in labels})
        return httpx.Response(
            200,
            json={
                "model": policy.model,
                "answers": {"direction": answer},
                "usage": {"input_tokens": 10, "output_tokens": 10},
            },
        )

    store = DecisionStore(tmp_path / "decisions.sqlite")
    transport = JevHTTPTransport("synthetic-only", transport=httpx.MockTransport(handler))
    service = DecisionService(
        store,
        evidence_store=evidence,
        transport=transport,
        limits=OperatingLimits(min_interval_seconds=0),
    )
    evaluator = EvaluationService(store, evidence_store=evidence)
    cohort = evaluator.freeze(plan, policy=policy, cases=cases)
    return SimpleNamespace(
        service=service,
        store=store,
        evaluator=evaluator,
        evidence=evidence,
        policy=policy,
        cases=cases,
        cohort=cohort,
        seen=seen,
    )


def score(env):
    observations = tuple(
        (c.case_id, env.service.evaluate(c.request, mode="refresh").observation_id)
        for c in env.cases
    )
    return env.evaluator.score(env.cohort.content_hash(), observation_ids=observations)


def publish(env):
    receipt = score(env)
    proof = env.evaluator.publish(env.policy, evaluation_ids=(receipt.content_hash(),))
    policy = env.policy.model_copy(
        update={"activation": "evaluated", "evaluation_id": proof.content_hash()}
    )
    request = env.cases[0].request.model_copy(
        update={
            "policy": policy,
            "context": env.cases[0].request.context.model_copy(
                update={"as_of": datetime.now(timezone.utc), "run_id": "operating"}
            ),
        }
    )
    return receipt, proof, request


def test_only_reverified_published_policy_accepts_and_restart_replay(evaluation_env):
    env = evaluation_env
    receipt, publication, request = publish(env)
    assert json.loads(receipt.report_json)["passed"]
    assert all(c["support"] == 1 for c in json.loads(receipt.report_json)["per_class"].values())
    assert receipt.observation_hashes and receipt.predictions
    observed = env.service.evaluate(request)
    assert observed.status == DecisionStatus.OK
    assert observed.judgments[0].disposition == "accepted"
    # Parser is incapable of granting policy authority itself.
    assert parse_response(request, observed.raw_response)[1][0].disposition == "observed"
    service = DecisionService(
        DecisionStore(env.store.path),
        evidence_store=env.evidence,
        transport=env.service.transport,
        limits=env.service.limits,
    )
    replay = service.evaluate(request, mode="replay", observation_id=observed.observation_id)
    assert replay.replayed and replay.judgments == observed.judgments
    assert len(env.seen) == len(env.cases) + 1
    with env.store.connect() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM evaluation_artifacts")


def test_declared_class_with_zero_support_never_passes(decision_context):
    plan = EvaluationPlan(
        task_id="no-quality-claim",
        question_hash="a" * 64,
        model="jev-1.14.0",
        policy_hash="b" * 64,
        dataset_hash="c" * 64,
        split_id="test",
        frozen_at=decision_context.as_of,
        labels=("down", "up"),
        min_macro_recall=0.0,
        max_abstention_rate=1.0,
        max_false_block_rate=1.0,
        max_missed_material_rate=1.0,
    )
    case = LabeledPrediction(
        case_id="one",
        source_hash="d" * 64,
        gold="down",
        prediction="down",
        material=True,
        probabilities=(
            ProbabilityValue(option="down", probability=1.0),
            ProbabilityValue(option="up", probability=0.0),
        ),
        observation_id="not-authority",
        observed_at=decision_context.as_of + timedelta(seconds=1),
    )
    result = evaluate_classification(plan, (case,))
    assert not result["passed"] and result["missing_classes"] == ["up"]


@pytest.mark.parametrize("mode", ["cache", "refresh", "replay"])
def test_declared_evaluated_label_has_no_authority(tmp_path, decision_request, mode):
    policy = DecisionPolicy(
        activation="evaluated",
        evaluation_id="invented",
        thresholds=(
            TaskThreshold(
                question_hash=decision_request.questions[0].content_hash(), min_probability=0.5
            ),
        ),
    )
    request = decision_request.model_copy(update={"policy": policy})
    assert (
        parse_response(request, canonical_json(response_for(request)))[1][0].disposition
        == "observed"
    )

    def forbidden(_):
        pytest.fail("unproved policy must be rejected before network")

    service = DecisionService(
        DecisionStore(tmp_path / "observations"),
        transport=JevHTTPTransport("test", transport=httpx.MockTransport(forbidden)),
    )
    result = service.evaluate(request, mode=mode)
    assert (
        result.status == DecisionStatus.UNAVAILABLE
        and not result.judgments
        and result.attempts == 0
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "threshold",
        "model",
        "question",
        "future",
        "missing_sources",
        "cohort",
        "raw",
        "metric",
        "ordering",
        "prediction",
        "budget",
    ],
)
def test_substituted_policy_source_or_proof_blocks_cache_and_replay(evaluation_env, mutation):
    env = evaluation_env
    receipt, publication, request = publish(env)
    assert env.service.evaluate(request).status == DecisionStatus.OK
    calls = len(env.seen)
    if mutation == "threshold":
        request = request.model_copy(
            update={
                "policy": request.policy.model_copy(
                    update={
                        "thresholds": (
                            request.policy.thresholds[0].model_copy(
                                update={"min_probability": 0.9}
                            ),
                        )
                    }
                )
            }
        )
    elif mutation == "model":
        request = request.model_copy(
            update={"policy": request.policy.model_copy(update={"model": "jev-1.15.0"})}
        )
    elif mutation == "question":
        request = request.model_copy(
            update={"questions": (request.questions[0].model_copy(update={"version": "changed"}),)}
        )
    elif mutation == "ordering":
        request = request.model_copy(
            update={
                "questions": (
                    request.questions[0].model_copy(
                        update={"criteria": tuple(reversed(request.questions[0].criteria))}
                    ),
                )
            }
        )
    elif mutation in ("prediction", "budget"):
        with env.store.connect() as db:
            if mutation == "budget":
                db.execute("DROP TRIGGER reservation_update")
                db.execute(
                    "UPDATE reservations SET at=0 WHERE observation_id=?",
                    (receipt.predictions[0].observation_id,),
                )
            else:
                db.execute("DROP TRIGGER observation_immutable_update")
                raw = json.loads(
                    db.execute(
                        "SELECT result_json FROM observations WHERE observation_id=?",
                        (receipt.predictions[0].observation_id,),
                    ).fetchone()[0]
                )
                raw["judgments"][0]["disposition"] = "abstain"
                db.execute(
                    "UPDATE observations SET result_json=? WHERE observation_id=?",
                    (json.dumps(raw), receipt.predictions[0].observation_id),
                )
    elif mutation == "future":
        request = request.model_copy(
            update={
                "context": request.context.model_copy(
                    update={"as_of": publication.published_at - timedelta(microseconds=1)}
                )
            }
        )
    elif mutation == "missing_sources":
        env.service.evidence_store = EvidenceStore(env.store.path.parent / "empty.sqlite")
    elif mutation in ("cohort", "metric"):
        identity = env.cohort.content_hash() if mutation == "cohort" else receipt.content_hash()
        with env.store.connect() as db:
            db.execute("DROP TRIGGER evaluation_immutable_update")
            payload = json.loads(
                db.execute(
                    "SELECT payload FROM evaluation_artifacts WHERE identity=?", (identity,)
                ).fetchone()[0]
            )
            if mutation == "cohort":
                payload["cases"][0]["gold"] = "unrelated"
            else:
                payload["report_json"] = "{}"
            db.execute(
                "UPDATE evaluation_artifacts SET payload=? WHERE identity=?",
                (json.dumps(payload), identity),
            )
    else:
        with env.store.connect() as db:
            db.execute("DROP TRIGGER attempt_immutable_update")
            db.execute(
                "UPDATE attempts SET body='{}' WHERE observation_id=?",
                (receipt.predictions[0].observation_id,),
            )
    for mode in ("cache", "replay"):
        denied = env.service.evaluate(request, mode=mode)
        assert denied.status == DecisionStatus.UNAVAILABLE and not denied.judgments
    assert len(env.seen) == calls


def test_exact_cohort_and_first_observation_cannot_be_selected_after_results(evaluation_env):
    env = evaluation_env
    first = tuple(
        (c.case_id, env.service.evaluate(c.request, mode="refresh").observation_id)
        for c in env.cases
    )
    with pytest.raises(ValueError, match="already observed"):
        env.evaluator.freeze(env.cohort.plan, policy=env.policy, cases=env.cases)
    with pytest.raises(ValueError, match="coverage"):
        env.evaluator.score(env.cohort.content_hash(), observation_ids=first[:1])
    later = env.service.evaluate(env.cases[0].request, mode="refresh")
    with pytest.raises(ValueError, match="first admitted"):
        env.evaluator.score(
            env.cohort.content_hash(),
            observation_ids=((first[0][0], later.observation_id), *first[1:]),
        )


def test_invented_observation_and_changed_question_are_not_evaluation(evaluation_env):
    env = evaluation_env
    with pytest.raises(ValueError, match="unknown authoritative"):
        env.evaluator.score(
            env.cohort.content_hash(),
            observation_ids=tuple((c.case_id, "invented-" + c.case_id) for c in env.cases),
        )
    changed = env.cases[0].model_copy(
        update={"request": env.cases[0].request.model_copy(update={"state_json": '{"new":true}'})}
    )
    with pytest.raises(ValueError, match="cohort identity"):
        env.evaluator.freeze(env.cohort.plan, policy=env.policy, cases=(changed, *env.cases[1:]))


def test_zero_support_blocks_authoritative_publication(evaluation_env):
    env = evaluation_env
    cases = env.cases[:1]
    plan = env.cohort.plan.model_copy(update={"dataset_hash": evaluation_dataset_hash(cases)})
    cohort = env.evaluator.freeze(plan, policy=env.policy, cases=cases)
    observation = env.service.evaluate(cases[0].request)
    receipt = env.evaluator.score(
        cohort.content_hash(), observation_ids=((cases[0].case_id, observation.observation_id),)
    )
    report = json.loads(receipt.report_json)
    assert not report["passed"] and report["missing_classes"]
    with pytest.raises(ValueError, match="did not pass"):
        env.evaluator.publish(env.policy, evaluation_ids=(receipt.content_hash(),))


def test_missing_and_duplicate_task_evaluations_cannot_publish(evaluation_env):
    env = evaluation_env
    receipt = score(env)
    with pytest.raises(ValueError, match="unique policy task"):
        env.evaluator.publish(
            env.policy, evaluation_ids=(receipt.content_hash(), receipt.content_hash())
        )
    with pytest.raises(ValueError, match="missing authoritative"):
        env.evaluator.publish(env.policy, evaluation_ids=("0" * 64,))


def test_evaluated_deadline_includes_proof_resolution_and_no_new_charge(evaluation_env):
    env = evaluation_env
    _, _, request = publish(env)
    calls = len(env.seen)
    denied = env.service.evaluate(request, timeout_seconds=0)
    assert denied.status == DecisionStatus.TIMED_OUT
    assert denied.attempts == 0 and len(env.seen) == calls


def test_cli_replay_requires_actual_evidence_authority(evaluation_env, tmp_path, capsys):
    from src.decision.__main__ import main

    env = evaluation_env
    _, _, request = publish(env)
    result = env.service.evaluate(request)
    path = tmp_path / "evaluated-request.json"
    path.write_text(request.model_dump_json())
    args = [
        str(path),
        "--store",
        str(env.store.path),
        "--mode",
        "replay",
        "--observation-id",
        result.observation_id,
    ]
    assert main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"
    assert main([*args, "--evidence", str(env.evidence.path)]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["replayed"] and replay["judgments"][0]["disposition"] == "accepted"


def test_cli_explicit_absent_authority_is_not_created(tmp_path, decision_request, capsys):
    from src.decision.__main__ import main

    request_path = tmp_path / "request.json"
    request_path.write_text(decision_request.model_dump_json())
    missing = tmp_path / "missing-evidence.sqlite"
    assert (
        main(
            [
                str(request_path),
                "--store",
                str(tmp_path / "observations.sqlite"),
                "--evidence",
                str(missing),
            ]
        )
        == 2
    )
    assert not missing.exists()
    assert "inaccessible observation/evidence authority" in capsys.readouterr().err
