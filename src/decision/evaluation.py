"""Labeled task evaluation; thresholds are frozen before scoring held-out cases.

No bundled synthetic benchmark is used as runtime accuracy evidence. Callers
supply independently labeled financial passages and comparator predictions from
the same examples. This measures classification, not profitable trading.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone

from pydantic import Field, model_validator

from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.domain.decisions import (
    DecisionPolicy,
    DecisionRequest,
    DecisionResult,
    DecisionSource,
    DecisionStatus,
    Probability,
    ProbabilityValue,
    canonical_json,
    digest,
)


class EvaluationPlan(VersionedContract):
    task_id: NonEmptyString
    question_hash: Sha256
    model: NonEmptyString
    policy_hash: Sha256
    dataset_hash: Sha256
    split_id: NonEmptyString
    frozen_at: UtcTimestamp
    min_macro_recall: Probability
    max_abstention_rate: Probability
    max_false_block_rate: Probability
    max_missed_material_rate: Probability
    labels: tuple[NonEmptyString, ...] = ()
    blocking_labels: tuple[NonEmptyString, ...] = ()
    required_stressors: tuple[NonEmptyString, ...] = ()
    min_variant_agreement: Probability | None = None
    max_ordinal_mae: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique_labels(self):
        if (
            len(set(self.labels)) != len(self.labels)
            or not set(self.blocking_labels).issubset(self.labels)
            or len(set(self.required_stressors)) != len(self.required_stressors)
        ):
            raise ValueError("evaluation label/stressor declaration differs")
        return self


class LabeledPrediction(VersionedContract):
    case_id: NonEmptyString
    source_hash: Sha256
    gold: NonEmptyString
    prediction: str | None
    probabilities: tuple[ProbabilityValue, ...]
    material: bool = False
    blocked: bool = False
    gold_blocked: bool = False
    variant_group: str | None = None
    stressor: str | None = None
    observation_id: NonEmptyString
    observed_at: UtcTimestamp
    ordinal_score: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def distribution(self):
        probs = {p.option: p.probability for p in self.probabilities}
        if len(probs) != len(self.probabilities) or self.gold not in probs:
            raise ValueError("labeled distribution lacks unique gold option")
        if self.prediction is not None and self.prediction not in probs:
            raise ValueError("prediction outside label set")
        if not math.isclose(sum(probs.values()), 1, abs_tol=1e-5):
            raise ValueError("invalid evaluation probability distribution")
        return self


def evaluate_classification(plan: EvaluationPlan, cases: tuple[LabeledPrediction, ...]) -> dict:
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("nonempty unique labeled cases required")
    if any(c.observed_at <= plan.frozen_at for c in cases):
        raise ValueError("evaluation plan must be frozen before held-out observations")
    labels = sorted({p.option for c in cases for p in c.probabilities})
    if any({p.option for p in c.probabilities} != set(labels) for c in cases):
        raise ValueError("all cases must share the exact task label space")
    if plan.labels and set(plan.labels) != set(labels):
        raise ValueError("prediction labels differ from frozen declared classes")
    per_class = {}
    for label in labels:
        tp = sum(c.gold == label and c.prediction == label for c in cases)
        predicted = sum(c.prediction == label for c in cases)
        actual = sum(c.gold == label for c in cases)
        per_class[label] = {
            "precision": tp / predicted if predicted else None,
            "recall": tp / actual if actual else None,
            "support": actual,
        }
    brier = log_loss = 0.0
    bins = defaultdict(list)
    groups = defaultdict(list)
    for case in cases:
        p = {v.option: v.probability for v in case.probabilities}
        brier += sum((v - float(label == case.gold)) ** 2 for label, v in p.items())
        log_loss -= math.log(max(p[case.gold], 1e-15))
        selected = max(p, key=p.get)
        bins[min(9, int(p[selected] * 10))].append((p[selected], selected == case.gold))
        if case.variant_group:
            groups[case.variant_group].append(case)
    reliability = [
        {
            "lower": i / 10,
            "upper": (i + 1) / 10,
            "count": len(values),
            "mean_probability": sum(p for p, _ in values) / len(values),
            "accuracy": sum(correct for _, correct in values) / len(values),
        }
        for i, values in sorted(bins.items())
    ]
    recall = sum(v["recall"] for v in per_class.values() if v["recall"] is not None) / sum(
        v["support"] > 0 for v in per_class.values()
    )
    abstention = sum(c.prediction is None for c in cases) / len(cases)
    unblocked_gold = [c for c in cases if not c.gold_blocked]
    false_block = (
        sum(c.blocked for c in unblocked_gold) / len(unblocked_gold) if unblocked_gold else None
    )
    material = [c for c in cases if c.material]
    missed = sum(c.prediction != c.gold for c in material) / len(material) if material else None
    comparisons = [group for group in groups.values() if len(group) > 1]
    order_agreement = (
        sum(len({c.prediction for c in group}) == 1 for group in comparisons) / len(comparisons)
        if comparisons
        else None
    )
    # Missing critical evaluation populations cannot pass a declared gate.
    missing_classes = [k for k, v in per_class.items() if not v["support"]]
    missing_stressors = sorted(set(plan.required_stressors) - {c.stressor for c in cases})
    ordinal_mae = None
    if plan.max_ordinal_mae is not None:
        if all(c.ordinal_score is not None and c.gold.isdigit() for c in cases):
            ordinal_mae = sum(abs(c.ordinal_score - int(c.gold)) for c in cases) / len(cases)
    passed = (
        not missing_classes
        and not missing_stressors
        and (
            plan.min_variant_agreement is None
            or (order_agreement is not None and order_agreement >= plan.min_variant_agreement)
        )
        and (
            plan.max_ordinal_mae is None
            or (ordinal_mae is not None and ordinal_mae <= plan.max_ordinal_mae)
        )
        and recall >= plan.min_macro_recall
        and abstention <= plan.max_abstention_rate
        and false_block is not None
        and false_block <= plan.max_false_block_rate
        and missed is not None
        and missed <= plan.max_missed_material_rate
    )
    return {
        "plan_hash": plan.content_hash(),
        "n": len(cases),
        "per_class": per_class,
        "macro_recall": recall,
        "abstention_rate": abstention,
        "false_block_rate": false_block,
        "missed_material_rate": missed,
        "brier_score": brier / len(cases),
        "log_loss": log_loss / len(cases),
        "reliability": reliability,
        "option_variant_agreement": order_agreement,
        "missing_classes": missing_classes,
        "missing_stressors": missing_stressors,
        "ordinal_mae": ordinal_mae,
        "passed": passed,
        "observation_ids": [c.observation_id for c in cases],
    }


def compare_classifiers(plan, candidate_cases, baseline_cases):
    """Compare identical labeled source cases; never compare mismatched cohorts."""
    candidate = {
        c.case_id: (c.source_hash, c.gold, c.material, c.gold_blocked) for c in candidate_cases
    }
    baseline = {
        c.case_id: (c.source_hash, c.gold, c.material, c.gold_blocked) for c in baseline_cases
    }
    if candidate != baseline:
        raise ValueError("comparators require identical case IDs, sources and labels")
    evaluated = evaluate_classification(plan, candidate_cases)
    reference = evaluate_classification(plan, baseline_cases)
    return {
        "candidate": evaluated,
        "baseline": reference,
        "brier_improvement": reference["brier_score"] - evaluated["brier_score"],
        "macro_recall_change": evaluated["macro_recall"] - reference["macro_recall"],
    }


class EvaluationCase(VersionedContract):
    """A frozen provider request plus independently archived annotation span."""

    case_id: NonEmptyString
    request: DecisionRequest
    label_source: "DecisionSource"
    gold: NonEmptyString
    material: bool = False
    gold_blocked: bool = False
    variant_group: str | None = None
    stressor: str | None = None

    def annotation(self):
        return self.model_dump(mode="json", exclude={"request", "label_source", "schema_version"})


class FrozenEvaluation(VersionedContract):
    plan: EvaluationPlan
    policy: DecisionPolicy
    cases: tuple[EvaluationCase, ...] = Field(min_length=1, max_length=10000)
    recorded_at: UtcTimestamp


class EvaluationReceipt(VersionedContract):
    cohort_id: Sha256
    observation_ids: tuple[tuple[NonEmptyString, NonEmptyString], ...]
    observation_hashes: tuple[Sha256, ...]
    predictions: tuple[LabeledPrediction, ...]
    report_json: str
    completed_at: UtcTimestamp


class PublishedEvaluation(VersionedContract):
    subject_hash: Sha256
    evaluation_ids: tuple[Sha256, ...] = Field(min_length=1)
    published_at: UtcTimestamp


# These are records in DecisionStore, not a second artifact/promotion registry.
EVALUATION_TYPES = {
    t.__name__: t for t in (FrozenEvaluation, EvaluationReceipt, PublishedEvaluation)
}


def evaluation_dataset_hash(cases):
    return digest(canonical_json([c.model_dump(mode="json") for c in cases]))


class EvaluationService:
    """Resolve measured immutable task evidence; never approve trading/research promotion.

    Public inputs are exact IDs. Reads reparse actual source bytes and recorded
    provider responses. A caller-supplied metric report or evaluation label is
    never accepted as authority. No benchmark data or successful policy ships
    with this implementation.
    """

    def __init__(self, store, *, evidence_store):
        self.store, self.evidence = store, evidence_store

    def _source(self, source, at):
        if self.evidence is None:
            raise ValueError("evaluation source authority is unavailable")
        source.record.require_available(at)
        document = self.evidence.document(source.record.evidence_id, as_of=at)
        import hashlib

        if (
            document.evidence != source.record
            or document.document_hash != source.document_hash
            or document.text[source.start : source.end] != source.text
            or hashlib.sha256(self.evidence.raw(source.record.evidence_id)).hexdigest()
            != source.record.raw_hash
        ):
            raise ValueError("evaluation source differs from exact authoritative span")

    def _cohort(self, cohort):
        plan, policy, cases = cohort.plan, cohort.policy, cohort.cases
        if (
            policy.activation != "shadow"
            or policy.evaluation_id is not None
            or plan.model != policy.model
            or plan.policy_hash != policy.evaluation_subject_hash()
            or plan.dataset_hash != evaluation_dataset_hash(cases)
            or plan.frozen_at > cohort.recorded_at
            or len({c.case_id for c in cases}) != len(cases)
            or len({c.request.content_hash() for c in cases}) != len(cases)
        ):
            raise ValueError("frozen evaluation policy/cohort identity differs")
        thresholds = {t.question_hash: t for t in policy.thresholds}
        if plan.question_hash not in thresholds or not plan.labels:
            raise ValueError("evaluation requires exact declared question threshold and classes")
        for case in cases:
            request = case.request
            if request.policy != policy or len(request.questions) != 1 or not request.sources:
                raise ValueError("case must freeze one exact shadow-policy question and sources")
            q = request.questions[0]
            labels = {c.key for c in q.criteria} if q.kind != "noul" else {"true", "false"}
            if (
                q.content_hash() != plan.question_hash
                or set(plan.labels) != labels
                or case.gold not in labels
            ):
                raise ValueError("case question or gold differs from declared classes")
            if q.kind == "score" and plan.max_ordinal_mae is None:
                raise ValueError("Score activation requires a frozen ordinal error gate")
            for source in request.sources:
                self._source(source, request.context.as_of)
                if (
                    source.record.instrument_ids
                    and request.instrument.instrument_id not in source.record.instrument_ids
                ):
                    raise ValueError("case source belongs to another instrument")
                age = policy.max_source_age_seconds
                if (
                    age is not None
                    and (request.context.as_of - source.record.available_at).total_seconds() > age
                ):
                    raise ValueError("case source exceeds frozen freshness policy")
            self._source(case.label_source, cohort.recorded_at)
            if (
                case.label_source.record.evidence_id
                in {s.record.evidence_id for s in request.sources}
                or case.label_source.document_hash in {s.document_hash for s in request.sources}
                or case.label_source.record.raw_hash in {s.record.raw_hash for s in request.sources}
                or json.loads(case.label_source.text) != case.annotation()
            ):
                raise ValueError("gold annotation must be exact independent archived label data")

    def freeze(self, plan, *, policy, cases):
        cohort = FrozenEvaluation(
            plan=plan, policy=policy, cases=tuple(cases), recorded_at=datetime.now(timezone.utc)
        )
        self._cohort(cohort)
        if any(
            self.store.find(c.request.cache_key(), successful_only=False) is not None for c in cases
        ):
            raise ValueError("evaluation cohort requests already observed before freeze")
        self.store._append_evaluation(cohort)
        return cohort

    def _predictions(self, cohort, observation_ids):
        from src.decision.service import parse_response

        mapping = dict(observation_ids)
        if (
            len(mapping) != len(observation_ids)
            or set(mapping) != {c.case_id for c in cohort.cases}
            or len(set(mapping.values())) != len(mapping)
        ):
            raise ValueError("exact unique frozen cohort observation coverage required")
        values, hashes = [], []
        for case in cohort.cases:
            identity = mapping[case.case_id]
            exported = self.store.export_observation(identity)
            request = DecisionRequest.model_validate(exported["request"])
            result = DecisionResult.model_validate(exported["result"])
            if (
                request != case.request
                or result.status != DecisionStatus.OK
                or result.requested_model != cohort.plan.model
                or result.actual_model != cohort.plan.model
                or result.started_at <= cohort.recorded_at
                or result.completed_at < result.started_at
                or result.reused_observation_id
                or result.cache_hit
                or result.replayed
                or exported["reused"] is not None
                or not result.raw_response
            ):
                raise ValueError("prediction lacks fresh exact frozen request/model observation")
            # Selecting a later lucky attempt cannot replace the first observation.
            first = self.store.find(request.cache_key(), successful_only=False)
            if first is None or first.observation_id != identity:
                raise ValueError("evaluation must retain the first admitted case observation")
            attempts, reservations = exported["attempts"], exported["reservations"]
            if (
                not attempts
                or len(attempts) != result.attempts
                or [a["attempt"] for a in attempts] != list(range(1, result.attempts + 1))
                or [r["attempt"] for r in reservations] != list(range(1, result.attempts + 1))
                or attempts[-1]["status"] != 200
                or attempts[-1]["body"] != result.raw_response
                or any(
                    not result.started_at.timestamp() <= r["at"] <= result.completed_at.timestamp()
                    for r in reservations
                )
            ):
                raise ValueError("prediction provider attempts/admission chronology differs")
            model, judgments, input_tokens, output_tokens = parse_response(
                request, result.raw_response
            )
            if (
                judgments != result.judgments
                or model != result.actual_model
                or input_tokens != result.input_tokens
                or output_tokens != result.output_tokens
            ):
                raise ValueError("prediction differs from recorded raw provider response")
            judgment = judgments[0]
            probabilities = judgment.probabilities
            if judgment.kind == "noul":
                probabilities = (
                    ProbabilityValue(option="true", probability=judgment.yes_probability),
                    ProbabilityValue(option="false", probability=1 - judgment.yes_probability),
                )
            distribution = {p.option: p.probability for p in probabilities}
            prediction = max(distribution, key=distribution.get)
            if (
                judgment.disposition == "abstain"
                or list(distribution.values()).count(distribution[prediction]) > 1
            ):
                prediction = None
            values.append(
                LabeledPrediction(
                    case_id=case.case_id,
                    source_hash=digest(canonical_json([s.content_hash() for s in request.sources])),
                    gold=case.gold,
                    prediction=prediction,
                    probabilities=probabilities,
                    material=case.material,
                    gold_blocked=case.gold_blocked,
                    blocked=prediction in cohort.plan.blocking_labels,
                    variant_group=case.variant_group,
                    stressor=case.stressor,
                    observation_id=identity,
                    observed_at=result.completed_at,
                    ordinal_score=judgment.score,
                )
            )
            hashes.append(digest(canonical_json(exported)))
        return tuple(values), tuple(hashes)

    def score(self, cohort_id, *, observation_ids):
        cohort = self.store.evaluation_artifact(cohort_id, FrozenEvaluation)
        self._cohort(cohort)
        observations = tuple(sorted(observation_ids))
        predictions, hashes = self._predictions(cohort, observations)
        receipt = EvaluationReceipt(
            cohort_id=cohort_id,
            observation_ids=observations,
            observation_hashes=hashes,
            predictions=predictions,
            report_json=canonical_json(evaluate_classification(cohort.plan, predictions)),
            completed_at=datetime.now(timezone.utc),
        )
        self._receipt(receipt)
        self.store._append_evaluation(receipt)
        return receipt

    def _receipt(self, receipt):
        cohort = self.store.evaluation_artifact(receipt.cohort_id, FrozenEvaluation)
        self._cohort(cohort)
        predictions, hashes = self._predictions(cohort, receipt.observation_ids)
        report = evaluate_classification(cohort.plan, predictions)
        if (
            receipt.predictions != predictions
            or receipt.observation_hashes != hashes
            or receipt.report_json != canonical_json(report)
            or receipt.completed_at < max(p.observed_at for p in predictions)
        ):
            raise ValueError("evaluation metrics/predictions or chronology differ from authority")
        return cohort, report

    def _publication(self, publication, policy):
        if publication.subject_hash != policy.evaluation_subject_hash():
            raise ValueError("evaluated policy differs from frozen policy subject")
        questions = []
        for identity in publication.evaluation_ids:
            receipt = self.store.evaluation_artifact(identity, EvaluationReceipt)
            if self.store.evaluation_cohort_role(receipt.cohort_id) in ("selection", "calibration"):
                raise ValueError(
                    "selection/calibration cohorts cannot authorize policy publication"
                )
            cohort, report = self._receipt(receipt)
            if (
                not report["passed"]
                or cohort.policy.evaluation_subject_hash() != publication.subject_hash
                or receipt.completed_at > publication.published_at
            ):
                raise ValueError("evaluation did not pass the exact policy before publication")
            questions.append(cohort.plan.question_hash)
        if len(set(questions)) != len(questions) or set(questions) != {
            t.question_hash for t in policy.thresholds
        }:
            raise ValueError("publication requires exact unique policy task coverage")
        return publication

    def publish(self, policy, *, evaluation_ids):
        if policy.activation != "shadow" or policy.evaluation_id is not None:
            raise ValueError("publish the frozen shadow candidate before referencing evaluation")
        publication = PublishedEvaluation(
            subject_hash=policy.evaluation_subject_hash(),
            evaluation_ids=tuple(sorted(evaluation_ids)),
            published_at=datetime.now(timezone.utc),
        )
        self._publication(publication, policy)
        self.store._append_evaluation(publication)
        return publication

    def resolve(self, policy, *, question_hashes, as_of):
        if policy.activation != "evaluated" or not policy.evaluation_id or as_of.tzinfo is None:
            raise ValueError("explicit evaluated policy and aware cutoff required")
        publication = self.store.evaluation_artifact(policy.evaluation_id, PublishedEvaluation)
        self._publication(publication, policy)
        if publication.published_at > as_of or not set(question_hashes).issubset(
            {t.question_hash for t in policy.thresholds}
        ):
            raise ValueError("policy publication is future or lacks requested question")
        return publication
