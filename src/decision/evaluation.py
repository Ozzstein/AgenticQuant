"""Labeled task evaluation; thresholds are frozen before scoring held-out cases.

No bundled synthetic benchmark is used as runtime accuracy evidence. Callers
supply independently labeled financial passages and comparator predictions from
the same examples. This measures classification, not profitable trading.
"""

from __future__ import annotations

import math
from collections import defaultdict

from pydantic import model_validator

from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.domain.decisions import Probability, ProbabilityValue


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
    passed = (
        recall >= plan.min_macro_recall
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
