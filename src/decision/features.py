"""J5 text features for research; no fitting, promotion or allocation authority."""

from __future__ import annotations

from src.decision.questions import DEMAND_AVAILABLE, DEMAND_WEAKNESS
from src.domain.common import UtcTimestamp, VersionedContract
from src.domain.decisions import DecisionQuestion, DecisionStatus


class TextFeature(VersionedContract):
    name: str
    transformation_version: str
    value: float | None
    status: str
    evidence_available_at: UtcTimestamp | None
    generated_at: UtcTimestamp
    observation_ids: tuple[str, ...]
    question_hash: str
    model: str
    policy_hash: str
    source_hashes: tuple[str, ...]
    historical_knowledge_caveat: str = (
        "archived inputs do not remove knowledge in pretrained weights"
    )


class TextFeatureDefinition(VersionedContract):
    name: str
    question: DecisionQuestion
    availability_question: DecisionQuestion
    available_choice: str = "present"
    transformation_version: str = "normalized-score-or-yes-probability-v1"


def demand_weakness_feature(workflow, sources) -> TextFeature:
    """Dependent scoring follows an availability judgment; missing is never zero.

    Uses probability-weighted ordinal score / 3, with source and generation times
    separate. Research must respect both, plus question selection/holdout gates.
    """
    return compute_text_feature(
        workflow,
        sources,
        TextFeatureDefinition(
            name="demand_weakness",
            question=DEMAND_WEAKNESS,
            availability_question=DEMAND_AVAILABLE,
            transformation_version="ordinal_expected_div3-v1",
        ),
    )


def compute_text_feature(workflow, sources, definition: TextFeatureDefinition) -> TextFeature:
    """Reusable proposed-question adapter; training and promotion stay external.

    A source availability Choice precedes dependent Score/Noul evaluation. All
    definitions and transformations must be selected inside development windows.
    """
    if (
        definition.question.kind not in ("score", "noul")
        or definition.availability_question.kind != "choice"
    ):
        raise ValueError("numeric feature requires Score/Noul and an availability Choice")
    if definition.available_choice not in {
        c.key for c in definition.availability_question.criteria
    }:
        raise ValueError("availability label missing from question")
    available = workflow.ask(sources, (definition.availability_question,))
    observations = [available]
    value, status = None, available.status.value
    if available.status == DecisionStatus.OK:
        judgment = available.judgments[0]
        if judgment.choice == definition.available_choice and judgment.disposition != "abstain":
            result = workflow.ask(sources, (definition.question,))
            observations.append(result)
            status = result.status.value
            if result.status == DecisionStatus.OK and result.judgments[0].disposition != "abstain":
                scored = result.judgments[0]
                value = (
                    scored.score / (len(definition.question.criteria) - 1)
                    if scored.kind == "score"
                    else scored.yes_probability
                )
        else:
            status = "insufficient_evidence"
    return TextFeature(
        name=definition.name,
        transformation_version=definition.transformation_version,
        value=value,
        status=status,
        evidence_available_at=max(
            (s.record.available_at for s in sources if s.record.available_at), default=None
        ),
        generated_at=observations[-1].completed_at,
        observation_ids=tuple(r.observation_id for r in observations),
        question_hash=definition.content_hash(),
        model=workflow.policy.model,
        policy_hash=workflow.policy.content_hash(),
        source_hashes=tuple(s.content_hash() for s in sources),
    )
