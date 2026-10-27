"""Immutable decision/evidence contracts; judgments carry no execution authority."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from src.domain.common import (
    EvidenceRecord,
    Instrument,
    NonEmptyString,
    RunContext,
    Sha256,
    UtcTimestamp,
    VersionedContract,
)

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
PINNED_MODEL = "jev-1.14.0"
API_CONTRACT = "typesafe-openapi-0.2.0@2026-10-27"


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DecisionSource(VersionedContract):
    """Exact bounded span in a stored document; full content belongs to EvidenceStore.

    Use the store adapter to establish document identity. This contract validates
    span integrity, not the truth of the source or its provider's metadata.
    """

    record: EvidenceRecord
    document_hash: Sha256
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=12000)
    passage_hash: Sha256
    kind: Literal["news", "transcript", "fundamental", "technical", "macro", "memory", "other"]

    @model_validator(mode="after")
    def validate_span(self):
        if self.end - self.start != len(self.text) or digest(self.text) != self.passage_hash:
            raise ValueError("passage offsets/hash do not match exact text")
        return self

    @classmethod
    def from_document(
        cls, record: EvidenceRecord, document: str, *, start: int, end: int, kind: str
    ) -> DecisionSource:
        if not 0 <= start < end <= len(document):
            raise ValueError("invalid document span")
        return cls(
            record=record,
            document_hash=digest(document),
            start=start,
            end=end,
            text=document[start:end],
            passage_hash=digest(document[start:end]),
            kind=kind,
        )


class Criterion(VersionedContract):
    key: NonEmptyString
    description: NonEmptyString


class DecisionQuestion(VersionedContract):
    question_id: NonEmptyString
    version: NonEmptyString
    kind: Literal["choice", "score", "noul"]
    instructions: NonEmptyString
    criteria: tuple[Criterion, ...] = ()
    unknown_options: tuple[str, ...] = ("unknown", "insufficient")

    @model_validator(mode="after")
    def check_criteria(self):
        keys = [c.key for c in self.criteria]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate criteria keys")
        if self.kind == "choice" and not 2 <= len(keys) <= 255:
            raise ValueError("Choice requires 2..255 distinct options")
        if self.kind == "score" and (
            not 2 <= len(keys) <= 10 or keys != [str(i) for i in range(len(keys))]
        ):
            raise ValueError("Score requires 2..10 ordered zero-based levels")
        if self.kind == "noul" and keys and set(keys) != {"true", "false"}:
            raise ValueError("Noul criteria must define true and false")
        return self

    def wire(self) -> dict:
        result = {"type": self.kind, "instructions": self.instructions}
        if self.kind == "score":
            result["criteria"] = [c.description for c in self.criteria]
        elif self.criteria:
            result["criteria"] = {c.key: c.description for c in self.criteria}
        return result


class TaskThreshold(VersionedContract):
    question_hash: Sha256
    min_probability: Probability
    min_provider_confidence: Probability | None = None


class DecisionPolicy(VersionedContract):
    policy_id: NonEmptyString = "jev-shadow"
    version: NonEmptyString = "1.0"
    model: NonEmptyString = PINNED_MODEL
    activation: Literal["shadow", "evaluated"] = "shadow"
    evaluation_id: NonEmptyString | None = None
    thresholds: tuple[TaskThreshold, ...] = ()
    max_source_age_seconds: float | None = Field(default=None, gt=0)

    @field_validator("model")
    @classmethod
    def pinned(cls, value):
        if not re.fullmatch(r"jev-\d+\.\d+\.\d+", value):
            raise ValueError("an exact versioned Jev model is required, never an alias")
        return value

    @model_validator(mode="after")
    def activation_evidence(self):
        if self.activation == "evaluated" and (not self.evaluation_id or not self.thresholds):
            raise ValueError("activated policy requires evaluation identity and task thresholds")
        hashes = [t.question_hash for t in self.thresholds]
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate task thresholds")
        return self


class DecisionRequest(VersionedContract):
    context: RunContext
    instrument: Instrument
    questions: tuple[DecisionQuestion, ...] = Field(min_length=1, max_length=64)
    sources: tuple[DecisionSource, ...] = ()
    state_json: str = "{}"
    policy: DecisionPolicy = DecisionPolicy()
    api_contract: Literal[API_CONTRACT] = API_CONTRACT

    @field_validator("state_json")
    @classmethod
    def canonical_state(cls, value):
        def invalid(v):
            raise ValueError("non-finite JSON")

        state = json.loads(value, parse_constant=invalid)
        if not isinstance(state, (dict, list, str)):
            raise ValueError("state must be a JSON object, array, or string")
        return canonical_json(state)

    @model_validator(mode="after")
    def identities(self):
        ids = [q.question_id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate question IDs")
        sources = [(s.record.evidence_id, s.start, s.end) for s in self.sources]
        if len(sources) != len(set(sources)):
            raise ValueError("duplicate source spans")
        if self.policy.activation == "evaluated":
            thresholds = {t.question_hash for t in self.policy.thresholds}
            if any(q.content_hash() not in thresholds for q in self.questions):
                raise ValueError("activated policy lacks threshold for exact question")
        return self

    @property
    def state_hash(self) -> str:
        return digest(self.state_json)

    def cache_key(self) -> str:
        # Run ID is lineage rather than semantic identity. All other context,
        # including as_of, config/code/snapshot IDs and mode, stays in the key.
        data = self.model_dump(mode="json")
        del data["context"]["run_id"]
        return digest(canonical_json(data))

    def wire(self) -> dict:
        return {
            "model": self.policy.model,
            "state": {
                "context": json.loads(self.state_json),
                "instrument": self.instrument.model_dump(mode="json"),
                "as_of": self.context.as_of.isoformat(),
                "untrusted_evidence": [
                    {
                        "evidence_id": s.record.evidence_id,
                        "raw_hash": s.record.raw_hash,
                        "passage_hash": s.passage_hash,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                    }
                    for s in self.sources
                ],
            },
            "questions": {q.question_id: q.wire() for q in self.questions},
        }


class ProbabilityValue(VersionedContract):
    option: NonEmptyString
    probability: Probability


class Judgment(VersionedContract):
    question_id: NonEmptyString
    question_hash: Sha256
    kind: Literal["choice", "score", "noul"]
    choice: str | None = None
    score: float | None = None
    yes_probability: Probability | None = None
    probabilities: tuple[ProbabilityValue, ...] = ()
    provider_confidence: Probability | None = None
    confidence_definition: str | None = None
    disposition: Literal["observed", "accepted", "abstain"] = "observed"
    # This is task classification, never a financial-return estimate.
    calibrated_task_accuracy: Probability | None = None


class DecisionStatus(StrEnum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"
    MALFORMED = "malformed"
    STALE = "stale"
    INSUFFICIENT = "insufficient_evidence"
    VERSION_MISMATCH = "version_mismatch"
    BUDGET = "budget_exhausted"
    CANCELLED = "cancelled"
    REPLAY_MISS = "replay_miss"


class DecisionResult(VersionedContract):
    observation_id: NonEmptyString
    request_hash: Sha256
    cache_key: Sha256
    run_id: NonEmptyString
    status: DecisionStatus
    reason: str = ""
    requested_model: str
    actual_model: str | None = None
    policy_hash: Sha256
    state_hash: Sha256
    source_hashes: tuple[Sha256, ...]
    judgments: tuple[Judgment, ...] = ()
    started_at: UtcTimestamp
    completed_at: UtcTimestamp
    latency_ms: float = Field(ge=0)
    attempts: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    pricing_version: str | None = None
    raw_response: str | None = None
    cache_hit: bool = False
    replayed: bool = False
    reused_observation_id: str | None = None


class ConfidenceSemantics(VersionedContract):
    analyst_conviction: Probability | None = None
    evidence_completeness: Probability | None = None
    calibrated_task_accuracy: Probability | None = None
    legacy_confidence_0_100: float | None = Field(default=None, ge=0, le=100)
    legacy_meaning: Literal["legacy_unvalidated"] | None = None


class AgentAction(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    ABSTAIN = "ABSTAIN"
    VETO = "VETO"


class CitedClaim(VersionedContract):
    claim: NonEmptyString
    evidence_id: NonEmptyString
    passage_hash: Sha256
    quote: NonEmptyString


class SpecialistReport(VersionedContract):
    role: NonEmptyString
    action: AgentAction
    status: Literal["ok", "unavailable", "insufficient", "not_applicable"]
    reasoning: NonEmptyString
    confidence: ConfidenceSemantics = ConfidenceSemantics()
    claims: tuple[CitedClaim, ...] = ()
    risk_flags: tuple[str, ...] = ()
    catalysts: tuple[str, ...] = ()
    target_price: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    position_size_pct: float | None = Field(default=None, ge=0, le=100)
    time_horizon: str | None = None
    provider_model: str | None = None
    derived_from: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unavailable_is_not_neutral(self):
        if self.status != "ok" and self.action not in (AgentAction.ABSTAIN, AgentAction.VETO):
            raise ValueError("unavailable reports must abstain or retain veto")
        return self


class AgentDecision(VersionedContract):
    instrument: Instrument
    context: RunContext
    action: AgentAction
    reasoning: NonEmptyString
    confidence: ConfidenceSemantics = ConfidenceSemantics()
    risk_veto: bool = False
    new_exposure_blocked: bool = True
    reports: tuple[SpecialistReport, ...] = ()
    debate_transcript: tuple[str, ...] = ()
    judgment_observation_ids: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()
    catalysts: tuple[str, ...] = ()
    target_price: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    position_size_pct: float | None = Field(default=None, ge=0, le=100)
    time_horizon: str | None = None

    @model_validator(mode="after")
    def safe_boundary(self):
        if self.risk_veto and self.action != AgentAction.VETO:
            raise ValueError("a binding veto must remain VETO, never become EXIT")
        if self.action in (AgentAction.ABSTAIN, AgentAction.VETO) and not self.new_exposure_blocked:
            raise ValueError("abstention and veto block new exposure")
        if self.action == AgentAction.BUY and self.new_exposure_blocked:
            raise ValueError("blocked new exposure cannot emit BUY")
        return self
