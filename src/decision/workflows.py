"""Shared J1–J6 workflows. Storage/retrieval eligibility and exact matches stay in code."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.decision import questions as q
from src.decision.service import DecisionService
from src.domain.decisions import (
    CitedClaim,
    DecisionPolicy,
    DecisionRequest,
    DecisionResult,
    DecisionSource,
    DecisionStatus,
    canonical_json,
)


@dataclass(frozen=True)
class CitationCheck:
    claim: CitedClaim
    status: str
    reason: str
    result: DecisionResult | None = None


class EvidenceJudgments:
    def __init__(
        self,
        service: DecisionService,
        context,
        instrument,
        *,
        policy=None,
        timeout_seconds=30,
        mode="cache",
        deadline=None,
    ):
        self.service, self.context, self.instrument = service, context, instrument
        self.policy = policy or DecisionPolicy()
        self.timeout_seconds, self.mode = timeout_seconds, mode
        self.deadline = deadline

    def ask(self, sources, questions, state=None):
        return self.service.evaluate(
            DecisionRequest(
                context=self.context,
                instrument=self.instrument,
                policy=self.policy,
                sources=tuple(sources),
                questions=tuple(questions),
                state_json=canonical_json(state or {}),
            ),
            timeout_seconds=min(self.timeout_seconds, max(0, self.deadline - time.monotonic()))
            if self.deadline
            else self.timeout_seconds,
            mode=self.mode,
        )

    def news(self, source: DecisionSource):
        return self.ask((source,), (q.NEWS_TONE, q.RELEVANCE, q.NEWS_EVENT))

    def transcript(self, source: DecisionSource):
        return self.ask((source,), (q.MANAGEMENT_TONE, q.GUIDANCE, q.HEDGING, q.RESPONSIVENESS))

    def verify(self, claim: CitedClaim, sources) -> CitationCheck:
        matches = [
            s
            for s in sources
            if s.record.evidence_id == claim.evidence_id and s.passage_hash == claim.passage_hash
        ]
        if len(matches) != 1:
            return CitationCheck(claim, "invalid_reference", "unknown or ambiguous source/hash")
        source = matches[0]
        if claim.quote not in source.text:
            return CitationCheck(
                claim, "quote_missing", "exact quote absent from referenced passage"
            )
        try:
            source.record.require_available(self.context.as_of)
        except ValueError:
            return CitationCheck(claim, "ineligible", "citation was unavailable at as_of")
        result = self.ask(
            (source,), (q.CLAIM_SUPPORT,), {"claim": claim.claim, "quote": claim.quote}
        )
        label = (
            result.judgments[0].choice
            if result.status == DecisionStatus.OK
            else result.status.value
        )
        return CitationCheck(claim, label, "semantic source support, not source truth", result)

    def retrieve(self, sources, *, query: str, reference: str):
        """Score all candidates; preserve conflicts AND unknown/unavailable passages.

        Returns scored candidates, never a falsely clean corpus. The caller may
        order them but must retain contradictory evidence inside its context bound.
        """
        return tuple(
            (
                source,
                self.ask(
                    (source,), (q.RELEVANCE, q.CONFLICT), {"query": query, "reference": reference}
                ),
            )
            for source in sources
        )

    def debate(self, sources, reports):
        return self.ask(
            sources,
            (q.DEBATE_DISCREPANCY,),
            {
                "reports": [
                    {
                        "role": r.role,
                        "action": r.action.value,
                        "reasoning": r.reasoning,
                        "claims": [c.model_dump(mode="json") for c in r.claims],
                    }
                    for r in reports
                ]
            },
        )
