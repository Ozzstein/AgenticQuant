"""Authoritative prospective desk history for point-in-time consensus research.

Export accepts an identity, never a user-supplied receipt. The receipt is an
archiveable snapshot after re-verification against source, model and budget
stores. Its hash is not a substitute for those checks. Legacy memory is not an
admissible generation history and cannot be upgraded by backdating an import.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.agents.receipts import (
    AgentCallReceipt,
    GenerationReceipt,
    canonical,
    read_call,
    read_generation,
    session_calls,
)
from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.domain.decisions import AgentDecision, DecisionRequest, DecisionResult, DecisionSource


def verify_model_projections(receipt, *, decision_store, evidence_store, require_current=False):
    """Original prompt bytes remain immutable; old unisolated prompts are not memory."""
    from src.decision.workflows import PROJECTION_VERSION, exact_citation, semantic_projection
    from src.domain.decisions import CitedClaim

    calls = {c.call_id: c for c in receipt.calls}
    observed = set(receipt.decision.judgment_observation_ids)
    for generation in receipt.generations:
        messages = json.loads(generation.request_json)
        payload = json.loads(messages[-1]["content"])
        if payload.get("judgment_projection_version") != PROJECTION_VERSION:
            if require_current:
                raise ValueError("legacy prompt lacks certified shadow-isolation boundary")
            continue  # Preserve legacy audit export; never silently upgrade memory eligibility.
        entries = payload.get("accepted_task_judgments", [])
        seen = set()
        for entry in entries:
            identity = entry["observation_id"]
            if identity not in observed:
                raise ValueError("prompt judgment is outside recorded desk observations")
            result = DecisionResult.model_validate(
                decision_store.export_observation(identity)["result"]
            )
            projections = semantic_projection(
                decision_store,
                evidence_store,
                result,
                context=receipt.decision.context,
                instrument=receipt.decision.instrument,
                sources=receipt.sources,
                as_of=calls[generation.call_id].started_at,
            )
            key = (identity, entry["judgment"]["question_hash"])
            if key in seen or entry not in projections:
                raise ValueError("model-visible judgment differs from evaluated source authority")
            seen.add(key)
        extra = payload.get("additional_context") or {}
        for entry in extra.get("citation_checks", []):
            check = exact_citation(
                CitedClaim.model_validate(entry["claim"]), receipt.sources, receipt.decision.context
            )
            if entry != {
                "claim": check.claim.model_dump(mode="json"),
                "status": check.status,
                "reason": check.reason,
            }:
                raise ValueError("model-visible citation includes unapproved semantic status")


class DeskHistoryReceipt(VersionedContract):
    decision_hash: Sha256
    decision: AgentDecision
    session_id: NonEmptyString
    started_at: UtcTimestamp
    completed_at: UtcTimestamp
    available_at: UtcTimestamp
    sources: tuple[DecisionSource, ...]
    calls: tuple[AgentCallReceipt, ...]
    generations: tuple[GenerationReceipt, ...]
    jev_observations_json: str
    inputs_json: str


def memory_projection(receipts):
    """Bound prose without truncating its exact authority references."""
    entries = []
    for receipt in reversed(receipts):
        entry = {
            "decision_hash": receipt.decision_hash,
            "receipt_hash": receipt.content_hash(),
            "as_of": receipt.decision.context.as_of.isoformat(),
            "available_at": receipt.available_at.isoformat(),
            "action": receipt.decision.action.value,
            "reasoning": receipt.decision.reasoning[:500],
        }
        if len(canonical([entry, *entries])) > 2000:
            break
        entries.insert(0, entry)
    references = tuple((v["decision_hash"], v["receipt_hash"]) for v in entries)
    return (canonical(entries) if entries else ""), references


def _validate(receipt):
    if (
        receipt.decision.content_hash() != receipt.decision_hash
        or not receipt.started_at <= receipt.completed_at <= receipt.available_at
        or receipt.decision.context.as_of > receipt.available_at
    ):
        raise ValueError("history identities or clocks disagree")
    calls = {c.call_id: c for c in receipt.calls}
    source_hashes = {s.content_hash() for s in receipt.sources}
    if len(calls) != len(receipt.calls):
        raise ValueError("duplicate generation admission")
    for call in calls.values():
        if (
            call.session_id != receipt.session_id
            or call.run_id
            != receipt.decision.context.run_id + "/" + receipt.decision.instrument.instrument_id
            or not receipt.started_at
            <= call.started_at
            <= call.completed_at
            <= receipt.completed_at
        ):
            raise ValueError("call does not belong to this history session")
    for generation in receipt.generations:
        call = calls.get(generation.call_id)
        if (
            call is None
            or call.status != "ok"
            or call.model != call.actual_model
            or generation.report.provider_model != call.actual_model
            or generation.report.role != generation.role
            or not set(generation.source_hashes).issubset(source_hashes)
            or not call.completed_at <= generation.available_at <= receipt.completed_at
        ):
            raise ValueError("report lacks a settled exact-model source-bound generation")
        source_map = {
            (s.record.evidence_id, s.passage_hash): s
            for s in receipt.sources
            if s.content_hash() in generation.source_hashes
        }
        for claim in generation.report.claims:
            source = source_map.get((claim.evidence_id, claim.passage_hash))
            if source is None or claim.quote not in source.text:
                raise ValueError("report quote absent from its exact generation sources")
    for report in receipt.decision.reports:
        if report.status != "ok":
            continue
        # Graph-owned transformations are limited to derived strategist provenance
        # and monotone CRO veto, never a new independent vote or liquidation.
        possible = []
        for generation in receipt.generations:
            raw = generation.report
            updated = raw.model_dump()
            if report.role == "strategist":
                updated["derived_from"] = report.derived_from
            if report.role == "risk_manager" and report.action.value == "VETO":
                if report.reasoning.endswith("\nQualitative review: " + raw.reasoning):
                    updated.update(action=report.action, reasoning=report.reasoning)
            if updated == report.model_dump():
                possible.append(generation)
        if not possible:
            raise ValueError("decision report is not backed by exact generation history")
    observations = json.loads(receipt.jev_observations_json)
    if [o["result"]["observation_id"] for o in observations] != list(
        receipt.decision.judgment_observation_ids
    ):
        raise ValueError("decision Jev references differ from history")

    def observation(value, *, reused=False):
        request = DecisionRequest.model_validate(value["request"])
        result = DecisionResult.model_validate(value["result"])
        if (
            (not reused and request.context != receipt.decision.context)
            or request.context.as_of != receipt.decision.context.as_of
            or request.instrument != receipt.decision.instrument
            or not set(result.source_hashes).issubset(source_hashes)
            or result.completed_at > receipt.completed_at
        ):
            raise ValueError("Jev observation belongs to a different decision or future")
        if value["reused"] is not None:
            if value["reused"]["result"]["cache_key"] != result.cache_key:
                raise ValueError("reused Jev observation has a different question/model/state key")
            observation(value["reused"], reused=True)

    for value in observations:
        observation(value)


class DeskHistoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS certified_desk_history(decision_hash TEXT PRIMARY KEY, receipt_hash TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            for action in ("UPDATE", "DELETE"):
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS certified_desk_{action.lower()} BEFORE {action} ON certified_desk_history BEGIN SELECT RAISE(ABORT, 'immutable desk history'); END"
                )

    def record_graph_completion(self, state, *, decision_store):
        """Graph write boundary; timestamps are measured here, never supplied."""
        budget = state["call_budget"]
        decision = AgentDecision.model_validate(state["agent_decision"].model_dump())
        calls = session_calls(budget.path, budget.session_id)
        generations = []
        for call in calls:
            try:
                generations.append(read_generation(budget.path, call.call_id))
            except ValueError:
                # Failed/malformed calls retain a charge but generate no report.
                continue
        observations = [
            decision_store.export_observation(identity)
            for identity in decision.judgment_observation_ids
        ]
        now = datetime.now(timezone.utc)
        inputs = {
            key: state.get(key)
            for key in (
                "qlib_context",
                "memory_context",
                "memory_receipts",
                "memory_status",
                "market_context",
                "macro_regime",
                "macro_confidence",
                "evidence_issues",
                "risk_veto_reason",
            )
        }
        for key in ("hmm_context", "portfolio_context"):
            inputs[key] = state[key].model_dump(mode="json") if state.get(key) is not None else None
        receipt = DeskHistoryReceipt(
            decision_hash=decision.content_hash(),
            decision=decision,
            session_id=budget.session_id,
            started_at=budget.started_at,
            completed_at=now,
            available_at=max(now, decision.context.as_of),
            sources=tuple(state.get("sources", ())),
            calls=calls,
            generations=tuple(generations),
            jev_observations_json=canonical(observations),
            inputs_json=canonical(inputs),
        )
        _validate(receipt)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT OR IGNORE INTO certified_desk_history VALUES(?,?,?)",
                (receipt.decision_hash, receipt.content_hash(), receipt.model_dump_json()),
            )
        return receipt.decision_hash

    def retrieve(self, instrument_id, *, as_of, limit=10, **authorities):
        """Bounded projection from existing immutable history, never a second store."""
        if as_of.tzinfo is None or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("aware cutoff and bounded positive limit required")
        candidates = []
        with sqlite3.connect(self.path) as db:
            # Validate headers before filtering: a changed instrument/time cannot
            # silently hide a corrupt formerly eligible row from the query.
            for identity, receipt_hash, payload in db.execute(
                "SELECT decision_hash,receipt_hash,payload FROM certified_desk_history"
            ):
                receipt = DeskHistoryReceipt.model_validate_json(payload)
                if receipt.decision_hash != identity or receipt.content_hash() != receipt_hash:
                    raise ValueError("history payload hash mismatch")
                _validate(receipt)
                if (
                    receipt.decision.instrument.instrument_id == instrument_id
                    and receipt.available_at <= as_of
                    and receipt.decision.context.as_of <= as_of
                ):
                    candidates.append(receipt)
        selected = sorted(candidates, key=lambda r: (r.decision.context.as_of, r.decision_hash))[
            -limit:
        ]
        receipts = tuple(self.export(r.decision_hash, as_of=as_of, **authorities) for r in selected)
        for receipt in receipts:
            verify_model_projections(
                receipt,
                decision_store=authorities["decision_store"],
                evidence_store=authorities["evidence_store"],
                require_current=True,
            )
        return receipts

    def export(
        self,
        decision_hash: str,
        *,
        as_of: datetime,
        evidence_store,
        decision_store,
        budget_store_path: str | Path,
        _ancestors: frozenset[str] = frozenset(),
    ) -> DeskHistoryReceipt:
        """Verify recorded proof against stores and historical cutoff; no network."""
        if as_of.tzinfo is None:
            raise ValueError("history export requires timezone-aware signal cutoff")
        if decision_hash in _ancestors:
            raise ValueError("cyclic memory history")
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT receipt_hash,payload FROM certified_desk_history WHERE decision_hash=?",
                (decision_hash,),
            ).fetchone()
        if row is None:
            raise ValueError("no authoritative generation history for decision")
        receipt = DeskHistoryReceipt.model_validate_json(row[1])
        if receipt.decision_hash != decision_hash or receipt.content_hash() != row[0]:
            raise ValueError("history payload hash mismatch")
        _validate(receipt)
        if receipt.available_at > as_of or receipt.decision.context.as_of > as_of:
            raise ValueError("decision generation/publication is after signal_at")
        for source in receipt.sources:
            source.record.require_available(receipt.decision.context.as_of)
            document = evidence_store.document(
                source.record.evidence_id, as_of=receipt.decision.context.as_of
            )
            if (
                document.evidence != source.record
                or document.document_hash != source.document_hash
                or document.text[source.start : source.end] != source.text
                or hashlib.sha256(evidence_store.raw(source.record.evidence_id)).hexdigest()
                != source.record.raw_hash
            ):
                raise ValueError("history source differs from authoritative exact document span")
        for call in receipt.calls:
            if read_call(budget_store_path, call.call_id) != call:
                raise ValueError("history budget receipt differs from authority")
        for generation in receipt.generations:
            if read_generation(budget_store_path, generation.call_id) != generation:
                raise ValueError("history generation differs from authority")
        expected = [
            decision_store.export_observation(identity)
            for identity in receipt.decision.judgment_observation_ids
        ]
        from src.decision.evaluation import EvaluationService

        def verify_policy(observation):
            request = DecisionRequest.model_validate(observation["request"])
            if request.policy.activation == "evaluated":
                EvaluationService(decision_store, evidence_store=evidence_store).resolve(
                    request.policy,
                    question_hashes=tuple(q.content_hash() for q in request.questions),
                    as_of=request.context.as_of,
                )
            if observation["reused"] is not None:
                verify_policy(observation["reused"])

        for observation in expected:
            verify_policy(observation)
        if canonical(expected) != receipt.jev_observations_json:
            raise ValueError("history Jev observation differs from authority")
        verify_model_projections(
            receipt, decision_store=decision_store, evidence_store=evidence_store
        )
        inputs = json.loads(receipt.inputs_json)
        references = inputs.get("memory_receipts") or ()
        prior = []
        for identity, proof_hash in references:
            value = self.export(
                identity,
                as_of=receipt.decision.context.as_of,
                evidence_store=evidence_store,
                decision_store=decision_store,
                budget_store_path=budget_store_path,
                _ancestors=_ancestors | {decision_hash},
            )
            if (
                value.content_hash() != proof_hash
                or value.decision.instrument != receipt.decision.instrument
            ):
                raise ValueError("memory receipt identity differs")
            verify_model_projections(
                value,
                decision_store=decision_store,
                evidence_store=evidence_store,
                require_current=True,
            )
            prior.append(value)
        projection, verified = memory_projection(prior)
        if (
            projection != (inputs.get("memory_context") or "")
            or tuple(map(tuple, references)) != verified
        ):
            raise ValueError("memory projection lacks exact certified history")
        return receipt
