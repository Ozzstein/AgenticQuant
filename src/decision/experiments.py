"""Bounded, source-bound evaluation execution in the existing DecisionStore.

This composes task evidence. It neither publishes policies nor certifies financial
efficacy, source truth, provider entitlement, or prospective trading performance.
"""

from __future__ import annotations

import fcntl
import json
import math
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from src.decision.evaluation import (
    EvaluationCase,
    EvaluationPlan,
    EvaluationReceipt,
    EvaluationService,
    FrozenEvaluation,
    LabeledPrediction,
    compare_classifiers,
    evaluate_classification,
)
from src.decision.service import DecisionService, OperatingLimits, parse_response
from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.domain.decisions import (
    DecisionPolicy,
    DecisionSource,
    DecisionStatus,
    ProbabilityValue,
    canonical_json,
    digest,
)


class ExperimentBudget(VersionedContract):
    """Whole-experiment reservations, including retries; never per-case caps.

    Pricing is an explicit pinned declaration, not a vendor-verified invoice.
    Unknown pricing prevents paid admission under the finite spend ceiling.
    """

    requests: int = Field(gt=0, le=10000, strict=True)
    tokens: int = Field(gt=0, le=1_000_000_000, strict=True)
    spend_usd: float = Field(ge=0, le=10000)
    duration_seconds: float = Field(gt=0, le=86400)
    case_timeout_seconds: float = Field(gt=0, le=600)
    max_attempts: int = Field(ge=1, le=5, strict=True)
    input_usd_per_million: float | None = Field(ge=0)
    pricing_version: NonEmptyString | None
    max_request_bytes: int = Field(default=24000, gt=0, le=100000, strict=True)
    min_interval_seconds: float = Field(default=0.1, ge=0, le=60)

    @model_validator(mode="after")
    def price_identity(self):
        if (self.input_usd_per_million is None) != (self.pricing_version is None):
            raise ValueError("price and pinned pricing identity must be supplied together")
        return self

    def limits(self):
        return OperatingLimits(
            run_requests=self.requests,
            day_requests=self.requests,
            run_tokens=self.tokens,
            day_tokens=self.tokens,
            run_spend_usd=self.spend_usd,
            day_spend_usd=self.spend_usd,
            max_request_bytes=self.max_request_bytes,
            max_attempts=self.max_attempts,
            attempt_timeout_seconds=self.case_timeout_seconds,
            min_interval_seconds=self.min_interval_seconds,
            input_usd_per_million=self.input_usd_per_million,
            pricing_version=self.pricing_version,
        )


class ExperimentCohort(VersionedContract):
    name: NonEmptyString
    role: Literal["selection", "calibration", "test"]
    plan: EvaluationPlan
    policy: DecisionPolicy
    cases: tuple[EvaluationCase, ...] = Field(min_length=1, max_length=10000)


PairKind = Literal[
    "model-baseline",
    "option-order",
    "missing-evidence",
    "conflicting-evidence",
    "misleading-wording",
    "prompt-injection",
    "numeric-boundary",
    "date-boundary",
]
STRESSORS = (
    "option-order",
    "missing-evidence",
    "conflicting-evidence",
    "misleading-wording",
    "prompt-injection",
    "numeric-boundary",
    "date-boundary",
)


class ExperimentPair(VersionedContract):
    name: NonEmptyString
    kind: PairKind
    base: NonEmptyString
    variant: NonEmptyString
    # Explicit bijection: no cherry-picked subset or implicit row-order joins.
    cases: tuple[tuple[NonEmptyString, NonEmptyString], ...] = Field(min_length=1, max_length=10000)
    annotation_source: DecisionSource

    def annotation(self):
        return self.model_dump(mode="json", exclude={"annotation_source", "schema_version"})


class PriorBaseline(VersionedContract):
    name: NonEmptyString
    training: NonEmptyString
    test: NonEmptyString
    algorithm: Literal["empirical-label-prior-1"] = "empirical-label-prior-1"


class CalibrationSpec(VersionedContract):
    name: NonEmptyString
    calibration: NonEmptyString
    test: NonEmptyString
    temperatures: tuple[float, ...] = Field(min_length=1, max_length=64)
    algorithm: Literal["temperature-grid-logloss-1"] = "temperature-grid-logloss-1"

    @model_validator(mode="after")
    def grid(self):
        if tuple(sorted(set(self.temperatures))) != self.temperatures or any(
            not 0.05 <= v <= 20 for v in self.temperatures
        ):
            raise ValueError("temperature grid must be unique, sorted and within [0.05,20]")
        return self


class EvaluationManifest(VersionedContract):
    experiment_id: NonEmptyString
    run_id: NonEmptyString
    budget: ExperimentBudget
    cohorts: tuple[ExperimentCohort, ...] = Field(min_length=1, max_length=64)
    pairs: tuple[ExperimentPair, ...] = Field(default=(), max_length=256)
    priors: tuple[PriorBaseline, ...] = Field(default=(), max_length=64)
    calibrations: tuple[CalibrationSpec, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def bounds(self):
        if sum(len(c.cases) for c in self.cohorts) > 10000:
            raise ValueError("experiment exceeds 10000 total cases")
        for values in (self.cohorts, self.pairs, self.priors, self.calibrations):
            if len({v.name for v in values}) != len(values):
                raise ValueError("experiment names must be unique within each collection")
        names = {c.name for c in self.cohorts}
        if (
            any(p.base not in names or p.variant not in names for p in self.pairs)
            or any(p.training not in names or p.test not in names for p in self.priors)
            or any(p.calibration not in names or p.test not in names for p in self.calibrations)
        ):
            raise ValueError("comparison references an undeclared cohort")
        return self


class FrozenExperiment(VersionedContract):
    manifest: EvaluationManifest
    cohort_ids: tuple[tuple[NonEmptyString, Sha256], ...]
    recorded_at: UtcTimestamp
    deadline_at: UtcTimestamp


class EvaluationDispatch(VersionedContract):
    experiment_hash: Sha256
    cohort_id: Sha256
    case_id: NonEmptyString
    request_hash: Sha256
    cache_key: Sha256
    observation_id: NonEmptyString
    recorded_at: UtcTimestamp


class ExperimentReport(VersionedContract):
    experiment_hash: Sha256
    dispatch_ids: tuple[Sha256, ...]
    cohort_receipts: tuple[tuple[NonEmptyString, Sha256], ...]
    report_json: str
    observed_through: UtcTimestamp
    recorded_at: UtcTimestamp


EXPERIMENT_TYPES = {t.__name__: t for t in (FrozenExperiment, EvaluationDispatch, ExperimentReport)}


@contextmanager
def experiment_lock(store):
    """One local workflow writer per authoritative DB; held across provider calls."""
    with store.path.resolve().with_suffix(".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("evaluation workflow already running for this store") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _context(request):
    return request.context.model_dump(mode="json", exclude={"run_id"})


def _source_ids(cases):
    # Whole documents/raw blobs, not only spans: sibling passages cannot leak
    # labels across selection/calibration/test roles under different IDs.
    return {
        h
        for case in cases
        for s in (*case.request.sources, case.label_source)
        for h in (s.document_hash, s.record.raw_hash)
    }


class ExperimentService:
    def __init__(self, store, *, evidence_store, transport=None):
        self.store = store
        self.evaluator = EvaluationService(store, evidence_store=evidence_store)
        self.transport = transport

    def _validate_manifest(self, manifest, at):
        cohorts = {c.name: c for c in manifest.cohorts}
        keys = [case.request.cache_key() for c in manifest.cohorts for case in c.cases]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate semantic requests across experiment cohorts")
        visible = {
            h
            for c in manifest.cohorts
            for case in c.cases
            for s in case.request.sources
            for h in (s.document_hash, s.record.raw_hash)
        }
        annotations = [case.label_source for c in manifest.cohorts for case in c.cases]
        annotations.extend(pair.annotation_source for pair in manifest.pairs)
        if visible & {h for s in annotations for h in (s.document_hash, s.record.raw_hash)}:
            raise ValueError(
                "experiment label/pair annotations leak into provider-visible documents"
            )
        for c in manifest.cohorts:
            frozen = FrozenEvaluation(plan=c.plan, policy=c.policy, cases=c.cases, recorded_at=at)
            self.evaluator._cohort(frozen)
            if any(case.request.context.run_id != manifest.run_id for case in c.cases):
                raise ValueError("all cohort requests must share the exact experiment budget run")
            if any(case.request.context.as_of > at for case in c.cases):
                raise ValueError("case context is future to experiment freeze")
        ranks = [{"selection": 0, "calibration": 1, "test": 2}[c.role] for c in manifest.cohorts]
        if ranks != sorted(ranks):
            raise ValueError("execute selection then calibration then test cohorts")
        for i, left in enumerate(manifest.cohorts):
            for right in manifest.cohorts[i + 1 :]:
                if left.role != right.role and (
                    left.plan.split_id == right.plan.split_id
                    or _source_ids(left.cases) & _source_ids(right.cases)
                ):
                    raise ValueError(
                        "selection/calibration/test splits overlap in identity or bytes"
                    )
        for prior in manifest.priors:
            train, test = cohorts[prior.training], cohorts[prior.test]
            if (
                train.role not in ("selection", "calibration")
                or test.role != "test"
                or set(train.plan.labels) != set(test.plan.labels)
                or train.plan.task_id != test.plan.task_id
                or train.plan.question_hash != test.plan.question_hash
            ):
                raise ValueError("prior baseline requires disjoint development and exact test task")
        for calibration in manifest.calibrations:
            train, test = cohorts[calibration.calibration], cohorts[calibration.test]
            if (
                train.role != "calibration"
                or test.role != "test"
                or train.policy != test.policy
                or train.plan.task_id != test.plan.task_id
                or train.plan.question_hash != test.plan.question_hash
                or set(train.plan.labels) != set(test.plan.labels)
            ):
                raise ValueError(
                    "calibration requires exact task/policy and disjoint calibration/test roles"
                )
        for pair in manifest.pairs:
            self.evaluator._source(pair.annotation_source, at)
            if json.loads(pair.annotation_source.text) != pair.annotation():
                raise ValueError("pair annotation differs from archived span")
            a, b = cohorts[pair.base], cohorts[pair.variant]
            if (
                a.name == b.name
                or a.role != b.role
                or a.plan.split_id != b.plan.split_id
                or a.plan.task_id != b.plan.task_id
                or set(a.plan.labels) != set(b.plan.labels)
            ):
                raise ValueError("paired cohorts must share exact task and split role/identity")
            ac, bc = {c.case_id: c for c in a.cases}, {c.case_id: c for c in b.cases}
            if (
                len(pair.cases) != len(ac)
                or len(pair.cases) != len(bc)
                or {x for x, _ in pair.cases} != set(ac)
                or {y for _, y in pair.cases} != set(bc)
            ):
                raise ValueError("pairing requires a complete unique case bijection")
            for x, y in pair.cases:
                self._validate_pair(pair, ac[x], bc[y])

    def _validate_pair(self, pair, a, b):
        x, y = a.request, b.request
        if x.instrument != y.instrument or _context(x) != _context(y):
            raise ValueError("paired examples differ in instrument/context")
        if pair.annotation_source.document_hash in {
            s.document_hash for s in (*x.sources, *y.sources)
        }:
            raise ValueError("pair annotation leaked into model context")
        qx, qy = x.questions[0], y.questions[0]
        px = x.policy.model_dump(exclude={"model", "thresholds"})
        py = y.policy.model_dump(exclude={"model", "thresholds"})
        tx = next(t for t in x.policy.thresholds if t.question_hash == qx.content_hash())
        ty = next(t for t in y.policy.thresholds if t.question_hash == qy.content_hash())
        if (
            px != py
            or tx.model_dump(exclude={"question_hash"}) != ty.model_dump(exclude={"question_hash"})
            or (pair.kind != "model-baseline" and x.policy.model != y.policy.model)
            or x.state_json != y.state_json
        ):
            raise ValueError("paired comparison changes unapproved policy/model/state semantics")
        if pair.kind in ("option-order", "model-baseline"):
            if (
                x.sources != y.sources
                or a.gold != b.gold
                or a.material != b.material
                or a.gold_blocked != b.gold_blocked
                or a.label_source != b.label_source
            ):
                raise ValueError("order/model comparison requires identical sources and labels")
        if pair.kind == "model-baseline":
            if qx != qy or x.policy.model == y.policy.model:
                raise ValueError("model baseline must change only pinned model")
        elif pair.kind == "option-order":
            if (
                qx.kind != "choice"
                or qx.model_dump(exclude={"criteria"}) != qy.model_dump(exclude={"criteria"})
                or sorted(c.model_dump_json() for c in qx.criteria)
                != sorted(c.model_dump_json() for c in qy.criteria)
                or qx.criteria == qy.criteria
            ):
                raise ValueError(
                    "option-order pair must be a real Choice permutation, not relabeling"
                )
        elif pair.kind == "misleading-wording":
            if (
                x.sources != y.sources
                or qx.instructions == qy.instructions
                or qx.model_dump(exclude={"instructions"})
                != qy.model_dump(exclude={"instructions"})
            ):
                raise ValueError("wording pair must change only the actual question instructions")
        else:
            if qx != qy or x.sources == y.sources:
                raise ValueError(
                    "evidence stressor requires exact question and changed source spans"
                )
            if pair.kind == "missing-evidence" and not (
                {s.content_hash() for s in y.sources} < {s.content_hash() for s in x.sources}
            ):
                raise ValueError("missing-evidence variant must remove an actual source subset")

    def freeze(self, manifest: EvaluationManifest) -> FrozenExperiment:
        manifest = EvaluationManifest.model_validate(manifest.model_dump())
        with experiment_lock(self.store):
            existing = self.store.experiment_for_manifest(manifest.content_hash())
            if existing:
                return self.inspect(existing)
            at = datetime.now(timezone.utc)
            self._validate_manifest(manifest, at)
            cohorts = tuple(
                FrozenEvaluation(plan=c.plan, policy=c.policy, cases=c.cases, recorded_at=at)
                for c in manifest.cohorts
            )
            frozen = FrozenExperiment(
                manifest=manifest,
                cohort_ids=tuple(
                    (c.name, f.content_hash())
                    for c, f in zip(manifest.cohorts, cohorts, strict=True)
                ),
                recorded_at=at,
                deadline_at=at + timedelta(seconds=manifest.budget.duration_seconds),
            )
            self.store.freeze_experiment(frozen, cohorts)
            return frozen

    def inspect(self, identity: str) -> FrozenExperiment:
        frozen = self.store.evaluation_artifact(identity, FrozenExperiment)
        self._validate_manifest(frozen.manifest, frozen.recorded_at)
        if (
            frozen.deadline_at
            != frozen.recorded_at + timedelta(seconds=frozen.manifest.budget.duration_seconds)
            or [n for n, _ in frozen.cohort_ids] != [c.name for c in frozen.manifest.cohorts]
            or self.store.experiment_for_manifest(frozen.manifest.content_hash()) != identity
        ):
            raise ValueError("experiment manifest/admission identity differs")
        for spec, (_, cohort_id) in zip(frozen.manifest.cohorts, frozen.cohort_ids, strict=True):
            cohort = self.store.evaluation_artifact(cohort_id, FrozenEvaluation)
            if cohort != FrozenEvaluation(
                plan=spec.plan, policy=spec.policy, cases=spec.cases, recorded_at=frozen.recorded_at
            ):
                raise ValueError("cohort differs from atomically frozen experiment")
        return frozen

    def run(self, identity: str) -> ExperimentReport:
        with experiment_lock(self.store):
            frozen = self.inspect(identity)
            service = DecisionService(
                self.store,
                evidence_store=self.evaluator.evidence,
                transport=self.transport,
                limits=frozen.manifest.budget.limits(),
            )
            for spec, (_, cohort_id) in zip(
                frozen.manifest.cohorts, frozen.cohort_ids, strict=True
            ):
                for case in spec.cases:
                    claim = self.store.evaluation_dispatch_for(case.request.cache_key())
                    if claim is not None:
                        self._dispatch(frozen, claim, cohort_id, case)
                        if (
                            self.store.find(
                                case.request.cache_key(),
                                observation_id=claim.observation_id,
                                successful_only=False,
                            )
                            is None
                        ):
                            # A pre-dispatch crash is intentionally indistinguishable
                            # from an unrecorded network outcome. Never resend.
                            return self._report(frozen)
                        continue
                    remaining = (frozen.deadline_at - datetime.now(timezone.utc)).total_seconds()
                    if remaining <= 0:
                        return self._report(frozen)
                    claim = EvaluationDispatch(
                        experiment_hash=identity,
                        cohort_id=cohort_id,
                        case_id=case.case_id,
                        request_hash=case.request.content_hash(),
                        cache_key=case.request.cache_key(),
                        observation_id=str(uuid4()),
                        recorded_at=datetime.now(timezone.utc),
                    )
                    self.store.claim_evaluation_dispatch(claim)
                    service.evaluate(
                        case.request,
                        mode="refresh",
                        timeout_seconds=min(remaining, frozen.manifest.budget.case_timeout_seconds),
                        _evaluation_dispatch_id=claim.content_hash(),
                    )
            return self._report(frozen)

    def _dispatch(self, frozen, claim, cohort_id, case):
        if (
            claim.experiment_hash != frozen.content_hash()
            or claim.cohort_id != cohort_id
            or claim.case_id != case.case_id
            or claim.request_hash != case.request.content_hash()
            or claim.cache_key != case.request.cache_key()
            or not frozen.recorded_at <= claim.recorded_at < frozen.deadline_at
        ):
            raise ValueError("case dispatch differs from frozen authority")

    def report(self, identity: str) -> ExperimentReport:
        with experiment_lock(self.store):
            return self._report(self.inspect(identity))

    def _report(self, frozen):
        # Score only whole cohorts with actual successful first observations.
        # Provider failures are retained as operational failures, never omitted
        # from a supposedly complete semantic evaluation population.
        receipts = []
        for spec, (_, cohort_id) in zip(frozen.manifest.cohorts, frozen.cohort_ids, strict=True):
            mapping = []
            for case in spec.cases:
                claim = self.store.evaluation_dispatch_for(case.request.cache_key())
                if claim is None:
                    break
                self._dispatch(frozen, claim, cohort_id, case)
                result = self.store.find(
                    case.request.cache_key(),
                    observation_id=claim.observation_id,
                    successful_only=False,
                )
                if result is None or result.status != DecisionStatus.OK:
                    break
                mapping.append((case.case_id, claim.observation_id))
            if len(mapping) == len(spec.cases):
                receipt = self.evaluator.score(cohort_id, observation_ids=tuple(mapping))
                receipts.append((spec.name, receipt.content_hash()))
        at = datetime.now(timezone.utc)
        report, dispatches = self._compose(frozen, tuple(receipts), at)
        value = ExperimentReport(
            experiment_hash=frozen.content_hash(),
            dispatch_ids=dispatches,
            cohort_receipts=tuple(receipts),
            report_json=canonical_json(report),
            observed_through=at,
            recorded_at=datetime.now(timezone.utc),
        )
        self.store._append_evaluation(value)
        return value

    def verify_report(self, identity: str) -> ExperimentReport:
        value = self.store.evaluation_artifact(identity, ExperimentReport)
        frozen = self.inspect(value.experiment_hash)
        if value.recorded_at < value.observed_through:
            raise ValueError("report creation precedes its observation cutoff")
        report, dispatches = self._compose(frozen, value.cohort_receipts, value.observed_through)
        if value.report_json != canonical_json(report) or value.dispatch_ids != dispatches:
            raise ValueError("experiment report differs from actual immutable authorities")
        return value

    def _compose(self, frozen, receipt_ids, at):
        if at < frozen.recorded_at or len(dict(receipt_ids)) != len(receipt_ids):
            raise ValueError("invalid experiment report chronology or receipt map")
        receipts = {}
        for name, identity in receipt_ids:
            receipt = self.store.evaluation_artifact(identity, EvaluationReceipt)
            _, metrics = self.evaluator._receipt(receipt)
            if receipt.cohort_id != dict(frozen.cohort_ids).get(name) or receipt.completed_at > at:
                raise ValueError("report receipt differs from exact frozen cohort/time")
            receipts[name] = (receipt, metrics)
        rows, dispatches, exports = [], [], []
        all_reservations, all_attempts = [], []
        for spec, (_, cohort_id) in zip(frozen.manifest.cohorts, frozen.cohort_ids, strict=True):
            for case in spec.cases:
                row = {
                    "cohort": spec.name,
                    "role": spec.role,
                    "case_id": case.case_id,
                    "request_hash": case.request.content_hash(),
                    "status": "not_dispatched",
                }
                claim = self.store.evaluation_dispatch_for(case.request.cache_key())
                if claim is not None and claim.recorded_at <= at:
                    self._dispatch(frozen, claim, cohort_id, case)
                    dispatches.append(claim.content_hash())
                    row.update(
                        dispatch_id=claim.content_hash(),
                        observation_id=claim.observation_id,
                        status="ambiguous",
                    )
                    result = self.store.find(
                        case.request.cache_key(),
                        observation_id=claim.observation_id,
                        successful_only=False,
                    )
                    attempts, reservations = self.store.evaluation_dispatch_usage(claim, at=at)
                    expected_tokens = len(canonical_json(case.request.wire()).encode()) + 1280
                    price = frozen.manifest.budget.input_usd_per_million
                    if (
                        [r["attempt"] for r in reservations]
                        != list(range(1, len(reservations) + 1))
                        or len(reservations) > frozen.manifest.budget.max_attempts
                        or any(
                            r["run_id"] != frozen.manifest.run_id
                            or r["reserved_tokens"] != expected_tokens
                            or r["reserved_cost"]
                            != (expected_tokens * price / 1_000_000 if price is not None else None)
                            or not claim.recorded_at.timestamp()
                            <= r["at"]
                            < frozen.deadline_at.timestamp()
                            for r in reservations
                        )
                    ):
                        raise ValueError(
                            "case reservations differ from frozen request/cap/price/clock"
                        )
                    all_reservations.extend(reservations)
                    all_attempts.extend(attempts)
                    row["usage_proof_hash"] = digest(
                        canonical_json({"attempts": attempts, "reservations": reservations})
                    )
                    if result is not None and result.completed_at <= at:
                        proof = self.store.export_observation(result.observation_id)
                        self._observation(frozen, case, claim, result, proof)
                        exports.append(proof)
                        row.update(
                            status=result.status.value,
                            observation_hash=digest(canonical_json(proof)),
                            latency_ms=result.latency_ms,
                            attempts=result.attempts,
                            requested_model=result.requested_model,
                            actual_model=result.actual_model,
                            started_at=result.started_at.isoformat(),
                            completed_at=result.completed_at.isoformat(),
                        )
                rows.append(row)
        admitted = self.store.evaluation_run_reservations(frozen.manifest.run_id, at=at)
        if admitted != sorted(all_reservations, key=lambda r: r["seq"]):
            raise ValueError("experiment report omits or substitutes admitted run usage")
        if (
            len(admitted) > frozen.manifest.budget.requests
            or sum(r["reserved_tokens"] for r in admitted) > frozen.manifest.budget.tokens
            or any(r["reserved_cost"] is None for r in admitted)
            or sum(r["reserved_cost"] for r in admitted) > frozen.manifest.budget.spend_usd
        ):
            raise ValueError("experiment durable reservations exceed frozen total caps")
        complete = all(row["status"] == "ok" for row in rows)
        if set(receipts) != {
            c.name
            for c in frozen.manifest.cohorts
            if all(r["status"] == "ok" for r in rows if r["cohort"] == c.name)
        }:
            raise ValueError(
                "report must retain every complete cohort and no partial success subset"
            )
        # Usage on a successful final retry does not account for earlier uncertain
        # attempts. Keep every conservative reservation and report unknown usage.
        known = [p["result"] for p in exports if p["result"]["input_tokens"] is not None]
        known_attempts = {
            (p["result"]["observation_id"], p["result"]["attempts"])
            for p in exports
            if p["result"]["input_tokens"] is not None
        }
        unknown = sum(
            (r["observation_id"], r["attempt"]) not in known_attempts for r in all_reservations
        )
        cost_known = all(v["cost_usd"] is not None for v in known)
        operating = {
            "reserved_requests": len(all_reservations),
            "reserved_tokens": sum(r["reserved_tokens"] for r in all_reservations),
            "reserved_cost_usd": sum(r["reserved_cost"] for r in all_reservations)
            if all(r["reserved_cost"] is not None for r in all_reservations)
            else None,
            "known_input_tokens": sum(r["input_tokens"] for r in known),
            "known_output_tokens": sum(r["output_tokens"] for r in known),
            "known_cost_usd": sum(r["cost_usd"] for r in known) if cost_known else None,
            "unknown_usage_reservations": unknown,
            "total_cost_usd": sum(r["cost_usd"] for r in known)
            if cost_known and not unknown
            else None,
            "pricing_version": frozen.manifest.budget.pricing_version,
            "pricing_authority": "manifest declaration; not a provider invoice",
            "provider_attempts_recorded": len(all_attempts),
            "observation_latency_ms": sum(p["result"]["latency_ms"] for p in exports),
            "elapsed_since_freeze_seconds": (at - frozen.recorded_at).total_seconds(),
            "generative_calls_saved": None,
            "first_observation_started_at": min(p["result"]["started_at"] for p in exports)
            if exports
            else None,
            "last_observation_completed_at": max(p["result"]["completed_at"] for p in exports)
            if exports
            else None,
        }
        pairs = self._pair_reports(frozen, receipts)
        priors = self._prior_reports(frozen, receipts, at)
        represented = {p["kind"] for p in pairs if p["status"] == "measured"}
        report = {
            "experiment_hash": frozen.content_hash(),
            "manifest_hash": frozen.manifest.content_hash(),
            "status": "complete"
            if complete
            else ("ambiguous" if any(r["status"] == "ambiguous" for r in rows) else "incomplete"),
            "deadline_expired": at >= frozen.deadline_at,
            "cases": rows,
            "operating": operating,
            "cohorts": {
                c.name: {
                    "role": c.role,
                    "split_id": c.plan.split_id,
                    "plan_hash": c.plan.content_hash(),
                    "dataset_hash": c.plan.dataset_hash,
                    "question_hash": c.plan.question_hash,
                    "model": c.policy.model,
                    "policy_hash": c.policy.content_hash(),
                    "receipt_id": dict(receipt_ids)[c.name],
                    "metrics": receipts[c.name][1],
                }
                for c in frozen.manifest.cohorts
                if c.name in receipts
            },
            "pairs": pairs,
            "prior_baselines": priors,
            "calibrations": self._calibration_reports(frozen, receipts, at),
            "acceptance": {
                "publication_performed": False,
                "release_status": "unverified",
                "roles_present": sorted({c.role for c in frozen.manifest.cohorts}),
                "missing_roles": sorted(
                    {"selection", "calibration", "test"} - {c.role for c in frozen.manifest.cohorts}
                ),
                "missing_stressor_pairs": sorted(set(STRESSORS) - represented),
                "probability_calibration": "raw proper scores/reliability; optional disjoint temperature fit is descriptive, never analyst conviction or financial-return confidence",
                "stressor_semantics": "exact transformations and archived human annotation; semantic sufficiency requires review",
                "outstanding": [
                    "reviewed financial cohort/source/label sufficiency for J1-J4/J6 and proposed J7",
                    "authenticated provider and independently reviewed efficacy acceptance",
                    "matched incumbent heuristic/generative baselines and measured calls saved",
                    "J5 six-month causal walk-forward/common-registry/holdout/cost evidence",
                    "prospective with/without Jev and quant-only/quant-plus-veto/full-agent controls",
                    "three-month paper observation and common release gates",
                ],
            },
        }
        return report, tuple(dispatches)

    def _observation(self, frozen, case, claim, result, proof):
        first = self.store.find(case.request.cache_key(), successful_only=False)
        started = self.store.evaluation_dispatch_started_at(claim.content_hash())
        if (
            proof["request"] != case.request.model_dump(mode="json")
            or result.started_at < claim.recorded_at
            or result.replayed
            or result.cache_hit
            or result.reused_observation_id
            or first is None
            or first.observation_id != claim.observation_id
            or result.requested_model != case.request.policy.model
            or started is None
            or not claim.recorded_at <= started <= result.completed_at
            or (
                result.input_tokens is None
                and (result.output_tokens is not None or result.cost_usd is not None)
            )
        ):
            raise ValueError("experiment observation differs from exact original dispatch")
        if result.input_tokens is not None:
            model, judgments, tokens_in, tokens_out = parse_response(
                case.request, result.raw_response
            )
            price = frozen.manifest.budget.input_usd_per_million
            if (
                model != result.actual_model
                or judgments != result.judgments
                and result.status == DecisionStatus.OK
                or tokens_in != result.input_tokens
                or tokens_out != result.output_tokens
                or result.cost_usd != (tokens_in * price / 1_000_000 if price is not None else None)
                or result.pricing_version
                != (frozen.manifest.budget.pricing_version if price is not None else None)
            ):
                raise ValueError(
                    "experiment usage/model differs from raw response and frozen price"
                )

    def _pair_reports(self, frozen, receipts):
        values = []
        for pair in frozen.manifest.pairs:
            value = {
                "name": pair.name,
                "kind": pair.kind,
                "pair_hash": pair.content_hash(),
                "status": "unavailable",
            }
            if pair.base in receipts and pair.variant in receipts:
                a, b = receipts[pair.base][0], receipts[pair.variant][0]
                ap, bp = (
                    {p.case_id: p for p in a.predictions},
                    {p.case_id: p for p in b.predictions},
                )
                comparisons = [
                    {
                        "base": x,
                        "variant": y,
                        "base_observation": ap[x].observation_id,
                        "variant_observation": bp[y].observation_id,
                        "same_gold": ap[x].gold == bp[y].gold,
                        "same_prediction": ap[x].prediction == bp[y].prediction,
                        "both_correct": ap[x].prediction == ap[x].gold
                        and bp[y].prediction == bp[y].gold,
                    }
                    for x, y in pair.cases
                ]
                value.update(
                    status="measured",
                    comparisons=comparisons,
                    agreement=sum(c["same_prediction"] for c in comparisons) / len(comparisons),
                    both_correct_rate=sum(c["both_correct"] for c in comparisons)
                    / len(comparisons),
                )
                if pair.kind == "model-baseline":
                    value["brier_change"] = (
                        receipts[pair.variant][1]["brier_score"]
                        - receipts[pair.base][1]["brier_score"]
                    )
                    value["macro_recall_change"] = (
                        receipts[pair.variant][1]["macro_recall"]
                        - receipts[pair.base][1]["macro_recall"]
                    )
            values.append(value)
        return values

    def _calibration_reports(self, frozen, receipts, at):
        cohorts = {c.name: c for c in frozen.manifest.cohorts}
        values = []
        for spec in frozen.manifest.calibrations:
            value = {"name": spec.name, "algorithm": spec.algorithm, "status": "unavailable"}
            if spec.calibration not in receipts or spec.test not in receipts:
                values.append(value)
                continue
            training, training_metrics = receipts[spec.calibration]
            test, test_metrics = receipts[spec.test]
            if training_metrics["missing_classes"] or test_metrics["missing_classes"]:
                value["reason"] = "every declared class requires calibration and test support"
                values.append(value)
                continue
            scores = []
            for temperature in spec.temperatures:
                loss = -sum(
                    math.log(max(dict(_temperature(p, temperature))[p.gold], 1e-15))
                    for p in training.predictions
                ) / len(training.predictions)
                scores.append((temperature, loss))
            # Frozen sorted grid gives a deterministic smallest-temperature tie break.
            chosen = min(scores, key=lambda v: (v[1], v[0]))[0]
            predictions = []
            for prediction in test.predictions:
                probs = _temperature(prediction, chosen)
                predictions.append(
                    prediction.model_copy(
                        update={
                            "probabilities": tuple(
                                ProbabilityValue(option=k, probability=v) for k, v in probs
                            ),
                            "ordinal_score": sum(int(k) * v for k, v in probs)
                            if prediction.ordinal_score is not None
                            else None,
                        }
                    )
                )
            fitted = {
                "spec_hash": spec.content_hash(),
                "calibration_receipt_id": training.content_hash(),
                "calibration_losses": scores,
                "temperature": chosen,
            }
            value.update(
                status="measured",
                fit=fitted,
                fit_hash=digest(canonical_json(fitted)),
                test_receipt_id=test.content_hash(),
                decision_rule="original prediction/abstention unchanged; transforms task probabilities only",
                availability="derived fit/metrics available at enclosing report.recorded_at; original observation clocks retained",
                metrics=evaluate_classification(cohorts[spec.test].plan, tuple(predictions)),
            )
            values.append(value)
        return values

    def _prior_reports(self, frozen, receipts, at):
        cohorts = {c.name: c for c in frozen.manifest.cohorts}
        values = []
        for baseline in frozen.manifest.priors:
            train, test = cohorts[baseline.training], cohorts[baseline.test]
            counts = Counter(c.gold for c in train.cases)
            support = {label: counts[label] for label in train.plan.labels}
            value = {
                "name": baseline.name,
                "algorithm": baseline.algorithm,
                "training_cohort_id": dict(frozen.cohort_ids)[baseline.training],
                "training_support": support,
                "status": "unavailable",
                "provider_requests": 0,
                "generative_calls_saved": None,
            }
            if baseline.test in receipts and all(support.values()):
                probs = tuple(
                    ProbabilityValue(option=k, probability=v / len(train.cases))
                    for k, v in support.items()
                )
                top = max(support, key=support.get)
                selected = top if list(support.values()).count(support[top]) == 1 else None
                receipt = receipts[baseline.test][0]
                predictions = tuple(
                    LabeledPrediction(
                        case_id=p.case_id,
                        source_hash=p.source_hash,
                        gold=p.gold,
                        prediction=selected,
                        probabilities=probs,
                        material=p.material,
                        gold_blocked=p.gold_blocked,
                        blocked=selected in test.plan.blocking_labels,
                        variant_group=p.variant_group,
                        stressor=p.stressor,
                        observation_id="derived-prior:" + baseline.content_hash() + ":" + p.case_id,
                        observed_at=p.observed_at,
                        ordinal_score=sum(int(k) * v / len(train.cases) for k, v in support.items())
                        if test.cases[0].request.questions[0].kind == "score"
                        else None,
                    )
                    for p in receipt.predictions
                )
                value.update(
                    status="measured",
                    predictions=[p.model_dump(mode="json") for p in predictions],
                    comparison=compare_classifiers(test.plan, receipt.predictions, predictions),
                    availability="derived prior predictions available at enclosing report.recorded_at; observed_at retains paired case observation clock",
                )
            values.append(value)
        return values


def _temperature(prediction, value):
    logs = [
        (p.option, math.log(max(p.probability, 1e-15)) / value) for p in prediction.probabilities
    ]
    largest = max(v for _, v in logs)
    weights = [(k, math.exp(v - largest)) for k, v in logs]
    total = sum(v for _, v in weights)
    return tuple((k, v / total) for k, v in weights)
