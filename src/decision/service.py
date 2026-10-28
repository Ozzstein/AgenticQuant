"""Provider-backed typed decisions with replay, bounded operating costs and failures."""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import Field

from src.decision.store import DecisionStore
from src.decision.transport import JevHTTPTransport
from src.domain.common import VersionedContract
from src.domain.decisions import (
    DecisionRequest,
    DecisionResult,
    DecisionStatus,
    Judgment,
    ProbabilityValue,
    canonical_json,
)


class OperatingLimits(VersionedContract):
    run_requests: int = Field(default=200, gt=0)
    day_requests: int = Field(default=2000, gt=0)
    run_tokens: int = Field(default=2_000_000, gt=0)
    day_tokens: int = Field(default=20_000_000, gt=0)
    run_spend_usd: float | None = Field(default=0.10, ge=0)
    day_spend_usd: float | None = Field(default=1.00, ge=0)
    max_request_bytes: int = Field(default=24_000, gt=0, le=100_000)
    max_attempts: int = Field(default=3, ge=1, le=5)
    attempt_timeout_seconds: float = Field(default=10, gt=0)
    min_interval_seconds: float = Field(default=0.1, ge=0)
    input_usd_per_million: float | None = Field(default=0.042, ge=0)
    pricing_version: str | None = "typesafe-jev-1.14.0-2026-10-27"


class ModelVersionMismatch(ValueError):
    pass


def _number(value, *, low=0, high=1):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("expected finite numeric value")
    if not low <= value <= high:
        raise ValueError("value outside permitted range")
    return float(value)


def parse_response(
    request: DecisionRequest, raw: str
) -> tuple[str, tuple[Judgment, ...], int, int]:
    """Validate exact question keys, types, distributions, expected score and model."""

    def bad_constant(_):
        raise ValueError("non-finite JSON value")

    data = json.loads(raw, parse_constant=bad_constant)
    if not isinstance(data, dict) or set(data) != {"model", "answers", "usage"}:
        raise ValueError("unexpected response envelope")
    if not isinstance(data["model"], str):
        raise ValueError("model identity must be a string")
    if data["model"] != request.policy.model:
        raise ModelVersionMismatch("provider returned a different model version")
    answers, usage = data["answers"], data["usage"]
    if not isinstance(answers, dict) or set(answers) != {q.question_id for q in request.questions}:
        raise ValueError("answer IDs do not match request")
    if not isinstance(usage, dict) or set(usage) != {"input_tokens", "output_tokens"}:
        raise ValueError("usage missing or malformed")
    if any(type(v) is not int or v < 0 for v in usage.values()):
        raise ValueError("token usage must be nonnegative integers")
    thresholds = {t.question_hash: t for t in request.policy.thresholds}
    judgments = []
    for q in request.questions:
        answer = answers[q.question_id]
        if not isinstance(answer, dict) or answer.get("type") != q.kind:
            raise ValueError("answer type mismatch")
        kwargs = dict(question_id=q.question_id, question_hash=q.content_hash(), kind=q.kind)
        threshold = thresholds.get(q.content_hash())
        abstain = False
        if q.kind == "noul":
            if set(answer) != {"type", "noul"}:
                raise ValueError("unexpected Noul fields")
            p = _number(answer["noul"])
            kwargs.update(yes_probability=p)
            top = max(p, 1 - p)
            abstain = top == 0.5
            # Noul has no provider confidence; never manufacture one.
            if threshold and threshold.min_provider_confidence is not None:
                abstain = True
        else:
            fields = {"type", "confidence", "probabilities", q.kind}
            if q.kind == "score":
                fields.add("legend")
            if set(answer) != fields:
                raise ValueError("unexpected Choice/Score fields")
            probabilities = answer["probabilities"]
            if not isinstance(probabilities, dict) or set(probabilities) != {
                c.key for c in q.criteria
            }:
                raise ValueError("probability options differ from criteria")
            probs = {k: _number(v) for k, v in probabilities.items()}
            if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5):
                raise ValueError("probabilities must sum to one")
            confidence = _number(answer["confidence"])
            kwargs.update(
                provider_confidence=confidence,
                probabilities=tuple(
                    ProbabilityValue(option=c.key, probability=probs[c.key]) for c in q.criteria
                ),
            )
            top = max(probs.values())
            if q.kind == "choice":
                choice = answer["choice"]
                if choice not in probs or probs[choice] != top:
                    raise ValueError("selected Choice must maximize probability")
                kwargs.update(
                    choice=choice,
                    confidence_definition="typesafe-choice-distribution-concentration-v1",
                )
                abstain = choice in q.unknown_options or sum(v == top for v in probs.values()) > 1
            else:
                if answer["legend"] != {c.key: c.description for c in q.criteria}:
                    raise ValueError("returned Score rubric differs from request")
                score = _number(answer["score"], high=len(q.criteria) - 1)
                expected = sum(int(k) * p for k, p in probs.items())
                if not math.isclose(score, expected, abs_tol=1e-5):
                    raise ValueError("Score is not the distribution's expected value")
                kwargs.update(
                    score=score, confidence_definition="typesafe-score-ordinal-distribution-v1"
                )
            if threshold and threshold.min_provider_confidence is not None:
                abstain |= confidence < threshold.min_provider_confidence
        if threshold:
            abstain |= top < threshold.min_probability
        # Wire validation has no evaluation authority. Only the service may
        # promote an observation after resolving the exact published proof.
        kwargs["disposition"] = "abstain" if abstain else "observed"
        judgments.append(Judgment(**kwargs))
    return data["model"], tuple(judgments), usage["input_tokens"], usage["output_tokens"]


class DecisionService:
    """One service shared by graph, research and CLI; no trading capabilities.

    Cache keys include source, question, ordering, policy, model, as-of and state.
    Replay may specify an exact observation ID and never accesses the network.
    Refresh always appends a new observation; it cannot rewrite prior evidence.
    """

    def __init__(self, store: DecisionStore, *, transport=None, limits=None, evidence_store=None):
        self.store = store
        self.transport = transport if transport is not None else JevHTTPTransport()
        self.limits = limits or OperatingLimits()
        self.evidence_store = evidence_store

    def evaluate(self, request: DecisionRequest, **kwargs) -> DecisionResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aevaluate(request, **kwargs))
        raise RuntimeError("use await service.aevaluate() inside an event loop")

    async def aevaluate(
        self,
        request: DecisionRequest,
        *,
        timeout_seconds: float = 30,
        mode: Literal["cache", "refresh", "replay"] = "cache",
        observation_id: str | None = None,
        _evaluation_dispatch_id: str | None = None,
    ) -> DecisionResult:
        if mode not in ("cache", "refresh", "replay"):
            raise ValueError("invalid observation mode")
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("deadline must be finite and nonnegative")
        if observation_id and mode != "replay":
            raise ValueError("observation_id is only meaningful in recorded replay")
        started, clock = datetime.now(timezone.utc), time.monotonic()
        deadline = clock + timeout_seconds
        if _evaluation_dispatch_id is not None:
            if mode != "refresh" or observation_id is not None:
                raise ValueError("evaluation dispatch requires a fresh exact observation")
            identity, experiment_remaining = self.store.start_evaluation_dispatch(
                _evaluation_dispatch_id, request
            )
            deadline = min(deadline, time.monotonic() + max(0, experiment_remaining))
        else:
            identity = str(uuid4())
        attempts, raw, actual, input_tokens, output_tokens = 0, None, None, None, None
        evaluated = False

        def finish(status, reason="", *, judgments=(), reused=None):
            if evaluated and status == DecisionStatus.OK:
                judgments = tuple(
                    j.model_copy(update={"disposition": "accepted"})
                    if j.disposition == "observed"
                    else j
                    for j in judgments
                )
            cost = None
            price = self.limits.input_usd_per_million
            if input_tokens is not None and price is not None and actual == request.policy.model:
                cost = input_tokens * price / 1_000_000
            result = DecisionResult(
                observation_id=identity,
                request_hash=request.content_hash(),
                cache_key=request.cache_key(),
                run_id=request.context.run_id,
                status=status,
                reason=reason,
                requested_model=request.policy.model,
                actual_model=actual,
                policy_hash=request.policy.content_hash(),
                state_hash=request.state_hash,
                source_hashes=tuple(s.content_hash() for s in request.sources),
                judgments=judgments,
                started_at=started,
                completed_at=datetime.now(timezone.utc),
                latency_ms=(time.monotonic() - clock) * 1000,
                attempts=attempts,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                pricing_version=self.limits.pricing_version if price is not None else None,
                raw_response=raw,
                cache_hit=reused is not None and mode == "cache",
                replayed=mode == "replay",
                reused_observation_id=reused,
            )
            self.store.append(request, result)
            return result

        if request.policy.activation == "evaluated":
            from src.decision.evaluation import EvaluationService

            try:
                authority = EvaluationService(self.store, evidence_store=self.evidence_store)
                authority.resolve(
                    request.policy,
                    question_hashes=tuple(q.content_hash() for q in request.questions),
                    as_of=request.context.as_of,
                )
                for source in request.sources:
                    authority._source(source, request.context.as_of)
                evaluated = True
            except Exception:
                return finish(
                    DecisionStatus.UNAVAILABLE, "evaluated policy proof unavailable or ineligible"
                )
            if time.monotonic() >= deadline:
                return finish(
                    DecisionStatus.TIMED_OUT, "deadline expired while resolving evaluation proof"
                )

        if mode in ("cache", "replay"):
            cached = self.store.find(
                request.cache_key(), observation_id=observation_id, successful_only=mode == "cache"
            )
            if cached:
                if evaluated:
                    try:
                        exported = self.store.export_observation(cached.observation_id)
                        original = DecisionRequest.model_validate(exported["request"])
                        _, parsed, _, _ = parse_response(original, cached.raw_response)
                        normalized = tuple(
                            j.model_copy(update={"disposition": "observed"})
                            if j.disposition == "accepted"
                            else j
                            for j in cached.judgments
                        )
                        if parsed != normalized or cached.actual_model != request.policy.model:
                            raise ValueError(
                                "cached judgment differs from raw provider observation"
                            )
                    except Exception:
                        return finish(
                            DecisionStatus.MALFORMED, "evaluated cached observation proof differs"
                        )
                actual, raw = cached.actual_model, cached.raw_response
                # Reuse preserves original usage on its observation, not a new charge.
                return finish(
                    cached.status,
                    cached.reason,
                    judgments=cached.judgments,
                    reused=cached.observation_id,
                )
            if mode == "replay":
                return finish(DecisionStatus.REPLAY_MISS, "no compatible recorded observation")
        if not request.sources:
            return finish(DecisionStatus.INSUFFICIENT, "no immutable evidence spans")
        for source in request.sources:
            try:
                source.record.require_available(request.context.as_of)
                if (
                    source.record.instrument_ids
                    and request.instrument.instrument_id not in source.record.instrument_ids
                ):
                    raise ValueError("source belongs to a different instrument")
            except ValueError:
                return finish(
                    DecisionStatus.STALE
                    if source.record.quality == "stale"
                    else DecisionStatus.INSUFFICIENT,
                    "source unavailable or ineligible at as_of",
                )
            age = request.policy.max_source_age_seconds
            if (
                age is not None
                and (request.context.as_of - source.record.available_at).total_seconds() > age
            ):
                return finish(DecisionStatus.STALE, "source exceeds question freshness policy")
        payload = request.wire()
        size = len(canonical_json(payload).encode())
        if size > self.limits.max_request_bytes:
            return finish(DecisionStatus.BUDGET, "bounded context byte limit exceeded")
        if not self.transport.available:
            return finish(DecisionStatus.UNAVAILABLE, "TYPESAFE_API_KEY is not configured")
        # Conservative UTF-8 byte reservation plus response/envelope allowance.
        # Provider usage is verified too; reservations are never refunded on uncertainty.
        reserve_tokens = size + 1024 + len(request.questions) * 256
        price = self.limits.input_usd_per_million
        reserve_cost = reserve_tokens * price / 1_000_000 if price is not None else None
        try:
            while attempts < self.limits.max_attempts:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return finish(DecisionStatus.TIMED_OUT, "caller deadline expired")
                now = datetime.now(timezone.utc)
                allowed, reason, wait = self.store.reserve(
                    run_id=request.context.run_id,
                    day=now.date().isoformat(),
                    tokens=reserve_tokens,
                    cost=reserve_cost,
                    now=now.timestamp(),
                    limits=self.limits,
                    observation_id=identity,
                    attempt=attempts + 1,
                )
                if not allowed:
                    if wait and wait < remaining:
                        await asyncio.sleep(wait)
                        continue
                    return finish(
                        DecisionStatus.TIMED_OUT if wait else DecisionStatus.BUDGET, reason
                    )
                attempts += 1
                attempt_clock = time.monotonic()
                remaining = deadline - attempt_clock
                if remaining <= 0:
                    return finish(DecisionStatus.TIMED_OUT, "deadline expired during admission")
                retry_delay = min(0.25 * 2 ** (attempts - 1), 2)
                try:
                    timeout = min(remaining, self.limits.attempt_timeout_seconds)
                    response = await asyncio.wait_for(
                        self.transport.post(payload, timeout), timeout
                    )
                    raw = response.body
                    self.store.attempt(
                        identity,
                        attempts,
                        response.status,
                        raw,
                        (time.monotonic() - attempt_clock) * 1000,
                    )
                    if response.status == 200:
                        try:
                            data = json.loads(raw)
                            actual = data.get("model") if isinstance(data, dict) else None
                            if not isinstance(actual, str):
                                actual = None
                            actual, judgments, input_tokens, output_tokens = parse_response(
                                request, raw
                            )
                        except ModelVersionMismatch:
                            return finish(
                                DecisionStatus.VERSION_MISMATCH,
                                "response model differs from pinned policy",
                            )
                        except (ValueError, TypeError, KeyError, OverflowError):
                            return finish(
                                DecisionStatus.MALFORMED,
                                "provider response violates typed contract",
                            )
                        if input_tokens + output_tokens > reserve_tokens:
                            return finish(
                                DecisionStatus.BUDGET,
                                "provider usage exceeds conservative reservation",
                            )
                        return finish(DecisionStatus.OK, judgments=judgments)
                    if response.status not in (408, 429, 500, 502, 503, 504):
                        return finish(
                            DecisionStatus.MALFORMED
                            if response.status in (400, 422)
                            else DecisionStatus.UNAVAILABLE,
                            f"provider HTTP {response.status}",
                        )
                    if response.retry_after:
                        try:
                            retry_delay = max(retry_delay, float(response.retry_after))
                        except ValueError:
                            try:
                                retry_delay = max(
                                    retry_delay,
                                    (
                                        parsedate_to_datetime(response.retry_after) - now
                                    ).total_seconds(),
                                )
                            except (ValueError, TypeError):
                                pass
                except (TimeoutError, httpx.TimeoutException):
                    self.store.attempt(
                        identity,
                        attempts,
                        None,
                        None,
                        (time.monotonic() - attempt_clock) * 1000,
                        "timeout",
                    )
                    if attempts == self.limits.max_attempts or time.monotonic() >= deadline:
                        return finish(DecisionStatus.TIMED_OUT, "provider deadline expired")
                except httpx.TransportError:
                    self.store.attempt(
                        identity,
                        attempts,
                        None,
                        None,
                        (time.monotonic() - attempt_clock) * 1000,
                        "transport_failure",
                    )
                except (ValueError, UnicodeError):
                    return finish(DecisionStatus.MALFORMED, "invalid or oversized response body")
                if attempts == self.limits.max_attempts:
                    return finish(
                        DecisionStatus.UNAVAILABLE, "transient provider failure; retries exhausted"
                    )
                if not math.isfinite(retry_delay) or retry_delay >= deadline - time.monotonic():
                    return finish(DecisionStatus.TIMED_OUT, "retry would exceed caller deadline")
                await asyncio.sleep(retry_delay)
        except asyncio.CancelledError:
            finish(DecisionStatus.CANCELLED, "caller cancelled provider evaluation")
            raise
