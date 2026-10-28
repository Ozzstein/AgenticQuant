"""Actual SQLite admission/receipt boundaries; HTTP bodies are test-only fixtures."""

# ruff: noqa: F811
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import httpx
import pytest

from src.decision.experiments import ExperimentService
from src.decision.service import DecisionService, OperatingLimits
from src.decision.store import DecisionStore, read_usage_summary
from src.decision.transport import JevHTTPTransport
from src.domain.decisions import DecisionStatus, canonical_json, digest
from tests.test_decision_contracts import (  # noqa: F401
    decision_context,
    decision_instrument,
    decision_question,
    decision_request,
    decision_source,
)
from tests.test_decision_experiments import experiment  # noqa: F401
from tests.test_decision_service import response_for
from tests.test_evaluated_policy import evaluation_env  # noqa: F401


def limits(**changes):
    return OperatingLimits(
        min_interval_seconds=0,
        run_tokens=10_000,
        day_tokens=10_000,
        run_spend_usd=0.01,
        day_spend_usd=0.01,
        input_usd_per_million=1.0,
        pricing_version="synthetic-test-rate",
        **changes,
    )


def service(path, handler, caps=None):
    return DecisionService(
        DecisionStore(path),
        limits=caps or limits(),
        transport=JevHTTPTransport("fixture-secret", transport=httpx.MockTransport(handler)),
    )


def summary(path, run_id):
    now = datetime.now(timezone.utc)
    return read_usage_summary(path, day=now.date().isoformat(), run_id=run_id, as_of=now)


def other_run(request):
    return request.model_copy(
        update={"context": request.context.model_copy(update={"run_id": "other-run"})}
    )


@pytest.mark.parametrize("new_run", [False, True])
def test_overrun_fences_restart_run_and_day(tmp_path, decision_request, new_run):
    seen = []

    def handler(_):
        seen.append(1)
        return httpx.Response(
            200,
            json=response_for(
                decision_request, usage={"input_tokens": 10_001, "output_tokens": 20}
            ),
        )

    path = tmp_path / "decisions.sqlite"
    first = service(path, handler).evaluate(decision_request, mode="refresh")
    assert first.status == DecisionStatus.BUDGET and first.cost_usd == pytest.approx(0.010001)
    restarted = service(path, handler)
    assert (
        restarted.evaluate(
            other_run(decision_request) if new_run else decision_request, mode="refresh"
        ).status
        == DecisionStatus.BUDGET
    )
    assert len(seen) == 1
    proof = restarted.store.export_observation(first.observation_id)
    usage = proof["usage_authority"][0]["usage"]
    assert usage["source_hashes"] == [s.content_hash() for s in decision_request.sources]
    assert usage["request_hash"] == decision_request.content_hash()
    assert usage["charged_tokens"] == 10_021
    assert set(usage["breaches"]) == {"token_reservation_exceeded", "cost_reservation_exceeded"}
    view = summary(path, decision_request.context.run_id)
    assert view["day_usage"]["breached"] and view["run_usage"]["breached"]
    assert view["day_usage"]["total_cost_usd"] == pytest.approx(0.010001)
    assert view["day_usage"]["unknown_usage_reservations"] == 0


@pytest.mark.parametrize(
    "change,status",
    [
        ("answers", DecisionStatus.MALFORMED),
        ("wrong-model", DecisionStatus.VERSION_MISMATCH),
        ("missing-model", DecisionStatus.MALFORMED),
        ("missing-output", DecisionStatus.MALFORMED),
        ("missing-input", DecisionStatus.MALFORMED),
        ("retryable-http", DecisionStatus.BUDGET),
    ],
)
def test_invalid_semantics_still_charge_reported_usage(tmp_path, decision_request, change, status):
    value = response_for(decision_request, usage={"input_tokens": 10_001, "output_tokens": 10_002})
    if change == "answers":
        value["answers"] = {}
    if change == "wrong-model":
        value["model"] = "jev-1.13.0"
    if change == "missing-model":
        del value["model"]
    if change == "missing-output":
        del value["usage"]["output_tokens"]
    if change == "missing-input":
        del value["usage"]["input_tokens"]
    seen = []

    def handler(_):
        seen.append(1)
        return httpx.Response(503 if change == "retryable-http" else 200, json=value)

    path = tmp_path / "decisions.sqlite"
    svc = service(path, handler)
    result = svc.evaluate(decision_request, mode="refresh")
    assert result.status == status
    assert svc.evaluate(other_run(decision_request), mode="refresh").status == DecisionStatus.BUDGET
    assert len(seen) == 1
    svc.store.export_observation(result.observation_id)
    v = summary(path, decision_request.context.run_id)["run_usage"]
    assert v["breached"] and v["conservative_tokens"] >= 10_001
    assert v["known_input_tokens"] == (0 if change == "missing-input" else 10_001)
    if change in {"wrong-model", "missing-model", "missing-input", "missing-output"}:
        assert result.cost_usd is None and v["total_cost_usd"] is None
        assert v["unknown_usage_reservations"] == 1
    else:
        assert v["total_cost_usd"] == pytest.approx(0.010001)


@pytest.mark.parametrize("known_first", [False, True])
def test_retries_never_refund_uncertainty_or_omit_known_attempt(
    tmp_path, decision_request, known_first
):
    seen = []

    def handler(_):
        seen.append(1)
        if len(seen) == 1:
            return httpx.Response(
                503,
                json=response_for(
                    decision_request, usage={"input_tokens": 200, "output_tokens": 30}
                )
                if known_first
                else {"error": "uncertain"},
            )
        return httpx.Response(200, json=response_for(decision_request))

    path = tmp_path / "decisions.sqlite"
    svc = service(path, handler)
    result = svc.evaluate(decision_request, mode="refresh")
    assert result.status == DecisionStatus.OK and len(seen) == 2
    proof = svc.store.export_observation(result.observation_id)
    v = summary(path, decision_request.context.run_id)["run_usage"]
    assert v["known_input_tokens"] == (300 if known_first else 100)
    assert v["unknown_usage_reservations"] == (0 if known_first else 1)
    assert v["conservative_tokens"] == sum(r["reserved_tokens"] for r in proof["reservations"])
    assert v["total_cost_usd"] == (pytest.approx(0.0003) if known_first else None)
    assert v["known_cost_usd"] == pytest.approx(0.0003 if known_first else 0.0001)


def reserve(svc, request, identity, at):
    tokens = len(canonical_json(request.wire()).encode()) + 1280
    return svc.store.reserve(
        run_id=request.context.run_id,
        day=at.date().isoformat(),
        now=at.timestamp(),
        tokens=tokens,
        cost=tokens / 1_000_000,
        limits=svc.limits,
        observation_id=identity,
        attempt=1,
        request=request,
    )


def test_original_utc_day_owns_late_response_and_run_fence(tmp_path, decision_request):
    path = tmp_path / "decisions.sqlite"
    svc = service(path, lambda _: httpx.Response(200, json=response_for(decision_request)))
    now = datetime.now(timezone.utc)
    yesterday = now.replace(hour=23, minute=59, second=59) - timedelta(days=1)
    assert reserve(svc, decision_request, "late", yesterday)[0]
    usage = svc.store.attempt(
        "late",
        1,
        200,
        canonical_json(
            response_for(decision_request, usage={"input_tokens": 10_001, "output_tokens": 20})
        ),
        1,
    )
    assert usage.recorded_at > yesterday
    assert svc.store.budget_fences(None, yesterday.date().isoformat())
    assert svc.evaluate(decision_request, mode="refresh").status == DecisionStatus.BUDGET
    assert svc.evaluate(other_run(decision_request), mode="refresh").status == DecisionStatus.OK
    original_day = read_usage_summary(
        path, day=yesterday.date().isoformat(), as_of=datetime.now(timezone.utc)
    )
    assert original_day["day_usage"]["requests"] == 1 and original_day["day_usage"]["breached"]
    before = read_usage_summary(
        path, day=yesterday.date().isoformat(), as_of=yesterday + timedelta(microseconds=1)
    )
    assert not before["day_usage"]["breached"] and before["day_usage"]["total_cost_usd"] is None


def test_attempt_and_fence_commit_before_concurrent_admission(
    tmp_path, decision_request, monkeypatch
):
    path = tmp_path / "decisions.sqlite"
    svc = service(
        path,
        lambda _: httpx.Response(
            200,
            json=response_for(
                decision_request, usage={"input_tokens": 10_001, "output_tokens": 20}
            ),
        ),
    )
    restarted = service(path, lambda _: pytest.fail("post-breach transport"))
    in_transaction, release, admission_started = Event(), Event(), Event()
    original = DecisionStore._record

    def pause(db, table, identity, value):
        if table == "attempt_usage":
            in_transaction.set()
            assert release.wait(10)
        return original(db, table, identity, value)

    monkeypatch.setattr(DecisionStore, "_record", staticmethod(pause))

    def second():
        admission_started.set()
        return restarted.evaluate(other_run(decision_request), mode="refresh")

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(svc.evaluate, decision_request, mode="refresh")
        assert in_transaction.wait(10)
        following = pool.submit(second)
        assert admission_started.wait(10)
        release.set()
        assert first.result().status == following.result().status == DecisionStatus.BUDGET
    assert summary(path, decision_request.context.run_id)["day_usage"]["requests"] == 1


def test_already_dispatched_work_retains_actual_usage_after_fence(tmp_path, decision_request):
    first_admitted, second_admitted, breached = Event(), Event(), Event()
    path = tmp_path / "decisions.sqlite"

    def first_handler(_):
        first_admitted.set()
        assert second_admitted.wait(10)
        return httpx.Response(
            200,
            json=response_for(
                decision_request, usage={"input_tokens": 10_001, "output_tokens": 20}
            ),
        )

    def second_handler(_):
        second_admitted.set()
        assert breached.wait(10)
        return httpx.Response(200, json=response_for(decision_request))

    first_svc, second_svc = service(path, first_handler), service(path, second_handler)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(first_svc.evaluate, decision_request, mode="refresh")
        assert first_admitted.wait(10)
        second = pool.submit(second_svc.evaluate, other_run(decision_request), mode="refresh")
        assert first.result().status == DecisionStatus.BUDGET
        breached.set()
        second_result = second.result()
    assert (
        second_result.status == DecisionStatus.OK
    )  # It was really dispatched; never fake a cancellation.
    second_svc.store.export_observation(second_result.observation_id)
    v = summary(path, decision_request.context.run_id)["day_usage"]
    assert v["requests"] == 2 and v["known_input_tokens"] == 10_101
    assert v["total_cost_usd"] == pytest.approx(0.010101)
    assert (
        second_svc.evaluate(other_run(decision_request), mode="refresh").status
        == DecisionStatus.BUDGET
    )


@pytest.mark.parametrize("table", ["reservation_authorities", "attempt_usage"])
def test_usage_authority_is_immutable_and_substitution_reverified(
    tmp_path, decision_request, table
):
    path = tmp_path / "decisions.sqlite"
    svc = service(path, lambda _: httpx.Response(200, json=response_for(decision_request)))
    result = svc.evaluate(decision_request)
    with svc.store.connect() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(f"DELETE FROM {table}")
        db.execute(f"DROP TRIGGER {table}_update")
        payload = json.loads(db.execute(f"SELECT payload FROM {table}").fetchone()[0])
        if table == "attempt_usage":
            payload["input_tokens"] = 0
        else:
            payload["request"]["state_json"] = '{"changed":true}'
        encoded = canonical_json(payload)
        db.execute(f"UPDATE {table} SET payload=?,hash=?", (encoded, digest(encoded)))
    with pytest.raises(ValueError):
        svc.store.export_observation(result.observation_id)
    with pytest.raises(ValueError):
        summary(path, decision_request.context.run_id)


def test_missing_store_and_read_only_monitor_never_create_or_leak(tmp_path, decision_request):
    path = tmp_path / "absent" / "budget.sqlite"
    assert summary(path, "missing")["status"] == "unavailable"
    assert not path.parent.exists()
    path = tmp_path / "real.sqlite"
    svc = service(path, lambda _: httpx.Response(200, json=response_for(decision_request)))
    result = svc.evaluate(decision_request)
    view = summary(path, decision_request.context.run_id)
    encoded = canonical_json(view)
    for secret in (
        "fixture-secret",
        "untrusted_evidence",
        "raw_response",
        decision_request.sources[0].text,
        str(path),
    ):
        assert secret not in encoded
    assert view["run_usage"]["known_input_tokens"] == result.input_tokens
    assert not view["run_usage"]["breached"]


def test_legacy_receipts_remain_identical_but_known_overrun_fences(tmp_path, decision_request):
    path = tmp_path / "decisions.sqlite"
    svc = service(
        path,
        lambda _: httpx.Response(
            200,
            json=response_for(
                decision_request, usage={"input_tokens": 10_001, "output_tokens": 20}
            ),
        ),
    )
    result = svc.evaluate(decision_request)
    original = svc.store.export_observation(result.observation_id)
    original.pop("usage_authority")
    # Simulate an actual pre-upgrade file, without altering any original rows.
    with svc.store.connect() as db:
        db.execute("DROP TABLE attempt_usage")
        db.execute("DROP TABLE reservation_authorities")
    assert summary(path, decision_request.context.run_id)["status"] == "unavailable"
    restarted = service(path, lambda _: pytest.fail("legacy overrun dispatched again"))
    assert restarted.store.export_observation(result.observation_id) == original
    assert (
        restarted.evaluate(other_run(decision_request), mode="refresh").status
        == DecisionStatus.BUDGET
    )
    assert summary(path, decision_request.context.run_id)["day_usage"]["total_cost_usd"] is None


@pytest.mark.parametrize("case", ["duplicate-answer", "duplicate-usage", "invalid-counter"])
def test_accounting_parser_keeps_valid_fields_without_claiming_ambiguous_usage(
    tmp_path, decision_request, case
):
    value = response_for(decision_request)
    if case == "invalid-counter":
        value["usage"]["input_tokens"] = True
        value["usage"]["output_tokens"] = 10_001
    body = canonical_json(value)
    if case == "duplicate-answer":
        body = body.replace('"confidence":0.8', '"confidence":0.8,"confidence":0.9')
    if case == "duplicate-usage":
        body = body[:-1] + ',"usage":{"input_tokens":10001,"output_tokens":20}}'
    svc = service(tmp_path / "budget.sqlite", lambda _: httpx.Response(200, text=body))
    result = svc.evaluate(decision_request)
    assert result.status == DecisionStatus.MALFORMED
    svc.store.export_observation(result.observation_id)
    v = summary(svc.store.path, decision_request.context.run_id)["run_usage"]
    if case == "duplicate-answer":
        assert v["known_input_tokens"] == 100 and not v["breached"]
        assert result.cost_usd == pytest.approx(0.0001)
    else:
        assert v["breached"] and v["total_cost_usd"] is None and result.cost_usd is None
        assert (
            svc.evaluate(other_run(decision_request), mode="refresh").status
            == DecisionStatus.BUDGET
        )


def test_attempt_failure_is_atomic_and_uncertain_reservation_not_released(
    tmp_path, decision_request, monkeypatch
):
    svc = service(
        tmp_path / "budget.sqlite",
        lambda _: httpx.Response(200, json=response_for(decision_request)),
    )
    at = datetime.now(timezone.utc)
    assert reserve(svc, decision_request, "crashed", at)[0]
    original = DecisionStore._record

    def crash(db, table, identity, value):
        if table == "attempt_usage":
            raise RuntimeError("test-only process failure")
        return original(db, table, identity, value)

    with monkeypatch.context() as patch:
        patch.setattr(DecisionStore, "_record", staticmethod(crash))
        with pytest.raises(RuntimeError):
            svc.store.attempt("crashed", 1, 200, canonical_json(response_for(decision_request)), 1)
    with svc.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM attempt_usage").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 1
    v = summary(svc.store.path, decision_request.context.run_id)["run_usage"]
    assert v["unknown_usage_reservations"] == 1 and v["total_cost_usd"] is None
    assert v["conservative_tokens"] > 0 and v["conservative_cost_usd"] > 0


def test_observation_failure_after_usage_commit_does_not_erase_fence(
    tmp_path, decision_request, monkeypatch
):
    path = tmp_path / "budget.sqlite"
    svc = service(
        path,
        lambda _: httpx.Response(
            200,
            json=response_for(
                decision_request, usage={"input_tokens": 10_001, "output_tokens": 20}
            ),
        ),
    )

    def crash(*args):
        raise RuntimeError("test-only lost observation write")

    monkeypatch.setattr(svc.store, "append", crash)
    with pytest.raises(RuntimeError):
        svc.evaluate(decision_request)
    restarted = service(path, lambda _: pytest.fail("orphaned overrun lost"))
    assert (
        restarted.evaluate(other_run(decision_request), mode="refresh").status
        == DecisionStatus.BUDGET
    )
    v = summary(path, decision_request.context.run_id)["run_usage"]
    assert v["breached"] and v["total_cost_usd"] == pytest.approx(0.010001)


def test_experiment_breach_stops_future_dispatch_and_is_reverified(experiment):
    x = experiment
    seen = []
    requests = {c.case_id: c.request for c in x.manifest.cohorts[0].cases}

    def handler(request):
        wire = json.loads(request.content)
        seen.append(wire)
        return httpx.Response(
            200,
            json=response_for(
                requests[wire["state"]["context"]["case_id"]],
                usage={"input_tokens": 10_001, "output_tokens": 20},
            ),
        )

    svc = ExperimentService(
        x.env.store,
        evidence_store=x.env.evidence,
        transport=JevHTTPTransport("fixture-secret", transport=httpx.MockTransport(handler)),
    )
    value = x.manifest.model_copy(
        update={
            "budget": x.manifest.budget.model_copy(
                update={"tokens": 10_000, "spend_usd": 0.01, "input_usd_per_million": 1.0}
            )
        }
    )
    frozen = svc.freeze(value)
    receipt = svc.run(frozen.content_hash())
    report = json.loads(receipt.report_json)
    assert svc.verify_report(receipt.content_hash()) == receipt
    assert len(seen) == 1 and report["status"] == "incomplete"
    assert report["cases"][1]["status"] == "not_dispatched"
    assert report["cases"][1]["nondispatch_reason"] == "durable_run_or_day_budget_breach"
    assert report["operating"]["accounting"]["breached"]
    assert report["operating"]["total_cost_usd"] == pytest.approx(0.010001)
    assert not report["acceptance"]["publication_performed"]
    svc.run(frozen.content_hash())
    assert len(seen) == 1


def test_legacy_experiment_report_hash_survives_additive_usage_schema(experiment):
    from src.decision.experiments import ExperimentReport

    x = experiment
    frozen = x.service.freeze(x.manifest)
    x.service.run(frozen.content_hash())
    # Exact pre-usage-schema source/attempt/observation records, unchanged.
    with x.env.store.connect() as db:
        db.execute("DROP TABLE attempt_usage")
        db.execute("DROP TABLE reservation_authorities")
    DecisionStore(x.env.store.path)
    receipts = []
    for spec, (name, identity) in zip(frozen.manifest.cohorts, frozen.cohort_ids, strict=True):
        scored = x.service.evaluator.score(
            identity,
            observation_ids=tuple(
                (
                    case.case_id,
                    x.env.store.evaluation_dispatch_for(case.request.cache_key()).observation_id,
                )
                for case in spec.cases
            ),
        )
        receipts.append((name, scored.content_hash()))
    at = datetime.now(timezone.utc)
    payload, dispatches = x.service._compose(frozen, tuple(receipts), at, usage_authority=False)
    old = ExperimentReport(
        experiment_hash=frozen.content_hash(),
        dispatch_ids=dispatches,
        cohort_receipts=tuple(receipts),
        report_json=canonical_json(payload),
        observed_through=at,
        recorded_at=datetime.now(timezone.utc),
    )
    x.env.store._append_evaluation(old)
    assert "usage_authority_version" not in payload["operating"]
    assert x.service.verify_report(old.content_hash()) == old
    assert summary(x.env.store.path, frozen.manifest.run_id)["run_usage"]["total_cost_usd"] is None
