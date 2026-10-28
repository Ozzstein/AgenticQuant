"""Real durable generative accounting; only provider responses and clocks are fixtures."""

import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from src.agents.budget import AgentBudgetError, AgentCallBudget
from src.agents.receipts import canonical, insert, read_call, read_usage_summary
from src.decision.research_gateway import CHAT_PATH, GatewayConfig, ResearchGateway

UTC = timezone.utc
MODEL = "fixture-model-version-1"


@pytest.fixture
def cfg(tmp_path):
    price = SimpleNamespace(
        provider="openai",
        model=MODEL,
        price_version="fixture-1",
        source_uri="https://example.test/fixture-rates",
        effective_at=datetime(2020, 1, 1, tzinfo=UTC),
        input_usd_per_million=Decimal(1),
        output_usd_per_million=Decimal(2),
    )
    return SimpleNamespace(
        budget_store_path=str(tmp_path / "usage.sqlite"),
        llm_provider="openai",
        pricing=[price],
        max_requests_per_analysis=100,
        max_requests_per_day=200,
        max_tokens_per_analysis=10000,
        max_tokens_per_day=10000,
        max_cost_per_analysis=Decimal(1),
        max_cost_per_day_usd=Decimal(2),
    )


def budget(cfg, run="run"):
    return AgentCallBudget(cfg, run, deadline=time.monotonic() + 30)


def raw(i=10, o=2, *, model=MODEL, **usage):
    return SimpleNamespace(
        usage_metadata={"input_tokens": i, "output_tokens": o, **usage},
        response_metadata={"model": model},
    )


def summary(cfg, run="run", *, at=None, day=None):
    at = at or datetime.now(UTC)
    return read_usage_summary(
        cfg.budget_store_path, day=day or at.date().isoformat(), run_id=run, as_of=at
    )


@pytest.mark.parametrize(
    "i,o,model,priced",
    [
        (1001, 20, "other", False),
        (1001, 20, None, False),
        (1001, None, MODEL, True),
        (None, 1001, MODEL, True),
    ],
)
def test_independent_counters_overrun_original_run_and_day_survive_restart(
    cfg, i, o, model, priced
):
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    with pytest.raises(AgentBudgetError, match="exceeded"):
        b.finish(identity, price, raw=raw(i, o, model=model), status="malformed")
    receipt = read_call(b.path, identity)
    settlement = json.loads(receipt.settlement_json)
    assert settlement["evidence"]["input_tokens"] == i
    assert settlement["evidence"]["output_tokens"] == o
    scope = summary(cfg)["run_usage"]
    assert scope["known_input_tokens"] == (i or 0)
    assert scope["known_output_tokens"] == (o or 0)
    assert scope["conservative_tokens"] == (i or 0) + (o or 0)
    assert scope["actual_cost_usd"] is None and scope["unpriced_requests"] == 1
    assert scope["usage_unknown_requests"] == (0 if i is not None and o is not None else 1)
    assert (Decimal(scope["known_cost_lower_bound_usd"]) > 0) == priced
    assert scope["breached"] and summary(cfg)["day_usage"]["breached"]
    for run in ("run", "new-run"):
        with pytest.raises(AgentBudgetError, match="violated admission"):
            budget(cfg, run).reserve_bound(MODEL, 1, 0)


def test_unknown_attempt_never_refunded_by_later_success_and_known_charge_has_floor(cfg):
    b = budget(cfg)
    first, price, _ = b.reserve_bound(MODEL, 100, 20)
    b.finish(first, price, status="timeout")
    second, price, _ = b.reserve_bound(MODEL, 100, 20)
    b.finish(second, price, raw=raw(), status="ok")
    s = summary(cfg)["run_usage"]
    assert s["requests"] == 2 and s["unpriced_requests"] == s["usage_unknown_requests"] == 1
    assert s["actual_tokens"] is None and s["actual_cost_usd"] is None
    assert s["known_input_tokens"] == 10 and s["known_cost_lower_bound_usd"] == "0.000014"
    assert s["conservative_tokens"] == 240 and Decimal(s["conservative_cost_usd"]) == Decimal(
        "0.00028"
    )
    assert s["uncertain_tokens"] == 120 and not s["breached"]
    call = read_call(b.path, second)
    assert call.status == "ok" and call.actual_model == MODEL
    assert json.loads(call.settlement_json)["usage"]["actual_cost_usd"] == "0.000014"


@pytest.mark.parametrize("bad", [None, True, -1, 1.5, float("nan"), float("inf"), "10", 2**62])
def test_invalid_one_counter_does_not_discard_other(cfg, bad):
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    with pytest.raises(AgentBudgetError, match="cannot certify success"):
        b.finish(identity, price, raw=raw(10, bad), status="ok")
    s = summary(cfg)["run_usage"]
    assert s["known_input_tokens"] == 10 and s["known_output_tokens"] == 0
    assert s["actual_tokens"] is None and s["actual_cost_usd"] is None
    assert s["known_cost_lower_bound_usd"] == "0.00001"
    assert read_call(b.path, identity).status == "usage_unknown"


def test_clock_original_day_fence_not_settlement_day_and_historical_projection(cfg, monkeypatch):
    import src.agents.budget as budget_module
    import src.agents.receipts as receipts_module

    clock = [datetime(2026, 1, 1, 23, 59, 59, tzinfo=UTC)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]

    monkeypatch.setattr(budget_module, "datetime", Clock)
    monkeypatch.setattr(receipts_module, "datetime", Clock)
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    admitted = clock[0]
    clock[0] += timedelta(seconds=2)
    with pytest.raises(AgentBudgetError):
        b.finish(identity, price, raw=raw(200, None), status="malformed")
    before = summary(cfg, at=admitted, day="2026-01-01")["run_usage"]
    assert (
        before["pending_requests"] == 1
        and not before["breached"]
        and before["known_input_tokens"] == 0
    )
    assert summary(cfg, at=clock[0], day="2026-01-01")["day_usage"]["breached"]
    assert not summary(cfg, at=clock[0])["day_usage"]["breached"]
    with pytest.raises(AgentBudgetError, match="violated"):
        budget(cfg).reserve_bound(MODEL, 1, 0)
    budget(cfg, "tomorrow").reserve_bound(MODEL, 1, 0)
    clock[0] = admitted
    # Going backwards cannot hide a committed terminal breach at admission.
    with pytest.raises(AgentBudgetError):
        budget(cfg, "backdated").reserve_bound(MODEL, 1, 0)


def test_already_dispatched_concurrent_results_all_persist_after_breach(cfg):
    b = budget(cfg)
    reservations = [b.reserve_bound(MODEL, 100, 20)[:2] for _ in range(8)]

    def settle(item):
        idx, (identity, price) = item
        try:
            b.finish(identity, price, raw=raw(200 if idx % 2 else 10, 2), status="ok")
            return "ok"
        except AgentBudgetError:
            return "overrun"

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(settle, enumerate(reservations)))
    assert statuses.count("overrun") == 4
    s = summary(cfg)["run_usage"]
    assert s["pending_requests"] == 0 and s["requests"] == 8 and s["known_input_tokens"] == 840
    assert s["actual_tokens"] == 856 and s["actual_cost_usd"] == "0.000872"
    assert s["conservative_tokens"] == 4 * 120 + 4 * 202

    def admit(idx):
        try:
            budget(cfg, f"new-{idx}").reserve_bound(MODEL, 1, 0)
            return True
        except AgentBudgetError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert not any(pool.map(admit, range(8)))
    assert summary(cfg)["day_usage"]["requests"] == 8


def test_read_only_missing_and_corrupted_authority(cfg, tmp_path):
    missing = tmp_path / "absent" / "never.sqlite"
    assert (
        read_usage_summary(missing, day="2026-01-01", as_of=datetime.now(UTC))["status"]
        == "unavailable"
    )
    assert not missing.parent.exists()
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    b.finish(identity, price, raw=raw(), status="ok")
    before = b.path.read_bytes()
    summary(cfg)
    read_call(b.path, identity)
    assert b.path.read_bytes() == before
    with sqlite3.connect(b.path) as db:
        db.execute("UPDATE calls SET charged_tokens=1 WHERE id=?", (identity,))
    with pytest.raises(ValueError, match="differs"):
        summary(cfg)
    with pytest.raises(AgentBudgetError, match="authority"):
        budget(cfg, "new").reserve_bound(MODEL, 1, 0)


def test_immutable_receipt_chain_recomputed_not_just_self_hash(cfg):
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    b.finish(identity, price, raw=raw(), status="ok")
    receipt = read_call(b.path, identity)
    with sqlite3.connect(b.path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM agent_settlements")
        s = json.loads(receipt.settlement_json)
        s["usage"]["actual_cost_usd"] = "0"
        body = canonical(s)
        db.execute("DROP TRIGGER agent_settlements_update")
        db.execute(
            "UPDATE agent_settlements SET payload=?,hash=?",
            (body, hashlib.sha256(body.encode()).hexdigest()),
        )
    with pytest.raises(ValueError, match="original admission/evidence"):
        summary(cfg)


def test_v1_receipts_byte_identical_and_conservative_upgrade_without_rewrite(cfg):
    b = budget(cfg)
    now = datetime.now(UTC).isoformat()
    p = cfg.pricing[0]
    a = {
        "schema_version": "1.0",
        "call_id": "legacy",
        "run_id": "run",
        "session_id": "legacy-session",
        "started_at": now,
        "provider": "openai",
        "model": MODEL,
        "reserved_tokens": 120,
        "reserved_cost_usd": "0.00014",
        "pricing": {
            k: (v.isoformat() if isinstance(v, datetime) else str(v)) for k, v in vars(p).items()
        },
    }
    row = dict(
        id="legacy",
        run_id="run",
        day=now[:10],
        provider="openai",
        model=MODEL,
        pricing_version=p.price_version,
        reserved_tokens=120,
        charged_tokens=12,
        reserved_cost="0.00014",
        charged_cost="0.000014",
        input_tokens=10,
        output_tokens=2,
        status="ok",
        started_at=now,
        latency_ms=1.0,
        actual_model=MODEL,
    )
    s = {"schema_version": "1.0", "completed_at": now, "accounting": row}
    with sqlite3.connect(b.path) as db:
        db.execute(
            "INSERT INTO calls VALUES(" + ",".join("?" for _ in row) + ")", tuple(row.values())
        )
        insert(db, "agent_admissions", "legacy", a)
        insert(db, "agent_settlements", "legacy", s)
    before = b.path.read_bytes()
    receipt = read_call(b.path, "legacy")
    assert receipt.admission_json == canonical(a) and receipt.settlement_json == canonical(s)
    scope = summary(cfg)["run_usage"]
    assert scope["actual_tokens"] == 12 and scope["conservative_tokens"] == 120
    assert scope["actual_cost_usd"] == "0.000014" and scope["conservative_cost_usd"] == "0.00014"
    assert b.path.read_bytes() == before
    cfg.max_tokens_per_analysis = 121
    with pytest.raises(AgentBudgetError, match="exhausted"):
        budget(cfg).reserve_bound(MODEL, 2, 0)
    assert read_call(b.path, "legacy") == receipt


def gateway(cfg, responder, run="native"):
    price = {k: v for k, v in vars(cfg.pricing[0]).items() if k != "provider"}
    config = GatewayConfig(
        run_id=run,
        budget_store_path=cfg.budget_store_path,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        chat=price,
        embedding={**price, "model": "fixture-embedding-1", "output_usd_per_million": 0},
        max_tokens=100000,
        max_tokens_per_day=100000,
        max_requests=30,
        max_requests_per_day=100,
        max_cost_usd=1,
        max_cost_per_day_usd=2,
        max_output_tokens=20,
    )
    token = "fixture-scoped-credential-" + run + "x" * 32
    return ResearchGateway(
        config,
        bearer=token,
        provider_key="fixture-provider-key",
        transport=httpx.MockTransport(responder),
    ), token


def dispatch(g, token):
    return g.dispatch(
        "POST",
        CHAT_PATH,
        "Bearer " + token,
        json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "fixture"}]}).encode(),
    )


@pytest.mark.parametrize(
    "case",
    [
        "wrong",
        "missing",
        "partial",
        "http_error",
        "duplicate_semantic",
        "duplicate_counter",
        "duplicate_usage",
        "contradictory",
        "total_only",
    ],
)
def test_gateway_usage_extracted_before_rejection_and_day_admission_fenced(cfg, case):
    sent = []

    def respond(request):
        sent.append(request)
        body = (
            '{"model":"'
            + MODEL
            + '","usage":{"prompt_tokens":100001,"completion_tokens":2,"total_tokens":100003}}'
        )
        if case == "wrong":
            body = body.replace(MODEL, "other-version")
        if case == "missing":
            body = body.replace('"model":"' + MODEL + '",', "")
        if case == "partial":
            body = body.replace('"completion_tokens":2,', "")
        if case == "duplicate_semantic":
            body = body[:-1] + ',"choices":[],"choices":[]}'
        if case == "duplicate_counter":
            body = body.replace(
                '"prompt_tokens":100001', '"prompt_tokens":100001,"prompt_tokens":1'
            )
        if case == "duplicate_usage":
            body = (
                body[:-1] + ',"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
            )
        if case == "contradictory":
            body = body.replace('"total_tokens":100003', '"total_tokens":1')
        if case == "total_only":
            body = body.replace('"prompt_tokens":100001,"completion_tokens":2,', "")
        return httpx.Response(500 if case == "http_error" else 200, content=body)

    g, token = gateway(cfg, respond)
    assert dispatch(g, token)[0] == 502
    r = g.receipt()
    s = r["accounting"]["run"]
    assert r["accounting"]["schema_version"] == "2.0" and r["accounting"]["run_id"] == "native"
    assert (
        r["accounting"]["day_id"]
        == datetime.fromisoformat(r["accounting"]["as_of"]).date().isoformat()
    )
    assert s["breached"] and r["blocked"] and not r["ready"]
    assert s["known_input_tokens"] == (
        0 if case in ("duplicate_counter", "duplicate_usage", "total_only") else 100001
    )
    if case not in ("duplicate_usage",):
        assert s["conservative_tokens"] >= 100001
    if case in (
        "wrong",
        "missing",
        "partial",
        "duplicate_counter",
        "duplicate_usage",
        "contradictory",
        "total_only",
    ):
        assert s["actual_cost_usd"] is None and r["actual"]["cost_usd"] is None
    else:
        assert s["actual_cost_usd"] == "0.100005"
    with sqlite3.connect(cfg.budget_store_path) as db:
        identity, request_hash, response_hash = db.execute(
            "SELECT call_id,request_hash,response_hash FROM gateway_audit"
        ).fetchone()
    call = read_call(cfg.budget_store_path, identity)
    assert json.loads(call.admission_json)["request_hash"] == request_hash
    assert json.loads(call.settlement_json)["evidence"]["response_hash"] == response_hash
    restarted, next_token = gateway(cfg, respond, run="distinct-native-run")
    assert dispatch(restarted, next_token)[0] == 503 and len(sent) == 1
    text = json.dumps(r)
    assert token not in text and "fixture-provider-key" not in text and "messages" not in text


def test_conservative_day_cap_uses_original_reservations_even_after_exact_small_usage(cfg):
    cfg.max_tokens_per_day = 239
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    b.finish(identity, price, raw=raw(), status="ok")
    with pytest.raises(AgentBudgetError, match="exhausted"):
        budget(cfg, "next-run").reserve_bound(MODEL, 100, 20)
    assert not summary(cfg)["run_usage"]["breached"]
    assert summary(cfg)["day_usage"]["conservative_tokens"] == 120


@pytest.mark.parametrize(
    "field,value",
    [
        ("day", "2001-01-01"),
        ("run_id", "different-run"),
        ("model", "different-model"),
        ("reserved_tokens", 1),
    ],
)
def test_admission_identity_cannot_be_hidden_by_changing_mutable_scope(cfg, field, value):
    b = budget(cfg)
    identity, _, _ = b.reserve_bound(MODEL, 100, 20)
    with sqlite3.connect(b.path) as db:
        db.execute(f"UPDATE calls SET {field}=? WHERE id=?", (value, identity))
    with pytest.raises(ValueError):
        summary(cfg)
    with pytest.raises(AgentBudgetError):
        budget(cfg, "different-run").reserve_bound(MODEL, 1, 0)


def test_usage_failure_is_not_a_false_overrun_or_success(cfg):
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    with pytest.raises(AgentBudgetError) as failure:
        b.finish(identity, price, raw=raw(10, 2, total_tokens=11), status="ok")
    assert failure.value.status == "usage_unknown"
    call = read_call(b.path, identity)
    assert call.status == "usage_unknown"
    s = summary(cfg)["run_usage"]
    assert s["actual_cost_usd"] is None and s["actual_tokens"] is None
    assert s["breach_reasons"] == ["contradictory_usage_response"]
    assert s["known_input_tokens"] == 10 and s["known_output_tokens"] == 2
    assert s["conservative_tokens"] == 120


def test_untrusted_model_response_text_not_exposed_in_public_summary(cfg):
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    private = "fixture-private-response-do-not-publish"
    b.finish(identity, price, raw=raw(model=private), status="model_mismatch")
    scope = summary(cfg)
    assert private not in json.dumps(scope)
    assert scope["run_usage"]["observed_models"] == []
    assert scope["run_usage"]["observed_model_hashes"] == [
        hashlib.sha256(private.encode()).hexdigest()
    ]
    assert read_call(b.path, identity).actual_model == private


def test_atomic_settlement_failure_retains_original_pending_charge(cfg, monkeypatch):
    import src.agents.receipts as receipts

    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("fixture lost settlement write")

    monkeypatch.setattr(receipts, "settlement", fail)
    with pytest.raises(sqlite3.OperationalError):
        b.finish(identity, price, raw=raw(1000, 2), status="ok")
    s = summary(cfg)["run_usage"]
    assert s["pending_requests"] == 1 and s["conservative_tokens"] == 120
    assert s["actual_cost_usd"] is None and not s["breached"]
    with sqlite3.connect(b.path) as db:
        assert db.execute("SELECT count(*) FROM agent_settlements").fetchone()[0] == 0
        assert db.execute("SELECT input_tokens,output_tokens,status FROM calls").fetchone() == (
            None,
            None,
            "reserved",
        )


def test_write_lock_wait_across_midnight_records_actual_admission_day(cfg, monkeypatch):
    import threading

    import src.agents.budget as budget_module
    import src.agents.receipts as receipts_module

    clock = [datetime(2026, 1, 1, 23, 59, 59, tzinfo=UTC)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]

    monkeypatch.setattr(budget_module, "datetime", Clock)
    monkeypatch.setattr(receipts_module, "datetime", Clock)
    b = budget(cfg)
    entered = threading.Event()
    blocker = sqlite3.connect(b.path)
    blocker.execute("BEGIN IMMEDIATE")
    connect = sqlite3.connect

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "BEGIN IMMEDIATE":
                entered.set()
            return super().execute(sql, *args, **kwargs)

    def observed_connect(*args, **kwargs):
        return connect(*args, factory=ObservedConnection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(b.reserve_bound, MODEL, 100, 20)
        assert entered.wait(5)
        clock[0] += timedelta(seconds=2)
        blocker.commit()
        identity, price, _ = pending.result(timeout=5)
    blocker.close()
    b.finish(identity, price, raw=raw(), status="ok")
    assert read_call(b.path, identity).started_at == clock[0]
    assert summary(cfg, at=clock[0], day="2026-01-01")["day_usage"]["requests"] == 0
    assert summary(cfg, at=clock[0])["day_usage"]["requests"] == 1


def test_invalid_model_encoding_retains_valid_usage_without_unserializable_receipt(cfg):
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    b.finish(identity, price, raw=raw(model="\ud800"), status="invalid_response")
    s = summary(cfg)["run_usage"]
    assert s["known_input_tokens"] == 10 and s["known_output_tokens"] == 2
    assert s["unpriced_requests"] == 1 and s["actual_cost_usd"] is None
    assert s["breached"] and read_call(b.path, identity).actual_model is None


def test_changed_settlement_price_cannot_reprice_or_discard_actual_counter(cfg):
    b = budget(cfg)
    identity, price, _ = b.reserve_bound(MODEL, 100, 20)
    changed = SimpleNamespace(**{**vars(price), "input_usd_per_million": Decimal(999)})
    with pytest.raises(AgentBudgetError) as error:
        b.finish(identity, changed, raw=raw(), status="ok")
    assert error.value.status == "pricing_mismatch"
    receipt = read_call(b.path, identity)
    assert receipt.status == "pricing_mismatch"
    scope = summary(cfg)["run_usage"]
    assert scope["actual_cost_usd"] == "0.000014"  # Only original admitted rates.
    assert scope["known_input_tokens"] == 10 and scope["known_output_tokens"] == 2
    assert scope["pending_requests"] == 0 and scope["breached"]
    assert scope["breach_reasons"] == ["settlement_pricing_identity_mismatch"]
    with pytest.raises(AgentBudgetError, match="violated"):
        budget(cfg, "later").reserve_bound(MODEL, 1, 0)


@pytest.mark.parametrize("case", ["wrong_model", "unexpected_generated_tokens"])
def test_embedding_protocol_retains_usage_before_model_and_counter_rejection(cfg, case):
    from src.decision.research_gateway import EMBEDDING_PATH

    def respond(request):
        count = 100001 if case == "wrong_model" else 5
        usage = {"prompt_tokens": count, "total_tokens": count}
        if case == "unexpected_generated_tokens":
            usage["completion_tokens"] = 2
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "wrong" if case == "wrong_model" else "fixture-embedding-1",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0]}],
                "usage": usage,
            },
        )

    g, token = gateway(cfg, respond)
    code, _ = g.dispatch(
        "POST",
        EMBEDDING_PATH,
        "Bearer " + token,
        json.dumps({"model": "fixture-embedding-1", "input": "fixture"}).encode(),
    )
    assert code == 502
    s = g.receipt()["accounting"]["run"]
    assert s["breached"] and s["actual_cost_usd"] is None
    assert s["known_input_tokens"] == (100001 if case == "wrong_model" else 5)
    assert s["known_output_tokens"] == (0 if case == "wrong_model" else 2)
