"""Native provider fixtures prove observations/gaps, never invented positive income."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from src.domain.portfolio import ActionCoverage
from src.research.evidence import json_pointer

from src.data.adapters.base import ProviderError, ReadHTTP, store_observation
from src.data.adapters.dividends import (
    DATA,
    INCOME_KINDS,
    PAPER,
    AlpacaDividendAdapter,
    verify_operating_dividend_capture,
)
from src.data.evidence.store import EvidenceStore, canonical_bytes
from src.domain.common import Instrument, RunContext

UTC = timezone.utc
ACCOUNT = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
def capture(tmp_path):
    store = EvidenceStore(tmp_path / "native-income.sqlite")
    requested = datetime(2024, 3, 11, 20, tzinfo=UTC)
    context = RunContext(
        run_id="income-refresh",
        mode="paper",
        as_of=requested,
        account_id="paper:source-book",
        config_hash="0" * 64,
        code_version="test",
    )
    instrument = Instrument(
        instrument_id="equity:identity",
        symbol="FIX",
        asset_class="equity",
        quote_currency="USD",
        venue="XNYS",
    )
    now = [requested + timedelta(seconds=1)]
    state = dict(
        terms=[
            dict(
                corporate_actions={
                    "cash_dividends": [
                        dict(
                            id="native-cash-action",
                            symbol="FIX",
                            cusip="fixture-cusip",
                            rate="0.123456789123456789123456789",
                            special=False,
                            foreign=False,
                            currency="USD",
                            process_date="2024-03-11",
                            ex_date="2024-03-15",
                            record_date="2024-03-18",
                            payable_date="2024-04-01",
                        )
                    ]
                },
                next_page_token=None,
            )
        ],
        activities=[
            [
                dict(
                    activity_type="DIV",
                    id="paid-1",
                    date="2024-03-01",
                    net_amount="1.02",
                    symbol="FIX",
                    cusip="fixture-cusip",
                    qty="2",
                    per_share_amount="0.51",
                )
            ]
        ],
        account=ACCOUNT,
    )
    requests = []

    def handler(request):
        assert request.method == "GET"
        assert "APCA-API-KEY-ID" in request.headers
        requests.append(request)
        now[0] += timedelta(seconds=2)
        if request.url.path == "/v2/account":
            return httpx.Response(200, json={"id": state["account"], "status": "ACTIVE"})
        if request.url.path == "/v1/corporate-actions":
            page = (
                0
                if "page_token" not in request.url.params
                else int(request.url.params["page_token"])
            )
            return httpx.Response(200, json=state["terms"][page])
        if request.url.path == "/v2/account/activities":
            assert request.url.params["category"] == "non_trade_activity"
            page = 0 if "page_token" not in request.url.params else 1
            return httpx.Response(200, json=state["activities"][page])
        raise AssertionError("unapproved endpoint")

    data = ReadHTTP(
        "alpaca", DATA, min_interval=0, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    trading = ReadHTTP(
        "alpaca", PAPER, min_interval=0, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    adapter = AlpacaDividendAdapter(
        store,
        "fixture-key",
        "fixture-secret",
        data_http=data,
        trading_http=trading,
        clock=lambda: now[0],
    )
    return SimpleNamespace(
        store=store,
        context=context,
        instrument=instrument,
        adapter=adapter,
        now=now,
        state=state,
        requests=requests,
        started_at=requested - timedelta(days=30),
        ended_at=requested,
    )


def run(f, *, account=False, **kwargs):
    method = f.adapter.account_dividend_sources if account else f.adapter.cash_dividend_sources
    args = dict(started_at=f.started_at, ended_at=f.ended_at, **kwargs)
    if account:
        args.update(account_id=f.context.account_id, provider_account_id=ACCOUNT)
    result = method(f.instrument, f.context, **args)
    proof = result.evidence[-1]
    return (
        result,
        proof,
        verify_operating_dividend_capture(f.store, proof.record.evidence_id, as_of=f.now[0]),
    )


@pytest.mark.parametrize("account", [False, True])
def test_native_capture_retains_complete_original_and_canonical_unknown_coverage(capture, account):
    f = capture
    result, proof, value = run(f, account=account)
    assert result.status == "available"  # The native capture succeeded.
    assert set(INCOME_KINDS) <= set(value)
    assert all(value[kind] == [] for kind in INCOME_KINDS if kind != "action_coverage")
    assert value["income_authorities"] == []
    assert len(value["native_observations"]) == 1
    fact = value["native_observations"][0]
    original = f.store.get(fact["source_id"], as_of=f.now[0])
    assert fact["source_hash"] == original.content_hash()
    assert fact["source_hash"] != proof.content_hash()
    normalized = f.store.normalized(original.record.evidence_id, as_of=f.now[0])
    raw_native = json.loads(f.store.raw(original.record.evidence_id), parse_float=str)
    assert normalized["payload"] == raw_native
    assert json_pointer(normalized, fact["pointer"]) == fact["observation"]
    assert f.context.as_of < original.record.ingested_at <= proof.record.ingested_at
    assert "fixture-key" not in canonical_bytes(value).decode()
    assert "fixture-secret" not in canonical_bytes(value).decode()
    coverage = ActionCoverage.model_validate(value["action_coverage"][0])
    assert coverage.source_id == original.record.evidence_id
    assert coverage.completeness == "unknown" and coverage.gaps
    assert coverage.account_id == (f.context.account_id if account else None)
    assert coverage.available_at == proof.record.available_at
    with pytest.raises(ValueError, match="coverage"):
        coverage.require_complete(
            instrument_ids=(f.instrument.instrument_id,),
            started_at=f.started_at,
            ended_at=f.ended_at,
            as_of=f.now[0],
            scope=coverage.scope,
            event_types=coverage.event_types,
            account_id=coverage.account_id,
        )
    if account:
        assert value["time_basis"] == "provider-creation-time"
        assert fact["observation"]["per_share_amount"] == "0.51"
        assert (
            "actual-gross-fee-and-withholding-authorities-required-including-zeros" in coverage.gaps
        )
    else:
        assert fact["observation"]["rate"] == "0.123456789123456789123456789"
        assert fact["observation"]["payable_date"] == "2024-04-01"
        assert value["time_basis"] == "provider-process-date"
        assert "pre-ex-entitlement-convention-authority-required" in coverage.gaps


@pytest.mark.parametrize("account", [False, True])
def test_observed_empty_query_is_not_empty_economic_event_coverage(capture, account):
    f = capture
    f.state["terms"] = [dict(corporate_actions={}, next_page_token=None)]
    f.state["activities"] = [[]]
    _, _, value = run(f, account=account)
    assert value["native_observations"] == []
    assert value["action_coverage"][0]["completeness"] == "unknown"
    assert value["action_coverage"][0]["gaps"]


def test_unsupported_actions_and_unlinked_deductions_remain_exact_observations(capture):
    f = capture
    f.state["terms"][0]["corporate_actions"]["cash_mergers"] = [
        dict(id="merger", acquiree_symbol="FIX", rate="3")
    ]
    _, _, terms = run(f)
    assert [r["category"] for r in terms["native_observations"]] == [
        "cash_dividends",
        "cash_mergers",
    ]
    assert "unsupported-action-type:cash_mergers" in terms["action_coverage"][0]["gaps"]
    f.state["activities"][0].append(
        dict(
            activity_type="DIVNRA",
            id="withholding",
            date="2024-03-01",
            net_amount="-0.15",
            symbol="FIX",
        )
    )
    _, _, payments = run(f, account=True)
    assert payments["native_observations"][-1]["observation"]["net_amount"] == "-0.15"
    assert payments["dividend_payments"] == []


def _variant(f, proof, value):
    manifest = value["source_manifest"]
    return store_observation(
        f.store,
        raw=canonical_bytes(manifest),
        normalized=value,
        provider="operating",
        version="operating-wire-2",
        source_uri=proof.record.source_uri,
        kind=proof.kind,
        logical_key=proof.logical_key,
        context=f.context,
        ingested_at=f.now[0],
        instrument=f.instrument,
    )


@pytest.mark.parametrize(
    "change", ["cash", "payment", "distribution", "policy", "complete", "scope", "native_rate"]
)
def test_normalized_flags_cannot_invent_income_or_economic_coverage(capture, change):
    f = capture
    _, proof, value = run(f)
    if change == "complete":
        value["action_coverage"][0].update(completeness="complete", gaps=[])
    elif change == "scope":
        value["capture_request"]["started_at"] = "2000-01-01T00:00:00Z"
    elif change == "native_rate":
        value["native_observations"][0]["observation"]["rate"] = "999"
    else:
        key = dict(
            cash="cash_dividends",
            payment="dividend_payments",
            distribution="dividend_distributions",
            policy="paper_dividend_policies",
        )[change]
        value[key] = [{"source_id": value["action_coverage"][0]["source_id"], "amount": "1"}]
    altered = _variant(f, proof, value)
    with pytest.raises(ProviderError):
        verify_operating_dividend_capture(f.store, altered.record.evidence_id, as_of=f.now[0])


def test_real_pagination_account_identity_and_no_mutation_dispatch(capture):
    f = capture
    original = f.state["terms"][0]
    f.state["terms"] = [
        {**original, "next_page_token": "1"},
        dict(corporate_actions={}, next_page_token=None),
    ]
    _, _, value = run(f)
    assert len(value["native_observations"]) == 1
    assert f.requests[-1].url.params["page_token"] == "1"
    f.state["activities"] = [
        [
            dict(activity_type="DIV", id="native-" + str(i), symbol="FIX", net_amount=".1")
            for i in range(100)
        ],
        [],
    ]
    _, _, activities = run(f, account=True)
    assert len(activities["native_observations"]) == 100
    assert f.requests[-1].url.params["page_token"] == "native-99"
    assert all(r.method == "GET" for r in f.requests)
    f.state["account"] = "00000000-0000-4000-8000-000000000002"
    with pytest.raises(ProviderError, match="account_identity"):
        run(f, account=True)


def test_page_budget_failure_retains_raw_and_never_claims_complete(capture):
    f = capture
    f.state["terms"][0]["next_page_token"] = "1"
    with pytest.raises(ProviderError, match="pagination_budget"):
        run(f, max_pages=1)
    originals = f.store.query(as_of=f.now[0], kind="operating_raw").records
    assert len(originals) == 2  # Measured raw receipt + exact native normalization.
    assert all(f.store.raw(e.record.evidence_id) for e in originals)
    assert not f.store.query(as_of=f.now[0], kind="operating_dividend_terms").records


@pytest.mark.parametrize("change", ["payload", "first_receipt"])
def test_independently_reparse_original_body_and_first_receipt_binding(capture, change):
    f = capture
    _, proof, value = run(f)
    original = f.store.get(value["native_observations"][0]["source_id"])
    normalized = f.store.normalized(original.record.evidence_id, as_of=f.now[0])
    if change == "payload":
        normalized["payload"]["corporate_actions"]["cash_dividends"][0]["rate"] = "999"
    else:
        normalized["capture_source"]["envelope_hash"] = "0" * 64
    altered_source = store_observation(
        f.store,
        raw=f.store.raw(original.record.evidence_id),
        normalized=normalized,
        provider="operating",
        version="operating-wire-2",
        source_uri=original.record.source_uri,
        kind="operating_raw",
        logical_key="false-normalized-authority",
        context=f.context,
        ingested_at=f.now[0],
        instrument=f.instrument,
    )
    value["source_manifest"]["sources"] = [
        dict(
            evidence_id=altered_source.record.evidence_id,
            envelope_hash=altered_source.content_hash(),
            raw_hash=altered_source.record.raw_hash,
        )
    ]
    altered = _variant(f, proof, value)
    with pytest.raises(ProviderError, match="native_payload|receipt_binding"):
        verify_operating_dividend_capture(f.store, altered.record.evidence_id, as_of=f.now[0])


def test_future_archive_receipt_and_repeated_cursor_fail_closed(capture):
    f = capture
    _, proof, _ = run(f)
    with pytest.raises(ProviderError):
        verify_operating_dividend_capture(
            f.store,
            proof.record.evidence_id,
            as_of=proof.record.ingested_at - timedelta(microseconds=1),
        )
    f.state["terms"] = [
        dict(corporate_actions={}, next_page_token="1"),
        dict(corporate_actions={}, next_page_token="1"),
    ]
    with pytest.raises(ProviderError, match="did_not_advance"):
        run(f)
