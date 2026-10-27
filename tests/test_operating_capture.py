"""Synthetic provider wires only; no authenticated or public provider calls."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from src.data.adapters.base import ProviderError, ReadHTTP, store_observation
from src.data.adapters.operating import (
    AlpacaOperatingAdapter,
    BinanceOperatingAdapter,
    KrakenOperatingAdapter,
    OperatingCapture,
    economic_sources,
    verify_operating_capture,
)
from src.data.calendars import sessions
from src.data.evidence.service import EvidenceService
from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, RunContext

UTC = timezone.utc


class Clock:
    def __init__(self, at):
        self.at = at

    def __call__(self):
        return self.at


def context(at):
    return RunContext(
        run_id="operating-test",
        mode="paper",
        as_of=at,
        config_hash="0" * 64,
        code_version="test",
        account_id="paper:operating",
    )


def transport(clock, handler):
    requests = []

    def dispatch(request):
        assert request.method == "GET"
        requests.append(request)
        clock.at += timedelta(seconds=2)
        return handler(request)

    return ReadHTTP(
        "test",
        "https://provider.test",
        client=httpx.Client(transport=httpx.MockTransport(dispatch)),
        min_interval=0,
        attempts=1,
    ), requests


def primary(result, kind):
    return [e for e in result.evidence if e.kind == kind][-1]


def read(store, envelope, clock):
    return store.normalized(envelope.record.evidence_id, as_of=clock())


@pytest.fixture
def equity_capture(tmp_path):
    store = EvidenceStore(tmp_path / "equity.sqlite")
    # Calendar/reference captured on Mar11, strictly before Mar12 opens.
    clock = Clock(datetime(2024, 3, 11, 20, 5, tzinfo=UTC))
    requested = context(datetime(2024, 3, 11, 20, tzinfo=UTC))
    instrument = Instrument(
        instrument_id="us:asset-123",
        symbol="EXAMPLE",
        asset_class="equity",
        quote_currency="USD",
        venue="XNYS",
    )
    schedule = sessions("2024-02-01", "2024-03-15", calendar="XNYS")
    raw_calendar = [
        dict(
            date=label,
            open=o.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).strftime(
                "%H:%M"
            ),
            close=c.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).strftime(
                "%H:%M"
            ),
        )
        for label, (o, c) in schedule.items()
    ]
    completed = [(s, o, c) for s, (o, c) in schedule.items() if c <= requested.as_of][-20:]
    bars = [
        dict(
            t=o.isoformat(),
            o="100.000000000000000001",
            h="102",
            l="99",
            c="101",
            v="1000",
            vw="100.123456789123456789",
        )
        for _, o, _ in completed
    ]
    state = {"actions": {}, "trade_pages": None}

    def handler(req):
        path = req.url.path
        if path == "/v2/calendar":
            return httpx.Response(200, json=raw_calendar)
        if path == "/v2/stocks/bars":
            assert req.url.params["adjustment"] == "raw"
            assert req.url.params["asof"] == "2024-03-11"
            return httpx.Response(
                200, json={"bars": {instrument.symbol: bars}, "next_page_token": None}
            )
        if path == "/v2/assets/asset-123":
            return httpx.Response(
                200,
                json={
                    **dict(
                        id="asset-123",
                        symbol=instrument.symbol,
                        **{"class": "us_equity"},
                        status="active",
                        tradable=True,
                        min_trade_increment="0.001",
                        min_order_size="0.001",
                        price_increment="0.01",
                    ),
                    **state.get("asset_changes", {}),
                },
            )
        if path == "/stable/profile":
            return httpx.Response(
                200,
                json=[
                    {
                        **dict(
                            symbol=instrument.symbol,
                            sector="Technology",
                            currency="USD",
                            isActivelyTrading=True,
                        ),
                        **state.get("profile_changes", {}),
                    }
                ],
            )
        if path == "/v2/stocks/trades":
            pages = state["trade_pages"] or [
                dict(
                    trades={
                        instrument.symbol: [
                            dict(
                                i=9007199254740993123,
                                x="V",
                                t="2024-03-12T13:30:00.123456789Z",
                                p="101.000000000000000001",
                                s="7",
                                c=["@"],
                            )
                        ]
                    },
                    next_page_token=None,
                )
            ]
            i = 0 if "page_token" not in req.url.params else int(req.url.params["page_token"])
            return httpx.Response(200, json=pages[i])
        if path == "/v1/corporate-actions":
            assert "types" not in req.url.params and req.url.params["data_quality"] == "all"
            return httpx.Response(
                200, json={"corporate_actions": state["actions"], "next_page_token": None}
            )
        raise AssertionError(path)

    http, requests = transport(clock, handler)
    adapter = AlpacaOperatingAdapter(
        store,
        "test-key",
        "test-secret",
        fmp_key="test-fmp",
        data_http=http,
        trading_http=http,
        profile_http=http,
        clock=clock,
    )
    calendar = primary(
        adapter.calendar(requested, start="2024-02-01", end="2024-03-15"), "operating_calendar"
    )
    result = adapter.bars(
        instrument,
        requested,
        start=completed[0][1],
        end=requested.as_of,
        calendar_id=calendar.record.evidence_id,
        feed="iex",
        symbol_asof="2024-03-11",
    )
    bar_ids = tuple(e.record.evidence_id for e in result.evidence if e.kind == "bar")
    reference = primary(
        adapter.reference(
            instrument,
            requested,
            asset_id="asset-123",
            bar_ids=bar_ids,
            calendar_id=calendar.record.evidence_id,
        ),
        "operating_reference",
    )
    return SimpleNamespace(
        store=store,
        clock=clock,
        requested=requested,
        instrument=instrument,
        adapter=adapter,
        calendar=calendar,
        bars=result,
        bar_ids=bar_ids,
        reference=reference,
        state=state,
        requests=requests,
    )


def test_measured_post_fetch_clocks_exact_adv_and_capture_only_membership(equity_capture):
    f = equity_capture
    value = read(f.store, f.reference, f.clock)
    risk = value["instrument_metadata"][0]
    assert Decimal(risk["adv_base"]) == Decimal("100123.456789123456789000")
    assert risk["as_of"] == f.reference.record.model_dump(mode="json")["available_at"]
    assert f.reference.record.ingested_at > f.requested.as_of
    with pytest.raises(ValueError, match="unavailable"):
        f.store.get(f.reference.record.evidence_id, as_of=f.requested.as_of)
    raw = f.store.query(as_of=f.clock(), kind="operating_raw").records
    assert len(raw) == 4
    assert raw[0].record.ingested_at == datetime(2024, 3, 11, 20, 5, 2, tzinfo=UTC)
    assert all(
        "test-key" not in e.model_dump_json() and "test-fmp" not in e.model_dump_json() for e in raw
    )
    assert all(
        "apikey"
        not in f.store.normalized(e.record.evidence_id, as_of=f.clock())["request"]["params"]
        for e in raw
    )
    result = OperatingCapture(f.store, clock=f.clock).current_universe(
        (f.instrument,),
        f.requested,
        reference_ids=(f.reference.record.evidence_id,),
        universe_version="capture-only",
    )
    universe = read(f.store, result.evidence[0], f.clock)["universe_history"][0]
    assert (
        universe["effective_at"]
        == universe["available_at"]
        == result.evidence[0].record.model_dump(mode="json")["available_at"]
    )
    assert universe["instrument_ids"] == [f.instrument.instrument_id]
    assert (
        verify_operating_capture(f.store, f.reference.record.evidence_id, as_of=f.clock()) == value
    )


def test_eight_source_arrays_bind_late_individual_print_and_full_audit(equity_capture):
    from src.domain.portfolio import (
        AccountSpec,
        CorporateAction,
        InstrumentRisk,
        OpeningEvent,
        UniverseEligibility,
    )
    from src.research.evidence import expected_artifacts, json_pointer

    f = equity_capture
    universe = (
        OperatingCapture(f.store, clock=f.clock)
        .current_universe(
            (f.instrument,),
            f.requested,
            reference_ids=(f.reference.record.evidence_id,),
            universe_version="operating-universe",
        )
        .evidence[0]
    )
    actions = primary(
        f.adapter.corporate_actions(
            f.instrument,
            f.requested,
            start="2024-03-01",
            end="2024-03-11",
            calendar_id=f.calendar.record.evidence_id,
        ),
        "operating_actions",
    )
    f.clock.at = datetime(2024, 3, 12, 13, 32, tzinfo=UTC)
    requested = context(datetime(2024, 3, 12, 13, 31, tzinfo=UTC))
    prints = primary(
        f.adapter.trades(
            f.instrument,
            requested,
            start=datetime(2024, 3, 12, 13, 30, tzinfo=UTC),
            end=requested.as_of,
            calendar_id=f.calendar.record.evidence_id,
            feed="iex",
            symbol_asof="2024-03-12",
        ),
        "operating_prints",
    )
    values = read(f.store, prints, f.clock)
    event = values["opening_events"][0]
    assert event["opening_volume"] == "7"
    assert {"open", "high", "low", "close", "volume", "closed_at"}.isdisjoint(event)
    assert Decimal(event["price"]) == Decimal("101.000000000000000001")
    assert event["opened_at"] == "2024-03-12T13:30:00.123457Z"
    assert values["prints"][0]["trade_id"] == "V:9007199254740993123"
    assert values["prints"][0]["event_time_wire"] == "2024-03-12T13:30:00.123456789Z"
    assert (
        prints.record.ingested_at
        == prints.record.available_at
        > datetime.fromisoformat(event["opened_at"])
    )
    final = context(f.clock())
    market = EvidenceService(f.store).market_snapshot(
        (f.instrument,), final, start=datetime(2024, 2, 1, tzinfo=UTC)
    )
    sources = economic_sources(
        f.store,
        market,
        reference_ids=(f.reference.record.evidence_id,),
        universe_ids=(universe.record.evidence_id,),
        calendar_ids=(f.calendar.record.evidence_id,),
        print_ids=(prints.record.evidence_id,),
        action_ids=(actions.record.evidence_id,),
        context=final,
        clock=f.clock,
    )
    arrays = {
        kind: [
            row
            for b in bindings
            for value in [
                json_pointer(f.store.normalized(b.evidence_id, as_of=f.clock()), b.pointer)
            ]
            for row in (value if isinstance(value, list) else [value])
        ]
        for kind, bindings in sources
    }
    inputs = SimpleNamespace(
        context=final,
        account=AccountSpec(
            account_id=final.account_id,
            mode="paper",
            base_currency="USD",
            initial_cash=10000,
            opened_at=datetime(2024, 3, 11, tzinfo=UTC),
        ),
        snapshot=market,
        bars=(),
        opening_events=tuple(OpeningEvent.model_validate(v) for v in arrays["opening_events"]),
        metadata=tuple(InstrumentRisk.model_validate(v) for v in arrays["instrument_metadata"]),
        universes=tuple(UniverseEligibility.model_validate(v) for v in arrays["universe_history"]),
        actions=tuple(CorporateAction.model_validate(v) for v in arrays["corporate_actions"]),
    )
    expected = expected_artifacts(inputs)
    assert len(sources) == 8 and {k: v for k, v in arrays.items() if k != "venue_rules"} == expected
    assert arrays["availability_audit"] == [r.model_dump(mode="json") for r in market.evidence]
    binding = dict(sources)["opening_liquidity"][0]
    assert (
        binding.evidence_id == prints.record.evidence_id
        and binding.available_at == prints.record.available_at
    )
    assert f.store.get(binding.evidence_id).content_hash() != prints.record.raw_hash
    assert dict(sources)["calendar"][0].available_at < datetime.fromisoformat(event["opened_at"])

    # Exercise the real root verifier over typed economics. The structural
    # fixture adds H's receipt container, not a parallel runtime schema.
    from src.execution.venue import VenueRules
    from src.operations.evidence import verify_paper_provenance
    from src.research.evidence import BoundArtifact, verify_provenance
    from src.research.registry import RegistryError, ResearchRegistry

    from src.data.evidence.store import canonical_bytes, digest

    inputs.context = final
    inputs.account = AccountSpec(
        account_id=final.account_id,
        mode="paper",
        base_currency="USD",
        initial_cash=10000,
        opened_at=datetime(2024, 3, 11, tzinfo=UTC),
    )
    inputs.rules = tuple(VenueRules.model_validate(v) for v in arrays["venue_rules"])
    receipt = SimpleNamespace(
        event_id=event["event_id"],
        evidence_id=prints.record.evidence_id,
        evidence_hash=prints.content_hash(),
        observed_at=prints.record.ingested_at,
        available_at=prints.record.available_at,
    )
    inputs.receipts = (receipt,)
    inputs.content_hash = lambda: digest(canonical_bytes(expected))
    registry = ResearchRegistry(f.store.path.parent / "registry")
    names = {
        "universe_history": "universe_history_id",
        "availability_audit": "availability_audit_id",
        "calendar": "calendar_artifact_id",
        "corporate_actions": "corporate_actions_id",
        "opening_events": "opening_events_id",
        "opening_liquidity": "opening_liquidity_id",
        "instrument_metadata": "instrument_metadata_id",
    }
    refs = {
        names[kind]: registry.archive_contract(
            BoundArtifact(
                kind=kind,
                inputs_hash=inputs.content_hash(),
                value_json=canonical_bytes(value).decode(),
                sources=dict(sources)[kind],
            )
        )
        for kind, value in expected.items()
    }
    with pytest.raises(RegistryError, match="unavailable at the observed open"):
        verify_provenance(registry, f.store, inputs, refs)
    assert (
        verify_paper_provenance(
            registry,
            f.store,
            inputs,
            refs,
            receipts=inputs.receipts,
            rule_sources=dict(sources)["venue_rules"],
        )
        == refs
    )
    from src.domain.portfolio import FillPolicy, OrderIntent, RiskPolicy
    from src.execution.ledger import PortfolioLedger
    from src.execution.venue import PaperVenue, VenueExecution

    ledger = PortfolioLedger(f.store.path.parent / "actual-print.sqlite")
    ledger.create_account(inputs.account)
    costs = FillPolicy(opening_participation=1)
    risk = RiskPolicy(min_adv=100000)  # Explicit small-liquidity synthetic fixture.
    rules = {r.instrument.instrument_id: r for r in inputs.rules}
    venue = PaperVenue(
        ledger, inputs.account.account_id, rules, fill_policy=costs, risk_policy=risk
    )
    execution = VenueExecution(
        ledger,
        inputs.account.account_id,
        {"paper": venue},
        {f.instrument.instrument_id: "paper"},
        owner="direct",
        fill_policy=costs,
        risk_policy=risk,
    )
    admitted = f.reference.record.available_at + timedelta(minutes=1)
    order = OrderIntent(
        intent_id="source-bound-print",
        account_id=inputs.account.account_id,
        run_id=final.run_id,
        instrument=f.instrument,
        side="buy",
        quantity=1,
        decision_at=admitted,
        eligible_at=admitted + timedelta(microseconds=1),
        reference_price=101,
        estimated_fee=".101",
        strategy_id="fixture",
        tif="DAY",
        expires_at=inputs.opening_events[0].session_close,
    )
    metadata = {m.instrument.instrument_id: m for m in inputs.metadata}
    execution.prepare(order, metadata, as_of=admitted)
    execution.dispatch(order.intent_id, as_of=order.eligible_at)
    assert order.intent_id in venue.accepted_intent_ids_before(inputs.opening_events[0].opened_at)
    retained = {order.intent_id: execution.admission_metadata(order.intent_id)}
    receipt_kwargs = dict(
        observed_at=receipt.observed_at,
        available_at=receipt.available_at,
        as_of=final.as_of,
        evidence_id=receipt.evidence_id,
        evidence_hash=receipt.evidence_hash,
    )
    fills = venue.settle_open(inputs.opening_events[0], retained, **receipt_kwargs)
    assert len(fills) == 1 and fills[0].quantity == 1
    assert fills[0].price == Decimal("101.000000000000000001")
    assert venue.settle_open(inputs.opening_events[0], retained, **receipt_kwargs) == []
    assert ledger.reconcile(inputs.account.account_id)["reconciled"]
    ledger.close()
    receipt.evidence_hash = prints.record.raw_hash
    with pytest.raises(RegistryError, match="source identity"):
        verify_paper_provenance(
            registry,
            f.store,
            inputs,
            refs,
            receipts=inputs.receipts,
            rule_sources=dict(sources)["venue_rules"],
        )


def test_pagination_failure_unsupported_actions_and_missing_adv_are_explicit(equity_capture):
    f = equity_capture
    with pytest.raises(ProviderError, match="twenty_complete"):
        f.adapter.reference(
            f.instrument,
            f.requested,
            asset_id="asset-123",
            bar_ids=f.bar_ids[:-1],
            calendar_id=f.calendar.record.evidence_id,
        )
    f.state["actions"] = {
        "cash_dividends": [
            dict(
                id="cash",
                symbol=f.instrument.symbol,
                rate="0.23",
                ex_date="2024-03-12",
                payable_date="2024-04-01",
            )
        ]
    }
    count = len(f.store.query(as_of=f.clock(), kind="operating_raw").records)
    with pytest.raises(ProviderError, match="unsupported_action_economics_wire_archived"):
        f.adapter.corporate_actions(
            f.instrument,
            f.requested,
            start="2024-03-01",
            end="2024-03-11",
            calendar_id=f.calendar.record.evidence_id,
        )
    assert len(f.store.query(as_of=f.clock(), kind="operating_raw").records) == count + 1
    f.clock.at = datetime(2024, 3, 12, 13, 32, tzinfo=UTC)
    f.state["trade_pages"] = [
        dict(trades={}, next_page_token="1"),
        dict(trades={}, next_page_token="1"),
    ]
    req = context(datetime(2024, 3, 12, 13, 31, tzinfo=UTC))
    with pytest.raises(ProviderError, match="pagination_did_not_advance"):
        f.adapter.trades(
            f.instrument,
            req,
            start=datetime(2024, 3, 12, 13, 30, tzinfo=UTC),
            end=req.as_of,
            calendar_id=f.calendar.record.evidence_id,
            feed="iex",
            symbol_asof="2024-03-12",
        )


def test_raw_invalid_json_is_archived_and_401_never_becomes_empty_success(tmp_path):
    store = EvidenceStore(tmp_path / "fail.sqlite")
    clock = Clock(datetime(2024, 3, 12, tzinfo=UTC))
    req = context(clock())
    http, _ = transport(clock, lambda r: httpx.Response(200, content=b"{broken"))
    cap = OperatingCapture(store, clock=clock)
    with pytest.raises(ProviderError, match="malformed_json_wire_archived"):
        cap.wire(http, "/public", req, upstream="test")
    assert (
        store.raw(store.query(as_of=clock(), kind="operating_raw").records[0].record.evidence_id)
        == b"{broken"
    )
    http, _ = transport(clock, lambda r: httpx.Response(401, json={"secret": "never surfaced"}))
    with pytest.raises(ProviderError, match="credentials_missing_or_rejected") as error:
        cap.wire(http, "/public", req, upstream="test")
    assert "never surfaced" not in str(error.value)


@pytest.fixture(params=["kraken", "binance"])
def crypto_capture(tmp_path, request):
    venue = request.param
    store = EvidenceStore(tmp_path / (venue + ".sqlite"))
    clock = Clock(datetime(2024, 3, 22, 0, 5, tzinfo=UTC))
    req = context(datetime(2024, 3, 22, tzinfo=UTC))
    instrument = Instrument(
        instrument_id="spot:" + venue + ":btc",
        symbol="XBT/USD" if venue == "kraken" else "BTC/USDT",
        asset_class="crypto",
        quote_currency="USD" if venue == "kraken" else "USDT",
        venue=venue,
    )
    start = datetime(2024, 3, 2, tzinfo=UTC)
    state = {"truncated": False, "market_error": False}
    daily = []
    for i in range(21):
        opened = start + timedelta(days=i)
        if venue == "kraken":
            daily.append(
                [
                    int(opened.timestamp()),
                    "100",
                    "102",
                    "99",
                    "101",
                    "100.123456789123456789",
                    "1000",
                    20,
                ]
            )
        else:
            daily.append(
                [
                    int(opened.timestamp() * 1000),
                    "100",
                    "102",
                    "99",
                    "101",
                    "1000",
                    int((opened + timedelta(days=1)).timestamp() * 1000) - 1,
                    "100123.456789123456789",
                    20,
                    "300",
                    "30000",
                    "0",
                ]
            )

    def handler(r):
        path = r.url.path
        if path == "/0/public/OHLC":
            rows = daily[1:] if state["truncated"] else daily
            return httpx.Response(
                200, json=dict(error=[], result={"XXBTZUSD": rows, "last": "1711065600"})
            )
        if path == "/0/public/AssetPairs":
            row = dict(
                wsname=instrument.symbol,
                altname="XBTUSD",
                aclass_base="currency",
                base="XXBT",
                quote="ZUSD",
                lot="unit",
                lot_multiplier=1,
                lot_decimals=8,
                tick_size="0.1",
                ordermin="0.0001",
                costmin="0.5",
                status="online",
            )
            if state["market_error"]:
                row["wsname"] = "OTHER/USD"
            return httpx.Response(200, json=dict(error=[], result={"XXBTZUSD": row}))
        if path == "/0/public/Trades":
            first = [
                [
                    "101.123456789123456789",
                    "0.000000123456789",
                    "1711065960.123456789",
                    "b",
                    "m",
                    "",
                    9007199254740993123,
                ]
            ]
            last = [["102", "0.2", "1711066021", "b", "m", "", 9007199254740993124]]
            initial = r.url.params["since"] == "1711065960000000000"
            return httpx.Response(
                200,
                json=dict(
                    error=[],
                    result={
                        "XXBTZUSD": first if initial else last,
                        "last": "1711065960123456789" if initial else "1711066021000000000",
                    },
                ),
            )
        if path == "/api/v3/klines":
            return httpx.Response(200, json=daily[:20][1:] if state["truncated"] else daily[:20])
        if path == "/api/v3/exchangeInfo":
            row = dict(
                symbol="BTCUSDT",
                baseAsset="BTC",
                quoteAsset="USDT",
                isSpotTradingAllowed=True,
                orderTypes=["MARKET"],
                status="TRADING",
                filters=[
                    dict(
                        filterType="LOT_SIZE", minQty="0.00001", maxQty="1000", stepSize="0.00001"
                    ),
                    dict(filterType="MARKET_LOT_SIZE", minQty="0", maxQty="500", stepSize="0"),
                    dict(filterType="PRICE_FILTER", tickSize="0.01"),
                    dict(filterType="MIN_NOTIONAL", minNotional="5", applyToMarket=True),
                ],
            )
            if state["market_error"]:
                row["quoteAsset"] = "USD"
            if state.get("maximum_notional"):
                row["filters"].append(
                    dict(
                        filterType="NOTIONAL",
                        applyMaxToMarket=True,
                        maxNotional="100",
                        applyMinToMarket=True,
                        minNotional="5",
                    )
                )
            return httpx.Response(200, json={"symbols": [row]})
        if path == "/api/v3/trades":
            events = [
                dict(
                    id=9007199254740993122,
                    price="100",
                    qty="0.1",
                    time=1711065959000,
                    isBestMatch=True,
                ),
                dict(
                    id=9007199254740993123,
                    price="101.123456789123456789",
                    qty="0.000000123456789",
                    time=1711065960123,
                    isBestMatch=True,
                ),
                dict(
                    id=9007199254740993124,
                    price="102",
                    qty="0.2",
                    time=1711066021000,
                    isBestMatch=True,
                ),
            ]
            return httpx.Response(200, json=events[1:] if state["truncated"] else events)
        raise AssertionError(path)

    http, requests = transport(clock, handler)
    adapter = (KrakenOperatingAdapter if venue == "kraken" else BinanceOperatingAdapter)(
        store, http=http, clock=clock
    )
    kwargs = {"market_id": "XXBTZUSD"} if venue == "kraken" else {}
    result = adapter.bars(instrument, req, start=start, end=req.as_of, **kwargs)
    bar_ids = tuple(e.record.evidence_id for e in result.evidence if e.kind == "bar")
    fx = None
    if venue == "binance":
        fx = store_observation(
            store,
            raw=b'{"observed-usdt-usd":"0.9997"}',
            normalized=dict(
                base_currency="USDT",
                quote_currency="USD",
                rate="0.9997",
                observed_at=clock().isoformat(),
            ),
            provider="synthetic-fx-test",
            version="1",
            source_uri="test://fx",
            kind="fx",
            logical_key="USDT/USD",
            context=req,
            ingested_at=clock(),
        )
    reference = primary(
        adapter.reference(
            instrument,
            req,
            bar_ids=bar_ids,
            fx_evidence_id=fx.record.evidence_id if fx else None,
            **kwargs,
        ),
        "operating_reference",
    )
    calendar = primary(
        adapter.calendar(
            instrument,
            req,
            start="2024-03-22",
            end="2024-03-23",
            reference_id=reference.record.evidence_id,
        ),
        "operating_calendar",
    )
    return SimpleNamespace(
        venue=venue,
        store=store,
        clock=clock,
        req=req,
        instrument=instrument,
        adapter=adapter,
        kwargs=kwargs,
        bars=result,
        bar_ids=bar_ids,
        reference=reference,
        calendar=calendar,
        state=state,
        requests=requests,
        fx=fx,
    )


def test_crypto_exact_turnover_limits_individual_print_size_and_wire_ids(crypto_capture):
    f = crypto_capture
    risk = read(f.store, f.reference, f.clock)["instrument_metadata"][0]
    expected = Decimal("100123.456789123456789") * (Decimal(".9997") if f.fx else 1)
    assert Decimal(risk["adv_base"]) == expected
    assert len(f.bar_ids) == 20
    if f.venue == "binance":
        rules = read(f.store, f.reference, f.clock)["venue_rules"][0]
        assert rules["quantity_step"] == "0.00001" and rules["maximum_quantity"] == "500"
    f.clock.at = datetime(2024, 3, 22, 0, 8, tzinfo=UTC)
    req = context(datetime(2024, 3, 22, 0, 7, tzinfo=UTC))
    result = f.adapter.trades(
        f.instrument,
        req,
        start=datetime(2024, 3, 22, 0, 6, tzinfo=UTC),
        end=req.as_of,
        calendar_id=f.calendar.record.evidence_id,
        fx_evidence_id=f.fx.record.evidence_id if f.fx else None,
        **f.kwargs,
    )
    proof = primary(result, "operating_prints")
    normalized = verify_operating_capture(f.store, proof.record.evidence_id, as_of=f.clock())
    assert len(normalized["opening_events"]) == 1
    bar = normalized["opening_events"][0]
    assert Decimal(bar["opening_volume"]) == Decimal("0.000000123456789")
    assert {"open", "high", "low", "close", "volume", "closed_at"}.isdisjoint(bar)
    assert Decimal(bar["price"]) == Decimal("101.123456789123456789")
    assert normalized["prints"][0]["trade_id"] == "9007199254740993123"
    assert Decimal(bar["fx_rate"]) == (Decimal(".9997") if f.fx else 1)
    assert all("/aggTrades" not in r.url.path for r in f.requests)


def test_crypto_upstream_identity_uses_exact_native_print_not_projection(crypto_capture):
    from src.data.opening_identity import verified_opening_source_identity

    f = crypto_capture
    f.clock.at = datetime(2024, 3, 22, 0, 8, tzinfo=UTC)
    req = context(datetime(2024, 3, 22, 0, 7, tzinfo=UTC))
    proof = primary(
        f.adapter.trades(
            f.instrument,
            req,
            start=datetime(2024, 3, 22, 0, 6, tzinfo=UTC),
            end=req.as_of,
            calendar_id=f.calendar.record.evidence_id,
            fx_evidence_id=f.fx.record.evidence_id if f.fx else None,
            **f.kwargs,
        ),
        "operating_prints",
    )
    event = read(f.store, proof, f.clock)["opening_events"][0]
    identity = verified_opening_source_identity(
        f.store, proof.record.evidence_id, event["event_id"], as_of=f.clock()
    )
    assert identity.provider == f.venue
    assert identity.instrument == f.instrument
    assert identity.trade_id == "9007199254740993123"
    assert identity.wire_timestamp == (
        "1711065960123456789" if f.venue == "kraken" else "1711065960123000000"
    )


def test_crypto_missing_history_wrong_identity_and_unknown_fx_fail(crypto_capture):
    f = crypto_capture
    f.state["truncated"] = True
    with pytest.raises(ProviderError, match="truncated|missing"):
        f.adapter.bars(
            f.instrument, f.req, start=datetime(2024, 3, 2, tzinfo=UTC), end=f.req.as_of, **f.kwargs
        )
    f.state["market_error"] = True
    with pytest.raises(ProviderError, match="identity"):
        f.adapter.reference(
            f.instrument,
            f.req,
            bar_ids=f.bar_ids,
            fx_evidence_id=f.fx.record.evidence_id if f.fx else None,
            **f.kwargs,
        )
    if f.venue == "binance":
        f.state["market_error"] = False
        with pytest.raises(ProviderError, match="explicit_usd_fx_evidence_required"):
            f.adapter.reference(f.instrument, f.req, bar_ids=f.bar_ids)
        f.clock.at = datetime(2024, 3, 22, 0, 8, tzinfo=UTC)
        req = context(datetime(2024, 3, 22, 0, 7, tzinfo=UTC))
        with pytest.raises(ProviderError, match="recent_trade_window_not_fully_observed"):
            f.adapter.trades(
                f.instrument,
                req,
                start=datetime(2024, 3, 22, 0, 6, tzinfo=UTC),
                end=req.as_of,
                calendar_id=f.calendar.record.evidence_id,
                fx_evidence_id=f.fx.record.evidence_id,
            )


def test_split_ratio_source_clock_and_explicit_action_scope(equity_capture):
    f = equity_capture
    f.state["actions"] = {
        "forward_splits": [
            dict(
                id="actual-split",
                symbol=f.instrument.symbol,
                old_rate="3",
                new_rate="7.5",
                process_date="2024-03-11",
                ex_date="2024-03-12",
            )
        ]
    }
    result = f.adapter.corporate_actions(
        f.instrument,
        f.requested,
        start="2024-03-01",
        end="2024-03-11",
        calendar_id=f.calendar.record.evidence_id,
    )
    proof = primary(result, "operating_actions")
    value = verify_operating_capture(f.store, proof.record.evidence_id, as_of=f.clock())
    action = value["corporate_actions"][0]
    assert action["value"] == "2.5" and action["effective_at"] == "2024-03-12T13:30:00Z"
    assert f.store.get(action["source_id"]).record.available_at == proof.record.available_at
    assert value["coverage"]["historical_completeness"] == "not-certified"
    assert value["coverage"]["instrument_ids"] == [f.instrument.instrument_id]
    f.clock.at -= timedelta(seconds=1)
    with pytest.raises(ProviderError, match="clock_moved_backwards"):
        f.adapter.current_universe(
            (f.instrument,),
            f.requested,
            reference_ids=(f.reference.record.evidence_id,),
            universe_version="bad-clock",
        )


def test_missing_or_ambiguous_calendar_coverage_fails(tmp_path):
    store = EvidenceStore(tmp_path / "coverage.sqlite")
    clock = Clock(datetime(2024, 3, 12, tzinfo=UTC))
    req = context(clock())
    http, _ = transport(clock, lambda r: httpx.Response(200, json=[]))
    adapter = AlpacaOperatingAdapter(
        store, "test", "test", data_http=http, trading_http=http, clock=clock
    )
    # March11 is an actual session; an empty provider result is not a holiday.
    with pytest.raises(ProviderError, match="provider_calendar_coverage"):
        adapter.calendar(req, start="2024-03-11", end="2024-03-11")
    # March10 is a Sunday and may truthfully return an empty calendar array.
    assert (
        read(
            store,
            primary(
                adapter.calendar(req, start="2024-03-10", end="2024-03-10"), "operating_calendar"
            ),
            clock,
        )["calendar"]
        == []
    )


@pytest.mark.parametrize("field", ["sector", "lot", "asset_identity"])
def test_missing_reference_authority_never_uses_default_sector_or_lot(equity_capture, field):
    f = equity_capture
    if field == "sector":
        f.state["profile_changes"] = {"sector": None}
    elif field == "lot":
        f.state["asset_changes"] = {"min_trade_increment": None}
    else:
        f.state["asset_changes"] = {"id": "different-security"}
    with pytest.raises(ProviderError, match="coverage_missing|invalid_decimal|identity_mismatch"):
        f.adapter.reference(
            f.instrument,
            f.requested,
            asset_id="asset-123",
            bar_ids=f.bar_ids,
            calendar_id=f.calendar.record.evidence_id,
        )


def test_equity_reference_uses_actual_request_cutoff_not_truncated_calendar(equity_capture):
    f = equity_capture
    # Before today's close, yesterday's completed 20 are the actual window.
    f.clock.at = datetime(2024, 3, 12, 13, 32, tzinfo=UTC)
    before = context(datetime(2024, 3, 12, 13, 31, tzinfo=UTC))
    current = f.adapter.reference(
        f.instrument,
        before,
        asset_id="asset-123",
        bar_ids=f.bar_ids,
        calendar_id=f.calendar.record.evidence_id,
    )
    assert (
        read(f.store, primary(current, "operating_reference"), f.clock)["adv_sessions"][-1]
        == "2024-03-11"
    )
    # After that close, the same 20 are missing today's actual completed day.
    f.clock.at = datetime(2024, 3, 12, 20, 5, tzinfo=UTC)
    after = context(datetime(2024, 3, 12, 20, tzinfo=UTC))
    with pytest.raises(ProviderError, match="twenty_complete"):
        f.adapter.reference(
            f.instrument,
            after,
            asset_id="asset-123",
            bar_ids=f.bar_ids,
            calendar_id=f.calendar.record.evidence_id,
        )
    f.clock.at = datetime(2026, 3, 12, 20, 5, tzinfo=UTC)
    later = context(datetime(2026, 3, 12, 20, tzinfo=UTC))
    with pytest.raises(ProviderError, match="current_twenty_session_calendar"):
        f.adapter.reference(
            f.instrument,
            later,
            asset_id="asset-123",
            bar_ids=f.bar_ids,
            calendar_id=f.calendar.record.evidence_id,
        )


def test_crypto_reference_cannot_relabel_old_turnover_as_current(crypto_capture):
    f = crypto_capture
    f.clock.at = datetime(2026, 3, 22, 0, 5, tzinfo=UTC)
    later = context(datetime(2026, 3, 22, tzinfo=UTC))
    with pytest.raises(ProviderError, match="twenty_complete|fx_missing_stale"):
        f.adapter.reference(
            f.instrument,
            later,
            bar_ids=f.bar_ids,
            fx_evidence_id=f.fx.record.evidence_id if f.fx else None,
            **f.kwargs,
        )


def test_v1_capture_readability_and_mixed_format_rejection(equity_capture):
    # Legacy fixtures are test-only archived v1 observations. Runtime v2 never
    # constructs this completed-bar shape from a print.
    from src.domain.portfolio import ExecutionBar

    from src.data.adapters.operating import VERSION
    from src.data.evidence.store import canonical_bytes, digest

    f = equity_capture
    f.clock.at = datetime(2024, 3, 12, 13, 32, tzinfo=UTC)
    requested = context(datetime(2024, 3, 12, 13, 31, tzinfo=UTC))
    result = f.adapter.trades(
        f.instrument,
        requested,
        start=datetime(2024, 3, 12, 13, 30, tzinfo=UTC),
        end=requested.as_of,
        calendar_id=f.calendar.record.evidence_id,
        feed="iex",
        symbol_asof="2024-03-12",
    )
    current = primary(result, "operating_prints")
    opening = read(f.store, current, f.clock)["opening_events"][0]
    legacy = ExecutionBar(
        event_id="archived-v1-event",
        instrument=f.instrument,
        opened_at=opening["opened_at"],
        closed_at="2024-03-12T13:31:00Z",
        open=101,
        high=102,
        low=100,
        close=102,
        volume=1000,
        opening_volume=7,
        calendar=opening["calendar"],
        calendar_version=opening["calendar_version"],
        session=opening["session"],
        session_open=opening["session_open"],
        session_close=opening["session_close"],
    )
    wire_bytes = canonical_bytes({"archived_completed_bar": legacy.model_dump(mode="json")})
    old_wire = store_observation(
        f.store,
        raw=wire_bytes,
        normalized={
            "wire_hash": digest(wire_bytes),
            "upstream": "synthetic-v1-fixture",
            "request": {"path": "/test-archive", "params": {}},
        },
        provider="operating",
        version="operating-wire-1",
        source_uri="test://legacy",
        kind="operating_raw",
        logical_key="old-wire",
        context=requested,
        ingested_at=f.clock(),
        instrument=f.instrument,
    )
    manifest = dict(
        implementation="operating-wire-1",
        policy="individual-print-size-1",
        sources=[
            dict(
                evidence_id=old_wire.record.evidence_id,
                envelope_hash=old_wire.content_hash(),
                raw_hash=old_wire.record.raw_hash,
            )
        ],
    )
    raw = canonical_bytes(manifest)
    old = store_observation(
        f.store,
        raw=raw,
        normalized=dict(
            source_manifest=manifest,
            execution_bars=[legacy.model_dump(mode="json")],
            opening_liquidity=[
                {
                    k: legacy.model_dump(mode="json")[k]
                    for k in ("event_id", "instrument", "opened_at", "opening_volume", "fx_rate")
                }
            ],
        ),
        provider="operating",
        version="operating-wire-1",
        source_uri="test://legacy",
        kind="operating_prints",
        logical_key=f.instrument.instrument_id,
        context=requested,
        ingested_at=f.clock(),
        instrument=f.instrument,
    )
    original_hash = old.content_hash()
    assert VERSION == "operating-wire-2"
    assert verify_operating_capture(f.store, old.record.evidence_id, as_of=f.clock())[
        "execution_bars"
    ] == [legacy.model_dump(mode="json")]
    assert digest(f.store.raw(old.record.evidence_id)) == old.record.raw_hash
    snapshot = EvidenceService(f.store).market_snapshot(
        (f.instrument,), context(f.clock()), start=datetime(2024, 2, 1, tzinfo=UTC)
    )
    with pytest.raises(ProviderError, match="mixed_or_unsupported"):
        economic_sources(
            f.store,
            snapshot,
            reference_ids=(f.reference.record.evidence_id,),
            universe_ids=(),
            calendar_ids=(f.calendar.record.evidence_id,),
            print_ids=(old.record.evidence_id, current.record.evidence_id),
            action_ids=(),
            context=context(f.clock()),
            clock=f.clock,
        )
    assert f.store.get(old.record.evidence_id).content_hash() == original_hash


@pytest.mark.parametrize("crypto_capture", ["binance"], indirect=True)
def test_unsupported_actual_market_maximum_is_archived_and_blocks_reference(crypto_capture):
    f = crypto_capture
    f.state["maximum_notional"] = True
    with pytest.raises(ProviderError, match="unsupported_market_maximum_notional_wire_archived"):
        f.adapter.reference(
            f.instrument, f.req, bar_ids=f.bar_ids, fx_evidence_id=f.fx.record.evidence_id
        )
