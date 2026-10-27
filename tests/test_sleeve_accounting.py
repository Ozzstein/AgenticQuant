"""Exact virtual ownership of a single real canonical ledger, synthetic fixtures."""

import json
from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.core.backtester import PortfolioReplay, load_replay_artifact, persist_replay_artifact
from src.domain.portfolio import (
    CorporateAction,
    FillPolicy,
    PortfolioTarget,
    PriceMark,
    TargetWeight,
)
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.sleeves import SleeveBook, exact_sum
from tests.test_core_backtest import A as REPLAY_A
from tests.test_durable_accounting import A, T, account, fill, intent, metadata
from tests.test_replay_decision_provider import inputs_for


def setup(path):
    ledger = account(path, cash="10000")
    book = SleeveBook(ledger, "backtest:test").initialize({"alpha": D(5000), "beta": D(5000)})
    ledger.mark("backtest:test", [PriceMark(instrument=A, price=100, as_of=T)])
    return ledger, book


def allocated_buy(ledger, book):
    order = intent(qty="10", fee="1")
    book.register_intent(
        order, {"alpha": D(3), "beta": D(7)}, source_hash="f" * 64, quantity_quantum=D(1)
    )
    ledger.submit(order)
    return order


def test_partial_fill_residual_cash_fee_and_restart(tmp_path):
    path = tmp_path / "sleeves.sqlite3"
    ledger, book = setup(path)
    order = allocated_buy(ledger, book)
    first = fill(order, qty="5", fee=".50")
    ledger.apply_fill(first)
    parts = book.snapshot(first.executed_at)
    assert [parts[k].portfolio.positions[0].quantity for k in sorted(parts)] == [2, 3]
    assert exact_sum(p.portfolio.cash for p in parts.values()) == D("9499.50")
    assert exact_sum(p.portfolio.fees for p in parts.values()) == D(".50")
    assert exact_sum(p.portfolio.reserved_cash for p in parts.values()) == D("500.5")
    assert parts["alpha"].portfolio.positions[0].opened_at == first.executed_at
    ledger.close()
    ledger = PortfolioLedger(path)
    book = SleeveBook(ledger, order.account_id).initialize({"alpha": D(5000), "beta": D(5000)})
    assert book.snapshot(first.executed_at) == parts
    assert ledger.apply_fill(first) is False
    second = fill(
        order, name="second", qty="5", fee=".50", at=first.executed_at + timedelta(seconds=1)
    )
    ledger.apply_fill(second)
    parts = book.snapshot(second.executed_at)
    assert [parts[k].portfolio.positions[0].quantity for k in sorted(parts)] == [3, 7]
    assert exact_sum(p.portfolio.cash for p in parts.values()) == D(8999)
    assert exact_sum(p.portfolio.nav for p in parts.values()) == D(9999)
    assert parts["alpha"].portfolio.positions[0].opened_at == first.executed_at
    assert ledger.reconcile(order.account_id)["reconciled"]
    assert len(list(ledger.db.execute("SELECT * FROM sleeve_events WHERE kind='fill'"))) == 2
    ledger.close()


def test_internal_cross_changes_ownership_not_canonical_trades_or_nav(tmp_path):
    ledger, book = setup(tmp_path / "cross.sqlite3")
    order = intent(qty="10")
    book.register_intent(order, {"alpha": D(10)}, source_hash="a" * 64, quantity_quantum=D(1))
    ledger.submit(order)
    execution = fill(order)
    ledger.apply_fill(execution)
    at = T + timedelta(days=1)
    ledger.mark(order.account_id, [PriceMark(instrument=A, price=110, as_of=at)])
    before = book.snapshot(at)
    target = PortfolioTarget(
        target_id="rotate",
        decision_at=at,
        universe_version="fixture",
        strategy_id="blend",
        artifact_version="fixture",
        data_snapshot_id="fixture",
        weights=(TargetWeight(instrument=A, weight=D(1100) / D(10100)),),
        sleeve_contributions={
            "alpha": {A.instrument_id: D(0)},
            "beta": {A.instrument_id: D(1100) / D(10100)},
        },
        sleeve_sources={"alpha": "a" * 64, "beta": "b" * 64},
    )
    assert (
        book.net_target(target, {A.instrument_id: metadata(at=at)}, FillPolicy(equity_cost_bps=0))
        == []
    )
    after = book.snapshot(at)
    assert not after["alpha"].portfolio.positions
    assert after["beta"].portfolio.positions[0].quantity == 10
    assert after["beta"].portfolio.positions[0].opened_at == at
    assert after["alpha"].portfolio.cash == 5100
    assert after["beta"].portfolio.cash == 3900
    assert {k: v.portfolio.nav for k, v in before.items()} == {
        k: v.portfolio.nav for k, v in after.items()
    }
    assert exact_sum(v.net_capital_flow for v in after.values()) == 0
    assert len(ledger.fills(order.account_id)) == 1
    assert (
        book.net_target(target, {A.instrument_id: metadata(at=at)}, FillPolicy(equity_cost_bps=0))
        == []
    )
    assert book.snapshot(at) == after
    ledger.close()


def test_capital_transfer_is_explicit_conserved_and_restart_idempotent(tmp_path):
    path = tmp_path / "transfers.sqlite3"
    ledger, book = setup(path)
    book.transfer_cash("rotate", "alpha", "beta", D(1000), T, "1" * 64)
    snapshot = book.snapshot(T)
    assert [snapshot[k].portfolio.cash for k in sorted(snapshot)] == [4000, 6000]
    assert [snapshot[k].net_capital_flow for k in sorted(snapshot)] == [-1000, 1000]
    ledger.close()
    ledger = PortfolioLedger(path)
    book = SleeveBook(ledger, "backtest:test")
    book.transfer_cash("rotate", "alpha", "beta", D(1000), T, "1" * 64)
    assert book.snapshot(T) == snapshot
    with pytest.raises(LedgerError, match="changed content"):
        book.transfer_cash("rotate", "alpha", "beta", D(999), T, "1" * 64)
    with pytest.raises(LedgerError, match="invested/unavailable"):
        book.transfer_cash("impossible", "alpha", "beta", D(4001), T, "1" * 64)
    ledger.close()


def test_corporate_actions_and_full_exit_ownership(tmp_path):
    ledger, book = setup(tmp_path / "actions.sqlite3")
    order = allocated_buy(ledger, book)
    ledger.apply_fill(fill(order, fee="1"))
    book.sync()
    for day, kind, value in ((1, "split", 2), (2, "dividend", 3), (3, "delist", 50)):
        at = T + timedelta(days=day)
        ledger.corporate_action(
            order.account_id,
            CorporateAction(
                action_id=kind,
                instrument=A,
                effective_at=at,
                kind=kind,
                value=value,
                source_id="fixture",
            ),
        )
        parts = book.snapshot(at)
        assert exact_sum(p.portfolio.cash for p in parts.values()) == D(
            ledger._account(order.account_id)["cash"]
        )
        if kind == "split":
            assert [parts[k].portfolio.positions[0].quantity for k in sorted(parts)] == [6, 14]
            assert parts["alpha"].portfolio.positions[0].opened_at == order.eligible_at
        if kind == "delist":
            assert not any(p.portfolio.positions for p in parts.values())
    ledger.close()


def test_actual_multisleeve_callback_replay_and_verified_artifact(tmp_path):
    seen = []

    def provider(at, snapshot):
        seen.append(snapshot)
        weight = D(".25")
        return (
            PortfolioTarget(
                target_id="sleeves:" + at.isoformat(),
                decision_at=at,
                universe_version="test-universe",
                strategy_id="blend",
                artifact_version="test",
                data_snapshot_id="fixture",
                weights=(TargetWeight(instrument=REPLAY_A, weight=weight * 2),),
                sleeve_contributions={"alpha": {"A": weight}, "beta": {"A": weight}},
                sleeve_sources={"alpha": "a" * 64, "beta": "b" * 64},
            )
            if not snapshot.positions
            else None
        )

    inputs = inputs_for(provider)
    inputs["sleeve_capital"] = {"alpha": D(5000), "beta": D(5000)}
    path = tmp_path / "replay.sqlite3"
    result = PortfolioReplay(path).run(**inputs)
    assert len(seen[1].sleeves) == 2
    assert [v.portfolio.positions[0].quantity for v in seen[1].sleeves.values()] == [25, 25]
    assert exact_sum(s.portfolio.nav for s in seen[1].sleeves.values()) == seen[1].nav
    reference = persist_replay_artifact(result, path, tmp_path / "result.json")
    assert load_replay_artifact(reference) == result
    ledger = PortfolioLedger(path)
    final = SleeveBook(ledger, result.account_id).snapshot(inputs["context"].as_of)
    assert [s.portfolio.nav for s in final.values()] == [4700, 4700]
    assert result.equity_curve[-1] == 9400
    ledger.close()


def test_reject_unallocated_fills_and_changed_source(tmp_path):
    ledger, book = setup(tmp_path / "unallocated.sqlite3")
    order = intent()
    ledger.submit(order)
    ledger.apply_fill(fill(order))
    with pytest.raises(LedgerError, match="no explicit sleeve allocation"):
        book.sync()
    book.register_intent(order, {"alpha": D(10)}, source_hash="a" * 64, quantity_quantum=D(1))
    book.sync()
    row = ledger.db.execute("SELECT payload FROM fills").fetchone()
    payload = json.loads(row[0])
    payload["price"] = "101"
    ledger.db.execute("UPDATE fills SET payload=?", (json.dumps(payload),))
    with pytest.raises(LedgerError, match="source fill/economics changed"):
        book.snapshot(order.eligible_at)
    ledger.close()


def test_projection_tamper_cannot_reassign_cash_even_if_totals_still_match(tmp_path):
    ledger, book = setup(tmp_path / "tampered-projection.sqlite3")
    order = allocated_buy(ledger, book)
    ledger.apply_fill(fill(order, fee="1"))
    book.sync()
    row = ledger.db.execute("SELECT id,payload FROM sleeve_events WHERE kind='fill'").fetchone()
    payload = json.loads(row[1])
    payload["cash"]["alpha"] = str(D(payload["cash"]["alpha"]) + 100)
    payload["cash"]["beta"] = str(D(payload["cash"]["beta"]) - 100)
    ledger.db.execute(
        "UPDATE sleeve_events SET payload=? WHERE id=?",
        (json.dumps(payload, sort_keys=True), row[0]),
    )
    with pytest.raises(LedgerError, match="deterministic source allocation"):
        book.snapshot(order.eligible_at)
    ledger.close()


def test_typed_paper_target_to_outbox_to_real_fill_to_sleeve_nav(tmp_path):
    from tests.test_durable_accounting import bar
    from tests.test_venue_execution import setup as venue_setup

    ledger, adapter, service, original = venue_setup(tmp_path)
    book = SleeveBook(ledger, original.account_id).initialize({"alpha": D(50000), "beta": D(50000)})
    ledger.mark(original.account_id, [PriceMark(instrument=A, price=100, as_of=T)])
    target = PortfolioTarget(
        target_id="live-plan",
        decision_at=T,
        universe_version="fixture",
        strategy_id="blend",
        artifact_version="fixture",
        data_snapshot_id="fixture",
        weights=(TargetWeight(instrument=A, weight=D(".01")),),
        sleeve_contributions={
            "alpha": {A.instrument_id: D(".003")},
            "beta": {A.instrument_id: D(".007")},
        },
        sleeve_sources={"alpha": "a" * 64, "beta": "b" * 64},
    )
    prepared = service.prepare_target(target, {A.instrument_id: metadata()}, sleeve_book=book)
    assert len(prepared) == 1
    assert (
        service.prepare_target(target, {A.instrument_id: metadata()}, sleeve_book=book) == prepared
    )
    order = ledger.order(original.account_id, prepared[0]["intent"])["intent"]
    service.dispatch(order.intent_id, as_of=order.eligible_at)
    adapter.advance_open(
        bar(at=order.eligible_at, opening_volume="500"), {A.instrument_id: metadata()}
    )
    service.reconcile_all(as_of=order.eligible_at)
    parts = book.snapshot(order.eligible_at)
    assert [p.portfolio.positions[0].quantity for p in parts.values()] == [2, 3]
    assert exact_sum(p.portfolio.nav for p in parts.values()) == 100000
    assert ledger.reconcile(original.account_id)["reconciled"]
    ledger.close()


def test_hold_frame_keeps_exact_three_shares_despite_float_weight_round_trip(tmp_path):
    import pandas as pd

    from src.domain.common import RunContext
    from src.execution.routing import target_from_blended_frame

    ledger = account(tmp_path / "hold.sqlite3", cash="10004")
    book = SleeveBook(ledger, "backtest:test").initialize({"alpha": D(5004), "beta": D(5000)})
    order = intent(qty="3")
    book.register_intent(order, {"alpha": D(3)}, source_hash="a" * 64, quantity_quantum=D(1))
    ledger.submit(order)
    ledger.apply_fill(fill(order))
    at = T + timedelta(days=1)
    ledger.mark(order.account_id, [PriceMark(instrument=A, price=100, as_of=at)])
    root = ledger.snapshot(order.account_id, at, 86400, persist=False)
    root = type(root).model_validate({**root.model_dump(), "sleeves": book.snapshot(at)})
    context = RunContext(
        run_id="hold",
        account_id=order.account_id,
        mode="backtest",
        as_of=at,
        code_version="fixture",
        config_hash="1" * 64,
        data_snapshot_ids=("fixture",),
    )
    actual = float(D(300) / root.nav)
    assert (root.nav * D(str(actual)) / 100).to_integral_value(rounding="ROUND_DOWN") == 2
    contributions = {"alpha": {A.instrument_id: actual}, "beta": {A.instrument_id: 0.0}}
    outputs = {}
    for name in contributions:
        frame = pd.DataFrame(
            {
                "target_weight": [0.0],
                "action": ["HOLD"],
                "reason": ["cadence"],
                "score": [float("nan")],
            },
            index=[A.instrument_id],
        )
        frame.attrs.update(
            strategy=name,
            strategy_hash=("a" if name == "alpha" else "b") * 64,
            model_hash=None,
            context=context.model_dump(mode="json"),
            signal_at=at.isoformat(),
            data_snapshot_hash="fixture",
            universe_version="fixture",
        )
        outputs[name] = frame
    blended = pd.DataFrame({"target_weight": [actual]}, index=[A.instrument_id])
    blended.attrs.update(
        context=context.model_dump(mode="json"),
        allocations={"alpha": 0.2, "beta": 0.8},
        data_snapshot_hash="fixture",
        universe_version="fixture",
        signal_at=at.isoformat(),
        strategy_hashes={n: f.attrs["strategy_hash"] for n, f in outputs.items()},
        model_artifact_hashes={n: None for n in outputs},
        sleeve_provenance={n: f.attrs for n, f in outputs.items()},
        contributions=contributions,
        vetoed=(),
    )
    target = target_from_blended_frame(
        blended, context, [A], target_id="hold", current_snapshot=root, outputs=outputs
    )
    assert target.sleeve_preserve_quantities["alpha"][A.instrument_id] == 3
    with pytest.raises(TypeError, match="immutable"):
        target.sleeve_contributions["alpha"][A.instrument_id] = D(1)
    assert (
        book.net_target(target, {A.instrument_id: metadata(at=at)}, FillPolicy(equity_cost_bps=0))
        == []
    )
    assert book.snapshot(at)["alpha"].portfolio.positions[0].quantity == 3
    assert len(ledger.fills(order.account_id)) == 1
    ledger.close()


def test_performance_projection_uses_flow_boundary_not_a_fabricated_return(tmp_path):
    ledger, book = setup(tmp_path / "performance.sqlite3")
    opening = book.snapshot(T)
    closing_at = T + timedelta(days=1)
    book.transfer_cash("allocation", "alpha", "beta", D(1000), closing_at, "1" * 64)
    periods = book.performance_period(opening, closing_at)
    for period in periods.values():
        assert (period["closing_nav"] - period["end_flow"]) / period["opening_nav"] - 1 == 0
        assert len(period["source_event_hash"]) == 64
    with pytest.raises(LedgerError, match="intermediate capital transfer"):
        book.performance_period(opening, closing_at + timedelta(days=1))
    ledger.close()


def test_single_strategy_mixed_hold_and_buy_preserves_actual_quantity(tmp_path):
    import pandas as pd

    from src.domain.common import Instrument, RunContext
    from src.execution.routing import target_from_strategy_frame, target_to_intents

    ledger = account(tmp_path / "single-hold.sqlite3", cash="10004")
    order = intent(qty="3")
    ledger.submit(order)
    ledger.apply_fill(fill(order))
    at = T + timedelta(days=1)
    new = Instrument(instrument_id="us:C", symbol="C", asset_class="equity", quote_currency="USD")
    marks = {i.instrument_id: PriceMark(instrument=i, price=100, as_of=at) for i in (A, new)}
    ledger.mark(order.account_id, list(marks.values()))
    snapshot = ledger.snapshot(order.account_id, at, 86400, persist=False)
    context = RunContext(
        run_id="mixed",
        account_id=order.account_id,
        mode="backtest",
        as_of=at,
        code_version="fixture",
        config_hash="1" * 64,
        data_snapshot_ids=("fixture",),
    )
    frame = pd.DataFrame(
        {
            "target_weight": [float(D(300) / snapshot.nav), 0.1],
            "action": ["HOLD", "BUY"],
            "reason": ["preserve", "active"],
            "score": [None, 0.2],
        },
        index=[A.instrument_id, new.instrument_id],
    )
    frame.attrs.update(
        strategy="test",
        strategy_hash="a" * 64,
        context=context.model_dump(mode="json"),
        signal_at=at.isoformat(),
        data_snapshot_hash="fixture",
        universe_version="fixture",
        model_hash="b" * 64,
    )
    target = target_from_strategy_frame(
        frame, context, [A, new], target_id="mixed", current_snapshot=snapshot
    )
    orders = target_to_intents(
        target,
        snapshot,
        marks,
        {i.instrument_id: metadata(at=at, instrument=i) for i in (A, new)},
        FillPolicy(equity_cost_bps=0),
        max_price_age_seconds=86400,
    )
    assert [(o.instrument.instrument_id, o.side, o.quantity) for o in orders] == [
        (new.instrument_id, "buy", D(10))
    ]
    assert target.preserve_quantities[A.instrument_id] == 3
    bad = target.model_copy(update={"preserve_quantities": {A.instrument_id: D(4)}})
    with pytest.raises(LedgerError, match="actual current holdings"):
        target_to_intents(
            bad,
            snapshot,
            marks,
            {i.instrument_id: metadata(at=at, instrument=i) for i in (A, new)},
            FillPolicy(equity_cost_bps=0),
            max_price_age_seconds=86400,
        )
    ledger.close()


def test_long_decimal_money_and_full_exit_reentry_conserve_all_projection_totals(tmp_path):
    ledger, book = setup(tmp_path / "exact.sqlite3")
    basis = D("37.84564546343524345566432112")
    qty = D("3.001")
    fee = basis - qty * D(12)
    for index, (side, price, charge) in enumerate(
        (
            ("buy", D(12), fee),
            ("sell", D(13), D(".001")),
            ("buy", D(12), fee),
            ("sell", D(13), D(".001")),
        )
    ):
        at = T + timedelta(days=index)
        order = intent(
            name=f"precise-{index}",
            side=side,
            qty=str(qty),
            price=str(price),
            fee=str(charge),
            at=at,
        )
        book.register_intent(
            order,
            {"alpha": D(1), "beta": D("2.001")},
            source_hash="1" * 64,
            quantity_quantum=D(".001"),
        )
        ledger.submit(order)
        execution = fill(order, name=f"exact-{index}", price=str(price), fee=str(charge))
        ledger.apply_fill(execution)
        ledger.mark(
            order.account_id, [PriceMark(instrument=A, price=price, as_of=execution.executed_at)]
        )
        parts = book.snapshot(execution.executed_at)
        canonical = ledger.snapshot(order.account_id, execution.executed_at, 86400, persist=False)
        for field in ("cash", "nav", "fees", "realized_pnl"):
            assert exact_sum(getattr(p.portfolio, field) for p in parts.values()) == getattr(
                canonical, field
            )
    ledger.close()


def test_purchase_plan_cannot_spend_same_sleeve_cash_twice(tmp_path):
    ledger, book = setup(tmp_path / "budget.sqlite3")
    other = A.model_copy(update={"instrument_id": "us:C", "symbol": "C"})
    ledger.mark("backtest:test", [PriceMark(instrument=other, price=100, as_of=T)])
    target = PortfolioTarget(
        target_id="overfunded-sleeve",
        decision_at=T,
        universe_version="fixture",
        strategy_id="blend",
        artifact_version="fixture",
        data_snapshot_id="fixture",
        weights=tuple(TargetWeight(instrument=i, weight=D(".3")) for i in (A, other)),
        sleeve_contributions={"alpha": {i.instrument_id: D(".3") for i in (A, other)}, "beta": {}},
        sleeve_sources={"alpha": "a" * 64, "beta": "b" * 64},
    )
    with pytest.raises(LedgerError, match="exceeds sleeve cash"):
        book.net_target(
            target,
            {i.instrument_id: metadata(instrument=i) for i in (A, other)},
            FillPolicy(equity_cost_bps=0),
        )
    assert not list(ledger.db.execute("SELECT * FROM sleeve_intents"))
    assert not ledger.pending("backtest:test")
    assert book.snapshot(T)["alpha"].portfolio.cash == 5000
    ledger.close()


@pytest.mark.parametrize("blended", [False, True])
def test_actual_decision_may_follow_signal_close_but_never_precede_it(tmp_path, blended):
    import pandas as pd

    from src.domain.common import RunContext
    from src.execution.routing import target_from_blended_frame, target_from_strategy_frame

    ledger, book = setup(tmp_path / "decision-lag.sqlite3")
    decision = T + timedelta(minutes=15)
    context = RunContext(
        run_id="cycle",
        mode="paper",
        as_of=decision,
        account_id="backtest:test",
        config_hash="a" * 64,
        code_version="test",
        data_snapshot_ids=("fixture",),
    )
    snapshot = ledger.snapshot("backtest:test", decision, 86400, persist=False)
    output = pd.DataFrame(
        {"target_weight": [0.2], "action": ["BUY"], "reason": ["fixture"], "score": [0.3]},
        index=[A.instrument_id],
    )
    output.attrs.update(
        strategy="alpha",
        strategy_hash="a" * 64,
        context=context.model_dump(mode="json"),
        signal_at=T.isoformat(),
        data_snapshot_hash="fixture",
        universe_version="fixture",
        model_hash="b" * 64,
    )
    if blended:
        snapshot = snapshot.model_copy(update={"sleeves": book.snapshot(decision)})
        beta = output.copy(deep=True)
        beta.attrs = {**output.attrs, "strategy": "beta", "strategy_hash": "c" * 64}
        outputs = {"alpha": output, "beta": beta}
        frame = pd.DataFrame({"target_weight": [0.2]}, index=[A.instrument_id])
        frame.attrs.update(
            context=context.model_dump(mode="json"),
            signal_at=T.isoformat(),
            allocations={"alpha": 0.5, "beta": 0.5},
            data_snapshot_hash="fixture",
            universe_version="fixture",
            strategy_hashes={k: v.attrs["strategy_hash"] for k, v in outputs.items()},
            model_artifact_hashes={k: v.attrs["model_hash"] for k, v in outputs.items()},
            sleeve_provenance={k: dict(v.attrs) for k, v in outputs.items()},
            contributions={k: {A.instrument_id: 0.1} for k in outputs},
            vetoed=(),
        )

        def adapt():
            return target_from_blended_frame(
                frame, context, [A], target_id="cycle", current_snapshot=snapshot, outputs=outputs
            )
    else:
        frame = output

        def adapt():
            return target_from_strategy_frame(
                frame, context, [A], target_id="cycle", current_snapshot=snapshot
            )

    assert adapt().decision_at == decision
    for invalid in (decision + timedelta(seconds=1), decision.replace(tzinfo=None)):
        frame.attrs["signal_at"] = invalid.isoformat()
        with pytest.raises(LedgerError, match="causal context"):
            adapt()
    ledger.close()
