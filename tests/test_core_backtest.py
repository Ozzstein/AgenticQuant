"""Synthetic, independently calculated chronological accounting cases."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from scripts.run_backtest import app
from src.core.backtester import PortfolioReplay, WalkForwardBacktester
from src.domain.common import Instrument, RunContext
from src.domain.portfolio import (
    AccountSpec,
    ExecutionBar,
    FillPolicy,
    InstrumentRisk,
    RankedSignal,
    RiskPolicy,
    UniverseEligibility,
)
from src.execution.ledger import LedgerError, PortfolioLedger
from src.utils.exceptions import BacktestError

START = datetime(2024, 1, 2, 14, tzinfo=timezone.utc)
A = Instrument(instrument_id="A", symbol="A", asset_class="equity", quote_currency="USD")
B = Instrument(instrument_id="B", symbol="B", asset_class="equity", quote_currency="USD")


def replay_inputs():
    bars = []
    for day, (opening, close) in enumerate([(100, 100), (100, 110), (110, 88)]):
        at = START + timedelta(days=day)
        for instrument in (A, B):
            op, cl = (opening, close) if instrument == A else (100, 100)
            bars.append(
                ExecutionBar(
                    event_id=f"{instrument.symbol}-{day}",
                    instrument=instrument,
                    opened_at=at,
                    closed_at=at + timedelta(hours=6),
                    open=op,
                    high=max(op, cl),
                    low=min(op, cl),
                    close=cl,
                    volume=1000000,
                    opening_volume=100000,
                    calendar="XNYS",
                    calendar_version="test",
                    session=at.date(),
                    session_open=at,
                    session_close=at + timedelta(hours=6),
                )
            )
    signals = [
        RankedSignal(
            signal_id=f"{i.symbol}-{day}",
            instrument=i,
            decision_at=START + timedelta(days=day, hours=6),
            available_at=START + timedelta(days=day, hours=6),
            score=1 if i == A else 0,
            model_version="test-frozen",
            universe_version="test-universe",
            data_snapshot_id="fixture",
            strategy_id="test",
        )
        for day in (0, 1)
        for i in (A, B)
    ]
    account = AccountSpec(
        account_id="backtest:case",
        mode="backtest",
        base_currency="USD",
        initial_cash=10000,
        opened_at=START - timedelta(seconds=1),
    )
    context = RunContext(
        run_id="case",
        account_id=account.account_id,
        mode="backtest",
        as_of=bars[-1].closed_at,
        code_version="fixture",
        config_hash="1" * 64,
        data_snapshot_ids=("fixture",),
    )
    return dict(
        context=context,
        account=account,
        bars=bars,
        universes=[
            UniverseEligibility(
                universe_version="test-universe",
                data_snapshot_id="fixture",
                effective_at=account.opened_at,
                available_at=account.opened_at,
                instrument_ids=("A", "B"),
            )
        ],
        metadata=[
            InstrumentRisk(
                instrument=i,
                sector=i.symbol,
                adv_base=100000000,
                as_of=account.opened_at,
                lot_size=1,
            )
            for i in (A, B)
        ],
        signals=signals,
        topk=1,
        fill_policy=FillPolicy(equity_cost_bps=0, crypto_cost_bps=0),
        risk_policy=RiskPolicy(
            equity_weight=1,
            crypto_weight=1,
            sector_weight=1,
            crypto_total=1,
            min_cash=0,
            max_daily_turnover=1,
            max_drawdown=1,
        ),
        samples_per_year=252,
    )


def write_bundle(path, inputs):
    data = {}
    for k, v in inputs.items():
        if hasattr(v, "model_dump"):
            data[k] = v.model_dump(mode="json")
        elif isinstance(v, list):
            data[k] = [x.model_dump(mode="json") if hasattr(x, "model_dump") else x for x in v]
        else:
            data[k] = v
    path.write_text(json.dumps(data))


def test_reproduced_selection_failure_is_date_local():
    index = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=2), ["A", "B"]])
    labels = pd.Series([0.1, 0.01, 0, -0.2], index=index)
    assert WalkForwardBacktester._top_k_period_returns(labels, np.array([9, 1, 0, 8]), 1, 0) == [
        0.1,
        -0.2,
    ]


def test_real_losing_nav_drawdown_and_timestamped_fills(tmp_path):
    inputs = replay_inputs()
    result = PortfolioReplay(tmp_path / "replay.db").run(**inputs)
    assert result.equity_curve == [10000, 10000, 11000, 8800]
    assert result.daily_returns == pytest.approx([0, 0.1, -0.2])
    assert result.metrics.total_return == pytest.approx(-0.12)
    assert result.metrics.max_drawdown == pytest.approx(-0.2)
    assert result.metrics.total_trades == 1
    assert pd.Timestamp(result.trades[0]["executed_at"]) > pd.Timestamp(
        result.trades[0]["decision_at"]
    )
    assert result.reconciliation["reconciled"]
    assert len(result.timestamps) == len(set(result.timestamps))


def test_future_perturbation_cannot_change_past_decisions_or_nav(tmp_path):
    inputs = replay_inputs()
    before = PortfolioReplay(tmp_path / "before.db").run(**inputs)
    future = inputs["signals"][-2].decision_at
    inputs["signals"] = [
        s.model_copy(update={"score": 100 if s.instrument == B else -100})
        if s.decision_at == future
        else s
        for s in inputs["signals"]
    ]
    inputs["bars"] = [
        b.model_copy(update={"close": D(22), "low": D(22)})
        if b.opened_at > future and b.instrument == A
        else b
        for b in inputs["bars"]
    ]
    after = PortfolioReplay(tmp_path / "after.db").run(**inputs)
    assert before.decisions[0] == after.decisions[0]
    assert before.equity_curve[:3] == after.equity_curve[:3]
    assert before.trades[0] == after.trades[0]


def test_replay_idempotent_after_restart_and_manifest_immutable(tmp_path):
    path = tmp_path / "replay.db"
    inputs = replay_inputs()
    one = PortfolioReplay(path).run(**inputs)
    two = PortfolioReplay(path).run(**inputs)
    assert one == two
    changed = {**inputs, "topk": 2}
    with pytest.raises(BacktestError, match="inputs changed"):
        PortfolioReplay(path).run(**changed)
    ledger = PortfolioLedger(path)
    with pytest.raises(LedgerError, match="immutable manifest"):
        ledger.save_run(one.account_id, one.run_id, {"manifest_hash": "changed"})
    stored = json.loads(ledger.db.execute("SELECT payload FROM runs").fetchone()[0])
    assert stored["status"] == "complete"
    ledger.close()


def test_costs_reconcile_to_cash_and_nav(tmp_path):
    inputs = replay_inputs()
    inputs["fill_policy"] = FillPolicy(equity_cost_bps=10)
    result = PortfolioReplay(tmp_path / "cost.db").run(**inputs)
    # 99 shares at 100; entry charge 9.90; 90.10 cash remains.
    assert result.trades[0]["quantity"] == "99"
    assert D(result.trades[0]["fee"]) == D("9.9")
    assert result.snapshots[2].cash == D("90.1")
    assert result.snapshots[2].nav == D("10980.1")
    assert result.reconciliation["reconciled"]


def test_label_only_replay_and_missing_inputs_fail():
    with pytest.raises(BacktestError, match="model_factory"):
        WalkForwardBacktester().run(pd.DataFrame({"x": [1]}), pd.Series([1]), model=object())
    args = replay_inputs()
    args["bars"] = []
    with pytest.raises(BacktestError, match="actual bars"):
        PortfolioReplay(":memory:").run(**args)


def test_cli_actual_run_and_failed_data_cannot_report_success(tmp_path):
    runner = CliRunner()
    bundle, output, ledger = (
        tmp_path / "input.json",
        tmp_path / "result.json",
        tmp_path / "ledger.db",
    )
    write_bundle(bundle, replay_inputs())
    result = runner.invoke(
        app, ["run", "--input", str(bundle), "--ledger", str(ledger), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["metrics"]["total_return"] == pytest.approx(-0.12)
    bad = replay_inputs()
    bad["bars"] = []
    write_bundle(bundle, bad)
    failed = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(bundle),
            "--ledger",
            str(tmp_path / "bad.db"),
            "--output",
            str(tmp_path / "bad.json"),
        ],
    )
    assert failed.exit_code == 1
    assert not (tmp_path / "bad.json").exists()
    assert json.loads((tmp_path / "bad.json.failure.json").read_text())["status"] == "failed"
    validation = runner.invoke(
        app,
        [
            "validate",
            "--result",
            str(output),
            "--ledger",
            str(ledger),
            "--output",
            str(tmp_path / "validation.json"),
        ],
    )
    assert validation.exit_code == 2
    assert json.loads((tmp_path / "validation.json").read_text())["overall"] == "NOT RELIABLE"


def test_missing_open_liquidity_and_future_signal_rejected():
    args = replay_inputs()
    with pytest.raises(ValueError):
        RankedSignal(
            **{
                **args["signals"][0].model_dump(),
                "available_at": args["signals"][0].decision_at + timedelta(days=1),
            }
        )
    with pytest.raises(ValueError):
        ExecutionBar(
            **{k: v for k, v in args["bars"][0].model_dump().items() if k != "opening_volume"}
        )


def test_future_universe_membership_cannot_qualify_a_signal(tmp_path):
    args = replay_inputs()
    args["universes"] = [
        args["universes"][0].model_copy(update={"available_at": args["context"].as_of})
    ]
    with pytest.raises(BacktestError, match="universe membership"):
        PortfolioReplay(tmp_path / "ledger.db").run(**args)


def test_typed_paper_and_replay_have_identical_economic_state(tmp_path):
    from src.domain.portfolio import OrderIntent
    from src.execution.paper_trader import PaperTrader

    inputs = replay_inputs()
    result = PortfolioReplay(tmp_path / "replay.db").run(**inputs)
    paper = PaperTrader(
        10000,
        ledger_path=tmp_path / "paper.db",
        account_id="paper:equivalent",
        opened_at=inputs["account"].opened_at,
        fill_policy=inputs["fill_policy"],
        risk_policy=inputs["risk_policy"],
    )
    meta = {m.instrument.instrument_id: m for m in inputs["metadata"]}
    timeline = {}
    for b in inputs["bars"]:
        timeline.setdefault(b.opened_at, {"open": [], "close": [], "decisions": []})["open"].append(
            b
        )
        timeline.setdefault(b.closed_at, {"open": [], "close": [], "decisions": []})[
            "close"
        ].append(b)
    for d in result.decisions:
        timeline[pd.Timestamp(d["target"]["decision_at"]).to_pydatetime()]["decisions"].append(d)
    for at, event in sorted(timeline.items()):
        for b in event["open"]:
            paper.execution.open_bar(b, meta)
        for b in event["close"]:
            paper.execution.close_bar(b)
        for d in event["decisions"]:
            for o in d["orders"]:
                order = OrderIntent.model_validate(
                    {**o["intent"], "account_id": "paper:equivalent"}
                )
                paper.execution.submit(order, meta)
        if event["close"]:
            snap = paper.ledger.snapshot("paper:equivalent", at, 86400)
            replay = next(s for s in result.snapshots if s.as_of == at)
            assert (snap.cash, snap.nav, snap.fees, snap.realized_pnl) == (
                replay.cash,
                replay.nav,
                replay.fees,
                replay.realized_pnl,
            )
            assert snap.positions == replay.positions
    assert paper.ledger.reconcile("paper:equivalent")["reconciled"]
