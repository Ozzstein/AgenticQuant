"""Synthetic causal model evaluations, purging and failed-training acceptance cases."""

from datetime import timedelta
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.core.backtester import split_folds
from src.core.walk_forward import WalkForwardRunner, cross_sectional_ic
from src.domain.common import Instrument, RunContext
from src.domain.portfolio import (
    AccountSpec,
    ExecutionBar,
    FillPolicy,
    InstrumentRisk,
    RiskPolicy,
    UniverseEligibility,
)
from src.utils.exceptions import BacktestError


def wf_case(tmp_path, *, two_folds=False):
    # Shorter initial training is explicit in test; OOS remains six months.
    dates = pd.date_range(
        "2023-01-02 20:00",
        "2024-03-07 20:00" if two_folds else "2023-09-07 20:00",
        tz="UTC",
        freq="D",
    )
    instruments = [
        Instrument(instrument_id=s, symbol=s, asset_class="crypto", quote_currency="USD")
        for s in ("A", "B", "C")
    ]
    rows, ys, ends, bars, metadata = [], [], [], [], []
    index = []
    for day, at in enumerate(dates):
        for j, instrument in enumerate(instruments):
            opening = D(str(round(100 + 0.02 * day + j * 3 + np.sin(day / 13 + j), 6)))
            close = opening * D(str(round(1 + 0.002 * np.sin(day / 7 + j), 8)))
            midnight = at.normalize()
            bars.append(
                ExecutionBar(
                    event_id=f"{j}-{day}",
                    instrument=instrument,
                    opened_at=at.to_pydatetime() - timedelta(hours=6),
                    closed_at=at.to_pydatetime(),
                    open=opening,
                    high=max(opening, close),
                    low=min(opening, close),
                    close=close,
                    volume=1000000,
                    opening_volume=100000,
                    calendar="24/7",
                    calendar_version="fixture",
                    session=at.date(),
                    session_open=midnight,
                    session_close=midnight + pd.Timedelta(days=1),
                )
            )
            metadata.append(
                InstrumentRisk(
                    instrument=instrument,
                    sector=str(j),
                    adv_base=100000000,
                    as_of=midnight,
                    lot_size=".01",
                )
            )
            if day < len(dates) - 1:
                index.append((at, instrument.instrument_id))
                rows.append([np.sin(day / 7 + j), float(close / opening - 1)])
                ys.append(0.002 * np.sin((day + 1) / 7 + j))
                ends.append(at + pd.Timedelta(days=1))
    idx = pd.MultiIndex.from_tuples(index, names=["decision_at", "instrument_id"])
    features, labels = (
        pd.DataFrame(rows, index=idx, columns=["lag", "return"]),
        pd.Series(ys, index=idx),
    )
    label_end, available = pd.Series(ends, index=idx), pd.Series(idx.get_level_values(0), index=idx)
    account = AccountSpec(
        account_id="backtest:wf",
        mode="backtest",
        base_currency="USD",
        initial_cash=10000,
        opened_at=dates[0] - pd.Timedelta(days=1),
    )
    context = RunContext(
        run_id="wf",
        account_id=account.account_id,
        mode="backtest",
        as_of=dates[-1],
        config_hash="2" * 64,
        code_version="test",
        data_snapshot_ids=("fixture",),
    )
    kwargs = dict(
        label_end_times=label_end,
        feature_available_at=available,
        bars=bars,
        metadata=metadata,
        context=context,
        account=account,
        universes=[
            UniverseEligibility(
                universe_version="fixture",
                data_snapshot_id="fixture",
                effective_at=account.opened_at,
                available_at=account.opened_at,
                instrument_ids=("A", "B", "C"),
            )
        ],
        ledger_path=tmp_path / "wf.db",
        artifact_dir=tmp_path / "models",
        universe_version="fixture",
        initial_train_months=1,
        validation_months=1,
        embargo_days=5,
        fill_policy=FillPolicy(equity_cost_bps=10, crypto_cost_bps=20),
        risk_policy=RiskPolicy(),
        samples_per_year=365,
        valuation_calendar="24/7",
        topk=1,
    )
    return features, labels, kwargs


def test_six_month_expanding_splits_honor_train_start():
    dates = pd.date_range("2020-01-01", "2024-01-01", tz="UTC")
    folds = split_folds(dates, "2021-01-01")
    assert pd.Timestamp(folds[0]["train_start"]) == pd.Timestamp("2021-01-01", tz="UTC")
    assert pd.Timestamp(folds[0]["test_end"]) == pd.Timestamp(
        folds[0]["test_start"]
    ) + pd.DateOffset(months=6)
    assert folds[1]["train_start"] == folds[0]["train_start"]
    assert folds[1]["train_end"] > folds[0]["train_end"]
    assert folds[0]["test_end"] == folds[1]["test_start"]


def test_real_oos_nav_purging_and_holdings_carried(tmp_path):
    x, y, args = wf_case(tmp_path, two_folds=True)
    instances = []

    def factory():
        instance = Ridge(alpha=1)
        instances.append(instance)
        return instance

    result = WalkForwardRunner().run(x, y, model_factory=factory, **args)
    assert len(instances) == len(result.walk_forward_results) == 2
    assert result.reconciliation["reconciled"]
    assert result.metrics.total_trades > 0
    assert len(result.timestamps) == len(set(result.timestamps))
    assert len(result.daily_returns) == len(result.equity_curve) - 1
    assert np.std(result.daily_returns) > 0
    assert len(list((tmp_path / "models").glob("fold-*.joblib"))) == 2
    for fold in result.walk_forward_results:
        assert fold.purged_rows > 0
        assert (
            pd.Timestamp(fold.label_end_max)
            < pd.Timestamp(fold.train_end)
            < pd.Timestamp(fold.validation_start)
        )
    boundary = pd.Timestamp(result.walk_forward_results[1].test_start)
    before = [s for s in result.snapshots if s.as_of < boundary][-1]
    after = [s for s in result.snapshots if s.as_of >= boundary][0]
    assert before.positions and after.positions
    assert before.cash != args["account"].initial_cash
    assert abs(float(after.nav / before.nav) - 1) < 0.02  # no capital reset at fold boundary


def test_future_feature_perturbation_does_not_rewrite_first_fold(tmp_path):
    x, y, args = wf_case(tmp_path / "one", two_folds=True)
    one = WalkForwardRunner().run(x, y, model_factory=Ridge, **args)
    boundary = pd.Timestamp(one.walk_forward_results[1].test_start)
    changed = x.copy()
    changed.loc[changed.index.get_level_values(0) >= boundary] *= 100
    args.update(ledger_path=tmp_path / "two" / "wf.db", artifact_dir=tmp_path / "two" / "models")
    two = WalkForwardRunner().run(changed, y, model_factory=Ridge, **args)
    assert [d for d in one.decisions if pd.Timestamp(d["target"]["decision_at"]) < boundary] == [
        d for d in two.decisions if pd.Timestamp(d["target"]["decision_at"]) < boundary
    ]
    assert [(s.as_of, s.nav) for s in one.snapshots if s.as_of < boundary] == [
        (s.as_of, s.nav) for s in two.snapshots if s.as_of < boundary
    ]


def test_future_availability_overlap_and_incomplete_folds(tmp_path):
    x, y, args = wf_case(tmp_path)
    args["feature_available_at"] = args["feature_available_at"] + pd.Timedelta(days=1)
    with pytest.raises(BacktestError, match="future feature"):
        WalkForwardRunner().run(x, y, model_factory=Ridge, **args)
    args["feature_available_at"] -= pd.Timedelta(days=1)
    # Labels spanning all of training and validation leave no eligible model fit.
    args["label_end_times"] = args["label_end_times"] + pd.Timedelta(days=400)
    with pytest.raises(BacktestError, match="empty purged"):
        WalkForwardRunner().run(x, y, model_factory=Ridge, **args)
    assert not (tmp_path / "models" / "result.json").exists()


def test_training_failure_never_returns_performance(tmp_path):
    x, y, args = wf_case(tmp_path)

    class Failed:
        def fit(self, *args):
            raise RuntimeError("training failed")

    with pytest.raises(BacktestError, match="training failed"):
        WalkForwardRunner().run(x, y, model_factory=Failed, **args)
    assert (tmp_path / "models" / "failure.json").exists()
    assert not (tmp_path / "models" / "result.json").exists()


def test_ic_is_cross_sectional_and_undefined_is_preserved():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2, tz="UTC"), ["A", "B", "C"]]
    )
    values = cross_sectional_ic(
        pd.Series([1, 2, 3, 3, 2, 1], index=idx), pd.Series([2, 3, 4, 2, 3, 4], index=idx)
    )
    assert list(values.values()) == pytest.approx([1, -1])
    undefined = cross_sectional_ic(pd.Series([1] * 6, index=idx), pd.Series([1] * 6, index=idx))
    assert list(undefined.values()) == [None, None]


@pytest.mark.parametrize("name", ["Linear", "LightGBM", "CatBoost", "XGBoost", "Ensemble"])
def test_real_model_family_executes_oos_ledger(name, tmp_path):
    from src.core.ensemble import EnsembleModel
    from src.core.model_zoo import ModelWrapper
    from src.utils.config import AppConfig

    x, y, args = wf_case(tmp_path)
    cfg = AppConfig(output_dir=str(tmp_path / "output"), log_dir=str(tmp_path / "logs"))

    def factory():
        if name == "Ensemble":
            return EnsembleModel(model_names=["Linear"], weighting="equal", config=cfg)
        model = ModelWrapper(name, cfg)
        if name == "LightGBM":
            model._params.update(n_estimators=4, n_jobs=1, min_child_samples=5)
        elif name == "CatBoost":
            model._params.update(iterations=4, thread_count=1, allow_writing_files=False)
        elif name == "XGBoost":
            model._params.update(n_estimators=4, n_jobs=1)
        return model

    result = WalkForwardRunner(cfg).run(x, y, model_factory=factory, **args)
    assert result.metrics.total_trades > 0
    assert result.walk_forward_results[0].model_version
    assert result.reconciliation["reconciled"]


def test_incomplete_window_and_cli_failed_training_are_non_success(tmp_path):
    import json

    from typer.testing import CliRunner

    from scripts.run_backtest import app
    from tests.test_core_backtest import write_bundle

    x, y, args = wf_case(tmp_path / "partial")
    cutoff = x.index.get_level_values(0).max() - pd.Timedelta(days=30)
    mask = x.index.get_level_values(0) <= cutoff
    args["label_end_times"] = args["label_end_times"].loc[mask]
    args["feature_available_at"] = args["feature_available_at"].loc[mask]
    result = WalkForwardRunner().run(x.loc[mask], y.loc[mask], model_factory=Ridge, **args)
    assert result.status == "incomplete"
    assert "promotion prohibited" in result.diagnostics[-1]
    # Exercise a real unsupported model/training failure through the public CLI.
    bundle = tmp_path / "bad-training.json"
    inputs = {
        k: args[k]
        for k in (
            "context",
            "account",
            "bars",
            "metadata",
            "universes",
            "fill_policy",
            "risk_policy",
            "topk",
            "samples_per_year",
        )
    }
    write_bundle(bundle, inputs)
    payload = json.loads(bundle.read_text())
    payload["walk_forward"] = {
        "model_name": "MissingModel",
        "universe_version": "fixture",
        "initial_train_months": 1,
        "validation_months": 1,
        "rows": [
            {
                "decision_at": i[0].isoformat(),
                "instrument_id": i[1],
                "features": row.to_dict(),
                "label": float(y.loc[i]),
                "label_end_at": (i[0] + pd.Timedelta(days=1)).isoformat(),
                "feature_available_at": i[0].isoformat(),
            }
            for i, row in x.iterrows()
        ],
    }
    bundle.write_text(json.dumps(payload))
    output = tmp_path / "failed.json"
    invocation = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(bundle),
            "--ledger",
            str(tmp_path / "failed.db"),
            "--output",
            str(output),
        ],
    )
    assert invocation.exit_code == 1, invocation.output
    assert not output.exists()
    assert json.loads((tmp_path / "failed.json.failure.json").read_text())["status"] == "failed"


class TemporalRidge:
    """Real numerical fit with the declared temporal API, registered by test below."""

    def train(
        self,
        x,
        y,
        *,
        context,
        feature_available_at,
        label_available_at,
        feature_spec_hash,
        data_snapshot_hash,
    ):
        assert x.index.names == ["datetime", "instrument"]
        assert x.index.equals(feature_available_at.index) == x.index.equals(
            label_available_at.index
        )
        assert (label_available_at <= context.as_of).all()
        assert (feature_available_at <= context.as_of).all()
        self.fit_context = context
        self.label_times = label_available_at
        self.feature_times = feature_available_at
        self.spec_hash, self.data_hash = feature_spec_hash, data_snapshot_hash
        self.model = Ridge().fit(x, y)
        self.prediction_contexts = []

    def predict(self, x, *, context, feature_available_at):
        dates = x.index.get_level_values("datetime")
        assert dates.nunique() == 1
        assert (dates > self.fit_context.as_of).all()
        assert (dates <= context.as_of).all()
        assert feature_available_at.index.equals(x.index)
        assert (feature_available_at.to_numpy() <= dates.to_numpy()).all()
        self.prediction_contexts.append(context)
        return self.model.predict(x)


def test_contextual_metadata_is_purged_aligned_and_prediction_date_local(tmp_path, monkeypatch):
    import src.core.ensemble as ensemble
    import src.core.model_zoo as zoo

    # Protocol unit test: numerical fitting is real, registration is test-only.
    monkeypatch.setattr(zoo, "ModelWrapper", TemporalRidge)
    monkeypatch.setattr(ensemble, "ModelWrapper", TemporalRidge)
    x, y, args = wf_case(tmp_path)
    models = []

    def factory():
        model = TemporalRidge()
        models.append(model)
        return model

    result = WalkForwardRunner().run(
        x,
        y,
        model_factory=factory,
        feature_spec_hash="a" * 64,
        data_snapshot_hash="b" * 64,
        **args,
    )
    assert result.metrics.total_trades > 0
    assert result.manifest["provenance"]["temporal_training"] == "contextual-v1"
    for fold, model in zip(result.walk_forward_results, models, strict=True):
        assert (
            pd.Timestamp(fold.train_end) - pd.Timedelta(microseconds=1) == model.fit_context.as_of
        )
        mask = (x.index.get_level_values(0) < pd.Timestamp(fold.train_end)) & (
            args["label_end_times"] < pd.Timestamp(fold.train_end)
        )
        assert model.label_times.equals(
            args["label_end_times"].loc[mask].rename_axis(["datetime", "instrument"])
        )
        assert model.spec_hash == "a" * 64 and model.data_hash == "b" * 64
        assert all(c.as_of > model.fit_context.as_of for c in model.prediction_contexts)


def test_contextual_fit_rejects_unsupported_protocol_and_unpaired_hashes(tmp_path):
    x, y, args = wf_case(tmp_path)
    with pytest.raises(BacktestError, match="must be SHA256"):
        WalkForwardRunner().run(x, y, model_factory=Ridge, feature_spec_hash="a" * 64, **args)
    with pytest.raises(BacktestError, match="requires ModelWrapper or EnsembleModel"):
        WalkForwardRunner().run(
            x,
            y,
            model_factory=Ridge,
            feature_spec_hash="a" * 64,
            data_snapshot_hash="b" * 64,
            **args,
        )
    assert not (tmp_path / "models" / "result.json").exists()
