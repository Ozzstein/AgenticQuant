"""Synthetic fixtures exercise real HMM fitting/filtering; no provider claims."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.core.macro_regime import MACRO_SIGNALS, CausalHMMRegimeDetector, macro_signals
from src.data.evidence.store import canonical_bytes, digest
from src.domain.common import EvidenceRecord, RunContext
from src.utils.exceptions import RegimeError


@dataclass
class Inputs:
    raw: pd.DataFrame
    available: pd.Series
    sources: pd.Series
    records: tuple

    def through(self, count):
        raw = self.raw.iloc[:count]
        context = RunContext(run_id="causal-test", mode="backtest", as_of=raw.index[-1], config_hash="0"*64, code_version="test")
        return raw, context, dict(available_at=self.available.iloc[:count], source_ids=self.sources.iloc[:count], evidence=self.records[:count])


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(44)
    index = pd.date_range("2020-01-01", periods=180, freq="B", tz="UTC")
    raw = pd.DataFrame(rng.normal(size=(180, 8)).cumsum(axis=0), index=index, columns=MACRO_SIGNALS)
    records = tuple(EvidenceRecord(evidence_id=f"row-{i}", provider="test", provider_version="1", source_uri=f"archive:{i}",
                                  raw_hash=digest(canonical_bytes(row.tolist())), available_at=t, requested_as_of=t,
                                  observed_at=t, ingested_at=t, quality="usable", availability_basis="documented")
                    for i, (t, row) in enumerate(raw.iterrows()))
    return Inputs(raw, pd.Series(index, index=index), pd.Series([(e.evidence_id,) for e in records], index=index), records)


@pytest.fixture
def fitted(inputs, tmp_path):
    model = CausalHMMRegimeDetector(tmp_path / "state.sqlite", normalization_window=20, n_init=1, n_iter=20)
    raw, context, provenance = inputs.through(140)
    model.fit(raw, context, **provenance)
    return model


def test_actual_fit_filter_restart_and_future_extension(inputs, fitted, tmp_path):
    raw, context, provenance = inputs.through(150)
    before = fitted.predict(raw, context, **provenance)
    assert before.status == "available"
    assert sum(dict(before.state_probabilities).values()) == pytest.approx(1)
    artifact = fitted.save(tmp_path / "model.joblib")
    restarted = CausalHMMRegimeDetector(tmp_path / "state.sqlite").load(artifact)
    assert restarted.predict(raw, context, **provenance) == before
    all_raw, all_context, all_provenance = inputs.through(170)
    future = all_raw.copy()
    future.iloc[150:] *= -1000
    after = restarted.predict(future, all_context, **all_provenance)
    assert after.context.as_of == all_context.as_of
    with restarted._db() as db:
        persisted = db.execute("SELECT result FROM regime_events WHERE timestamp=?", (raw.index[-1].isoformat(),)).fetchone()[0]
    assert before.model_validate_json(persisted) == before
    # Fresh state must obtain exactly the same prefix by filtering a longer run.
    fresh = CausalHMMRegimeDetector(tmp_path / "fresh.sqlite").load(artifact)
    fresh.predict(all_raw, all_context, **all_provenance)
    with fresh._db() as db:
        earlier = db.execute("SELECT result FROM regime_events WHERE timestamp=?", (raw.index[-1].isoformat(),)).fetchone()[0]
    assert before.model_validate_json(earlier) == before


def test_revised_training_and_recorded_prefix_rejected(inputs, fitted):
    raw, context, provenance = inputs.through(150)
    fitted.predict(raw, context, **provenance)
    revised = raw.copy()
    revised.iloc[1, 0] += 1
    with pytest.raises(RegimeError, match="training history"):
        fitted.predict(revised, context, **provenance)
    revised = raw.copy()
    revised.iloc[145, 0] += 1
    with pytest.raises(RegimeError, match="historical regime inputs changed"):
        fitted.predict(revised, context, **provenance)


def test_future_and_late_evidence_rejected(inputs, tmp_path):
    raw, context, provenance = inputs.through(140)
    model = CausalHMMRegimeDetector(tmp_path / "state")
    with pytest.raises(RegimeError, match="future"):
        model.fit(raw, context.model_copy(update={"as_of": raw.index[-2].to_pydatetime()}), **provenance)
    late = provenance["available_at"].copy()
    late.iloc[1] = raw.index[-1]
    with pytest.raises(RegimeError, match="historical decision"):
        model.fit(raw, context, **{**provenance, "available_at": late})
    sources = provenance["source_ids"].copy()
    sources.iloc[1] = sources.iloc[-1]
    with pytest.raises(ValueError, match="unavailable"):
        model.fit(raw, context, **{**provenance, "source_ids": sources})


def test_unknown_and_low_confidence_never_report_neutral(inputs, tmp_path):
    raw, context, provenance = inputs.through(140)
    model = CausalHMMRegimeDetector(tmp_path / "state", normalization_window=20, n_init=1, n_iter=20, confidence_threshold=1)
    absent = model.predict(raw, context, **provenance)
    assert absent.status == "insufficient_data" and absent.regime is None
    model.fit(raw, context, **provenance)
    # Controlled probability fixture verifies gating independently of fitted data.
    model._forward = lambda values, previous=None: np.tile([.3, .25, .25, .2], (len(values), 1))
    raw, context, provenance = inputs.through(150)
    uncertain = model.predict(raw, context, **provenance)
    assert uncertain.status == "insufficient_data" and uncertain.regime is None
    assert uncertain.held_reason == "confidence_below_threshold"


def test_minimum_duration_survives_restart(inputs, fitted, tmp_path):
    fitted.min_days = 5
    # Initial winner then opposite winner: hold first state for five observations.
    fitted._forward = lambda values, previous=None: np.tile([.97, .01, .01, .01], (len(values), 1))
    raw, context, provenance = inputs.through(141)
    first = fitted.predict(raw, context, **provenance)
    del fitted._forward
    artifact = fitted.save(tmp_path / "restart.joblib")
    restarted = CausalHMMRegimeDetector(tmp_path / "state.sqlite").load(artifact)
    restarted._forward = lambda values, previous=None: np.tile([.01, .97, .01, .01], (len(values), 1))
    raw, context, provenance = inputs.through(145)
    held = restarted.predict(raw, context, **provenance)
    assert held.regime == first.regime and held.held_reason == "minimum_duration"
    assert held.confidence == pytest.approx(.01)  # effective state, not winning state's confidence
    raw, context, provenance = inputs.through(146)
    changed = restarted.predict(raw, context, **provenance)
    assert changed.regime != first.regime


def test_actual_breadth_and_yield_slope_no_proxy():
    index = pd.date_range("2020-01-01", periods=220, tz="UTC")
    t = np.arange(220)
    prices = pd.DataFrame({"rising": 100+t, "falling": 400-t}, index=index)
    membership = pd.DataFrame(True, index=index, columns=prices.columns)
    membership.loc[index[-1], "falling"] = False
    raw = pd.DataFrame({"vix": 20., "dgs10": 4., "dgs2": 5., "sp500": 100+t, "dxy": 100., "hyg": 80., "tlt": 100.}, index=index)
    signals = macro_signals(raw, prices, membership)
    assert signals.iloc[-2].sp500_breadth == .5
    assert signals.iloc[-1].sp500_breadth == 1
    assert signals.iloc[-1].yield_curve_10y2y == -1
    prices.iloc[-2, 0] = np.nan
    assert np.isnan(macro_signals(raw, prices, membership).iloc[-2].sp500_breadth)
