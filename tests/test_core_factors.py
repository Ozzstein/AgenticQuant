"""Tests for src.core.factor_engine.FactorEngine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.factor_engine import FactorEngine, _FACTOR_NAMES
from src.utils.exceptions import FactorError
from src.utils.schemas import FactorDefinition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _monotone_up(n: int = 100) -> pd.DataFrame:
    """Return a DataFrame with strictly increasing close prices."""
    prices = np.arange(100.0, 100.0 + n)
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.ones(n) * 1_000_000,
        },
        index=pd.bdate_range("2023-01-01", periods=n),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> FactorEngine:
    return FactorEngine()


@pytest.fixture
def aapl_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    return sample_ohlcv_df[sample_ohlcv_df["ticker"] == "AAPL"].copy()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_rsi_range(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """RSI values must be in [0, 100]."""
        rsi = engine.compute_rsi(aapl_df)
        valid = rsi.dropna()
        assert (valid >= 0).all(), "RSI below 0"
        assert (valid <= 100).all(), "RSI above 100"

    def test_rsi_perfect_up_trend(self, engine: FactorEngine) -> None:
        """RSI for monotonically increasing prices approaches 100."""
        df = _monotone_up(60)
        rsi = engine.compute_rsi(df, period=14)
        last = rsi.dropna().iloc[-1]
        assert last > 90, f"Expected RSI near 100 for up-trend, got {last}"

    def test_rsi_missing_column(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """Missing 'close' column raises FactorError."""
        with pytest.raises(FactorError):
            engine.compute_rsi(aapl_df.drop(columns=["close"]))

    def test_rsi_has_nans_at_start(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """First rows should be NaN due to lookback."""
        rsi = engine.compute_rsi(aapl_df, period=14)
        assert rsi.isna().sum() > 0


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_macd_returns_three_series(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """MACD dict must contain macd, signal, histogram."""
        result = engine.compute_macd(aapl_df)
        assert set(result.keys()) == {"macd", "signal", "histogram"}

    def test_macd_histogram_equals_diff(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """histogram = macd - signal by definition."""
        d = engine.compute_macd(aapl_df)
        diff = (d["macd"] - d["signal"]).dropna()
        hist = d["histogram"].dropna()
        pd.testing.assert_series_equal(diff, hist, check_names=False, atol=1e-10)

    def test_macd_length(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """Output length matches input length."""
        d = engine.compute_macd(aapl_df)
        assert len(d["macd"]) == len(aapl_df)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollinger:
    def test_bollinger_returns_four_series(
        self, engine: FactorEngine, aapl_df: pd.DataFrame
    ) -> None:
        """BB dict must contain upper, middle, lower, pct_b."""
        result = engine.compute_bollinger(aapl_df)
        assert set(result.keys()) == {"upper", "middle", "lower", "pct_b"}

    def test_bollinger_upper_above_lower(
        self, engine: FactorEngine, aapl_df: pd.DataFrame
    ) -> None:
        """Upper band must always be >= lower band where both are defined."""
        bb = engine.compute_bollinger(aapl_df)
        valid = pd.concat([bb["upper"], bb["lower"]], axis=1).dropna()
        assert (valid.iloc[:, 0] >= valid.iloc[:, 1]).all()

    def test_bollinger_middle_is_sma(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """Middle band should match a rolling SMA."""
        bb = engine.compute_bollinger(aapl_df, period=20)
        sma = aapl_df["close"].rolling(window=20, min_periods=20).mean()
        pd.testing.assert_series_equal(
            bb["middle"].dropna(), sma.dropna(), check_names=False, atol=1e-8
        )


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_atr_positive(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """ATR must be >= 0 for all valid rows."""
        atr = engine.compute_atr(aapl_df)
        assert (atr.dropna() >= 0).all()

    def test_atr_length(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """Output length matches input length."""
        assert len(engine.compute_atr(aapl_df)) == len(aapl_df)


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------


class TestOBV:
    def test_obv_increases_on_up_day(self, engine: FactorEngine) -> None:
        """OBV should increase when close > previous close."""
        prices = [100.0, 105.0, 110.0, 108.0]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1_000_000] * 4,
            }
        )
        obv = engine.compute_obv(df)
        # Day 2 close > day 1 close → OBV should rise
        assert obv.iloc[2] > obv.iloc[1]

    def test_obv_decreases_on_down_day(self, engine: FactorEngine) -> None:
        """OBV should decrease when close < previous close."""
        prices = [110.0, 105.0, 100.0]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1_000_000] * 3,
            }
        )
        obv = engine.compute_obv(df)
        assert obv.iloc[2] < obv.iloc[1]

    def test_obv_length(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        assert len(engine.compute_obv(aapl_df)) == len(aapl_df)


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


class TestVWAP:
    def test_vwap_positive(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """VWAP should be positive for positive prices."""
        vwap = engine.compute_vwap(aapl_df)
        assert (vwap.dropna() > 0).all()

    def test_vwap_length(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        assert len(engine.compute_vwap(aapl_df)) == len(aapl_df)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class TestMomentum:
    def test_momentum_all_periods(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """All four momentum keys must be present."""
        result = engine.compute_momentum(aapl_df)
        assert set(result.keys()) == {"mom_1m", "mom_3m", "mom_6m", "mom_12m"}

    def test_momentum_positive_up_trend(self, engine: FactorEngine) -> None:
        """Momentum should be positive for a sustained up-trend."""
        df = _monotone_up(300)
        result = engine.compute_momentum(df)
        last = result["mom_1m"].dropna().iloc[-1]
        assert last > 0


# ---------------------------------------------------------------------------
# Realized Vol
# ---------------------------------------------------------------------------


class TestRealizedVol:
    def test_realized_vol_positive(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """vol_20d and vol_60d should be positive where defined."""
        result = engine.compute_realized_vol(aapl_df)
        assert (result["vol_20d"].dropna() > 0).all()
        assert (result["vol_60d"].dropna() > 0).all()

    def test_realized_vol_returns_three_series(
        self, engine: FactorEngine, aapl_df: pd.DataFrame
    ) -> None:
        result = engine.compute_realized_vol(aapl_df)
        assert set(result.keys()) == {"vol_20d", "vol_60d", "vol_ratio"}


# ---------------------------------------------------------------------------
# Garman-Klass
# ---------------------------------------------------------------------------


class TestGarmanKlass:
    def test_garman_klass_positive(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """GK vol should be > 0 where defined."""
        gk = engine.compute_garman_klass(aapl_df)
        assert (gk.dropna() > 0).all()

    def test_garman_klass_length(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        assert len(engine.compute_garman_klass(aapl_df)) == len(aapl_df)


# ---------------------------------------------------------------------------
# Parkinson
# ---------------------------------------------------------------------------


class TestParkinson:
    def test_parkinson_positive(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        """Parkinson vol should be > 0 where defined."""
        park = engine.compute_parkinson(aapl_df)
        assert (park.dropna() > 0).all()

    def test_parkinson_length(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        assert len(engine.compute_parkinson(aapl_df)) == len(aapl_df)


# ---------------------------------------------------------------------------
# Mean Reversion
# ---------------------------------------------------------------------------


class TestMeanReversion:
    def test_mean_reversion_zscore_near_zero_mean(
        self, engine: FactorEngine, aapl_df: pd.DataFrame
    ) -> None:
        """Z-score should have mean close to 0 over long period."""
        z = engine.compute_mean_reversion(aapl_df)
        assert abs(z.dropna().mean()) < 2.0

    def test_mean_reversion_length(self, engine: FactorEngine, aapl_df: pd.DataFrame) -> None:
        assert len(engine.compute_mean_reversion(aapl_df)) == len(aapl_df)


# ---------------------------------------------------------------------------
# Batch compute_factors
# ---------------------------------------------------------------------------


class TestComputeFactors:
    def test_compute_factors_returns_dataframe(
        self, engine: FactorEngine, aapl_df: pd.DataFrame
    ) -> None:
        """compute_factors should return a pd.DataFrame."""
        result = engine.compute_factors(aapl_df)
        assert isinstance(result, pd.DataFrame)

    def test_compute_factors_all_columns_present(
        self, engine: FactorEngine, aapl_df: pd.DataFrame
    ) -> None:
        """All factor names should be columns in the output."""
        result = engine.compute_factors(aapl_df)
        for name in _FACTOR_NAMES:
            assert name in result.columns, f"Factor '{name}' not in result"

    def test_compute_factors_subset(
        self, engine: FactorEngine, aapl_df: pd.DataFrame
    ) -> None:
        """Specifying names should return only those columns."""
        subset = ["rsi", "atr", "obv"]
        result = engine.compute_factors(aapl_df, names=subset)
        assert set(result.columns) == set(subset)

    def test_compute_factors_unknown_name(
        self, engine: FactorEngine, aapl_df: pd.DataFrame
    ) -> None:
        """Unknown factor name should raise FactorError."""
        with pytest.raises(FactorError):
            engine.compute_factors(aapl_df, names=["rsi", "nonexistent_factor"])

    def test_compute_factors_multi_ticker(
        self, engine: FactorEngine, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """compute_factors should work on multi-ticker DataFrames."""
        result = engine.compute_factors(sample_ohlcv_df, names=["rsi", "atr"])
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns

    def test_handles_single_ticker(
        self, engine: FactorEngine, aapl_df: pd.DataFrame
    ) -> None:
        """Works correctly for a single-ticker slice without 'ticker' column."""
        df = aapl_df.drop(columns=["ticker"])
        result = engine.compute_factors(df, names=["rsi", "obv"])
        assert "rsi" in result.columns
        assert "obv" in result.columns
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# IC and factor evaluation
# ---------------------------------------------------------------------------


class TestFactorIC:
    def test_compute_factor_ic_perfect_foresight(self, engine: FactorEngine) -> None:
        """IC ≈ 1 when factor equals forward return (perfect foresight)."""
        np.random.seed(0)
        ret = pd.Series(np.random.randn(200))
        ic = engine.compute_factor_ic(ret, ret)
        assert ic > 0.95

    def test_compute_factor_ic_random(self, engine: FactorEngine) -> None:
        """IC ≈ 0 for uncorrelated factor and returns."""
        np.random.seed(1)
        factor = pd.Series(np.random.randn(500))
        returns = pd.Series(np.random.randn(500))
        ic = engine.compute_factor_ic(factor, returns)
        assert abs(ic) < 0.3

    def test_compute_factor_ic_anticorrelated(self, engine: FactorEngine) -> None:
        """IC ≈ -1 for perfectly anti-correlated factor."""
        np.random.seed(2)
        ret = pd.Series(np.random.randn(200))
        ic = engine.compute_factor_ic(-ret, ret)
        assert ic < -0.95


class TestEvaluateFactor:
    def test_evaluate_factor_perfect_foresight(self, engine: FactorEngine) -> None:
        """IC ≈ 1.0 for a perfect (factor = return) predictor."""
        np.random.seed(10)
        ret = pd.Series(np.random.randn(252))
        result = engine.evaluate_factor(ret, ret)
        assert result["ic_mean"] > 0.7
        assert result["pct_positive"] >= 0.5

    def test_evaluate_factor_random(self, engine: FactorEngine) -> None:
        """IC ≈ 0 for a random factor (within [-0.3, 0.3])."""
        np.random.seed(20)
        factor = pd.Series(np.random.randn(252))
        returns = pd.Series(np.random.randn(252))
        result = engine.evaluate_factor(factor, returns)
        assert abs(result["ic_mean"]) < 0.3

    def test_evaluate_factor_keys(self, engine: FactorEngine) -> None:
        """Result dict must contain all required keys."""
        np.random.seed(30)
        f = pd.Series(np.random.randn(100))
        r = pd.Series(np.random.randn(100))
        result = engine.evaluate_factor(f, r)
        assert set(result.keys()) == {"ic_mean", "ic_std", "icir", "t_stat", "pct_positive", "turnover"}

    def test_evaluate_factor_icir_finite(self, engine: FactorEngine) -> None:
        """ICIR should be a finite float."""
        np.random.seed(40)
        f = pd.Series(np.random.randn(252))
        r = pd.Series(np.random.randn(252))
        result = engine.evaluate_factor(f, r)
        assert np.isfinite(result["icir"])


class TestEvaluateFactors:
    def test_evaluate_factors_dataframe(self, engine: FactorEngine) -> None:
        """evaluate_factors returns DataFrame with correct columns."""
        np.random.seed(50)
        idx = pd.RangeIndex(200)
        factors = pd.DataFrame(
            {"rsi": np.random.randn(200), "mom": np.random.randn(200)},
            index=idx,
        )
        returns = pd.Series(np.random.randn(200), index=idx)
        result = engine.evaluate_factors(factors, returns)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"ic_mean", "ic_std", "icir", "t_stat", "pct_positive", "turnover"}
        assert expected_cols.issubset(set(result.columns))
        assert set(result.index) == {"rsi", "mom"}


# ---------------------------------------------------------------------------
# Quintile returns
# ---------------------------------------------------------------------------


class TestQuintileReturns:
    def test_quintile_returns_five_rows(self, engine: FactorEngine) -> None:
        """Should return 5 bins for quintile split."""
        np.random.seed(60)
        factor = pd.Series(np.random.randn(200))
        returns = pd.Series(np.random.randn(200))
        result = engine.compute_quintile_returns(factor, returns)
        assert len(result) == 5

    def test_quintile_returns_q5_gt_q1_for_perfect_factor(
        self, engine: FactorEngine
    ) -> None:
        """Q5 should have higher return than Q1 when factor = return."""
        np.random.seed(70)
        ret = pd.Series(np.random.randn(500))
        result = engine.compute_quintile_returns(ret, ret)
        assert result.iloc[4] > result.iloc[0], "Q5 should be > Q1 for perfect factor"


# ---------------------------------------------------------------------------
# Factor library save/load
# ---------------------------------------------------------------------------


class TestFactorLibrary:
    def test_save_load_factor_library(
        self, engine: FactorEngine, tmp_path: Path
    ) -> None:
        """Round-trip save/load produces the same FactorDefinition."""
        lib_path = str(tmp_path / "factor_library.json")
        fd = FactorDefinition(
            name="test_rsi",
            expression="RSI(14)",
            category="technical",
            ic_mean=0.05,
            icir=0.45,
            source="builtin",
            description="14-day RSI",
        )
        engine.save_factor(fd, path=lib_path)
        loaded = engine.load_factor_library(path=lib_path)
        assert len(loaded) == 1
        assert loaded[0].name == "test_rsi"
        assert loaded[0].ic_mean == pytest.approx(0.05)

    def test_save_factor_updates_existing(
        self, engine: FactorEngine, tmp_path: Path
    ) -> None:
        """Saving a factor with the same name updates it."""
        lib_path = str(tmp_path / "factor_library.json")
        fd1 = FactorDefinition(name="rsi", ic_mean=0.01)
        fd2 = FactorDefinition(name="rsi", ic_mean=0.09)
        engine.save_factor(fd1, path=lib_path)
        engine.save_factor(fd2, path=lib_path)
        loaded = engine.load_factor_library(path=lib_path)
        assert len(loaded) == 1
        assert loaded[0].ic_mean == pytest.approx(0.09)

    def test_save_multiple_factors(
        self, engine: FactorEngine, tmp_path: Path
    ) -> None:
        """Multiple distinct factors are all persisted."""
        lib_path = str(tmp_path / "factor_library.json")
        for name in ["rsi", "macd", "atr"]:
            engine.save_factor(FactorDefinition(name=name), path=lib_path)
        loaded = engine.load_factor_library(path=lib_path)
        assert {f.name for f in loaded} == {"rsi", "macd", "atr"}

    def test_load_missing_file_returns_empty(
        self, engine: FactorEngine, tmp_path: Path
    ) -> None:
        """Loading from a non-existent path returns empty list."""
        missing = str(tmp_path / "no_such_file.json")
        result = engine.load_factor_library(path=missing)
        assert result == []
