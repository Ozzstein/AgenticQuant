"""Tests for CryptoFactorEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.crypto_factors import CryptoFactorEngine
from src.utils.exceptions import FactorError

N = 100  # enough data for all factors
DATES = pd.date_range("2023-01-01", periods=N, freq="D")


@pytest.fixture
def engine():
    from src.utils.config import AppConfig

    return CryptoFactorEngine(config=AppConfig())


@pytest.fixture
def funding_df():
    np.random.seed(42)
    return pd.DataFrame(
        {"funding_rate": np.random.normal(0.0001, 0.0002, N)},
        index=DATES,
    )


@pytest.fixture
def price_df():
    np.random.seed(42)
    prices = 40000 * np.cumprod(1 + np.random.normal(0.001, 0.02, N))
    return pd.DataFrame({"close": prices}, index=DATES)


@pytest.fixture
def oi_df():
    np.random.seed(42)
    return pd.DataFrame(
        {"open_interest": np.abs(np.random.normal(1e9, 1e8, N))},
        index=DATES,
    )


@pytest.fixture
def price_series():
    np.random.seed(42)
    prices = 40000 * np.cumprod(1 + np.random.normal(0.001, 0.02, N))
    return pd.Series(prices, index=DATES, name="close")


@pytest.fixture
def returns_series(price_series):
    return price_series.pct_change().dropna()


@pytest.fixture
def mcap_series():
    np.random.seed(42)
    return pd.Series(np.abs(np.random.normal(1e12, 1e11, N)), index=DATES)


@pytest.fixture
def volume_series():
    np.random.seed(42)
    return pd.Series(np.abs(np.random.normal(5e10, 1e10, N)), index=DATES)


# --- funding_rate_zscore ---


def test_funding_rate_zscore_returns_series(engine, funding_df):
    result = engine.funding_rate_zscore(funding_df)
    assert isinstance(result, pd.Series)


def test_funding_rate_zscore_nan_prefix(engine, funding_df):
    result = engine.funding_rate_zscore(funding_df, window=30)
    # First 29 rows should be NaN (rolling window requires `window` periods)
    assert result.iloc[:29].isna().all()


def test_funding_rate_zscore_values_valid(engine, funding_df):
    result = engine.funding_rate_zscore(funding_df, window=30)
    non_nan = result.dropna()
    assert len(non_nan) > 0
    assert np.isfinite(non_nan.values).all()


def test_funding_rate_missing_column(engine):
    bad_df = pd.DataFrame({"wrong_col": [0.1, 0.2, 0.3]})
    with pytest.raises(FactorError):
        engine.funding_rate_zscore(bad_df)


# --- oi_price_divergence ---


def test_oi_price_divergence_returns_series(engine, oi_df, price_df):
    result = engine.oi_price_divergence(oi_df, price_df)
    assert isinstance(result, pd.Series)


def test_oi_price_divergence_bounded(engine, oi_df, price_df):
    result = engine.oi_price_divergence(oi_df, price_df)
    non_nan = result.dropna()
    assert len(non_nan) > 0
    assert (non_nan >= -1.0).all() and (non_nan <= 1.0).all()


# --- btc_dominance_signal ---


def test_btc_dominance_signal_returns_series(engine, mcap_series):
    np.random.seed(7)
    total_mcap = mcap_series * np.random.uniform(2.0, 3.0, N)
    result = engine.btc_dominance_signal(mcap_series, total_mcap)
    assert isinstance(result, pd.Series)


def test_btc_dominance_signal_nan_prefix(engine, mcap_series):
    np.random.seed(7)
    total_mcap = mcap_series * np.random.uniform(2.0, 3.0, N)
    result = engine.btc_dominance_signal(mcap_series, total_mcap)
    # First 29 rows should be NaN (window=30)
    assert result.iloc[:29].isna().all()


# --- beta_adjusted_momentum ---


def test_beta_adjusted_momentum_returns_series(engine, returns_series):
    np.random.seed(42)
    btc_returns = pd.Series(
        np.random.normal(0.001, 0.025, len(returns_series)),
        index=returns_series.index,
    )
    result = engine.beta_adjusted_momentum(returns_series, btc_returns)
    assert isinstance(result, pd.Series)


# --- realized_vol_cone ---


def test_realized_vol_cone_range(engine, price_series):
    # Use smaller lookback to get non-NaN values within N=100
    result = engine.realized_vol_cone(price_series, window=10, lookback=50)
    non_nan = result.dropna()
    assert len(non_nan) > 0
    assert (non_nan >= 0.0).all() and (non_nan <= 1.0).all()


# --- nvt_proxy ---


def test_nvt_proxy_returns_series(engine, mcap_series, volume_series):
    result = engine.nvt_proxy(mcap_series, volume_series)
    assert isinstance(result, pd.Series)


def test_nvt_proxy_nan_prefix(engine, mcap_series, volume_series):
    result = engine.nvt_proxy(mcap_series, volume_series, window=30)
    # First 29 rows should be NaN
    assert result.iloc[:29].isna().all()


# --- futures_spot_ratio ---


def test_futures_spot_ratio_returns_series(engine, volume_series):
    np.random.seed(99)
    spot_vol = pd.Series(np.abs(np.random.normal(3e10, 5e9, N)), index=DATES)
    result = engine.futures_spot_ratio(volume_series, spot_vol)
    assert isinstance(result, pd.Series)


def test_futures_spot_ratio_nan_prefix(engine, volume_series):
    np.random.seed(99)
    spot_vol = pd.Series(np.abs(np.random.normal(3e10, 5e9, N)), index=DATES)
    result = engine.futures_spot_ratio(volume_series, spot_vol, window=20)
    # First 19 rows should be NaN
    assert result.iloc[:19].isna().all()


# --- all factors produce non-NaN values after warmup ---


def test_all_factors_sufficient_data(engine, funding_df, oi_df, price_df, price_series, mcap_series, volume_series):
    np.random.seed(42)
    returns = price_series.pct_change().dropna()
    btc_returns = pd.Series(
        np.random.normal(0.001, 0.025, len(returns)), index=returns.index
    )
    total_mcap = mcap_series * 2.5
    spot_vol = pd.Series(np.abs(np.random.normal(3e10, 5e9, N)), index=DATES)

    results = {
        "funding_zscore": engine.funding_rate_zscore(funding_df, window=30),
        "oi_divergence": engine.oi_price_divergence(oi_df, price_df, window=20),
        "btc_dominance": engine.btc_dominance_signal(mcap_series, total_mcap),
        "beta_mom": engine.beta_adjusted_momentum(returns, btc_returns, window=30),
        "nvt": engine.nvt_proxy(mcap_series, volume_series, window=30),
        "futures_spot": engine.futures_spot_ratio(volume_series, spot_vol, window=20),
    }

    for name, series in results.items():
        assert isinstance(series, pd.Series), f"{name} should be pd.Series"
        non_nan = series.dropna()
        assert len(non_nan) > 0, f"{name} should have non-NaN values with N={N} rows"
