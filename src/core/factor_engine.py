"""Factor Engine: custom alpha factor computation and IC evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import AppConfig, get_config
from src.utils.exceptions import FactorError
from src.utils.logger import get_logger
from src.utils.schemas import FactorDefinition

logger = get_logger(__name__)

_FACTOR_LIBRARY_FILE = "factor_library.json"

# Mapping from factor name to method name for batch computation
_FACTOR_NAMES = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_pct_b",
    "atr",
    "obv",
    "vwap",
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "mom_12m",
    "vol_20d",
    "vol_60d",
    "vol_ratio",
    "garman_klass",
    "parkinson",
    "mean_reversion",
]


class FactorEngine:
    """Computes custom alpha factors from OHLCV data using pure numpy/pandas.

    All factor computations operate on a DataFrame with columns:
    open, high, low, close, volume.  NaN values are returned at the start of
    each series where the lookback period is not yet satisfied.

    Args:
        config: Optional AppConfig instance.  Falls back to get_config().
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or get_config()
        self._output_dir = Path(self._config.output_dir)

    # ------------------------------------------------------------------
    # Individual factor computations
    # ------------------------------------------------------------------

    def compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index using Wilder's EMA method.

        Args:
            df: OHLCV DataFrame with a 'close' column.
            period: Lookback period (default 14).

        Returns:
            pd.Series of RSI values in [0, 100], NaN for initial rows.

        Raises:
            FactorError: If 'close' column is missing.
        """
        self._require_columns(df, ["close"])
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Wilder's smoothing = EMA with alpha = 1/period
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # When avg_loss == 0 (pure up-trend), RSI should be 100
        rsi = rsi.where(avg_loss != 0, 100.0)
        return rsi.rename("rsi")

    def compute_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict[str, pd.Series]:
        """Compute MACD, signal line, and histogram.

        Args:
            df: OHLCV DataFrame with a 'close' column.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal EMA period.

        Returns:
            dict with keys 'macd', 'signal', 'histogram'.
        """
        self._require_columns(df, ["close"])
        close = df["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd": macd_line.rename("macd"),
            "signal": signal_line.rename("signal"),
            "histogram": histogram.rename("histogram"),
        }

    def compute_bollinger(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_mult: float = 2.0,
    ) -> dict[str, pd.Series]:
        """Compute Bollinger Bands (upper, middle, lower) and %B indicator.

        Args:
            df: OHLCV DataFrame with a 'close' column.
            period: Rolling window for SMA and std.
            std_mult: Number of standard deviations for band width.

        Returns:
            dict with keys 'upper', 'middle', 'lower', 'pct_b'.
        """
        self._require_columns(df, ["close"])
        close = df["close"]
        middle = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()
        upper = middle + std_mult * std
        lower = middle - std_mult * std
        band_width = upper - lower
        pct_b = (close - lower) / band_width.replace(0, np.nan)
        return {
            "upper": upper.rename("bb_upper"),
            "middle": middle.rename("bb_middle"),
            "lower": lower.rename("bb_lower"),
            "pct_b": pct_b.rename("bb_pct_b"),
        }

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range using Wilder's EMA.

        Args:
            df: OHLCV DataFrame with 'high', 'low', 'close' columns.
            period: Lookback period for the EMA.

        Returns:
            pd.Series of ATR values, always >= 0.
        """
        self._require_columns(df, ["high", "low", "close"])
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        return atr.rename("atr")

    def compute_obv(self, df: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume.

        Args:
            df: OHLCV DataFrame with 'close' and 'volume' columns.

        Returns:
            pd.Series of cumulative OBV values.
        """
        self._require_columns(df, ["close", "volume"])
        close = df["close"]
        volume = df["volume"]
        direction = np.sign(close.diff()).fillna(0)
        signed_vol = direction * volume
        obv = signed_vol.cumsum()
        return obv.rename("obv")

    def compute_vwap(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute rolling 20-day VWAP.

        Args:
            df: OHLCV DataFrame with 'high', 'low', 'close', 'volume' columns.
            period: Rolling window in days.

        Returns:
            pd.Series of VWAP values.
        """
        self._require_columns(df, ["high", "low", "close", "volume"])
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"]
        rolling_tp_vol = (typical_price * vol).rolling(window=period, min_periods=period).sum()
        rolling_vol = vol.rolling(window=period, min_periods=period).sum()
        vwap = rolling_tp_vol / rolling_vol.replace(0, np.nan)
        return vwap.rename("vwap")

    def compute_momentum(
        self,
        df: pd.DataFrame,
        periods: list[int] | None = None,
    ) -> dict[str, pd.Series]:
        """Compute price momentum over multiple look-back windows.

        Args:
            df: OHLCV DataFrame with a 'close' column.
            periods: List of periods corresponding to [1m, 3m, 6m, 12m].
                     Defaults to [21, 63, 126, 252].

        Returns:
            dict with keys 'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m'.
        """
        self._require_columns(df, ["close"])
        if periods is None:
            periods = [21, 63, 126, 252]
        close = df["close"]
        keys = ["mom_1m", "mom_3m", "mom_6m", "mom_12m"]
        result: dict[str, pd.Series] = {}
        for key, p in zip(keys, periods):
            mom = close / close.shift(p) - 1.0
            result[key] = mom.rename(key)
        return result

    def compute_realized_vol(
        self,
        df: pd.DataFrame,
        short: int = 20,
        long: int = 60,
    ) -> dict[str, pd.Series]:
        """Compute short/long realized volatility and their ratio.

        Args:
            df: OHLCV DataFrame with a 'close' column.
            short: Short-window period (default 20).
            long: Long-window period (default 60).

        Returns:
            dict with keys 'vol_20d', 'vol_60d', 'vol_ratio'.
        """
        self._require_columns(df, ["close"])
        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol_short = log_ret.rolling(window=short, min_periods=short).std() * np.sqrt(252)
        vol_long = log_ret.rolling(window=long, min_periods=long).std() * np.sqrt(252)
        vol_ratio = vol_short / vol_long.replace(0, np.nan)
        return {
            "vol_20d": vol_short.rename("vol_20d"),
            "vol_60d": vol_long.rename("vol_60d"),
            "vol_ratio": vol_ratio.rename("vol_ratio"),
        }

    def compute_garman_klass(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Garman-Klass volatility estimator.

        Formula per bar: 0.5*(log(H/L))^2 - (2*log(2)-1)*(log(C/O))^2
        The rolling mean over `period` bars is returned, annualised.

        Args:
            df: OHLCV DataFrame with 'open', 'high', 'low', 'close' columns.
            period: Rolling window.

        Returns:
            pd.Series of GK volatility estimates (annualised), always > 0
            where data is sufficient.
        """
        self._require_columns(df, ["open", "high", "low", "close"])
        log_hl = np.log(df["high"] / df["low"])
        log_co = np.log(df["close"] / df["open"])
        gk_daily = 0.5 * log_hl**2 - (2.0 * np.log(2) - 1.0) * log_co**2
        gk_vol = (
            gk_daily.rolling(window=period, min_periods=period).mean() * 252
        ) ** 0.5
        return gk_vol.rename("garman_klass")

    def compute_parkinson(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Parkinson volatility estimator.

        Formula per bar: 1/(4*log(2)) * (log(H/L))^2
        Rolling mean over `period` bars is returned, annualised.

        Args:
            df: OHLCV DataFrame with 'high' and 'low' columns.
            period: Rolling window.

        Returns:
            pd.Series of Parkinson volatility estimates (annualised).
        """
        self._require_columns(df, ["high", "low"])
        log_hl = np.log(df["high"] / df["low"])
        park_daily = (1.0 / (4.0 * np.log(2))) * log_hl**2
        park_vol = (
            park_daily.rolling(window=period, min_periods=period).mean() * 252
        ) ** 0.5
        return park_vol.rename("parkinson")

    def compute_mean_reversion(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute mean-reversion signal as z-score of close vs rolling mean.

        Args:
            df: OHLCV DataFrame with a 'close' column.
            period: Rolling window for mean and std.

        Returns:
            pd.Series of z-scores; negative → price below mean (potential reversal up).
        """
        self._require_columns(df, ["close"])
        close = df["close"]
        rolling_mean = close.rolling(window=period, min_periods=period).mean()
        rolling_std = close.rolling(window=period, min_periods=period).std()
        z = (close - rolling_mean) / rolling_std.replace(0, np.nan)
        return z.rename("mean_reversion")

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_factors(
        self,
        df: pd.DataFrame,
        names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute a set of factors and return as a DataFrame.

        Args:
            df: OHLCV DataFrame (may contain multiple tickers via a 'ticker'
                column; factors are then computed per ticker).
            names: Subset of factor names to compute.  If None, all factors
                   are computed.  Valid names: ``_FACTOR_NAMES``.

        Returns:
            pd.DataFrame with one column per requested factor, same index as
            input.

        Raises:
            FactorError: If an unknown factor name is requested.
        """
        requested = set(names) if names else set(_FACTOR_NAMES)
        unknown = requested - set(_FACTOR_NAMES)
        if unknown:
            raise FactorError(f"Unknown factor names: {sorted(unknown)}")

        if "ticker" in df.columns:
            parts = []
            for ticker, grp in df.groupby("ticker", sort=False):
                part = self._compute_factors_single(grp, requested)
                parts.append(part)
            return pd.concat(parts).sort_index()

        return self._compute_factors_single(df, requested)

    def _compute_factors_single(
        self, df: pd.DataFrame, requested: set[str]
    ) -> pd.DataFrame:
        """Compute factors for a single-ticker DataFrame slice."""
        result: dict[str, pd.Series] = {}

        def _want(*names: str) -> bool:
            return not requested.isdisjoint(names)

        if _want("rsi"):
            result["rsi"] = self.compute_rsi(df)

        if _want("macd", "macd_signal", "macd_histogram"):
            macd_d = self.compute_macd(df)
            if "macd" in requested:
                result["macd"] = macd_d["macd"]
            if "macd_signal" in requested:
                result["macd_signal"] = macd_d["signal"]
            if "macd_histogram" in requested:
                result["macd_histogram"] = macd_d["histogram"]

        if _want("bb_upper", "bb_middle", "bb_lower", "bb_pct_b"):
            bb_d = self.compute_bollinger(df)
            for k, col in (
                ("bb_upper", "upper"),
                ("bb_middle", "middle"),
                ("bb_lower", "lower"),
                ("bb_pct_b", "pct_b"),
            ):
                if k in requested:
                    result[k] = bb_d[col]

        if _want("atr"):
            result["atr"] = self.compute_atr(df)

        if _want("obv"):
            result["obv"] = self.compute_obv(df)

        if _want("vwap"):
            result["vwap"] = self.compute_vwap(df)

        if _want("mom_1m", "mom_3m", "mom_6m", "mom_12m"):
            mom_d = self.compute_momentum(df)
            for k in ("mom_1m", "mom_3m", "mom_6m", "mom_12m"):
                if k in requested:
                    result[k] = mom_d[k]

        if _want("vol_20d", "vol_60d", "vol_ratio"):
            vol_d = self.compute_realized_vol(df)
            for k in ("vol_20d", "vol_60d", "vol_ratio"):
                if k in requested:
                    result[k] = vol_d[k]

        if _want("garman_klass"):
            result["garman_klass"] = self.compute_garman_klass(df)

        if _want("parkinson"):
            result["parkinson"] = self.compute_parkinson(df)

        if _want("mean_reversion"):
            result["mean_reversion"] = self.compute_mean_reversion(df)

        if not result:
            return pd.DataFrame(index=df.index)

        return pd.DataFrame(result, index=df.index)

    # ------------------------------------------------------------------
    # Factor evaluation
    # ------------------------------------------------------------------

    def compute_factor_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """Compute Information Coefficient (Spearman rank correlation).

        Args:
            factor: Factor values aligned with forward returns.
            returns: Forward return series.

        Returns:
            Spearman IC as a float.  Returns 0.0 if insufficient data.
        """
        combined = pd.concat([factor, returns], axis=1).dropna()
        if len(combined) < 5:
            return 0.0
        ic, _ = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
        return float(ic) if not np.isnan(ic) else 0.0

    def evaluate_factor(
        self,
        factor: pd.Series,
        forward_returns: pd.Series,
    ) -> dict[str, float]:
        """Evaluate a single factor's predictive power via IC analysis.

        Splits the common non-NaN observations into monthly (21-bar) windows
        and computes per-window Spearman ICs, then aggregates statistics.

        Args:
            factor: Factor time-series.
            forward_returns: Forward return series (same index as factor).

        Returns:
            dict with keys: ic_mean, ic_std, icir, t_stat, pct_positive,
            turnover.
        """
        combined = pd.concat([factor, forward_returns], axis=1).dropna()
        combined.columns = ["factor", "ret"]

        n = len(combined)
        if n < 10:
            return {
                "ic_mean": 0.0,
                "ic_std": 0.0,
                "icir": 0.0,
                "t_stat": 0.0,
                "pct_positive": 0.0,
                "turnover": 0.0,
            }

        # Compute rolling IC over 21-bar windows
        window = 21
        ic_list: list[float] = []
        for start in range(0, n - window + 1, window):
            chunk = combined.iloc[start : start + window]
            if len(chunk) < 5:
                continue
            ic, _ = stats.spearmanr(chunk["factor"], chunk["ret"])
            if not np.isnan(ic):
                ic_list.append(float(ic))

        if not ic_list:
            ic_val, _ = stats.spearmanr(combined["factor"], combined["ret"])
            ic_arr = np.array([float(ic_val) if not np.isnan(ic_val) else 0.0])
        else:
            ic_arr = np.array(ic_list)

        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0
        icir = ic_mean / ic_std if ic_std > 1e-9 else 0.0
        t_stat = (
            ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 1e-9 else 0.0
        )
        pct_positive = float(np.mean(ic_arr > 0))

        # Turnover: average absolute change in factor rank normalised by n_obs
        factor_rank = combined["factor"].rank(pct=True)
        turnover = float(factor_rank.diff().abs().mean()) if len(factor_rank) > 1 else 0.0

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "t_stat": t_stat,
            "pct_positive": pct_positive,
            "turnover": turnover,
        }

    def evaluate_factors(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> pd.DataFrame:
        """Evaluate multiple factors, returning a summary DataFrame.

        Args:
            factors: DataFrame where each column is a factor series.
            forward_returns: Forward return series (same index as factors).

        Returns:
            pd.DataFrame with one row per factor and columns:
            ic_mean, ic_std, icir, t_stat, pct_positive, turnover.
        """
        rows: dict[str, dict[str, float]] = {}
        for col in factors.columns:
            rows[col] = self.evaluate_factor(factors[col], forward_returns)
        return pd.DataFrame(rows).T

    def compute_quintile_returns(
        self,
        factor: pd.Series,
        returns: pd.Series,
        n: int = 5,
    ) -> pd.Series:
        """Compute mean return for each quantile bin.

        Args:
            factor: Factor values.
            returns: Corresponding return values.
            n: Number of bins (default 5 = quintiles).

        Returns:
            pd.Series indexed 1..n with mean return per bin.
        """
        combined = pd.concat([factor, returns], axis=1).dropna()
        combined.columns = ["factor", "ret"]
        combined["quintile"] = pd.qcut(combined["factor"], q=n, labels=False) + 1
        return combined.groupby("quintile")["ret"].mean()

    # ------------------------------------------------------------------
    # Factor library management
    # ------------------------------------------------------------------

    def save_factor(
        self,
        definition: FactorDefinition,
        path: str | None = None,
    ) -> None:
        """Save a FactorDefinition to the factor library JSON file.

        If a factor with the same name already exists, it is updated;
        otherwise it is appended.

        Args:
            definition: FactorDefinition to persist.
            path: Override path to JSON file.  Defaults to
                  ``outputs/factor_library.json``.
        """
        lib_path = Path(path) if path else self._output_dir / _FACTOR_LIBRARY_FILE
        lib_path.parent.mkdir(parents=True, exist_ok=True)

        existing = self.load_factor_library(str(lib_path))
        by_name = {f.name: f for f in existing}
        by_name[definition.name] = definition

        payload = [f.model_dump() for f in by_name.values()]
        lib_path.write_text(json.dumps(payload, indent=2))
        logger.debug("Saved factor '{}' to {}", definition.name, lib_path)

    def load_factor_library(self, path: str | None = None) -> list[FactorDefinition]:
        """Load factor definitions from the factor library JSON file.

        Args:
            path: Override path to JSON file.  Defaults to
                  ``outputs/factor_library.json``.

        Returns:
            List of FactorDefinition objects; empty list if file not found.
        """
        lib_path = Path(path) if path else self._output_dir / _FACTOR_LIBRARY_FILE
        try:
            data = json.loads(lib_path.read_text())
            return [FactorDefinition(**item) for item in data]
        except FileNotFoundError:
            return []
        except Exception as exc:
            logger.warning("Could not load factor library from {}: {}", lib_path, exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        """Raise FactorError if required columns are absent."""
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise FactorError(f"DataFrame missing required columns: {missing}")
