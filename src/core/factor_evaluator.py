"""FactorEvaluator — two-stage IC gate for factor validation.

Stage 1: fast 90-day IC check (rejects ~70% cheaply, < 10 seconds per factor).
Stage 2: 500-day walk-forward IC/ICIR computation on OOS windows.

The factor expression is evaluated using Python eval() in a restricted namespace
{close, high, low, open, volume} — full pandas method syntax is available.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.core.backtester import split_folds
from src.core.data_pipeline import DataPipeline
from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.schemas import EvalResult, FactorDefinition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EVAL_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "MA", "UNH", "JNJ", "PG", "HD", "CVX", "LLY", "ABBV",
    "MRK", "PEP", "KO", "AVGO", "COST", "MCD", "ACN", "TMO", "DHR",
    "WMT", "BAC", "XOM",
]

_STAGE1_LOOKBACK_DAYS = 90
_STAGE2_LOOKBACK_DAYS = 500
_STAGE2_WALK_FORWARD_MONTHS = 6
_STAGE2_EMBARGO_DAYS = 5


class FactorEvaluator:
    """Validate factor candidates with real market data via two-stage IC gate.

    Args:
        config: Full application config. Falls back to get_full_config() if None.
    """

    def __init__(self, config: FullAppConfig | None = None) -> None:
        self.config = config or get_full_config()
        self._pipeline = DataPipeline(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_factor(
        self,
        factor: FactorDefinition,
        tickers: list[str] | None = None,
    ) -> EvalResult:
        """Evaluate a single factor through the two-stage IC gate.

        Args:
            factor: FactorDefinition with name and expression to evaluate.
            tickers: Optional override for the evaluation universe.
                     Defaults to _EVAL_UNIVERSE (30 large-cap US equities).

        Returns:
            EvalResult with stage1_ic, stage1_passed, and optionally stage2_ic/icir.
        """
        universe = tickers if tickers is not None else _EVAL_UNIVERSE
        min_ic = self.config.rd_agent.min_ic
        min_icir = self.config.rd_agent.min_icir

        # Stage 1: fast 90-day IC check
        stage1_ic, stage1_error = self._compute_stage1_ic(factor, universe)

        if stage1_error:
            return EvalResult(
                factor_name=factor.name,
                stage1_ic=0.0,
                stage1_passed=False,
                passed=False,
                reason=stage1_error,
            )

        stage1_passed = abs(stage1_ic) >= min_ic

        if not stage1_passed:
            return EvalResult(
                factor_name=factor.name,
                stage1_ic=round(stage1_ic, 6),
                stage1_passed=False,
                passed=False,
                reason="low_ic",
            )

        # Stage 2: walk-forward IC/ICIR
        stage2_ic, stage2_icir, stage2_error = self._compute_stage2_ic(factor, universe)

        if stage2_error:
            return EvalResult(
                factor_name=factor.name,
                stage1_ic=round(stage1_ic, 6),
                stage1_passed=True,
                passed=False,
                reason=stage2_error,
            )

        passed = (
            (stage2_ic is not None)
            and (stage2_ic >= min_ic)
            and (stage2_icir is not None)
            and (stage2_icir >= min_icir)
        )

        return EvalResult(
            factor_name=factor.name,
            stage1_ic=round(stage1_ic, 6),
            stage1_passed=True,
            stage2_ic=round(stage2_ic, 6) if stage2_ic is not None else None,
            stage2_icir=round(stage2_icir, 6) if stage2_icir is not None else None,
            passed=passed,
            reason="passed" if passed else "low_ic",
        )

    def evaluate_factors_batch(
        self,
        factors: list[FactorDefinition],
    ) -> list[EvalResult]:
        """Evaluate a batch of factors sequentially.

        Args:
            factors: List of FactorDefinition objects to evaluate.

        Returns:
            List of EvalResult objects, one per input factor.
        """
        results: list[EvalResult] = []
        for factor in factors:
            result = self.evaluate_factor(factor)
            results.append(result)
            logger.debug(
                "FactorEvaluator: {} → passed={}, stage1_ic={:.4f}",
                factor.name,
                result.passed,
                result.stage1_ic,
            )
        return results

    # ------------------------------------------------------------------
    # Stage 1 — fast 90-day IC
    # ------------------------------------------------------------------

    def _compute_stage1_ic(
        self,
        factor: FactorDefinition,
        universe: list[str],
    ) -> tuple[float, str | None]:
        """Fetch 90-day data and compute cross-sectional Spearman IC.

        Returns:
            (ic_value, error_reason_or_None)
        """
        stage1_start = (datetime.today() - timedelta(days=_STAGE1_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        try:
            df = self._pipeline.yfinance_fallback(
                tickers=universe,
                start=stage1_start,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 1 data fetch failed: {}", exc)
            return 0.0, "no_data"

        if df is None or df.empty:
            return 0.0, "no_data"

        try:
            ic = self._cross_sectional_ic(df, factor.expression)
            return ic, None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 1 eval error for '{}': {}", factor.name, exc)
            return 0.0, "eval_error"

    # ------------------------------------------------------------------
    # Stage 2 — walk-forward IC/ICIR
    # ------------------------------------------------------------------

    def _compute_stage2_ic(
        self,
        factor: FactorDefinition,
        universe: list[str],
    ) -> tuple[float | None, float | None, str | None]:
        """Fetch 500-day data, split into OOS folds, compute IC per fold.

        Returns:
            (ic_mean, icir, error_reason_or_None)
        """
        stage2_start = (datetime.today() - timedelta(days=_STAGE2_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        try:
            df = self._pipeline.yfinance_fallback(
                tickers=universe,
                start=stage2_start,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 2 data fetch failed: {}", exc)
            return None, None, "no_data"

        if df is None or df.empty:
            return None, None, "no_data"

        # Build date index for fold splitting
        if "date" in df.columns:
            dates = pd.DatetimeIndex(df["date"].unique())
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.unique()
        else:
            try:
                dates = pd.DatetimeIndex(df.index.unique())
            except Exception:  # noqa: BLE001
                return None, None, "eval_error"

        dates = dates.sort_values()
        if len(dates) < 50:
            return None, None, "no_data"

        try:
            folds = split_folds(
                dates,
                str(dates[0].date()),
                walk_forward_months=_STAGE2_WALK_FORWARD_MONTHS,
                embargo_days=_STAGE2_EMBARGO_DAYS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 2 fold split failed: {}", exc)
            return None, None, "eval_error"

        if not folds:
            return None, None, "no_data"

        fold_ics: list[float] = []
        for fold in folds:
            try:
                if "date" in df.columns:
                    mask = (pd.to_datetime(df["date"]) >= fold["test_start"]) & (
                        pd.to_datetime(df["date"]) <= fold["test_end"]
                    )
                    fold_df = df[mask]
                else:
                    mask = (df.index >= fold["test_start"]) & (df.index <= fold["test_end"])
                    fold_df = df[mask]

                if fold_df.empty:
                    continue
                ic = self._cross_sectional_ic(fold_df, factor.expression)
                fold_ics.append(ic)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Stage 2 fold IC failed: {}", exc)
                continue

        if not fold_ics:
            return None, None, "eval_error"

        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics)) if len(fold_ics) > 1 else 1.0
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        return ic_mean, icir, None

    # ------------------------------------------------------------------
    # IC computation helper
    # ------------------------------------------------------------------

    def _cross_sectional_ic(self, df: pd.DataFrame, expression: str) -> float:
        """Compute cross-sectional Spearman IC of factor vs. next-day return.

        The expression is evaluated using Python eval() in a restricted namespace
        {close, high, low, open, volume} per ticker. IC is computed as the
        Spearman correlation between factor values and next-day returns,
        averaged across all dates.

        Args:
            df: Multi-ticker OHLCV DataFrame (index = date, or with 'date' + 'ticker' cols).
            expression: Pandas-compatible expression string.

        Returns:
            Mean cross-sectional Spearman IC (float).

        Raises:
            ValueError: If expression evaluation fails or data is insufficient.
        """
        # Normalise: need date and ticker columns
        df = df.copy()
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "date" not in df.columns:
                # Index may have any name (e.g. "Date", None→"index") — rename first col
                df = df.rename(columns={df.columns[0]: "date"})
        if "ticker" not in df.columns:
            # Single-ticker df — cannot compute cross-sectional IC
            raise ValueError("DataFrame must have a 'ticker' column for cross-sectional IC")

        # Ensure required columns exist
        required = {"close", "open", "high", "low", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Compute factor values per ticker
        factor_values: list[pd.Series] = []
        for ticker, grp in df.groupby("ticker"):
            grp = grp.sort_values("date").set_index("date")
            ns = {
                "close": grp["close"],
                "open": grp["open"],
                "high": grp["high"],
                "low": grp["low"],
                "volume": grp["volume"].astype(float),
                "pd": pd,
                "np": np,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fval = eval(expression, {"__builtins__": {}}, ns)  # noqa: S307
            if not isinstance(fval, pd.Series):
                fval = pd.Series(fval, index=grp.index)
            fval.name = ticker
            factor_values.append(fval)

        factor_df = pd.concat(factor_values, axis=1)  # dates × tickers

        # Next-day returns per ticker
        ret_values: list[pd.Series] = []
        for ticker, grp in df.groupby("ticker"):
            grp = grp.sort_values("date").set_index("date")
            ret = grp["close"].pct_change().shift(-1)
            ret.name = ticker
            ret_values.append(ret)
        ret_df = pd.concat(ret_values, axis=1)

        # Compute IC per date (cross-sectional Spearman)
        ic_list: list[float] = []
        for date_idx in factor_df.index:
            f_row = factor_df.loc[date_idx].dropna()
            r_row = ret_df.loc[date_idx].dropna() if date_idx in ret_df.index else pd.Series(dtype=float)
            common = f_row.index.intersection(r_row.index)
            if len(common) < 5:
                continue
            corr, _ = stats.spearmanr(f_row[common].values, r_row[common].values)
            if not np.isnan(corr):
                ic_list.append(float(corr))

        if not ic_list:
            raise ValueError("No valid IC observations computed")

        return float(np.mean(ic_list))
