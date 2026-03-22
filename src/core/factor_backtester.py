"""FactorBacktester — walk-forward backtest gate for the factor discovery pipeline.

Validates IC-passing FactorDefinition objects against a full walk-forward backtest.
Only factors that also pass the backtest gate are admitted to the factor library.

Imports DataPipeline at module level so patch("src.core.factor_backtester.DataPipeline")
works in tests (same pattern as FactorEvaluator).
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

# Module-level imports for patchability in tests
from src.core.backtester import BacktestValidator, WalkForwardBacktester
from src.core.data_pipeline import DataPipeline
from src.core.model_zoo import ModelWrapper
from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.schemas import BacktestValidationResult, FactorDefinition, ValidationVerdict

# ---------------------------------------------------------------------------
# Constants — mirror FactorEvaluator Stage 2 constants
# If either file's lookback is tuned, the other must also be updated.
# ---------------------------------------------------------------------------
_BT_LOOKBACK_DAYS = 500        # same as _STAGE2_LOOKBACK_DAYS in factor_evaluator.py
_BT_TOPK = 10
_BT_WALK_FORWARD_MONTHS = 6    # same as _STAGE2_WALK_FORWARD_MONTHS
_BT_EMBARGO_DAYS = 5           # same as _STAGE2_EMBARGO_DAYS

_BT_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "MA", "UNH", "JNJ", "PG", "HD", "CVX", "LLY", "ABBV",
    "MRK", "PEP", "KO", "AVGO", "COST", "MCD", "ACN", "TMO", "DHR",
    "WMT", "BAC", "XOM",
]


class FactorBacktester:
    """Validate IC-passing factors via a full walk-forward backtest gate.

    Args:
        config: Full application config. Falls back to get_full_config() if None.
    """

    def __init__(self, config: FullAppConfig | None = None) -> None:
        self.config = config or get_full_config()
        self._pipeline = DataPipeline(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        factor: FactorDefinition,
        tickers: list[str] | None = None,
        df: pd.DataFrame | None = None,
    ) -> BacktestValidationResult:
        """Validate a single factor through the walk-forward backtest gate.

        Args:
            factor: IC-passing FactorDefinition to validate.
            tickers: Optional universe override. Defaults to _BT_UNIVERSE.
            df: Optional pre-fetched OHLCV DataFrame (avoids double-fetch in batch).

        Returns:
            BacktestValidationResult with passed, sharpe, checks, reason.
        """
        universe = tickers if tickers is not None else _BT_UNIVERSE

        # Step 1: Fetch OHLCV (or reuse pre-fetched df from validate_batch)
        if df is None:
            start = (datetime.today() - timedelta(days=_BT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            try:
                df = self._pipeline.yfinance_fallback(tickers=universe, start=start)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FactorBacktester: data fetch failed for '{}': {}", factor.name, exc)
                return BacktestValidationResult(factor_name=factor.name, passed=False, reason="no_data")

        if df is None or df.empty:
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="no_data")

        # Normalise: ensure date and ticker columns exist
        df = df.copy()
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "date" not in df.columns:
                df = df.rename(columns={df.columns[0]: "date"})
        if "ticker" not in df.columns:
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="no_data")

        # Step 2 + 3: Build factor values per ticker, then stack to (date, ticker) MultiIndex
        factor_series_list: list[pd.Series] = []
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
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fval = eval(factor.expression, {"__builtins__": {}}, ns)  # noqa: S307
                if not isinstance(fval, pd.Series):
                    fval = pd.Series(fval, index=grp.index)
                fval.name = ticker
                factor_series_list.append(fval)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FactorBacktester: eval error for '{}' on ticker '{}': {}",
                    factor.name, ticker, exc,
                )
                continue

        if not factor_series_list:
            logger.warning("FactorBacktester: all tickers failed eval for '{}' — treating as eval_error.", factor.name)
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="eval_error")

        factor_wide = pd.concat(factor_series_list, axis=1)  # dates × tickers
        features = factor_wide.stack().rename(factor.name).to_frame()

        # Step 4: Build labels — (date, ticker) MultiIndex Series of next-day returns
        returns_by_ticker: list[pd.Series] = []
        for ticker, grp in df.groupby("ticker"):
            grp = grp.sort_values("date").set_index("date")
            ret = grp["close"].pct_change().shift(-1)
            ret.name = ticker
            returns_by_ticker.append(ret)
        returns_wide = pd.concat(returns_by_ticker, axis=1)  # dates × tickers
        labels = returns_wide.stack()  # (date, ticker) MultiIndex Series, date at level 0

        # Step 5: Align and joint-drop NaN rows (joint to preserve index alignment)
        combined = features.join(labels.rename("_label"), how="inner").dropna()
        features = combined.drop(columns=["_label"])
        labels = combined["_label"]

        # Step 6: Minimum row check
        if len(features) < 50:
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="no_data")

        # Steps 7–9: Run walk-forward backtest
        try:
            model = ModelWrapper(model_name="Linear", config=self.config)  # type: ignore[arg-type]
            bt_result = WalkForwardBacktester(self.config).run(  # type: ignore[arg-type]
                features,
                labels,
                model,
                topk=_BT_TOPK,
                walk_forward_months=_BT_WALK_FORWARD_MONTHS,
                embargo_days=_BT_EMBARGO_DAYS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FactorBacktester: backtest error for '{}': {}", factor.name, exc)
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="backtest_error")

        try:
            val_result = BacktestValidator().validate(bt_result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FactorBacktester: validation error for '{}': {}", factor.name, exc)
            return BacktestValidationResult(factor_name=factor.name, passed=False, reason="backtest_error")

        # Step 10: Build checks dict from ValidationCheck list
        checks = {vc.name: vc.passed for vc in val_result.checks}

        # Step 11: Gate
        # ValidationVerdict has APPROVED / CAUTION / REJECTED.
        # CAUTION passes (Sharpe floor is the primary guard); only REJECTED is blocked.
        sharpe = bt_result.metrics.sharpe_ratio
        passed = (
            val_result.verdict != ValidationVerdict.REJECTED
            and sharpe >= self.config.rd_agent.min_backtest_sharpe
        )
        if passed:
            reason = "passed"
        elif val_result.verdict == ValidationVerdict.REJECTED:
            # Determine most specific reason from validator checks
            if not checks.get("overfitting_signals", True):
                reason = "overfitting"
            elif not checks.get("look_ahead_bias", True):
                reason = "look_ahead_bias"
            else:
                reason = "backtest_rejected"
        else:
            reason = "low_sharpe"

        return BacktestValidationResult(
            factor_name=factor.name,
            passed=passed,
            sharpe=sharpe,
            max_drawdown=bt_result.metrics.max_drawdown,
            checks=checks,
            reason=reason,
        )

    def validate_batch(
        self,
        factors: list[FactorDefinition],
        tickers: list[str] | None = None,
    ) -> list[BacktestValidationResult]:
        """Validate a batch of factors, fetching OHLCV only once.

        Args:
            factors: List of IC-passing FactorDefinition objects to validate.
            tickers: Optional universe override.

        Returns:
            List of BacktestValidationResult, one per input factor.
        """
        if not factors:
            return []

        universe = tickers if tickers is not None else _BT_UNIVERSE
        start = (datetime.today() - timedelta(days=_BT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        try:
            df = self._pipeline.yfinance_fallback(tickers=universe, start=start)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FactorBacktester.validate_batch: data fetch failed: {}", exc)
            df = None

        results: list[BacktestValidationResult] = []
        for factor in factors:
            result = self.validate(factor, tickers=tickers, df=df)
            results.append(result)
            logger.debug(
                "FactorBacktester: {} → passed={}, reason={}",
                factor.name, result.passed, result.reason,
            )
        return results
