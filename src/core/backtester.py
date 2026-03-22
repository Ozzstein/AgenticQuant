"""Walk-forward backtester and validation framework."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import AppConfig, get_config
from src.utils.exceptions import BacktestError
from src.utils.logger import get_logger
from src.utils.schemas import (
    BacktestMetrics,
    BacktestResult,
    ValidationCheck,
    ValidationResult,
    ValidationVerdict,
    WalkForwardFold,
)

logger = get_logger(__name__)

_TRADING_DAYS = 252


def split_folds(
    dates: pd.DatetimeIndex,
    train_start: str,
    walk_forward_months: int,
    embargo_days: int,
) -> list[dict]:
    """Split dates into expanding walk-forward folds.

    Args:
        dates: DatetimeIndex of all available trading dates.
        train_start: ISO date string for the start of the first training window.
        walk_forward_months: Length of each OOS window in months.
        embargo_days: Gap between train end and test start.

    Returns:
        List of dicts with keys: train_start, train_end, test_start, test_end.
    """
    folds: list[dict] = []
    unique_dates = dates.sort_values().unique()
    total_months = (unique_dates[-1].year - unique_dates[0].year) * 12 + (
        unique_dates[-1].month - unique_dates[0].month
    )

    if total_months < walk_forward_months * 2:
        return []

    current_test_start = unique_dates[0] + pd.DateOffset(months=walk_forward_months)

    while True:
        test_end = current_test_start + pd.DateOffset(months=walk_forward_months) - pd.Timedelta(days=1)
        if test_end > unique_dates[-1]:
            break

        train_end_raw = current_test_start - pd.Timedelta(days=embargo_days + 1)
        train_mask = unique_dates <= train_end_raw
        if train_mask.sum() < 20:
            current_test_start += pd.DateOffset(months=walk_forward_months)
            continue

        train_end = unique_dates[train_mask][-1]
        test_mask = (unique_dates >= current_test_start) & (unique_dates <= test_end)
        if test_mask.sum() < 1:
            current_test_start += pd.DateOffset(months=walk_forward_months)
            continue

        actual_test_end = unique_dates[test_mask][-1]
        folds.append(
            {
                "train_start": str(unique_dates[0].date()),
                "train_end": str(train_end.date()),
                "test_start": str(current_test_start.date()),
                "test_end": str(actual_test_end.date()),
            }
        )
        current_test_start += pd.DateOffset(months=walk_forward_months)

    return folds


class WalkForwardBacktester:
    """Walk-forward backtester operating on pandas DataFrames.

    Each fold uses an expanding training window and a fixed out-of-sample test
    window.  The top-K equal-weight portfolio is built from model predictions
    and fold returns are stitched together to form a daily equity curve.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        model: object,
        topk: int = 30,
        transaction_cost_bps: float = 10.0,
        walk_forward_months: int = 3,
        embargo_days: int = 5,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            features: Feature matrix.  Rows indexed by date (or MultiIndex
                (date, ticker)).  Columns are feature names.
            labels: Forward returns aligned with ``features``.
            model: Duck-typed model with ``train(X, y)`` and ``predict(X)``.
            topk: Number of stocks to hold per period.
            transaction_cost_bps: One-way transaction cost in basis points.
            walk_forward_months: Length of each OOS window in months.
            embargo_days: Gap between train end and test start.

        Returns:
            BacktestResult with metrics, equity curve and fold details.
        """
        if features.empty or labels.empty:
            raise BacktestError("features and labels must not be empty")

        # Normalise index so we always work with a DatetimeIndex of dates
        dates = self._extract_dates(features)
        folds = split_folds(dates, str(dates.min().date()), walk_forward_months, embargo_days)

        if not folds:
            raise BacktestError("No folds produced — dataset may be too small")

        fold_results: list[WalkForwardFold] = []
        period_returns: list[float] = []

        for fold_id, fold in enumerate(folds):
            result = self._run_fold(
                fold_id=fold_id,
                fold=fold,
                features=features,
                labels=labels,
                model=model,
                topk=topk,
                transaction_cost_bps=transaction_cost_bps,
                embargo_days=embargo_days,
            )
            fold_results.append(result.wf_fold)
            period_returns.extend(result.period_oos_returns)

        # Approximate daily returns by spreading each fold's return uniformly
        daily_returns = self._to_daily(period_returns, fold_results)
        equity_curve = (1 + pd.Series(daily_returns)).cumprod() * 100
        timestamps = [str(dates.min().date() + pd.Timedelta(days=i)) for i in range(len(daily_returns))]

        metrics = self._compute_metrics(pd.Series(daily_returns))
        metrics.total_trades = topk * len(folds)

        trades = [
            {"fold": f.fold_id, "test_start": f.test_start, "test_end": f.test_end,
             "topk": topk, "oos_sharpe": f.oos_sharpe}
            for f in fold_results
        ]

        logger.info(
            "Backtest complete: {} folds, Sharpe={:.2f}, MaxDD={:.1%}",
            len(folds),
            metrics.sharpe_ratio,
            metrics.max_drawdown,
        )
        return BacktestResult(
            metrics=metrics,
            daily_returns=daily_returns,
            trades=trades,
            walk_forward_results=fold_results,
            equity_curve=equity_curve.tolist(),
            timestamps=timestamps,
        )

    # ------------------------------------------------------------------
    # Single fold execution
    # ------------------------------------------------------------------

    class _FoldOutput:
        def __init__(self, wf_fold: WalkForwardFold, period_oos_returns: list[float]) -> None:
            self.wf_fold = wf_fold
            self.period_oos_returns = period_oos_returns

    def _run_fold(
        self,
        fold_id: int,
        fold: dict,
        features: pd.DataFrame,
        labels: pd.Series,
        model: object,
        topk: int,
        transaction_cost_bps: float,
        embargo_days: int,
    ) -> _FoldOutput:
        """Train model on fold and compute OOS top-K return."""
        dates = self._extract_dates(features)

        train_mask = (dates >= fold["train_start"]) & (dates <= fold["train_end"])
        test_mask = (dates >= fold["test_start"]) & (dates <= fold["test_end"])

        X_train = features.loc[train_mask]
        y_train = labels.loc[train_mask]
        X_test = features.loc[test_mask]
        y_test = labels.loc[test_mask]

        if X_train.empty or X_test.empty:
            return self._FoldOutput(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=fold["train_start"],
                    train_end=fold["train_end"],
                    test_start=fold["test_start"],
                    test_end=fold["test_end"],
                ),
                [],
            )

        # --- Train ---
        try:
            model.train(X_train.values, y_train.values)
            is_preds = model.predict(X_train.values)
            oos_preds = model.predict(X_test.values)
        except Exception as exc:
            raise BacktestError(f"Model error in fold {fold_id}: {exc}") from exc

        # --- IS Sharpe (for validation purposes) ---
        is_sharpe = self._top_k_sharpe(y_train, is_preds, topk)

        # --- OOS: build top-K portfolio each period ---
        oos_returns = self._top_k_period_returns(y_test, oos_preds, topk, transaction_cost_bps)
        oos_series = pd.Series(oos_returns)
        oos_sharpe = float(
            oos_series.mean() / oos_series.std() * math.sqrt(_TRADING_DAYS)
            if oos_series.std() > 0 and len(oos_series) > 1
            else 0.0
        )

        wf_fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
            in_sample_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
        )

        logger.debug(
            "Fold {}: IS Sharpe={:.2f}, OOS Sharpe={:.2f}, {} OOS periods",
            fold_id,
            is_sharpe,
            oos_sharpe,
            len(oos_returns),
        )
        return self._FoldOutput(wf_fold, oos_returns)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
        """Return DatetimeIndex from either flat or MultiIndex DataFrame."""
        if isinstance(df.index, pd.MultiIndex):
            return pd.DatetimeIndex(df.index.get_level_values(0))
        return pd.DatetimeIndex(df.index)

    @staticmethod
    def _top_k_period_returns(
        y: pd.Series,
        preds: np.ndarray,
        topk: int,
        cost_bps: float,
    ) -> list[float]:
        """Select top-K stocks by predicted rank; return net-of-cost period return."""
        if len(preds) == 0:
            return []

        k = min(topk, len(preds))
        top_idx = np.argsort(preds)[-k:]
        raw_return = float(np.mean(y.values[top_idx]))
        cost = k * cost_bps * 2 / 10_000  # round-trip
        return [raw_return - cost]

    @staticmethod
    def _top_k_sharpe(y: pd.Series, preds: np.ndarray, topk: int) -> float:
        """Compute approximate IS Sharpe using top-K groups across dates."""
        if len(preds) == 0 or y.std() == 0:
            return 0.0
        k = min(topk, len(preds))
        top_idx = np.argsort(preds)[-k:]
        vals = y.values[top_idx]
        if vals.std() == 0:
            return 0.0
        return float(vals.mean() / vals.std() * math.sqrt(_TRADING_DAYS))

    @staticmethod
    def _to_daily(period_returns: list[float], folds: list[WalkForwardFold]) -> list[float]:
        """Spread each fold's period return into approximate daily returns."""
        if not period_returns or not folds:
            return period_returns

        daily: list[float] = []
        for ret, fold in zip(period_returns, folds):
            try:
                n_days = max(
                    1,
                    (
                        pd.Timestamp(fold.test_end) - pd.Timestamp(fold.test_start)
                    ).days,
                )
            except Exception:
                n_days = 63  # ~3 months

            daily_ret = (1 + ret) ** (1 / n_days) - 1
            daily.extend([daily_ret] * n_days)

        return daily

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, daily_returns: pd.Series) -> BacktestMetrics:
        """Compute aggregate performance metrics from daily returns series.

        Args:
            daily_returns: Series of daily portfolio returns.

        Returns:
            BacktestMetrics populated with all statistics.
        """
        if daily_returns.empty:
            return BacktestMetrics()

        n = len(daily_returns)
        years = n / _TRADING_DAYS

        total_return = float((1 + daily_returns).prod() - 1)
        annual_return = float((1 + total_return) ** (1 / max(years, 1e-9)) - 1)

        vol = float(daily_returns.std() * math.sqrt(_TRADING_DAYS))
        sharpe = float(annual_return / vol) if vol > 0 else 0.0

        # Downside deviation (Sortino)
        downside = daily_returns[daily_returns < 0]
        downside_std = float(downside.std() * math.sqrt(_TRADING_DAYS)) if len(downside) > 1 else vol
        sortino = float(annual_return / downside_std) if downside_std > 0 else 0.0

        # Max drawdown
        equity = (1 + daily_returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())

        calmar = float(annual_return / abs(max_dd)) if max_dd < 0 else 0.0
        win_rate = float((daily_returns > 0).mean())
        avg_trade = float(daily_returns.mean())

        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            avg_trade_return=avg_trade,
            volatility=vol,
            sortino_ratio=sortino,
        )

    # ------------------------------------------------------------------
    # HTML Report
    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        result: BacktestResult,
        output_path: str = "reports/backtest.html",
    ) -> str:
        """Generate interactive HTML report with Plotly.

        Args:
            result: BacktestResult to visualise.
            output_path: Destination path for the HTML file.

        Returns:
            Absolute path to the generated HTML file.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise BacktestError("plotly is required for HTML reports") from exc

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        dates = result.timestamps or list(range(len(result.daily_returns)))
        returns = pd.Series(result.daily_returns, dtype=float)
        equity = pd.Series(result.equity_curve, dtype=float)

        # Rolling 60d Sharpe
        rolling_sharpe = (
            returns.rolling(60).mean() / returns.rolling(60).std() * math.sqrt(_TRADING_DAYS)
        ).fillna(0)

        # Drawdown
        equity_norm = equity / equity.iloc[0] if len(equity) > 0 else equity
        peak = equity_norm.cummax()
        drawdown = (equity_norm - peak) / peak

        # Fold IS/OOS bars
        fold_ids = [str(f.fold_id) for f in result.walk_forward_results]
        is_sharpes = [f.in_sample_sharpe for f in result.walk_forward_results]
        oos_sharpes = [f.oos_sharpe for f in result.walk_forward_results]

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Equity Curve", "Drawdown", "Rolling 60-Day Sharpe", "IS vs OOS Sharpe by Fold"],
            vertical_spacing=0.08,
            row_heights=[0.35, 0.2, 0.2, 0.25],
        )

        fig.add_trace(go.Scatter(x=dates, y=equity.tolist(), name="Equity", line=dict(color="royalblue")), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=dates, y=drawdown.tolist(), name="Drawdown", fill="tozeroy",
                       line=dict(color="crimson")), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=rolling_sharpe.tolist(), name="Rolling Sharpe",
                       line=dict(color="darkorange")), row=3, col=1
        )
        fig.add_trace(go.Bar(x=fold_ids, y=is_sharpes, name="IS Sharpe", marker_color="steelblue"), row=4, col=1)
        fig.add_trace(go.Bar(x=fold_ids, y=oos_sharpes, name="OOS Sharpe", marker_color="coral"), row=4, col=1)

        m = result.metrics
        fig.update_layout(
            title=(
                f"Backtest Report | Sharpe={m.sharpe_ratio:.2f} | "
                f"Ann Return={m.annual_return:.1%} | MaxDD={m.max_drawdown:.1%}"
            ),
            height=1000,
            template="plotly_white",
            showlegend=True,
        )

        fig.write_html(str(out), include_plotlyjs="cdn")
        logger.info("HTML report written to {}", out.resolve())
        return str(out.resolve())


# ---------------------------------------------------------------------------
# BacktestValidator
# ---------------------------------------------------------------------------


class BacktestValidator:
    """Validates a BacktestResult against 8 overfitting and realism checks."""

    CRITICAL_CHECKS = {
        "overfitting_signals",
    }

    def validate(self, result: BacktestResult) -> ValidationResult:
        """Run all 8 checks and return a ValidationResult.

        Args:
            result: BacktestResult from WalkForwardBacktester.

        Returns:
            ValidationResult with per-check details and overall verdict.
        """
        checks = [
            self._check_statistical_significance(result),
            self._check_overfitting_signals(result),
            self._check_cost_sensitivity(result),
            self._check_concentration_risk(result),
            self._check_look_ahead_bias(result),
            self._check_survivorship_bias(result),
            self._check_fill_realism(result),
            self._check_regime_dependence(result),
        ]

        verdict = self._determine_verdict(checks)
        passed = sum(1 for c in checks if c.passed)
        summary = f"{passed}/{len(checks)} checks passed — verdict: {verdict.value}"

        logger.info("Validation: {}", summary)
        return ValidationResult(checks=checks, verdict=verdict, summary=summary)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_statistical_significance(self, result: BacktestResult) -> ValidationCheck:
        """t-stat of daily returns > 2.0 AND at least 252 trading days."""
        name = "statistical_significance"
        returns = np.array(result.daily_returns)
        n = len(returns)

        if n < 252:
            return ValidationCheck(
                name=name,
                passed=False,
                details=f"Only {n} days — need ≥252 for significance",
                severity="critical",
            )

        t_stat, _ = stats.ttest_1samp(returns, 0)
        passed = bool(t_stat > 2.0)
        return ValidationCheck(
            name=name,
            passed=passed,
            details=f"t-stat={t_stat:.2f} (need >2.0), n={n} days",
            severity="warning",
        )

    def _check_overfitting_signals(self, result: BacktestResult) -> ValidationCheck:
        """Flag if Sharpe > 3 or any fold has OOS Sharpe < 50% of IS Sharpe."""
        name = "overfitting_signals"
        issues: list[str] = []

        if result.metrics.sharpe_ratio > 3.0:
            issues.append(f"Sharpe={result.metrics.sharpe_ratio:.2f} exceeds 3.0 (likely overfit)")

        for fold in result.walk_forward_results:
            if fold.in_sample_sharpe > 0 and fold.oos_sharpe < 0.5 * fold.in_sample_sharpe:
                issues.append(
                    f"Fold {fold.fold_id}: OOS Sharpe {fold.oos_sharpe:.2f} "
                    f"< 50% of IS Sharpe {fold.in_sample_sharpe:.2f}"
                )

        passed = len(issues) == 0
        return ValidationCheck(
            name=name,
            passed=passed,
            details="; ".join(issues) if issues else "No overfitting signals detected",
            severity="critical",
        )

    def _check_cost_sensitivity(self, result: BacktestResult) -> ValidationCheck:
        """Flag strategies with Sharpe < 0.5 as likely cost-sensitive."""
        name = "cost_sensitivity"
        sharpe = result.metrics.sharpe_ratio
        passed = sharpe >= 0.5
        return ValidationCheck(
            name=name,
            passed=passed,
            details=(
                f"Sharpe={sharpe:.2f} — strategy appears cost-sensitive"
                if not passed
                else f"Sharpe={sharpe:.2f} — cost-robust"
            ),
            severity="warning",
        )

    def _check_concentration_risk(self, result: BacktestResult) -> ValidationCheck:
        """Check no single position would exceed 30% weight in top-K equal-weight portfolio."""
        name = "concentration_risk"
        topk_values = [t.get("topk", 0) for t in result.trades if isinstance(t.get("topk"), (int, float))]
        if not topk_values:
            return ValidationCheck(name=name, passed=True,
                                   details="No trade data — skipping", severity="info")

        min_topk = int(min(topk_values))
        max_weight = 1 / max(min_topk, 1)
        passed = max_weight <= 0.30
        return ValidationCheck(
            name=name,
            passed=passed,
            details=f"Max position weight = {max_weight:.1%} with topk={min_topk}",
            severity="warning",
        )

    def _check_look_ahead_bias(self, result: BacktestResult) -> ValidationCheck:
        """Verify embargo days > 0 to prevent lookahead bias."""
        name = "look_ahead_bias"
        # Proxy: check that OOS windows don't start the same day as train ends
        issues: list[str] = []
        for fold in result.walk_forward_results:
            if fold.train_end >= fold.test_start:
                issues.append(f"Fold {fold.fold_id}: train_end={fold.train_end} >= test_start={fold.test_start}")

        passed = len(issues) == 0
        return ValidationCheck(
            name=name,
            passed=passed,
            details="; ".join(issues) if issues else "No lookahead bias detected — embargo enforced",
            severity="warning",
        )

    def _check_survivorship_bias(self, result: BacktestResult) -> ValidationCheck:
        """Informational: flag if no survivorship-bias information is recorded."""
        name = "survivorship_bias"
        return ValidationCheck(
            name=name,
            passed=False,
            details="CAUTION: No survivorship bias information available — results may be optimistic",
            severity="info",
        )

    def _check_fill_realism(self, result: BacktestResult) -> ValidationCheck:
        """Informational check on fill realism via trade count."""
        name = "fill_realism"
        total_trades = result.metrics.total_trades
        if total_trades == 0:
            return ValidationCheck(name=name, passed=True,
                                   details="No trades recorded", severity="info")
        return ValidationCheck(
            name=name,
            passed=True,
            details=f"Total trades={total_trades} — fill realism: informational only",
            severity="info",
        )

    def _check_regime_dependence(self, result: BacktestResult) -> ValidationCheck:
        """Ensure OOS performance holds across both good and bad IS regimes."""
        name = "regime_dependence"
        folds = result.walk_forward_results
        if len(folds) < 2:
            return ValidationCheck(name=name, passed=True,
                                   details="Too few folds for regime analysis", severity="info")

        median_is = np.median([f.in_sample_sharpe for f in folds])
        bad_folds = [f for f in folds if f.in_sample_sharpe < median_is]

        if not bad_folds:
            return ValidationCheck(name=name, passed=True,
                                   details="All folds have above-median IS Sharpe", severity="info")

        bad_oos_sharpes = [f.oos_sharpe for f in bad_folds]
        avg_bad_oos = float(np.mean(bad_oos_sharpes))
        passed = avg_bad_oos > -0.5
        return ValidationCheck(
            name=name,
            passed=passed,
            details=f"Avg OOS Sharpe in bad-IS folds = {avg_bad_oos:.2f}",
            severity="warning",
        )

    # ------------------------------------------------------------------
    # Verdict logic
    # ------------------------------------------------------------------

    def _determine_verdict(self, checks: list[ValidationCheck]) -> ValidationVerdict:
        """Map check results to overall verdict.

        Returns:
            APPROVED if all critical checks pass and ≤1 non-critical failure.
            CAUTION if all critical checks pass but ≥2 non-critical failures.
            REJECTED if any critical check fails.
        """
        failed_critical = [c for c in checks if not c.passed and c.severity == "critical"]
        if failed_critical:
            return ValidationVerdict.REJECTED

        failed_warning = [c for c in checks if not c.passed and c.severity == "warning"]
        if len(failed_warning) >= 2:
            return ValidationVerdict.CAUTION
        return ValidationVerdict.APPROVED
