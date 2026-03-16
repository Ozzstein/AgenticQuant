"""Enhanced walk-forward runner with feature importance drift and rolling IC."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.core.backtester import WalkForwardBacktester
from src.core.model_zoo import ModelWrapper
from src.utils.config import AppConfig, get_config
from src.utils.exceptions import BacktestError
from src.utils.logger import get_logger
from src.utils.schemas import (
    WalkForwardFold,
    WalkForwardResult,
)

logger = get_logger(__name__)

_TRADING_DAYS = 252


class WalkForwardRunner:
    """Run enhanced walk-forward backtests with feature drift and rolling IC tracking.

    Builds on WalkForwardBacktester but adds:
    - Per-fold feature importance tracking
    - Spearman rank correlation drift between consecutive folds
    - Per-fold Information Coefficient (IC) between predictions and realized returns
    - Enhanced 5-panel HTML report
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialise WalkForwardRunner.

        Args:
            config: Application config; falls back to ``get_config()`` singleton.
        """
        self.config = config or get_config()
        self._backtester = WalkForwardBacktester(self.config)

    def run(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        model_name: str = "LightGBM",
        topk: int = 30,
        cost_bps: float = 10.0,
        walk_months: int = 3,
        embargo_days: int = 5,
        mode: str = "expanding",
    ) -> WalkForwardResult:
        """Run enhanced walk-forward backtest with feature importance drift and rolling IC.

        Args:
            features: Feature matrix indexed by date (or MultiIndex(date, ticker)).
            labels: Forward returns aligned with ``features``.
            model_name: One of the ModelWrapper registry keys (e.g. ``LightGBM``).
            topk: Number of stocks to hold per OOS period.
            cost_bps: One-way transaction cost in basis points.
            walk_months: Length of each OOS window in months.
            embargo_days: Gap between train end and test start.
            mode: Walk-forward mode (currently only ``expanding`` is supported).

        Returns:
            WalkForwardResult with stitched returns, per-fold metrics, feature drift
            and rolling IC.

        Raises:
            BacktestError: If no folds are produced from the dataset.
        """
        if features.empty or labels.empty:
            raise BacktestError("features and labels must not be empty")

        dates = WalkForwardBacktester._extract_dates(features)
        folds = self._backtester._split_folds(
            dates, str(dates.min().date()), walk_months, embargo_days
        )

        if not folds:
            raise BacktestError("No folds produced — dataset may be too small")

        fold_results: list[WalkForwardFold] = []
        rolling_ic: list[float] = []
        feature_importances: list[pd.Series] = []
        # Each entry: (test_start_str, list[float]) for stitching
        fold_period_returns: list[tuple[str, list[float]]] = []

        for i, fold in enumerate(folds):
            train_mask = (dates >= fold["train_start"]) & (dates <= fold["train_end"])
            test_mask = (dates >= fold["test_start"]) & (dates <= fold["test_end"])

            X_train = features.loc[train_mask]
            y_train = labels.loc[train_mask]
            X_test = features.loc[test_mask]
            y_test = labels.loc[test_mask]

            # Skip folds with insufficient data
            if X_train.empty or X_test.empty:
                fold_results.append(
                    WalkForwardFold(
                        fold_id=i,
                        train_start=fold["train_start"],
                        train_end=fold["train_end"],
                        test_start=fold["test_start"],
                        test_end=fold["test_end"],
                    )
                )
                rolling_ic.append(0.0)
                feature_importances.append(pd.Series(dtype=float))
                fold_period_returns.append((fold["test_start"], []))
                logger.debug("Fold {}: skipped — empty train or test split", i)
                continue

            # Train model
            model = ModelWrapper(model_name, self.config)
            model.train(X_train.values, y_train.values)
            is_preds = model.predict(X_train.values)
            oos_preds = model.predict(X_test.values)

            # Feature importance
            feature_imp = model.get_feature_importance()
            feature_importances.append(feature_imp)

            # IS Sharpe
            is_sharpe = WalkForwardBacktester._top_k_sharpe(y_train, is_preds, topk)

            # OOS period returns
            oos_returns = WalkForwardBacktester._top_k_period_returns(
                y_test, oos_preds, topk, cost_bps
            )

            # OOS Sharpe — cross-sectional (same pattern as IS Sharpe)
            oos_sharpe = WalkForwardBacktester._top_k_sharpe(y_test, oos_preds, topk)

            # Per-fold IC
            try:
                if len(oos_preds) >= 2 and len(y_test.values) >= 2:
                    ic_val = float(spearmanr(oos_preds, y_test.values).statistic)
                    if not math.isfinite(ic_val):
                        ic_val = 0.0
                else:
                    ic_val = 0.0
            except Exception:
                ic_val = 0.0
            rolling_ic.append(ic_val)

            # Store top-10 feature importances in model_params
            if not feature_imp.empty:
                top_imp = feature_imp.nlargest(10)
                model_params: dict[str, float | int | str | bool] = {
                    f"fi_{name}": float(val) for name, val in top_imp.items()
                }
            else:
                model_params = {}

            wf_fold = WalkForwardFold(
                fold_id=i,
                train_start=fold["train_start"],
                train_end=fold["train_end"],
                test_start=fold["test_start"],
                test_end=fold["test_end"],
                in_sample_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                model_params=model_params,
            )
            fold_results.append(wf_fold)
            fold_period_returns.append((fold["test_start"], oos_returns))

            logger.debug(
                "Fold {}: IS Sharpe={:.2f}, OOS Sharpe={:.2f}, IC={:.3f}",
                i,
                is_sharpe,
                oos_sharpe,
                ic_val,
            )

        # Compute feature importance drift (Spearman rank corr of consecutive folds)
        # Always produces exactly n_folds - 1 entries; pairs with empty importances yield 0.0.
        n_folds = len(folds)
        feature_importance_drift: list[float] = []
        for j in range(len(feature_importances) - 1):
            imp_a = feature_importances[j]
            imp_b = feature_importances[j + 1]
            if imp_a.empty or imp_b.empty:
                feature_importance_drift.append(0.0)
                continue
            # Align on common index
            common = imp_a.index.intersection(imp_b.index)
            if len(common) < 2:
                feature_importance_drift.append(0.0)
                continue
            a_vals = imp_a[common].values
            b_vals = imp_b[common].values
            if a_vals.std() == 0 or b_vals.std() == 0:
                feature_importance_drift.append(0.0)
                continue
            try:
                corr = float(spearmanr(a_vals, b_vals).statistic)
                feature_importance_drift.append(corr if math.isfinite(corr) else 0.0)
            except Exception:
                feature_importance_drift.append(0.0)

        # Stitch returns and timestamps
        # Sort by test_start then flatten
        fold_period_returns_sorted = sorted(fold_period_returns, key=lambda x: x[0])
        stitched_returns: list[float] = []
        stitched_timestamps: list[str] = []
        for test_start_str, ret_list in fold_period_returns_sorted:
            for r in ret_list:
                stitched_returns.append(r)
                stitched_timestamps.append(test_start_str)

        # Aggregate metrics
        aggregate_metrics = self._backtester._compute_metrics(pd.Series(stitched_returns))
        aggregate_metrics.total_trades = topk * len(folds)

        logger.info(
            "WalkForwardRunner complete: {} folds, Sharpe={:.2f}, IC_mean={:.3f}",
            n_folds,
            aggregate_metrics.sharpe_ratio,
            float(np.mean(rolling_ic)) if rolling_ic else 0.0,
        )

        return WalkForwardResult(
            folds=fold_results,
            stitched_returns=stitched_returns,
            stitched_timestamps=stitched_timestamps,
            aggregate_metrics=aggregate_metrics,
            feature_importance_drift=feature_importance_drift,
            rolling_ic=rolling_ic,
            model_name=model_name,
        )

    def generate_html_report(self, result: WalkForwardResult, output_path: str) -> str:
        """Generate 5-panel Plotly HTML report.

        Panels:
        1. Cumulative returns with vertical retrain lines at each fold train_end
        2. Per-window Sharpe bars (OOS Sharpe per fold)
        3. Feature importance heatmap (fold × feature top-10)
        4. Drawdown
        5. Rolling IC per fold

        Args:
            result: WalkForwardResult produced by :meth:`run`.
            output_path: Destination file path for the HTML report.

        Returns:
            Absolute path to the written HTML file.

        Raises:
            BacktestError: If plotly is not installed.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise BacktestError("plotly is required for HTML reports") from exc

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # --- Build cumulative returns series ---
        returns_series = pd.Series(result.stitched_returns, dtype=float)
        timestamps = result.stitched_timestamps or list(range(len(result.stitched_returns)))

        cum_returns = (1 + returns_series).cumprod().tolist() if len(returns_series) > 0 else []

        # Drawdown
        if cum_returns:
            cum_s = pd.Series(cum_returns)
            peak = cum_s.cummax()
            drawdown = ((cum_s - peak) / peak).tolist()
        else:
            drawdown = []

        # Fold train_end dates for vertical retrain lines
        retrain_dates = [f.train_end for f in result.folds]
        fold_ids = [str(f.fold_id) for f in result.folds]
        oos_sharpes = [f.oos_sharpe for f in result.folds]

        # Feature heatmap data
        fi_data: list[dict[str, float]] = []
        for fold in result.folds:
            fi_row = {
                k.replace("fi_", ""): v
                for k, v in fold.model_params.items()
                if k.startswith("fi_") and isinstance(v, float)
            }
            fi_data.append(fi_row)

        all_feat_names = sorted(
            {feat for row in fi_data for feat in row}
        )

        has_feature_data = bool(all_feat_names)

        # Always produce exactly 5 panels
        subplot_titles_list = [
            "Cumulative Returns",
            "OOS Sharpe per Fold",
            "Feature Importance Heatmap",
            "Drawdown",
            "Rolling IC per Fold",
        ]

        fig = make_subplots(
            rows=5,
            cols=1,
            subplot_titles=subplot_titles_list,
            vertical_spacing=0.07,
        )

        # Panel 1: cumulative returns
        fig.add_trace(
            go.Scatter(
                x=list(timestamps),
                y=cum_returns,
                name="Cumulative Return",
                line={"color": "royalblue"},
            ),
            row=1,
            col=1,
        )

        # Vertical retrain lines as shapes
        shapes = []
        for rd in retrain_dates:
            shapes.append(
                {
                    "type": "line",
                    "x0": rd,
                    "x1": rd,
                    "y0": 0,
                    "y1": 1,
                    "xref": "x1",
                    "yref": "paper",
                    "line": {"color": "grey", "dash": "dot", "width": 1},
                }
            )

        # Panel 2: OOS Sharpe bars
        fig.add_trace(
            go.Bar(x=fold_ids, y=oos_sharpes, name="OOS Sharpe", marker_color="coral"),
            row=2,
            col=1,
        )

        # Panel 3: feature importance heatmap (or empty placeholder)
        if has_feature_data:
            z_matrix = [
                [row.get(feat, 0.0) for feat in all_feat_names] for row in fi_data
            ]
            fig.add_trace(
                go.Heatmap(
                    z=z_matrix,
                    x=all_feat_names,
                    y=fold_ids,
                    colorscale="Blues",
                    name="Feature Importance",
                ),
                row=3,
                col=1,
            )
        else:
            fig.add_trace(
                go.Heatmap(z=[], x=[], y=[], name="Feature Importance"),
                row=3,
                col=1,
            )

        # Panel 4: drawdown
        fig.add_trace(
            go.Scatter(
                x=list(timestamps),
                y=drawdown,
                name="Drawdown",
                fill="tozeroy",
                line={"color": "crimson"},
            ),
            row=4,
            col=1,
        )

        # Panel 5: rolling IC
        fig.add_trace(
            go.Bar(
                x=fold_ids,
                y=result.rolling_ic,
                name="Rolling IC",
                marker_color="steelblue",
            ),
            row=5,
            col=1,
        )

        m = result.aggregate_metrics
        fig.update_layout(
            title=(
                f"Walk-Forward Report | Model={result.model_name} | "
                f"Sharpe={m.sharpe_ratio:.2f} | "
                f"Ann Return={m.annual_return:.1%} | MaxDD={m.max_drawdown:.1%}"
            ),
            height=1200,
            template="plotly_white",
            showlegend=True,
            shapes=shapes,
        )

        fig.write_html(str(out), include_plotlyjs="cdn")
        logger.info("WalkForward HTML report written to {}", out.resolve())
        return str(out.resolve())
