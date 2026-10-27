"""Point-in-time factor IC diagnostics using the same interpreter as inference.

Diagnostics select candidates for a registered experiment. They do not approve
activation or substitute for chronological portfolio evaluation.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.core.walk_forward import cross_sectional_ic
from src.data.features.expressions import evaluate_expression
from src.domain.common import _utc
from src.domain.market import FeatureSpec, MarketSnapshot
from src.domain.portfolio import UniverseEligibility
from src.research.datasets import eligible_features
from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.schemas import EvalResult, FactorDefinition


class FactorEvaluator:
    def __init__(self, config: FullAppConfig | None = None):
        self.config = config or get_full_config()

    def evaluate_factor(self, factor: FactorDefinition, *, snapshot: MarketSnapshot,
                        universes: tuple[UniverseEligibility, ...], universe_version: str,
                        stage1: tuple[datetime, datetime], stage2: tuple[datetime, datetime],
                        min_dates: int = 20) -> EvalResult:
        windows = [tuple(pd.Timestamp(_utc(t)) for t in bounds) for bounds in (stage1, stage2)]
        if not windows[0][0] < windows[0][1] <= windows[1][0] < windows[1][1] <= snapshot.context.as_of:
            raise ValueError("IC screening windows must be disjoint, ordered and inside snapshot cutoff")
        if min_dates < 2:
            raise ValueError("at least two date-local IC observations required")
        data = eligible_features(snapshot, (FeatureSpec(name=factor.name, expression=factor.expression),),
                                 universes, universe_version)
        dates = data.features.index.get_level_values(0)
        means, ratios, counts = [], [], []
        for start, end in windows:
            mask = (dates >= start) & (dates < end) & (data.label_available_at < end)
            series = cross_sectional_ic(data.features.loc[mask, factor.name], data.labels.loc[mask])
            values = [v for v in series.values() if v is not None]
            count = len(values)
            mean = float(np.mean(values)) if count >= min_dates else None
            std = float(np.std(values, ddof=1)) if count >= min_dates else None
            means.append(mean)
            ratios.append(mean / std if std and std > 0 else None)
            counts.append(count)
        stage1_ok = means[0] is not None and means[0] > self.config.rd_agent.min_ic
        passed = (stage1_ok and means[1] is not None and means[1] > self.config.rd_agent.min_ic
                  and ratios[1] is not None and ratios[1] > self.config.rd_agent.min_icir)
        return EvalResult(factor_name=factor.name, stage1_ic=means[0], stage1_passed=stage1_ok,
                          stage2_ic=means[1], stage2_icir=ratios[1], passed=passed,
                          reason="passed" if passed else "insufficient_observations" if min(counts) < min_dates
                          else "undefined_icir" if ratios[1] is None else "low_ic")

    def evaluate_factors_batch(self, factors: list[FactorDefinition], **explicit_inputs) -> list[EvalResult]:
        return [self.evaluate_factor(factor, **explicit_inputs) for factor in factors]

    def _cross_sectional_ic(self, frame: pd.DataFrame, expression: str) -> float:
        """Legacy arithmetic diagnostic only; no data fetching or approval authority."""
        data = frame.reset_index() if "date" not in frame.columns else frame.copy()
        if "date" not in data or "ticker" not in data:
            raise ValueError("date and ticker columns required")
        data = data.set_index(["date", "ticker"]).sort_index()
        values = evaluate_expression(expression, data)
        labels = data.close.groupby(level="ticker").shift(-1) / data.close - 1
        ics = [v for v in cross_sectional_ic(values, labels).values() if v is not None]
        if not ics:
            raise ValueError("no defined cross-sectional IC observations")
        return float(np.mean(ics))
