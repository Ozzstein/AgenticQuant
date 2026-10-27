"""Causal expanding walk-forward: six-month OOS windows and one carried ledger."""

from __future__ import annotations

import hashlib
import io
import json
from decimal import Decimal as D
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.core.backtester import (
    FoldEvidence,
    PortfolioReplay,
    performance_metrics,
    split_folds,
)
from src.domain.portfolio import FillPolicy, RankedSignal
from src.utils.exceptions import BacktestError


def frame_hash(frame) -> str:
    """Include instrument/date alignment, column names and values."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        + str(getattr(frame, "columns", getattr(frame, "name", None))).encode()
    ).hexdigest()


def cross_sectional_ic(predictions: pd.Series, labels: pd.Series) -> dict[str, float | None]:
    if not predictions.index.equals(labels.index) or not isinstance(labels.index, pd.MultiIndex):
        raise BacktestError("IC requires exactly aligned (date,instrument) predictions/labels")
    combined = pd.DataFrame({"pred": predictions, "label": labels})
    result = {}
    for at, group in combined.groupby(level=0):
        eligible = group.replace([np.inf, -np.inf], np.nan).dropna()
        value = (
            eligible["pred"].corr(eligible["label"], method="spearman")
            if len(eligible) >= 3
            and eligible["pred"].nunique() > 1
            and eligible["label"].nunique() > 1
            else None
        )
        result[pd.Timestamp(at).isoformat()] = (
            float(value) if value is not None and np.isfinite(value) else None
        )
    return result


class WalkForwardRunner:
    def __init__(self, config=None):
        self.config = config

    def run(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        model_name="LightGBM",
        topk=30,
        cost_bps=10.0,
        walk_months=6,
        embargo_days=5,
        mode="expanding",
        model_factory=None,
        *,
        label_end_times: pd.Series | None = None,
        feature_available_at: pd.Series | None = None,
        bars=None,
        metadata=None,
        universes=None,
        context=None,
        account=None,
        ledger_path=None,
        artifact_dir=None,
        universe_version=None,
        validation_months=1,
        initial_train_months=12,
        risk_policy=None,
        fill_policy=None,
        samples_per_year=252,
        valuation_calendar="declared-equity-sessions",
        valuation_times=None,
        provenance=None,
        actions=None,
    ):
        if mode != "expanding":
            raise BacktestError("only expanding walk-forward is implemented")
        if features.empty or labels.empty:
            raise BacktestError("features and labels must not be empty")
        if any(
            v is None
            for v in (
                label_end_times,
                feature_available_at,
                bars,
                metadata,
                universes,
                context,
                account,
                ledger_path,
                artifact_dir,
                universe_version,
            )
        ):
            raise BacktestError(
                "walk-forward requires label end/feature availability, actual execution bars/metadata, context/account, ledger_path, artifact_dir and universe_version"
            )
        if (
            not isinstance(features.index, pd.MultiIndex)
            or features.index.nlevels != 2
            or features.index.has_duplicates
        ):
            raise BacktestError("unique (decision_at,instrument_id) feature index required")
        if not all(
            features.index.equals(s.index) for s in (labels, label_end_times, feature_available_at)
        ):
            raise BacktestError(
                "features, labels and availability must retain exact instrument/date alignment"
            )
        if (
            not features.index.is_monotonic_increasing
            or not np.isfinite(features.to_numpy(dtype=float)).all()
        ):
            raise BacktestError("sorted finite features required")
        dates = pd.DatetimeIndex(features.index.get_level_values(0))
        if dates.tz is None:
            raise BacktestError("decision timestamps require explicit timezone")
        ends, available = pd.DatetimeIndex(label_end_times), pd.DatetimeIndex(feature_available_at)
        if ends.tz is None or available.tz is None or ends.hasnans or available.hasnans:
            raise BacktestError("explicit timezone-aware label/feature availability required")
        if (available > dates).any() or (ends <= dates).any():
            raise BacktestError("future feature availability or invalid forward-label horizon")
        if not context.data_snapshot_ids:
            raise BacktestError("walk-forward data snapshot provenance required")
        realized_labels = labels.where(ends <= context.as_of)
        folds = split_folds(
            dates,
            dates.min().isoformat(),
            walk_months,
            embargo_days,
            validation_months,
            initial_train_months,
        )
        if not folds:
            raise BacktestError("no folds: insufficient history for training/validation/test")
        if model_factory is None:
            from src.core.ensemble import create_model

            def model_factory():
                return create_model(model_name, self.config)

        artifacts = Path(artifact_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        instruments = {b.instrument.instrument_id: b.instrument for b in bars}
        signals, evidence, models = [], [], []
        predicted = []
        incomplete = False
        try:
            for number, fold in enumerate(folds):
                train_raw = (dates >= fold["train_start"]) & (dates < fold["train_end"])
                train_mask = train_raw & (ends < pd.Timestamp(fold["train_end"]))
                val_raw = (dates >= fold["validation_start"]) & (dates < fold["validation_end"])
                val_mask = val_raw & (ends < pd.Timestamp(fold["validation_end"]))
                test_mask = (dates >= fold["test_start"]) & (dates < fold["test_end"])
                if not train_mask.any() or not val_mask.any() or not test_mask.any():
                    raise BacktestError(f"fold {number} has an empty purged segment")
                if (
                    not np.isfinite(labels.loc[train_mask]).all()
                    or not np.isfinite(labels.loc[val_mask]).all()
                ):
                    raise BacktestError(f"fold {number} has unavailable training/validation labels")
                x_train, y_train = features.loc[train_mask], labels.loc[train_mask]
                x_val, y_val = features.loc[val_mask], labels.loc[val_mask]
                model = model_factory()
                if any(model is old for old in models):
                    raise BacktestError("model_factory reused fitted state across folds")
                models.append(model)
                if hasattr(model, "train_with_validation"):
                    model.train_with_validation(x_train, y_train, x_val, y_val)
                elif hasattr(model, "train"):
                    model.train(x_train, y_train)
                else:
                    model.fit(x_train, y_train)
                validation_preds = np.asarray(model.predict(x_val), dtype=float).reshape(-1)
                if len(validation_preds) != len(x_val) or not np.isfinite(validation_preds).all():
                    raise BacktestError(f"fold {number} invalid validation predictions")
                # Serialize the actual fitted model; no invented artifact version.
                model_path = artifacts / f"fold-{number}.joblib"
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                model_bytes = buffer.getvalue()
                model_version = hashlib.sha256(model_bytes).hexdigest()
                if model_path.exists() and model_path.read_bytes() != model_bytes:
                    raise BacktestError(
                        "fitted artifact changed; use a new immutable run/artifact directory"
                    )
                if not model_path.exists():
                    with model_path.open("xb") as destination:
                        destination.write(model_bytes)
                fold_predictions = []
                # Date-local inference prevents transforms that normalize over a future
                # test batch from changing earlier predictions.
                for at, frame in features.loc[test_mask].groupby(level=0, sort=True):
                    output = np.asarray(model.predict(frame), dtype=float).reshape(-1)
                    if len(output) != len(frame) or not np.isfinite(output).all():
                        raise BacktestError(f"fold {number} invalid OOS predictions at {at}")
                    fold_predictions.append(pd.Series(output, index=frame.index))
                    for index, score in zip(frame.index, output, strict=True):
                        key = index[1]
                        if key not in instruments:
                            raise BacktestError(
                                f"prediction instrument lacks execution prices: {key}"
                            )
                        signals.append(
                            RankedSignal(
                                signal_id=f"fold-{number}:{at.isoformat()}:{key}",
                                instrument=instruments[key],
                                decision_at=at.to_pydatetime(),
                                available_at=pd.Timestamp(
                                    feature_available_at.loc[index]
                                ).to_pydatetime(),
                                score=float(score),
                                model_version=model_version,
                                universe_version=universe_version,
                                data_snapshot_id=context.data_snapshot_ids[0],
                                strategy_id="model-topk",
                            )
                        )
                p = pd.concat(fold_predictions).sort_index()
                predicted.append(p)
                ic = cross_sectional_ic(p, realized_labels.loc[p.index])
                values = [v for v in ic.values() if v is not None]
                importance = {}
                if hasattr(model, "get_feature_importance"):
                    importance = {
                        str(k): float(v)
                        for k, v in model.get_feature_importance().items()
                        if np.isfinite(v)
                    }
                evidence.append(
                    FoldEvidence(
                        fold_id=number,
                        **{
                            k: fold[k]
                            for k in (
                                "train_start",
                                "train_end",
                                "validation_start",
                                "validation_end",
                                "test_start",
                                "test_end",
                            )
                        },
                        train_rows=int(train_mask.sum()),
                        validation_rows=int(val_mask.sum()),
                        test_rows=int(test_mask.sum()),
                        purged_rows=int(
                            train_raw.sum() - train_mask.sum() + val_raw.sum() - val_mask.sum()
                        ),
                        embargo_days=embargo_days,
                        model_version=model_version,
                        train_hash=frame_hash(x_train) + ":" + frame_hash(y_train),
                        validation_hash=frame_hash(x_val) + ":" + frame_hash(y_val),
                        predictions_hash=frame_hash(p),
                        label_end_max=ends[train_mask].max().isoformat(),
                        validation_label_end_max=ends[val_mask].max().isoformat(),
                        ic=float(np.mean(values)) if values else None,
                        feature_importance=importance,
                    )
                )
                incomplete |= not fold["complete_window"]
            predictions = pd.concat(predicted).sort_index()
            if predictions.index.has_duplicates:
                raise BacktestError("overlapping OOS prediction observations")
            # Replay the entire continuous OOS span once: no fold reset/liquidation.
            first = min(s.decision_at for s in signals)
            execution_bars = [b for b in bars if b.closed_at >= first]
            oos_start = min(b.opened_at for b in execution_bars)
            oos_account = account.model_copy(
                update={"opened_at": oos_start - pd.Timedelta(microseconds=1)}
            )
            run_provenance = {
                **(provenance or {}),
                "folds": [e.model_dump(mode="json") for e in evidence],
                "features_hash": frame_hash(features),
                "labels_hash": frame_hash(labels),
                "feature_availability_hash": frame_hash(feature_available_at),
                "label_end_hash": frame_hash(label_end_times),
                "universe_version": universe_version,
                "mode": "expanding",
                "walk_months": walk_months,
                "model_artifacts": [
                    str((artifacts / f"fold-{i}.joblib").resolve()) for i in range(len(evidence))
                ],
            }
            result = PortfolioReplay(ledger_path).run(
                context=context,
                account=oos_account,
                bars=execution_bars,
                metadata=metadata,
                universes=universes,
                signals=signals,
                topk=topk,
                actions=[a for a in (actions or []) if a.effective_at >= oos_start],
                fill_policy=fill_policy or FillPolicy(equity_cost_bps=D(str(cost_bps))),
                risk_policy=risk_policy,
                samples_per_year=samples_per_year,
                valuation_calendar=valuation_calendar,
                valuation_times=[t for t in valuation_times if t >= first]
                if valuation_times
                else None,
                provenance=run_provenance,
            )
            for fold in evidence:
                observations = [
                    s
                    for s in result.snapshots
                    if pd.Timestamp(fold.test_start) <= s.as_of < pd.Timestamp(fold.test_end)
                ]
                preceding = [s for s in result.snapshots if s.as_of < pd.Timestamp(fold.test_start)]
                if observations and preceding:
                    fold.oos_sharpe = performance_metrics(
                        [float(preceding[-1].nav), *[float(s.nav) for s in observations]],
                        samples_per_year,
                    ).sharpe_ratio
            diagnostics = (
                ["final OOS window is incomplete; promotion prohibited"] if incomplete else []
            )
            result = result.model_copy(
                update={
                    "walk_forward_results": evidence,
                    "cross_sectional_ic": cross_sectional_ic(
                        predictions, realized_labels.loc[predictions.index]
                    ),
                    "status": "incomplete" if incomplete else "complete",
                    "diagnostics": diagnostics,
                }
            )
            (artifacts / "signals.json").write_text(
                json.dumps([s.model_dump(mode="json") for s in signals], indent=2)
            )
            (artifacts / "result.json").write_text(result.model_dump_json(indent=2))
            # Store the enhanced result in the same ledger run record.
            from src.execution.ledger import PortfolioLedger

            ledger = PortfolioLedger(ledger_path)
            try:
                row = ledger.db.execute(
                    "SELECT payload FROM runs WHERE account=? AND id=?",
                    (account.account_id, context.run_id),
                ).fetchone()
                payload = json.loads(row[0])
                payload.update(status=result.status, result=result.model_dump(mode="json"))
                ledger.save_run(account.account_id, context.run_id, payload)
            finally:
                ledger.close()
            return result
        except Exception as exc:
            (artifacts / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "run_id": context.run_id,
                        "error": str(exc),
                        "completed_fold_evidence": [e.model_dump(mode="json") for e in evidence],
                    },
                    indent=2,
                )
            )
            if isinstance(exc, BacktestError):
                raise
            raise BacktestError(f"walk-forward failed: {exc}") from exc

    def generate_html_report(self, result, output_path: str) -> str:
        """Export actual NAV and fills; no synthetic timestamp interpolation."""
        import plotly.graph_objects as go

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = go.Figure(go.Scatter(x=result.timestamps, y=result.equity_curve, name="OOS NAV"))
        fig.update_layout(title=f"{result.run_id} — {result.status}", yaxis_title="Account NAV")
        fig.write_html(str(out), include_plotlyjs=True)
        return str(out.resolve())
