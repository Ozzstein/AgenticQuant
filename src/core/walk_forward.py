"""Causal expanding walk-forward: six-month OOS windows and one carried ledger."""

from __future__ import annotations

import hashlib
import inspect
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
    ReplayResult,
    canonical_hash,
    file_hash,
    performance_metrics,
    require_matching_inputs,
    split_folds,
    write_immutable,
)
from src.domain.common import RunContext
from src.domain.portfolio import AccountSpec, FillPolicy, RankedSignal, RiskPolicy
from src.utils.exceptions import BacktestError


def frame_hash(frame) -> str:
    """Include instrument/date alignment, column names and values."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        + str(getattr(frame, "columns", getattr(frame, "name", None))).encode()
    ).hexdigest()


def _registered_model(model):
    """Only these model implementations implement the temporal training API."""
    from src.core.ensemble import EnsembleModel
    from src.core.model_zoo import ModelWrapper

    return isinstance(model, (ModelWrapper, EnsembleModel))


def _model_rows(value):
    # The shared feature/model contract uses these level names. Preserve every
    # value and row identity; the replay's caller-facing index remains unchanged.
    return value.rename_axis(index=["datetime", "instrument"])


def _fit_fold(model, x, y, available, ends, *, fit_context, feature_spec_hash, data_snapshot_hash):
    contextual = feature_spec_hash is not None
    if _registered_model(model):
        parameters = inspect.signature(model.train).parameters
        temporal = "label_available_at" in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        if contextual and not temporal:
            raise BacktestError("registered model lacks the integrated temporal training API")
        kwargs = {}
        if temporal:
            kwargs.update(
                feature_available_at=_model_rows(available),
                label_available_at=_model_rows(ends),
            )
        if contextual:
            kwargs.update(
                context=fit_context,
                feature_spec_hash=feature_spec_hash,
                data_snapshot_hash=data_snapshot_hash,
            )
        model.train(_model_rows(x), _model_rows(y), **kwargs)
        return contextual
    if contextual:
        raise BacktestError("contextual walk-forward requires ModelWrapper or EnsembleModel")
    # Explicit generic fit protocol is diagnostic only; no invented provenance.
    if not callable(getattr(model, "fit", None)):
        raise BacktestError("unsupported model: require registered temporal model or fit protocol")
    model.fit(x, y)
    return False


def _predict_fold(model, frame, available, context, contextual):
    kwargs = {}
    if _registered_model(model):
        frame = _model_rows(frame)
        if contextual:
            kwargs.update(context=context, feature_available_at=_model_rows(available))
    return np.asarray(model.predict(frame, **kwargs), dtype=float).reshape(-1)


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


def reconstruct_oos_inputs(
    result: ReplayResult, original_inputs: dict, *, signals_path=None
) -> dict:
    """Rebuild exactly the observed OOS replay without fitting or guessing signals.

    The supplied development bundle supplies market/metadata/universe events;
    the immutable result fixes the OOS span and conventions. Every reconstructed
    input hash is compared before any benchmark/cost rerun may execute.
    """
    provenance = result.manifest.get("provenance", {})
    reference = provenance.get("oos_signals_artifact")
    if not reference:
        raise BacktestError("walk-forward result lacks bound OOS signal artifact; rerun training")
    path = Path(signals_path or reference["path"])
    if file_hash(path) != reference["sha256"]:
        raise BacktestError("OOS signals artifact bytes differ from frozen hash")
    signals = [RankedSignal.model_validate(s) for s in json.loads(path.read_text())]
    if (
        canonical_hash(
            [s.model_dump(mode="json") for s in sorted(signals, key=lambda s: s.signal_id)]
        )
        != result.manifest["signals_hash"]
    ):
        raise BacktestError("OOS signal content differs from replay manifest")
    first = pd.Timestamp(provenance["oos_first_decision_at"])
    manifest = result.manifest
    account = AccountSpec.model_validate(manifest["account"])
    args = {
        **original_inputs,
        "account": account,
        "context": RunContext.model_validate(manifest["context"]),
        "bars": [b for b in original_inputs["bars"] if b.closed_at >= first],
        "signals": signals,
        "targets": [],
        "actions": [
            a for a in original_inputs.get("actions") or [] if a.effective_at > account.opened_at
        ],
        "fill_policy": FillPolicy.model_validate(manifest["fill_policy"]),
        "risk_policy": RiskPolicy.model_validate(manifest["risk_policy"]),
        "topk": manifest["topk"],
        "samples_per_year": manifest["samples_per_year"],
        "valuation_calendar": manifest["valuation_calendar"],
        "valuation_times": [pd.Timestamp(t).to_pydatetime() for t in manifest["valuation_times"]],
        "provenance": provenance,
    }
    require_matching_inputs(result, args)
    return args


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
        feature_spec_hash: str | None = None,
        data_snapshot_hash: str | None = None,
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
        if (feature_spec_hash is None) != (data_snapshot_hash is None) or any(
            value is not None
            and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value))
            for value in (feature_spec_hash, data_snapshot_hash)
        ):
            raise BacktestError("explicit feature_spec_hash and data_snapshot_hash must be SHA256")
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
                fit_context = context.model_copy(
                    update={
                        "as_of": (
                            pd.Timestamp(fold["train_end"]) - pd.Timedelta(microseconds=1)
                        ).to_pydatetime()
                    }
                )
                contextual = _fit_fold(
                    model,
                    x_train,
                    y_train,
                    feature_available_at.loc[train_mask],
                    label_end_times.loc[train_mask],
                    fit_context=fit_context,
                    feature_spec_hash=feature_spec_hash,
                    data_snapshot_hash=data_snapshot_hash,
                )
                validation_preds = np.concatenate(
                    [
                        _predict_fold(
                            model,
                            frame,
                            feature_available_at.loc[frame.index],
                            context.model_copy(update={"as_of": at.to_pydatetime()}),
                            contextual,
                        )
                        for at, frame in x_val.groupby(level=0, sort=True)
                    ]
                )
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
                    output = _predict_fold(
                        model,
                        frame,
                        feature_available_at.loc[frame.index],
                        context.model_copy(update={"as_of": at.to_pydatetime()}),
                        contextual,
                    )
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
            signal_path = (artifacts / "signals.json").resolve()
            write_immutable(
                signal_path,
                json.dumps(
                    [s.model_dump(mode="json") for s in sorted(signals, key=lambda s: s.signal_id)],
                    sort_keys=True,
                    indent=2,
                ),
            )
            run_provenance = {
                **(provenance or {}),
                "oos_signals_artifact": {
                    "path": str(signal_path),
                    "sha256": file_hash(signal_path),
                },
                "oos_first_decision_at": first.isoformat(),
                "folds": [e.model_dump(mode="json") for e in evidence],
                "features_hash": frame_hash(features),
                "labels_hash": frame_hash(labels),
                "feature_availability_hash": frame_hash(feature_available_at),
                "label_end_hash": frame_hash(label_end_times),
                "universe_version": universe_version,
                "mode": "expanding",
                "temporal_training": "contextual-v1"
                if feature_spec_hash
                else "diagnostic-unversioned",
                "feature_spec_hash": feature_spec_hash,
                "data_snapshot_hash": data_snapshot_hash,
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
