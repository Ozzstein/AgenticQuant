"""Multi-model ensemble that duck-types ModelWrapper."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.model_zoo import REGISTRY, ModelWrapper
from src.data.evidence.store import canonical_bytes
from src.data.features.identity import training_frame_identity
from src.data.features.metrics import prediction_metrics
from src.domain.common import RunContext
from src.domain.market import EnsembleComposition
from src.utils.config import AppConfig, get_config
from src.utils.exceptions import ModelError
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_MIN_VAL_SAMPLES = 20


class EnsembleModel:
    """Multi-model ensemble with IC-weighted prediction blending.

    Duck-types ModelWrapper: implements train(), predict(), evaluate(),
    get_feature_importance(), save(), load().
    """

    def __init__(
        self,
        model_names: list[str] | None = None,
        weighting: str = "ic_weighted",
        custom_weights: dict[str, float] | None = None,
        val_fraction: float = 0.15,
        config: AppConfig | None = None,
        *,
        composition: EnsembleComposition | None = None,
    ) -> None:
        """Initialise EnsembleModel.

        Args:
            model_names: Names of sub-models (must be in REGISTRY).
                Defaults to config.models.ensemble_models.
            weighting: "ic_weighted" | "equal" | "custom".
            custom_weights: Required when weighting="custom". Dict mapping
                model name → weight (will be normalised to sum to 1).
            val_fraction: Fraction of training data used for IC validation
                split (temporal, not random). Ignored for "equal" / "custom".
            config: AppConfig; falls back to get_config() singleton.

        Raises:
            ModelError: If any name in model_names is not in REGISTRY.
        """
        self._composition = composition
        self.artifact = None
        self._selection_models = {}
        self._training_frames = None
        self._selection_frames = None
        self._validation_proof = None
        self._canonical_pair = None
        self._canonical_fingerprint = None
        if composition is not None:
            # A canonical composition is the complete constructor authority.
            # Even a supplied diagnostic config cannot override it.
            config = None
            model_names = [c.model_name for c in composition.children]
            weighting = composition.weighting
            custom_weights = dict(composition.custom_weights)
            val_fraction = composition.validation_fraction or .15
        self.config = config if composition is not None else config or get_config()
        self._weighting = weighting
        self._val_fraction = val_fraction
        self._custom_weights = custom_weights or {}

        names = model_names if model_names is not None else self.config.models.ensemble_models
        # Validate all names
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise ModelError(
                f"Unknown model(s) {unknown} for EnsembleModel. Available: {list(REGISTRY)}"
            )
        if not names or len(set(names)) != len(names):
            raise ModelError("nonempty unique ensemble models required")
        if weighting not in ("ic_weighted", "equal", "custom") or not 0 < val_fraction < 1:
            raise ModelError("invalid weighting or validation fraction")
        self._model_names: list[str] = list(names)
        self._train_metadata = {}
        self.validation_evidence = {}
        self._models: dict[str, ModelWrapper] = {}
        self._weights: dict[str, float] = {}
        self._trained = False

    @property
    def model_name(self) -> str:
        """Human-readable name showing constituent models."""
        return "Ensemble" if self._composition is not None else self.display_name

    @property
    def display_name(self) -> str:
        return f"Ensemble({'+'.join(self._model_names)})"

    def _new_child(self, name):
        if self._composition is not None:
            spec = next(c for c in self._composition.children if c.model_name == name)
            return ModelWrapper(name, _exact_parameters=dict(spec.parameters))
        return ModelWrapper(name, self.config)

    def _to_df_series(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Coerce ndarray inputs to DataFrame/Series."""
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]) if isinstance(X, np.ndarray) else X
        y_s: pd.Series | None = None
        if y is not None:
            y_s = pd.Series(y) if isinstance(y, np.ndarray) else y
        return X_df, y_s

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **temporal_metadata,
    ) -> "EnsembleModel":
        """Train all sub-models and compute blend weights.

        For "ic_weighted": temporal split → per-model IC on val set → weights.
        Sub-models are re-trained on the full dataset after weight computation.
        Temporal inputs use ModelWrapper's explicit availability policy/clock
        keywords. Captured fitting additionally requires true ``label_end_at``;
        internal IC selection remains retrospective within the actual fit.

        Args:
            X_train: Feature matrix (DataFrame or ndarray).
            y_train: Target labels (Series or ndarray).

        Returns:
            self

        Raises:
            ModelError: If training fails for any sub-model.
        """
        X_df, y_s = self._to_df_series(X_train, y_train)
        if y_s is None:
            raise ModelError("y_train must not be None.")

        if X_df.empty or not X_df.index.equals(y_s.index) or not X_df.index.is_unique:
            raise ModelError("ensemble feature/label identity mismatch")
        self._validate_fit_inputs(X_df, y_s, temporal_metadata)
        if self._composition is not None:
            if not isinstance(temporal_metadata.get("context"), RunContext) or temporal_metadata.get("label_end_at") is None:
                raise ModelError("canonical ensemble requires actual context and true label_end_at")
            if any(not isinstance(temporal_metadata.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", temporal_metadata[key]) for key in ("feature_spec_hash", "data_snapshot_hash")):
                raise ModelError("canonical feature and data SHA256 identities required")
            if pd.Timestamp(temporal_metadata["context"].as_of).value % 1000:
                raise ModelError("canonical datetime contracts cannot round sub-microsecond fit cutoff")
            if not isinstance(X_df.index, pd.MultiIndex) or X_df.index.names != ["datetime", "instrument"] or X_df.index.nlevels != 2:
                raise ModelError("canonical datetime/instrument identities required")
            if any(not isinstance(v, str) or not v for v in X_df.index.get_level_values("instrument")):
                raise ModelError("canonical instrument IDs required")
            policy = temporal_metadata.get("availability_policy", "decision-time-vintage-1")
            purge = "outcome-end-before-validation-1" if policy == "captured-vintage-at-fit-1" else "label-availability-before-validation-1"
            if self._weighting == "ic_weighted" and self._composition.purge_policy != purge:
                raise ModelError("composition purge differs from training availability policy")
            from src.data.model_artifacts import validate_canonical_inputs
            validate_canonical_inputs(X_df, y_s, **{name:temporal_metadata[name]
                for name in ("feature_available_at", "label_available_at", "label_end_at")})
        # Once a new fit starts, failure must not leave old executable weights
        # paired with new validation evidence or partially refitted children.
        self._trained = False
        self._models = {}
        self._weights = {}
        self.validation_evidence = {}
        self.artifact = None
        self._selection_models = {}
        self._selection_frames = None
        self._validation_proof = None
        self._canonical_pair = None
        self._canonical_fingerprint = None
        self._train_metadata = temporal_metadata
        if self._composition is not None:
            self._training_frames = training_frame_identity(X_df, y_s, temporal_metadata["feature_available_at"],
                temporal_metadata["label_available_at"], temporal_metadata["label_end_at"])
        n = len(X_df)

        if self._weighting == "equal":
            weights = self._equal_weights()
        elif self._weighting == "ic_weighted":
            weights = self._ic_weighted_train(X_df, y_s, n)
        elif self._weighting == "custom":
            if set(self._custom_weights) != set(self._model_names):
                raise ModelError("custom weights must cover exactly the configured models")
            weights = dict(self._custom_weights) if self._composition is not None else self._normalise(self._custom_weights)
        else:
            raise ModelError("unsupported weighting")

        # Re-train all sub-models on the full dataset
        for name in self._model_names:
            wrapper = self._new_child(name)
            try:
                wrapper.train(X_df, y_s, **temporal_metadata)
            except Exception as exc:
                raise ModelError(f"EnsembleModel: sub-model '{name}' failed: {exc}") from exc
            self._models[name] = wrapper

        self._weights = weights
        self._trained = True
        _logger.info("EnsembleModel trained: {} weights={}", self.model_name, weights)
        return self

    def _equal_weights(self) -> dict[str, float]:
        n = len(self._model_names)
        return {name: 1.0 / n for name in self._model_names}

    @staticmethod
    def _clock_series(value, index: pd.Index, field: str) -> pd.Series:
        """Validate exact identities before any positional split or estimator fit."""
        if not isinstance(value, pd.Series) or not value.index.equals(index) or value.isna().any():
            raise ModelError(f"explicit aligned {field} required")
        try:
            if any(pd.Timestamp(t).tzinfo is None for t in value):
                raise ValueError("timezone missing")
            return pd.to_datetime(value, utc=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelError(f"{field} requires timezone-aware timestamps") from exc

    def _validate_fit_inputs(self, X: pd.DataFrame, y: pd.Series, metadata: dict) -> None:
        """Check the entire fit, including validation rows, before selecting weights.

        Child training checks only its own subset. In particular, retrospective
        weight selection must never inspect a future or unidentified validation
        outcome before the final full-data child training rejects it.
        """
        if not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns):
            raise ModelError("unique string feature names required")
        try:
            if not np.isfinite(y.to_numpy(dtype=float)).all():
                raise ModelError("unrealized or invalid training labels")
        except (TypeError, ValueError) as exc:
            raise ModelError("unrealized or invalid training labels") from exc
        policy = metadata.get("availability_policy", "decision-time-vintage-1")
        context = metadata.get("context")
        if policy not in ("decision-time-vintage-1", "captured-vintage-at-fit-1") or (
            context is None and policy != "decision-time-vintage-1"
        ):
            raise ModelError("supported availability policy and actual temporal context required")
        if context is None:
            return  # Non-contextual diagnostics cannot yield temporal inference authority.
        dates = ModelWrapper._dates(X.index)
        if dates.isna().any() or (dates > context.as_of).any():
            raise ModelError("missing or future feature observations cannot enter training")
        features = self._clock_series(metadata.get("feature_available_at"), X.index, "feature availability")
        labels = self._clock_series(metadata.get("label_available_at"), X.index, "label availability")
        if (features > context.as_of).any() or (labels > context.as_of).any():
            raise ModelError("future observations cannot enter training")
        if policy == "decision-time-vintage-1" and (features.to_numpy() > dates.to_numpy()).any():
            raise ModelError("revised/future features unavailable at their historical decision date")
        end_values = metadata.get("label_end_at")
        if end_values is None and policy == "captured-vintage-at-fit-1":
            raise ModelError("captured ensemble requires explicit aligned label_end_at")
        ends = self._clock_series(labels if end_values is None else end_values, X.index, "outcome ends")
        if (ends > context.as_of).any() or (ends.to_numpy() < dates.to_numpy()).any() or (ends > labels).any():
            raise ModelError("future or inconsistent outcome end/availability")
        if not metadata.get("feature_spec_hash") or not metadata.get("data_snapshot_hash"):
            raise ModelError("feature and snapshot hashes required for temporal training")

    def _ic_weighted_train(
        self, X: pd.DataFrame, y: pd.Series, n: int
    ) -> dict[str, float]:
        """Select weights retrospectively within this fit; never claim OOS inference."""
        policy = self._train_metadata.get("availability_policy", "decision-time-vintage-1")
        captured = policy == "captured-vintage-at-fit-1"
        if isinstance(X.index, (pd.DatetimeIndex, pd.MultiIndex)):
            dates = ModelWrapper._dates(X.index)
            if dates.isna().any():
                raise ModelError("missing feature observation dates")
            unique = dates.unique().sort_values()
            boundary = int(len(unique) * (1 - self._val_fraction))
            if boundary < 1 or boundary >= len(unique):
                raise ModelError("insufficient distinct validation dates")
            cutoff = unique[boundary]
            train_mask, val_mask = dates < cutoff, dates >= cutoff
            # Current captures retain their actual receipt/fit clocks. Their
            # economic outcome ends, not those later receipts, determine overlap.
            # Strict historical fitting keeps its original publication purge.
            field = "label_end_at" if captured else "label_available_at"
            purge_at = self._clock_series(self._train_metadata.get(field), X.index, field)
            train_mask &= (purge_at < cutoff).to_numpy()
        else:
            # Generic non-temporal diagnostics only. Contextual training below
            # still rejects this index; it cannot yield an eligible artifact.
            split_idx = int(n * (1.0 - self._val_fraction))
            train_mask, val_mask = np.arange(n) < split_idx, np.arange(n) >= split_idx
            cutoff = None
        if sum(val_mask) < _MIN_VAL_SAMPLES or sum(train_mask) < 2:
            raise ModelError("insufficient eligible ensemble validation data")
        X_sub, y_sub = X.loc[train_mask], y.loc[train_mask]
        X_val, y_val = X.loc[val_mask], y.loc[val_mask]
        metadata = dict(self._train_metadata)
        for field in ("feature_available_at", "label_available_at", "label_end_at"):
            if metadata.get(field) is not None:
                metadata[field] = metadata[field].loc[train_mask]
        if metadata.get("context") is not None and cutoff is not None and not captured:
            metadata["context"] = metadata["context"].model_copy(update={"as_of": (cutoff - pd.Timedelta(nanoseconds=1000)).to_pydatetime()})
        ics = {}
        predictions = {}
        for name in self._model_names:
            wrapper = self._new_child(name)
            wrapper.train(X_sub, y_sub, **metadata)
            # Intentionally non-contextual: this is internal retrospective
            # selection after whole-input eligibility checks, not external
            # temporal inference at a fabricated post-fit timestamp.
            prediction = wrapper.predict(X_val)
            ic = prediction_metrics(prediction, y_val)["ic"]
            if not math.isfinite(ic):
                raise ModelError(f"undefined validation IC for {name}")
            ics[name] = ic
            if self._composition is not None:
                self._selection_models[name] = wrapper
                predictions[name] = prediction
        if self._composition is not None:
            from src.data.model_artifacts import validation_proof
            self._selection_frames = tuple(training_frame_identity(X.loc[mask], y.loc[mask],
                self._train_metadata["feature_available_at"].loc[mask],
                self._train_metadata["label_available_at"].loc[mask],
                self._train_metadata["label_end_at"].loc[mask]) for mask in (train_mask, val_mask))
            self._validation_proof = validation_proof(y_val, predictions,
                **{field:self._train_metadata[field].loc[val_mask]
                   for field in ("feature_available_at", "label_available_at", "label_end_at")})
        fit_context = self._train_metadata.get("context")
        weight_context = metadata.get("context")
        self.validation_evidence = {
            "ic": ics, "boundary": str(cutoff), "train_rows": len(X_sub), "validation_rows": len(X_val),
            "scope": "retrospective-within-fit-weight-selection-1",
            "prediction_policy": "non-contextual-internal-validation-1",
            "availability_policy": policy,
            "fit_cutoff": fit_context.as_of.isoformat() if fit_context is not None else None,
            "weight_fit_cutoff": weight_context.as_of.isoformat() if weight_context is not None else None,
            "purge_policy": "outcome-end-before-validation-1" if captured else "label-availability-before-validation-1",
        }
        positive = {name: max(ic, 0.0) for name, ic in ics.items()}
        total = sum(positive.values())
        if total <= 0:
            raise ModelError("no model has positive held-out IC; ensemble is ineligible")
        return {name: weight / total for name, weight in positive.items()}

    def _normalise(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalise weights to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0 or any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ModelError("ensemble weights must be finite, nonnegative, and nonzero")
        return {k: v / total for k, v in weights.items()}

    def predict(self, X: pd.DataFrame | np.ndarray, **temporal_metadata) -> pd.Series | np.ndarray:
        """Generate blended predictions.

        Args:
            X: Feature matrix. If ndarray, returns ndarray; if DataFrame, returns Series.

        Returns:
            Blended predictions (weighted average of sub-model predictions).

        Raises:
            ModelError: If the ensemble has not been trained.
        """
        if not self._trained:
            raise ModelError("EnsembleModel has not been trained. Call train() first.")

        is_ndarray = isinstance(X, np.ndarray)
        X_df, _ = self._to_df_series(X)

        pred_arrays: list[np.ndarray] = []
        for name, wrapper in self._models.items():
            p = wrapper.predict(X_df, **temporal_metadata)
            pred_arrays.append(np.asarray(p))

        blended = sum(
            self._weights[name] * arr
            for name, arr in zip(self._models, pred_arrays)
        )
        blended = np.asarray(blended)

        if is_ndarray:
            return blended
        return pd.Series(blended, index=X_df.index, name="prediction")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Compute evaluation metrics on a hold-out set.

        Returns:
            Dict with ic, icir, mse, r2 keys.
        """
        return prediction_metrics(self.predict(X), y)

    def get_feature_importance(self) -> pd.Series:
        """Return weighted average of sub-model feature importances.

        Returns:
            Series mapping feature name → weighted average importance.
        """
        if not self._models:
            return pd.Series(dtype=float)

        weighted_imp: dict[str, float] = {}
        for name, wrapper in self._models.items():
            w = self._weights.get(name, 0.0)
            imp = wrapper.get_feature_importance()
            if imp.empty:
                continue
            for feat, val in imp.items():
                weighted_imp[feat] = weighted_imp.get(feat, 0.0) + w * float(val)

        return pd.Series(weighted_imp, name="importance")

    def get_weights(self) -> dict[str, float]:
        """Return current blend weights (empty if not trained).

        Returns:
            Dict mapping model name → weight.
        """
        return dict(self._weights)

    def save(self, path: str | Path) -> None:
        if not self._trained:
            raise ModelError("Cannot save an untrained EnsembleModel.")
        if self._composition is not None:
            from src.data.model_artifacts import save_ensemble
            save_ensemble(self, path)
            return
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        models = {}
        for name, wrapper in self._models.items():
            model_path = target / f"{name}.joblib"
            wrapper.save(str(model_path))
            models[name] = json.loads(Path(str(model_path) + ".json").read_text())["payload_hash"]
        manifest = {"schema_version": "2.0", "models": models, "weights": self._weights,
                    "weighting": self._weighting, "validation": self.validation_evidence}
        temporary = target / "manifest.tmp"
        temporary.write_bytes(canonical_bytes(manifest))
        os.replace(temporary, target / "manifest.json")

    def load(self, path: str | Path) -> "EnsembleModel":
        if self._composition is not None:
            from src.core.model_zoo import load_model, model_file_proof
            model = load_model(path, expected=model_file_proof(path), ensemble_composition=self._composition)
            self.__dict__.update(model.__dict__)
            return self
        # Backward-compatible direct diagnostic API only; shared canonical
        # loader never falls back to legacy directories.
        return self.load_legacy_directory(path)

    def load_legacy_directory(self, path: str | Path) -> "EnsembleModel":
        self.artifact = None
        self._composition = None
        try:
            target = Path(path)
            manifest = json.loads((target / "manifest.json").read_text())
            if manifest.get("schema_version") != "2.0" or any(name not in REGISTRY for name in manifest["models"]):
                raise ModelError("unsupported ensemble artifact")
            self._model_names = list(manifest["models"])
            self._weighting = manifest["weighting"]
            self._weights = self._normalise(manifest["weights"])
            if set(self._weights) != set(self._model_names):
                raise ModelError("ensemble weight/model mismatch")
            self._models = {}
            for name, expected_hash in manifest["models"].items():
                model_path = target / f"{name}.joblib"
                metadata = json.loads(Path(str(model_path) + ".json").read_text())
                if metadata["payload_hash"] != expected_hash:
                    raise ModelError("ensemble child hash mismatch")
                self._models[name] = ModelWrapper(name, self.config).load(str(model_path))
            self.validation_evidence = manifest["validation"]
            self._trained = True
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Failed to load ensemble artifact: {exc}") from exc
        return self


def create_model(
    name: str, config: AppConfig | None = None
) -> ModelWrapper | EnsembleModel:
    """Factory: returns EnsembleModel for 'Ensemble', ModelWrapper otherwise.

    Args:
        name: Model name. Use "Ensemble" for an ensemble; otherwise must be
            a key in REGISTRY (e.g. "LightGBM", "Linear").
        config: AppConfig; falls back to get_config() singleton.

    Returns:
        EnsembleModel if name == "Ensemble", else ModelWrapper.

    Raises:
        ModelError: If name is not "Ensemble" and not in REGISTRY.
    """
    if name == "Ensemble":
        cfg = config or get_config()
        return EnsembleModel(
            model_names=cfg.models.ensemble_models,
            weighting=cfg.models.ensemble_method,
            val_fraction=cfg.models.ensemble_val_fraction,
            config=cfg,
        )
    return ModelWrapper(name, config)
