"""Multi-model ensemble that duck-types ModelWrapper."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.model_zoo import REGISTRY, ModelWrapper
from src.data.evidence.store import canonical_bytes
from src.data.features.metrics import prediction_metrics
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
        self.config = config or get_config()
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
        return f"Ensemble({'+'.join(self._model_names)})"

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

        if not X_df.index.equals(y_s.index) or not X_df.index.is_unique:
            raise ModelError("ensemble feature/label identity mismatch")
        self._train_metadata = temporal_metadata
        n = len(X_df)

        if self._weighting == "equal":
            weights = self._equal_weights()
        elif self._weighting == "ic_weighted":
            weights = self._ic_weighted_train(X_df, y_s, n)
        elif self._weighting == "custom":
            if set(self._custom_weights) != set(self._model_names):
                raise ModelError("custom weights must cover exactly the configured models")
            weights = self._normalise(self._custom_weights)
        else:
            raise ModelError("unsupported weighting")

        # Re-train all sub-models on the full dataset
        self._models = {}
        for name in self._model_names:
            wrapper = ModelWrapper(name, self.config)
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

    def _ic_weighted_train(
        self, X: pd.DataFrame, y: pd.Series, n: int
    ) -> dict[str, float]:
        """Compute IC weights via temporal split, with equal fallback."""
        if isinstance(X.index, (pd.DatetimeIndex, pd.MultiIndex)):
            dates = ModelWrapper._dates(X.index)
            unique = dates.unique().sort_values()
            boundary = int(len(unique) * (1 - self._val_fraction))
            if boundary < 1 or boundary >= len(unique):
                raise ModelError("insufficient distinct validation dates")
            cutoff = unique[boundary]
            train_mask, val_mask = dates < cutoff, dates >= cutoff
            # Outcomes overlapping the validation period cannot fit even the
            # submodels used to select weights.
            labels_available = self._train_metadata.get("label_available_at")
            if labels_available is None:
                raise ModelError("IC ensemble requires label availability for temporal purging")
            train_mask &= (labels_available < cutoff).to_numpy()
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
        for field in ("feature_available_at", "label_available_at"):
            if field in metadata:
                metadata[field] = metadata[field].loc[train_mask]
        if "context" in metadata and cutoff is not None:
            metadata["context"] = metadata["context"].model_copy(update={"as_of": (cutoff - pd.Timedelta(nanoseconds=1000)).to_pydatetime()})
        ics = {}
        for name in self._model_names:
            wrapper = ModelWrapper(name, self.config)
            wrapper.train(X_sub, y_sub, **metadata)
            ic = prediction_metrics(wrapper.predict(X_val), y_val)["ic"]
            if not math.isfinite(ic):
                raise ModelError(f"undefined validation IC for {name}")
            ics[name] = ic
        self.validation_evidence = {"ic": ics, "boundary": str(cutoff), "train_rows": len(X_sub), "validation_rows": len(X_val)}
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
