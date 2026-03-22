"""Multi-model ensemble that duck-types ModelWrapper."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from src.core.model_zoo import REGISTRY, ModelWrapper
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
        self._model_names: list[str] = list(names)
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
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        y_s: pd.Series | None = None
        if y is not None:
            y_s = pd.Series(y) if isinstance(y, np.ndarray) else y
        return X_df, y_s

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
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
        assert y_s is not None

        n = len(X_df)

        if self._weighting == "equal":
            weights = self._equal_weights()
        elif self._weighting == "ic_weighted":
            weights = self._ic_weighted_train(X_df, y_s, n)
        elif self._weighting == "custom":
            weights = self._normalise(self._custom_weights)
        else:
            _logger.warning("Unknown weighting '{}'; falling back to equal.", self._weighting)
            weights = self._equal_weights()

        # Re-train all sub-models on the full dataset
        self._models = {}
        for name in self._model_names:
            wrapper = ModelWrapper(name, self.config)
            try:
                wrapper.train(X_df, y_s)
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
        split_idx = int(n * (1.0 - self._val_fraction))
        n_val = n - split_idx

        if n_val < _MIN_VAL_SAMPLES:
            _logger.warning(
                "Validation set has {} samples (< {}). Falling back to equal weights.",
                n_val,
                _MIN_VAL_SAMPLES,
            )
            return self._equal_weights()

        X_sub = X.iloc[:split_idx]
        y_sub = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]

        ics: dict[str, float] = {}
        for name in self._model_names:
            wrapper = ModelWrapper(name, self.config)
            try:
                wrapper.train(X_sub, y_sub)
                val_preds = wrapper.predict(X_val)
                corr_result = spearmanr(np.asarray(val_preds), np.asarray(y_val))
                ic = float(corr_result.statistic)
                if not math.isfinite(ic):
                    ic = 0.0
            except Exception as exc:
                _logger.warning("IC computation failed for '{}': {}", name, exc)
                ic = 0.0
            ics[name] = ic
            _logger.debug("IC for '{}': {:.4f}", name, ic)

        # If all ICs are <= 0, fall back to equal
        positive = {n: max(ic, 0.0) for n, ic in ics.items()}
        total = sum(positive.values())
        if total <= 0:
            _logger.warning("All validation ICs <= 0. Falling back to equal weights.")
            return self._equal_weights()

        return {name: w / total for name, w in positive.items()}

    def _normalise(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalise weights to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            return self._equal_weights()
        return {k: v / total for k, v in weights.items()}

    def predict(self, X: pd.DataFrame | np.ndarray) -> pd.Series | np.ndarray:
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
            p = wrapper.predict(X_df)
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
        preds = self.predict(X)
        y_arr = np.asarray(y)
        p_arr = np.asarray(preds)

        corr_result = spearmanr(p_arr, y_arr)
        ic = float(corr_result.statistic) if not np.isnan(corr_result.statistic) else 0.0
        mse = float(mean_squared_error(y_arr, p_arr))
        r2 = float(r2_score(y_arr, p_arr))

        return {"ic": ic, "icir": ic, "mse": mse, "r2": r2}

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
        """Serialise the ensemble to disk.

        Args:
            path: Destination file path.

        Raises:
            ModelError: If not trained or saving fails.
        """
        if not self._trained:
            raise ModelError("Cannot save an untrained EnsembleModel.")
        try:
            data: dict[str, Any] = {
                "model_names": self._model_names,
                "weighting": self._weighting,
                "weights": self._weights,
                "models": {name: w.model for name, w in self._models.items()},
            }
            joblib.dump(data, str(path))
            _logger.info("Saved EnsembleModel to {}", path)
        except Exception as exc:
            raise ModelError(f"Failed to save EnsembleModel: {exc}") from exc

    def load(self, path: str | Path) -> "EnsembleModel":
        """Load a previously saved ensemble from disk.

        Args:
            path: File path produced by save().

        Returns:
            self with restored state.

        Raises:
            ModelError: If loading fails.
        """
        try:
            data = joblib.load(str(path))
            self._model_names = data["model_names"]
            self._weighting = data["weighting"]
            self._weights = data["weights"]
            self._models = {}
            for name, raw_model in data["models"].items():
                wrapper = ModelWrapper(name, self.config)
                wrapper.model = raw_model
                self._models[name] = wrapper
            self._trained = True
            _logger.info("Loaded EnsembleModel from {}", path)
        except Exception as exc:
            raise ModelError(f"Failed to load EnsembleModel: {exc}") from exc
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
