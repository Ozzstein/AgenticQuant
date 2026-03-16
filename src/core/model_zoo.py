"""Model Zoo: unified ML model registry for LightGBM, CatBoost, XGBoost, and Ridge."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from src.utils.config import PROJECT_ROOT, AppConfig, get_config
from src.utils.exceptions import ModelError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, str] = {
    "LightGBM": "lightgbm.LGBMRegressor",
    "CatBoost": "catboost.CatBoostRegressor",
    "XGBoost": "xgboost.XGBRegressor",
    "Linear": "sklearn.linear_model.Ridge",
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "LightGBM": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 63,
        "n_jobs": -1,
        "verbose": -1,
    },
    "CatBoost": {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "RMSE",
        "verbose": 0,
    },
    "XGBoost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_jobs": -1,
        "verbosity": 0,
    },
    "Linear": {"alpha": 1.0},
}


class ModelWrapper:
    """Unified interface for ML regression models used in the quant pipeline.

    Supports LightGBM, CatBoost, XGBoost, and sklearn Ridge via a common
    train / predict / evaluate / save / load API.
    """

    def __init__(self, model_name: str = "LightGBM", config: AppConfig | None = None) -> None:
        """Initialise the wrapper.

        Args:
            model_name: One of ``REGISTRY`` keys (LightGBM, CatBoost, XGBoost, Linear).
            config: Application config; falls back to ``get_config()`` singleton.

        Raises:
            ModelError: If *model_name* is not in ``REGISTRY``.
        """
        if model_name not in REGISTRY:
            raise ModelError(
                f"Unknown model '{model_name}'. Available: {list(REGISTRY.keys())}"
            )
        self.model_name = model_name
        self.config = config or get_config()
        self.model: Any = None
        self._params: dict[str, Any] = self._load_params()

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_available(cls) -> list[str]:
        """Return the names of all registered models.

        Returns:
            List of model name strings.
        """
        return list(REGISTRY.keys())

    @classmethod
    def load_rd_agent_config(cls, config_path: str | None = None) -> dict[str, Any]:
        """Load the best_model_config.yaml written by RD-Agent.

        Args:
            config_path: Explicit path to the YAML file. Falls back to
                ``outputs/best_model_config.yaml`` then ``config/best_model_config.yaml``.

        Returns:
            Parsed YAML contents as a dict, or ``{}`` if the file is not found.
        """
        candidates: list[Path] = []
        if config_path:
            candidates.append(Path(config_path))

        try:
            cfg = get_config()
            candidates.append(Path(cfg.output_dir) / "best_model_config.yaml")
        except Exception:
            pass

        # Also check the project config/ directory
        candidates.append(PROJECT_ROOT / "config" / "best_model_config.yaml")

        for path in candidates:
            if path.exists():
                try:
                    with open(path) as fh:
                        data = yaml.safe_load(fh) or {}
                    logger.debug("Loaded RD-Agent config from {}", path)
                    return data
                except Exception as exc:
                    logger.warning("Failed to parse {}: {}", path, exc)

        return {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_params(self) -> dict[str, Any]:
        """Load hyperparameters, preferring RD-Agent config when available.

        Priority:
        1. ``best_model_config.yaml`` if it contains params for this model.
        2. ``config.qlib.model_params`` merged over defaults when model matches.
        3. Hard-coded ``DEFAULT_PARAMS``.

        Returns:
            Merged hyperparameter dict.
        """
        defaults = dict(DEFAULT_PARAMS.get(self.model_name, {}))

        # Try RD-Agent config
        rd_cfg = self.load_rd_agent_config()
        if rd_cfg:
            rd_model = rd_cfg.get("model_name", "")
            if rd_model == self.model_name or not rd_model:
                rd_params = rd_cfg.get("params", rd_cfg.get("model_params", {}))
                if rd_params:
                    logger.debug(
                        "Using RD-Agent params for {}: {}", self.model_name, rd_params
                    )
                    defaults.update(rd_params)
                    return defaults

        # Merge config.qlib.model_params when model names match
        try:
            cfg_model = self.config.qlib.model_name
            cfg_params = self.config.qlib.model_params or {}
            if cfg_model == self.model_name and cfg_params:
                defaults.update(cfg_params)
        except Exception:
            pass

        return defaults

    def _instantiate_model(self) -> Any:
        """Lazily import and instantiate the model class.

        Returns:
            A fresh (untrained) model instance.

        Raises:
            ModelError: If the underlying library cannot be imported.
        """
        dotted = REGISTRY[self.model_name]
        module_path, class_name = dotted.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls(**self._params)
        except ImportError as exc:
            raise ModelError(
                f"Cannot import '{dotted}'. Is the library installed? ({exc})"
            ) from exc
        except Exception as exc:
            raise ModelError(f"Failed to instantiate {self.model_name}: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModelWrapper":
        """Fit the model on training data.

        Args:
            X_train: Feature matrix.
            y_train: Target labels.

        Returns:
            ``self`` for chaining.

        Raises:
            ModelError: On any training failure.
        """
        logger.info("Training {} on {} samples × {} features", self.model_name, *X_train.shape)
        try:
            self.model = self._instantiate_model()
            self.model.fit(X_train, y_train)
            logger.info("{} training complete.", self.model_name)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"{self.model_name} training failed: {exc}") from exc
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for *X*.

        Args:
            X: Feature matrix with the same columns used during training.

        Returns:
            Prediction Series indexed identically to *X*.

        Raises:
            ModelError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ModelError(f"{self.model_name} has not been trained. Call train() first.")
        try:
            preds = self.model.predict(X)
            return pd.Series(preds, index=X.index, name="prediction")
        except Exception as exc:
            raise ModelError(f"{self.model_name} prediction failed: {exc}") from exc

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Compute evaluation metrics on a hold-out set.

        Metrics:
        - **ic**: Spearman rank correlation between predictions and *y*.
        - **icir**: IC / std(IC) — approximated as the Spearman statistic itself
          (single-period; callers aggregate over time for a true ICIR).
        - **mse**: Mean squared error.
        - **r2**: Coefficient of determination.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Dict with keys ``ic``, ``icir``, ``mse``, ``r2``.

        Raises:
            ModelError: If the model has not been trained.
        """
        preds = self.predict(X)
        y_arr = np.asarray(y)
        p_arr = np.asarray(preds)

        corr, _ = spearmanr(p_arr, y_arr)
        ic = float(corr) if not np.isnan(corr) else 0.0
        mse = float(mean_squared_error(y_arr, p_arr))
        r2 = float(r2_score(y_arr, p_arr))

        # ICIR approximation: single-period IC / assumed std of 1
        # (true ICIR requires multiple time-period ICs)
        icir = ic

        return {"ic": ic, "icir": icir, "mse": mse, "r2": r2}

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances (or coefficients for linear models).

        Returns:
            Series mapping feature name → importance value, or an empty
            Series if the model has not been trained or doesn't expose
            importances.
        """
        if self.model is None:
            return pd.Series(dtype=float)

        try:
            match self.model_name:
                case "LightGBM" | "XGBoost":
                    imp = self.model.feature_importances_
                    names = self.model.feature_name_ if hasattr(self.model, "feature_name_") else None
                    if names is None and hasattr(self.model, "feature_names_in_"):
                        names = self.model.feature_names_in_
                    if names is None:
                        names = [f"f{i}" for i in range(len(imp))]
                    return pd.Series(imp, index=names, name="importance")

                case "CatBoost":
                    imp = self.model.get_feature_importance()
                    names = self.model.feature_names_ if hasattr(self.model, "feature_names_") else None
                    if names is None:
                        names = [f"f{i}" for i in range(len(imp))]
                    return pd.Series(imp, index=names, name="importance")

                case "Linear":
                    coef = self.model.coef_
                    names = (
                        list(self.model.feature_names_in_)
                        if hasattr(self.model, "feature_names_in_")
                        else [f"f{i}" for i in range(len(coef))]
                    )
                    return pd.Series(coef, index=names, name="importance")

                case _:
                    return pd.Series(dtype=float)

        except Exception as exc:
            logger.warning("Could not retrieve feature importance: {}", exc)
            return pd.Series(dtype=float)

    def save(self, path: str) -> None:
        """Serialise the trained model to disk using joblib.

        Args:
            path: File path to write (created if parent dirs exist).

        Raises:
            ModelError: If the model has not been trained or saving fails.
        """
        if self.model is None:
            raise ModelError("Cannot save an untrained model.")
        try:
            joblib.dump({"model_name": self.model_name, "model": self.model}, path)
            logger.info("Saved {} to {}", self.model_name, path)
        except Exception as exc:
            raise ModelError(f"Failed to save model to '{path}': {exc}") from exc

    def load(self, path: str) -> "ModelWrapper":
        """Load a previously saved model from disk.

        Args:
            path: File path produced by :meth:`save`.

        Returns:
            ``self`` with the restored model.

        Raises:
            ModelError: If loading fails.
        """
        try:
            data = joblib.load(path)
            if isinstance(data, dict):
                self.model = data["model"]
                self.model_name = data.get("model_name", self.model_name)
            else:
                # Legacy / raw pickle
                self.model = data
            logger.info("Loaded {} from {}", self.model_name, path)
        except Exception as exc:
            raise ModelError(f"Failed to load model from '{path}': {exc}") from exc
        return self
