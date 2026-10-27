"""Model Zoo: unified ML model registry for LightGBM, CatBoost, XGBoost, and Ridge."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.evidence.store import canonical_bytes, digest
from src.data.features.metrics import prediction_metrics
from src.domain.common import RunContext
from src.domain.market import ModelArtifact
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
        "allow_writing_files": False,
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
        self._feature_names: tuple[str, ...] = ()
        self._imputer = None
        self._scaler = None
        self._provenance = None
        self.artifact: ModelArtifact | None = None

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

        # Research candidates cannot mutate active training by dropping a YAML
        # file into outputs. Only explicit evaluated config is consumed here.
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

    @staticmethod
    def _dates(index: pd.Index) -> pd.DatetimeIndex:
        if isinstance(index, pd.MultiIndex):
            level = next((name for name in ("datetime", "date", "timestamp") if name in index.names), None)
            if level is None:
                raise ModelError("date index level is required")
            index = index.get_level_values(level)
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise ModelError("temporal training requires timezone-aware dates")
        return index

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, *, context: RunContext | None = None,
              feature_available_at: pd.Series | None = None, label_available_at: pd.Series | None = None,
              feature_spec_hash: str | None = None, data_snapshot_hash: str | None = None,
              label_end_at: pd.Series | None = None,
              availability_policy: str = "decision-time-vintage-1") -> "ModelWrapper":
        if availability_policy not in ("decision-time-vintage-1", "captured-vintage-at-fit-1") or (context is None and availability_policy != "decision-time-vintage-1"):
            raise ModelError("supported availability policy and actual temporal context required")
        if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
            raise ModelError("training requires indexed DataFrame/Series")
        if X_train.empty or not X_train.index.is_unique or not X_train.index.equals(y_train.index):
            raise ModelError("training data/label identity must match exactly and be unique/nonempty")
        if not X_train.columns.is_unique or not all(isinstance(c, str) for c in X_train.columns):
            raise ModelError("unique string feature names required")
        if not np.isfinite(y_train.to_numpy(dtype=float)).all():
            raise ModelError("unrealized or invalid training labels")
        provenance = None
        if context is not None:
            dates = self._dates(X_train.index)
            for availability in (feature_available_at, label_available_at):
                if availability is None or not availability.index.equals(X_train.index) or availability.isna().any():
                    raise ModelError("explicit aligned feature and label availability required")
                if any(pd.Timestamp(t).tzinfo is None for t in availability):
                    raise ModelError("availability requires a timezone")
                if (availability > context.as_of).any():
                    raise ModelError("future observations cannot enter training")
            if (dates > context.as_of).any():
                raise ModelError("future feature observations cannot enter training")
            if availability_policy == "decision-time-vintage-1" and (feature_available_at.to_numpy() > dates.to_numpy()).any():
                raise ModelError("revised/future features unavailable at their historical decision date")
            ends = label_available_at if label_end_at is None else label_end_at
            if not ends.index.equals(X_train.index) or ends.isna().any() or any(pd.Timestamp(t).tzinfo is None for t in ends):
                raise ModelError("explicit aligned outcome ends require timezone-aware timestamps")
            if (ends > context.as_of).any() or (ends.to_numpy() < dates.to_numpy()).any() or (ends > label_available_at).any():
                raise ModelError("future or inconsistent outcome end/availability")
            if not feature_spec_hash or not data_snapshot_hash:
                raise ModelError("feature and snapshot hashes required for temporal training")
            provenance = dict(feature_spec_hash=feature_spec_hash, training_data_hash=data_snapshot_hash,
                              fitted_at=context.as_of, training_start=dates.min().to_pydatetime(),
                              training_end=dates.max().to_pydatetime(),
                              labels_available_through=pd.Timestamp(label_available_at.max()).to_pydatetime(),
                              features_available_through=pd.Timestamp(feature_available_at.max()).to_pydatetime(),
                              label_end_through=pd.Timestamp(ends.max()).to_pydatetime(),
                              label_end_policy="availability-upper-bound-1" if label_end_at is None else "explicit-outcome-end-1",
                              vintage_policy=availability_policy,
                              code_version=context.code_version)
        try:
            clean = X_train.astype(float).replace([np.inf, -np.inf], np.nan)
            if clean.isna().all().any():
                raise ModelError("required feature has no eligible training coverage")
            self._feature_names = tuple(X_train.columns)
            self._imputer = SimpleImputer(strategy="median")
            self._scaler = StandardScaler()
            values = self._scaler.fit_transform(self._imputer.fit_transform(clean))
            transformed = pd.DataFrame(values, index=X_train.index, columns=self._feature_names)
            self.model = self._instantiate_model()
            self.model.fit(transformed, y_train)
            self._provenance = provenance
            self.artifact = None
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"{self.model_name} training failed: {exc}") from exc
        return self

    def predict(self, X: pd.DataFrame, *, context: RunContext | None = None,
                feature_available_at: pd.Series | None = None,
                availability_policy: str = "decision-time-vintage-1") -> pd.Series:
        if availability_policy not in ("decision-time-vintage-1", "captured-vintage-at-fit-1") or (context is None and availability_policy != "decision-time-vintage-1"):
            raise ModelError("supported availability policy and actual temporal context required")
        if self.model is None or self._imputer is None:
            raise ModelError(f"{self.model_name} has not been trained. Call train() first.")
        if tuple(X.columns) != self._feature_names or not X.index.is_unique:
            raise ModelError("prediction feature schema/order or instrument/date identity mismatch")
        if context is not None:
            if self._provenance is None:
                raise ModelError("unversioned model cannot perform temporal inference")
            dates = self._dates(X.index)
            if self._provenance["fitted_at"] >= context.as_of or (dates > context.as_of).any():
                raise ModelError("model fit must precede inference decision and observations must precede cutoff")
            if availability_policy == "decision-time-vintage-1" and (dates <= self._provenance["fitted_at"]).any():
                raise ModelError("inference observations must follow model fit and precede cutoff")
            if feature_available_at is None or not feature_available_at.index.equals(X.index) or feature_available_at.isna().any():
                raise ModelError("explicit prediction feature availability required")
            if any(pd.Timestamp(t).tzinfo is None for t in feature_available_at):
                raise ModelError("prediction availability requires a timezone")
            if (feature_available_at > context.as_of).any():
                raise ModelError("future feature in inference")
            if availability_policy == "decision-time-vintage-1" and (feature_available_at.to_numpy() > dates.to_numpy()).any():
                raise ModelError("future feature in historical inference")
        try:
            clean = X.astype(float).replace([np.inf, -np.inf], np.nan)
            transformed = pd.DataFrame(self._scaler.transform(self._imputer.transform(clean)), index=X.index, columns=self._feature_names)
            preds = self.model.predict(transformed)
            if not np.isfinite(preds).all():
                raise ModelError("model produced invalid predictions")
            return pd.Series(preds, index=X.index, name="prediction")
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"{self.model_name} prediction failed: {exc}") from exc

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        return prediction_metrics(self.predict(X), y)

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
        """Save model+fitted transforms with a hashed, versioned sidecar.

        Only load application-owned artifacts: joblib deserialization is for
        trusted local files, never raw research/provider downloads.
        """
        if self.model is None:
            raise ModelError("Cannot save an untrained model.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": "2.0", "model_name": self.model_name, "model": self.model,
                "feature_names": self._feature_names, "imputer": self._imputer, "scaler": self._scaler,
                "provenance": self._provenance, "params": self._params}
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".model-")
        os.close(fd)
        try:
            joblib.dump(data, temporary)
            payload_hash = digest(Path(temporary).read_bytes())
            metadata = {"schema_version": "2.0", "payload_hash": payload_hash, "temporal_provenance": self._provenance is not None}
            if self._provenance:
                packages = ("numpy", "pandas", "scikit-learn", "joblib", REGISTRY[self.model_name].split(".")[0])
                environment = []
                for package in dict.fromkeys(packages):
                    try:
                        environment.append((package, importlib.metadata.version(package)))
                    except importlib.metadata.PackageNotFoundError:
                        pass
                self.artifact = ModelArtifact(model_name=self.model_name, payload_hash=payload_hash,
                                              feature_names=self._feature_names, environment=tuple(environment),
                                              parameters_json=canonical_bytes(self._params).decode(), **self._provenance)
                metadata["artifact"] = self.artifact.model_dump(mode="json")
            # Content mismatch during a concurrent read fails closed. Activation
            # is a coordinator-owned atomic manifest pointer, not these files.
            os.replace(temporary, target)
            sidecar = Path(str(target) + ".json")
            side_temp = Path(str(sidecar) + ".tmp")
            side_temp.write_bytes(canonical_bytes(metadata))
            os.replace(side_temp, sidecar)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def load(self, path: str) -> "ModelWrapper":
        try:
            metadata = json.loads(Path(str(path) + ".json").read_text())
            if metadata.get("schema_version") != "2.0" or digest(Path(path).read_bytes()) != metadata["payload_hash"]:
                raise ModelError("model artifact version/integrity mismatch")
            data = joblib.load(path)
            if data.get("schema_version") != "2.0" or data["model_name"] not in REGISTRY:
                raise ModelError("unsupported model artifact")
            self.model, self.model_name = data["model"], data["model_name"]
            self._feature_names = tuple(data["feature_names"])
            self._imputer, self._scaler = data["imputer"], data["scaler"]
            self._provenance, self._params = data["provenance"], data["params"]
            self.artifact = ModelArtifact.model_validate(metadata["artifact"]) if metadata.get("artifact") else None
            if self.artifact and self.artifact.payload_hash != metadata["payload_hash"]:
                raise ModelError("model metadata integrity mismatch")
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Failed to load versioned model: {exc}") from exc
        return self
