"""Unit tests for src/core/model_zoo.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.core.model_zoo import DEFAULT_PARAMS, REGISTRY, ModelWrapper
from src.utils.exceptions import ModelError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_data(n: int = 100, p: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(RNG.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(RNG.standard_normal(n), name="label")
    return X, y


# ---------------------------------------------------------------------------
# Registry / listing
# ---------------------------------------------------------------------------


def test_list_available() -> None:
    models = ModelWrapper.list_available()
    assert isinstance(models, list)
    for name in ("LightGBM", "CatBoost", "XGBoost", "Linear"):
        assert name in models


def test_registry_keys_match_list() -> None:
    assert set(ModelWrapper.list_available()) == set(REGISTRY.keys())


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------


def test_default_params_lightgbm() -> None:
    w = ModelWrapper("LightGBM")
    assert w._params["n_estimators"] == 500


def test_default_params_catboost() -> None:
    w = ModelWrapper("CatBoost")
    assert w._params["iterations"] == 500
    assert w._params["verbose"] == 0


def test_default_params_xgboost() -> None:
    w = ModelWrapper("XGBoost")
    assert w._params["n_estimators"] == 500


def test_default_params_linear() -> None:
    w = ModelWrapper("Linear")
    assert w._params["alpha"] == 1.0


def test_unknown_model_raises() -> None:
    with pytest.raises(ModelError, match="Unknown model"):
        ModelWrapper("FancyNet")


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_instantiate_lightgbm() -> None:
    w = ModelWrapper("LightGBM")
    model = w._instantiate_model()
    assert model is not None


def test_instantiate_linear() -> None:
    w = ModelWrapper("Linear")
    model = w._instantiate_model()
    assert model is not None


# ---------------------------------------------------------------------------
# Train / predict — LightGBM
# ---------------------------------------------------------------------------


def test_train_predict_lightgbm() -> None:
    X, y = _make_data()
    w = ModelWrapper("LightGBM")
    w.train(X, y)
    preds = w.predict(X)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(X)
    assert preds.index.equals(X.index)


def test_train_returns_self_lightgbm() -> None:
    X, y = _make_data()
    w = ModelWrapper("LightGBM")
    result = w.train(X, y)
    assert result is w


# ---------------------------------------------------------------------------
# Train / predict — Linear (Ridge)
# ---------------------------------------------------------------------------


def test_train_predict_linear() -> None:
    X, y = _make_data()
    w = ModelWrapper("Linear")
    w.train(X, y)
    preds = w.predict(X)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# Predict before train raises
# ---------------------------------------------------------------------------


def test_predict_before_train_raises() -> None:
    w = ModelWrapper("LightGBM")
    X, _ = _make_data()
    with pytest.raises(ModelError, match="not been trained"):
        w.predict(X)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_metrics() -> None:
    X, y = _make_data()
    w = ModelWrapper("LightGBM")
    w.train(X, y)
    metrics = w.evaluate(X, y)
    assert isinstance(metrics, dict)
    for key in ("ic", "icir", "mse", "r2"):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_evaluate_ic_range() -> None:
    X, y = _make_data()
    w = ModelWrapper("Linear")
    w.train(X, y)
    metrics = w.evaluate(X, y)
    assert -1.0 <= metrics["ic"] <= 1.0


def test_evaluate_mse_nonnegative() -> None:
    X, y = _make_data()
    w = ModelWrapper("Linear")
    w.train(X, y)
    assert w.evaluate(X, y)["mse"] >= 0.0


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def test_feature_importance_lightgbm() -> None:
    X, y = _make_data()
    w = ModelWrapper("LightGBM")
    w.train(X, y)
    imp = w.get_feature_importance()
    assert isinstance(imp, pd.Series)
    assert len(imp) == X.shape[1]
    assert (imp >= 0).all()


def test_feature_importance_linear() -> None:
    X, y = _make_data()
    w = ModelWrapper("Linear")
    w.train(X, y)
    imp = w.get_feature_importance()
    assert isinstance(imp, pd.Series)
    assert len(imp) == X.shape[1]


def test_feature_importance_untrained_empty() -> None:
    w = ModelWrapper("LightGBM")
    imp = w.get_feature_importance()
    assert isinstance(imp, pd.Series)
    assert len(imp) == 0


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path) -> None:
    X, y = _make_data()
    w = ModelWrapper("LightGBM")
    w.train(X, y)
    preds_before = w.predict(X).to_numpy()

    save_path = str(tmp_path / "lgbm_model.pkl")
    w.save(save_path)

    w2 = ModelWrapper("LightGBM")
    w2.load(save_path)
    preds_after = w2.predict(X).to_numpy()

    np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_save_untrained_raises(tmp_path) -> None:
    w = ModelWrapper("LightGBM")
    with pytest.raises(ModelError, match="untrained"):
        w.save(str(tmp_path / "model.pkl"))


def test_load_returns_self(tmp_path) -> None:
    X, y = _make_data()
    w = ModelWrapper("Linear")
    w.train(X, y)
    path = str(tmp_path / "linear.pkl")
    w.save(path)

    w2 = ModelWrapper("Linear")
    result = w2.load(path)
    assert result is w2


# ---------------------------------------------------------------------------
# RD-Agent config loading
# ---------------------------------------------------------------------------


def test_load_rd_agent_config_missing() -> None:
    cfg = ModelWrapper.load_rd_agent_config("/nonexistent/path/best_model_config.yaml")
    assert cfg == {}


def test_load_rd_agent_config_found(tmp_path) -> None:
    cfg_data = {"model_name": "LightGBM", "params": {"n_estimators": 200, "learning_rate": 0.01}}
    cfg_file = tmp_path / "best_model_config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data))

    loaded = ModelWrapper.load_rd_agent_config(str(cfg_file))
    assert loaded["model_name"] == "LightGBM"
    assert loaded["params"]["n_estimators"] == 200


def test_load_params_uses_best_config(tmp_path, monkeypatch) -> None:
    """When best_model_config.yaml exists with matching model, override defaults."""
    cfg_data = {"model_name": "LightGBM", "params": {"n_estimators": 42, "learning_rate": 0.001}}
    cfg_file = tmp_path / "best_model_config.yaml"
    cfg_file.write_text(yaml.dump(cfg_data))

    # Patch load_rd_agent_config to return our test data
    monkeypatch.setattr(
        ModelWrapper,
        "load_rd_agent_config",
        classmethod(lambda cls, p=None: cfg_data),
    )

    w = ModelWrapper("LightGBM")
    assert w._params["n_estimators"] == 42
    assert w._params["learning_rate"] == 0.001
