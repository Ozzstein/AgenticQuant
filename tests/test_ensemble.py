"""Tests for src/core/ensemble.py — EnsembleModel and create_model factory."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.ensemble import EnsembleModel, create_model
from src.utils.exceptions import ModelError


def _make_data(n: int = 200, p: int = 10):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.standard_normal(n), name="target")
    return X, y


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_models_from_config(self):
        """EnsembleModel uses config defaults when model_names is None."""
        e = EnsembleModel()
        assert set(e._model_names) == {"LightGBM", "XGBoost", "Linear"}

    def test_custom_model_list(self):
        """EnsembleModel accepts a custom model list."""
        e = EnsembleModel(model_names=["LightGBM", "Linear"])
        assert set(e._model_names) == {"LightGBM", "Linear"}

    def test_invalid_model_raises(self):
        """Unknown model name raises ModelError."""
        with pytest.raises(ModelError):
            EnsembleModel(model_names=["LightGBM", "NotAModel"])

    def test_model_name_property(self):
        """model_name property returns 'Ensemble(...)' format."""
        e = EnsembleModel(model_names=["LightGBM", "Linear"])
        assert e.model_name.startswith("Ensemble(")
        assert "LightGBM" in e.model_name
        assert "Linear" in e.model_name


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

class TestTrain:
    def test_equal_weight(self):
        """Equal weighting assigns uniform weights summing to 1."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["LightGBM", "Linear"], weighting="equal")
        e.train(X, y)
        weights = e.get_weights()
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        for w in weights.values():
            assert abs(w - 0.5) < 1e-6

    def test_ic_weighted(self):
        """IC-weighted training assigns weights summing to 1."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["LightGBM", "Linear"], weighting="ic_weighted")
        e.train(X, y)
        weights = e.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_custom_weights(self):
        """Custom weights are used directly after normalization."""
        X, y = _make_data()
        e = EnsembleModel(
            model_names=["LightGBM", "Linear"],
            weighting="custom",
            custom_weights={"LightGBM": 0.7, "Linear": 0.3},
        )
        e.train(X, y)
        weights = e.get_weights()
        assert abs(weights["LightGBM"] - 0.7) < 1e-6
        assert abs(weights["Linear"] - 0.3) < 1e-6

    def test_train_returns_self(self):
        """train() returns self for chaining."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["Linear"], weighting="equal")
        result = e.train(X, y)
        assert result is e

    def test_train_ndarray_input(self):
        """train() accepts numpy ndarrays."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["Linear"], weighting="equal")
        e.train(X.values, y.values)
        weights = e.get_weights()
        assert len(weights) == 1

    def test_train_dataframe_input(self):
        """train() accepts DataFrames and Series."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["Linear"], weighting="equal")
        e.train(X, y)
        weights = e.get_weights()
        assert len(weights) == 1

    def test_ic_fallback_to_equal_when_all_ics_nonpositive(self):
        """Falls back to equal weights when all validation ICs are <= 0."""
        # Use a tiny val_fraction that gives < 20 samples → equal fallback
        X, y = _make_data(n=50, p=5)
        e = EnsembleModel(
            model_names=["LightGBM", "Linear"],
            weighting="ic_weighted",
            val_fraction=0.3,  # 50 * 0.3 = 15 samples < 20 → fallback
        )
        e.train(X, y)
        weights = e.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_ndarray_in_ndarray_out(self):
        """ndarray input → ndarray output."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["Linear"], weighting="equal")
        e.train(X.values, y.values)
        preds = e.predict(X.values)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_dataframe_in_series_out(self):
        """DataFrame input → Series output."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["Linear"], weighting="equal")
        e.train(X, y)
        preds = e.predict(X)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X)

    def test_predict_before_train_raises(self):
        """predict() before train() raises ModelError."""
        X, _ = _make_data()
        e = EnsembleModel(model_names=["Linear"], weighting="equal")
        with pytest.raises(ModelError):
            e.predict(X)

    def test_predict_shape_matches_input(self):
        """Prediction length matches number of input rows."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["Linear", "LightGBM"], weighting="equal")
        e.train(X, y)
        preds = e.predict(X)
        assert len(preds) == len(X)

    def test_weighted_average_is_correct(self):
        """Blended predictions match manual weighted average."""
        X, y = _make_data()

        # Manually blend at 0.7 / 0.3
        ensemble = EnsembleModel(
            model_names=["Linear", "LightGBM"],
            weighting="custom",
            custom_weights={"Linear": 0.7, "LightGBM": 0.3},
        )
        ensemble.train(X, y)
        blended = np.asarray(ensemble.predict(X))

        # Retrain each sub-model individually (they're trained inside ensemble)
        from src.core.model_zoo import ModelWrapper
        w_linear = ModelWrapper("Linear")
        w_linear.train(X, y)
        w_lgbm = ModelWrapper("LightGBM")
        w_lgbm.train(X, y)
        expected = 0.7 * np.asarray(w_linear.predict(X)) + 0.3 * np.asarray(w_lgbm.predict(X))

        np.testing.assert_allclose(blended, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_metrics(self):
        """evaluate() returns dict with ic, icir, mse, r2 keys."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["Linear"], weighting="equal")
        e.train(X, y)
        metrics = e.evaluate(X, y)
        for key in ("ic", "icir", "mse", "r2"):
            assert key in metrics
            assert isinstance(metrics[key], float)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_get_feature_importance(self):
        """get_feature_importance() returns Series with feature names."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["Linear", "LightGBM"], weighting="equal")
        e.train(X, y)
        imp = e.get_feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        """save() then load() preserves predictions."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["Linear"], weighting="equal")
        e.train(X, y)
        preds_before = np.asarray(e.predict(X))

        path = tmp_path / "ensemble.joblib"
        e.save(str(path))

        e2 = EnsembleModel(model_names=["Linear"], weighting="equal")
        e2.load(str(path))
        preds_after = np.asarray(e2.predict(X))

        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_model_lightgbm(self):
        """create_model('LightGBM') returns ModelWrapper."""
        from src.core.model_zoo import ModelWrapper
        m = create_model("LightGBM")
        assert isinstance(m, ModelWrapper)

    def test_create_model_ensemble(self):
        """create_model('Ensemble') returns EnsembleModel."""
        m = create_model("Ensemble")
        assert isinstance(m, EnsembleModel)

    def test_create_model_unknown_raises(self):
        """create_model('Unknown') raises ModelError."""
        with pytest.raises(ModelError):
            create_model("Unknown")


# ---------------------------------------------------------------------------
# Weights introspection
# ---------------------------------------------------------------------------

class TestWeightsIntrospection:
    def test_get_weights_after_train_sums_to_one(self):
        """get_weights() after train sums to 1.0."""
        X, y = _make_data()
        e = EnsembleModel(model_names=["LightGBM", "Linear"], weighting="ic_weighted")
        e.train(X, y)
        weights = e.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_get_weights_before_train_returns_empty(self):
        """get_weights() before train returns empty dict."""
        e = EnsembleModel(model_names=["Linear"], weighting="equal")
        assert e.get_weights() == {}
