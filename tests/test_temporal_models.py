from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.core.model_zoo import ModelWrapper
from src.data.features.metrics import prediction_metrics
from src.domain.common import RunContext
from src.utils.exceptions import ModelError


def data():
    dates = pd.date_range("2020-01-01", periods=30, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["a", "b", "c"]], names=["datetime", "instrument"])
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(len(index), 3)), index=index, columns=["x", "y", "z"])
    features.loc[index[0], "x"] = np.nan
    labels = (features.y * 0.3 - features.z * 0.2).rename("label")
    available = pd.Series(index.get_level_values("datetime"), index=index)
    label_available = available + timedelta(days=1)
    context = RunContext(run_id="model-test", mode="research", as_of=dates[-1] + timedelta(days=2),
                         config_hash="0"*64, code_version="test")
    return features, labels, available, label_available, context


@pytest.mark.parametrize("name", ["LightGBM", "CatBoost", "XGBoost", "Linear"])
def test_actual_model_family_transforms_artifact_roundtrip(tmp_path, mock_config, name):
    X, y, available, outcomes, context = data()
    wrapper = ModelWrapper(name, mock_config)
    wrapper._params = {"LightGBM": {"n_estimators": 5, "n_jobs": 1, "verbosity": -1},
                       "CatBoost": {"iterations": 5, "verbose": 0, "thread_count": 1, "allow_writing_files": False},
                       "XGBoost": {"n_estimators": 5, "n_jobs": 1}, "Linear": {"alpha": 1}}[name]
    wrapper.train(X, y, context=context, feature_available_at=available,
                  label_available_at=outcomes, feature_spec_hash="1"*64, data_snapshot_hash="2"*64)
    before = wrapper._scaler.mean_.copy()
    imputed = wrapper._imputer.statistics_.copy()
    future = X.copy() * 1e5
    wrapper.predict(future)
    np.testing.assert_array_equal(before, wrapper._scaler.mean_)
    np.testing.assert_array_equal(imputed, wrapper._imputer.statistics_)
    path = tmp_path / f"{name}.joblib"
    wrapper.save(str(path))
    loaded = ModelWrapper(name, mock_config).load(str(path))
    pd.testing.assert_series_equal(loaded.predict(X), wrapper.predict(X))
    assert loaded.artifact.feature_spec_hash == "1"*64
    assert loaded.artifact.training_end <= loaded.artifact.labels_available_through <= context.as_of
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ModelError, match="integrity"):
        loaded.load(str(path))


def test_future_revised_or_misaligned_training_rejected(mock_config):
    X, y, available, outcomes, context = data()
    wrapper = ModelWrapper("Linear", mock_config)
    with pytest.raises(ModelError, match="identity"):
        wrapper.train(X, y.iloc[::-1])
    args = dict(context=context, feature_available_at=available, label_available_at=outcomes,
                feature_spec_hash="1"*64, data_snapshot_hash="2"*64)
    outcomes.iloc[0] = context.as_of + timedelta(days=1)
    with pytest.raises(ModelError, match="future observations"):
        wrapper.train(X, y, **args)
    outcomes.iloc[0] = available.iloc[0] + timedelta(days=1)
    available.iloc[0] += timedelta(days=1)
    with pytest.raises(ModelError, match="historical decision"):
        wrapper.train(X, y, **args)


def test_icir_is_cross_sectional_not_pooled():
    index = pd.MultiIndex.from_product([pd.date_range("2020-01-01", periods=2), ["a", "b", "c"]], names=["datetime", "instrument"])
    p = pd.Series([1, 2, 3, 1, 2, 3], index=index)
    y = pd.Series([1, 2, 3, 3, 2, 1], index=index)
    metrics = prediction_metrics(p, y)
    assert metrics["ic"] == 0 and metrics["icir"] == 0
    assert np.isnan(prediction_metrics(p.iloc[:3], y.iloc[:3])["icir"])


def test_explicit_captured_fit_and_decision_preserve_economic_rows(tmp_path,mock_config):
    X,y,available,ends,context=data()
    captured=pd.Series(context.as_of-timedelta(minutes=1),index=X.index)
    args=dict(context=context,feature_available_at=captured,label_available_at=captured,
        label_end_at=ends,feature_spec_hash='1'*64,data_snapshot_hash='2'*64)
    model=ModelWrapper('Linear',mock_config)
    with pytest.raises(ModelError,match='historical decision'):
        model.train(X,y,**args)
    model.train(X,y,availability_policy='captured-vintage-at-fit-1',**args)
    model.save(str(tmp_path/'captured.joblib'))
    assert model.artifact.vintage_policy=='captured-vintage-at-fit-1'
    assert model.artifact.features_available_through==captured.max()
    assert model.artifact.label_end_through==ends.max()
    assert model.artifact.label_end_policy=='explicit-outcome-end-1'
    assert model.artifact.training_data_hash=='2'*64
    old_index=X.tail(3).index.copy()
    decision=context.model_copy(update={'as_of':context.as_of+timedelta(minutes=1)})
    with pytest.raises(ModelError,match='follow model fit'):
        model.predict(X.tail(3),context=decision,feature_available_at=captured.tail(3))
    pred=model.predict(X.tail(3),context=decision,feature_available_at=captured.tail(3),
        availability_policy='captured-vintage-at-fit-1')
    assert pred.index.equals(old_index) and np.isfinite(pred).all()
    loaded=ModelWrapper('Linear',mock_config).load(str(tmp_path/'captured.joblib'))
    assert loaded.artifact==model.artifact
    historical=context.model_copy(update={'as_of':context.as_of-timedelta(days=2)})
    with pytest.raises(ModelError,match='future observations'):
        ModelWrapper('Linear',mock_config).train(X,y,**{**args,'context':historical},
            availability_policy='captured-vintage-at-fit-1')
    with pytest.raises(ModelError,match='future feature'):
        model.predict(X.tail(3),context=decision,
            feature_available_at=pd.Series(decision.as_of+timedelta(microseconds=1),index=old_index),
            availability_policy='captured-vintage-at-fit-1')
    with pytest.raises(ModelError,match='outcome end'):
        model.train(X,y,**{**args,'label_end_at':ends.map(lambda _:context.as_of+timedelta(days=1))},
            availability_policy='captured-vintage-at-fit-1')
