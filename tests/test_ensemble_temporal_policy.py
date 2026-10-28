"""Synthetic fixtures exercise real estimators, never research/performance authority."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.ensemble import EnsembleModel
from src.core.model_zoo import DEFAULT_PARAMS, ModelWrapper
from src.domain.common import RunContext
from src.utils.exceptions import ModelError


@pytest.fixture
def fit_data():
    dates = pd.bdate_range("2020-01-02", periods=80, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["a", "b", "c", "d"]], names=["datetime", "instrument"])
    rng = np.random.default_rng(2026)
    X = pd.DataFrame(rng.normal(size=(len(index), 2)), index=index, columns=["x", "z"])
    y = (3 * X.x - X.z).rename("label")
    X.iloc[5, 1] = np.nan  # Imputation must use the eligible training subset only.
    context = RunContext(run_id="ensemble-policy-test", mode="research",
                         as_of=pd.Timestamp("2026-10-28T18:00:00Z").to_pydatetime(),
                         config_hash="0" * 64, code_version="test")
    metadata = dict(context=context, feature_spec_hash="1" * 64, data_snapshot_hash="2" * 64,
                    feature_available_at=pd.Series(context.as_of - pd.Timedelta(minutes=2), index=index),
                    label_available_at=pd.Series(context.as_of - pd.Timedelta(minutes=1), index=index),
                    label_end_at=pd.Series(index.get_level_values("datetime") + pd.Timedelta(days=3), index=index),
                    availability_policy="captured-vintage-at-fit-1")
    return X, y, metadata


@pytest.fixture
def actual_fits(monkeypatch):
    """Record real fits; small real estimators keep the four-family test bounded."""
    parameters = {
        "Linear": {"alpha": 1},
        "LightGBM": {"n_estimators": 20, "n_jobs": 1, "verbosity": -1},
        "CatBoost": {"iterations": 20, "verbose": 0, "thread_count": 1, "allow_writing_files": False},
        "XGBoost": {"n_estimators": 20, "n_jobs": 1},
    }
    for name, params in parameters.items():
        monkeypatch.setitem(DEFAULT_PARAMS, name, params)
    fits = []
    original = ModelWrapper.train

    def train(wrapper, X, y, **metadata):
        result = original(wrapper, X, y, **metadata)
        fits.append((wrapper, X.copy(deep=True), y.copy(deep=True), metadata))
        return result

    monkeypatch.setattr(ModelWrapper, "train", train)
    return fits


def ensemble(config, names=None):
    return EnsembleModel(model_names=names or ["Linear"], val_fraction=0.3, config=config)


def forbid_fit(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid whole-fit input reached child weight selection/training")

    monkeypatch.setattr(ModelWrapper, "train", unexpected)


def test_four_actual_estimators_use_economic_purge_and_real_fit_clock(
    fit_data, actual_fits, mock_config, tmp_path, monkeypatch
):
    X, y, metadata = fit_data
    input_before = X.copy(deep=True)
    names = ["Linear", "LightGBM", "CatBoost", "XGBoost"]
    predictions = []
    original_predict = ModelWrapper.predict

    def predict(wrapper, frame, **kwargs):
        predictions.append((frame.index, kwargs))
        return original_predict(wrapper, frame, **kwargs)

    monkeypatch.setattr(ModelWrapper, "predict", predict)
    model = ensemble(mock_config, names).train(X, y, **metadata)
    boundary = pd.Timestamp(model.validation_evidence["boundary"])
    mask = (X.index.get_level_values("datetime") < boundary) & (metadata["label_end_at"] < boundary)
    assert len(actual_fits) == 8
    for wrapper, training, labels, clocks in actual_fits[:4]:
        assert training.index.equals(X.loc[mask].index)
        assert labels.index.equals(training.index)
        assert clocks["context"] == metadata["context"]
        assert wrapper._provenance["fitted_at"] == metadata["context"].as_of
        assert wrapper._provenance["vintage_policy"] == "captured-vintage-at-fit-1"
        assert wrapper.model.__class__.__module__.split(".")[0] in {"sklearn", "lightgbm", "catboost", "xgboost"}
        np.testing.assert_allclose(wrapper._imputer.statistics_, X.loc[mask].median().to_numpy())
        np.testing.assert_allclose(wrapper._scaler.mean_, X.loc[mask].fillna(X.loc[mask].median()).mean())
    assert all(training.index.equals(X.index) for _, training, _, _ in actual_fits[4:])
    assert all(clocks["context"] == metadata["context"] for _, _, _, clocks in actual_fits)
    assert all(kwargs == {} for _, kwargs in predictions)  # Internal, explicitly non-contextual selection.
    assert all((index.get_level_values("datetime") >= boundary).all() for index, _ in predictions)
    evidence = model.validation_evidence
    assert evidence["train_rows"] == mask.sum()
    assert evidence["validation_rows"] == (X.index.get_level_values("datetime") >= boundary).sum()
    assert evidence["scope"] == "retrospective-within-fit-weight-selection-1"
    assert evidence["prediction_policy"] == "non-contextual-internal-validation-1"
    assert evidence["purge_policy"] == "outcome-end-before-validation-1"
    assert evidence["fit_cutoff"] == evidence["weight_fit_cutoff"] == metadata["context"].as_of.isoformat()
    assert set(model.get_weights()) == set(names) and sum(model.get_weights().values()) == pytest.approx(1)
    pd.testing.assert_frame_equal(X, input_before)
    model.save(tmp_path / "ensemble")
    loaded = ensemble(mock_config).load(tmp_path / "ensemble")
    assert loaded.validation_evidence == evidence
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_internal_selection_does_not_authorize_temporal_inference(fit_data, mock_config):
    X, y, metadata = fit_data
    model = ensemble(mock_config).train(X, y, **metadata)
    args = dict(context=metadata["context"], feature_available_at=metadata["feature_available_at"],
                availability_policy="captured-vintage-at-fit-1")
    with pytest.raises(ModelError, match="fit must precede inference"):
        model.predict(X, **args)
    # An explicit later decision is supplied by the caller, never invented by the ensemble.
    decision = metadata["context"].model_copy(update={"as_of": metadata["context"].as_of + pd.Timedelta(minutes=1)})
    prediction = model.predict(X, **{**args, "context": decision})
    assert prediction.index.equals(X.index) and np.isfinite(prediction).all()
    with pytest.raises(ModelError, match="follow model fit"):
        model.predict(X, context=decision, feature_available_at=metadata["feature_available_at"])


@pytest.mark.parametrize("field", ["feature_available_at", "label_available_at", "label_end_at"])
@pytest.mark.parametrize("problem", ["future", "nat", "misaligned", "naive", "missing"])
@pytest.mark.parametrize("position", [0, -1], ids=["training", "validation"])
def test_all_clock_rows_rejected_before_any_weight_fit(
    fit_data, mock_config, monkeypatch, field, problem, position
):
    X, y, metadata = fit_data
    value = metadata[field].copy()
    if problem == "future":
        value.iloc[position] = metadata["context"].as_of + pd.Timedelta(microseconds=1)
    elif problem == "nat":
        value.iloc[position] = pd.NaT
    elif problem == "misaligned":
        value = value.iloc[::-1]
    elif problem == "naive":
        value = value.dt.tz_localize(None)
    elif problem == "missing":
        value = None
    metadata[field] = value
    forbid_fit(monkeypatch)
    with pytest.raises(ModelError):
        ensemble(mock_config).train(X, y, **metadata)


@pytest.mark.parametrize("problem", ["future", "nat", "naive"])
@pytest.mark.parametrize("position", [0, -1], ids=["training", "validation"])
def test_all_observation_rows_rejected_before_any_weight_fit(
    fit_data, mock_config, monkeypatch, problem, position
):
    X, y, metadata = fit_data
    dates = X.index.get_level_values("datetime").to_list()
    dates[position] = metadata["context"].as_of + pd.Timedelta(microseconds=1) if problem == "future" else pd.NaT
    if problem == "naive":
        dates = X.index.get_level_values("datetime").tz_localize(None)
    index = pd.MultiIndex.from_arrays([dates, X.index.get_level_values("instrument")], names=X.index.names)
    X.index = y.index = index
    for field in ("feature_available_at", "label_available_at", "label_end_at"):
        metadata[field].index = index
    forbid_fit(monkeypatch)
    with pytest.raises(ModelError):
        ensemble(mock_config).train(X, y, **metadata)


@pytest.mark.parametrize("problem", ["before_observation", "after_publication"])
def test_true_outcome_end_must_be_consistent_for_validation_too(fit_data, mock_config, monkeypatch, problem):
    X, y, metadata = fit_data
    metadata["label_end_at"].iloc[-1] = (
        X.index[-1][0] - pd.Timedelta(microseconds=1) if problem == "before_observation"
        else metadata["label_available_at"].iloc[-1] + pd.Timedelta(microseconds=1)
    )
    forbid_fit(monkeypatch)
    with pytest.raises(ModelError, match="inconsistent outcome"):
        ensemble(mock_config).train(X, y, **metadata)


@pytest.mark.parametrize("problem", ["nan_label", "infinite_label", "reordered_labels", "duplicate_identity", "missing_hash"])
def test_complete_identity_and_label_validation_precedes_weight_fit(fit_data, mock_config, monkeypatch, problem):
    X, y, metadata = fit_data
    if problem == "nan_label":
        y.iloc[-1] = np.nan
    elif problem == "infinite_label":
        y.iloc[-1] = np.inf
    elif problem == "reordered_labels":
        y = y.iloc[::-1]
    elif problem == "duplicate_identity":
        index = pd.MultiIndex.from_tuples([X.index[1], *X.index[1:]], names=X.index.names)
        X.index = y.index = index
    elif problem == "missing_hash":
        metadata.pop("feature_spec_hash")
    forbid_fit(monkeypatch)
    with pytest.raises(ModelError):
        ensemble(mock_config).train(X, y, **metadata)


def test_same_backfill_fails_strict_policy_and_actual_earlier_cutoff(fit_data, mock_config, monkeypatch):
    X, y, metadata = fit_data
    forbid_fit(monkeypatch)
    with pytest.raises(ModelError, match="historical decision"):
        ensemble(mock_config).train(X, y, **{k: v for k, v in metadata.items() if k != "availability_policy"})
    earlier = metadata["context"].model_copy(update={"as_of": metadata["context"].as_of - pd.Timedelta(days=1)})
    with pytest.raises(ModelError, match="future observations"):
        ensemble(mock_config).train(X, y, **{**metadata, "context": earlier})
    with pytest.raises(ModelError, match="actual temporal context"):
        ensemble(mock_config).train(X, y, **{**metadata, "context": None})


def test_strict_default_retains_publication_purge_and_original_child_clock(fit_data, actual_fits, mock_config):
    X, y, metadata = fit_data
    metadata.pop("availability_policy")
    metadata["feature_available_at"] = pd.Series(X.index.get_level_values("datetime"), index=X.index)
    metadata["label_available_at"] = metadata["label_end_at"] + pd.Timedelta(days=2)
    model = ensemble(mock_config).train(X, y, **metadata)
    boundary = pd.Timestamp(model.validation_evidence["boundary"])
    expected = (X.index.get_level_values("datetime") < boundary) & (metadata["label_available_at"] < boundary)
    assert actual_fits[0][1].index.equals(X.loc[expected].index)
    assert model.validation_evidence["train_rows"] < (
        (X.index.get_level_values("datetime") < boundary) & (metadata["label_end_at"] < boundary)
    ).sum()
    assert actual_fits[0][3]["context"].as_of == boundary - pd.Timedelta(microseconds=1)
    assert actual_fits[1][3]["context"] == metadata["context"]
    assert model.validation_evidence["purge_policy"] == "label-availability-before-validation-1"
    assert model.validation_evidence["availability_policy"] == "decision-time-vintage-1"


@pytest.mark.parametrize("field", ["feature_available_at", "label_available_at", "label_end_at"])
def test_strict_validation_future_clock_is_rejected_before_any_fit(fit_data, mock_config, monkeypatch, field):
    X, y, metadata = fit_data
    metadata.pop("availability_policy")
    metadata["feature_available_at"] = pd.Series(X.index.get_level_values("datetime"), index=X.index)
    metadata["label_available_at"] = metadata["label_end_at"] + pd.Timedelta(days=2)
    metadata[field].iloc[-1] = metadata["context"].as_of + pd.Timedelta(microseconds=1)
    forbid_fit(monkeypatch)
    with pytest.raises(ModelError):
        ensemble(mock_config).train(X, y, **metadata)


def test_outcome_end_equal_to_economic_boundary_is_purged(fit_data, actual_fits, mock_config):
    X, y, metadata = fit_data
    dates = X.index.get_level_values("datetime")
    boundary = dates.unique().sort_values()[56]
    just_before = dates == dates.unique().sort_values()[55]
    metadata["label_end_at"].loc[just_before] = boundary
    key = X.index[just_before][0]
    metadata["label_end_at"].loc[key] = boundary - pd.Timedelta(microseconds=1)
    ensemble(mock_config).train(X, y, **metadata)
    trained_index = actual_fits[0][1].index
    assert key in trained_index
    assert all(row not in trained_index for row in X.index[just_before][1:])


def test_permutation_preserves_identity_weights_and_predictions(fit_data, mock_config):
    X, y, metadata = fit_data
    first = ensemble(mock_config).train(X, y, **metadata)
    order = np.random.default_rng(7).permutation(len(X))
    shuffled = {key: value.iloc[order] if isinstance(value, pd.Series) else value for key, value in metadata.items()}
    second = ensemble(mock_config).train(X.iloc[order], y.iloc[order], **shuffled)
    assert first.validation_evidence == second.validation_evidence
    assert first.get_weights() == second.get_weights()
    np.testing.assert_allclose(first.predict(X), second.predict(X), rtol=1e-12, atol=1e-12)


def test_validation_perturbation_cannot_fit_submodel_transforms(fit_data, actual_fits, mock_config):
    X, y, metadata = fit_data
    first = ensemble(mock_config).train(X, y, **metadata)
    first_subset = actual_fits[0][0]
    changed = X.copy()
    val = X.index.get_level_values("datetime") >= pd.Timestamp(first.validation_evidence["boundary"])
    changed.loc[val] *= 1e6
    ensemble(mock_config).train(changed, y, **metadata)
    second_subset = actual_fits[2][0]
    np.testing.assert_array_equal(first_subset._imputer.statistics_, second_subset._imputer.statistics_)
    np.testing.assert_array_equal(first_subset._scaler.mean_, second_subset._scaler.mean_)
    np.testing.assert_array_equal(first_subset.model.coef_, second_subset.model.coef_)
    assert not np.allclose(first._models["Linear"]._scaler.mean_, actual_fits[-1][0]._scaler.mean_)


def test_nonpositive_validation_ic_never_falls_back_to_equal(fit_data, actual_fits, mock_config):
    X, y, metadata = fit_data
    dates = X.index.get_level_values("datetime")
    y.loc[dates >= dates.unique().sort_values()[56]] *= -1
    model = ensemble(mock_config)
    with pytest.raises(ModelError, match="no model has positive"):
        model.train(X, y, **metadata)
    assert len(actual_fits) == 1  # No full-data refit after failed selection.
    assert model.get_weights() == {}
    assert model.validation_evidence["ic"]["Linear"] < 0
    with pytest.raises(ModelError, match="has not been trained"):
        model.predict(X)


def test_undefined_validation_ic_never_falls_back_to_equal(fit_data, actual_fits, mock_config):
    X, y, metadata = fit_data
    dates = X.index.get_level_values("datetime")
    y.loc[dates >= dates.unique().sort_values()[56]] = 0
    model = ensemble(mock_config)
    with pytest.raises(ModelError, match="undefined validation IC"):
        model.train(X, y, **metadata)
    assert len(actual_fits) == 1 and model.get_weights() == {}


def test_failed_reselection_cannot_execute_or_save_previous_success(fit_data, actual_fits, mock_config, tmp_path):
    X, y, metadata = fit_data
    model = ensemble(mock_config).train(X, y, **metadata)
    assert model.get_weights() == {"Linear": 1}
    dates = X.index.get_level_values("datetime")
    y.loc[dates >= dates.unique().sort_values()[56]] *= -1
    with pytest.raises(ModelError, match="no model has positive"):
        model.train(X, y, **metadata)
    assert len(actual_fits) == 3 and model.get_weights() == {}
    with pytest.raises(ModelError, match="has not been trained"):
        model.predict(X)
    with pytest.raises(ModelError, match="untrained EnsembleModel"):
        model.save(tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_later_fit_cutoff_does_not_change_eligible_data_or_weight_selection(fit_data, actual_fits, mock_config):
    X, y, metadata = fit_data
    first = ensemble(mock_config).train(X, y, **metadata)
    later = metadata["context"].model_copy(update={"as_of": metadata["context"].as_of + pd.Timedelta(days=1)})
    second = ensemble(mock_config).train(X, y, **{**metadata, "context": later})
    assert first.get_weights() == second.get_weights()
    assert first.validation_evidence["ic"] == second.validation_evidence["ic"]
    assert actual_fits[0][1].index.equals(actual_fits[2][1].index)
    assert actual_fits[0][3]["context"] == metadata["context"]
    assert actual_fits[2][3]["context"] == later
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
