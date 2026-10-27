import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.core.ensemble import EnsembleModel
from src.core.model_zoo import ModelWrapper
from src.data.__main__ import main
from src.data.evidence.store import EvidenceStore
from src.data.features.dataset import feature_dataset
from src.domain.market import FeatureSpec
from tests.data_quant_fixtures import archived_snapshot


@pytest.mark.parametrize("crypto", [False, True])
def test_evidence_to_real_model_to_inference_cli(tmp_path, capsys, crypto):
    snapshot = archived_snapshot(crypto=crypto)
    store = EvidenceStore(tmp_path / "evidence.db")
    key = store.save_snapshot(snapshot.model_dump(mode="json"))
    specs = (
        FeatureSpec(name="return_5", expression="$close/Ref($close,5)-1"),
        FeatureSpec(name="vol_5", expression="Std($close/Ref($close,1)-1,5)"),
    )
    features = tmp_path / "features.json"
    features.write_text(json.dumps([s.model_dump(mode="json") for s in specs]))
    dates = sorted({bar.closed_at for bar in snapshot.bars})
    train_context = snapshot.context.model_copy(update={"as_of": dates[100] + timedelta(seconds=1)})
    train = tmp_path / "train.json"
    train.write_text(train_context.model_dump_json())
    test = tmp_path / "test.json"
    test.write_text(snapshot.context.model_dump_json())
    model_path = tmp_path / "model.joblib"
    common = ["--store", str(store.path)]
    assert (
        main(
            common
            + [
                "train",
                "--snapshot",
                key,
                "--features",
                str(features),
                "--context",
                str(train),
                "--model",
                "Linear",
                "--output",
                str(model_path),
            ]
        )
        == 0
    )
    training_result = json.loads(capsys.readouterr().out)
    assert training_result["training_rows"] == 300  # day 100 label cannot have realized yet
    assert (
        main(
            common
            + [
                "predict",
                "--snapshot",
                key,
                "--features",
                str(features),
                "--context",
                str(test),
                "--model",
                "Linear",
                "--artifact",
                str(model_path),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["predictions"]) == 3
    assert {p["instrument_id"] for p in result["predictions"]} == {
        bar.instrument.instrument_id for bar in snapshot.bars
    }


def test_ensemble_purges_labels_and_preserves_transformations(tmp_path, mock_config):
    snapshot = archived_snapshot()
    specs = (FeatureSpec(name="momentum", expression="$close/Ref($close,3)-1"),)
    dataset = feature_dataset(snapshot, specs, horizon=5).eligible_training(snapshot.context.as_of)
    # Use an explicitly nontrivial predictive target only inside this unit test.
    y = dataset.features.momentum.fillna(0) * 2
    model = EnsembleModel(model_names=["Linear"], val_fraction=0.3, config=mock_config)
    model.train(
        dataset.features,
        y,
        context=snapshot.context,
        feature_available_at=dataset.feature_available_at,
        label_available_at=dataset.label_available_at,
        feature_spec_hash=dataset.feature_spec_hash,
        data_snapshot_hash=dataset.snapshot_hash,
    )
    boundary = pd.Timestamp(model.validation_evidence["boundary"])
    expected = (
        (dataset.features.index.get_level_values("datetime") < boundary)
        & (dataset.label_available_at < boundary)
    ).sum()
    assert model.validation_evidence["train_rows"] == expected
    assert model.get_weights() == {"Linear": 1}
    before = model.predict(dataset.features)
    path = tmp_path / "ensemble"
    model.save(path)
    loaded = EnsembleModel(model_names=["Linear"], config=mock_config).load(path)
    np.testing.assert_array_equal(before, loaded.predict(dataset.features))
    assert loaded._models["Linear"]._imputer is not None


def test_alpha158_actual_qlib_local_bundle(tmp_path):
    pytest.importorskip("qlib")
    pytest.importorskip("exchange_calendars")
    from src.data.qlib_us import USQlibScenario, alpha158_dataset, export_bundle

    snapshot = archived_snapshot(use_exchange_calendar=True)
    root = tmp_path / "us_bundle"
    manifest = export_bundle(snapshot, root)
    dates = sorted({b.session for b in snapshot.bars})
    names = tuple(manifest["instrument_map"].values())
    scenario = USQlibScenario(
        provider_uri=str(root),
        bundle_hash=manifest["bundle_hash"],
        universe=names,
        benchmark=names[0],
        price_basis="raw",
        train=(dates[0], dates[99]),
        valid=(dates[102], dates[125]),
        test=(dates[128], dates[-1]),
    )
    dataset = alpha158_dataset(scenario)
    train = dataset.prepare("train", col_set="feature", data_key="learn")
    test = dataset.prepare("test", col_set="feature", data_key="infer")
    assert train.shape == (300, 158)
    assert test.shape == (96, 158)
    assert set(test.index.get_level_values("instrument")) == set(names)
    assert np.isfinite(train.to_numpy()).all()
    # Real Qlib transforms/features can train the actual local numeric wrapper.
    labels = dataset.prepare("train", col_set="label", data_key="learn").iloc[:, 0]
    model = ModelWrapper("Linear").train(train, labels)
    assert len(model.predict(test)) == 96
