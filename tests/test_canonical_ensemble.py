"""Actual local estimators and synthetic inputs; identity is not approval proof."""
from __future__ import annotations

import io
import json
import stat
import sys
import types
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.model_zoo import build_model, load_model, model_file_proof, resolved_child_spec
from src.data import model_artifacts as codec
from src.data.evidence.store import canonical_bytes, digest
from src.data.features.identity import frame_hash, training_frame_identity
from src.domain.common import RunContext
from src.domain.market import (
    EnsembleArtifact,
    EnsembleChildSpec,
    EnsembleComposition,
    ModelFileProof,
)
from src.utils.exceptions import ModelError

PARAMS = {
    "Linear": {"alpha": 2.},
    "LightGBM": {"n_estimators": 15, "n_jobs": 1, "verbosity": -1},
    "CatBoost": {"iterations": 15, "thread_count": 1, "verbose": 0, "allow_writing_files": False},
    "XGBoost": {"n_estimators": 15, "n_jobs": 1},
}


def composition(mode="ic_weighted", policy="captured-vintage-at-fit-1", names=None, weights=()):
    return EnsembleComposition(children=tuple(EnsembleChildSpec(model_name=name, parameters=tuple(sorted(PARAMS[name].items()))) for name in names or PARAMS),
        weighting=mode, weighting_algorithm={"ic_weighted":"positive-date-spearman-normalization-1", "equal":"equal-unit-weights-1", "custom":"exact-normalized-custom-1"}[mode],
        validation_fraction=.3 if mode == "ic_weighted" else None,
        selection_policy="retrospective-within-fit-weight-selection-1" if mode == "ic_weighted" else "fixed-composition-weights-1",
        purge_policy=("outcome-end-before-validation-1" if policy == "captured-vintage-at-fit-1" else "label-availability-before-validation-1") if mode == "ic_weighted" else None,
        custom_weights=weights)


def inputs(policy="captured-vintage-at-fit-1"):
    dates = pd.bdate_range("2020-01-02", periods=80, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["a", "b", "c", "d"]], names=["datetime", "instrument"])
    rng = np.random.default_rng(76)
    X = pd.DataFrame(rng.normal(size=(len(index), 2)), columns=["x", "z"], index=index)
    y = (X.x * 4 - X.z).rename("label")
    X.iloc[4, 1] = np.nan
    fit = pd.Timestamp("2026-10-28T18:00:00Z")
    ctx = RunContext(run_id="canonical-ensemble-test", mode="research", as_of=fit,
                     code_version="test", config_hash="0" * 64)
    ends = pd.Series(index.get_level_values(0) + pd.Timedelta(days=2), index=index)
    features = pd.Series(fit - pd.Timedelta(minutes=2), index=index)
    labels = pd.Series(fit - pd.Timedelta(minutes=1), index=index)
    if policy == "decision-time-vintage-1":
        features = pd.Series(index.get_level_values(0), index=index)
        labels = ends + pd.Timedelta(days=1)
    return X, y, dict(context=ctx, feature_available_at=features, label_available_at=labels,
                     label_end_at=ends, feature_spec_hash="1" * 64, data_snapshot_hash="2" * 64,
                     availability_policy=policy)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    X, y, metadata = inputs()
    spec = composition()
    model = build_model("Ensemble", ensemble_composition=spec).train(X, y, **metadata)
    path = tmp_path_factory.mktemp("canonical-ensemble") / "parent.joblib"
    model.save(path)
    return model, path, spec, X, y, metadata


def copy_pair(path, destination):
    destination.write_bytes(path.read_bytes())
    Path(str(destination) + ".json").write_bytes(Path(str(path) + ".json").read_bytes())
    return destination


def test_actual_four_family_parent_and_selection_identity(fitted, tmp_path):
    model, path, spec, X, y, metadata = fitted
    artifact = model.artifact
    assert isinstance(artifact, EnsembleArtifact) and model.model_name == "Ensemble"
    assert len(artifact.children) == len(artifact.selection_children) == 4
    frames = training_frame_identity(X, y, metadata["feature_available_at"], metadata["label_available_at"], metadata["label_end_at"])
    assert artifact.training_frames == frames and artifact.training_frame_hash == frames.aggregate_hash()
    assert artifact.weight_evidence.fit_cutoff == artifact.fitted_at == metadata["context"].as_of
    assert artifact.weight_evidence.selection_fit_cutoff == artifact.fitted_at
    proof = model_file_proof(path)
    loaded = load_model(path, expected=proof, ensemble_composition=spec)
    assert loaded.artifact == artifact and loaded.get_weights() == model.get_weights()
    assert loaded.validation_evidence == model.validation_evidence
    for name in PARAMS:
        for child in (loaded._models[name], loaded._selection_models[name]):
            assert child._params == PARAMS[name]
            assert json.loads(child.artifact.parameters_json) == PARAMS[name]
            assert all(child.model.get_params()[key] == value for key, value in PARAMS[name].items())
            assert child.artifact.training_data_hash == "2" * 64 and child.artifact.feature_spec_hash == "1" * 64
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
    second = tmp_path / "same.joblib"
    loaded.save(second)
    assert second.read_bytes() == path.read_bytes()
    assert Path(str(second) + ".json").read_bytes() == Path(str(path) + ".json").read_bytes()
    assert model_file_proof(second) == proof
    with pytest.raises(ModelError, match="fit must precede inference"):
        loaded.predict(X, context=metadata["context"], feature_available_at=metadata["feature_available_at"], availability_policy=metadata["availability_policy"])


def test_pure_frame_identity_preserves_existing_public_hash_and_h_aggregate():
    from src.core.walk_forward import frame_hash as original
    X, y, metadata = inputs()
    values = dict(features=X, labels=y, **{k:metadata[k] for k in ("feature_available_at", "label_available_at", "label_end_at")})
    assert all(frame_hash(value) == original(value) for value in values.values())
    identity = training_frame_identity(**values)
    assert identity.aggregate_hash() == digest(canonical_bytes({key: original(value) for key, value in values.items()}))


@pytest.mark.parametrize("mode", ["equal", "custom"])
def test_fixed_modes_preserve_exact_weights_and_have_no_fitted_ic(mode, tmp_path):
    X, y, metadata = inputs()
    weights = (("Linear", .4), ("LightGBM", .6000000000001)) if mode == "custom" else ()
    spec = composition(mode, names=["Linear", "LightGBM"], weights=weights)
    model = build_model("Ensemble", ensemble_composition=spec).train(X, y, **metadata)
    path = tmp_path / "fixed.joblib"
    model.save(path)
    loaded = load_model(path, expected=model_file_proof(path), ensemble_composition=spec)
    expected = weights if weights else (("Linear", .5), ("LightGBM", .5))
    assert loaded.artifact.resolved_weights == expected
    assert loaded.artifact.selection_children == () and loaded._selection_models == {}
    assert loaded.artifact.weight_evidence.scope == "fixed-composition-weights-1"
    assert loaded.artifact.weight_evidence.validation_rows == 0
    with zipfile.ZipFile(path) as archive:
        assert not any(name.startswith("selection/") for name in archive.namelist())


def test_strict_parent_uses_actual_true_end_and_original_publication_purge(tmp_path):
    X, y, metadata = inputs("decision-time-vintage-1")
    spec = composition(policy=metadata["availability_policy"], names=["Linear", "LightGBM"])
    model = build_model("Ensemble", ensemble_composition=spec).train(X, y, **metadata)
    model.save(tmp_path / "strict.joblib")
    artifact = model.artifact
    boundary = pd.Timestamp(artifact.weight_evidence.boundary)
    mask = (X.index.get_level_values(0) < boundary) & (metadata["label_available_at"] < boundary)
    assert artifact.weight_evidence.train_rows == mask.sum()
    assert artifact.selection_children[0].label_end_policy == "explicit-outcome-end-1"
    assert artifact.selection_children[0].fitted_at == boundary - pd.Timedelta(microseconds=1)
    assert artifact.label_end_through == metadata["label_end_at"].max()


def test_canonical_constructor_and_loader_ignore_all_ambient_overrides(monkeypatch, tmp_path):
    import src.core.ensemble as ensemble_module
    import src.core.model_zoo as zoo

    def forbidden(*args, **kwargs):
        pytest.fail("canonical constructor consumed ambient model configuration")

    monkeypatch.setattr(zoo, "get_config", forbidden)
    monkeypatch.setattr(zoo.ModelWrapper, "_load_params", forbidden)
    monkeypatch.setattr(ensemble_module, "get_config", forbidden)
    monkeypatch.setenv("AIQUANT_QLIB__MODEL_PARAMS", '{"alpha":999,"n_estimators":999}')
    monkeypatch.setenv("AIQUANT_MODELS__ENSEMBLE_MODELS", '["Linear"]')
    X, y, metadata = inputs()
    spec = composition("equal", names=["Linear", "LightGBM"])
    model = build_model("Ensemble", ensemble_composition=spec).train(X, y, **metadata)
    path = tmp_path / "explicit.joblib"
    model.save(path)
    loaded = load_model(path, expected=model_file_proof(path), ensemble_composition=spec)
    assert all(child._params == PARAMS[name] for name, child in loaded._models.items())
    assert all(child.model.get_params()[key] == value for name, child in loaded._models.items() for key, value in PARAMS[name].items())
    single = build_model("Linear", parameters=(("alpha", 7),)).train(X, y)
    assert single.model.alpha == 7
    assert resolved_child_spec("Linear", (("alpha", 3),)).parameters == (("alpha", 3),)


@pytest.mark.parametrize("problem", ["no_composition", "parent_parameters", "individual_composition", "unknown_family"])
def test_factory_rejects_ambiguous_admission(problem):
    with pytest.raises((ModelError, ValueError)):
        if problem == "no_composition":
            build_model("Ensemble")
        elif problem == "parent_parameters":
            build_model("Ensemble", parameters=(("alpha", 3),), ensemble_composition=composition())
        elif problem == "individual_composition":
            build_model("Linear", ensemble_composition=composition())
        else:
            build_model("Other")


@pytest.mark.parametrize("problem", ["label_end_at", "context", "wrong_purge"])
def test_canonical_training_needs_real_ends_context_and_matching_policy(problem):
    X, y, metadata = inputs()
    spec = composition()
    if problem == "wrong_purge":
        spec = composition(policy="decision-time-vintage-1")
    else:
        metadata.pop(problem)
    with pytest.raises(ModelError):
        build_model("Ensemble", ensemble_composition=spec).train(X, y, **metadata)


def repack_pair(path, destination, *, edit=None, raw=None):
    """Recompute outer identities so rejection must check the inner contract."""
    parent = json.loads(Path(str(path) + ".json").read_bytes())["artifact"]
    with zipfile.ZipFile(path) as archive:
        members = [(name, archive.read(name)) for name in archive.namelist()[1:]]
    if edit is not None:
        edit(parent, members)
    parent["member_hashes"] = [(name, digest(data)) for name, data in members]
    manifest = canonical_bytes({k:v for k,v in parent.items() if k not in ("payload_hash", "manifest_hash")})
    payload = codec._archive([("manifest.json", manifest), *members])
    if raw is not None:
        payload = raw(payload)
    parent["payload_hash"], parent["manifest_hash"] = digest(payload), digest(manifest)
    sidecar = canonical_bytes({"schema_version":"3.0", "artifact":parent})
    destination.write_bytes(payload)
    Path(str(destination) + ".json").write_bytes(sidecar)
    return ModelFileProof(model_name="Ensemble", payload_hash=digest(payload), sidecar_hash=digest(sidecar),
                          artifact_hash=digest(canonical_bytes(parent)),
                          ensemble_composition_hash=digest(canonical_bytes(parent["composition"])))


def no_deserialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid member/metadata reached child deserialization")

    monkeypatch.setattr(codec.joblib, "load", forbidden)


@pytest.mark.parametrize("problem", [
    "raw_last_child", "last_child_sidecar", "last_selection_sidecar", "full_clock", "missing_end_clock",
    "selection_clock", "selection_parameters", "composition_parameters", "composition_order",
    "weights", "reordered_weights", "environment", "training_hash", "source_hash", "prediction_value",
    "validation_future_clock", "validation_future_observation", "validation_label", "extra_member",
])
def test_all_components_checked_before_any_child_load(fitted, tmp_path, monkeypatch, problem):
    _, original, spec, _, _, _ = fitted

    def edit(parent, members):
        def change_member(name, transform):
            i = next(i for i, (key, _) in enumerate(members) if key == name)
            members[i] = (name, transform(members[i][1]))

        if problem == "raw_last_child":
            change_member("full/3.joblib", lambda data: data + b"wrong")
        elif problem in ("last_child_sidecar", "last_selection_sidecar"):
            change_member("full/3.joblib.json" if problem == "last_child_sidecar" else "selection/3.joblib.json", lambda data: data + b" ")
        elif problem == "full_clock":
            parent["children"][-1]["fitted_at"] = "2026-10-28T18:00:01Z"
        elif problem == "missing_end_clock":
            parent["label_end_through"] = None
        elif problem == "selection_clock":
            parent["selection_children"][-1]["training_end"] = parent["weight_evidence"]["boundary"]
        elif problem == "selection_parameters":
            parent["selection_children"][-1]["parameters_json"] = '{}'
        elif problem == "composition_parameters":
            parent["composition"]["children"][-1]["parameters"] = [["n_estimators", 999]]
        elif problem == "composition_order":
            parent["composition"]["children"].reverse()
        elif problem == "weights":
            parent["resolved_weights"][0][1] += 1e-13  # No tolerance-based renormalizing load.
        elif problem == "reordered_weights":
            parent["resolved_weights"].reverse()
        elif problem == "environment":
            parent["children"][-1]["environment"][-1][1] = "different"
        elif problem == "training_hash":
            parent["training_frame_hash"] = "e" * 64
        elif problem == "source_hash":
            parent["children"][-1]["training_data_hash"] = "e" * 64
        elif problem == "extra_member":
            members.append(("full/extra.joblib", b"wrong"))
        else:
            def corrupt(data):
                proof = json.loads(data)
                if problem == "prediction_value":
                    proof["predictions"][-1][1]["values"][0] += 1
                elif problem == "validation_future_clock":
                    proof["feature_available_at"]["values"][0] = str(pd.Timestamp("2026-10-29T00:00:00Z").value)
                elif problem == "validation_future_observation":
                    proof["index"][0][0] = str(pd.Timestamp("2026-10-29T00:00:00Z").value)
                elif problem == "validation_label":
                    proof["labels"]["values"][0] += 1
                result = canonical_bytes(proof)
                parent["weight_evidence"]["validation_proof_hash"] = digest(result)
                return result
            change_member("selection/validation.json", corrupt)

    path = tmp_path / "tampered.joblib"
    expected = repack_pair(original, path, edit=edit)
    no_deserialization(monkeypatch)
    with pytest.raises(ModelError):
        load_model(path, expected=expected, ensemble_composition=spec)


def alternative_zip(payload, problem):
    with zipfile.ZipFile(io.BytesIO(payload)) as original:
        members = [(name, original.read(name)) for name in original.namelist()]
    if problem == "prefix":
        return b"garbage" + payload
    if problem == "trailing":
        return payload + b"garbage"
    if problem == "concatenated":
        return payload + payload
    if problem == "duplicate":
        members.append(members[-1])
    elif problem == "unknown":
        members.append(("unknown", b"x"))
    elif problem == "reorder":
        members[1], members[2] = members[2], members[1]
    elif problem in ("traversal", "absolute", "alias", "backslash"):
        name = {"traversal":"../full/0.joblib", "absolute":"/full/0.joblib", "alias":"full/./0.joblib", "backslash":"full\\0.joblib"}[problem]
        members[1] = (name, members[1][1])
    elif problem == "missing":
        members.pop()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        if problem == "comment":
            archive.comment = b"not canonical"
        for i, (name, data) in enumerate(members):
            info = zipfile.ZipInfo(name, date_time=(1981 if problem == "timestamp" else 1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777 if problem == "symlink" and i == 1 else stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED if problem == "compression" else zipfile.ZIP_STORED
            if problem == "extra":
                info.extra = b'\x01\x00\x00\x00'
            archive.writestr(info, data)
    return output.getvalue()


@pytest.mark.parametrize("problem", ["prefix", "trailing", "concatenated", "duplicate", "unknown", "reorder", "missing", "traversal", "absolute", "alias", "backslash", "comment", "timestamp", "symlink", "compression", "extra"])
def test_container_alias_headers_and_expansion_fail_before_load(fitted, tmp_path, monkeypatch, problem):
    _, original, spec, _, _, _ = fitted
    path = tmp_path / "malformed.joblib"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:.*", category=UserWarning)
        expected = repack_pair(original, path, raw=lambda data: alternative_zip(data, problem))
    no_deserialization(monkeypatch)
    with pytest.raises(ModelError):
        load_model(path, expected=expected, ensemble_composition=spec)


@pytest.mark.parametrize("limit", ["MAX_MEMBER_BYTES", "MAX_PAYLOAD_BYTES", "MAX_JSON_BYTES", "MAX_MEMBERS"])
def test_actual_file_boundary_resource_caps(fitted, monkeypatch, limit):
    _, path, spec, _, _, _ = fitted
    expected = model_file_proof(path)
    monkeypatch.setattr(codec, limit, 10)
    no_deserialization(monkeypatch)
    with pytest.raises(ModelError, match="limit|budget|members"):
        load_model(path, expected=expected, ensemble_composition=spec)


def test_external_pair_and_recipe_expected_identity_before_load(fitted, monkeypatch):
    _, path, spec, _, _, _ = fitted
    expected = model_file_proof(path)
    no_deserialization(monkeypatch)
    for key in ("payload_hash", "sidecar_hash", "artifact_hash", "ensemble_composition_hash"):
        with pytest.raises(ModelError, match="expected model file"):
            load_model(path, expected=expected.model_copy(update={key:"e" * 64}), ensemble_composition=spec)
    with pytest.raises(ModelError, match="expected ensemble composition"):
        load_model(path, expected=expected, ensemble_composition=composition("equal"))


@pytest.mark.parametrize("change", ["weight", "estimator", "selection_estimator", "artifact"])
def test_republication_never_hides_loaded_state_mutation(fitted, tmp_path, change):
    _, path, spec, _, _, _ = fitted
    loaded = load_model(path, expected=model_file_proof(path), ensemble_composition=spec)
    if change == "weight":
        loaded._weights["Linear"] += .1
    elif change == "estimator":
        loaded._models["Linear"].model.coef_[0] += .1
    elif change == "selection_estimator":
        loaded._selection_models["Linear"].model.coef_[0] += .1
    elif change == "artifact":
        loaded.artifact = loaded.artifact.model_copy(update={"training_data_hash":"e" * 64})
    with pytest.raises(ModelError, match="state changed"):
        loaded.save(tmp_path / "mutated.joblib")
    assert not (tmp_path / "mutated.joblib").exists()


def test_immutable_pair_retries_collisions_and_symlinks(fitted, tmp_path):
    model, path, _, _, _, _ = fitted
    target = tmp_path / "same.joblib"
    model.save(target)
    before = (target.read_bytes(), Path(str(target) + ".json").read_bytes())
    model.save(target)
    assert before == (target.read_bytes(), Path(str(target) + ".json").read_bytes())
    target.write_bytes(b"other")
    with pytest.raises(ModelError, match="collision"):
        model.save(target)
    assert target.read_bytes() == b"other"
    target.unlink()
    target.symlink_to(path)
    with pytest.raises(ModelError, match="alias|collision"):
        model.save(target)


def test_independent_four_family_refit_preserves_native_payload_identity(fitted, tmp_path):
    _, path, spec, X, y, metadata = fitted
    independent = build_model("Ensemble", ensemble_composition=spec).train(X, y, **metadata)
    new_path = tmp_path / "fresh.joblib"
    independent.save(new_path)
    np.testing.assert_array_equal(independent.predict(X), fitted[0].predict(X))
    assert independent.get_weights() == fitted[0].get_weights()
    with zipfile.ZipFile(path) as first, zipfile.ZipFile(new_path) as second:
        for prefix in ("full", "selection"):
            for i, name in enumerate(PARAMS):
                left, right = first.read(f"{prefix}/{i}.joblib"), second.read(f"{prefix}/{i}.joblib")
                if name != "CatBoost":
                    assert left == right
                else:
                    # Actual native GUID/time is retained, not normalized away
                    # to pretend independent full-refit byte reconstruction.
                    a = codec.joblib.load(io.BytesIO(left))["model"]
                    b = codec.joblib.load(io.BytesIO(right))["model"]
                    assert a.get_metadata()["model_guid"] != b.get_metadata()["model_guid"]
                    assert left != right
    assert path.read_bytes() != new_path.read_bytes()
    assert model_file_proof(path).artifact_hash != model_file_proof(new_path).artifact_hash


def test_independent_fresh_three_family_fit_has_identical_container(tmp_path):
    X, y, metadata = inputs()
    spec = composition(names=["Linear", "LightGBM", "XGBoost"])
    paths = [tmp_path / f"fresh-{i}.joblib" for i in range(2)]
    for path in paths:
        build_model("Ensemble", ensemble_composition=spec).train(X, y, **metadata).save(path)
    for suffix in ("", ".json"):
        assert Path(str(paths[0]) + suffix).read_bytes() == Path(str(paths[1]) + suffix).read_bytes()


def historical_reference():
    """Frozen test-only serializer/artifact; portable without private Git objects."""
    name = "tests._before_stage1_reference"
    reference = Path(__file__).parent / "fixtures" / "model_v2_reference.py.txt"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(reference.read_bytes(), str(reference), "exec"), module.__dict__)
    return module


def test_existing_single_family_serializer_and_artifact_bytes_unchanged(tmp_path):
    from src.core.model_zoo import ModelWrapper
    legacy = historical_reference()
    X, y, metadata = inputs()
    model = ModelWrapper("Linear").train(X, y, **metadata)
    original = types.SimpleNamespace(**model.__dict__)
    legacy.save(original, str(tmp_path / "before.joblib"))
    model.save(str(tmp_path / "after.joblib"))
    for suffix in ("", ".json"):
        assert Path(str(tmp_path / "before.joblib") + suffix).read_bytes() == Path(str(tmp_path / "after.joblib") + suffix).read_bytes()
    assert model.artifact.model_dump(mode="json") == original.artifact.model_dump(mode="json")
    assert model.artifact.content_hash() == original.artifact.content_hash()
    proof = model_file_proof(tmp_path / "after.joblib")
    assert "ensemble_composition_hash" not in proof.model_dump(mode="json")
    loaded = load_model(tmp_path / "after.joblib", expected=proof)
    pd.testing.assert_series_equal(loaded.predict(X), model.predict(X))


def test_legacy_directory_stays_diagnostic_and_is_never_canonical(tmp_path):
    from src.core.ensemble import EnsembleModel
    X, y, _ = inputs()
    model = EnsembleModel(model_names=["Linear"], weighting="equal").train(X, y)
    path = tmp_path / "legacy"
    model.save(path)
    loaded = EnsembleModel(model_names=["Linear"]).load_legacy_directory(path)
    assert loaded.artifact is None and loaded._composition is None
    pd.testing.assert_series_equal(loaded.predict(X), model.predict(X))
    with pytest.raises(ModelError, match="regular model file"):
        model_file_proof(path)


def test_no_consumer_admission_widening():
    from src.operations.contracts import ModelRecipe
    from src.research.evaluator import ExperimentPlan
    from src.strategies.specification import StrategySpec, baseline_specs
    assert "Ensemble" not in str(ModelRecipe.model_fields["model_name"].annotation)
    value = baseline_specs()["momentum_topk"].model_dump(mode="json")
    value["model_name"] = "Ensemble"
    with pytest.raises(ValueError):
        StrategySpec.model_validate(value)
    assert "ensemble_composition" not in ExperimentPlan.model_fields


@pytest.mark.parametrize("problem", ["one_child", "duplicate", "nested", "bad_algorithm", "no_fraction", "nan_fraction", "fixed_purge", "extra_custom", "wrong_custom_keys", "negative_custom", "unnormalized_custom", "duplicate_params", "unsorted_params", "nonfinite_params"])
def test_closed_composition_rejects_ambiguous_recipes(problem):
    data = composition().model_dump(mode="json")
    if problem == "one_child":
        data["children"] = data["children"][:1]
    elif problem == "duplicate":
        data["children"][1] = data["children"][0]
    elif problem == "nested":
        data["children"][0]["model_name"] = "Ensemble"
    elif problem == "bad_algorithm":
        data["weighting_algorithm"] = "equal-unit-weights-1"
    elif problem in ("no_fraction", "nan_fraction"):
        data["validation_fraction"] = None if problem == "no_fraction" else float("nan")
    elif problem == "fixed_purge":
        data = composition("equal").model_dump(mode="json")
        data["purge_policy"] = "outcome-end-before-validation-1"
    elif problem == "extra_custom":
        data["custom_weights"] = [["Linear", 1]]
    elif problem in ("wrong_custom_keys", "negative_custom", "unnormalized_custom"):
        weights = (("Linear", .4), ("LightGBM", .6))
        data = composition("custom", names=["Linear", "LightGBM"], weights=weights).model_dump(mode="json")
        data["custom_weights"] = {"wrong_custom_keys":[["Linear", 1]], "negative_custom":[["Linear", -.1],["LightGBM", 1.1]], "unnormalized_custom":[["Linear", 4],["LightGBM", 6]]}[problem]
    elif problem == "duplicate_params":
        data["children"][0]["parameters"] = [["alpha", 1],["alpha", 2]]
    elif problem == "unsorted_params":
        data["children"][0]["parameters"] = [["z", 1],["alpha", 2]]
    elif problem == "nonfinite_params":
        data["children"][0]["parameters"] = [["alpha", float("inf")]]
    with pytest.raises(ValueError):
        EnsembleComposition.model_validate(data)


@pytest.mark.parametrize("problem", ["estimator_kwargs", "provenance", "features", "preprocessing"])
def test_loaded_actual_child_must_match_declared_artifact(fitted, tmp_path, monkeypatch, problem):
    from src.domain.market import ModelArtifact
    _, original, spec, _, _, _ = fitted

    def edit(parent, members):
        entry = next(i for i, (name, _) in enumerate(members) if name == "full/0.joblib")
        data = codec.joblib.load(io.BytesIO(members[entry][1]))
        if problem == "estimator_kwargs":
            data["model"].set_params(alpha=999)
        elif problem == "provenance":
            data["provenance"]["fitted_at"] += pd.Timedelta(seconds=1)
        elif problem == "features":
            data["feature_names"] = tuple(reversed(data["feature_names"]))
        else:
            data["imputer"].strategy = "constant"
        buffer = io.BytesIO()
        codec.joblib.dump(data, buffer)
        raw = buffer.getvalue()
        members[entry] = ("full/0.joblib", raw)
        parent["children"][0]["payload_hash"] = digest(raw)
        sidecar_entry = next(i for i, (name, _) in enumerate(members) if name == "full/0.joblib.json")
        members[sidecar_entry] = ("full/0.joblib.json", canonical_bytes(codec._child_metadata(ModelArtifact.model_validate(parent["children"][0]))))

    path = tmp_path / "contradictory.joblib"
    expected = repack_pair(original, path, edit=edit)
    calls = []
    original_load = codec.joblib.load

    def observed(*args, **kwargs):
        calls.append(1)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(codec.joblib, "load", observed)
    with pytest.raises(ModelError, match="loaded|actual estimator"):
        load_model(path, expected=expected, ensemble_composition=spec)
    assert len(calls) == 1


def test_huge_central_directory_count_fails_before_zip_parser(fitted, tmp_path, monkeypatch):
    _, original, spec, _, _, _ = fitted

    def raw(payload):
        # Canonical EOCD directory count is checked before ZipFile allocation.
        changed = bytearray(payload)
        changed[-14:-12] = (65535).to_bytes(2, "little")
        changed[-12:-10] = (65535).to_bytes(2, "little")
        return bytes(changed)

    path = tmp_path / "directory.joblib"
    expected = repack_pair(original, path, raw=raw)
    monkeypatch.setattr(codec.zipfile, "ZipFile", lambda *a, **k: pytest.fail("unbounded directory reached ZIP parser"))
    no_deserialization(monkeypatch)
    with pytest.raises(ModelError, match="directory"):
        load_model(path, expected=expected, ensemble_composition=spec)


@pytest.mark.parametrize("problem", ["object_labels", "object_features", "non_utc_clocks", "submicro_end", "submicro_observation", "invalid_hash"])
def test_canonical_representation_errors_fail_before_any_fit(monkeypatch, problem):
    from src.core.model_zoo import ModelWrapper
    X, y, metadata = inputs()
    if problem == "object_labels":
        y = y.astype(object)
    elif problem == "object_features":
        X = X.astype(object)
    elif problem == "non_utc_clocks":
        metadata["feature_available_at"] = metadata["feature_available_at"].dt.tz_convert("Europe/Rome")
    elif problem == "submicro_end":
        metadata["label_end_at"].iloc[-1] += pd.Timedelta(nanoseconds=1)
    elif problem == "invalid_hash":
        metadata["feature_spec_hash"] = "invalid"
    else:
        index = pd.MultiIndex.from_tuples([(at + pd.Timedelta(nanoseconds=1), key) for at, key in X.index], names=X.index.names)
        X.index = y.index = index
        for key in ("feature_available_at", "label_available_at", "label_end_at"):
            metadata[key].index = index
    monkeypatch.setattr(ModelWrapper, "train", lambda *a, **k: pytest.fail("invalid canonical representation reached fit"))
    with pytest.raises(ModelError, match="canonical"):
        build_model("Ensemble", ensemble_composition=composition()).train(X, y, **metadata)
