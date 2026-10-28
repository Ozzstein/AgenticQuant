"""Closed model file identities and bounded ensemble containers.

Hash/type checks establish identity and consistency, not reconstructed fitting,
source certification or approval. Joblib is only for application-owned bytes
whose external hashes have been authenticated by the caller's authority.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
import math
import os
import re
import stat
import struct
import tempfile
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.evidence.store import canonical_bytes, digest
from src.data.features.identity import frame_hash
from src.data.features.metrics import prediction_metrics
from src.domain.market import (
    EnsembleArtifact,
    EnsembleWeightEvidence,
    ModelArtifact,
    ModelFileProof,
)
from src.utils.exceptions import ModelError

MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_MEMBERS = 18
_NUMERIC = {f"{kind}{bits}" for kind in ("int", "uint") for bits in (8, 16, 32, 64)} | {"float32", "float64"}
_CLOCKS = {f"datetime64[{unit}, UTC]" for unit in ("ns", "us", "ms", "s")}
_NS = re.compile(r"(?:0|-?[1-9][0-9]*)\Z", re.ASCII)


def _read(path, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ModelError("regular model file required; no path alias")
    with path.open("rb") as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise ModelError("model resource byte limit exceeded")
    return data


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if canonical_bytes(value) != data:
            raise ValueError("noncanonical JSON")
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ModelError(f"invalid canonical model JSON: {exc}") from exc


def _archive(members):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name, data in members:
            if len(data) > (MAX_JSON_BYTES if name == "manifest.json" else MAX_MEMBER_BYTES):
                raise ModelError("model member byte limit exceeded")
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            entry.compress_type = zipfile.ZIP_STORED
            archive.writestr(entry, data)
    result = output.getvalue()
    if len(result) > MAX_PAYLOAD_BYTES or sum(len(data) for _, data in members) > MAX_PAYLOAD_BYTES:
        raise ModelError("model container byte limit exceeded")
    return result


def _unarchive(payload):
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ModelError("model container byte limit exceeded")
    try:
        # Bound central-directory work BEFORE ZipFile allocates member objects.
        # Canonical archives have a fixed 22-byte, comment-free, single-disk EOCD.
        if len(payload) < 22:
            raise ModelError("missing model container directory")
        signature, disk, directory_disk, count_disk, count, size, offset, comment = struct.unpack("<4s4H2LH", payload[-22:])
        if (signature != b"PK\x05\x06" or disk or directory_disk or comment or count_disk != count
                or not 1 <= count <= MAX_MEMBERS or size > MAX_MEMBERS * 110
                or offset + size != len(payload) - 22 or not payload.startswith(b"PK\x03\x04")):
            raise ModelError("invalid or excessive model directory members/bytes")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(entries) != count or not 1 <= len(entries) <= MAX_MEMBERS or len(set(names)) != len(names):
                raise ModelError("duplicate or excessive model members")
            total = 0
            members = []
            for entry in entries:
                if (entry.filename != "manifest.json" and not re.fullmatch(r"(?:full|selection)/[0-3]\.joblib(?:\.json)?|selection/validation\.json", entry.filename)):
                    raise ModelError("unknown model member or path alias")
                maximum = MAX_JSON_BYTES if entry.filename == "manifest.json" else MAX_MEMBER_BYTES
                total += entry.file_size
                if (entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits != 0
                        or entry.file_size != entry.compress_size or entry.file_size > maximum
                        or total > MAX_PAYLOAD_BYTES or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ModelError("invalid model member/resource budget")
                with archive.open(entry) as handle:
                    data = handle.read(maximum + 1)
                if len(data) != entry.file_size or len(data) > maximum:
                    raise ModelError("model member size mismatch")
                members.append((entry.filename, data))
        # Reject headers, aliases, comments, timestamps, prefix/trailing garbage
        # and alternative encodings too. No archive extraction is performed.
        if _archive(members) != payload:
            raise ModelError("noncanonical model container bytes")
        return members
    except ModelError:
        raise
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ModelError(f"invalid model container: {exc}") from exc


def _series_document(series, clock=False):
    dtype = str(series.dtype)
    if dtype not in (_CLOCKS if clock else _NUMERIC) or series.name is not None and not isinstance(series.name, str):
        raise ModelError("canonical numeric/UTC clock series dtype and name required")
    values = [str(pd.Timestamp(v).value) for v in series] if clock else series.to_list()
    return {"dtype": dtype, "name": series.name, "values": values}


def validation_proof(labels, predictions, **clocks):
    """Losslessly retain actual numerical selection outputs and original clocks."""
    index = labels.index
    if index.names != ["datetime", "instrument"] or str(index.get_level_values(0).tz) != "UTC":
        raise ModelError("canonical UTC datetime/instrument proof required")
    return canonical_bytes({
        "index": [[str(pd.Timestamp(at).value), key] for at, key in index],
        "labels": _series_document(labels),
        "predictions": [[name, _series_document(value)] for name, value in predictions.items()],
        **{name: _series_document(value, clock=True) for name, value in clocks.items()},
    })


def validate_canonical_inputs(features, labels, **clocks):
    """Check lossless canonical representations before any selection fitting."""
    try:
        if str(features.index.get_level_values(0).tz) != "UTC" or any(str(dtype) not in _NUMERIC for dtype in features.dtypes):
            raise ModelError("canonical UTC observations and numeric feature dtypes required")
        for value in features.index.get_level_values(0):
            if pd.Timestamp(value).value % 1000:
                raise ModelError("canonical datetime contracts cannot round sub-microsecond observations")
        _series_document(labels)
        for value in clocks.values():
            _series_document(value, True)
            if any(pd.Timestamp(t).value % 1000 for t in value):
                raise ModelError("canonical datetime contracts cannot round sub-microsecond clocks")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ModelError(f"canonical input representation is invalid: {exc}") from exc


def _restore_series(document, index, clock=False):
    if not isinstance(document, dict) or set(document) != {"dtype", "name", "values"}:
        raise ModelError("closed validation series required")
    dtype, name, values = document["dtype"], document["name"], document["values"]
    if dtype not in (_CLOCKS if clock else _NUMERIC) or name is not None and not isinstance(name, str) or not isinstance(values, list) or len(values) != len(index):
        raise ModelError("validation series identity/dtype differs")
    if clock:
        if any(not isinstance(v, str) or not _NS.fullmatch(v) for v in values):
            raise ModelError("canonical UTC-ns clock text required")
        values = pd.to_datetime([int(v) for v in values], unit="ns", utc=True)
    elif any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ModelError("finite actual validation values required")
    result = pd.Series(values, index=index, dtype=dtype, name=name)
    if _series_document(result, clock) != document:
        raise ModelError("lossy validation series conversion")
    return result


def _verify_validation(data, artifact):
    evidence = artifact.weight_evidence
    if digest(data) != evidence.validation_proof_hash:
        raise ModelError("validation proof hash differs")
    document = _json(data)
    if set(document) != {"index", "labels", "predictions", "feature_available_at", "label_available_at", "label_end_at"}:
        raise ModelError("closed validation proof required")
    rows = document["index"]
    if not isinstance(rows, list) or len(rows) != evidence.validation_rows:
        raise ModelError("validation row count differs")
    if any(not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str) or not _NS.fullmatch(row[0]) or not isinstance(row[1], str) or not row[1] for row in rows):
        raise ModelError("exact validation row identities required")
    dates = pd.to_datetime([int(row[0]) for row in rows], unit="ns", utc=True)
    index = pd.MultiIndex.from_arrays([dates, [row[1] for row in rows]], names=["datetime", "instrument"])
    if not index.is_unique or (dates < evidence.boundary).any() or (dates > artifact.fitted_at).any():
        raise ModelError("invalid validation observation identities/clocks")
    labels = _restore_series(document["labels"], index)
    clocks = {name: _restore_series(document[name], index, True) for name in ("feature_available_at", "label_available_at", "label_end_at")}
    if any(frame_hash(value) != getattr(evidence.validation_frames, name) for name, value in {"labels": labels, **clocks}.items()):
        raise ModelError("validation input frame hash differs")
    features, available, ends = (clocks[name] for name in ("feature_available_at", "label_available_at", "label_end_at"))
    if (any(value.isna().any() or (value > artifact.fitted_at).any() for value in clocks.values())
            or (ends.to_numpy() < dates.to_numpy()).any() or (ends > available).any()
            or artifact.vintage_policy == "decision-time-vintage-1" and (features.to_numpy() > dates.to_numpy()).any()):
        raise ModelError("invalid validation end/availability clocks")
    predictions = document["predictions"]
    if not isinstance(predictions, list) or any(not isinstance(v, list) or len(v) != 2 for v in predictions) or [name for name, _ in predictions] != [c.model_name for c in artifact.children]:
        raise ModelError("exact ordered validation predictions required")
    for (name, values), (_, expected_hash), (_, expected_ic) in zip(predictions, evidence.prediction_hashes, evidence.ic):
        prediction = _restore_series(values, index)
        ic = prediction_metrics(prediction, labels)["ic"]
        if frame_hash(prediction) != expected_hash or not math.isfinite(ic) or ic != expected_ic:
            raise ModelError(f"actual archived prediction/IC differs for {name}")


def _child_metadata(artifact):
    return {"schema_version": "2.0", "payload_hash": artifact.payload_hash,
            "temporal_provenance": True, "artifact": artifact.model_dump(mode="json")}


def _check_environment(artifact):
    for package, version in artifact.environment:
        if importlib.metadata.version(package) != version:
            raise ModelError(f"model environment differs for {package}")


def _checked_container(payload, artifact):
    members = _unarchive(payload)
    if not members or members[0][0] != "manifest.json":
        raise ModelError("manifest must be first")
    manifest = _json(members[0][1])
    if digest(members[0][1]) != artifact.manifest_hash or manifest != artifact.model_dump(mode="json", exclude={"payload_hash", "manifest_hash"}):
        raise ModelError("parent manifest differs")
    if tuple((name, digest(data)) for name, data in members[1:]) != artifact.member_hashes:
        raise ModelError("member inventory/hash differs")
    mapping = dict(members)
    for prefix, artifacts in (("full", artifact.children), ("selection", artifact.selection_children)):
        for i, child in enumerate(artifacts):
            model, sidecar = mapping[f"{prefix}/{i}.joblib"], mapping[f"{prefix}/{i}.joblib.json"]
            if digest(model) != child.payload_hash or sidecar != canonical_bytes(_child_metadata(child)):
                raise ModelError("child payload/sidecar differs from typed artifact")
            _check_environment(child)
    if artifact.selection_children:
        _verify_validation(mapping["selection/validation.json"], artifact)
    return mapping


def checked_pair(path):
    """Validate every file/member/typed component without deserializing a child."""
    try:
        sidecar = _read(str(path) + ".json", MAX_JSON_BYTES)
        metadata = _json(sidecar)
        if not isinstance(metadata, dict):
            raise ModelError("closed model sidecar required")
        payload = _read(path, MAX_PAYLOAD_BYTES)
        if metadata.get("schema_version") == "3.0" and set(metadata) == {"schema_version", "artifact"}:
            artifact = EnsembleArtifact.model_validate(metadata["artifact"])
            if artifact.payload_hash != digest(payload):
                raise ModelError("parent payload hash differs")
            _checked_container(payload, artifact)
        elif metadata.get("schema_version") == "2.0" and set(metadata) == {"schema_version", "payload_hash", "temporal_provenance", "artifact"}:
            artifact = ModelArtifact.model_validate(metadata["artifact"])
            from src.core.model_zoo import REGISTRY
            if artifact.model_name not in REGISTRY or not metadata["temporal_provenance"] or metadata["payload_hash"] != artifact.payload_hash or digest(payload) != artifact.payload_hash:
                raise ModelError("individual temporal artifact differs")
            _check_environment(artifact)
        else:
            raise ModelError("canonical versioned model pair required; legacy diagnostics are separate")
        return payload, sidecar, artifact
    except ModelError:
        raise
    except (ValueError, TypeError, KeyError, OSError, ImportError, OverflowError) as exc:
        raise ModelError(f"invalid model pair: {exc}") from exc


def _load_child(payload, artifact):
    """Called only after ALL outer/component checks have completed."""
    from src.core.model_zoo import REGISTRY, ModelWrapper
    data = joblib.load(io.BytesIO(payload))
    if not isinstance(data, dict) or set(data) != {"schema_version", "model_name", "model", "feature_names", "imputer", "scaler", "provenance", "params"} or data["schema_version"] != "2.0" or data["model_name"] != artifact.model_name:
        raise ModelError("loaded model structure/family differs")
    module, class_name = REGISTRY[artifact.model_name].rsplit(".", 1)
    if type(data["model"]) is not getattr(importlib.import_module(module), class_name):
        raise ModelError("loaded estimator class differs")
    if canonical_bytes(data["params"]).decode() != artifact.parameters_json or tuple(data["feature_names"]) != artifact.feature_names:
        raise ModelError("loaded parameters/features differ")
    actual_params = data["model"].get_params(deep=False)
    if any(key not in actual_params or actual_params[key] != value for key, value in data["params"].items()):
        raise ModelError("actual estimator kwargs differ")
    provenance = data["provenance"]
    expected_keys = {"feature_spec_hash", "training_data_hash", "fitted_at", "training_start", "training_end", "labels_available_through", "features_available_through", "label_end_through", "label_end_policy", "vintage_policy", "code_version"}
    if not isinstance(provenance, dict) or set(provenance) != expected_keys or any(value != getattr(artifact, key) for key, value in provenance.items()):
        raise ModelError("loaded training clocks/provenance differ")
    if type(data["imputer"]) is not SimpleImputer or data["imputer"].strategy != "median" or type(data["scaler"]) is not StandardScaler or not data["scaler"].with_mean or not data["scaler"].with_std:
        raise ModelError("loaded preprocessing differs")
    if len(data["imputer"].statistics_) != len(artifact.feature_names) or len(data["scaler"].mean_) != len(artifact.feature_names):
        raise ModelError("fitted preprocessing feature count differs")
    model = ModelWrapper(artifact.model_name, _exact_parameters=data["params"])
    model.model, model._feature_names = data["model"], tuple(data["feature_names"])
    model._imputer, model._scaler = data["imputer"], data["scaler"]
    model._provenance, model.artifact = provenance, artifact
    return model


def _state_fingerprint(model):
    """Detect in-memory mutations before republishing the original loaded bytes.

    Joblib's memoization/array object graph need not be a byte fixed point after
    loading. This is a same-instance mutation guard, not a reconstructed fit.
    """
    state = {key:value for key,value in vars(model).items()
             if key not in {"_canonical_pair", "_canonical_fingerprint"}}
    buffer = io.BytesIO()
    joblib.dump(state, buffer)
    return digest(buffer.getvalue())


def load_canonical_model(path, *, expected, composition=None):
    if not isinstance(expected, ModelFileProof):
        raise ModelError("typed external file identity required")
    payload, sidecar, artifact = checked_pair(path)
    actual = ModelFileProof(model_name=artifact.model_name, payload_hash=digest(payload), sidecar_hash=digest(sidecar),
                            artifact_hash=artifact.content_hash(), ensemble_composition_hash=artifact.composition.content_hash() if isinstance(artifact, EnsembleArtifact) else None)
    if actual != expected:
        raise ModelError("external expected model file identity differs")
    try:
        if not isinstance(artifact, EnsembleArtifact):
            if composition is not None:
                raise ModelError("individual model cannot accept ensemble composition")
            return _load_child(payload, artifact)
        if composition != artifact.composition:
            raise ModelError("exact expected ensemble composition required")
        members = _checked_container(payload, artifact)
        from src.core.ensemble import EnsembleModel
        model = EnsembleModel(composition=composition)
        # All members, artifacts, clocks and weights have already been checked.
        model._models = {a.model_name: _load_child(members[f"full/{i}.joblib"], a) for i, a in enumerate(artifact.children)}
        model._selection_models = {a.model_name: _load_child(members[f"selection/{i}.joblib"], a) for i, a in enumerate(artifact.selection_children)}
        model._weights = dict(artifact.resolved_weights)
        model._training_frames = artifact.training_frames
        evidence = artifact.weight_evidence
        model._selection_frames = (evidence.train_frames, evidence.validation_frames) if artifact.selection_children else None
        model._validation_proof = members.get("selection/validation.json")
        model._train_metadata = {"context": None}  # Source receipt is not a fabricated new RunContext.
        model.artifact, model._trained = artifact, True
        if artifact.selection_children:
            model.validation_evidence = dict(ic=dict(evidence.ic), boundary=str(pd.Timestamp(evidence.boundary)),
                train_rows=evidence.train_rows, validation_rows=evidence.validation_rows, scope=evidence.scope,
                prediction_policy=evidence.prediction_policy, availability_policy=evidence.availability_policy,
                fit_cutoff=evidence.fit_cutoff.isoformat(), weight_fit_cutoff=evidence.selection_fit_cutoff.isoformat(),
                purge_policy=composition.purge_policy)
        model._canonical_pair = (payload, sidecar)
        model._canonical_fingerprint = _state_fingerprint(model)
        return model
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(f"application-owned model deserialization failed: {exc}") from exc


def _publish_pair(path, payload, sidecar):
    paths = ((Path(path), payload), (Path(str(path) + ".json"), sidecar))
    for target, data in paths:
        if target.exists() and _read(target, MAX_PAYLOAD_BYTES) != data or target.is_symlink():
            raise ModelError("immutable model path collision; use a new artifact identity")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for target, data in paths:
            fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".ensemble-")
            staged.append(temporary)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for (target, data), temporary in zip(paths, staged):
            try:
                os.link(temporary, target)
            except FileExistsError:
                if _read(target, MAX_PAYLOAD_BYTES) != data:
                    raise ModelError("immutable concurrent model collision") from None
    finally:
        for temporary in staged:
            Path(temporary).unlink(missing_ok=True)


def save_ensemble(model, path):
    """Serialize real full/selection children into one immutable canonical pair."""
    if not model._trained or model._composition is None or model._training_frames is None:
        raise ModelError("canonical fitted ensemble required")
    if getattr(model, "_canonical_pair", None) is not None:
        if model._canonical_fingerprint != _state_fingerprint(model):
            raise ModelError("canonical loaded/saved model state changed; retrain under an explicit recipe")
        _publish_pair(path, *model._canonical_pair)
        return
    members, full, selected = [], [], []
    with tempfile.TemporaryDirectory(prefix="quantagent-ensemble-save-") as temporary:
        for prefix, wrappers, artifacts in (("full", model._models, full), ("selection", model._selection_models, selected)):
            for i, name in enumerate(model._model_names):
                if name not in wrappers:
                    if prefix == "selection" and model._weighting != "ic_weighted":
                        continue
                    raise ModelError("all actual configured children required")
                child = wrappers[name]
                child_path = Path(temporary) / f"{prefix}-{i}.joblib"
                child.save(str(child_path))
                if child.artifact is None:
                    raise ModelError("actual temporal child artifact required")
                artifacts.append(child.artifact)
                for suffix in ("", ".json"):
                    members.append((f"{prefix}/{i}.joblib{suffix}", _read(str(child_path) + suffix, MAX_MEMBER_BYTES)))
    if selected:
        members.append(("selection/validation.json", model._validation_proof))
    source = full[0]
    if selected:
        evidence = model.validation_evidence
        proof = _json(model._validation_proof)
        index = pd.MultiIndex.from_tuples([(pd.Timestamp(int(at), unit="ns", tz="UTC"), key) for at, key in proof["index"]], names=["datetime", "instrument"])
        hashes = tuple((name, frame_hash(_restore_series(values, index))) for name, values in proof["predictions"])
        weight_evidence = EnsembleWeightEvidence(scope="retrospective-within-fit-weight-selection-1",
            availability_policy=source.vintage_policy, fit_cutoff=source.fitted_at,
            prediction_policy="non-contextual-internal-validation-1", boundary=pd.Timestamp(evidence["boundary"]),
            selection_fit_cutoff=pd.Timestamp(evidence["weight_fit_cutoff"]),
            train_frames=model._selection_frames[0], validation_frames=model._selection_frames[1],
            train_rows=evidence["train_rows"], validation_rows=evidence["validation_rows"],
            ic=tuple((name, evidence["ic"][name]) for name in model._model_names), prediction_hashes=hashes,
            validation_proof_hash=digest(model._validation_proof))
    else:
        weight_evidence = EnsembleWeightEvidence(scope="fixed-composition-weights-1", availability_policy=source.vintage_policy, fit_cutoff=source.fitted_at)
    environment = dict(item for artifact in full for item in artifact.environment)
    common = source.model_dump(exclude={"model_name", "payload_hash", "environment", "parameters_json", "preprocessing"})
    artifact = EnsembleArtifact(**common, payload_hash="0" * 64, manifest_hash="0" * 64,
        composition=model._composition, training_frames=model._training_frames,
        training_frame_hash=model._training_frames.aggregate_hash(), children=tuple(full), selection_children=tuple(selected),
        environment=tuple(environment.items()), parameters_json=canonical_bytes(model._composition.model_dump(mode="json")).decode(),
        resolved_weights=tuple((name, model._weights[name]) for name in model._model_names), weight_evidence=weight_evidence,
        member_hashes=tuple((name, digest(data)) for name, data in members))
    manifest = canonical_bytes(artifact.model_dump(mode="json", exclude={"payload_hash", "manifest_hash"}))
    payload = _archive([("manifest.json", manifest), *members])
    artifact = EnsembleArtifact.model_validate({**artifact.model_dump(), "payload_hash": digest(payload), "manifest_hash": digest(manifest)})
    _checked_container(payload, artifact)
    sidecar = canonical_bytes({"schema_version": "3.0", "artifact": artifact.model_dump(mode="json")})
    if len(sidecar) > MAX_JSON_BYTES:
        raise ModelError("parent sidecar byte limit exceeded")
    _publish_pair(path, payload, sidecar)
    model.artifact = artifact
    model._canonical_pair = (payload, sidecar)
    model._canonical_fingerprint = _state_fingerprint(model)
