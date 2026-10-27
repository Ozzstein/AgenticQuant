"""Real append-only registry authority for interrupted native execution."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from src.research.native import NativeSupervisor
from src.research.native_recovery import (
    NativeContainerBinding,
    NativeFile,
    NativeLaunch,
    NativeMount,
    NativeRecoveryEvent,
)
from src.research.registry import RegistryError, ResearchRegistry, canonical
from tests.test_native_discovery_exposure import mounted_request


@pytest.fixture
def authority(tmp_path):
    request, snapshot = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    exposure = supervisor.admit_discovery(request, registry)
    frozen = supervisor.root / ".inputs" / request.run_id
    work = supervisor.root / request.run_id
    work.mkdir()
    originals = {
        "request.json": request.model_dump(mode="json"),
        "scenario.json": {**request.scenario.model_dump(mode="json"), "provider_uri": "/data/qlib"},
        "admission.json": {
            "discovery_exposure": exposure.model_dump(mode="json"),
            "discovery_exposure_hash": exposure.content_hash(),
            "image_id": "sha256:fixture",
            "network_id": "network-fixture",
        },
    }
    files = []
    for name, value in originals.items():
        raw = canonical(value)
        (work / name).write_bytes(raw)
        files.append(
            NativeFile(tree="work", path=name, blob_hash=registry.archive(raw), size=len(raw))
        )
    for path in sorted(frozen.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            files.append(
                NativeFile(
                    tree="frozen",
                    path=path.relative_to(frozen).as_posix(),
                    blob_hash=registry.archive(raw),
                    size=len(raw),
                )
            )
    token = tmp_path / "bearer"
    args = supervisor.command(request, work, token, frozen_root=frozen)
    mounts = []
    for i, arg in enumerate(args):
        if arg == "--mount":
            values = dict(
                part.split("=", 1) if "=" in part else (part, True)
                for part in args[i + 1].split(",")
            )
            mounts.append(
                NativeMount(
                    source=values["src"],
                    destination=values["dst"],
                    read_only=bool(values.get("readonly")),
                )
            )
    launch = NativeLaunch(
        run_id=request.run_id,
        request_hash=request.content_hash(),
        discovery_exposure_hash=exposure.content_hash(),
        account_id=snapshot.context.account_id,
        recorded_at=exposure.admitted_at + timedelta(seconds=1),
        registry_path=str(registry.path),
        supervisor_root=str(supervisor.root),
        gateway_token_path=str(token),
        image_id="sha256:fixture",
        network_id="network-fixture",
        user="1000:1000",
        files=tuple(files),
        mounts=tuple(mounts),
    )
    return registry, launch


def bound(registry, launch):
    registry.record_native_launch(launch.model_dump(mode="json"))
    created = launch.recorded_at + timedelta(seconds=1)
    identity = {
        "container_id": "a" * 64,
        "created_at": created.isoformat(),
        "name": "/quantagent-rd-" + launch.run_id,
        "launch_hash": launch.content_hash(),
        "image_id": launch.image_id,
        "network_id": launch.network_id,
        "mounts": [m.model_dump(mode="json") for m in launch.mounts],
    }
    value = NativeContainerBinding(
        run_id=launch.run_id,
        request_hash=launch.request_hash,
        discovery_exposure_hash=launch.discovery_exposure_hash,
        launch_hash=launch.content_hash(),
        container_id="a" * 64,
        created_at=created,
        bound_at=created + timedelta(seconds=1),
        identity_hash=registry.archive(canonical(identity)),
    )
    return value


def test_launch_and_original_container_are_immutable_and_readable_after_restart(authority):
    registry, launch = authority
    assert registry.native_launch(launch.run_id) is None
    container = bound(registry, launch)
    key = registry.record_native_container(container.model_dump(mode="json"))
    assert key == container.content_hash()
    restarted = ResearchRegistry(registry.path)
    assert restarted.native_launch(launch.run_id) == launch.model_dump(mode="json")
    assert restarted.native_container(launch.run_id) == container.model_dump(mode="json")
    assert restarted.record_native_container(container.model_dump(mode="json")) == key
    with pytest.raises(RegistryError, match="immutable"):
        restarted.record_native_launch(
            launch.model_copy(
                update={"recorded_at": launch.recorded_at + timedelta(seconds=1)}
            ).model_dump(mode="json")
        )


@pytest.mark.parametrize("substitution", ["size", "missing", "bytes", "mount", "registry"])
def test_launch_cannot_substitute_frozen_source_or_control_authority(authority, substitution):
    registry, launch = authority
    payload = launch.model_dump(mode="json")
    if substitution == "size":
        payload["files"][0]["size"] += 1
    elif substitution == "missing":
        payload["files"].pop()
    elif substitution == "bytes":
        target = next(
            f
            for f in payload["files"]
            if f["tree"] == "frozen" and f["path"] != "qlib/manifest.json"
        )
        target.update(blob_hash=registry.archive(b"foreign"), size=7)
    elif substitution == "mount":
        payload["mounts"][0]["source"] = str(registry.path)
    else:
        payload["registry_path"] = "relative.db"
    with pytest.raises((ValueError, KeyError)):
        registry.record_native_launch(payload)
    assert registry.native_launch(launch.run_id) is None


def test_container_projection_must_bind_exact_original_identity(authority):
    registry, launch = authority
    container = bound(registry, launch)
    projection = json.loads(registry.blob(container.identity_hash))
    projection["container_id"] = "b" * 64
    changed = container.model_copy(
        update={"identity_hash": registry.archive(canonical(projection))}
    )
    with pytest.raises(RegistryError, match="projection differs"):
        registry.record_native_container(changed.model_dump(mode="json"))
    assert registry.native_container(launch.run_id) is None


def test_recovery_append_is_cas_and_cannot_forge_completion_or_release_exposure(authority):
    registry, launch = authority
    container = bound(registry, launch)
    registry.record_native_container(container.model_dump(mode="json"))
    first = NativeRecoveryEvent(
        run_id=launch.run_id,
        request_hash=launch.request_hash,
        discovery_exposure_hash=launch.discovery_exposure_hash,
        sequence=1,
        observed_at=container.bound_at,
        launch_hash=launch.content_hash(),
        container_binding_hash=container.content_hash(),
        observation="absent",
        reason="auto_removed_exit_unknown",
        cost_state="uncertain",
    )
    first_hash = registry.append_native_recovery(first.model_dump(mode="json"))
    assert registry.append_native_recovery(first.model_dump(mode="json")) == first_hash

    def append(index):
        value = first.model_copy(
            update={
                "sequence": 2,
                "previous_hash": first_hash,
                "observed_at": first.observed_at + timedelta(seconds=index + 1),
            }
        )
        try:
            registry.append_native_recovery(value.model_dump(mode="json"))
            return "appended"
        except RegistryError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(append, (0, 1))) == ["appended", "blocked"]
    history = ResearchRegistry(registry.path).native_recoveries(launch.run_id)
    assert len(history) == 2 and all(row["exposure_retained"] for row in history)
    assert registry.status()["native_discovery"]["admissions"] == 1
    last = NativeRecoveryEvent.model_validate(history[-1])
    forged = last.model_copy(
        update={
            "sequence": 3,
            "previous_hash": last.content_hash(),
            "result_hash": registry.archive(canonical({"status": "discovery_complete"})),
        }
    )
    with pytest.raises(RegistryError, match="no authoritative completion"):
        registry.append_native_recovery(forged.model_dump(mode="json"))


def test_historical_unbound_observation_survives_later_container_binding(authority):
    registry, launch = authority
    container = bound(registry, launch)
    first = NativeRecoveryEvent(
        run_id=launch.run_id,
        request_hash=launch.request_hash,
        discovery_exposure_hash=launch.discovery_exposure_hash,
        sequence=1,
        observed_at=launch.recorded_at,
        launch_hash=launch.content_hash(),
        observation="ambiguous",
        reason="container_unbound",
    )
    registry.append_native_recovery(first.model_dump(mode="json"))
    registry.record_native_container(container.model_dump(mode="json"))
    assert registry.native_recoveries(launch.run_id) == (first.model_dump(mode="json"),)
