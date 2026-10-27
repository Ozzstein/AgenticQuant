"""Cross-run host paths are reserved before any source bytes are staged."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from src.research.gateway_deployment import GatewayDeploymentManager
from src.research.native import NativeSupervisor
from src.research.native_mounts import NativeMountAuthority
from src.research.registry import ResearchRegistry
from tests.test_native_discovery_exposure import mounted_request


def test_nested_supervisor_cannot_stage_into_another_writable_job(tmp_path):
    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    first = NativeSupervisor(tmp_path / "jobs")
    first.admit_discovery(req, registry)
    work = first.root / req.run_id
    work.mkdir()
    second = NativeSupervisor(work / "nested-supervisor")
    document = tmp_path / "reserved-observations.txt"
    document.write_text("Source bytes must never be visible to the other job")
    nested = req.model_copy(
        update={"run_id": "nested", "command": "general_model", "document": str(document)}
    )
    with pytest.raises(ValueError, match="another reserved"):
        second.admit_discovery(nested, registry)
    assert not (second.root / ".inputs").exists()
    assert registry.status()["native_discovery"]["admissions"] == 1
    with pytest.raises(ValueError, match="another reserved"):
        NativeSupervisor(tmp_path).admit_discovery(
            req.model_copy(update={"run_id": "jobs"}), registry
        )


def test_control_creation_is_blocked_before_registry_or_gateway_writes(tmp_path):
    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    supervisor.admit_discovery(req, registry)
    work = supervisor.root / req.run_id
    with pytest.raises(ValueError, match="control state overlaps|Control state overlaps"):
        ResearchRegistry(work / "research.db")
    with pytest.raises(ValueError, match="control state overlaps|Control state overlaps"):
        GatewayDeploymentManager(work / "gateway")
    assert not (work / "research.db").exists()
    assert not (work / "gateway").exists()


@pytest.mark.parametrize("existing_other_db", [False, True])
def test_relative_registry_identity_survives_working_directory_change(
    tmp_path, monkeypatch, existing_other_db
):
    import sqlite3

    original = tmp_path / "control"
    original.mkdir()
    monkeypatch.chdir(original)
    registry = ResearchRegistry("research.db")
    assert registry.path == original / "research.db"
    work = tmp_path / "jobs/A"
    work.mkdir(parents=True)
    if existing_other_db:
        with sqlite3.connect(work / "research.db") as db:
            db.execute("CREATE TABLE blobs(hash TEXT PRIMARY KEY, content BLOB NOT NULL)")
    NativeMountAuthority().reserve({"run": "A"}, writable=work, inputs=(tmp_path / "inputs/A",))
    monkeypatch.chdir(work)
    digest = registry.archive(b"private-fixture-data")
    assert registry.blob(digest) == b"private-fixture-data"
    with sqlite3.connect(original / "research.db") as db:
        assert (
            db.execute("SELECT content FROM blobs WHERE hash=?", (digest,)).fetchone()[0]
            == b"private-fixture-data"
        )
    if existing_other_db:
        with sqlite3.connect(work / "research.db") as db:
            assert db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 0
    else:
        assert not (work / "research.db").exists()


def test_other_registry_control_and_input_roots_are_protected(tmp_path):
    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "primary.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    work = supervisor.root / req.run_id
    ResearchRegistry(work / "other-registry.db")
    with pytest.raises(ValueError, match="reserved control state"):
        supervisor.admit_discovery(req, registry)
    assert not (supervisor.root / ".inputs").exists()


def test_overlapping_concurrent_reservations_have_only_one_winner(tmp_path):
    authority = NativeMountAuthority()
    barrier = Barrier(2)

    def reserve(index):
        barrier.wait(timeout=10)
        work = tmp_path / "jobs/A"
        if index:
            work = work / "nested/B"
        try:
            authority.reserve({"run": index}, writable=work, inputs=(tmp_path / f"inputs/{index}",))
            return "admitted"
        except ValueError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, range(2)))
    assert sorted(results) == ["admitted", "blocked"]


def test_same_run_can_add_protected_paths_but_no_other_run_can_mount_them(tmp_path):
    authority = NativeMountAuthority()
    identity = {"run": "A"}
    work = tmp_path / "jobs/A"
    token = tmp_path / "private/bearer"
    authority.reserve(identity, writable=work, inputs=(tmp_path / "inputs/A",), controls=(token,))
    authority.reserve(identity, writable=work, inputs=(tmp_path / "inputs/A",))
    with pytest.raises(ValueError, match="another reserved"):
        authority.reserve({"run": "B"}, writable=token.parent, inputs=(tmp_path / "inputs/B",))


@pytest.mark.parametrize("source_kind", ["secret", "registry", "control-directory"])
def test_never_expose_controls_cannot_be_reclassified_as_source_documents(tmp_path, source_kind):
    from src.research.registry import RegistryError, content_hash

    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    key = tmp_path / "private/provider"
    key.parent.mkdir()
    value = b"private-provider-fixture-not-a-real-key"
    key.write_bytes(value)
    key.chmod(0o600)
    authority = NativeMountAuthority()
    authority.protect(key)
    source = (
        key
        if source_kind == "secret"
        else registry.path
        if source_kind == "registry"
        else key.parent
    )
    req = req.model_copy(update={"command": "general_model", "document": str(source)})
    supervisor = NativeSupervisor(tmp_path / "jobs")
    with pytest.raises(ValueError, match="never-expose"):
        supervisor.reserve_mounts(req, registry, control_paths=(key,))
    with pytest.raises(ValueError, match="never-expose"):
        supervisor.admit_discovery(req, registry)
    assert not (supervisor.root / ".inputs").exists()
    with pytest.raises(RegistryError, match="missing"):
        registry.blob(content_hash(value))


@pytest.mark.parametrize("kind", ["secret", "registry", "source", "authority"])
def test_hardlinked_control_and_source_files_fail_before_exposure(tmp_path, kind):
    import os

    from src.research.gateway_deployment import private_secret

    authority = NativeMountAuthority()
    inside = tmp_path / "writable/file"
    inside.parent.mkdir()
    inside.write_bytes(b"offline-hardlink-fixture")
    inside.chmod(0o600)
    alias = tmp_path / "alias"
    os.link(inside, alias)
    assert alias.stat().st_ino == inside.stat().st_ino
    if kind == "secret":
        with pytest.raises(ValueError, match="hardlinks"):
            authority.protect(alias)
        with pytest.raises(ValueError, match="ordinary"):
            private_secret(alias)
    elif kind == "registry":
        with pytest.raises(ValueError, match="hardlinks"):
            ResearchRegistry(alias)
    elif kind == "authority":
        with pytest.raises(ValueError, match="hardlinks"):
            NativeMountAuthority(alias)
    else:
        with pytest.raises(ValueError, match="hardlinks"):
            authority.reserve({"run": "A"}, writable=tmp_path / "jobs/A", inputs=(alias,))
