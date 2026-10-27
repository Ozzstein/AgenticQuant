"""Real bundle and SQLite exposure tests; subprocess fixtures are not Docker acceptance."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.data.qlib_us import USQlibScenario, export_bundle
from src.research.native import NativeSupervisor
from src.research.registry import Holdout, RegistryError, ResearchRegistry, canonical, content_hash
from tests.data_quant_fixtures import archived_snapshot
from tests.test_native_rd_adapter import request


def mounted_request(tmp_path):
    snapshot = archived_snapshot(periods=60, use_exchange_calendar=True)
    path = tmp_path / "data"
    bundle = export_bundle(snapshot, path)
    days = sorted({bar.session for bar in snapshot.bars})
    first = bundle["instrument_map"]["equity:0"]
    scenario = USQlibScenario(
        provider_uri=str(path),
        bundle_hash=bundle["bundle_hash"],
        universe=(first,),
        benchmark=first,
        price_basis="raw",
        train=(days[0], days[20]),
        valid=(days[23], days[30]),
        test=(days[33], days[40]),
    )
    return request().model_copy(update={"scenario": scenario}), snapshot


def held_out(registry, snapshot, *, key="reserved", later=False):
    # This instrument is in the physical mount but outside the chosen universe;
    # its last observations are also beyond the configured OOS interval.
    bars = tuple(bar for bar in snapshot.bars if bar.instrument.instrument_id == "equity:2")
    keys = {bar.evidence_id for bar in bars}
    revised = snapshot.model_copy(
        update={
            "context": snapshot.context.model_copy(update={"run_id": "different-data-revision"}),
            "bars": bars,
            "evidence": tuple(e for e in snapshot.evidence if e.evidence_id in keys),
        }
    )
    start, end = bars[-2].opened_at, bars[-1].closed_at + timedelta(microseconds=1)
    if later:
        start, end = end, end + timedelta(days=10)
    return Holdout(
        holdout_id=key,
        dataset_hash=registry.archive_contract(revised),
        starts_at=start,
        ends_at=end,
        registered_at=datetime.now(timezone.utc),
    )


def test_full_mounted_bundle_cannot_touch_existing_holdout(tmp_path):
    req, snapshot = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    registry.register_holdout(held_out(registry, snapshot))
    supervisor = NativeSupervisor(tmp_path / "jobs")
    with pytest.raises(RegistryError, match="mounted data touches"):
        supervisor.admit_discovery(req, registry)
    assert registry.status()["native_discovery"]["admissions"] == 0


def test_failed_launch_still_exposes_all_rows_and_forbids_resubmission(tmp_path, monkeypatch):
    req, snapshot = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    monkeypatch.setattr(supervisor, "preflight", lambda *a, **k: {"boundary": "offline"})
    monkeypatch.setattr(
        supervisor, "command", lambda *a, **k: [str(tmp_path / "missing-executable")]
    )
    with pytest.raises(FileNotFoundError):
        supervisor.run(req, gateway_token_path=tmp_path / "unused-token", registry=registry)
    reopened = ResearchRegistry(registry.path)
    exposure = reopened.discovery_exposure("native:" + req.run_id)
    assert exposure.starts_at == min(bar.opened_at for bar in snapshot.bars)
    assert exposure.ends_at == snapshot.context.as_of + timedelta(microseconds=1)
    assert exposure.internal_trials is None
    assert reopened.status()["native_discovery"] == {"admissions": 1, "internal_trials": None}
    with pytest.raises(RegistryError, match="already exposed"):
        reopened.register_holdout(held_out(reopened, snapshot))
    reopened.register_holdout(held_out(reopened, snapshot, key="later", later=True))
    with pytest.raises(RegistryError, match="already admitted"):
        supervisor.run(req, gateway_token_path=tmp_path / "unused-token", registry=reopened)
    with (
        sqlite3.connect(registry.path) as db,
        pytest.raises(sqlite3.IntegrityError, match="immutable"),
    ):
        db.execute("DELETE FROM discovery_exposures")


@pytest.mark.parametrize(
    "mutation", ["extra", "symlink", "parent-symlink", "snapshot-omission", "binary"]
)
def test_only_reproducible_complete_bundle_bytes_can_be_exposed(tmp_path, mutation):
    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    root = Path(req.scenario.provider_uri)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if mutation == "extra":
        (root / "unrecorded-holdout.csv").write_text("private holdout")
    elif mutation == "symlink":
        (root / "extra-link").symlink_to(root / "manifest.json")
    elif mutation == "parent-symlink":
        link = tmp_path / "linked-parent"
        link.symlink_to(tmp_path, target_is_directory=True)
        req = req.model_copy(
            update={
                "scenario": req.scenario.model_copy(update={"provider_uri": str(link / "data")})
            }
        )
    else:
        if mutation == "snapshot-omission":
            manifest["snapshot"]["bars"] = [
                bar
                for bar in manifest["snapshot"]["bars"]
                if bar["instrument"]["instrument_id"] != "equity:2"
            ]
        else:
            name = next(name for name in manifest["files"] if name.endswith("close.day.bin"))
            altered = bytearray((root / name).read_bytes())
            altered[-1] ^= 1
            (root / name).write_bytes(altered)
            manifest["files"][name] = content_hash(altered)
        raw = canonical(manifest)
        (root / "manifest.json").write_bytes(raw)
        req = req.model_copy(
            update={"scenario": req.scenario.model_copy(update={"bundle_hash": content_hash(raw)})}
        )
    with pytest.raises(ValueError, match="symlink|symbolic links|unsupported links|unrecorded|exposure snapshot"):
        NativeSupervisor(tmp_path / "jobs").admit_discovery(req, registry)
    assert registry.status()["native_discovery"]["admissions"] == 0


def test_discovery_requires_registry_before_any_provider_or_container_call(tmp_path, monkeypatch):
    supervisor = NativeSupervisor(tmp_path / "jobs")
    monkeypatch.setattr(
        supervisor, "preflight", lambda *a, **k: pytest.fail("no preflight without registry")
    )
    with pytest.raises(ValueError, match="authoritative research registry"):
        supervisor.run(request(), gateway_token_path=tmp_path / "unused", registry=None)


def test_mutating_original_mount_cannot_change_admitted_data(tmp_path):
    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    supervisor.admit_discovery(req, registry)
    frozen = supervisor.root / ".inputs" / req.run_id
    original = Path(req.scenario.provider_uri)
    (original / "private-holdout.csv").write_text("new reserved observations")
    (original / "calendars/day.txt").write_text("changed after admission")
    work = supervisor.root / req.run_id
    work.mkdir()
    command = supervisor.command(req, work, tmp_path / "token", frozen_root=frozen)
    assert f"type=bind,src={frozen}/qlib,dst=/data/qlib,readonly" in command
    assert not any(f"src={original}," in part for part in command)
    assert not (frozen / "qlib/private-holdout.csv").exists()
    assert (frozen / "qlib/calendars/day.txt").read_text() != "changed after admission"
    assert not frozen.is_relative_to(work)


@pytest.mark.parametrize("command", ["general_model", "fin_factor_report"])
def test_opaque_document_scope_is_unbounded_and_frozen_without_writable_alias(tmp_path, command):
    req, snapshot = mounted_request(tmp_path)
    document = tmp_path / "paper.txt"
    document.write_text("A paper may contain market outcomes outside the chosen Qlib interval.")
    req = req.model_copy(update={"command": command, "document": str(document)})
    registry = ResearchRegistry(tmp_path / "registry.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    exposure = supervisor.admit_discovery(req, registry)
    assert exposure.observation_scope == "unbounded"
    proof = json.loads(registry.blob(exposure.document_manifest_hash))
    assert registry.blob(proof["files"][document.name]["sha256"]) == document.read_bytes()
    with pytest.raises(RegistryError, match="already exposed"):
        registry.register_holdout(held_out(registry, snapshot, later=True))
    frozen = supervisor.root / ".inputs" / req.run_id
    command = supervisor.command(
        req, supervisor.root / req.run_id, tmp_path / "token", frozen_root=frozen
    )
    document.write_text("substitution")
    assert (frozen / "sources" / document.name).read_text() != "substitution"
    assert f"type=bind,src={frozen}/sources,dst=/sources,readonly" in command
    assert not (supervisor.root / req.run_id / "sources").exists()


def test_document_cannot_bypass_an_existing_disjoint_market_holdout(tmp_path):
    req, snapshot = mounted_request(tmp_path)
    document = tmp_path / "holdout-paper.txt"
    document.write_text("Opaque original observations")
    req = req.model_copy(update={"command": "general_model", "document": str(document)})
    registry = ResearchRegistry(tmp_path / "registry.db")
    registry.register_holdout(held_out(registry, snapshot, later=True))
    with pytest.raises(RegistryError, match="touches a reserved holdout"):
        NativeSupervisor(tmp_path / "jobs").admit_discovery(req, registry)


@pytest.mark.parametrize("source", ["knowledge", "ui"])
def test_legacy_self_certified_source_outputs_are_not_admissible(tmp_path, source):
    from src.research.native import NativeKnowledgeRef

    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    predecessor = supervisor.root / "legacy"
    predecessor.mkdir()
    (predecessor / "result.json").write_text('{"status":"discovery_complete"}')
    if source == "knowledge":
        ref = NativeKnowledgeRef(
            source_run="legacy",
            source_result_hash=content_hash((predecessor / "result.json").read_bytes()),
            artifact="factor",
            artifact_hash=content_hash(b"opaque legacy pickle"),
        )
        req = req.model_copy(update={"knowledge": (ref,)})
    else:
        req = req.model_copy(update={"command": "ui", "ui_source_run": "legacy"})
    with pytest.raises(RegistryError, match="unknown native discovery exposure"):
        supervisor.admit_discovery(req, registry)


def record_fixture_completion(supervisor, registry, req, outputs):
    """Coordinator completion fixture; no Docker/provider success is asserted."""
    exposure = supervisor.admit_discovery(req, registry)
    work = supervisor.root / req.run_id
    work.mkdir()
    files = {"request.json": req.model_dump_json().encode(), **outputs}
    for name, raw in files.items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    result = dict(
        run_id=req.run_id,
        request_hash=req.content_hash(),
        status="discovery_complete",
        activation="quarantined",
        discovery_exposure_hash=exposure.content_hash(),
        internal_search_trials=None,
        upstream_revision="484776c211e4fbbeef03e0ec00d6bbee7362a4f4",
        gateway_audit_error=None,
        gateway_final_receipt={"fixture": "offline"},
        files={name: content_hash(raw) for name, raw in files.items()},
    )
    registry.record_native_result(result)
    raw = canonical(result)
    (work / "result.json").write_bytes(raw)
    return exposure, content_hash(raw)


@pytest.mark.parametrize("inherited", ["knowledge", "ui"])
def test_inherited_opaque_scope_and_actual_source_bytes_remain_bound(tmp_path, inherited):
    from src.research.native import NativeKnowledgeRef

    req, snapshot = mounted_request(tmp_path)
    document = tmp_path / "paper.txt"
    document.write_text("Unknown market observation scope")
    prior = req.model_copy(
        update={"run_id": "prior", "command": "general_model", "document": str(document)}
    )
    registry = ResearchRegistry(tmp_path / "registry.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    data = b"immutable inherited content"
    exposure, result_hash = record_fixture_completion(
        supervisor, registry, prior, {"knowledge/factor.pkl": data, "log/session.txt": data}
    )
    if inherited == "knowledge":
        ref = NativeKnowledgeRef(
            source_run="prior",
            source_result_hash=result_hash,
            artifact="factor",
            artifact_hash=content_hash(data),
        )
        child = req.model_copy(update={"knowledge": (ref,)})
        staged = "knowledge/factor.pkl"
    else:
        child = req.model_copy(update={"command": "ui", "ui_source_run": "prior"})
        staged = "log/session.txt"
    admitted = supervisor.admit_discovery(child, registry)
    assert admitted.inherited_exposures == (exposure.content_hash(),)
    assert admitted.observation_scope == "unbounded"
    frozen = supervisor.root / ".inputs" / child.run_id
    (
        supervisor.root / "prior" / ("knowledge/factor.pkl" if inherited == "knowledge" else staged)
    ).write_bytes(b"later mutation")
    assert (frozen / staged).read_bytes() == data
    with pytest.raises(RegistryError, match="already exposed"):
        registry.register_holdout(held_out(registry, snapshot, later=True))
    command = supervisor.command(
        child, supervisor.root / child.run_id, tmp_path / "token", frozen_root=frozen
    )
    assert any(f"src={frozen}/" in item and item.endswith("readonly") for item in command)


def test_self_consistent_replacement_cannot_override_authoritative_completion(tmp_path):
    from src.research.native import NativeKnowledgeRef

    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    prior = req.model_copy(update={"run_id": "prior"})
    data = b"original"
    record_fixture_completion(supervisor, registry, prior, {"knowledge/factor.pkl": data})
    result_path = supervisor.root / "prior/result.json"
    forged = json.loads(result_path.read_bytes())
    forged["files"]["knowledge/factor.pkl"] = content_hash(b"replacement")
    raw = canonical(forged)
    result_path.write_bytes(raw)
    (supervisor.root / "prior/knowledge/factor.pkl").write_bytes(b"replacement")
    ref = NativeKnowledgeRef(
        source_run="prior",
        source_result_hash=content_hash(raw),
        artifact="factor",
        artifact_hash=content_hash(b"replacement"),
    )
    with pytest.raises(ValueError, match="authoritative registry completion"):
        supervisor.knowledge_bytes(ref, registry=registry)
    with pytest.raises(RegistryError, match="result is immutable"):
        registry.record_native_result(forged)


@pytest.mark.parametrize("descendant", [False, True])
def test_native_or_relabelled_descendant_cannot_activate_on_unknown_search_count(
    tmp_path, descendant
):
    from tests.test_research_registry import T, admitted

    registry = ResearchRegistry(tmp_path / "registry.db")
    _, _, outcome, policy = admitted(registry, "native", origin="native_rd_agent")
    if descendant:
        _, _, outcome, policy = admitted(registry, "renamed", parent_ids=("native",))
    registry.record_evaluation(outcome)
    with pytest.raises(RegistryError, match="multiplicity and exact implementation audit"):
        registry.activate(
            "factor", outcome.trial_id, policy, expected_generation=0, at=T + timedelta(hours=2)
        )
    assert registry.active("factor") is None


def test_upgrade_refuses_preexisting_native_activation_and_rollback(tmp_path, monkeypatch):
    from tests.test_research_registry import T, admitted

    registry = ResearchRegistry(tmp_path / "registry.db")
    with monkeypatch.context() as old:
        # The previous registry admitted these origins through its normal APIs.
        old.setattr(ResearchRegistry, "_require_audited_origin", staticmethod(lambda *args: None))
        _, _, parent, policy = admitted(registry, "native", origin="native_rd_agent")
        registry.record_evaluation(parent)
        registry.activate(
            "factor", "native", policy, expected_generation=0, at=T + timedelta(hours=2)
        )
        candidate, _, child, _ = admitted(registry, "child", parent_ids=("native",))
        registry.record_evaluation(child)
        registry.activate(
            "factor", "child", policy, expected_generation=1, at=T + timedelta(hours=3)
        )
    reopened = ResearchRegistry(registry.path)
    with pytest.raises(RegistryError, match="multiplicity"):
        reopened.resolve(
            "factor", payload_hash=candidate.payload_hash, as_of=T + timedelta(hours=4)
        )
    with pytest.raises(RegistryError, match="multiplicity"):
        reopened.rollback("factor", 1, expected_generation=2, at=T + timedelta(hours=4))
    assert reopened.status()["active"] == []
    assert reopened.status()["blocked_activations"][0]["activation"]["candidate_id"] == "child"
    with sqlite3.connect(registry.path) as db:
        assert db.execute("SELECT count(*) FROM activations").fetchone()[0] == 2
