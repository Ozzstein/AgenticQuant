"""Reconcile an admitted native job; never restart discovery or forgive exposure.

Docker is the exit authority. A missing auto-removed container is not a successful
exit. The registry owns immutable launch, container and sequenced recovery records.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract


class NativeFile(VersionedContract):
    tree: Literal["work", "frozen"]
    path: NonEmptyString
    blob_hash: Sha256
    size: int = Field(ge=0)

    @model_validator(mode="after")
    def relative_name(self):
        from pathlib import PurePosixPath

        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or str(path) != self.path or "\\" in self.path:
            raise ValueError("native file needs an exact relative POSIX path")
        return self


class NativeMount(VersionedContract):
    source: NonEmptyString
    destination: NonEmptyString
    read_only: bool


class NativeLaunch(VersionedContract):
    run_id: NonEmptyString
    request_hash: Sha256
    discovery_exposure_hash: Sha256
    account_id: str | None = None  # Original dataset provenance, never trading authority.
    recorded_at: UtcTimestamp
    registry_path: NonEmptyString
    supervisor_root: NonEmptyString
    gateway_token_path: NonEmptyString  # Path only; never credential bytes or a token hash.
    image_id: NonEmptyString
    network_id: NonEmptyString
    user: NonEmptyString
    files: tuple[NativeFile, ...]
    mounts: tuple[NativeMount, ...]
    auto_remove: Literal[True] = True
    policy: Literal["native-recovery-1"] = "native-recovery-1"

    @model_validator(mode="after")
    def unique_files(self):
        if len({(f.tree, f.path) for f in self.files}) != len(self.files):
            raise ValueError("duplicate launch file")
        if len({m.destination for m in self.mounts}) != len(self.mounts):
            raise ValueError("duplicate launch mount")
        if {f.path for f in self.files if f.tree == "work"} != {
            "request.json",
            "scenario.json",
            "admission.json",
        }:
            raise ValueError("exact original native control files required")
        return self


class NativeContainerBinding(VersionedContract):
    run_id: NonEmptyString
    request_hash: Sha256
    discovery_exposure_hash: Sha256
    launch_hash: Sha256
    container_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: UtcTimestamp
    bound_at: UtcTimestamp
    identity_hash: Sha256

    @model_validator(mode="after")
    def chronology(self):
        if self.bound_at < self.created_at:
            raise ValueError("container binding precedes actual creation")
        return self


class NativeRecoveryEvent(VersionedContract):
    run_id: NonEmptyString
    request_hash: Sha256
    discovery_exposure_hash: Sha256
    sequence: int = Field(ge=1)
    previous_hash: Sha256 | None = None
    observed_at: UtcTimestamp
    launch_hash: Sha256 | None = None
    container_binding_hash: Sha256 | None = None
    observation: Literal["absent", "created", "running", "exited", "unavailable", "ambiguous"]
    action: Literal["observe", "stop_requested", "stopped", "stop_uncertain"] = "observe"
    reason: Literal[
        "observed",
        "legacy_launch_unbound",
        "container_unbound",
        "source_changed",
        "identity_changed",
        "docker_unavailable",
        "inspection_ambiguous",
        "auto_removed_exit_unknown",
        "stop_pending",
        "stop_acknowledged",
        "stop_unknown",
        "exit_failed",
        "gateway_unavailable",
        "cost_uncertain",
        "artifacts_unavailable",
        "completion_recovered",
        "authoritative_result_retained",
        "result_conflict",
    ]
    inspection_hash: Sha256 | None = None
    artifact_manifest_hash: Sha256 | None = None
    gateway_receipt_hash: Sha256 | None = None
    result_hash: Sha256 | None = None
    cost_state: Literal["unknown", "uncertain", "audited"] = "unknown"
    exposure_retained: Literal[True] = True
    activation: Literal["quarantined"] = "quarantined"
    internal_search_trials: None = None

    @model_validator(mode="after")
    def chain(self):
        if (self.sequence == 1) != (self.previous_hash is None):
            raise ValueError("recovery events require exact append-only sequencing")
        return self
