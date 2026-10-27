"""Closed immutable recovery contracts; no Docker/provider acceptance implied."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.research.native_recovery import NativeContainerBinding, NativeFile, NativeRecoveryEvent


def test_native_file_rejects_escaped_and_ambiguous_names():
    for path in ("../provider", "/provider", "a/../provider", "a//b", "a\\b"):
        with pytest.raises(ValidationError):
            NativeFile(tree="frozen", path=path, blob_hash="1" * 64, size=1)
    item = NativeFile(tree="frozen", path="qlib/manifest.json", blob_hash="1" * 64, size=1)
    with pytest.raises(ValidationError):
        item.path = "different"


def test_recovery_requires_exact_sequence_and_retained_exposure():
    payload = dict(
        run_id="job",
        request_hash="1" * 64,
        discovery_exposure_hash="2" * 64,
        sequence=1,
        observed_at=datetime.now(timezone.utc),
        observation="absent",
        reason="auto_removed_exit_unknown",
    )
    assert NativeRecoveryEvent(**payload).cost_state == "unknown"
    for update in (
        {"sequence": 2},
        {"previous_hash": "3" * 64},
        {"exposure_retained": False},
        {"activation": "approved"},
        {"internal_search_trials": 1},
    ):
        with pytest.raises(ValidationError):
            NativeRecoveryEvent(**(payload | update))


def test_container_binding_requires_actual_creation_before_binding():
    with pytest.raises(ValidationError):
        NativeContainerBinding(
            run_id="job",
            request_hash="1" * 64,
            discovery_exposure_hash="2" * 64,
            launch_hash="3" * 64,
            container_id="4" * 64,
            identity_hash="5" * 64,
            created_at="2026-10-27T12:00:01Z",
            bound_at="2026-10-27T12:00:00Z",
        )
