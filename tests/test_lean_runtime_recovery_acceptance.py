"""Read-only assertions over actual engine evidence, never synthetic runtime mocks.

Set QAL_LEAN_RECOVERY_EVIDENCE to a recovery_driver output directory. These tests
launch no engine and intentionally skip when retained private evidence is absent.
"""

import json
import os
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def evidence():
    path = os.environ.get("QAL_LEAN_RECOVERY_EVIDENCE")
    if not path:
        pytest.skip("requires retained actual LEAN recovery evidence")
    return Path(path)


def read(root, name):
    return json.loads((root / name).read_text())


@pytest.mark.parametrize("phase", ["origin", "pending-restart"])
def test_bounded_engine_and_exact_cleanup(evidence, phase):
    launch = read(evidence / phase, "launch.json")
    assert not launch.get("timed_out")
    inspection = launch["inspection"]
    assert inspection["Id"] == launch["container_id"]
    assert inspection["Image"] == launch["image"]
    assert inspection["HostConfig"]["NetworkMode"] == "none"
    assert inspection["HostConfig"]["Memory"] <= 4 * 1024**3
    assert inspection["HostConfig"]["NanoCpus"] <= 2 * 10**9
    assert inspection["HostConfig"]["PidsLimit"] == 256
    assert not inspection["State"]["OOMKilled"]
    assert not inspection["State"]["Running"]
    mount = next(m for m in inspection["Mounts"] if m["Destination"] == "/app/src")
    assert mount["Source"] == launch["source"] + "/src"
    assert mount["RW"] is False
    assert launch["cleanup"]["exit"] == 0
    assert launch["after_cleanup"]["exit"] != 0
    assert "no such object" in launch["after_cleanup"]["stderr"].lower()
    if phase == "origin":
        assert inspection["State"]["ExitCode"] == launch["attach_exit"] == 0

    else:
        assert inspection["State"]["ExitCode"] == launch["attach_exit"] == 1


def test_completed_retry_has_no_duplicate_submission_or_fill(evidence):
    result = read(evidence / "origin", "retry-result.json")
    assert len(result["before"]["fills"]) == 1
    assert len(result["retries"]) == 3
    assert all(row["state"] == "filled" for row in result["retries"])
    for key in ("bindings", "attempts", "fills", "native_orders", "outbox"):
        assert result["after"][key] == result["before"][key]
    assert len(result["after"]["attempts"]) == 1
    assert len(result["after"]["native_orders"]) == 1


def test_pending_restart_retains_exact_original_ownership(evidence):
    original = read(evidence / "origin", "timeout.json")
    rejected = read(evidence / "pending-restart", "initialization-error.json")
    assert (
        rejected["error"]
        == "LEAN paper restart has unresolved accepted orders; original proof required"
    )
    restarted = rejected["snapshot"]
    assert restarted["runtime"]["runtime_id"] != original["runtime"]["runtime_id"]
    assert len(original["native_orders"]) == 1
    assert restarted["native_orders"] == []
    assert len(original["attempts"]) == 2
    for key in ("bindings", "attempts", "fills", "outbox", "pending"):
        assert restarted[key] == original[key]
    pending = [row for row in original["outbox"] if row["intent"] == "recovery-pending"]
    assert len(pending) == 1 and pending[0]["state"] == "ambiguous"
    assert original["pending"]
    assert not (evidence / "pending-restart" / "initialized.json").exists()


def test_fresh_readonly_ledger_preserves_original_receipts(evidence):
    def rows(phase, table):
        path = evidence / phase / "ledger.sqlite"
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            return db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()

    for table in ("lean_ticket_bindings", "venue_outbox", "venue_reports", "fills"):
        assert rows("origin", table) == rows("pending-restart", table)
