"""CLI boundaries using synthetic XML/receipts only; no broker data or retrieval."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.data import __main__ as cli
from src.data.ibkr_flex import read_flex_capture

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-10-31T00:00:00+00:00"


@pytest.fixture
def synthetic_capture(tmp_path):
    directory = tmp_path / "synthetic-capture"
    directory.mkdir(mode=0o700)
    # Deliberately identifiable synthetic values must never appear in CLI output.
    report = (
        '<FlexQueryResponse><FlexStatements count="1">'
        '<FlexStatement accountId="SYNTHETIC-ACCOUNT" fromDate="20261001" toDate="20261030">'
        '<CashTransactions><CashTransaction currency="USD" amount="12345.6789" '
        'type="Dividends" transactionID="SYNTHETIC-TXN" description="SYNTHETIC-PRIVATE"/>'
        "</CashTransactions></FlexStatement></FlexStatements></FlexQueryResponse>"
    ).encode()
    ack = b"<FlexStatementResponse><Status>Success</Status><ReferenceCode>12345</ReferenceCode></FlexStatementResponse>"
    for index, (operation, raw) in enumerate((("SendRequest", ack), ("GetStatement", report)), 1):
        name = f"{index:02d}-{operation}.xml"
        receipt = {
            "raw_file": name,
            "operation": operation,
            "http_status": 200,
            "request_version": "3",
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "query_id_sha256": hashlib.sha256(b"synthetic-query").hexdigest(),
            "started_at": f"2026-10-30T12:00:0{index}+00:00",
            "available_at": f"2026-10-30T12:00:0{index}+00:00",
        }
        (directory / name).write_bytes(raw)
        (directory / (name + ".receipt.json")).write_text(json.dumps(receipt))
    (directory / "capture.json").write_text(
        json.dumps(
            {
                "status": "downloaded",
                "directory": str(directory),
                "requests": 2,
                "report_file": "02-GetStatement.xml",
                "statements": 1,
            }
        )
    )
    for path in directory.iterdir():
        path.chmod(0o600)
    return directory


def invoke(route, args, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    # Resolve the installed entry point from this checkout, not root's editable package.
    if route == "canonical":
        command = [sys.executable, "-c", "from src.utils.cli import app; app()", "data"]
    elif route == "standalone":
        command = [sys.executable, "-m", "scripts.inspect_ibkr_flex_report"]
    else:
        command = [sys.executable, "-m", "src.data"]
    return subprocess.run(
        command + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize("route", ["direct", "canonical"])
@pytest.mark.parametrize("supply_store", [False, True])
def test_actual_cli_safe_summary_read_only(route, supply_store, synthetic_capture, tmp_path):
    before = {p.name: p.read_bytes() for p in synthetic_capture.iterdir()}
    store = tmp_path / "never-created" / "evidence.sqlite3"
    args = (["--store", str(store)] if supply_store else []) + [
        "inspect-ibkr-flex",
        "--capture",
        str(synthetic_capture),
        "--as-of",
        CUTOFF,
    ]
    result = invoke(route, args, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(result.stdout)
        == read_flex_capture(synthetic_capture, as_of=datetime.fromisoformat(CUTOFF)).safe_summary()
    )
    assert not store.parent.exists()
    assert {p.name: p.read_bytes() for p in synthetic_capture.iterdir()} == before
    for secret in ("SYNTHETIC-ACCOUNT", "SYNTHETIC-TXN", "SYNTHETIC-PRIVATE", "12345.6789"):
        assert secret not in result.stdout + result.stderr


@pytest.mark.parametrize("route", ["direct", "canonical"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_actual_cli_failure_is_closed_and_private(route, damage, synthetic_capture, tmp_path):
    if damage == "missing":
        (synthetic_capture / "capture.json").unlink()
        reason = "private_source_unavailable"
    else:
        (synthetic_capture / "capture.json").write_text("SYNTHETIC-PRIVATE invalid JSON")
        reason = "invalid_capture_metadata"
    store = tmp_path / "never-created" / "evidence.sqlite3"
    result = invoke(
        route,
        ["--store", str(store), "inspect-ibkr-flex", "--capture", str(synthetic_capture)],
        tmp_path,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"status": "unavailable", "reason": reason}
    assert str(synthetic_capture) not in result.stdout + result.stderr
    assert "SYNTHETIC-PRIVATE" not in result.stdout + result.stderr
    assert not store.parent.exists()


@pytest.mark.parametrize("cutoff", ["", "SYNTHETIC-PRIVATE", "2026-10-31T00:00:00"])
def test_invalid_cutoff_is_sanitized(cutoff, synthetic_capture, capsys):
    assert (
        cli.main(["inspect-ibkr-flex", "--capture", str(synthetic_capture), "--as-of", cutoff]) == 1
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "unavailable",
        "reason": "aware_cutoff_required",
    }


def test_reader_cutoff_and_unexpected_failure(synthetic_capture, monkeypatch, capsys):
    args = ["inspect-ibkr-flex", "--capture", str(synthetic_capture)]
    assert cli.main(args + ["--as-of", "2020-01-01T00:00:00+00:00"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "capture_clock_mismatch"

    def fail(*args, **kwargs):
        raise RuntimeError("SYNTHETIC-PRIVATE financial text")

    monkeypatch.setattr("src.data.ibkr_flex.read_flex_capture", fail)
    assert cli.main(args) == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "unavailable",
        "reason": "capture_inspection_failed",
    }


def test_inspection_dispatch_never_imports_evidence(synthetic_capture, monkeypatch, capsys):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert not name.startswith("src.data.evidence"), "inspection imported evidence machinery"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert (
        cli.main(
            [
                "--store",
                "/unused/evidence.sqlite3",
                "inspect-ibkr-flex",
                "--capture",
                str(synthetic_capture),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["income_admission"] == "blocked"


LEGACY_COMMANDS = [
    ["import-archive", "--manifest", "unused"],
    ["query", "--as-of", CUTOFF],
    ["document", "--as-of", CUTOFF, "--evidence-id", "unused"],
    ["snapshot", "--context", "unused", "--instruments", "unused", "--start", CUTOFF],
    [
        "train",
        "--snapshot",
        "unused",
        "--context",
        "unused",
        "--features",
        "unused",
        "--output",
        "unused",
    ],
    [
        "predict",
        "--snapshot",
        "unused",
        "--context",
        "unused",
        "--features",
        "unused",
        "--artifact",
        "unused",
    ],
    ["qlib-export", "--snapshot", "unused", "--output", "unused"],
    ["qlib-alpha158", "--scenario", "unused", "--output", "unused"],
    [
        "fmp-transcript",
        "--instrument",
        "unused",
        "--context",
        "unused",
        "--year",
        "2026",
        "--quarter",
        "3",
    ],
]


@pytest.mark.parametrize("args", LEGACY_COMMANDS)
def test_every_existing_command_still_requires_store(args, monkeypatch, capsys):
    def fail(*args):
        pytest.fail("must reject missing store before executing any command")

    monkeypatch.setattr(cli, "execute", fail)
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2
    assert "required: --store" in capsys.readouterr().err


@pytest.mark.parametrize("route", ["direct", "canonical"])
def test_existing_query_still_uses_explicit_store(route, tmp_path):
    store = tmp_path / "evidence.sqlite3"
    result = invoke(route, ["--store", str(store), "query", "--as-of", CUTOFF], tmp_path)
    assert result.returncode == 0, result.stderr
    assert store.is_file()
    assert json.loads(result.stdout)["records"] == []


def test_standalone_remains_compatible(synthetic_capture, tmp_path):
    result = invoke("standalone", ["--capture", str(synthetic_capture)], tmp_path)
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(result.stdout)
        == read_flex_capture(synthetic_capture, as_of=datetime.now(timezone.utc)).safe_summary()
    )


def test_loaded_implementation_is_this_worktree():
    assert Path(cli.__file__).resolve() == ROOT / "src/data/__main__.py"


def test_fresh_direct_cli_imports_no_evidence(synthetic_capture, tmp_path):
    # SystemExit is expected from module execution; inspect imports in finally.
    program = """
import runpy
import sys
try:
    runpy.run_module("src.data", run_name="__main__")
finally:
    assert not any(n.startswith("src.data.evidence") for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, "inspect-ibkr-flex", "--capture", str(synthetic_capture)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["income_admission"] == "blocked"


def test_cutoff_defaults_to_invocation_clock_and_preserves_explicit_offset(
    synthetic_capture, monkeypatch, capsys
):
    observed = []

    def read(directory, *, as_of):
        observed.append(as_of)
        return read_flex_capture(directory, as_of=as_of)

    monkeypatch.setattr("src.data.ibkr_flex.read_flex_capture", read)
    args = ["inspect-ibkr-flex", "--capture", str(synthetic_capture)]
    before = datetime.now(timezone.utc)
    assert cli.main(args) == 0
    assert before <= observed[-1] <= datetime.now(timezone.utc)
    cutoff = "2026-10-31T02:00:00+02:00"
    assert cli.main(args + ["--as-of", cutoff]) == 0
    assert observed[-1].isoformat() == cutoff
    capsys.readouterr()
