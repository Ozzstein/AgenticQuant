"""Tests for src/utils/audit.py — AuditLogger lifecycle and persistence."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

import src.utils.audit as audit_mod
from src.utils.audit import AuditLogger
from src.utils.schemas import AgentReport, AnalysisResult, AssetClass, Decision


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect _PROJECT_ROOT so audit files land in tmp_path."""
    monkeypatch.setattr(audit_mod, "_PROJECT_ROOT", tmp_path)
    return tmp_path / "data" / "audit"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        ticker="AAPL",
        asset_class=AssetClass.EQUITY,
        decision=Decision.BUY,
        confidence=75.0,
        target_price=200.0,
        stop_loss=160.0,
        position_size_pct=5.0,
        time_horizon="3M",
        reasoning="Strong fundamentals.",
        agent_reports={
            "fundamental": AgentReport(
                agent_name="fundamental",
                decision=Decision.BUY,
                confidence=80.0,
                reasoning="Good revenue growth.",
            )
        },
        risk_flags=["High valuation"],
        catalysts=["Product launch"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_audit_logger_creates_file(audit_dir: Path) -> None:
    """finalize_run() should create the audit JSON file on disk."""
    al = AuditLogger()
    al.start_run()
    al.log_step("equity_data", "ok", {"rows": 100})
    path = al.finalize_run()

    assert path is not None
    assert path.exists(), "Audit file should exist after finalize_run()."


def test_all_required_keys_in_output(audit_dir: Path) -> None:
    """The audit JSON must contain all required top-level keys."""
    al = AuditLogger()
    al.start_run()
    path = al.finalize_run()

    assert path is not None
    payload: dict[str, Any] = json.loads(path.read_text())

    required_keys = {
        "date",
        "run_id",
        "pipeline_steps",
        "predictions",
        "analyses",
        "risk_decisions",
        "trades",
        "portfolio_before",
        "portfolio_after",
        "api_costs",
    }
    assert required_keys.issubset(set(payload.keys())), (
        f"Missing keys: {required_keys - set(payload.keys())}"
    )


def test_finalize_run_returns_valid_path(audit_dir: Path) -> None:
    """finalize_run() must return a Path that exists and contains valid JSON."""
    al = AuditLogger()
    al.start_run()
    path = al.finalize_run()

    assert isinstance(path, Path), "Return value should be a Path."
    assert path.exists(), "Returned path must point to an existing file."
    # Must be valid JSON
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)


def test_multiple_log_step_calls_append(audit_dir: Path) -> None:
    """Three log_step() calls should produce 3 entries in pipeline_steps."""
    al = AuditLogger()
    al.start_run()
    al.log_step("step_a", "ok")
    al.log_step("step_b", "ok")
    al.log_step("step_c", "error", {"msg": "oops"})
    path = al.finalize_run()

    assert path is not None
    payload = json.loads(path.read_text())
    assert len(payload["pipeline_steps"]) == 3


def test_start_run_with_custom_date(audit_dir: Path) -> None:
    """finalize_run() filename should contain the custom date supplied to start_run()."""
    al = AuditLogger()
    al.start_run(run_date="2025-01-15")
    path = al.finalize_run()

    assert path is not None
    assert "2025-01-15" in path.name, "Filename should contain the custom date."


def test_log_analysis_extracts_fields(audit_dir: Path) -> None:
    """log_analysis() should extract decision, confidence and reasoning correctly."""
    al = AuditLogger()
    al.start_run()
    result = _make_analysis_result()
    al.log_analysis("AAPL", result)
    path = al.finalize_run()

    assert path is not None
    payload = json.loads(path.read_text())
    analyses = payload["analyses"]
    assert len(analyses) == 1

    entry = analyses[0]
    assert entry["ticker"] == "AAPL"
    assert entry["decision"] == "BUY"
    assert entry["confidence"] == 75.0
    assert "Strong fundamentals" in entry["reasoning"]


@pytest.mark.skipif(sys.platform == "win32", reason="chmod not supported on Windows")
def test_finalize_returns_none_on_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """finalize_run() must return None (not raise) when the audit dir is read-only."""
    # Create the audit dir, then make it read-only
    audit_dir = tmp_path / "data" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x only

    monkeypatch.setattr(audit_mod, "_PROJECT_ROOT", tmp_path)

    al = AuditLogger()
    al.start_run()
    result = al.finalize_run()

    # Restore permissions for cleanup
    audit_dir.chmod(stat.S_IRWXU)

    assert result is None, "finalize_run() should return None on write error."


def test_log_step_is_graceful_on_bad_input(audit_dir: Path) -> None:
    """log_step() with empty string and None values should not raise."""
    al = AuditLogger()
    al.start_run()
    # Should not raise
    al.log_step("", None, None)  # type: ignore[arg-type]
    path = al.finalize_run()
    assert path is not None
