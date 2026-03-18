"""Smoke tests for src/dashboard.py — import and data-loading functions only."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dashboard_imports_without_error() -> None:
    """Importing src.dashboard must not raise any exception."""
    import src.dashboard  # noqa: F401

    assert True


def test_load_portfolio_handles_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_portfolio() should return a dict even when the file is missing."""
    import src.dashboard as dash

    # Clear any cached values before patching paths
    try:
        dash._load_portfolio.clear()
        dash._load_json.clear()
    except AttributeError:
        pass

    monkeypatch.setattr(dash, "_ROOT", tmp_path)
    monkeypatch.setattr(dash, "_OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(dash, "_DATA", tmp_path / "data")

    result = dash._load_portfolio.__wrapped__() if hasattr(dash._load_portfolio, "__wrapped__") else dash._load_portfolio()
    assert isinstance(result, dict)


def test_load_audit_files_handles_empty_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_audit_files() should return a list even when audit dir is empty."""
    import src.dashboard as dash

    try:
        dash._load_audit_files.clear()
        dash._load_json.clear()
    except AttributeError:
        pass

    audit_dir = tmp_path / "data" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dash, "_ROOT", tmp_path)
    monkeypatch.setattr(dash, "_OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(dash, "_DATA", tmp_path / "data")

    result = dash._load_audit_files.__wrapped__() if hasattr(dash._load_audit_files, "__wrapped__") else dash._load_audit_files()
    assert isinstance(result, list)


def test_load_factor_library_handles_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_factor_library() should return a dict even when the file is missing."""
    import src.dashboard as dash

    try:
        dash._load_factor_library.clear()
        dash._load_json.clear()
    except AttributeError:
        pass

    monkeypatch.setattr(dash, "_ROOT", tmp_path)
    monkeypatch.setattr(dash, "_OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(dash, "_DATA", tmp_path / "data")

    result = dash._load_factor_library.__wrapped__() if hasattr(dash._load_factor_library, "__wrapped__") else dash._load_factor_library()
    assert isinstance(result, dict)
