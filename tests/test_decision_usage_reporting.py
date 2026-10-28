"""Real immutable decision usage reaches read-only CLI and rendered operations UI."""

# ruff: noqa: F811 -- imported pytest fixtures are injected by name

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from scripts.run_operations import app
from src.decision.store import read_usage_summary
from src.operations.reporting import decision_budget_view, system_view
from src.utils.config import AppConfig, reset_config
from tests.test_decision_contracts import (  # noqa: F401
    decision_context,
    decision_instrument,
    decision_question,
    decision_request,
    decision_source,
)
from tests.test_decision_service import response_for
from tests.test_decision_usage import limits, service


def capture(path, request, kind):
    def handler(_):
        if kind == "unknown":
            return httpx.Response(503, text="fixture unavailable")
        return httpx.Response(
            200, json=response_for(request, usage={"input_tokens": 10_001, "output_tokens": 20})
        )

    return service(path, handler, limits(max_attempts=1)).evaluate(request, mode="refresh")


@pytest.mark.parametrize("kind", ["overrun", "unknown"])
def test_status_uses_exact_verified_usage_without_creating_or_mutating_sources(
    tmp_path, monkeypatch, decision_request, kind
):
    config = AppConfig(_env_file=None, output_dir=str(tmp_path / "outputs"))
    path = tmp_path / "outputs" / "decisions.sqlite"
    observed = capture(path, decision_request, kind)
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    files = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    view = system_view(config)["decision_budgets"]
    assert view == read_usage_summary(
        path, day=view["day"], as_of=datetime.fromisoformat(view["as_of"])
    )
    usage = view["day_usage"]
    assert usage["requests"] == 1
    assert usage["breached"] == (kind == "overrun")
    assert usage["unknown_usage_reservations"] == (kind == "unknown")
    if kind == "overrun":
        assert usage["known_cost_usd"] == pytest.approx(0.010001)
        assert usage["breaches"][0]["observation_id"] == observed.observation_id
        assert usage["conservative_tokens"] == 10_021
    else:
        assert usage["total_cost_usd"] is None
        assert usage["conservative_cost_usd"] > 0
    with monkeypatch.context() as patch:
        patch.setattr("src.utils.config.get_config", lambda: config)
        result = CliRunner().invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    output = json.loads(result.output)["decision_budgets"]
    assert output["day_usage"] == usage
    for secret in ("fixture-secret", "Management said:", "Direction?", "raw_response", str(path)):
        assert secret not in json.dumps(output)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == files


def test_corrupt_decision_usage_is_unavailable_not_zero_or_raw_error(tmp_path, decision_request):
    path = tmp_path / "decisions.sqlite"
    capture(path, decision_request, "overrun")
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER attempt_usage_update")
        db.execute("UPDATE attempt_usage SET payload='private-corrupt-fixture'")
    original = path.read_bytes()
    result = decision_budget_view(path, as_of=datetime.now(timezone.utc))
    assert result == {
        "status": "unavailable",
        "reason": "decision_usage_authority_invalid",
        "error_type": "ValueError",
    }
    assert "private-corrupt-fixture" not in json.dumps(result)
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["overrun", "unknown"])
def test_operations_dashboard_renders_verified_breach_and_unknown_total(
    tmp_path, monkeypatch, decision_request, kind
):
    from streamlit.testing.v1 import AppTest

    output = tmp_path / "outputs"
    capture(output / "decisions.sqlite", decision_request, kind)
    monkeypatch.setenv("AIQUANT_OUTPUT_DIR", str(output))
    reset_config()
    try:
        ui = AppTest.from_file(str(Path("src/dashboard.py").resolve())).run(timeout=20)
        ui.sidebar.radio[0].set_value("Operations").run(timeout=20)
        assert not ui.exception
        metrics = {metric.label: metric.value for metric in ui.metric}
        assert set(metrics) == {
            "Known reported cost",
            "Total reported cost",
            "Conservative budget charge",
        }
        if kind == "overrun":
            assert metrics["Total reported cost"] == "USD 0.010001"
            assert any("breach blocks further admission" in error.value for error in ui.error)
        else:
            assert metrics["Total reported cost"] == "Unknown"
            assert not ui.error
            assert any("1 with uncertain usage" in caption.value for caption in ui.caption)
    finally:
        reset_config()
