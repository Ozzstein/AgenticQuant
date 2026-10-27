"""Explicit agent input contract and canonical CLI preserve non-order actions."""
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from scripts.run_agents import DeskAnalysisInput
from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, RunContext
from src.domain.decisions import AgentAction, AgentDecision
from src.utils.cli import app


def request():
    return DeskAnalysisInput(instrument=Instrument(instrument_id="equity:1", symbol="TEST", asset_class="equity", quote_currency="USD"),
                             context=RunContext(run_id="desk", mode="paper", as_of=datetime(2026,1,1,tzinfo=timezone.utc),
                                                config_hash="0"*64, code_version="test", account_id="paper:test"))


@pytest.mark.parametrize("action", [AgentAction.HOLD, AgentAction.ABSTAIN, AgentAction.VETO, AgentAction.EXIT])
def test_canonical_analysis_passes_explicit_context_and_preserves_action(tmp_path, action):
    req = request()
    evidence = tmp_path/"evidence.db"
    EvidenceStore(evidence)
    input_path, output = tmp_path/"input.json", tmp_path/"decision.json"
    input_path.write_text(req.model_dump_json())
    decision = AgentDecision(instrument=req.instrument, context=req.context, action=action,
                              risk_veto=action==AgentAction.VETO, reasoning="Test boundary only")
    with patch("src.agents.graph.analyze_instrument", return_value=decision) as analyze:
        result = CliRunner().invoke(app, ["analyze", "--input", str(input_path), "--evidence", str(evidence), "--output", str(output)])
    assert result.exit_code == (2 if action in (AgentAction.ABSTAIN,AgentAction.VETO) else 0), result.output
    args, kwargs = analyze.call_args
    assert args == (req.instrument,req.context)
    assert kwargs["sources"] == () and kwargs["hmm_context"] is None
    assert json.loads(output.read_text())["decision"]["action"] == action.value


def test_input_omission_fails_with_nonzero():
    result = CliRunner().invoke(app, ["analyze", "TEST"])
    assert result.exit_code != 0


def test_canonical_delegates_real_data_and_decision_help():
    for command in ("data", "decision"):
        result = CliRunner().invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--store" in result.output
