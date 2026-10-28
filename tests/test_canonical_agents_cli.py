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
    assert kwargs["source_coverage"]
    assert all(not value["delivery_complete"] and not value["evidence_ids"]
               for kinds in kwargs["source_coverage"].values() for value in kinds.values())
    assert json.loads(output.read_text())["decision"]["action"] == action.value


def test_input_omission_fails_with_nonzero():
    result = CliRunner().invoke(app, ["analyze", "TEST"])
    assert result.exit_code != 0


def test_cli_forwards_exact_partial_delivery_and_preserves_short_contrary_source(tmp_path):
    from src.domain.common import EvidenceRecord
    from src.domain.decisions import digest

    req = request()
    store = EvidenceStore(tmp_path / "evidence.db")
    for identity, kind, text in (
        ("a-long", "income_statement", "Synthetic long filing. " * 1000),
        ("z-contrary", "news", "Synthetic contrary news."),
    ):
        record = EvidenceRecord(evidence_id=identity, provider="test", provider_version="fixture-1",
            source_uri="fixture://" + identity, raw_hash=digest(text), ingested_at=req.context.as_of,
            available_at=req.context.as_of, requested_as_of=req.context.as_of,
            availability_basis="documented", quality="usable", instrument_ids=(req.instrument.instrument_id,))
        store.put(record, text.encode(), kind=kind, logical_key=identity, document=text)
    input_path, output = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(req.model_dump_json())
    decision = AgentDecision(instrument=req.instrument, context=req.context,
                             action=AgentAction.ABSTAIN, reasoning="CLI boundary fixture")
    with patch("src.agents.graph.analyze_instrument", return_value=decision) as analyze:
        result = CliRunner().invoke(app, ["analyze", "--input", str(input_path),
            "--evidence", str(store.path), "--output", str(output)])
    assert result.exit_code == 2, result.output
    kwargs = analyze.call_args.kwargs
    assert {s.record.evidence_id for s in kwargs["sources"]} == {"a-long", "z-contrary"}
    coverage = kwargs["source_coverage"]
    assert json.loads(output.read_text())["coverage"] == coverage
    assert coverage["fundamental"]["income_statement"]["partial_ids"] == ["a-long"]
    assert not coverage["fundamental"]["income_statement"]["delivery_complete"]
    assert coverage["sentiment"]["news"]["evidence_ids"] == ["z-contrary"]


def test_canonical_delegates_real_data_and_decision_help():
    for command in ("data", "decision"):
        result = CliRunner().invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--store" in result.output
