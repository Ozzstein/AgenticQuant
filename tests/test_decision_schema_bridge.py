"""Owned decisions survive legacy envelopes without changing their meaning."""

from datetime import datetime, timezone

import pytest

from src.domain.common import Instrument, RunContext
from src.domain.decisions import AgentAction, AgentDecision, ConfidenceSemantics
from src.utils.schemas import AgentReport, AnalysisResult, Decision


@pytest.mark.parametrize("action", [AgentAction.HOLD, AgentAction.ABSTAIN, AgentAction.VETO])
def test_non_liquidating_decisions_round_trip_without_confidence_scaling(action):
    owned = AgentDecision(
        instrument=Instrument(instrument_id="equity:1", symbol="AAPL", asset_class="equity", quote_currency="USD"),
        context=RunContext(run_id="run", mode="paper", as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
                           config_hash="a" * 64, code_version="test"),
        action=action, reasoning="Keep existing exposure; no new orders.",
        risk_veto=action == AgentAction.VETO,
        confidence=ConfidenceSemantics(legacy_confidence_0_100=82.0, legacy_meaning="legacy_unvalidated"),
    )
    envelope = AnalysisResult(ticker="AAPL", decision=Decision.HOLD, confidence=82.0, agent_decision=owned)
    restored = AnalysisResult.model_validate_json(envelope.model_dump_json())
    assert restored.schema_version == "2.0"
    assert restored.agent_decision.action == action
    assert restored.agent_decision.new_exposure_blocked
    assert restored.confidence == 82.0
    assert restored.confidence_semantics == "legacy_unvalidated"
    assert restored.agent_decision.confidence.analyst_conviction is None
    assert restored.agent_decision.confidence.calibrated_task_accuracy is None


def test_legacy_numeric_confidence_cannot_claim_calibration():
    with pytest.raises(ValueError):
        AgentReport(agent_name="test", decision="BUY", confidence=80, reasoning="test",
                    confidence_semantics="calibrated_accuracy")
