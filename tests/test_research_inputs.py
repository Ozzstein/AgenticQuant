"""Actual archived-fixture features, trained models and chronological fills."""
from datetime import timedelta
from decimal import Decimal

import pandas as pd
import pytest

from src.core.factor_backtester import FactorBacktester
from src.core.factor_evaluator import FactorEvaluator
from src.domain.market import FeatureSpec, MarketSnapshot
from src.domain.portfolio import AccountSpec, ExecutionBar, InstrumentRisk, UniverseEligibility
from src.research.datasets import ResearchInputs, eligible_features
from src.utils.schemas import FactorDefinition
from tests.data_quant_fixtures import archived_snapshot


def research_inputs():
    snapshot = archived_snapshot(crypto=True, periods=252)
    key = snapshot.content_hash()
    account = AccountSpec(account_id="backtest:research", mode="backtest", base_currency="USD",
                          initial_cash=10000, opened_at=snapshot.bars[0].opened_at-timedelta(days=1))
    context = snapshot.context.model_copy(update={"mode": "backtest", "account_id": account.account_id,
                                                  "data_snapshot_ids": (key,)})
    bars, metadata = [], []
    for bar in snapshot.bars:
        bars.append(ExecutionBar(event_id=bar.evidence_id, instrument=bar.instrument,
                                 **{f: getattr(bar, f) for f in ("opened_at", "closed_at", "open", "high", "low", "close", "volume", "calendar", "session")},
                                 opening_volume=Decimal("100"), calendar_version="synthetic-test-only",
                                 session_open=bar.opened_at, session_close=bar.opened_at+timedelta(days=1)))
        metadata.append(InstrumentRisk(instrument=bar.instrument, sector=bar.instrument.instrument_id,
                                       adv_base=10000000, as_of=bar.opened_at, lot_size=".001"))
    universe = UniverseEligibility(universe_version="test", data_snapshot_id=key,
                                   effective_at=account.opened_at, available_at=account.opened_at,
                                   instrument_ids=tuple(sorted({b.instrument.instrument_id for b in bars})))
    return ResearchInputs(snapshot=snapshot, context=context, account=account, bars=tuple(bars),
                          metadata=tuple(metadata), universes=(universe,), universe_version="test",
                          valuation_calendar="24/7", samples_per_year=365, initial_train_months=1,
                          topk=1, synthetic=True)


def test_requires_explicit_inputs_instead_of_ambient_yahoo():
    with pytest.raises(ValueError, match="ResearchInputs"):
        FactorBacktester().validate(FactorDefinition(name="f", expression="$close"))


def test_exact_interpreter_preserves_membership_and_late_revision():
    inputs = research_inputs()
    specs = (FeatureSpec(name="momentum", expression="$close/Ref($close,3)-1"),)
    full = inputs.materialize(specs)
    assert len(full.features) > 600
    at = inputs.snapshot.bars[300].closed_at
    first = inputs.universes[0]
    narrow = first.model_copy(update={"effective_at": at, "available_at": at, "instrument_ids": (first.instrument_ids[0],)})
    subset = eligible_features(inputs.snapshot, specs, (first, narrow), "test")
    tail = subset.features.loc[subset.features.index.get_level_values(0) >= at]
    assert set(tail.index.get_level_values(1)) == {first.instrument_ids[0]}
    records = list(inputs.snapshot.evidence)
    revised_id = inputs.snapshot.bars[300].evidence_id
    records = [r.model_copy(update={"available_at": inputs.context.as_of}) if r.evidence_id == revised_id else r for r in records]
    revised = MarketSnapshot(context=inputs.snapshot.context, bars=inputs.snapshot.bars, evidence=tuple(records))
    masked = eligible_features(revised, specs, (first,), "test")
    assert len(masked.features) < len(full.features)
    assert not masked.features.index.has_duplicates


def test_real_factor_fit_and_shared_ledger(tmp_path):
    inputs = research_inputs()
    result = FactorBacktester().run(FactorDefinition(name="momentum", expression="$close/Ref($close,3)-1"),
                                    inputs=inputs, ledger_path=tmp_path/"ledger.db", artifact_dir=tmp_path/"models")
    assert result.engine == "chronological-ledger-v1"
    assert result.walk_forward_results and result.trades
    assert list((tmp_path/"models").glob("fold-*.joblib"))
    assert all(pd.Timestamp(t["executed_at"]) > pd.Timestamp(t["decision_at"]) for t in result.trades)
    assert len(result.daily_returns) == len(result.equity_curve)-1
    assert result.manifest["provenance"]["walk_months"] == 6


def test_ic_diagnostics_reject_overlapping_screen_windows():
    inputs = research_inputs()
    dates = sorted({b.closed_at for b in inputs.snapshot.bars})
    factor = FactorDefinition(name="momentum", expression="$close/Ref($close,3)-1")
    args = dict(snapshot=inputs.snapshot, universes=inputs.universes, universe_version="test")
    with pytest.raises(ValueError, match="disjoint"):
        FactorEvaluator().evaluate_factor(factor, stage1=(dates[0], dates[50]), stage2=(dates[30], dates[-1]), **args)
    result = FactorEvaluator().evaluate_factor(factor, stage1=(dates[0], dates[50]), stage2=(dates[50], dates[-1]), **args)
    assert result.stage1_ic is not None
    assert result.stage2_ic is not None


def test_snapshot_substitution_rejected():
    inputs = research_inputs()
    payload = inputs.model_dump(mode="json")
    payload["context"]["data_snapshot_ids"] = ["unrelated"]
    with pytest.raises(ValueError, match="exact market snapshot"):
        ResearchInputs.model_validate(payload)
