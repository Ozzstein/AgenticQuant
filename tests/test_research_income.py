"""Registered income composition, separated from actual source certification."""

from datetime import timedelta

import pytest

from src.core.backtester import PortfolioReplay, canonical_hash
from src.data.evidence.store import EvidenceStore
from src.domain.common import EvidenceRecord
from src.domain.market import MarketBar, MarketSnapshot
from src.domain.portfolio import ActionCoverage
from src.research.datasets import ResearchInputs
from src.research.evidence import BoundArtifact, SourceBinding, expected_artifacts
from src.research.income import RESEARCH_INCOME_KINDS, verify_research_income_provenance
from src.research.registry import RegistryError, ResearchRegistry, canonical
from tests.test_dividend_accounting import replay_dividends
from tests.test_research_inputs import research_inputs

pytest_plugins = ("tests.test_operating_dividend_capture",)


def income_inputs():
    raw = replay_dividends()
    records, bars = [], []
    for bar in raw["bars"]:
        record = EvidenceRecord(
            evidence_id=bar.event_id,
            provider="synthetic-test-only",
            provider_version="1",
            source_uri="test://income-composition",
            raw_hash="a" * 64,
            ingested_at=bar.closed_at,
            available_at=bar.closed_at,
            observed_at=bar.closed_at,
            requested_as_of=raw["context"].as_of,
            availability_basis="documented",
            quality="usable",
            instrument_ids=(bar.instrument.instrument_id,),
        )
        records.append(record)
        bars.append(
            MarketBar(
                instrument=bar.instrument,
                evidence_id=record.evidence_id,
                price_basis="raw",
                market_type="equity",
                session=bar.session.isoformat(),
                **{
                    name: getattr(bar, name)
                    for name in (
                        "opened_at",
                        "closed_at",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "calendar",
                    )
                },
            )
        )
    snapshot = MarketSnapshot(context=raw["context"], bars=tuple(bars), evidence=tuple(records))
    key = snapshot.content_hash()
    inputs = ResearchInputs(
        snapshot=snapshot,
        context=raw["context"].model_copy(update={"data_snapshot_ids": (key,)}),
        account=raw["account"],
        bars=tuple(raw["bars"]),
        metadata=tuple(raw["metadata"]),
        universes=tuple(u.model_copy(update={"data_snapshot_id": key}) for u in raw["universes"]),
        universe_version=raw["universes"][0].universe_version,
        fill_policy=raw["fill_policy"],
        risk_policy=raw["risk_policy"],
        topk=raw["topk"],
        valuation_calendar="XNYS",
        samples_per_year=252,
        income_policy="source-confirmed-replay-distribution-1",
        cash_dividends=tuple(raw["cash_dividends"]),
        dividend_distributions=tuple(raw["dividend_distributions"]),
        synthetic=True,
    )
    return inputs, [signal.model_copy(update={'data_snapshot_id':key}) for signal in raw["signals"]]


def test_legacy_identity_is_unchanged_but_missing_income_blocks_certification(tmp_path):
    inputs = research_inputs()
    payload = inputs.model_dump(mode="json")
    assert not set(("income_policy", *RESEARCH_INCOME_KINDS)) & set(payload)
    assert inputs.content_hash() == canonical_hash(payload)
    assert ResearchInputs.model_validate_json(inputs.model_dump_json()) == inputs
    with pytest.raises(RegistryError, match="not certifiable"):
        verify_research_income_provenance(
            ResearchRegistry(tmp_path / "registry"),
            EvidenceStore(tmp_path / "evidence"),
            inputs,
            {},
        )


def test_research_roundtrip_passes_actual_notices_to_own_ledger(tmp_path):
    inputs, signals = income_inputs()
    restored = ResearchInputs.model_validate_json(inputs.model_dump_json())
    assert restored == inputs and restored.content_hash() == inputs.content_hash()
    assert set(RESEARCH_INCOME_KINDS) <= set(expected_artifacts(inputs))
    args = restored.replay_kwargs()
    result = PortfolioReplay(tmp_path / "income.sqlite").run(signals=signals, **args)
    assert result.snapshots[-1].nav == 8880
    assert result.snapshots[-1].fees == 5 and result.snapshots[-1].withholding_taxes == 15
    assert [event["kind"] for event in result.economic_events] == [
        "dividend_entitlement",
        "dividend_payment",
    ]
    assert result.manifest["cash_dividends_hash"] == canonical_hash(
        [t.model_dump(mode="json") for t in inputs.cash_dividends]
    )
    assert inputs.income_replay_kwargs(
        start=inputs.cash_dividends[0].entitlement_at + timedelta(microseconds=1)
    ) == {
        "cash_dividends": [],
        "dividend_distributions": [],
    }
    # Actual accounting success does not authenticate a provider's source fields.
    with pytest.raises(RegistryError, match="missing historical income source"):
        verify_research_income_provenance(
            ResearchRegistry(tmp_path / "registry"),
            EvidenceStore(tmp_path / "evidence"),
            inputs,
            {},
        )


@pytest.mark.parametrize("change", ["policy", "terms", "duplicate"])
def test_research_income_cannot_be_silently_discarded_or_reassociated(change):
    inputs, _ = income_inputs()
    payload = inputs.model_dump(mode="json")
    if change == "policy":
        payload["income_policy"] = None
    elif change == "terms":
        payload["dividend_distributions"][0]["terms_hash"] = "f" * 64
    else:
        payload["dividend_distributions"] *= 2
    with pytest.raises(ValueError):
        ResearchInputs.model_validate(payload)


def test_native_empty_income_arrays_cannot_certify_historical_coverage(capture, tmp_path):
    from tests.test_operating_dividend_capture import run

    _, envelope, parsed = run(capture)
    inputs, _ = income_inputs()
    # Keep the real native source and its declared gap scope; never upgrade it.
    inputs = inputs.model_copy(
        update={
            "context": inputs.context.model_copy(
                update={"as_of": max(inputs.context.as_of, capture.now[0])}
            ),
            "cash_dividends": (),
            "dividend_distributions": (),
            "action_coverage": tuple(
                ActionCoverage.model_validate(row) for row in parsed["action_coverage"]
            ),
        }
    )
    registry = ResearchRegistry(tmp_path / "registry")
    references = {
        kind + "_id": registry.archive_contract(
            BoundArtifact(
                kind=kind,
                inputs_hash=inputs.content_hash(),
                value_json=canonical(parsed[kind]).decode(),
                sources=(
                    SourceBinding(
                        evidence_id=envelope.record.evidence_id,
                        pointer="/" + kind,
                        available_at=envelope.record.available_at,
                    ),
                ),
            )
        )
        for kind in RESEARCH_INCOME_KINDS
    }
    with pytest.raises(RegistryError, match="coverage"):
        verify_research_income_provenance(registry, capture.store, inputs, references)
