"""Temporal and provenance invariants for shared domain contracts."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.domain.common import EvidenceRecord, Instrument, RunContext

UTC = timezone.utc


def evidence(**overrides: object) -> EvidenceRecord:
    """Build a source published in 2020 and backfilled in 2026."""
    fields: dict[str, object] = {
        "evidence_id": "filing-1",
        "provider": "fixture",
        "provider_version": "1",
        "source_uri": "https://example.test/filing-1",
        "raw_hash": "a" * 64,
        "available_at": datetime(2020, 1, 3, 12, tzinfo=UTC),
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
        "quality": "usable",
        "availability_basis": "documented",
    }
    fields.update(overrides)
    return EvidenceRecord.model_validate(fields)


def test_historical_backfill_uses_publication_not_ingestion_time() -> None:
    """A later download does not invalidate documented past availability."""
    evidence().require_available(datetime(2020, 1, 3, 12, tzinfo=UTC))


def test_future_evidence_cannot_enter_a_past_decision() -> None:
    """A single second of future information is still leakage."""
    with pytest.raises(ValueError, match="unavailable"):
        evidence().require_available(datetime(2020, 1, 3, 11, 59, 59, tzinfo=UTC))


@pytest.mark.parametrize("basis", ["inferred", "unknown"])
def test_a_plausible_date_does_not_establish_provenance(basis: str) -> None:
    with pytest.raises(ValueError, match="documented"):
        evidence(availability_basis=basis).require_available(datetime(2021, 1, 1, tzinfo=UTC))


def test_missing_date_is_not_neutral_evidence() -> None:
    with pytest.raises(ValueError, match="documented"):
        evidence(available_at=None).require_available(datetime(2021, 1, 1, tzinfo=UTC))


@pytest.mark.parametrize("quality", ["unknown", "missing", "stale", "invalid"])
def test_unusable_evidence_cannot_pass_a_historical_gate(quality: str) -> None:
    with pytest.raises(ValueError, match="quality"):
        evidence(quality=quality).require_available(datetime(2021, 1, 1, tzinfo=UTC))


def test_later_requested_vintage_cannot_be_reused_for_an_earlier_cutoff() -> None:
    record = evidence(requested_as_of=datetime(2022, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="later historical cutoff"):
        record.require_available(datetime(2021, 1, 1, tzinfo=UTC))


def test_timestamps_require_timezones_in_storage_and_queries() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        evidence(ingested_at=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="timezone"):
        evidence().require_available(datetime(2021, 1, 1))


def test_equal_instants_have_equal_hashes_and_persisted_replay() -> None:
    original = evidence()
    offset = evidence(available_at=datetime(2020, 1, 3, 14, tzinfo=timezone(timedelta(hours=2))))
    restored = EvidenceRecord.model_validate_json(original.model_dump_json())
    assert original.content_hash() == offset.content_hash() == restored.content_hash()
    assert original.content_hash() != evidence(raw_hash="b" * 64).content_hash()


def test_contract_cannot_be_mutated_or_gain_undeclared_authority() -> None:
    record = evidence()
    with pytest.raises(ValidationError, match="frozen"):
        record.quality = "invalid"
    with pytest.raises(ValidationError, match="Extra inputs"):
        evidence(approved_to_trade=True)


def test_run_context_requires_explicit_clock_and_config_identity() -> None:
    fields = {
        "run_id": "replay-1",
        "mode": "backtest",
        "code_version": "test-commit",
        "config_hash": "c" * 64,
    }
    with pytest.raises(ValidationError, match="as_of"):
        RunContext.model_validate(fields)
    with pytest.raises(ValidationError, match="config_hash"):
        RunContext.model_validate({**fields, "as_of": "2020-01-01T00:00:00Z", "config_hash": ""})


def test_symbol_alone_is_not_a_security_master_identity() -> None:
    with pytest.raises(ValidationError, match="instrument_id"):
        Instrument(symbol="ABC", asset_class="equity", quote_currency="USD")
    with pytest.raises(ValidationError, match="instrument_id"):
        Instrument(instrument_id=" ", symbol="ABC", asset_class="equity", quote_currency="USD")
