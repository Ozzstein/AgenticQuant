"""Scope arithmetic and real archive rejection; no positive vendor proof fixture."""

from datetime import datetime, timedelta, timezone

import pytest

from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import EvidenceRecord
from src.domain.portfolio import ActionCoverage
from src.operations.income import _parsed_capture, require_coverage_union
from src.research.registry import RegistryError

pytest_plugins = ["tests.test_operating_dividend_capture"]

T = datetime(2026, 10, 26, tzinfo=timezone.utc)


def row(start, end, **changes):
    return ActionCoverage(
        coverage_id=str(start),
        instrument_ids=("equity:x",),
        venue="fixture",
        scope="custodial-credit-ledger",
        account_id="paper:x",
        started_at=T + timedelta(hours=start),
        ended_at=T + timedelta(hours=end),
        available_at=T + timedelta(days=1),
        event_types=("cash_dividend_payments",),
        completeness="complete",
        gaps=(),
        source_id="untrusted-fixture",
        source_hash="0" * 64,
    ).model_copy(update=changes)


def require(rows):
    return require_coverage_union(
        rows,
        instrument_id="equity:x",
        account_id="paper:x",
        started_at=T,
        ended_at=T + timedelta(hours=6),
        as_of=T + timedelta(days=1),
        scope="custodial-credit-ledger",
        event_type="cash_dividend_payments",
    )


def test_coverage_interval_arithmetic_does_not_skip_holes_or_wrong_accounts():
    # Pure interval arithmetic only: these untrusted rows cannot enter the source
    # verifier. The separate test below proves that boundary rejects them.
    require((row(0, 2), row(2, 4), row(4, 6)))
    require((row(-1, 3), row(2, 7)))
    for rows in (
        (),
        (row(0, 2), row(3, 6)),
        (row(0, 6, account_id="paper:other"),),
        (row(0, 6, event_types=("cash_dividend_terms",)),),
    ):
        with pytest.raises(RegistryError, match="missing complete"):
            require(rows)
    for change in (
        {"completeness": "unknown"},
        {"gaps": ("missing page",)},
        {"available_at": T + timedelta(days=2)},
    ):
        with pytest.raises(RegistryError, match="unavailable, partial, or unknown"):
            require((row(0, 6), row(1, 2, **change)))


@pytest.mark.parametrize(
    "provider,version,kind",
    [
        ("attestation", "1", "income"),
        ("operating", "operating-wire-2", "operating_raw"),
        ("operating", "future-parser", "operating_dividend_payments"),
    ],
)
def test_archived_normalized_complete_claim_cannot_authorize_income(
    tmp_path, provider, version, kind
):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    payload = {
        "action_coverage": [row(0, 6).model_dump(mode="json")],
        "income_authorities": [{"role": "coverage", "complete": True}],
    }
    raw = canonical_bytes(payload)
    record = EvidenceRecord(
        evidence_id="claimed",
        provider=provider,
        provider_version=version,
        source_uri="test://untrusted-claim",
        raw_hash=digest(raw),
        ingested_at=T,
        available_at=T,
        requested_as_of=T,
        availability_basis="documented",
        quality="usable",
    )
    store.put(record, raw, kind=kind, logical_key="claimed", normalized=payload)
    with pytest.raises(RegistryError, match="no supported independent native parser"):
        _parsed_capture(store, record.evidence_id, as_of=T + timedelta(days=1))


def test_actual_native_capture_cannot_certify_full_operating_income(capture, tmp_path):
    from types import SimpleNamespace

    from src.domain.portfolio import AccountSpec
    from src.operations.income import INCOME_KINDS, verify_income_provenance
    from src.research.evidence import BoundArtifact, SourceBinding
    from src.research.registry import ResearchRegistry, canonical
    from tests.test_operating_dividend_capture import run

    f = capture
    _, proof, parsed = run(f)
    assert _parsed_capture(f.store, proof.record.evidence_id, as_of=f.now[0])[1] == parsed
    values = {kind: () for kind in INCOME_KINDS}
    values["action_coverage"] = tuple(
        ActionCoverage.model_validate(r) for r in parsed["action_coverage"]
    )
    inputs = SimpleNamespace(
        **values,
        income_policy="source-confirmed-paper-distribution-1",
        context=f.context.model_copy(update={"as_of": f.now[0]}),
        account=AccountSpec(
            account_id=f.context.account_id,
            mode="paper",
            base_currency="USD",
            initial_cash=100,
            opened_at=f.started_at,
        ),
        snapshot=SimpleNamespace(bars=()),
        metadata=(SimpleNamespace(instrument=f.instrument),),
        content_hash=lambda: "0" * 64,
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
                        evidence_id=proof.record.evidence_id,
                        pointer="/" + kind,
                        available_at=proof.record.available_at,
                    ),
                ),
            )
        )
        for kind in INCOME_KINDS
    }
    with pytest.raises(RegistryError, match="missing complete cash_dividend_terms"):
        verify_income_provenance(
            registry,
            f.store,
            inputs,
            references,
            income_policy=inputs.income_policy,
            required_started_at=f.started_at,
            required_ended_at=f.now[0],
        )


@pytest.mark.parametrize("account", [False, True])
def test_closed_refresh_dispatch_preserves_actual_native_gaps(capture, monkeypatch, account):
    from src.data.adapters.base import ReadHTTP
    from src.operations import operating_fetch
    from src.operations.refresh import FetchTask
    from tests.test_operating_dividend_capture import ACCOUNT

    f = capture
    monkeypatch.setenv("ALPACA_API_KEY", "synthetic-boundary")
    monkeypatch.setenv("ALPACA_API_SECRET", "synthetic-boundary")
    # Only replace HTTP; run the actual closed dispatcher, adapter and parser.
    monkeypatch.setattr(
        operating_fetch,
        "ReadHTTP",
        lambda provider, base, **kw: ReadHTTP(
            provider, base, client=f.adapter.data.client, min_interval=0, **kw
        ),
    )
    arguments = {"started_at": f.started_at.isoformat(), "ended_at": f.ended_at.isoformat()}
    if account:
        arguments.update(account_id=f.context.account_id, provider_account_id=ACCOUNT)
    task = FetchTask(
        task_id="native-income",
        provider="operating",
        operation="alpaca_account_dividend_sources" if account else "alpaca_cash_dividend_sources",
        instrument=f.instrument,
        arguments_json=canonical_bytes(arguments).decode(),
        max_http_requests=3,
    )
    result, requests = operating_fetch.fetch_operating(task, f.context, f.store)
    assert result.status == "available"
    assert requests == (2 if account else 1)
    proof = result.evidence[-1]
    _, parsed = _parsed_capture(f.store, proof.record.evidence_id, as_of=proof.record.available_at)
    assert parsed["native_observations"] and not parsed["dividend_payments"]
    assert all(r["completeness"] == "unknown" for r in parsed["action_coverage"])
