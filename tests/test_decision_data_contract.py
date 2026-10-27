"""Actual data-lane EvidenceStore contract tests after lane integration."""
# ruff: noqa: F401, F811

import pytest

from src.decision.evidence import load_desk_sources, source_from_passage
from src.domain.decisions import digest
from tests.test_decision_contracts import decision_context, decision_instrument, decision_source

store_module = pytest.importorskip(
    "src.data.evidence.store", reason="data-lane contract not integrated in this isolated lane"
)
service_module = pytest.importorskip(
    "src.data.evidence.service", reason="data-lane role service not integrated"
)


def test_actual_store_full_document_and_exact_quote(tmp_path, decision_context, decision_source):
    store = store_module.EvidenceStore(tmp_path / "evidence.sqlite")
    document = "Full prefix. " + decision_source.text + " Full suffix."
    raw = b'{"provider":"fixture","original":"preserved"}'
    record = decision_source.record.model_copy(
        update={"raw_hash": digest(raw.decode()), "requested_as_of": decision_context.as_of}
    )
    store.put(record, raw, kind="transcript", logical_key="fixture-quarter", document=document)
    passage = store.passage(
        record.evidence_id, 13, 13 + len(decision_source.text), as_of=decision_context.as_of
    )
    source = source_from_passage(store, passage, as_of=decision_context.as_of, kind="transcript")
    assert source.text == decision_source.text and source.record.raw_hash == digest(raw.decode())
    assert source.document_hash == digest(document)
    assert store.document(record.evidence_id, as_of=decision_context.as_of).text == document


def test_actual_role_service_into_agent_sources(
    tmp_path, decision_context, decision_instrument, decision_source
):
    store = store_module.EvidenceStore(tmp_path / "roles.sqlite")
    for kind in ("news", "transcript", "technical", "filing"):
        text = kind + ": " + decision_source.text
        record = decision_source.record.model_copy(
            update={
                "evidence_id": "actual-" + kind,
                "raw_hash": digest(text),
                "requested_as_of": decision_context.as_of,
            }
        )
        store.put(record, text.encode(), kind=kind, logical_key=kind, document=text)
    sources, coverage = load_desk_sources(
        service_module.EvidenceService(store), decision_instrument, decision_context
    )
    assert {s.kind for s in sources} == {"news", "transcript", "technical", "fundamental"}
    assert len(sources) == 4
    assert coverage["sentiment"]["news"].status == "available"
    assert coverage["fundamental"]["ratios"].status == "missing"
