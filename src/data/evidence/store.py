"""Transactional content-addressed evidence storage; no provider credentials.

Raw bytes, complete documents and normalized data live in SQLite, outside the
bounded context supplied to agents. Inserts are idempotent by content and reject
ID reuse. Query defaults preserve revisions and contradictory sources; callers
must resolve conflicts explicitly instead of letting ingestion order decide.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from src.domain.common import (
    EvidenceRecord,
    NonEmptyString,
    Sha256,
    UtcTimestamp,
    VersionedContract,
    _utc,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class EvidenceEnvelope(VersionedContract):
    record: EvidenceRecord
    kind: NonEmptyString
    logical_key: NonEmptyString
    published_at: UtcTimestamp | None = None
    revised_at: UtcTimestamp | None = None
    document_hash: Sha256 | None = None
    normalized_hash: Sha256 | None = None
    supersedes_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def chronology(self) -> "EvidenceEnvelope":
        available = self.record.available_at
        if available is not None:
            if self.published_at is not None and self.published_at > available:
                raise ValueError("publication is after availability")
            if self.revised_at is not None and self.revised_at > available:
                raise ValueError("revision is after availability")
        return self


class Document(VersionedContract):
    evidence: EvidenceRecord
    text: str
    document_hash: Sha256


class Passage(VersionedContract):
    evidence: EvidenceRecord
    document_hash: Sha256
    start: int
    end: int
    text: str

    @model_validator(mode="after")
    def exact_offsets(self) -> "Passage":
        if self.start < 0 or self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("passage offsets must describe its exact Unicode text")
        return self

    @property
    def citation(self) -> str:
        return f"evidence:{self.evidence.evidence_id}#chars={self.start}:{self.end}"


class EvidenceQuery(VersionedContract):
    as_of: UtcTimestamp
    status: Literal["available", "missing", "ineligible", "conflicting"]
    records: tuple[EvidenceEnvelope, ...]
    excluded: tuple[tuple[str, str], ...] = ()


class EvidenceStore:
    """An owned SQLite path, safe for concurrent readers and append-only writers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS blobs (
                    hash TEXT PRIMARY KEY, payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, logical_key TEXT NOT NULL,
                    envelope TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS evidence_lookup ON evidence(kind, logical_key);
                CREATE TABLE IF NOT EXISTS snapshots (
                    hash TEXT PRIMARY KEY, payload BLOB NOT NULL);
                CREATE TRIGGER IF NOT EXISTS immutable_blob_update BEFORE UPDATE ON blobs
                    BEGIN SELECT RAISE(ABORT, 'immutable blob'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_blob_delete BEFORE DELETE ON blobs
                    BEGIN SELECT RAISE(ABORT, 'immutable blob'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_evidence_update BEFORE UPDATE ON evidence
                    BEGIN SELECT RAISE(ABORT, 'immutable evidence'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_evidence_delete BEFORE DELETE ON evidence
                    BEGIN SELECT RAISE(ABORT, 'immutable evidence'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_snapshot_update BEFORE UPDATE ON snapshots
                    BEGIN SELECT RAISE(ABORT, 'immutable snapshot'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_snapshot_delete BEFORE DELETE ON snapshots
                    BEGIN SELECT RAISE(ABORT, 'immutable snapshot'); END;
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def put(self, record: EvidenceRecord, raw: bytes, *, kind: str,
            logical_key: str, document: str | None = None,
            normalized: Any = None, published_at: datetime | None = None,
            revised_at: datetime | None = None, supersedes_id: str | None = None) -> EvidenceEnvelope:
        if digest(raw) != record.raw_hash:
            raise ValueError("raw payload hash mismatch")
        if record.requested_as_of is None:
            raise ValueError("ingestion must record an explicit requested cutoff")
        blobs = {record.raw_hash: raw}
        document_hash = None
        normalized_hash = None
        if document is not None:
            encoded = document.encode("utf-8")
            document_hash = digest(encoded)
            blobs[document_hash] = encoded
        if normalized is not None:
            encoded = canonical_bytes(normalized)
            normalized_hash = digest(encoded)
            blobs[normalized_hash] = encoded
        envelope = EvidenceEnvelope(record=record, kind=kind, logical_key=logical_key,
                                    published_at=published_at, revised_at=revised_at,
                                    document_hash=document_hash, normalized_hash=normalized_hash,
                                    supersedes_id=supersedes_id)
        serialized = envelope.model_dump_json()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT envelope FROM evidence WHERE id=?", (record.evidence_id,)).fetchone()
            if old and old[0] != serialized:
                raise ValueError("evidence ID is immutable; create a new revision ID")
            if supersedes_id:
                previous = db.execute("SELECT envelope FROM evidence WHERE id=?", (supersedes_id,)).fetchone()
                if previous is None:
                    raise ValueError("superseded evidence is missing")
                prior = EvidenceEnvelope.model_validate_json(previous[0])
                if (prior.kind, prior.logical_key, prior.record.provider) != (kind, logical_key, record.provider):
                    raise ValueError("revision cannot supersede a different fact/provider")
                if prior.record.available_at is None or record.available_at is None or record.available_at < prior.record.available_at:
                    raise ValueError("revision availability must not move backwards")
            for key, payload in blobs.items():
                db.execute("INSERT OR IGNORE INTO blobs VALUES (?, ?)", (key, payload))
            db.execute("INSERT OR IGNORE INTO evidence VALUES (?, ?, ?, ?)",
                       (record.evidence_id, kind, logical_key, serialized))
        return envelope

    def get(self, evidence_id: str, *, as_of: datetime | None = None) -> EvidenceEnvelope:
        with self._connect() as db:
            row = db.execute("SELECT envelope FROM evidence WHERE id=?", (evidence_id,)).fetchone()
        if row is None:
            raise KeyError(evidence_id)
        envelope = EvidenceEnvelope.model_validate_json(row[0])
        if as_of is not None:
            envelope.record.require_available(as_of)
        return envelope

    def blob(self, content_hash: str) -> bytes:
        with self._connect() as db:
            row = db.execute("SELECT payload FROM blobs WHERE hash=?", (content_hash,)).fetchone()
        if row is None:
            raise KeyError(content_hash)
        payload = bytes(row[0])
        if digest(payload) != content_hash:
            raise ValueError("stored blob integrity failure")
        return payload

    def raw(self, evidence_id: str) -> bytes:
        return self.blob(self.get(evidence_id).record.raw_hash)

    def normalized(self, evidence_id: str, *, as_of: datetime) -> Any:
        envelope = self.get(evidence_id, as_of=as_of)
        if envelope.normalized_hash is None:
            raise ValueError("evidence has no normalized payload")
        return json.loads(self.blob(envelope.normalized_hash))

    def document(self, evidence_id: str, *, as_of: datetime) -> Document:
        envelope = self.get(evidence_id, as_of=as_of)
        if envelope.document_hash is None:
            raise ValueError("evidence has no complete document")
        return Document(evidence=envelope.record, document_hash=envelope.document_hash,
                        text=self.blob(envelope.document_hash).decode("utf-8"))

    def passage(self, evidence_id: str, start: int, end: int, *, as_of: datetime) -> Passage:
        doc = self.document(evidence_id, as_of=as_of)
        if not 0 <= start < end <= len(doc.text):
            raise ValueError("passage is outside complete document")
        return Passage(evidence=doc.evidence, document_hash=doc.document_hash,
                       start=start, end=end, text=doc.text[start:end])

    def retrieve(self, evidence_ids: tuple[str, ...] | list[str], *, as_of: datetime,
                 query: str = "", max_chars: int = 6000, max_passages: int = 8,
                 passage_chars: int = 1200) -> tuple[Passage, ...]:
        """Deterministic lexical ranking of exact spans, bounded across ALL sources.

        This is retrieval, not claim validation. Callers retain the query's full
        evidence IDs/conflict state, even when context cannot fit every document.
        """
        if not 0 < max_chars <= 100_000 or not 0 < max_passages <= 100 or not 0 < passage_chars <= 10_000:
            raise ValueError("invalid retrieval budget")
        terms = set(re.findall(r"\w+", query.casefold()))
        candidates = []
        for evidence_id in dict.fromkeys(evidence_ids):
            doc = self.document(evidence_id, as_of=as_of)
            for start in range(0, len(doc.text), passage_chars):
                text = doc.text[start:start + passage_chars]
                score = sum(text.casefold().count(term) for term in terms)
                candidates.append((-score, evidence_id, start, doc))
        candidates.sort(key=lambda value: value[:3])
        result = []
        remaining = max_chars
        for _, evidence_id, start, doc in candidates:
            if not remaining or len(result) == max_passages:
                break
            end = min(start + passage_chars, start + remaining, len(doc.text))
            result.append(Passage(evidence=doc.evidence, document_hash=doc.document_hash,
                                  start=start, end=end, text=doc.text[start:end]))
            remaining -= end - start
        return tuple(result)

    def query(self, *, as_of: datetime, kind: str | None = None,
              logical_key: str | None = None, instrument_id: str | None = None,
              current_revisions_only: bool = False) -> EvidenceQuery:
        cutoff = _utc(as_of)
        sql, args = "SELECT envelope FROM evidence WHERE 1=1", []
        for column, value in (("kind", kind), ("logical_key", logical_key)):
            if value is not None:
                sql += f" AND {column}=?"
                args.append(value)
        with self._connect() as db:
            rows = db.execute(sql, args).fetchall()
        records, excluded = [], []
        for (payload,) in rows:
            envelope = EvidenceEnvelope.model_validate_json(payload)
            if instrument_id and instrument_id not in envelope.record.instrument_ids:
                continue
            try:
                envelope.record.require_available(cutoff)
            except ValueError as exc:
                excluded.append((envelope.record.evidence_id, str(exc)))
            else:
                records.append(envelope)
        superseded = {e.supersedes_id for e in records}
        current = [e for e in records if e.record.evidence_id not in superseded]
        values: dict[tuple, set] = {}
        for envelope in current:
            key = (envelope.kind, envelope.logical_key, envelope.record.instrument_ids)
            values.setdefault(key, set()).add(envelope.normalized_hash or envelope.document_hash or envelope.record.raw_hash)
        conflict = any(len(hashes) > 1 for hashes in values.values())
        if current_revisions_only:
            records = current
        records.sort(key=lambda e: (e.record.available_at, e.record.evidence_id))
        status = "conflicting" if conflict else "available" if records else "ineligible" if excluded else "missing"
        return EvidenceQuery(as_of=cutoff, status=status, records=tuple(records), excluded=tuple(sorted(excluded)))

    def save_snapshot(self, payload: dict) -> str:
        raw = canonical_bytes(payload)
        key = digest(raw)
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?)", (key, raw))
        return key

    def snapshot(self, snapshot_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT payload FROM snapshots WHERE hash=?", (snapshot_id,)).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        if digest(bytes(row[0])) != snapshot_id:
            raise ValueError("snapshot integrity failure")
        return json.loads(row[0])
