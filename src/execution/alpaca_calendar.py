"""Original native US regular-session authority for Alpaca DAY admission."""

import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.domain.common import _utc
from src.execution.ledger import LedgerError

ZONE = ZoneInfo("America/New_York")


def query(at):
    start = _utc(at).astimezone(ZONE).date()
    return {
        "start": start.isoformat(),
        "end": (start + timedelta(days=14)).isoformat(),
        "date_type": "TRADING",
    }


def next_session(rows, params, at):
    """Select only a supplied native session; no weekday/close synthesis."""
    if not isinstance(rows, list) or not rows or len(rows) > 15:
        raise LedgerError("Alpaca DAY calendar lacks bounded native session coverage")
    start, end = date.fromisoformat(params["start"]), date.fromisoformat(params["end"])
    sessions, dates = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise LedgerError("invalid native calendar row")
        day = date.fromisoformat(row["date"])
        if day.isoformat() != row["date"] or day in dates or not start <= day <= end:
            raise LedgerError("native calendar duplicate/date/query conflict")
        dates.add(day)
        boundaries = []
        for field in ("open", "close"):
            local = datetime.strptime(row["date"] + " " + row[field], "%Y-%m-%d %H:%M")
            if local.strftime("%H:%M") != row[field]:
                raise LedgerError("native calendar requires exact HH:MM boundaries")
            first, second = local.replace(tzinfo=ZONE, fold=0), local.replace(tzinfo=ZONE, fold=1)
            if (
                first.utcoffset() != second.utcoffset()
                or first.astimezone(timezone.utc).astimezone(ZONE).replace(tzinfo=None) != local
            ):
                raise LedgerError("native calendar clock is ambiguous/nonexistent")
            boundaries.append(first.astimezone(timezone.utc))
        opened, closed = boundaries
        if opened >= closed:
            raise LedgerError("native calendar session interval is invalid")
        if closed > _utc(at):
            sessions.append((closed, row))
    if not sessions:
        raise LedgerError("native calendar has no eligible future regular-session close")
    return min(sessions, key=lambda item: item[0])


def verify(venue, intent, proof):
    from src.execution.alpaca_venue import _hash

    row = venue.ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?", (venue.account_id, proof["source"])
    ).fetchone()
    if row is None:
        raise LedgerError("original Alpaca DAY calendar source missing")
    raw = json.loads(row[0])
    requested, selected = (
        _utc(datetime.fromisoformat(proof[name])) for name in ("requested_at", "selected_at")
    )
    observed = _utc(datetime.fromisoformat(raw["observed_at"]))
    if (
        proof["intent_hash"] != intent.content_hash()
        or proof["source"] != "alpaca-http:" + _hash(raw)
        or raw["method"] != "GET"
        or raw["path"] != "/v2/calendar"
        or raw["params"] != query(requested)
        or raw.get("request") is not None
        or raw["status"] != 200
        or raw.get("complete") is not True
        or raw.get("content_encoding") not in ("", "identity")
        or not intent.decision_at <= requested <= observed <= selected < intent.expires_at
    ):
        raise LedgerError("original Alpaca DAY source/intent/clock binding differs")
    expiry, session = next_session(json.loads(raw["response"]), raw["params"], selected)
    if (
        expiry != intent.expires_at
        or proof["expires_at"] != expiry.isoformat()
        or proof["session"] != session
    ):
        raise LedgerError("Alpaca DAY expiry differs from original native regular session")
    return proof


def read(venue, intent):
    from src.execution.alpaca_venue import _hash

    if intent.tif != "DAY":
        return None
    row = venue.ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (venue.account_id, "alpaca-day-calendar:" + intent.intent_id),
    ).fetchone()
    if row is None:
        raise LedgerError("Alpaca DAY order lacks original native calendar authority")
    proof = verify(venue, intent, json.loads(row[0]))
    owner = venue.ledger.db.execute(
        "SELECT owner FROM execution_books WHERE account=?", (venue.account_id,)
    ).fetchone()
    if owner is None or owner[0] not in ("direct", "lean"):
        raise LedgerError("original DAY source lacks execution ownership")
    key = ("alpaca" if owner[0] == "direct" else "lean") + "-submit-attempt:" + intent.intent_id
    attempt = venue.ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?", (venue.account_id, key)
    ).fetchone()
    if attempt is None:
        raise LedgerError("original DAY source lacks its submission claim")
    claim = json.loads(attempt[0])
    if (
        claim["intent_hash"] != intent.content_hash()
        or claim.get("day_calendar_hash") != _hash(proof)
        or not _utc(datetime.fromisoformat(proof["selected_at"]))
        <= _utc(datetime.fromisoformat(claim["at"]))
        < intent.expires_at
    ):
        raise LedgerError("original DAY calendar differs from its immutable submission claim")
    return proof


def bind(venue, intent, as_of):
    from src.execution.alpaca_venue import _json

    if intent.tif != "DAY":
        return None
    now = venue._now()
    rule = venue._rules.get(intent.instrument.instrument_id)
    if (
        rule is None
        or rule.instrument != intent.instrument
        or intent.instrument.asset_class not in ("equity", "etf")
    ):
        raise LedgerError("native DAY calendar requires the exact configured US equity")
    if intent.account_id != venue.account_id or not 0 <= (now - _utc(as_of)).total_seconds() <= 30:
        raise LedgerError("native DAY admission account/current cutoff differs")
    if intent.expires_at is None or now >= intent.expires_at:
        raise LedgerError("native DAY order already expired")
    key = "alpaca-day-calendar:" + intent.intent_id
    row = venue.ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?", (venue.account_id, key)
    ).fetchone()
    if row:
        proof = verify(venue, intent, json.loads(row[0]))
        if _utc(datetime.fromisoformat(proof["selected_at"])) > now:
            raise LedgerError("original native DAY proof is future")
        return proof
    rows, _, source = venue._request("GET", "/v2/calendar", params=query(now))
    selected = venue._now()
    expiry, session = next_session(rows, query(now), selected)
    proof = {
        "intent_hash": intent.content_hash(),
        "source": source,
        "requested_at": now.isoformat(),
        "selected_at": selected.isoformat(),
        "expires_at": expiry.isoformat(),
        "session": session,
    }
    verify(venue, intent, proof)
    with venue.ledger.atomic():
        venue.ledger.record_event(venue.account_id, key, _json(proof))
    return proof
