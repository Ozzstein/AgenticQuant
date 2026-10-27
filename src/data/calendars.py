"""Actual US exchange sessions and continuous UTC crypto bars."""
from datetime import date, datetime, timedelta, timezone


def sessions(start: str, end: str, *, calendar: str) -> dict[str, tuple[datetime, datetime]]:
    """Inclusive session labels; never insert prices for closed/missing sessions."""
    first_label, last_label = date.fromisoformat(start), date.fromisoformat(end)
    if first_label > last_label:
        raise ValueError("calendar start must not follow end")
    if calendar == "24/7":
        first = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        last = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        result = {}
        while first <= last:
            result[first.date().isoformat()] = (first, first + timedelta(days=1))
            first += timedelta(days=1)
        return result
    if calendar not in ("XNYS", "XNAS"):
        raise ValueError("unsupported market calendar")
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise RuntimeError("exchange-calendars is required for US session validation") from exc
    # The provider's bounds are sessions, not arbitrary requested dates. A
    # single holiday/weekend may have no sessions and still be a valid query.
    exchange = xcals.get_calendar("XNYS", start=(first_label-timedelta(days=14)).isoformat(),
                                  end=(last_label+timedelta(days=14)).isoformat())
    labels = exchange.sessions_in_range(start, end)
    return {label.date().isoformat(): (exchange.session_open(label).to_pydatetime(),
                                     exchange.session_close(label).to_pydatetime()) for label in labels}
