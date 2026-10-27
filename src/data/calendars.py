"""Actual US exchange sessions and continuous UTC crypto bars."""
from datetime import datetime, timedelta, timezone


def sessions(start: str, end: str, *, calendar: str) -> dict[str, tuple[datetime, datetime]]:
    """Inclusive session labels; never insert prices for closed/missing sessions."""
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
    exchange = xcals.get_calendar("XNYS", start=start, end=end)
    labels = exchange.sessions_in_range(start, end)
    return {label.date().isoformat(): (exchange.session_open(label).to_pydatetime(),
                                     exchange.session_close(label).to_pydatetime()) for label in labels}
