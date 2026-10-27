from datetime import datetime, timezone

import pytest

from src.data.calendars import sessions


def test_actual_single_session():
    result = sessions('2026-03-06','2026-03-06',calendar='XNYS')
    assert result == {'2026-03-06':(datetime(2026,3,6,14,30,tzinfo=timezone.utc),datetime(2026,3,6,21,tzinfo=timezone.utc))}


@pytest.mark.parametrize('day',['2026-03-07','2026-03-08','2026-12-25'])
def test_actual_single_closed_date_is_empty(day):
    assert sessions(day,day,calendar='XNYS') == {}


def test_weekend_boundary_and_actual_dst():
    result = sessions('2026-03-07','2026-04-08',calendar='XNYS')
    assert min(result) == '2026-03-09'
    assert result['2026-03-09'] == (datetime(2026,3,9,13,30,tzinfo=timezone.utc),datetime(2026,3,9,20,tzinfo=timezone.utc))
    span = sessions('2026-03-06','2026-03-09',calendar='XNYS')
    assert list(span) == ['2026-03-06','2026-03-09']
    assert span['2026-03-06'][1].hour == 21 and span['2026-03-09'][1].hour == 20


@pytest.mark.parametrize('calendar',['XNYS','24/7'])
def test_reverse_calendar_range_rejected(calendar):
    with pytest.raises(ValueError,match='start must not follow end'):
        sessions('2026-03-09','2026-03-06',calendar=calendar)
