"""Actual close boundaries and persisted Decimal subperiods; test values are synthetic."""
import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.core.strategy_tracker import AttributedPerformanceStore
from src.data.calendars import sessions


def record(store,key,start,end,opening,closing,flow=0,available=None):
    store.record_nav('sleeve',started_at=pd.Timestamp(start),ended_at=pd.Timestamp(end),
        available_at=pd.Timestamp(available or end),opening_nav=Decimal(str(opening)),closing_nav=Decimal(str(closing)),
        end_flow=Decimal(str(flow)),source_event_id=key,source_event_hash=('a' if key=='a' else 'b')*64)


def test_overnight_subperiods_actual_dst_closes_and_later_receipt(tmp_path):
    store = AttributedPerformanceStore(tmp_path/'nav.sqlite')
    close = list(sessions('2026-03-06','2026-03-09',calendar='XNYS').values())
    friday,monday = close[0][1],close[1][1]
    assert friday.hour==21 and monday.hour==20
    middle = pd.Timestamp(friday)+pd.Timedelta(minutes=15)
    observed = pd.Timestamp(monday)+pd.Timedelta(minutes=15)
    record(store,'a',friday,middle,100,150,40)
    record(store,'b',middle,monday,150,180,15,observed)
    assert store.daily_returns(as_of=monday,annualization=252).empty
    daily = store.daily_returns(as_of=observed,annualization=252)
    assert len(daily)==1
    row = daily.iloc[0]
    assert row.started_at==friday and row.ended_at==monday and row.available_at==observed
    assert row.period_return==Decimal('.21')
    assert row.source_event_ids==('a','b') and row.source_event_hashes==('a'*64,'b'*64)
    assert row.subperiods[0]['ended_at']==middle.isoformat()
    metrics = store.metrics(as_of=observed,annualization=252)
    assert metrics.loc['sleeve','observations']==1 and metrics.loc['sleeve','complete']
    assert metrics.loc['sleeve','total_return']==pytest.approx(.21)
    assert metrics.loc['sleeve','aggregation_hashes']==(row.aggregation_hash,)


def test_intraday_cycles_do_not_inflate_daily_sample_count(tmp_path):
    store = AttributedPerformanceStore(tmp_path/'intraday.sqlite')
    start = pd.Timestamp('2026-01-01T00:00:00Z')
    nav = Decimal(100)
    for i in range(24):
        end = start+pd.Timedelta(hours=i+1)
        close = nav*Decimal('1.001')
        record(store,str(i),end-pd.Timedelta(hours=1),end,nav,close)
        nav = close
    daily = store.daily_returns(as_of=end,annualization=365)
    assert len(daily)==1 and len(daily.iloc[0].subperiods)==24
    metrics = store.metrics(as_of=end,annualization=365)
    assert metrics.loc['sleeve','observations']==1 and np.isnan(metrics.loc['sleeve','sharpe'])
    assert metrics.loc['sleeve','total_return']==pytest.approx(float(Decimal('1.001')**24-1))


def test_crossing_midnight_is_never_split_or_relabelled(tmp_path):
    store = AttributedPerformanceStore(tmp_path/'crossing.sqlite')
    record(store,'a','2026-01-01T00:00:00Z','2026-01-02T00:15:00Z',100,101)
    record(store,'b','2026-01-02T00:15:00Z','2026-01-03T00:00:00Z',101,102)
    daily = store.daily_returns(as_of=pd.Timestamp('2026-01-03T00:00:00Z'),annualization=365)
    assert daily.empty and not daily.attrs['coverage']['sleeve']['complete']
    assert daily.attrs['coverage']['sleeve']['expected_periods']==2
    assert daily.attrs['coverage']['sleeve']['unaggregated_source_ids']==('a','b')


def test_twenty_actual_days_after_intraday_compounding_and_early_close(tmp_path):
    store = AttributedPerformanceStore(tmp_path/'days.sqlite')
    dates = [end for _,end in sessions('2025-10-20','2025-11-28',calendar='XNYS').values()]
    assert dates[-1].hour==18  # Actual Friday after Thanksgiving early close.
    nav = Decimal(100)
    for i,(start,end) in enumerate(zip(dates,dates[1:])):
        mid = pd.Timestamp(start)+pd.Timedelta(minutes=15)
        after = nav*Decimal('1.001' if i%2 else '.999')
        record(store,f'{i}:open',start,mid,nav,after)
        close = after*Decimal('1.003')
        record(store,f'{i}:close',mid,end,after,close)
        nav = close
    metrics = store.metrics(as_of=end,annualization=252)
    assert metrics.loc['sleeve','observations']==len(dates)-1
    assert metrics.loc['sleeve','complete'] and np.isfinite(metrics.loc['sleeve','sharpe'])
    assert len(metrics.loc['sleeve','source_event_ids'])==2*(len(dates)-1)


@pytest.mark.parametrize("damage",["gap","return"])
def test_archived_gap_or_wrong_return_fails_closed(tmp_path,damage):
    store = AttributedPerformanceStore(tmp_path/'corrupt.sqlite')
    record(store,'a','2026-01-01T00:00:00Z','2026-01-02T00:00:00Z',100,101)
    record(store,'b','2026-01-02T00:00:00Z','2026-01-03T00:00:00Z',101,102)
    with store._db() as db:
        db.execute('DROP TRIGGER nav_no_update')
        value = json.loads(db.execute("SELECT payload FROM attributed_nav WHERE source_event_id='b'").fetchone()[0])
        if damage == 'gap':
            value['started_at']='2026-01-02T01:00:00+00:00'
        else:
            value['period_return']='9'
        db.execute("UPDATE attributed_nav SET payload=? WHERE source_event_id='b'",(json.dumps(value),))
    with pytest.raises(ValueError,match='chain has a gap|return differs'):
        store.daily_returns(as_of=pd.Timestamp('2026-01-03T00:00:00Z'),annualization=365)


def test_explicit_funded_epochs_reset_samples_without_invented_returns(tmp_path):
    store = AttributedPerformanceStore(tmp_path/'epochs.sqlite')
    start = pd.Timestamp('2026-01-01T00:00:00Z')
    values = dict(strategy='sleeve',at=start,available_at=start,opening_nav=Decimal(100),
                  source_event_id='initial-funding',source_event_hash='c'*64)
    store.start_funded_epoch(**values)
    store.start_funded_epoch(**values)
    with pytest.raises(ValueError,match='rewritten'):
        store.start_funded_epoch(**{**values,'opening_nav':Decimal(101)})
    with pytest.raises(ValueError,match='exact funded snapshot'):
        record(store,'wrong',start,start+pd.Timedelta(days=1),99,100)
    record(store,'a',start,start+pd.Timedelta(days=1),100,101)
    with pytest.raises(ValueError,match='prior zero NAV'):
        store.start_funded_epoch(**{**values,'source_event_id':'premature','at':start+pd.Timedelta(days=1),'available_at':start+pd.Timedelta(days=1)})
    record(store,'b',start+pd.Timedelta(days=1),start+pd.Timedelta(days=2),101,0,-101)
    funded = start+pd.Timedelta(days=4)
    receipt = funded+pd.Timedelta(hours=1)
    store.start_funded_epoch('sleeve',at=funded,available_at=receipt,opening_nav=Decimal(50),
                            source_event_id='actual-refunding',source_event_hash='d'*64)
    before = store.metrics(as_of=receipt,annualization=365)
    assert before.loc['sleeve','observations']==0 and not before.loc['sleeve','complete']
    assert before.loc['sleeve','epoch_id']=='actual-refunding'
    assert np.isnan(before.loc['sleeve','sharpe'])
    record(store,'c',funded,funded+pd.Timedelta(days=1),50,55)
    daily = store.daily_returns(as_of=funded+pd.Timedelta(days=1),annualization=365)
    assert len(daily)==3  # Two real old periods and one real new period, no hiatus rows.
    current = store.metrics(as_of=funded+pd.Timedelta(days=1),annualization=365)
    assert current.loc['sleeve','observations']==1
    assert current.loc['sleeve','total_return']==pytest.approx(.1)
    assert current.loc['sleeve','source_event_ids']==('c',)
    assert current.loc['sleeve','funding_anchor']['source_event_hash']=='d'*64


def test_initial_funding_anchor_is_causal_and_cannot_reset_itself(tmp_path):
    store = AttributedPerformanceStore(tmp_path/'anchor.sqlite')
    at = pd.Timestamp('2026-01-01T00:00:00Z')
    args = dict(strategy='sleeve',at=at,available_at=at+pd.Timedelta(hours=2),opening_nav=Decimal(100),
                source_event_id='funded',source_event_hash='c'*64)
    store.start_funded_epoch(**args)
    assert store.daily_returns(as_of=at,annualization=365).empty
    with pytest.raises(ValueError,match='cannot be reset'):
        store.start_funded_epoch(**{**args,'source_event_id':'again'})
    with pytest.raises(ValueError,match='anchor availability'):
        record(store,'too-early',at,at+pd.Timedelta(hours=1),100,101)
