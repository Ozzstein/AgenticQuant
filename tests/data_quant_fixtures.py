"""Synthetic archival fixtures ONLY for deterministic data/quant tests."""
from datetime import timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from src.data.evidence.store import digest
from src.domain.common import EvidenceRecord, Instrument, RunContext
from src.domain.market import MarketBar, MarketSnapshot


def archived_snapshot(*, crypto=False, periods=160, use_exchange_calendar=False):
    if use_exchange_calendar:
        from src.data.calendars import sessions
        schedule = list(sessions("2020-01-02", "2020-12-31", calendar="XNYS").items())[:periods]
    else:
        dates = pd.date_range("2020-01-01", periods=periods, tz="UTC", freq="D" if crypto else "B")
        schedule = [(date.date().isoformat(), (date.to_pydatetime(), (date + pd.Timedelta(hours=20)).to_pydatetime())) for date in dates]
    last = schedule[-1][1][1]
    context = RunContext(run_id="fixture", mode="research", as_of=last, config_hash="0"*64, code_version="fixture")
    rng = np.random.default_rng(152)
    bars, evidence = [], []
    for j in range(3):
        instrument = Instrument(instrument_id=f"{'crypto' if crypto else 'equity'}:{j}", symbol=f"TEST{j}",
                                asset_class="crypto" if crypto else "equity", quote_currency="USD", venue="test")
        prices = (100 + j * 20) * np.cumprod(1 + rng.normal(0.001, 0.01, len(schedule)))
        for i, (session, (opened, closed)) in enumerate(schedule):
            key = f"{j}-{i}"
            price = Decimal(str(prices[i]))
            bars.append(MarketBar(instrument=instrument, opened_at=opened, closed_at=closed, open=price,
                                  high=price*Decimal("1.01"), low=price*Decimal("0.99"), close=price,
                                  volume=1000+i*10, vwap=price, evidence_id=key, price_basis="raw",
                                  market_type="spot" if crypto else "equity", calendar="24/7" if crypto else "XNYS", session=session))
            evidence.append(EvidenceRecord(evidence_id=key, provider="synthetic-test-only", provider_version="1",
                                            source_uri="test://archived", raw_hash=digest(key.encode()),
                                            ingested_at=last+timedelta(days=365), available_at=closed,
                                            requested_as_of=schedule[0][1][0], observed_at=closed,
                                            availability_basis="documented", quality="usable", instrument_ids=(instrument.instrument_id,)))
    return MarketSnapshot(context=context, bars=tuple(sorted(bars, key=lambda b:(b.closed_at,b.instrument.instrument_id))), evidence=tuple(evidence))
