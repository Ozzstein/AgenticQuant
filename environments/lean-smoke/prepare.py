"""Prepare a synthetic, network-disabled LEAN PaperBrokerage fixture (not live trading)."""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import json5

from src.domain.common import Instrument
from src.domain.portfolio import AccountSpec
from src.execution.lean_bridge import LeanLaunchConfig, prepare_project
from src.execution.lean_venue import LeanDeployment, LeanInstrument
from src.execution.venue import VenueRules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="pinned public LEAN checkout")
    parser.add_argument(
        "--output", type=Path, required=True, help="new private directory, mounted at /smoke"
    )
    args = parser.parse_args()
    instrument = Instrument(
        instrument_id="smoke:BTCUSD", symbol="BTC/USD", asset_class="crypto", quote_currency="USD"
    )
    account = AccountSpec(
        account_id="paper:lean-offline-smoke",
        mode="paper",
        base_currency="USD",
        initial_cash="100000",
        opened_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    )
    deployment = LeanDeployment(
        deployment_id="offline-smoke",
        account_id=account.account_id,
        broker_account_id="local-paper",
        brokerage="PaperBrokerage",
        mode="paper",
        instruments=(
            LeanInstrument(
                instrument=instrument, ticker="BTCUSD", market="coinbase", base_currency="BTC"
            ),
        ),
    )
    config = LeanLaunchConfig(
        ledger_path="/smoke/ledger.sqlite",
        account=account,
        deployment=deployment,
        rules={
            instrument.instrument_id: VenueRules(
                available_at=account.opened_at,
                market_type="spot",
                active=True,
                venue="lean",
                instrument=instrument,
                symbol="BTCUSD",
                quantity_step="0.00000001",
                minimum_quantity="0.00000001",
                price_tick="0.01",
                minimum_notional="1",
            )
        },
    )
    prepare_project(config, args.output)
    engine = json5.loads((args.source / "Launcher/config.json").read_text())
    engine.update(
        {
            "environment": "live-paper",
            "algorithm-type-name": "SmokeAlgorithm",
            "algorithm-language": "Python",
            "algorithm-location": "/smoke/main.py",
            "parameters": {"qal-config": "/smoke/qal-execution.json"},
            "data-folder": "/Lean/Data",
            "results-destination-folder": "/smoke/results",
            "object-store-root": "/smoke/storage",
            "python-venv": "/app/.venv",
            "python-additional-paths": ["/app", "/smoke"],
            "debugging": False,
            "user-id": 0,
            "api-access-token": "",
            "job-user-id": 0,
            "live-cash-balance": [{"currency": "USD", "amount": 100000, "conversionRate": 1}],
            "live-holdings": [],
            "close-automatically": True,
            "history-provider": [
                "QuantConnect.Lean.Engine.HistoricalData.SubscriptionDataReaderHistoryProvider"
            ],
            "log-file": "/smoke/engine.log",
            "composer-dll-directory": "/Lean/Launcher",
        }
    )
    engine["environments"]["live-paper"]["data-queue-handler"] = [
        "QuantConnect.Lean.Engine.DataFeeds.Queues.QalOfflineSmokeDataQueue"
    ]
    (args.output / "engine.json").write_text(json.dumps(engine, indent=2))
    (args.output / "main.py").write_bytes(
        Path(__file__).with_name("smoke_algorithm.py").read_bytes()
    )


if __name__ == "__main__":
    main()
