"""LEAN-loaded algorithm for the canonical execution outbox, with no strategy logic."""

import os
from pathlib import Path

from AlgorithmImports import DataNormalizationMode, QCAlgorithm, Resolution, TimeZones

from src.execution.broker import create_typed_lean_execution
from src.execution.lean_bridge import verify_project
from src.execution.lean_runtime import LeanAlgorithmPort, LeanOperatingBridge


class AIQuantStackAlgorithm(QCAlgorithm):
    def Initialize(self):
        path = self.GetParameter("qal-config")
        if not path:
            raise ValueError("explicit qal-config parameter is required")
        config = verify_project(Path(path))
        if not self.LiveMode:
            raise ValueError(
                "the canonical LEAN execution bridge requires actual live-mode runtime"
            )
        self.SetTimeZone(TimeZones.Utc)
        self.SetAccountCurrency(config.account.base_currency)
        symbols = {}
        for item in config.deployment.instruments:
            if item.instrument.asset_class == "crypto":
                security = self.AddCrypto(item.ticker, Resolution.Second, item.market)
            else:
                security = self.AddEquity(item.ticker, Resolution.Second, item.market)
                security.SetDataNormalizationMode(DataNormalizationMode.Raw)
            security.SetLeverage(1)
            symbols[item.instrument.instrument_id] = security.Symbol
        # This is deliberately separate from persisted config: a generated
        # project alone never authorizes committing real capital.
        allow_live = (
            os.environ.get("QAL_AUTHORIZE_LIVE_EXECUTION") == config.deployment.deployment_id
        )
        self.port = LeanAlgorithmPort(self, config.deployment, symbols, allow_live=allow_live)
        self.execution = create_typed_lean_execution(
            ledger_path=config.ledger_path,
            account=config.account,
            deployment=config.deployment,
            rules=config.rules,
            runtime=self.port,
            fill_policy=config.fill_policy,
            risk_policy=config.risk_policy,
            allow_live=allow_live,
        )
        if config.account.mode == "live":
            self._attach_native_reader(config)
        if config.account.mode == "paper":
            self._restore_paper_state(config)
        self.bridge = LeanOperatingBridge(self.execution)

    def _attach_native_reader(self, config):
        from src.execution.alpaca_venue import AlpacaVenue
        from src.execution.binance_venue import BinanceVenue
        from src.execution.kraken_venue import KrakenVenue

        native = {
            "AlpacaBrokerage": AlpacaVenue,
            "BinanceBrokerage": BinanceVenue,
            "KrakenBrokerage": KrakenVenue,
        }[config.deployment.brokerage]
        symbols = {item.instrument.instrument_id: item.symbol for item in config.native_markets}
        rules = {
            key: value.model_copy(
                update={"venue": native.venue, "symbol": symbols.get(key, value.symbol)}
            )
            for key, value in config.rules.items()
        }
        kwargs = {
            "broker_account_id": config.deployment.broker_account_id,
            "rules": rules,
            "api_key": os.environ[config.api_key_environment],
            "api_secret": os.environ[config.api_secret_environment],
            "allow_live": True,
        }
        if native is not AlpacaVenue:
            kwargs["markets"] = config.native_markets
        self.execution.adapters["lean"].native_reader = native(
            self.execution.ledger, config.account.account_id, **kwargs
        )

    def _restore_paper_state(self, config):
        self.execution.adapters["lean"]._record(
            "paper-restore",
            self.port.restore_paper_state(self.execution.ledger),
        )

    def OnData(self, data):
        if hasattr(self, "bridge") and not self.IsWarmingUp:
            result = self.bridge.step()
            if result["status"] == "blocked" and result["error"] != getattr(
                self, "_last_block", None
            ):
                self.Error("Canonical execution blocked: " + result["error"])
            self._last_block = result["error"]

    def OnOrderEvent(self, event):
        if hasattr(self, "bridge"):
            self.bridge.on_order_event(event)

    def OnEndOfAlgorithm(self):
        if hasattr(self, "execution"):
            reader = self.execution.adapters["lean"].native_reader
            if reader is not None:
                reader.close()
            self.execution.ledger.close()
