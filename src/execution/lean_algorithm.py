"""LEAN-loaded algorithm for the canonical execution outbox, with no strategy logic."""

import os
from pathlib import Path

from AlgorithmImports import DataNormalizationMode, QCAlgorithm, Resolution, TimeZones

from src.execution.broker import create_typed_lean_execution
from src.execution.lean_bridge import verify_project
from src.execution.lean_runtime import LeanAlgorithmPort, LeanOperatingBridge
from src.execution.ledger import LedgerError


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
        self._qal_port = LeanAlgorithmPort(self, config.deployment, symbols, allow_live=allow_live)
        evidence_store = None
        if config.evidence_store_path is not None:
            from src.data.evidence.store import EvidenceStore

            if not Path(config.evidence_store_path).is_file():
                raise ValueError("configured actual EvidenceStore does not exist")
            evidence_store = EvidenceStore(config.evidence_store_path)
        self._qal_execution = create_typed_lean_execution(
            ledger_path=config.ledger_path,
            account=config.account,
            deployment=config.deployment,
            rules=config.rules,
            runtime=self._qal_port,
            evidence_store=evidence_store,
            fill_policy=config.fill_policy,
            risk_policy=config.risk_policy,
            allow_live=allow_live,
        )
        if config.account.mode == "live":
            self._attach_native_reader(config)
        self._qal_paper_restore_pending = config.account.mode == "paper"
        if self._qal_paper_restore_pending:
            # Refuse unresolved original orders before brokerage setup can proceed.
            self._qal_port.validate_paper_restore(self._qal_execution.ledger)
        else:
            self._qal_bridge = LeanOperatingBridge(self._qal_execution)

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
        self._qal_execution.adapters["lean"].native_reader = native(
            self._qal_execution.ledger, config.account.account_id, **kwargs
        )

    def OnWarmupFinished(self):
        # LEAN invokes this after loading brokerage cash/holdings, including when
        # no warmup is requested. Repeated callbacks must never reset economics.
        if getattr(self, "_qal_paper_restore_pending", False):
            self._restore_paper_state()
            self._qal_bridge = LeanOperatingBridge(self._qal_execution)
            self._qal_paper_restore_pending = False

    def _restore_paper_state(self):
        self._qal_execution.adapters["lean"]._record(
            "paper-restore",
            self._qal_port.restore_paper_state(self._qal_execution.ledger),
        )

    def OnData(self, data):
        if not self.IsWarmingUp and getattr(self, "_qal_paper_restore_pending", False):
            raise LedgerError("LEAN paper initialization is not complete")
        if hasattr(self, "_qal_bridge") and not self.IsWarmingUp:
            result = self._qal_bridge.step()
            if result["status"] == "blocked" and result["error"] != getattr(
                self, "_last_block", None
            ):
                self.Error("Canonical execution blocked: " + result["error"])
            self._last_block = result["error"]

    def OnOrderEvent(self, event):
        if hasattr(self, "_qal_bridge"):
            self._qal_bridge.on_order_event(event)

    def OnEndOfAlgorithm(self):
        if hasattr(self, "_qal_execution"):
            reader = self._qal_execution.adapters["lean"].native_reader
            if reader is not None:
                reader.close()
            self._qal_execution.ledger.close()
