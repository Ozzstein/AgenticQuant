"""Strategy performance tracker for per-strategy live P&L and drift detection."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.config import AppConfig, get_config
from src.utils.schemas import StrategyPerformance


class StrategyTracker:
    """Tracks per-strategy live performance and detects drift from backtest.

    Maintains rolling Sharpe ratios, drawdown metrics, and win rates for each
    strategy. Alerts when live performance diverges significantly from the
    backtest expectation.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the tracker, loading any previously persisted performance data.

        Args:
            config: Application configuration. Falls back to get_config() if None.
        """
        self.config: AppConfig = config if config is not None else get_config()
        self.strategy_config = self.config.strategy
        self.perf_dir = Path("data/strategy_performance")
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.performances: dict[str, StrategyPerformance] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load all persisted strategy performance files from disk.

        Reads every *.json file in self.perf_dir and hydrates StrategyPerformance
        objects into self.performances. Missing or empty directories are handled
        gracefully.
        """
        try:
            for json_file in self.perf_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    strategy_name = json_file.stem
                    perf = StrategyPerformance(**data)
                    self.performances[strategy_name] = perf
                    logger.debug("Loaded performance data for strategy '{}'", strategy_name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to load performance file {}: {}", json_file, exc
                    )
        except FileNotFoundError:
            logger.debug("Performance directory not found; starting fresh.")

    def save(self) -> None:
        """Persist all strategy performance objects to disk as JSON.

        Each strategy is stored in a separate file named ``{strategy_name}.json``
        under self.perf_dir. Datetime fields are serialised as ISO-8601 strings
        via Pydantic's ``model_dump(mode='json')``.
        """
        for name, perf in self.performances.items():
            file_path = self.perf_dir / f"{name}.json"
            try:
                file_path.write_text(json.dumps(perf.model_dump(mode="json"), indent=2))
                logger.debug("Saved performance data for strategy '{}'", name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to save performance for '{}': {}", name, exc)

    # ------------------------------------------------------------------
    # P&L Attribution
    # ------------------------------------------------------------------

    def attribute_pnl(self, strategy_name: str, daily_pnl: float) -> None:
        """Record a single day's P&L for a strategy and refresh all metrics.

        Args:
            strategy_name: Unique strategy identifier.
            daily_pnl: Dollar (or %) P&L for the day to attribute.
        """
        if strategy_name not in self.performances:
            self.performances[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )

        perf = self.performances[strategy_name]
        perf.daily_pnl.append(daily_pnl)
        perf.total_pnl += daily_pnl
        perf.last_updated = datetime.now()

        self.update_metrics(strategy_name)
        self.save()

    # ------------------------------------------------------------------
    # Metric Computation
    # ------------------------------------------------------------------

    def update_metrics(self, strategy_name: str) -> None:
        """Recompute all rolling metrics for a strategy.

        Calculates rolling Sharpe ratios (30/60/90-day), maximum and current
        drawdown, and win rate, then checks for strategy drift.

        Args:
            strategy_name: Unique strategy identifier.
        """
        if strategy_name not in self.performances:
            self.performances[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )

        perf = self.performances[strategy_name]
        pnl = perf.daily_pnl

        # Rolling Sharpe ratios
        perf.rolling_sharpe_30d = self._rolling_sharpe(pnl, 30)
        perf.rolling_sharpe_60d = self._rolling_sharpe(pnl, 60)
        perf.rolling_sharpe_90d = self._rolling_sharpe(pnl, 90)

        # Drawdown metrics
        if len(pnl) > 0:
            cumulative_pnl = np.cumsum(pnl)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = cumulative_pnl - running_max
            perf.max_drawdown = float(np.min(drawdown))
            perf.current_drawdown = float(drawdown[-1])
        else:
            perf.max_drawdown = 0.0
            perf.current_drawdown = 0.0

        # Win rate
        total_days = len(pnl)
        if total_days > 0:
            wins = sum(1 for d in pnl if d > 0)
            perf.win_rate = wins / total_days
        else:
            perf.win_rate = 0.0

        # Drift detection
        self.check_drift(strategy_name)

    def _rolling_sharpe(self, pnl_list: list[float], window: int) -> float:
        """Compute annualised Sharpe ratio over the last *window* daily P&L values.

        Args:
            pnl_list: Full history of daily P&L values.
            window: Number of trailing days to include.

        Returns:
            Annualised Sharpe ratio, or 0.0 when there is insufficient data or
            the standard deviation is zero.
        """
        if window == 0 or len(pnl_list) < window:
            return 0.0

        recent = pnl_list[-window:]
        mean = float(np.mean(recent))
        std = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0

        if std == 0.0:
            return 0.0

        return mean * math.sqrt(252) / std

    # ------------------------------------------------------------------
    # Drift Detection
    # ------------------------------------------------------------------

    def check_drift(self, strategy_name: str) -> bool:
        """Check whether a strategy's live Sharpe has drifted below its backtest value.

        Drift is flagged when the rolling 60-day live Sharpe falls below
        ``drift_threshold * backtest_sharpe``.  If either value is unavailable
        (zero), the check is skipped and False is returned.

        Args:
            strategy_name: Unique strategy identifier.

        Returns:
            True if drift is detected, False otherwise.
        """
        if strategy_name not in self.performances:
            return False

        perf = self.performances[strategy_name]
        live_sharpe = perf.rolling_sharpe_60d
        backtest_sharpe = perf.backtest_sharpe

        if live_sharpe == 0.0 or backtest_sharpe == 0.0:
            return False

        drift_threshold = self.strategy_config.drift_threshold
        detected = live_sharpe < drift_threshold * backtest_sharpe
        perf.drift_detected = detected

        if detected:
            logger.warning(
                "Strategy {} drift detected: live={:.2f} vs backtest={:.2f}",
                strategy_name,
                live_sharpe,
                backtest_sharpe,
            )

        return detected

    # ------------------------------------------------------------------
    # Backtest Anchor
    # ------------------------------------------------------------------

    def set_backtest_sharpe(self, strategy_name: str, backtest_sharpe: float) -> None:
        """Store the historical backtest Sharpe ratio for a strategy.

        This acts as the reference against which live drift is measured.

        Args:
            strategy_name: Unique strategy identifier.
            backtest_sharpe: Annualised out-of-sample Sharpe from walk-forward backtest.
        """
        if strategy_name not in self.performances:
            self.performances[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )

        self.performances[strategy_name].backtest_sharpe = backtest_sharpe

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_performances(self) -> dict[str, StrategyPerformance]:
        """Return a copy of all tracked strategy performance objects.

        Returns:
            Shallow copy mapping strategy names to their StrategyPerformance.
        """
        return dict(self.performances)

    def report(self) -> dict:
        """Generate a summary report suitable for CLI output or a dashboard.

        Returns:
            Dict with ``total_strategies`` count and per-strategy key metrics.
        """
        return {
            "total_strategies": len(self.performances),
            "strategies": {
                name: {
                    "rolling_sharpe_60d": perf.rolling_sharpe_60d,
                    "max_drawdown": perf.max_drawdown,
                    "win_rate": perf.win_rate,
                    "total_pnl": perf.total_pnl,
                    "drift_detected": perf.drift_detected,
                }
                for name, perf in self.performances.items()
            },
        }


class AttributedPerformanceStore:
    """Append-only return observations derived from the accounting ledger.

    This stores no orders/positions/balances and is not a second portfolio ledger.
    Caller supplies reconciled sleeve NAV and the immutable source event hash.
    Flows are end-of-period; intraperiod flows require splitting the observation.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS attributed_nav (strategy TEXT, ended_at TEXT, available_at TEXT, source_event_id TEXT UNIQUE, payload TEXT, PRIMARY KEY(strategy,ended_at))")
            db.execute("CREATE TABLE IF NOT EXISTS funded_epochs (strategy TEXT, at TEXT, available_at TEXT, source_event_id TEXT UNIQUE, payload TEXT, PRIMARY KEY(strategy,at))")
            for operation in ("UPDATE", "DELETE"):
                db.execute(f"CREATE TRIGGER IF NOT EXISTS epoch_no_{operation.lower()} BEFORE {operation} ON funded_epochs BEGIN SELECT RAISE(ABORT, 'immutable funded epoch'); END")
                db.execute(f"CREATE TRIGGER IF NOT EXISTS nav_no_{operation.lower()} BEFORE {operation} ON attributed_nav BEGIN SELECT RAISE(ABORT, 'immutable attributed observation'); END")

    @contextmanager
    def _db(self):
        import sqlite3
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def start_funded_epoch(self, strategy: str, *, at, available_at, opening_nav,
                           source_event_id: str, source_event_hash: str):
        """Record a source-bound positive funding anchor, never a return or flow estimate.

        Initial anchoring requires no prior history. Reactivation requires the
        latest recorded NAV to be exactly zero and a new authoritative snapshot.
        The caller verifies the accounting snapshot/transfer source.
        """
        from decimal import Decimal

        from src.data.evidence.store import canonical_bytes
        from src.domain.common import _utc
        at, available_at = _utc(at), _utc(available_at)
        if not strategy or not source_event_id or available_at < at:
            raise ValueError("funding anchor requires identity and actual availability")
        if not isinstance(opening_nav, Decimal) or not opening_nav.is_finite() or opening_nav <= 0:
            raise ValueError("positive Decimal funded snapshot required")
        if len(source_event_hash) != 64 or any(c not in "0123456789abcdef" for c in source_event_hash):
            raise ValueError("immutable funding source hash required")
        payload = canonical_bytes(dict(strategy=strategy,at=at.isoformat(),available_at=available_at.isoformat(),
            opening_nav=str(opening_nav),source_event_id=source_event_id,source_event_hash=source_event_hash)).decode()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT payload FROM funded_epochs WHERE source_event_id=?",(source_event_id,)).fetchone()
            if existing:
                if existing[0] != payload:
                    raise ValueError("funding source cannot be rewritten")
                return
            previous_epoch = db.execute("SELECT payload FROM funded_epochs WHERE strategy=? ORDER BY at DESC LIMIT 1",(strategy,)).fetchone()
            previous = db.execute("SELECT payload FROM attributed_nav WHERE strategy=? ORDER BY ended_at DESC LIMIT 1",(strategy,)).fetchone()
            if previous:
                value = json.loads(previous[0])
                if Decimal(value["closing_nav"]) != 0 or at.isoformat() < value["ended_at"]:
                    raise ValueError("reactivation requires an actual prior zero NAV and forward funding clock")
                if previous_epoch and json.loads(previous_epoch[0])["at"] >= value["ended_at"]:
                    raise ValueError("funded epoch has no subsequent authoritative zero NAV")
            elif previous_epoch:
                raise ValueError("existing positive anchor cannot be reset without zero NAV history")
            db.execute("INSERT INTO funded_epochs VALUES (?,?,?,?,?)",(strategy,at.isoformat(),available_at.isoformat(),source_event_id,payload))

    def record_nav(self, strategy: str, *, started_at, ended_at, available_at, opening_nav,
                   closing_nav, end_flow, source_event_id: str, source_event_hash: str):
        from decimal import Decimal

        from src.data.evidence.store import canonical_bytes
        from src.domain.common import _utc
        times = [_utc(value) for value in (started_at, ended_at, available_at)]
        started_at, ended_at, available_at = times
        if started_at >= ended_at or ended_at > available_at or not strategy or not source_event_id:
            raise ValueError("valid period/source identity and explicit availability required")
        if len(source_event_hash) != 64 or any(c not in "0123456789abcdef" for c in source_event_hash):
            raise ValueError("immutable source event hash required")
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in (opening_nav, closing_nav, end_flow)):
            raise ValueError("finite Decimal ledger NAV and flow required")
        if opening_nav <= 0 or closing_nav < 0 or closing_nav-end_flow < 0:
            raise ValueError("invalid reconciled NAV")
        period_return = (closing_nav-end_flow)/opening_nav-1
        payload = canonical_bytes(dict(strategy=strategy, started_at=started_at.isoformat(), ended_at=ended_at.isoformat(),
                                       available_at=available_at.isoformat(), opening_nav=str(opening_nav), closing_nav=str(closing_nav),
                                       end_flow=str(end_flow), period_return=str(period_return), source_event_id=source_event_id,
                                       source_event_hash=source_event_hash, flow_convention="end_of_period")).decode()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT payload FROM attributed_nav WHERE source_event_id=?", (source_event_id,)).fetchone()
            if existing:
                if existing[0] != payload:
                    raise ValueError("source event cannot be rewritten")
                return
            prior = db.execute("SELECT payload FROM attributed_nav WHERE strategy=? ORDER BY ended_at DESC LIMIT 1", (strategy,)).fetchone()
            epoch = db.execute("SELECT payload FROM funded_epochs WHERE strategy=? ORDER BY at DESC LIMIT 1",(strategy,)).fetchone()
            epoch = json.loads(epoch[0]) if epoch else None
            previous = json.loads(prior[0]) if prior else None
            if epoch and (previous is None or previous["ended_at"] <= epoch["at"]):
                if started_at.isoformat() != epoch["at"] or opening_nav != Decimal(epoch["opening_nav"]):
                    raise ValueError("first return must begin at the exact funded snapshot")
                if available_at.isoformat() < epoch["available_at"]:
                    raise ValueError("return cannot precede funding anchor availability")
            elif previous:
                if started_at.isoformat() != previous["ended_at"] or opening_nav != Decimal(previous["closing_nav"]):
                    raise ValueError("attributed NAV must continue reconciled history without gaps or overlap")
            db.execute("INSERT INTO attributed_nav VALUES (?,?,?,?,?)", (strategy, ended_at.isoformat(), available_at.isoformat(), source_event_id, payload))

    def daily_returns(self, *, as_of, annualization: int):
        """Compound actual contiguous subperiods between exact completed closes.

        XNYS uses prior/current exchange closes (including overnight/weekends).
        Continuous markets use UTC midnights. A crossing interval is never split,
        a missing close is never inferred, and late evidence keeps its true clock.
        The returned frame retains exact constituent payloads and source hashes.
        """
        from decimal import Decimal

        import pandas as pd

        from src.data.evidence.store import canonical_bytes, digest
        from src.domain.common import _utc
        cutoff = _utc(as_of)
        if annualization not in (252, 365):
            raise ValueError("explicit equity or continuous calendar required")
        with self._db() as db:
            rows = db.execute("SELECT payload FROM attributed_nav WHERE ended_at<=? AND available_at<=? ORDER BY ended_at", (cutoff.isoformat(),cutoff.isoformat())).fetchall()
            anchors = db.execute("SELECT payload FROM funded_epochs WHERE at<=? AND available_at<=? ORDER BY at",(cutoff.isoformat(),cutoff.isoformat())).fetchall()
        groups, epochs = {}, {}
        for row in anchors:
            value = json.loads(row[0])
            epochs.setdefault(value["strategy"], []).append(value)
            groups.setdefault(value["strategy"], [])
        for row in rows:
            value = json.loads(row[0])
            groups.setdefault(value["strategy"], []).append(value)
        output, coverage, chains = [], {}, []
        for name, history in groups.items():
            starts = epochs.get(name, [])
            implicit = [item for item in history if not starts or item["started_at"] < starts[0]["at"]]
            if implicit:
                chains.append((name,implicit,None))
            for i, anchor in enumerate(starts):
                chain = [item for item in history if anchor["at"] <= item["started_at"]
                         and (i+1 == len(starts) or item["started_at"] < starts[i+1]["at"])]
                if i+1 < len(starts) and any(item["ended_at"] > starts[i+1]["at"] for item in chain):
                    raise ValueError("attributed period crosses a funding epoch")
                chains.append((name,chain,anchor))
        for name, history, anchor in chains:
            epoch_id = anchor["source_event_id"] if anchor else "initial-unanchored"
            if anchor and history and (history[0]["started_at"] != anchor["at"] or Decimal(history[0]["opening_nav"]) != Decimal(anchor["opening_nav"])):
                raise ValueError("return chain differs from its exact funding anchor")
            for previous, current in zip(history, history[1:]):
                if current["started_at"] != previous["ended_at"] or Decimal(current["opening_nav"]) != Decimal(previous["closing_nav"]):
                    raise ValueError("attributed return chain has a gap or unreconciled NAV")
            first = pd.Timestamp(anchor["at"] if anchor else history[0]["started_at"])
            last = pd.Timestamp(history[-1]["ended_at"]) if history else first
            if annualization == 365:
                boundaries = list(pd.date_range(first.normalize(),last.normalize(),freq="D"))
            else:
                from src.data.calendars import sessions
                calendar = sessions((first-pd.Timedelta(days=14)).date().isoformat(),last.date().isoformat(),calendar="XNYS")
                boundaries = [pd.Timestamp(close) for _,close in calendar.values() if close <= last]
            pairs = [(a,b) for a,b in zip(boundaries,boundaries[1:]) if first <= a and b <= cutoff]
            by_start = {pd.Timestamp(item["started_at"]):item for item in history}
            complete, used = 0, set()
            for begin,end in pairs:
                cursor, parts = begin, []
                while cursor < end:
                    item = by_start.get(cursor)
                    if item is None or pd.Timestamp(item["ended_at"]) > end or pd.Timestamp(item["available_at"]) > cutoff:
                        break
                    parts.append(item)
                    cursor = pd.Timestamp(item["ended_at"])
                if cursor != end:
                    continue
                factor = Decimal(1)
                for item in parts:
                    value = (Decimal(item["closing_nav"])-Decimal(item["end_flow"]))/Decimal(item["opening_nav"])
                    if value-1 != Decimal(item["period_return"]):
                        raise ValueError("stored return differs from actual NAV and end flow")
                    factor *= value
                    used.add(item["source_event_id"])
                record = dict(strategy=name,epoch_id=epoch_id,started_at=begin,ended_at=end,
                    available_at=pd.Timestamp(max(item["available_at"] for item in parts)),
                    period_return=factor-1,source_event_ids=tuple(item["source_event_id"] for item in parts),
                    source_event_hashes=tuple(item["source_event_hash"] for item in parts),subperiods=tuple(parts))
                record["aggregation_hash"] = digest(canonical_bytes({"implementation":"exact-close-compounded-1",
                    "annualization":annualization,"funding_anchor":anchor,"started_at":begin.isoformat(),"ended_at":end.isoformat(),"subperiods":parts}))
                output.append(record)
                complete += 1
            coverage[name] = dict(epoch_id=epoch_id,funding_anchor=anchor,complete=bool(pairs) and complete == len(pairs),expected_periods=len(pairs),
                completed_periods=complete,available_at=max([item["available_at"] for item in history]+([anchor["available_at"]] if anchor else [])),
                unaggregated_source_ids=tuple(item["source_event_id"] for item in history if item["source_event_id"] not in used))
        columns = ["strategy","epoch_id","started_at","ended_at","available_at","period_return","source_event_ids",
                   "source_event_hashes","subperiods","aggregation_hash"]
        result = pd.DataFrame(output,columns=columns)
        result.attrs["coverage"] = coverage
        return result

    def metrics(self, *, as_of, annualization: int, window: int = 60):
        import pandas as pd
        if window < 2:
            raise ValueError("valid daily window required")
        daily = self.daily_returns(as_of=as_of,annualization=annualization)
        metrics = []
        for name,coverage in daily.attrs["coverage"].items():
            history = daily.loc[(daily.strategy == name) & (daily.epoch_id == coverage["epoch_id"])]
            trailing = history.iloc[-window:]
            returns = np.asarray(trailing.period_return,dtype=float)
            complete = coverage["complete"]
            std = returns.std(ddof=1) if len(returns)>1 else np.nan
            sharpe = float(returns.mean()/std*np.sqrt(annualization)) if complete and std>0 and len(returns)>=20 else np.nan
            nav = np.r_[1.,np.cumprod(1+np.asarray(history.period_return,dtype=float))]
            drawdown = nav/np.maximum.accumulate(nav)-1
            metrics.append(dict(strategy=name,epoch_id=coverage["epoch_id"],funding_anchor=coverage["funding_anchor"],sharpe=sharpe,observations=len(returns),annualization=annualization,
                complete=complete,available_at=trailing.available_at.max() if len(trailing) else pd.Timestamp(coverage["available_at"]),
                source_event_ids=tuple(key for values in trailing.source_event_ids for key in values),
                source_event_hashes=tuple(key for values in trailing.source_event_hashes for key in values),
                aggregation_hashes=tuple(trailing.aggregation_hash),max_drawdown=float(drawdown.min()),
                current_drawdown=float(drawdown[-1]),total_return=float(nav[-1]-1)))
        columns = ["strategy","epoch_id","funding_anchor","sharpe","observations","annualization","complete","available_at","source_event_ids",
                   "source_event_hashes","aggregation_hashes","max_drawdown","current_drawdown","total_return"]
        return pd.DataFrame(metrics,columns=columns).set_index("strategy")
