"""Audit trail logger for the QuantAgentLab daily pipeline.

Records every significant step of a pipeline run — predictions, agent analyses,
risk decisions, trade fills, portfolio snapshots, and API costs — to a
structured JSON file under ``data/audit/YYYY-MM-DD.json``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.schemas import AnalysisResult

# Project root is three levels up from this file (src/utils/audit.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class AuditLogger:
    """Collects and persists a structured audit trail for one pipeline run.

    All ``log_*`` methods are designed to be graceful: if a passed object does
    not have the expected attributes, a warning is emitted and the available
    data is stored as-is.

    Usage::

        audit = AuditLogger()
        audit.start_run()
        audit.log_step("equity_data", "ok", {"rows": 1000})
        ...
        path = audit.finalize_run()   # writes data/audit/YYYY-MM-DD.json
    """

    def __init__(self, run_date: str | None = None) -> None:
        """Initialise a blank audit record.

        Args:
            run_date: Date string ``YYYY-MM-DD`` for this run. Defaults to today
                in UTC when ``start_run()`` is called.
        """
        self._run_date: str | None = run_date
        self._run_id: str = str(uuid.uuid4())
        self._pipeline_steps: list[dict[str, Any]] = []
        self._predictions: list[dict[str, Any]] = []
        self._analyses: list[dict[str, Any]] = []
        self._risk_decisions: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []
        self._portfolio_before: dict[str, Any] = {}
        self._portfolio_after: dict[str, Any] = {}
        self._api_costs: dict[str, Any] = {"total_usd": 0.0, "calls": []}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_date: str | None = None) -> None:
        """Mark the beginning of a pipeline run.

        Resets all in-memory buffers and captures a fresh run_id.

        Args:
            run_date: Optional date override (``YYYY-MM-DD``).  When omitted the
                value supplied to ``__init__`` is used, falling back to today.
        """
        if run_date is not None:
            self._run_date = run_date
        if self._run_date is None:
            self._run_date = datetime.now(UTC).date().isoformat()

        self._run_id = str(uuid.uuid4())
        self._pipeline_steps = []
        self._predictions = []
        self._analyses = []
        self._risk_decisions = []
        self._trades = []
        self._portfolio_before = {}
        self._portfolio_after = {}
        self._api_costs = {"total_usd": 0.0, "calls": []}
        logger.debug("AuditLogger: run started — id={} date={}", self._run_id, self._run_date)

    # ------------------------------------------------------------------
    # Step-level logging
    # ------------------------------------------------------------------

    def log_step(self, step: str, status: str, details: dict | None = None) -> None:
        """Record a named pipeline step with its completion status.

        Args:
            step: Human-readable step name, e.g. ``"equity_data"``.
            status: ``"ok"`` or ``"error"`` (or any custom string).
            details: Optional extra metadata dict to embed.
        """
        entry: dict[str, Any] = {
            "step": step,
            "status": status,
            "ts": _now_iso(),
            "details": details or {},
        }
        self._pipeline_steps.append(entry)
        logger.debug("AuditLogger: step={} status={}", step, status)

    # ------------------------------------------------------------------
    # Prediction logging
    # ------------------------------------------------------------------

    def log_prediction(self, ticker: str, score: float) -> None:
        """Record an ML alpha score for a ticker.

        Args:
            ticker: Asset symbol, e.g. ``"AAPL"``.
            score: Float alpha score produced by the ML model.
        """
        self._predictions.append({"ticker": ticker, "score": score, "ts": _now_iso()})

    # ------------------------------------------------------------------
    # Analysis logging
    # ------------------------------------------------------------------

    def log_analysis(self, ticker: str, result: AnalysisResult) -> None:
        """Record the outcome of a multi-agent analysis for a ticker.

        Args:
            ticker: Asset symbol.
            result: ``AnalysisResult`` Pydantic model from the agent graph.
        """
        try:
            decision = (
                result.decision.value
                if hasattr(result.decision, "value")
                else str(result.decision)
            )
            entry: dict[str, Any] = {
                "ticker": ticker,
                "decision": decision,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "risk_flags": list(result.risk_flags),
                "ts": _now_iso(),
            }
        except Exception as exc:
            logger.warning("AuditLogger: log_analysis failed to parse result for {}: {}", ticker, exc)
            entry = {"ticker": ticker, "raw": str(result), "ts": _now_iso()}
        self._analyses.append(entry)

    # ------------------------------------------------------------------
    # Risk decision logging
    # ------------------------------------------------------------------

    def log_risk_decision(self, order: Any, check_result: Any) -> None:
        """Record whether a pre-trade risk check passed or failed.

        Args:
            order: ``Order`` Pydantic model (or dict) representing the order.
            check_result: ``RiskCheckResult`` dataclass (or dict) from the risk
                controls module.
        """
        try:
            # Extract order fields defensively
            ticker = _get_attr(order, "ticker", "UNKNOWN")
            side = _get_attr(order, "side", None)
            side_str = side.value if hasattr(side, "value") else str(side) if side else "UNKNOWN"
            quantity = _get_attr(order, "quantity", 0.0)

            # Infer order value from fill_price or quantity alone
            fill_price = _get_attr(order, "fill_price", None) or _get_attr(order, "limit_price", None)
            order_value = (fill_price * quantity) if fill_price else 0.0

            # Extract risk result fields defensively
            passed = _get_attr(check_result, "passed", None)
            if passed is None:
                passed = _get_attr(check_result, "passed", False)
            failed_checks = list(_get_attr(check_result, "failed_checks", []))

            entry: dict[str, Any] = {
                "ticker": ticker,
                "side": side_str,
                "passed": passed,
                "failed_checks": failed_checks,
                "order_value": order_value,
                "ts": _now_iso(),
            }
        except Exception as exc:
            logger.warning("AuditLogger: log_risk_decision failed to parse inputs: {}", exc)
            entry = {
                "raw_order": str(order),
                "raw_check": str(check_result),
                "ts": _now_iso(),
            }
        self._risk_decisions.append(entry)

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

    def log_trade(self, order: Any, fill_price: float, pnl: float) -> None:
        """Record an executed trade fill.

        Args:
            order: ``Order`` Pydantic model (or dict).
            fill_price: Actual fill price per share/unit.
            pnl: Realised P&L from this fill (0 for new positions).
        """
        try:
            ticker = _get_attr(order, "ticker", "UNKNOWN")
            side = _get_attr(order, "side", None)
            side_str = side.value if hasattr(side, "value") else str(side) if side else "UNKNOWN"
            qty = _get_attr(order, "quantity", 0.0)

            entry: dict[str, Any] = {
                "ticker": ticker,
                "side": side_str,
                "qty": qty,
                "fill_price": fill_price,
                "pnl": pnl,
                "ts": _now_iso(),
            }
        except Exception as exc:
            logger.warning("AuditLogger: log_trade failed to parse order: {}", exc)
            entry = {
                "raw_order": str(order),
                "fill_price": fill_price,
                "pnl": pnl,
                "ts": _now_iso(),
            }
        self._trades.append(entry)

    # ------------------------------------------------------------------
    # Portfolio snapshot logging
    # ------------------------------------------------------------------

    def log_portfolio_state(self, label: str, portfolio: Any) -> None:
        """Capture a portfolio snapshot as either *before* or *after* the run.

        ``label`` must be ``"before"`` or ``"after"``.  Any other value is
        accepted but stored under ``portfolio_{label}``.

        Args:
            label: ``"before"`` or ``"after"``.
            portfolio: ``Portfolio`` Pydantic model (or dict).
        """
        snapshot = _serialise_portfolio(portfolio)
        if label == "before":
            self._portfolio_before = snapshot
        elif label == "after":
            self._portfolio_after = snapshot
        else:
            logger.warning("AuditLogger: unexpected portfolio label '{}' — storing anyway", label)
            # Store under a generic key in pipeline steps so it's not lost
            self.log_step(f"portfolio_{label}", "ok", snapshot)

    # ------------------------------------------------------------------
    # API cost logging
    # ------------------------------------------------------------------

    def log_api_costs(self, costs: dict[str, float]) -> None:
        """Merge additional API cost entries into the running total.

        Args:
            costs: Dict with at least ``total_usd`` and optionally ``calls``
                (list of per-call dicts).  Extra keys are preserved.
        """
        try:
            extra_total = float(costs.get("total_usd", 0.0))
            self._api_costs["total_usd"] = float(self._api_costs["total_usd"]) + extra_total

            extra_calls = costs.get("calls", [])
            if isinstance(extra_calls, list):
                self._api_costs["calls"].extend(extra_calls)

            # Preserve any extra keys from the caller
            for k, v in costs.items():
                if k not in ("total_usd", "calls"):
                    self._api_costs[k] = v
        except Exception as exc:
            logger.warning("AuditLogger: log_api_costs failed: {}", exc)

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def finalize_run(self) -> Path | None:
        """Write the audit record to ``data/audit/YYYY-MM-DD.json``.

        Creates the directory if it does not exist.

        Returns:
            Path to the written audit file, or None if write failed.
        """
        if self._run_date is None:
            self._run_date = datetime.now(UTC).date().isoformat()

        audit_dir = _PROJECT_ROOT / "data" / "audit"
        try:
            audit_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("AuditLogger: could not create audit directory {}: {}", audit_dir, exc)
            return None

        filepath = audit_dir / f"{self._run_date}.json"

        payload: dict[str, Any] = {
            "date": self._run_date,
            "run_id": self._run_id,
            "pipeline_steps": self._pipeline_steps,
            "predictions": self._predictions,
            "analyses": self._analyses,
            "risk_decisions": self._risk_decisions,
            "trades": self._trades,
            "portfolio_before": self._portfolio_before,
            "portfolio_after": self._portfolio_after,
            "api_costs": self._api_costs,
        }

        try:
            with filepath.open("w") as fh:
                json.dump(payload, fh, indent=2, default=str)
            logger.info(
                "AuditLogger: run {} written to {}",
                self._run_id,
                filepath,
            )
            return filepath
        except (OSError, TypeError) as exc:
            logger.error("AuditLogger: failed to write audit file {}: {}", filepath, exc)
            return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_attr(obj: Any, attr: str, default: Any) -> Any:
    """Return obj.attr if available, falling back to obj[attr] then default."""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def _serialise_portfolio(portfolio: Any) -> dict[str, Any]:
    """Convert a Portfolio (or dict) to a JSON-serialisable dict.

    Args:
        portfolio: ``Portfolio`` Pydantic model, plain dict, or unknown object.

    Returns:
        Dict with at minimum ``nav``, ``cash``, and ``positions`` keys.
    """
    try:
        if isinstance(portfolio, dict):
            return portfolio

        nav = _get_attr(portfolio, "nav", 0.0)
        cash = _get_attr(portfolio, "cash", 0.0)
        positions_raw = _get_attr(portfolio, "positions", {})

        positions: dict[str, Any] = {}
        if isinstance(positions_raw, dict):
            for ticker, pos in positions_raw.items():
                positions[ticker] = {
                    "qty": _get_attr(pos, "quantity", 0.0),
                    "avg_cost": _get_attr(pos, "avg_cost", 0.0),
                    "current_price": _get_attr(pos, "current_price", 0.0),
                    "unrealized_pnl": _get_attr(pos, "unrealized_pnl", 0.0),
                    "weight_pct": _get_attr(pos, "weight_pct", 0.0),
                }

        return {"nav": nav, "cash": cash, "positions": positions}
    except Exception as exc:
        logger.warning("AuditLogger: _serialise_portfolio failed: {}", exc)
        return {"raw": str(portfolio)}
