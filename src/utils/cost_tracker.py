"""API cost tracking singleton for LLM usage across the QuantAgentLab agent pipeline."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

_log = get_logger("utils.cost_tracker")

# Project root is three levels up from this file (src/utils/cost_tracker.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CostTracker:
    """Singleton that records LLM API token usage and computes USD cost.

    Maintains an in-session call log and supports persistence to daily JSON files
    under ``data/costs/YYYY-MM-DD.json``.

    Usage::

        ct = CostTracker()
        cost = ct.record_call("claude-sonnet-4-6", 1000, 500)
        print(ct.get_session_cost())  # 0.0105
    """

    # Pricing table: USD per million tokens
    PRICING: dict[str, dict[str, float]] = {
        "claude-opus-4-6": {"input": 15.0, "output": 75.0},
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    _instance: CostTracker | None = None

    def __new__(cls) -> CostTracker:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._session_calls: list[dict[str, Any]] = []
            cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_call(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Record a single LLM API call and return its USD cost.

        Args:
            model: Model identifier, e.g. ``"claude-sonnet-4-6"``.
            prompt_tokens: Number of input/prompt tokens consumed.
            completion_tokens: Number of output/completion tokens generated.

        Returns:
            The USD cost of this call, or ``0.0`` if the model is unknown.
        """
        pricing = self.PRICING.get(model)
        if pricing is None:
            _log.warning("CostTracker: unknown model '{}' — cost recorded as $0.00", model)
            cost = 0.0
        else:
            cost = (prompt_tokens / 1_000_000) * pricing["input"] + (
                completion_tokens / 1_000_000
            ) * pricing["output"]

        entry: dict[str, Any] = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "ts": datetime.utcnow().isoformat(),
        }
        self._session_calls.append(entry)
        _log.debug(
            "CostTracker: model={} prompt={} completion={} cost=${:.6f}",
            model,
            prompt_tokens,
            completion_tokens,
            cost,
        )
        return cost

    def get_session_cost(self) -> float:
        """Return the total USD cost accumulated in the current session.

        Returns:
            Sum of all recorded call costs since the last :meth:`reset_session`.
        """
        return sum(c["cost_usd"] for c in self._session_calls)

    def get_session_calls(self) -> list[dict[str, Any]]:
        """Return a copy of the current session call log.

        Returns:
            List of dicts, each containing ``model``, ``prompt_tokens``,
            ``completion_tokens``, ``cost_usd``, and ``ts`` (ISO timestamp).
        """
        return list(self._session_calls)

    def save_daily(self, date_str: str | None = None) -> None:
        """Persist session calls to a daily JSON file.

        The file is written to ``data/costs/YYYY-MM-DD.json`` relative to the
        project root.  Existing data for the same date is merged (new calls are
        appended).

        Args:
            date_str: Date string in ``YYYY-MM-DD`` format.  Defaults to today.
        """
        target_date = date_str or date.today().isoformat()
        costs_dir = _PROJECT_ROOT / "data" / "costs"
        costs_dir.mkdir(parents=True, exist_ok=True)

        filepath = costs_dir / f"{target_date}.json"

        # Load existing data if present so we can merge
        existing = self.load_daily(target_date)
        prior_calls: list[dict[str, Any]] = existing.get("calls", [])

        all_calls = prior_calls + self._session_calls
        payload = self._build_summary_payload(target_date, all_calls)

        with filepath.open("w") as fh:
            json.dump(payload, fh, indent=2)

        _log.info("CostTracker: saved {} call(s) to {}", len(all_calls), filepath)

    def load_daily(self, date_str: str) -> dict[str, Any]:
        """Load the daily cost record for the given date.

        Args:
            date_str: Date string in ``YYYY-MM-DD`` format.

        Returns:
            Parsed JSON dict, or an empty dict if the file does not exist.
        """
        filepath = _PROJECT_ROOT / "data" / "costs" / f"{date_str}.json"
        if not filepath.exists():
            return {}
        with filepath.open() as fh:
            return json.load(fh)

    def get_daily_summary(self, date_str: str | None = None) -> dict[str, Any]:
        """Return a cost summary for a given date or the current session.

        When ``date_str`` is ``None`` the summary is derived from the in-memory
        session.  Otherwise the daily JSON file is loaded.

        Args:
            date_str: Date in ``YYYY-MM-DD`` format, or ``None`` for the live session.

        Returns:
            Dict with keys ``date``, ``total_usd``, ``by_model``, ``call_count``.
        """
        if date_str is None:
            return self._build_summary_payload(date.today().isoformat(), self._session_calls)
        daily = self.load_daily(date_str)
        if not daily:
            return {
                "date": date_str,
                "total_usd": 0.0,
                "by_model": {},
                "call_count": 0,
                "calls": [],
            }
        return daily

    def reset_session(self) -> None:
        """Clear the in-memory session call log.

        Does **not** affect any persisted daily files.
        """
        self._session_calls = []
        _log.debug("CostTracker: session reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary_payload(
        target_date: str, calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build the canonical JSON payload from a list of call records.

        Args:
            target_date: Date string for the ``date`` key.
            calls: List of call dicts as returned by :meth:`get_session_calls`.

        Returns:
            Dict suitable for JSON serialisation.
        """
        total_usd = sum(c["cost_usd"] for c in calls)
        by_model: dict[str, dict[str, Any]] = {}
        for call in calls:
            m = call["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "total_usd": 0.0}
            by_model[m]["calls"] += 1
            by_model[m]["total_usd"] += call["cost_usd"]

        return {
            "date": target_date,
            "total_usd": total_usd,
            "by_model": by_model,
            "call_count": len(calls),
            "calls": calls,
        }
