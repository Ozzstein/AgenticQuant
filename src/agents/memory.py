"""Agent analysis memory store — JSONL-backed persistence with thread-safe writes."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.logger import get_logger
from src.utils.schemas import AnalysisResult

if TYPE_CHECKING:
    from src.utils.config import AppConfig

try:
    from src.agents.state import TradingDeskState
except ImportError:
    import operator
    from typing import Annotated, TypedDict

    from src.utils.schemas import AnalysisResult as _AR

    def _merge_reports(a: dict, b: dict) -> dict:  # type: ignore[return]
        return {**a, **b}

    class TradingDeskState(TypedDict):  # type: ignore[no-redef]
        ticker: str
        asset_class: str
        qlib_context: str
        market_context: str
        macro_regime: str
        macro_confidence: float
        memory_context: str
        agent_reports: Annotated[dict, _merge_reports]
        debate_round: int
        debate_transcript: Annotated[list, operator.add]
        risk_veto: bool
        risk_veto_reason: str
        result: _AR | None

logger = get_logger(__name__)

_CHAR_BUDGET = 2000
_REASONING_TRUNCATE = 150


class AnalysisMemoryStore:
    """Persist and retrieve past AnalysisResult records for a given ticker.

    Records are stored as JSONL (one JSON object per line) so that concurrent
    appends from multiple threads are safe on POSIX systems.  Reads are
    lock-free; only writes acquire the internal threading.Lock.

    Args:
        store_path: Path to the JSONL file.  Created (with parents) on first
            write if it does not exist.  Defaults to the path from
            MemoryConfig when ``None``.
        max_history: Default maximum number of records returned by
            :meth:`retrieve` when *max_n* is not supplied by the caller.
    """

    def __init__(self, store_path: Path | None = None, max_history: int = 10) -> None:
        if store_path is None:
            from src.utils.config import get_config

            cfg = get_config()
            store_path = cfg.memory.store_path
            max_history = cfg.memory.max_history

        self._path: Path = store_path
        self._max_history: int = max_history
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, result: AnalysisResult) -> None:
        """Append *result* to the JSONL store, creating the file if needed.

        Thread-safe: acquires an internal lock before writing so that
        concurrent callers do not interleave partial lines.

        Args:
            result: The completed analysis to persist.
        """
        line = result.model_dump_json() + "\n"
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        logger.debug("Saved analysis for {} to {}", result.ticker, self._path)

    def retrieve(self, ticker: str, max_n: int | None = None) -> list[AnalysisResult]:
        """Return past analyses for *ticker*, oldest to newest, sorted by timestamp.

        Args:
            ticker: Ticker symbol to filter on (case-insensitive).
            max_n: Maximum number of records to return.  Defaults to
                ``self._max_history``.  The *last* ``max_n`` matching
                lines are returned (i.e. the most recent ones).

        Returns:
            A list of :class:`AnalysisResult` objects, ordered oldest to
            newest, of length at most *max_n*.  Returns an empty list when
            the store file does not exist.
        """
        limit = max_n if max_n is not None else self._max_history
        ticker_upper = ticker.upper()

        if not self._path.exists():
            logger.debug("Memory store not found at {} — returning empty list.", self._path)
            return []

        results: list[AnalysisResult] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = AnalysisResult.model_validate_json(raw)
                except Exception as exc:
                    logger.warning(
                        "Skipping unparseable line {} in {}: {}",
                        lineno,
                        self._path,
                        exc,
                    )
                    continue
                if record.ticker.upper() == ticker_upper:
                    results.append(record)

        # Sort by timestamp (ascending) then take the tail to honour limit.
        results.sort(key=lambda r: r.timestamp)
        return results[-limit:] if limit > 0 else results

    def format_memory_context(self, results: list[AnalysisResult]) -> str:
        """Format past analyses into a compact text suitable for LLM context.

        The total output stays within a 2 000-character budget; if all
        entries together exceed the budget, the oldest entries are dropped
        first until it fits.

        Args:
            results: List of :class:`AnalysisResult` objects (typically
                returned by :meth:`retrieve`).

        Returns:
            A multi-line string summarising each result, or ``""`` if
            *results* is empty.
        """
        if not results:
            return ""

        def _format_one(r: AnalysisResult) -> str:
            date_str = r.timestamp.strftime("%Y-%m-%d")
            reasoning = r.reasoning[:_REASONING_TRUNCATE]
            if len(r.reasoning) > _REASONING_TRUNCATE:
                reasoning += "…"
            flags = ", ".join(r.risk_flags) if r.risk_flags else "none"
            return (
                f"[{date_str}] {r.ticker} | {r.decision.value} | "
                f"conf={r.confidence:.0f} | reasoning={reasoning!r} | "
                f"risk_flags=[{flags}]"
            )

        # Build from newest to oldest so we can drop oldest first when over budget.
        formatted = [_format_one(r) for r in reversed(results)]
        while formatted:
            candidate = "\n".join(reversed(formatted))
            if len(candidate) <= _CHAR_BUDGET:
                return candidate
            formatted.pop()  # drop the oldest (last in reversed list)

        return ""


# ---------------------------------------------------------------------------
# LangGraph node functions
# ---------------------------------------------------------------------------


def load_memory_node(state: TradingDeskState, config: "AppConfig | None" = None) -> dict:
    """Load prior analyses for state['ticker'] and format as context string.

    Args:
        state: Current trading desk state.
        config: Optional AppConfig override. Uses get_config() singleton if None.

    Returns:
        Partial state dict with memory_context set to formatted history string.
    """
    from src.utils.config import get_config

    if config is None:
        config = get_config()

    if not config.memory.enabled:
        return {"memory_context": ""}

    store = AnalysisMemoryStore(
        store_path=config.memory.store_path,
        max_history=config.memory.max_history,
    )
    ticker = state["ticker"]
    results = store.retrieve(ticker)
    return {"memory_context": store.format_memory_context(results)}


def save_memory_node(state: TradingDeskState, config: "AppConfig | None" = None) -> dict:
    """Persist the completed AnalysisResult to the JSONL memory store.

    Args:
        state: Current trading desk state (must have result set).
        config: Optional AppConfig override. Uses get_config() singleton if None.

    Returns:
        Empty dict (no state mutation needed).
    """
    from src.utils.config import get_config

    if config is None:
        config = get_config()

    if not config.memory.enabled:
        return {}

    result = state.get("result")
    if result is None:
        return {}

    store = AnalysisMemoryStore(
        store_path=config.memory.store_path,
        max_history=config.memory.max_history,
    )
    store.save(result)
    return {}
