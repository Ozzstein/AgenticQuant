"""Tests for src/agents/memory.py and MemoryConfig."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

from src.agents.memory import AnalysisMemoryStore
from src.utils.config import MemoryConfig
from src.utils.schemas import AnalysisResult, Decision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    ticker: str = "AAPL",
    decision: Decision = Decision.BUY,
    confidence: float = 75.0,
    reasoning: str = "Looks good.",
    risk_flags: list[str] | None = None,
    timestamp: datetime | None = None,
) -> AnalysisResult:
    """Construct a minimal AnalysisResult for testing."""
    return AnalysisResult(
        ticker=ticker,
        decision=decision,
        confidence=confidence,
        reasoning=reasoning,
        risk_flags=risk_flags or [],
        timestamp=timestamp or datetime.now(),
    )


# ---------------------------------------------------------------------------
# TestAnalysisMemoryStore
# ---------------------------------------------------------------------------


class TestAnalysisMemoryStore:
    """Unit tests for AnalysisMemoryStore."""

    # 1. save creates file and appends lines
    def test_save_creates_file_and_appends(self, tmp_path: Path) -> None:
        store_file = tmp_path / "analysis_log.jsonl"
        store = AnalysisMemoryStore(store_path=store_file, max_history=10)

        store.save(_make_result("AAPL"))
        store.save(_make_result("MSFT"))

        lines = store_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        # Each line must be valid JSON
        for line in lines:
            assert json.loads(line)  # raises if invalid

    # 2. save creates parent directories
    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        store_file = tmp_path / "a" / "b" / "c" / "log.jsonl"
        assert not store_file.parent.exists()

        store = AnalysisMemoryStore(store_path=store_file, max_history=10)
        store.save(_make_result("AAPL"))

        assert store_file.exists()

    # 3. retrieve returns empty list when file does not exist
    def test_retrieve_empty_when_no_file(self, tmp_path: Path) -> None:
        store_file = tmp_path / "nonexistent.jsonl"
        store = AnalysisMemoryStore(store_path=store_file, max_history=10)

        result = store.retrieve("AAPL")
        assert result == []

    # 4. retrieve filters by ticker
    def test_retrieve_filters_by_ticker(self, tmp_path: Path) -> None:
        store_file = tmp_path / "log.jsonl"
        store = AnalysisMemoryStore(store_path=store_file, max_history=10)

        store.save(_make_result("AAPL"))
        store.save(_make_result("MSFT"))
        store.save(_make_result("AAPL"))

        aapl_results = store.retrieve("AAPL")
        assert len(aapl_results) == 2
        assert all(r.ticker == "AAPL" for r in aapl_results)

        msft_results = store.retrieve("MSFT")
        assert len(msft_results) == 1
        assert msft_results[0].ticker == "MSFT"

    # 5. retrieve respects max_n
    def test_retrieve_respects_max_n(self, tmp_path: Path) -> None:
        store_file = tmp_path / "log.jsonl"
        store = AnalysisMemoryStore(store_path=store_file, max_history=100)

        base_time = datetime(2024, 1, 1)
        for i in range(15):
            store.save(_make_result("AAPL", timestamp=base_time + timedelta(hours=i)))

        results = store.retrieve("AAPL", max_n=5)
        assert len(results) == 5

    # 6. retrieve is case-insensitive
    def test_retrieve_case_insensitive(self, tmp_path: Path) -> None:
        store_file = tmp_path / "log.jsonl"
        store = AnalysisMemoryStore(store_path=store_file, max_history=10)

        store.save(_make_result("AAPL"))

        assert len(store.retrieve("aapl")) == 1
        assert len(store.retrieve("Aapl")) == 1
        assert len(store.retrieve("AAPL")) == 1

    # 7. format_memory_context returns empty string for empty list
    def test_format_memory_context_empty_list(self, tmp_path: Path) -> None:
        store = AnalysisMemoryStore(store_path=tmp_path / "log.jsonl", max_history=10)
        assert store.format_memory_context([]) == ""

    # 8. format_memory_context contains decision and confidence
    def test_format_memory_context_formats_results(self, tmp_path: Path) -> None:
        store = AnalysisMemoryStore(store_path=tmp_path / "log.jsonl", max_history=10)
        result = _make_result("AAPL", decision=Decision.STRONG_BUY, confidence=90.0)

        context = store.format_memory_context([result])
        assert "STRONG_BUY" in context
        assert "90" in context
        assert "AAPL" in context

    # 9. format_memory_context stays under 2000 chars even with many results
    def test_format_memory_context_truncates(self, tmp_path: Path) -> None:
        store = AnalysisMemoryStore(store_path=tmp_path / "log.jsonl", max_history=100)
        # Create results with long reasoning to stress the budget
        many_results = [
            _make_result(
                "AAPL",
                reasoning="A" * 300,
                risk_flags=["flag1", "flag2", "flag3"],
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
            )
            for i in range(50)
        ]
        context = store.format_memory_context(many_results)
        assert len(context) <= 2000

    # 10. thread safety — concurrent saves all persist
    def test_thread_safety(self, tmp_path: Path) -> None:
        store_file = tmp_path / "log.jsonl"
        store = AnalysisMemoryStore(store_path=store_file, max_history=100)

        n_threads = 10
        errors: list[Exception] = []

        def worker() -> None:
            try:
                store.save(_make_result("AAPL"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        lines = store_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == n_threads
        for line in lines:
            assert json.loads(line)

    # 11. corrupted line is skipped; valid line after it is returned
    def test_corrupted_line_skipped(self, tmp_path: Path) -> None:
        store_file = tmp_path / "log.jsonl"
        good_result = _make_result("AAPL")

        # Write a bad line followed by a valid JSON line
        store_file.write_text(
            "not-valid-json\n" + good_result.model_dump_json() + "\n",
            encoding="utf-8",
        )

        store = AnalysisMemoryStore(store_path=store_file, max_history=10)
        results = store.retrieve("AAPL")
        assert len(results) == 1
        assert results[0].ticker == "AAPL"

# ---------------------------------------------------------------------------
# TestMemoryConfig (standalone config tests)
# ---------------------------------------------------------------------------


class TestMemoryConfig:
    """Tests for MemoryConfig Pydantic model."""

    def test_config_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.enabled is True
        assert cfg.max_history == 10
        assert "analysis_log.jsonl" in cfg.store_path.name

    def test_config_override(self) -> None:
        cfg = MemoryConfig(enabled=False, max_history=25, store_path="/tmp/custom.jsonl")
        assert cfg.enabled is False
        assert cfg.max_history == 25
        assert cfg.store_path == Path("/tmp/custom.jsonl")
