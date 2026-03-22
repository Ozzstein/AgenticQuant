"""Tests for src/agents/memory.py and MemoryConfig."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

from src.agents.memory import AnalysisMemoryStore
from src.utils.config import AppConfig, MemoryConfig
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


# ---------------------------------------------------------------------------
# TestMemoryGraphNodes
# ---------------------------------------------------------------------------


class TestMemoryGraphNodes:
    """Tests for load_memory_node and save_memory_node."""

    def _make_config(self, tmp_path: Path, enabled: bool = True) -> AppConfig:
        return AppConfig(memory=MemoryConfig(enabled=enabled, store_path=tmp_path / "test.jsonl"))

    def _make_state(self, result=None) -> dict:
        return {
            "ticker": "AAPL",
            "asset_class": "EQUITY",
            "qlib_context": "",
            "market_context": "",
            "macro_regime": "neutral",
            "macro_confidence": 50.0,
            "memory_context": "",
            "agent_reports": {},
            "debate_round": 0,
            "debate_transcript": [],
            "risk_veto": False,
            "risk_veto_reason": "",
            "result": result,
        }

    def test_load_memory_returns_dict_with_key(self, tmp_path: Path) -> None:
        """load_memory_node always returns a dict containing 'memory_context'."""
        from src.agents.memory import load_memory_node

        config = self._make_config(tmp_path)
        result = load_memory_node(self._make_state(), config=config)
        assert isinstance(result, dict)
        assert "memory_context" in result

    def test_load_memory_empty_when_no_history(self, tmp_path: Path) -> None:
        """No prior analyses → memory_context is an empty string."""
        from src.agents.memory import load_memory_node

        config = self._make_config(tmp_path)
        result = load_memory_node(self._make_state(), config=config)
        assert result["memory_context"] == ""

    def test_load_memory_returns_context_when_history_exists(self, tmp_path: Path) -> None:
        """After one saved analysis, load_memory_node returns a non-empty context string."""
        from src.agents.memory import AnalysisMemoryStore, load_memory_node

        config = self._make_config(tmp_path)
        analysis = _make_result("AAPL", decision=Decision.BUY, confidence=80.0)
        store = AnalysisMemoryStore(store_path=tmp_path / "test.jsonl", max_history=10)
        store.save(analysis)

        result = load_memory_node(self._make_state(), config=config)
        assert result["memory_context"] != ""
        assert "AAPL" in result["memory_context"]

    def test_load_memory_disabled_returns_empty(self, tmp_path: Path) -> None:
        """When memory.enabled=False, load_memory_node returns empty string without reading."""
        from src.agents.memory import AnalysisMemoryStore, load_memory_node

        config = self._make_config(tmp_path, enabled=False)
        # Pre-populate store to confirm it's not read
        analysis = _make_result("AAPL")
        store = AnalysisMemoryStore(store_path=tmp_path / "test.jsonl", max_history=10)
        store.save(analysis)

        result = load_memory_node(self._make_state(), config=config)
        assert result == {"memory_context": ""}

    def test_save_memory_persists_result(self, tmp_path: Path) -> None:
        """save_memory_node writes the AnalysisResult to a JSONL file."""
        from src.agents.memory import save_memory_node

        config = self._make_config(tmp_path)
        analysis = _make_result("AAPL", decision=Decision.SELL, confidence=60.0)
        state = self._make_state(result=analysis)

        save_memory_node(state, config=config)

        store_file = tmp_path / "test.jsonl"
        assert store_file.exists()
        lines = store_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    def test_save_memory_noop_when_none(self, tmp_path: Path) -> None:
        """When state['result'] is None, no file is created."""
        from src.agents.memory import save_memory_node

        config = self._make_config(tmp_path)
        state = self._make_state(result=None)

        save_memory_node(state, config=config)

        store_file = tmp_path / "test.jsonl"
        assert not store_file.exists()

    def test_save_memory_disabled_does_not_save(self, tmp_path: Path) -> None:
        """When memory.enabled=False, save_memory_node does not create a file."""
        from src.agents.memory import save_memory_node

        config = self._make_config(tmp_path, enabled=False)
        analysis = _make_result("AAPL")
        state = self._make_state(result=analysis)

        save_memory_node(state, config=config)

        store_file = tmp_path / "test.jsonl"
        assert not store_file.exists()


# ---------------------------------------------------------------------------
# TestMemoryGraphIntegration
# ---------------------------------------------------------------------------


class TestMemoryGraphIntegration:
    """Integration tests for memory nodes + graph + prompt injection."""

    def test_graph_has_memory_nodes(self):
        """Build graph, verify load_memory and save_memory nodes exist."""
        from src.agents.graph import build_trading_desk_graph

        graph = build_trading_desk_graph()
        assert "load_memory" in graph.nodes
        assert "save_memory" in graph.nodes

    def test_initial_state_has_memory_context(self):
        """Verify analyze_ticker would produce state with memory_context key."""
        from unittest.mock import MagicMock, patch

        from src.agents.graph import analyze_ticker
        from src.utils.schemas import AnalysisResult, Decision

        mock_result = AnalysisResult(
            ticker="AAPL", decision=Decision.HOLD, confidence=50.0, reasoning="test"
        )

        captured_states = []

        def mock_invoke(state):
            captured_states.append(state)
            return {"result": mock_result}

        with patch("src.agents.graph.build_trading_desk_graph") as mock_build:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = mock_invoke
            mock_build.return_value = mock_graph
            analyze_ticker("AAPL")

        assert len(captured_states) == 1
        assert "memory_context" in captured_states[0]
        assert captured_states[0]["memory_context"] == ""

    def test_analyst_includes_memory_in_prompt(self):
        """Mock LLM, set memory_context in state, verify HumanMessage contains prior history."""
        from unittest.mock import MagicMock

        from langchain_core.messages import HumanMessage

        from src.agents.fundamental import fundamental_node
        from src.utils.schemas import AgentReport, Decision

        mock_report = AgentReport(
            agent_name="fundamental",
            decision=Decision.HOLD,
            confidence=50.0,
            reasoning="test",
            data_points={},
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_report

        state = {
            "ticker": "AAPL",
            "asset_class": "EQUITY",
            "memory_context": "2024-01-01 | BUY | conf=75 | Test reasoning",
            "qlib_context": "",
            "market_context": "",
            "macro_regime": "neutral",
            "macro_confidence": 50.0,
            "agent_reports": {},
            "debate_round": 0,
            "debate_transcript": [],
            "risk_veto": False,
            "risk_veto_reason": "",
            "result": None,
        }

        fundamental_node(state, llm=mock_llm)

        invoke_calls = mock_llm.with_structured_output.return_value.invoke.call_args_list
        assert len(invoke_calls) == 1
        messages = invoke_calls[0][0][0]  # first positional arg to invoke
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        assert "Prior analysis history for AAPL" in human_messages[0].content

    def test_analyst_omits_memory_when_empty(self):
        """Empty memory_context means prompt does NOT contain 'Prior analysis history'."""
        from unittest.mock import MagicMock

        from langchain_core.messages import HumanMessage

        from src.agents.fundamental import fundamental_node
        from src.utils.schemas import AgentReport, Decision

        mock_report = AgentReport(
            agent_name="fundamental",
            decision=Decision.HOLD,
            confidence=50.0,
            reasoning="test",
            data_points={},
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_report

        state = {
            "ticker": "AAPL",
            "asset_class": "EQUITY",
            "memory_context": "",  # empty
            "qlib_context": "",
            "market_context": "",
            "macro_regime": "neutral",
            "macro_confidence": 50.0,
            "agent_reports": {},
            "debate_round": 0,
            "debate_transcript": [],
            "risk_veto": False,
            "risk_veto_reason": "",
            "result": None,
        }

        fundamental_node(state, llm=mock_llm)

        invoke_calls = mock_llm.with_structured_output.return_value.invoke.call_args_list
        messages = invoke_calls[0][0][0]
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert "Prior analysis history" not in human_messages[0].content

    def test_graph_invoke_produces_result_with_memory(self):
        """Full mocked graph invoke with memory enabled produces a result."""
        from unittest.mock import MagicMock, patch

        from src.agents.graph import analyze_ticker
        from src.utils.schemas import AnalysisResult, Decision

        mock_result = AnalysisResult(
            ticker="AAPL", decision=Decision.BUY, confidence=80.0, reasoning="mocked"
        )

        with patch("src.agents.graph.build_trading_desk_graph") as mock_build:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"result": mock_result}
            mock_build.return_value = mock_graph
            result = analyze_ticker("AAPL")

        assert result is not None
        assert result.ticker == "AAPL"
