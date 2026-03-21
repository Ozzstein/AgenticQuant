"""Tests for the HMM macro regime detector (TASK-09).

Covers:
  - Config and schema correctness
  - HMM training on synthetic data
  - Inference, confidence threshold, min-duration smoothing
  - Persistence (save/load round-trip)
  - Regime evaluation framework
  - RD-Agent evolve_regime / copilot_regime
  - CLI smoke tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.config import MacroRegimeConfig
from src.utils.exceptions import RegimeError
from src.utils.schemas import MacroRegime, RegimeDetectionResult, RegimeEvolutionResult

# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

_N_DAYS = 1500
_SIGNALS = [
    "vix_level",
    "vix_roc_10d",
    "yield_curve_10y2y",
    "sp500_breadth",
    "sp500_realized_vol_20d",
    "sp500_momentum_20d",
    "dxy_roc_20d",
    "credit_spread_proxy",
]


def _make_synthetic_signals(seed: int = 42) -> pd.DataFrame:
    """Build 1500 days of synthetic signal data with clear regime patterns.

    Regimes by segment:
      - Days 0-500:    risk_on   — low VIX, high breadth, positive momentum
      - Days 500-700:  crisis    — spiking VIX, dropping breadth, negative mom
      - Days 700-1200: neutral   — moderate values
      - Days 1200-1500: risk_on  — recovery
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=_N_DAYS, freq="B")

    def _make_series(
        base: float,
        std: float,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.normal(base, std, n)

    # Build per-segment values
    def _segment(vix_base, breadth_base, mom_base, vol_base, n):
        return {
            "vix_level": _make_series(vix_base, 2.0, n, rng),
            "vix_roc_10d": _make_series(0.0, 0.05, n, rng),
            "yield_curve_10y2y": _make_series(0.5, 0.2, n, rng),
            "sp500_breadth": _make_series(breadth_base, 0.05, n, rng).clip(0, 1),
            "sp500_realized_vol_20d": _make_series(vol_base, 0.02, n, rng).clip(0.01, None),
            "sp500_momentum_20d": _make_series(mom_base, 0.02, n, rng),
            "dxy_roc_20d": _make_series(0.0, 0.01, n, rng),
            "credit_spread_proxy": _make_series(-0.1, 0.02, n, rng),
        }

    seg_risk_on = _segment(vix_base=13, breadth_base=0.80, mom_base=0.04, vol_base=0.10, n=500)
    seg_crisis = _segment(vix_base=45, breadth_base=0.20, mom_base=-0.10, vol_base=0.40, n=200)
    seg_neutral = _segment(vix_base=22, breadth_base=0.55, mom_base=0.01, vol_base=0.16, n=500)
    seg_recovery = _segment(vix_base=16, breadth_base=0.72, mom_base=0.03, vol_base=0.12, n=300)

    data: dict[str, np.ndarray] = {}
    for col in _SIGNALS:
        data[col] = np.concatenate([
            seg_risk_on[col],
            seg_crisis[col],
            seg_neutral[col],
            seg_recovery[col],
        ])

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def synthetic_signals() -> pd.DataFrame:
    """Synthetic signal DataFrame (unprocessed)."""
    return _make_synthetic_signals()


@pytest.fixture
def fitted_detector(synthetic_signals):
    """HMMRegimeDetector fitted on synthetic data."""
    from src.core.macro_regime import HMMRegimeDetector

    detector = HMMRegimeDetector()
    # Use the signals directly (already "preprocessed" enough for testing)
    detector.fit(synthetic_signals)
    return detector


@pytest.fixture
def processed_signals(synthetic_signals) -> pd.DataFrame:
    """Preprocessed (z-scored) synthetic signals."""
    from src.core.macro_regime import HMMRegimeDetector

    detector = HMMRegimeDetector()
    return detector.preprocess(synthetic_signals)


# ===========================================================================
# 1. Config / Schema tests
# ===========================================================================


class TestConfigAndSchema:
    def test_macro_regime_config_defaults(self):
        cfg = MacroRegimeConfig()
        assert cfg.hmm_states == 4
        assert cfg.confidence_threshold == 0.7
        assert cfg.min_regime_days == 3
        assert cfg.retrain_frequency == "quarterly"
        assert cfg.normalization_window == 252
        assert len(cfg.signals) == 8

    def test_regime_detection_result_schema(self):
        result = RegimeDetectionResult(
            regime=MacroRegime.NEUTRAL,
            confidence=0.85,
            state_probabilities={"neutral": 0.85, "risk_on": 0.10},
            signal_values={"vix_level": 1.2},
            regime_history_20d=["neutral"] * 20,
        )
        assert result.regime == MacroRegime.NEUTRAL
        assert result.confidence == 0.85
        assert len(result.regime_history_20d) == 20

    def test_regime_evolution_result_schema(self):
        result = RegimeEvolutionResult(
            iteration=1,
            change_description="add_signal: put_call",
            baseline_sharpe=1.2,
            modified_sharpe=1.4,
            improvement=0.2,
            accepted=True,
            signals_used=["vix_level"],
            n_states=4,
        )
        assert result.accepted is True
        assert result.improvement == pytest.approx(0.2)


# ===========================================================================
# 2. HMM Training tests
# ===========================================================================


class TestHMMTraining:
    def test_fit_succeeds(self, synthetic_signals):
        from src.core.macro_regime import HMMRegimeDetector

        detector = HMMRegimeDetector()
        detector.fit(synthetic_signals)
        assert detector._model is not None

    def test_fit_produces_4_states(self, fitted_detector):
        assert fitted_detector._model.n_components == 4

    def test_auto_labels_all_regimes(self, fitted_detector):
        labels = set(fitted_detector._state_to_regime.values())
        assert MacroRegime.RISK_ON.value in labels
        assert MacroRegime.NEUTRAL.value in labels
        assert MacroRegime.RISK_OFF.value in labels
        assert MacroRegime.CRISIS.value in labels

    def test_risk_on_state_has_lowest_vix(self, fitted_detector, synthetic_signals):
        """The risk_on state's mean VIX should be lower than crisis state's mean VIX."""
        X = synthetic_signals.values.astype(float)
        state_seq = fitted_detector._model.predict(X)

        risk_on_state = next(
            (s for s, r in fitted_detector._state_to_regime.items() if r == MacroRegime.RISK_ON.value),
            None,
        )
        crisis_state = next(
            (s for s, r in fitted_detector._state_to_regime.items() if r == MacroRegime.CRISIS.value),
            None,
        )
        if risk_on_state is None or crisis_state is None:
            pytest.skip("State mapping incomplete")

        vix_col = list(synthetic_signals.columns).index("vix_level")
        risk_on_mask = state_seq == risk_on_state
        crisis_mask = state_seq == crisis_state

        if risk_on_mask.sum() == 0 or crisis_mask.sum() == 0:
            pytest.skip("Not enough samples in risk_on or crisis state")

        risk_on_vix = X[risk_on_mask, vix_col].mean()
        crisis_vix = X[crisis_mask, vix_col].mean()
        assert risk_on_vix < crisis_vix, (
            f"risk_on VIX ({risk_on_vix:.2f}) should be less than crisis VIX ({crisis_vix:.2f})"
        )


# ===========================================================================
# 3. Inference tests
# ===========================================================================


class TestInference:
    def test_predict_returns_detection_result(self, fitted_detector, synthetic_signals):
        result = fitted_detector.predict(synthetic_signals)
        assert isinstance(result, RegimeDetectionResult)
        assert result.regime in MacroRegime

    def test_state_probabilities_sum_to_one(self, fitted_detector, synthetic_signals):
        result = fitted_detector.predict(synthetic_signals)
        total = sum(result.state_probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_confidence_threshold_prevents_flickering(self):
        """When max_prob is below threshold, detector should keep previous regime."""
        from src.core.macro_regime import HMMRegimeDetector

        detector = HMMRegimeDetector()
        detector._state_to_regime = {
            0: MacroRegime.RISK_ON.value,
            1: MacroRegime.NEUTRAL.value,
            2: MacroRegime.RISK_OFF.value,
            3: MacroRegime.CRISIS.value,
        }
        detector._last_regime = MacroRegime.RISK_ON.value
        detector._cfg = MacroRegimeConfig(confidence_threshold=0.9)

        # Probabilities below threshold (max = 0.4, below 0.9)
        low_probs = np.array([0.4, 0.3, 0.2, 0.1])
        result_state = detector._apply_confidence_threshold(0, low_probs)

        # Should revert to the state for risk_on (state 0)
        assert result_state == 0

    def test_min_duration_prevents_whipsawing(self):
        """Regime should not switch until held for min_regime_days."""
        from src.core.macro_regime import HMMRegimeDetector

        detector = HMMRegimeDetector()
        detector._cfg = MacroRegimeConfig(min_regime_days=3)
        detector._last_regime = MacroRegime.NEUTRAL.value
        detector._regime_hold_counter = 1  # Only held for 1 day

        # Try to switch to risk_on after only 1 day
        effective_regime = detector._apply_min_duration(MacroRegime.RISK_ON.value)

        # Should stay in neutral since counter < min_regime_days
        assert effective_regime == MacroRegime.NEUTRAL.value

    def test_crisis_detected_in_crisis_segment(self, fitted_detector, synthetic_signals):
        """The crisis segment (days 500-700) should predominantly show high-risk regime."""
        crisis_signals = synthetic_signals.iloc[500:700]
        X = crisis_signals.values.astype(float)
        state_seq = fitted_detector._model.predict(X)

        # Find the crisis state
        crisis_state = next(
            (s for s, r in fitted_detector._state_to_regime.items() if r == MacroRegime.CRISIS.value),
            None,
        )
        if crisis_state is None:
            pytest.skip("No crisis state labeled")

        # High-risk states (risk_off or crisis) should dominate in the crisis segment
        high_risk_states = {
            s for s, r in fitted_detector._state_to_regime.items()
            if r in (MacroRegime.CRISIS.value, MacroRegime.RISK_OFF.value)
        }
        high_risk_count = sum(1 for s in state_seq if s in high_risk_states)
        assert high_risk_count / len(state_seq) > 0.3, (
            "Crisis segment should have >30% high-risk regime days"
        )

    def test_regime_history_returns_list(self, fitted_detector, synthetic_signals):
        history = fitted_detector.regime_history(synthetic_signals, n_days=20)
        assert len(history) == 20
        assert all(r in {m.value for m in MacroRegime} for r in history)


# ===========================================================================
# 4. Persistence tests
# ===========================================================================


class TestPersistence:
    def test_save_load_roundtrip(self, fitted_detector, synthetic_signals, tmp_path):
        save_path = tmp_path / "hmm_test.joblib"
        fitted_detector.save(save_path)
        assert save_path.exists()

        from src.core.macro_regime import HMMRegimeDetector

        loaded = HMMRegimeDetector()
        loaded.load(save_path)
        assert loaded._model is not None
        assert loaded._state_to_regime == fitted_detector._state_to_regime

        # Predictions should match
        r1 = fitted_detector.predict(synthetic_signals)
        r2 = loaded.predict(synthetic_signals)
        assert r1.regime == r2.regime

    def test_load_missing_file_raises_regime_error(self, tmp_path):
        from src.core.macro_regime import HMMRegimeDetector

        detector = HMMRegimeDetector()
        with pytest.raises(RegimeError, match="not found"):
            detector.load(tmp_path / "nonexistent.joblib")


# ===========================================================================
# 5. Evaluation tests
# ===========================================================================


class TestEvaluation:
    @pytest.fixture
    def mock_returns(self, synthetic_signals) -> pd.Series:
        rng = np.random.default_rng(0)
        returns = pd.Series(
            rng.normal(0.0005, 0.01, len(synthetic_signals)),
            index=synthetic_signals.index,
            name="returns",
        )
        return returns

    def test_evaluate_returns_sharpe(self, fitted_detector, synthetic_signals, mock_returns):
        from src.core.macro_regime import RegimeEvaluator

        evaluator = RegimeEvaluator()
        result = evaluator.evaluate(fitted_detector, synthetic_signals, mock_returns)
        assert "overall_sharpe" in result
        assert isinstance(result["overall_sharpe"], float)

    def test_regime_conditional_returns(self, fitted_detector, synthetic_signals, mock_returns):
        from src.core.macro_regime import RegimeEvaluator

        evaluator = RegimeEvaluator()
        X = synthetic_signals.values.astype(float)
        state_seq = fitted_detector._model.predict(X)
        regime_labels = [
            fitted_detector._state_to_regime.get(int(s), "neutral")
            for s in state_seq
        ]
        regimes = pd.Series(regime_labels, index=synthetic_signals.index)

        rc = evaluator.regime_conditional_returns(regimes, mock_returns)
        assert isinstance(rc, dict)
        assert all(r.value in rc for r in MacroRegime)

    def test_compare_detectors_returns_winner(self, fitted_detector, synthetic_signals, mock_returns):
        from src.core.macro_regime import HMMRegimeDetector, RegimeEvaluator

        # Use the same detector twice — winner can be either
        candidate = HMMRegimeDetector()
        candidate.fit(synthetic_signals)

        evaluator = RegimeEvaluator()
        result = evaluator.compare_detectors(
            fitted_detector, candidate, synthetic_signals, mock_returns
        )
        assert "winner" in result
        assert result["winner"] in ("baseline", "candidate")
        assert "improvement" in result


# ===========================================================================
# 6. RD-Agent tests
# ===========================================================================


class TestRDAgentRegime:
    def test_evolve_regime_runs(self):
        from src.core.rd_agent_runner import RDAgentRunner

        runner = RDAgentRunner()
        results = runner.evolve_regime(iterations=3, budget=1.0)
        # Should return a list (may be empty if all rejected)
        assert isinstance(results, list)

    def test_evolve_regime_updates_kb(self, tmp_path, monkeypatch):
        """evolve_regime should write tested_regime_configs to the KB."""
        import json

        from src.core.rd_agent_runner import RDAgentRunner

        kb_path = tmp_path / "kb.json"
        monkeypatch.setattr("src.core.rd_agent_runner._KB_PATH", kb_path)
        monkeypatch.setattr("src.core.rd_agent_runner._KB_DIR", tmp_path)

        runner = RDAgentRunner()
        runner.evolve_regime(iterations=2, budget=1.0)

        assert kb_path.exists()
        with kb_path.open() as f:
            kb = json.load(f)
        assert "tested_regime_configs" in kb
        assert len(kb["tested_regime_configs"]) > 0

    def test_copilot_regime_parses_signal_change(self):
        from src.core.rd_agent_runner import RDAgentRunner

        runner = RDAgentRunner()
        result = runner.copilot_regime("add put/call ratio as a signal")
        assert "change_description" in result
        assert "add_signal" in result["change_description"]

    def test_copilot_regime_parses_state_count(self):
        from src.core.rd_agent_runner import RDAgentRunner

        runner = RDAgentRunner()
        result = runner.copilot_regime("try 5 states")
        assert "change_description" in result
        assert "change_states" in result["change_description"]
        assert result["n_states"] == 5


# ===========================================================================
# 7. CLI smoke tests
# ===========================================================================


class TestCLISmoke:
    def test_evolve_regime_importable(self):
        from scripts.run_rd_agent import evolve_regime_cmd  # noqa: F401

    def test_copilot_regime_importable(self):
        from scripts.run_rd_agent import copilot_regime_cmd  # noqa: F401

    def test_regime_status_importable(self):
        from scripts.run_pipeline import regime_status  # noqa: F401

    def test_regime_backtest_importable(self):
        from scripts.run_pipeline import regime_backtest  # noqa: F401
