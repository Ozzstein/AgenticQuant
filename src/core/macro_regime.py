"""HMM-based macro regime detector.

Discovers 4 latent market regimes (risk_on / neutral / risk_off / crisis)
from 8 daily market signals using a Gaussian Hidden Markov Model.

Typical usage::

    from src.core.macro_regime import HMMRegimeDetector, RegimeEvaluator
    detector = HMMRegimeDetector()
    result = detector.fit_and_predict("2018-01-01", "2024-12-31")
    print(result.regime, result.confidence)
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field, model_validator
from scipy.special import logsumexp

from src.data.evidence.store import canonical_bytes, digest
from src.domain.common import EvidenceRecord, RunContext, Sha256, UtcTimestamp, VersionedContract
from src.utils.config import AppConfig, get_config
from src.utils.exceptions import RegimeError
from src.utils.schemas import MacroRegime, RegimeDetectionResult

# The original detector below is retained for explicit diagnostics. Production
# callers use CausalHMMRegimeDetector, which requires time/evidence contracts.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGIME_MODEL_DIR = Path("data/regime_model")
_DEFAULT_MODEL_PATH = _REGIME_MODEL_DIR / "hmm_model.joblib"

# Composite risk score weights for state auto-labelling:
#   high VIX → risky, low breadth → risky, negative momentum → risky
_RISK_SCORE_WEIGHTS: dict[str, float] = {
    "vix_level": 0.35,
    "vix_roc_10d": 0.20,
    "sp500_breadth": -0.25,   # inverted: low breadth = high risk
    "sp500_momentum_20d": -0.20,  # inverted: negative momentum = high risk
}


# ---------------------------------------------------------------------------
# HMMRegimeDetector
# ---------------------------------------------------------------------------


class HMMRegimeDetector:
    """Gaussian HMM regime detector trained on 8 market signals.

    Attributes:
        config: Application configuration with ``macro_regime`` sub-config.
        _model: Fitted ``hmmlearn.hmm.GaussianHMM`` instance or ``None``.
        _state_to_regime: Mapping from HMM state index to ``MacroRegime`` name.
        _last_regime: Most recent predicted regime (for duration smoothing).
        _regime_hold_counter: Number of consecutive days in ``_last_regime``.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._cfg = self.config.macro_regime
        self._model: Any = None  # GaussianHMM after fit()
        self._state_to_regime: dict[int, str] = {}
        self._last_regime: str = MacroRegime.NEUTRAL.value
        self._regime_hold_counter: int = 0

    # ------------------------------------------------------------------
    # Signal fetching
    # ------------------------------------------------------------------

    def fetch_signals(self, start: str, end: str) -> pd.DataFrame:
        """Ambient proxy data cannot satisfy the production macro contract."""
        raise RegimeError("Use macro_signals with actual DGS10/DGS2, historical constituent membership and versioned source inputs; Yahoo/IRX/index-time-above-average are not substitutes")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise signals to rolling z-scores, forward-fill missing values.

        Each signal is normalised to a rolling z-score using a
        ``normalization_window``-day window (default 252).  Missing values
        are forward-filled up to 5 business days before remaining NaN rows
        are dropped.

        Args:
            raw: DataFrame of raw signal values with datetime index.

        Returns:
            Normalised DataFrame ready for HMM fitting/inference.
        """
        window = self._cfg.normalization_window

        normalized: dict[str, pd.Series] = {}
        for col in raw.columns:
            series = raw[col]
            rolling_mean = series.rolling(window, min_periods=max(20, window // 10)).mean()
            rolling_std = series.rolling(window, min_periods=max(20, window // 10)).std()
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.nan)
            z = (series - rolling_mean) / rolling_std
            normalized[col] = z

        df = pd.DataFrame(normalized, index=raw.index)
        df = df.ffill(limit=5)
        df = df.dropna()
        logger.debug("Preprocessed: {} rows after normalisation and NaN-drop", len(df))
        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, signals: pd.DataFrame) -> None:
        """Fit the Gaussian HMM on preprocessed signal data.

        Trains with ``n_init=10`` random initialisations and keeps the model
        with the highest log-likelihood to avoid local optima.

        Args:
            signals: Preprocessed DataFrame (output of :meth:`preprocess`).

        Raises:
            RegimeError: If ``hmmlearn`` is not installed or training fails.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise RegimeError("hmmlearn is not installed. Run: pip install hmmlearn>=0.3") from exc

        if signals.empty:
            raise RegimeError("Cannot fit HMM on empty signal data.")

        n_states = self._cfg.hmm_states
        X = signals.values.astype(float)

        logger.info("Fitting GaussianHMM: n_states={}, n_obs={}, n_features={}", n_states, len(X), X.shape[1])

        best_model = None
        best_score = -np.inf

        for seed in range(10):
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=100,
                    random_state=seed,
                )
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as exc:  # noqa: BLE001
                logger.debug("HMM init seed={} failed: {}", seed, exc)

        if best_model is None:
            raise RegimeError("All HMM initialisation attempts failed.")

        self._model = best_model
        self._state_to_regime = self._auto_label_states(signals)
        logger.info("HMM fitted. State labels: {}", self._state_to_regime)

    def _auto_label_states(self, signals: pd.DataFrame) -> dict[int, str]:
        """Auto-label HMM states as risk_on / neutral / risk_off / crisis.

        Computes a composite risk score for each state's mean values.
        Highest risk → crisis, lowest → risk_on.

        Args:
            signals: Preprocessed DataFrame used during fitting.

        Returns:
            Dict mapping state index → regime name string.
        """
        n_states = self._cfg.hmm_states
        X = signals.values.astype(float)

        # Get the most likely state sequence
        state_seq = self._model.predict(X)

        # Compute per-state mean signal values
        state_means: dict[int, dict[str, float]] = {}
        for s in range(n_states):
            mask = state_seq == s
            if mask.sum() == 0:
                state_means[s] = {col: 0.0 for col in signals.columns}
            else:
                state_means[s] = {col: float(signals[col].values[mask].mean()) for col in signals.columns}

        # Compute composite risk score
        risk_scores: dict[int, float] = {}
        for s, means in state_means.items():
            score = 0.0
            for signal, weight in _RISK_SCORE_WEIGHTS.items():
                if signal in means:
                    score += weight * means[signal]
            risk_scores[s] = score

        # Sort states by risk score ascending
        sorted_states = sorted(risk_scores, key=lambda s: risk_scores[s])

        # Map sorted order to regimes
        regime_names = [
            MacroRegime.RISK_ON.value,
            MacroRegime.NEUTRAL.value,
            MacroRegime.RISK_OFF.value,
            MacroRegime.CRISIS.value,
        ]

        label_map: dict[int, str] = {}
        for rank, state_idx in enumerate(sorted_states):
            if rank < len(regime_names):
                label_map[state_idx] = regime_names[rank]
            else:
                label_map[state_idx] = MacroRegime.NEUTRAL.value

        return label_map

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, signals: pd.DataFrame) -> RegimeDetectionResult:
        """Run inference on (preprocessed) signal data and return a regime result.

        Applies confidence-threshold gating and minimum-duration smoothing.

        Args:
            signals: Preprocessed DataFrame (from :meth:`preprocess`).

        Returns:
            :class:`~src.utils.schemas.RegimeDetectionResult` with regime,
            confidence, state probabilities, and 20-day history.

        Raises:
            RegimeError: If the model has not been fitted yet.
        """
        if self._model is None:
            raise RegimeError("Model has not been fitted. Call fit() or load() first.")

        X = signals.values.astype(float)
        # Posterior state probabilities for each day
        posteriors = self._model.predict_proba(X)  # shape (T, n_states)

        # Use the last day's probabilities for the current regime
        last_probs = posteriors[-1]  # shape (n_states,)
        raw_state = int(np.argmax(last_probs))
        max_prob = float(last_probs[raw_state])

        # Apply confidence threshold
        final_state = self._apply_confidence_threshold(raw_state, last_probs)

        # Apply minimum duration smoothing
        final_regime_str = self._apply_min_duration(
            self._state_to_regime.get(final_state, MacroRegime.NEUTRAL.value)
        )

        # Build state_probabilities dict
        state_probs: dict[str, float] = {}
        for state_idx, prob in enumerate(last_probs):
            regime_name = self._state_to_regime.get(state_idx, f"state_{state_idx}")
            state_probs[regime_name] = round(float(prob), 4)

        # Build signal_values from last row
        last_signal_row = signals.iloc[-1]
        signal_values = {col: round(float(val), 4) for col, val in last_signal_row.items()}

        # Compute 20-day regime history
        history = self.regime_history(signals, n_days=20)

        return RegimeDetectionResult(
            regime=MacroRegime(final_regime_str),
            confidence=round(max_prob, 4),
            state_probabilities=state_probs,
            signal_values=signal_values,
            regime_history_20d=history,
            timestamp=datetime.now(),
        )

    def _apply_confidence_threshold(self, raw_state: int, probs: np.ndarray) -> int:
        """Keep previous regime state if max probability is below threshold.

        Args:
            raw_state: The argmax state from the HMM posterior.
            probs: Array of state probabilities (length = n_states).

        Returns:
            Accepted state index (may differ from raw_state if confidence is low).
        """
        max_prob = float(probs[raw_state])
        if max_prob < self._cfg.confidence_threshold:
            # Revert to the state corresponding to the last confirmed regime
            for state_idx, regime_name in self._state_to_regime.items():
                if regime_name == self._last_regime:
                    return state_idx
        return raw_state

    def _apply_min_duration(self, new_regime: str) -> str:
        """Hold the current regime for at least ``min_regime_days`` before switching.

        Args:
            new_regime: Newly predicted regime name.

        Returns:
            The effective regime name after applying duration smoothing.
        """
        if new_regime == self._last_regime:
            self._regime_hold_counter += 1
            return self._last_regime

        # Potential regime change — check if hold counter has been satisfied
        if self._regime_hold_counter >= self._cfg.min_regime_days:
            self._last_regime = new_regime
            self._regime_hold_counter = 1
            return new_regime
        else:
            # Not yet held long enough — stay in current regime
            self._regime_hold_counter += 1
            return self._last_regime

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def fit_and_predict(self, start: str, end: str) -> RegimeDetectionResult:
        """Fetch signals, preprocess, fit, and predict in one call.

        Args:
            start: Start date "YYYY-MM-DD".
            end: End date "YYYY-MM-DD".

        Returns:
            :class:`~src.utils.schemas.RegimeDetectionResult`.
        """
        raw = self.fetch_signals(start, end)
        processed = self.preprocess(raw)
        self.fit(processed)
        return self.predict(processed)

    def regime_history(self, signals: pd.DataFrame, n_days: int = 20) -> list[str]:
        """Decode the regime label for each of the last ``n_days`` days.

        Args:
            signals: Preprocessed DataFrame.
            n_days: Number of most-recent days to decode.

        Returns:
            List of regime name strings, oldest first.
        """
        if self._model is None:
            return []

        X = signals.values.astype(float)
        # Filter prefix probabilities; full-sequence Viterbi rewrites old states.
        from scipy.special import logsumexp
        alpha = None
        state_seq = []
        transitions = np.log(np.maximum(self._model.transmat_, np.finfo(float).tiny))
        for emission in self._model._compute_log_likelihood(X):
            prior = np.log(np.maximum(self._model.startprob_, np.finfo(float).tiny)) if alpha is None else logsumexp(alpha[:, None]+transitions, axis=0)
            alpha = prior + emission
            alpha -= logsumexp(alpha)
            state_seq.append(int(alpha.argmax()))
        recent = state_seq[-n_days:]
        return [self._state_to_regime.get(int(s), MacroRegime.NEUTRAL.value) for s in recent]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Serialise the fitted model and state labels to disk.

        Args:
            path: File path for the joblib archive.  Defaults to
                  ``data/regime_model/hmm_model.joblib``.

        Returns:
            Path where the model was saved.

        Raises:
            RegimeError: If the model has not been fitted yet.
        """
        import joblib

        if self._model is None:
            raise RegimeError("Cannot save an unfitted model.")

        save_path = Path(path) if path else _DEFAULT_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self._model,
            "state_to_regime": self._state_to_regime,
            "last_regime": self._last_regime,
            "regime_hold_counter": self._regime_hold_counter,
        }
        joblib.dump(payload, save_path)
        logger.info("HMM model saved to {}", save_path)
        return save_path

    def load(self, path: str | Path | None = None) -> None:
        """Deserialise a previously saved model from disk.

        Args:
            path: File path of the joblib archive.  Defaults to
                  ``data/regime_model/hmm_model.joblib``.

        Raises:
            RegimeError: If the file does not exist or cannot be loaded.
        """
        import joblib

        load_path = Path(path) if path else _DEFAULT_MODEL_PATH

        if not load_path.exists():
            raise RegimeError(f"Model file not found: {load_path}")

        try:
            payload = joblib.load(load_path)
            self._model = payload["model"]
            self._state_to_regime = payload["state_to_regime"]
            self._last_regime = payload.get("last_regime", MacroRegime.NEUTRAL.value)
            self._regime_hold_counter = payload.get("regime_hold_counter", 0)
            logger.info("HMM model loaded from {}", load_path)
        except Exception as exc:  # noqa: BLE001
            raise RegimeError(f"Failed to load model from {load_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# RegimeEvaluator
# ---------------------------------------------------------------------------


class RegimeEvaluator:
    """Evaluate HMM regime detector quality via portfolio return analysis.

    The detector is evaluated by its downstream effect on trading performance:
    regime-conditional daily returns, overall Sharpe, and max drawdown.

    Attributes:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def evaluate(
        self,
        detector: HMMRegimeDetector,
        signals: pd.DataFrame,
        returns: pd.Series,
    ) -> dict[str, Any]:
        """Evaluate detector performance against a return series.

        Runs the detector over the full signal history, assigns each day a
        regime label, then computes regime-conditional portfolio returns and
        overall performance metrics.

        Args:
            detector: A *fitted* :class:`HMMRegimeDetector`.
            signals: Preprocessed signal DataFrame (same index as ``returns``).
            returns: Daily portfolio return series aligned with ``signals``.

        Returns:
            Dict with keys:
                - ``overall_sharpe``: annualised Sharpe of the full period.
                - ``max_drawdown``: maximum drawdown.
                - ``regime_conditional_sharpe``: per-regime Sharpe dict.
                - ``regime_transitions``: number of regime changes.
                - ``regime_distribution``: fraction of days per regime.
        """
        if detector._model is None:
            return {"error": "Detector not fitted"}

        # Decode regimes for the full history
        X = signals.values.astype(float)
        state_seq = detector._model.predict(X)
        regime_labels = [
            detector._state_to_regime.get(int(s), MacroRegime.NEUTRAL.value)
            for s in state_seq
        ]
        regime_series = pd.Series(regime_labels, index=signals.index, name="regime")

        # Align returns with regime series
        aligned_returns, aligned_regimes = returns.align(regime_series, join="inner")

        # Overall Sharpe
        overall_sharpe = self._sharpe(aligned_returns)

        # Max drawdown
        max_dd = self._max_drawdown(aligned_returns)

        # Regime-conditional Sharpe
        rc_sharpe = self.regime_conditional_returns(aligned_regimes, aligned_returns)

        # Regime transitions
        transitions = int(sum(
            1 for i in range(1, len(regime_labels)) if regime_labels[i] != regime_labels[i - 1]
        ))

        # Regime distribution
        total = len(regime_labels)
        dist: dict[str, float] = {}
        for reg in [r.value for r in MacroRegime]:
            count = sum(1 for r in regime_labels if r == reg)
            dist[reg] = round(count / total, 4) if total > 0 else 0.0

        return {
            "overall_sharpe": round(overall_sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "regime_conditional_sharpe": rc_sharpe,
            "regime_transitions": transitions,
            "regime_distribution": dist,
        }

    def regime_conditional_returns(
        self,
        regimes: pd.Series,
        returns: pd.Series,
    ) -> dict[str, float]:
        """Compute per-regime annualised Sharpe.

        Args:
            regimes: Series of regime name strings indexed by date.
            returns: Daily return series indexed by date.

        Returns:
            Dict mapping regime name → annualised Sharpe.
        """
        result: dict[str, float] = {}
        for regime in [r.value for r in MacroRegime]:
            mask = regimes == regime
            regime_rets = returns[mask]
            if len(regime_rets) < 5:
                result[regime] = float("nan")
            else:
                result[regime] = round(self._sharpe(regime_rets), 4)
        return result

    def compare_detectors(
        self,
        baseline: HMMRegimeDetector,
        candidate: HMMRegimeDetector,
        signals: pd.DataFrame,
        returns: pd.Series,
    ) -> dict[str, Any]:
        """A/B compare two detectors and return the winner.

        Args:
            baseline: Baseline fitted detector.
            candidate: Candidate fitted detector.
            signals: Preprocessed signal DataFrame.
            returns: Daily return series.

        Returns:
            Dict with ``baseline_sharpe``, ``candidate_sharpe``, and ``winner``.
        """
        baseline_result = self.evaluate(baseline, signals, returns)
        candidate_result = self.evaluate(candidate, signals, returns)

        b_sharpe = baseline_result.get("overall_sharpe", 0.0)
        c_sharpe = candidate_result.get("overall_sharpe", 0.0)
        winner = "candidate" if c_sharpe > b_sharpe else "baseline"

        return {
            "baseline_sharpe": b_sharpe,
            "candidate_sharpe": c_sharpe,
            "improvement": round(c_sharpe - b_sharpe, 4),
            "winner": winner,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(returns: pd.Series, annual_factor: float = 252.0) -> float:
        """Annualised Sharpe ratio (assumes mean / std * sqrt(252))."""
        if len(returns) < 2:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float(mean / std * math.sqrt(annual_factor))

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Maximum drawdown from cumulative return series."""
        if returns.empty:
            return 0.0
        cum = (1 + returns).cumprod()
        rolling_max = cum.expanding().max()
        drawdowns = cum / rolling_max - 1
        return float(drawdowns.min())

# ---------------------------------------------------------------------------
# Versioned causal production path
# ---------------------------------------------------------------------------


MACRO_SIGNALS = (
    "vix_level", "vix_roc_10d", "yield_curve_10y2y", "sp500_breadth",
    "sp500_realized_vol_20d", "sp500_momentum_20d", "dxy_roc_20d", "credit_spread_proxy",
)


class CausalRegimeResult(VersionedContract):
    context: RunContext
    status: Literal["available", "insufficient_data"]
    regime: Literal["risk_on", "neutral", "risk_off", "crisis"] | None = None
    state_probabilities: tuple[tuple[str, float], ...] = ()
    confidence: float | None = Field(default=None, ge=0, le=1)
    model_hash: Sha256 | None = None
    trained_through: UtcTimestamp | None = None
    input_hash: Sha256
    evidence_ids: tuple[str, ...]
    held_reason: str | None = None

    @model_validator(mode="after")
    def coherent_result(self):
        if self.status == "available" and (self.regime is None or self.model_hash is None or self.trained_through is None or self.confidence is None):
            raise ValueError("available regime requires a fitted model and effective state")
        if self.status == "insufficient_data" and self.regime is not None:
            raise ValueError("insufficient data cannot assert a neutral or other regime")
        if self.state_probabilities:
            probabilities = dict(self.state_probabilities)
            if len(self.state_probabilities) != 4 or set(probabilities) != {"risk_on", "neutral", "risk_off", "crisis"}:
                raise ValueError("four distinct state probabilities required")
            if any(not 0 <= p <= 1 for p in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-8):
                raise ValueError("state probabilities must sum to one")
            if self.regime is not None and not math.isclose(self.confidence, probabilities[self.regime], abs_tol=1e-8):
                raise ValueError("confidence is the probability of the effective state")
        return self


def macro_signals(raw: pd.DataFrame, constituent_prices: pd.DataFrame,
                  membership: pd.DataFrame) -> pd.DataFrame:
    """Build eight signals from actual yields and constituent breadth.

    Inputs are already as-of materialized: raw columns vix,dgs10,dgs2,sp500,dxy,
    hyg,tlt. Membership is an explicit boolean historical matrix, not today's
    index list. Missing constituents fail breadth coverage rather than counting
    as below their moving average.
    """
    required = {"vix", "dgs10", "dgs2", "sp500", "dxy", "hyg", "tlt"}
    if required - set(raw) or not constituent_prices.index.equals(membership.index) or not constituent_prices.columns.equals(membership.columns):
        raise RegimeError("actual yields and aligned historical constituent prices/membership required")
    if not raw.index.equals(constituent_prices.index) or not all(pd.api.types.is_bool_dtype(dtype) for dtype in membership.dtypes):
        raise RegimeError("explicit date-aligned boolean historical membership required")
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise RegimeError("macro history must be unique and chronological")
    average = constituent_prices.rolling(200, min_periods=200).mean()
    coverage = (constituent_prices.notna() & average.notna()) | ~membership
    breadth = ((constituent_prices > average) & membership).sum(axis=1) / membership.sum(axis=1).replace(0, np.nan)
    breadth = breadth.where(coverage.all(axis=1))
    return pd.DataFrame({
        "vix_level": raw.vix,
        "vix_roc_10d": raw.vix.pct_change(10, fill_method=None),
        "yield_curve_10y2y": raw.dgs10 - raw.dgs2,
        "sp500_breadth": breadth,
        "sp500_realized_vol_20d": np.log(raw.sp500 / raw.sp500.shift(1)).rolling(20).std() * np.sqrt(252),
        "sp500_momentum_20d": raw.sp500.pct_change(20, fill_method=None),
        "dxy_roc_20d": raw.dxy.pct_change(20, fill_method=None),
        "credit_spread_proxy": -np.log(raw.hyg / raw.tlt),
    }, index=raw.index)


class CausalHMMRegimeDetector:
    """Four-state HMM with causal filtering and durable per-session transitions.

    All fit/inference input availability is explicit. Historical diagnostic
    methods of HMMRegimeDetector are not used to manufacture trading evidence.
    """

    def __init__(self, state_path: str | Path, *, normalization_window: int = 252,
                 min_regime_days: int = 5, confidence_threshold: float = 0.6,
                 n_init: int = 10, n_iter: int = 100, risk_weights: dict[str, float] | None = None):
        if normalization_window < 20 or min_regime_days < 1 or not 0 <= confidence_threshold <= 1 or not 1 <= n_init <= 20:
            raise RegimeError("invalid causal HMM configuration")
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.window, self.min_days, self.threshold = normalization_window, min_regime_days, confidence_threshold
        self.n_init, self.n_iter = n_init, n_iter
        self.risk_weights = dict(_RISK_SCORE_WEIGHTS if risk_weights is None else risk_weights)
        if not self.risk_weights or set(self.risk_weights)-set(MACRO_SIGNALS) or not any(self.risk_weights.values()) or any(not np.isfinite(w) or abs(w)>10 for w in self.risk_weights.values()):
            raise RegimeError("invalid exact macro state labeling weights")
        self.model = None
        self.model_hash = None
        self.trained_through = None
        self.fitted_at = None
        self.training_hash = None
        self.training_code_version = None
        self.labels = {}
        self.initial_alpha = None
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS regime_events (model_hash TEXT, timestamp TEXT, input_hash TEXT, result TEXT, alpha TEXT, held_count INTEGER, PRIMARY KEY(model_hash,timestamp))")

    @contextmanager
    def _db(self):
        import sqlite3
        connection = sqlite3.connect(self.state_path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _validate(self, raw, available_at, context, evidence, source_ids):
        if tuple(raw.columns) != MACRO_SIGNALS or not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
            raise RegimeError("all eight ordered macro signals require timezone-aware dates")
        if raw.empty or not raw.index.is_unique or not raw.index.is_monotonic_increasing or raw.index[-1] > context.as_of:
            raise RegimeError("invalid/future macro history")
        if not isinstance(available_at, pd.Series) or not available_at.index.equals(raw.index) or available_at.isna().any():
            raise RegimeError("aligned explicit macro availability required")
        if any(pd.Timestamp(t).tzinfo is None for t in available_at) or (available_at > raw.index).any():
            raise RegimeError("macro observation was unavailable at its historical decision time")
        if not evidence:
            raise RegimeError("macro evidence provenance required")
        for record in evidence:
            record.require_available(context.as_of)
        records = {record.evidence_id: record for record in evidence}
        if len(records) != len(evidence) or not isinstance(source_ids, pd.Series) or not source_ids.index.equals(raw.index):
            raise RegimeError("distinct evidence and aligned per-row source IDs required")
        for timestamp, ids in source_ids.items():
            if not isinstance(ids, tuple) or not ids or set(ids) - records.keys():
                raise RegimeError("every macro row requires immutable source IDs")
            for source_id in ids:
                record = records[source_id]
                record.require_available(timestamp.to_pydatetime())
                if available_at.loc[timestamp] < max(record.available_at, record.requested_as_of or record.available_at):
                    raise RegimeError("macro availability precedes its source availability")

    def preprocess(self, raw):
        average = raw.rolling(self.window, min_periods=self.window).mean()
        std = raw.rolling(self.window, min_periods=self.window).std()
        # A documented constant feature normalizes to 0; missing data stays NaN.
        return ((raw - average) / std.where(std != 0, 1)).replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _input_hash(raw, evidence, available_at, source_ids):
        used = set(source_id for ids in source_ids for source_id in ids)
        return digest(canonical_bytes({"signals": raw.to_json(orient="split", date_format="iso", double_precision=15),
                                       "availability": [t.isoformat() for t in available_at], "sources": source_ids.tolist(),
                                       "evidence": sorted(e.content_hash() for e in evidence if e.evidence_id in used)}))

    def _forward(self, values, previous=None):
        emissions = self.model._compute_log_likelihood(np.asarray(values, dtype=float))
        result = []
        transition = np.log(np.maximum(self.model.transmat_, np.finfo(float).tiny))
        alpha = np.log(np.maximum(previous, np.finfo(float).tiny)) if previous is not None else None
        for emission in emissions:
            prior = np.log(np.maximum(self.model.startprob_, np.finfo(float).tiny)) if alpha is None else logsumexp(alpha[:, None] + transition, axis=0)
            alpha = prior + emission
            alpha -= logsumexp(alpha)
            result.append(np.exp(alpha))
        return np.asarray(result)

    def fit(self, raw: pd.DataFrame, context: RunContext, *, available_at: pd.Series,
            evidence: tuple[EvidenceRecord, ...], source_ids: pd.Series):
        from hmmlearn.hmm import GaussianHMM
        self._validate(raw, available_at, context, evidence, source_ids)
        normalized = self.preprocess(raw)
        if len(normalized) < 80:
            raise RegimeError("insufficient eligible macro training history after one-year normalization")
        best, score = None, -np.inf
        for seed in range(self.n_init):
            model = GaussianHMM(n_components=4, covariance_type="full", n_iter=self.n_iter, random_state=seed, min_covar=1e-4)
            try:
                model.fit(normalized.values)
                likelihood = model.score(normalized.values)
                if np.isfinite(likelihood) and likelihood > score:
                    best, score = model, likelihood
            except (ValueError, np.linalg.LinAlgError):
                continue
        if best is None:
            raise RegimeError("all causal HMM fits failed")
        self.model = best
        risk = sum(best.means_[:, MACRO_SIGNALS.index(key)] * weight for key, weight in self.risk_weights.items())
        self.labels = {int(state): label for state, label in zip(np.argsort(risk), ("risk_on", "neutral", "risk_off", "crisis"))}
        self.trained_through, self.fitted_at = raw.index[-1].to_pydatetime(), context.as_of
        self.training_code_version = context.code_version
        self.training_hash = self._input_hash(raw, evidence, available_at, source_ids)
        self.initial_alpha = self._forward(normalized.values)[-1]
        self.model_hash = digest(canonical_bytes({"means": best.means_.tolist(), "covars": best.covars_.tolist(),
                                                 "transitions": best.transmat_.tolist(), "start": best.startprob_.tolist(),
                                                 "training": self.training_hash, "window": self.window,
                                                 "min_days": self.min_days, "threshold": self.threshold,
                                                 "risk_weights": self.risk_weights, "n_init": self.n_init, "n_iter": self.n_iter,
                                                 "fitted_at": self.fitted_at.isoformat(), "code_version": context.code_version,
                                                 "implementation": "causal-gaussian-hmm-1"}))
        return self

    def predict(self, raw: pd.DataFrame, context: RunContext, *, available_at: pd.Series,
                evidence: tuple[EvidenceRecord, ...], source_ids: pd.Series) -> CausalRegimeResult:
        self._validate(raw, available_at, context, evidence, source_ids)
        input_hash = self._input_hash(raw, evidence, available_at, source_ids)
        if self.model is None:
            return CausalRegimeResult(context=context, status="insufficient_data", input_hash=input_hash,
                                     evidence_ids=tuple(e.evidence_id for e in evidence), held_reason="model_unavailable")
        normalized = self.preprocess(raw)
        training_raw = raw.loc[:self.trained_through]
        if self._input_hash(training_raw, evidence, available_at.loc[training_raw.index], source_ids.loc[training_raw.index]) != self.training_hash:
            raise RegimeError("training history changed or is incomplete")
        if normalized.empty or normalized.index[-1] != raw.index[-1] or raw.index[-1] <= self.fitted_at:
            return CausalRegimeResult(context=context, status="insufficient_data", input_hash=input_hash,
                                     model_hash=self.model_hash, trained_through=self.trained_through,
                                     evidence_ids=tuple(e.evidence_id for e in evidence), held_reason="insufficient_out_of_sample_signals")
        import json
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT timestamp,input_hash,result,alpha,held_count FROM regime_events WHERE model_hash=? ORDER BY timestamp", (self.model_hash,)).fetchall()
            last = rows[-1] if rows else None
            # Verify every supplied recorded prefix. Future additions never alter
            # the earlier normalization or result, and revised history fails.
            for timestamp, previous_hash, _, _, _ in rows:
                time = pd.Timestamp(timestamp)
                if time <= raw.index[-1]:
                    if self._input_hash(raw.loc[:time], evidence, available_at.loc[:time], source_ids.loc[:time]) != previous_hash:
                        raise RegimeError("historical regime inputs changed; use a new model/run namespace")
            if last and raw.index[-1] < pd.Timestamp(last[0]):
                raise RegimeError("regime state cannot move backwards")
            alpha = np.array(json.loads(last[3])) if last else self.initial_alpha.copy()
            effective = CausalRegimeResult.model_validate_json(last[2]).regime if last else None
            held = last[4] if last else 0
            after = pd.Timestamp(last[0]) if last else pd.Timestamp(self.fitted_at)
            pending = normalized.loc[normalized.index > after]
            if not pending.index.equals(raw.index[raw.index > after]):
                raise RegimeError("macro coverage gap in inference history")
            if not last:
                delayed = normalized.loc[(normalized.index > self.trained_through) & (normalized.index <= self.fitted_at)]
                if not delayed.empty:
                    alpha = self._forward(delayed.values, alpha)[-1]
            result = CausalRegimeResult.model_validate_json(last[2]) if last else None
            for timestamp, row in pending.iterrows():
                alpha = self._forward([row.values], alpha)[0]
                winner = int(alpha.argmax())
                proposed = self.labels[winner]
                reason = None
                if alpha[winner] < self.threshold:
                    reason = "confidence_below_threshold"
                elif effective is not None and proposed != effective and held < self.min_days:
                    reason = "minimum_duration"
                else:
                    if proposed != effective:
                        held = 0
                    effective = proposed
                held = held + 1 if effective is not None else 0
                source_records = tuple(e for e in evidence if e.available_at <= timestamp and (e.requested_as_of is None or e.requested_as_of <= timestamp))
                prefix_hash = self._input_hash(raw.loc[:timestamp], source_records, available_at.loc[:timestamp], source_ids.loc[:timestamp])
                probabilities = tuple((self.labels[i], float(p)) for i, p in enumerate(alpha))
                result = CausalRegimeResult(context=context.model_copy(update={"as_of": timestamp.to_pydatetime()}),
                                           status="available" if effective is not None else "insufficient_data", regime=effective, state_probabilities=probabilities,
                                           confidence=dict(probabilities)[effective] if effective else None, model_hash=self.model_hash,
                                           trained_through=self.trained_through, input_hash=prefix_hash,
                                           evidence_ids=tuple(e.evidence_id for e in source_records), held_reason=reason)
                db.execute("INSERT INTO regime_events VALUES (?,?,?,?,?,?)", (self.model_hash, timestamp.isoformat(), prefix_hash,
                           result.model_dump_json(), json.dumps(alpha.tolist()), held))
            if result is None:
                raise RegimeError("no eligible regime inference events")
        return result.model_copy(update={"context": context})

    def save(self, path: str | Path):
        import importlib.metadata
        import os
        import tempfile

        import joblib
        if self.model is None:
            raise RegimeError("cannot save an unfitted causal HMM")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in self.__dict__.items() if key != "state_path"}
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+".")
        os.close(descriptor)
        try:
            joblib.dump(payload, temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        payload_hash = digest(path.read_bytes())
        Path(str(path)+".sha256").write_text(payload_hash)
        Path(str(path)+".json").write_bytes(canonical_bytes({
            "schema_version": "1.0", "implementation": "causal-gaussian-hmm-1", "model_hash": self.model_hash,
            "payload_hash": payload_hash, "training_input_hash": self.training_hash,
            "trained_through": self.trained_through.isoformat(), "fitted_at": self.fitted_at.isoformat(),
            "code_version": self.training_code_version, "signals": MACRO_SIGNALS,
            "environment": {name: importlib.metadata.version(name) for name in ("hmmlearn", "numpy", "pandas", "scipy")},
            "normalization_window": self.window, "minimum_duration": self.min_days,
            "confidence_threshold": self.threshold, "state_labels": self.labels,
            "risk_weights": self.risk_weights, "n_init": self.n_init, "n_iter": self.n_iter,
        }))
        return path

    def load(self, path: str | Path):
        import joblib
        path = Path(path)
        if digest(path.read_bytes()) != Path(str(path)+".sha256").read_text().strip():
            raise RegimeError("causal HMM artifact integrity failure")
        # Trusted application-owned artifacts only; no provider/generated pickle.
        self.__dict__.update(joblib.load(path))
        return self
