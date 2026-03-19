"""HMM-based macro regime detector.

Discovers 4 latent market regimes (risk_on / neutral / risk_off / crisis)
from 8 daily market signals using a Gaussian Hidden Markov Model.

Typical usage::

    from src.core.macro_regime import HMMRegimeDetector, RegimeEvaluator
    from src.utils.config import get_config

    detector = HMMRegimeDetector()
    result = detector.fit_and_predict("2018-01-01", "2024-12-31")
    print(result.regime, result.confidence)
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import AppConfig, get_config
from src.utils.exceptions import RegimeError
from src.utils.schemas import MacroRegime, RegimeDetectionResult

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
        """Fetch all 8 market signals from yfinance and return a daily DataFrame.

        Columns: vix_level, vix_roc_10d, yield_curve_10y2y, sp500_breadth,
        sp500_realized_vol_20d, sp500_momentum_20d, dxy_roc_20d,
        credit_spread_proxy.

        Args:
            start: Start date string "YYYY-MM-DD".
            end: End date string "YYYY-MM-DD".

        Returns:
            DataFrame indexed by date with one column per signal.
        """
        import yfinance as yf

        logger.info("Fetching macro signals {} → {}", start, end)

        # Fetch raw price series with fallback per-ticker error handling
        raw: dict[str, pd.Series] = {}

        def _fetch(ticker: str) -> pd.Series | None:
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if df.empty:
                    logger.warning("Empty data returned for {}", ticker)
                    return None
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return close.squeeze()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch {}: {}", ticker, exc)
                return None

        raw["vix"] = _fetch("^VIX")
        raw["tnx"] = _fetch("^TNX")   # 10Y yield
        raw["irx"] = _fetch("^IRX")   # 13-week T-bill (proxy for 2Y)
        raw["sp500"] = _fetch("^GSPC")
        raw["dxy"] = _fetch("DX-Y.NYB")
        raw["hyg"] = _fetch("HYG")    # high-yield bond ETF
        raw["tlt"] = _fetch("TLT")    # 20Y treasury ETF

        signals: dict[str, pd.Series] = {}

        # 1. VIX level
        if raw["vix"] is not None:
            signals["vix_level"] = raw["vix"]
        else:
            logger.warning("VIX unavailable; using zeros for vix_level")
            signals["vix_level"] = pd.Series(dtype=float)

        # 2. VIX 10-day ROC
        if raw["vix"] is not None:
            signals["vix_roc_10d"] = raw["vix"].pct_change(10)
        else:
            signals["vix_roc_10d"] = pd.Series(dtype=float)

        # 3. Yield curve: 10Y minus 2Y (using IRX as 2Y proxy, both in %)
        if raw["tnx"] is not None and raw["irx"] is not None:
            signals["yield_curve_10y2y"] = raw["tnx"] - raw["irx"]
        elif raw["tnx"] is not None:
            signals["yield_curve_10y2y"] = raw["tnx"]
        else:
            signals["yield_curve_10y2y"] = pd.Series(dtype=float)

        # 4. S&P 500 breadth (approximate: rolling % of days above 200d SMA)
        #    We use a rolling 252-day "fraction above" its own expanding mean
        if raw["sp500"] is not None:
            sp = raw["sp500"]
            sma200 = sp.rolling(200, min_periods=50).mean()
            signals["sp500_breadth"] = (sp > sma200).rolling(20, min_periods=5).mean()
        else:
            signals["sp500_breadth"] = pd.Series(dtype=float)

        # 5. S&P 500 20-day realized volatility (annualised)
        if raw["sp500"] is not None:
            log_ret = np.log(raw["sp500"] / raw["sp500"].shift(1))
            signals["sp500_realized_vol_20d"] = log_ret.rolling(20, min_periods=10).std() * math.sqrt(252)
        else:
            signals["sp500_realized_vol_20d"] = pd.Series(dtype=float)

        # 6. S&P 500 20-day momentum
        if raw["sp500"] is not None:
            signals["sp500_momentum_20d"] = raw["sp500"].pct_change(20)
        else:
            signals["sp500_momentum_20d"] = pd.Series(dtype=float)

        # 7. DXY 20-day ROC
        if raw["dxy"] is not None:
            signals["dxy_roc_20d"] = raw["dxy"].pct_change(20)
        else:
            signals["dxy_roc_20d"] = pd.Series(dtype=float)

        # 8. Credit spread proxy: negative log(HYG/TLT) — wider spread = higher value = more risk
        if raw["hyg"] is not None and raw["tlt"] is not None:
            # Align on common index
            hyg_aligned, tlt_aligned = raw["hyg"].align(raw["tlt"], join="inner")
            ratio = hyg_aligned / tlt_aligned
            signals["credit_spread_proxy"] = -np.log(ratio)
        else:
            signals["credit_spread_proxy"] = pd.Series(dtype=float)

        # Combine into DataFrame
        df = pd.DataFrame(signals)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        logger.info("Fetched {} rows, {} signals", len(df), len(df.columns))
        return df

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
        state_seq = self._model.predict(X)
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
