"""Portfolio optimization using risk parity, mean-variance, Black-Litterman, HRP, max-diversification."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import AppConfig, get_config


@dataclass
class OptimizationResult:
    """Result of a portfolio optimization.

    Attributes:
        weights: Ticker → weight mapping (sum ≤ 1, remainder = cash).
        risk_contribution: Ticker → % of total portfolio risk.
        method: Name of the optimization method used.
    """

    weights: dict[str, float]
    risk_contribution: dict[str, float]
    method: str = field(default="")


class PortfolioOptimizer:
    """Portfolio optimization engine supporting multiple methods.

    Supports: risk parity (ERC), mean-variance, Black-Litterman, HRP,
    and max diversification. All methods apply hard position-size constraints
    from AppConfig and fall back to equal-weight on solver failure.

    Args:
        config: AppConfig instance. Uses get_config() if None.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or get_config()

    # ------------------------------------------------------------------
    # Public optimization methods
    # ------------------------------------------------------------------

    def risk_parity(
        self,
        cov: pd.DataFrame,
        **constraints,  # noqa: ARG002
    ) -> OptimizationResult:
        """Equal Risk Contribution (ERC) portfolio optimization.

        Uses scipy.optimize.minimize with SLSQP to find weights where each
        asset contributes equally to total portfolio risk.

        Args:
            cov: Covariance matrix as DataFrame with same index/columns (tickers).
            **constraints: Optional keyword args (ignored, for API consistency).

        Returns:
            OptimizationResult with equal risk contribution weights.
        """
        from scipy.optimize import minimize

        tickers = list(cov.columns)
        n = len(tickers)
        cov_matrix = cov.values.astype(float)

        def _erc_objective(w: np.ndarray, cov_mat: np.ndarray) -> float:
            portfolio_var = float(w @ cov_mat @ w)
            marginal_risk = cov_mat @ w
            risk_contrib = w * marginal_risk
            target = portfolio_var / n
            return float(np.sum((risk_contrib - target) ** 2))

        x0 = np.ones(n) / n
        bounds = [(0.0, None)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        try:
            result = minimize(
                _erc_objective,
                x0=x0,
                args=(cov_matrix,),
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            raw = result.x if result.success else x0
        except Exception as exc:  # noqa: BLE001
            logger.warning("risk_parity solver failed: {}. Using equal weight.", exc)
            raw = x0

        weights = self._apply_constraints(raw, tickers)
        risk_contrib = self._compute_risk_contributions(weights, cov)
        logger.debug("risk_parity complete. tickers={}", tickers)
        return OptimizationResult(weights=weights, risk_contribution=risk_contrib, method="risk_parity")

    def mean_variance(
        self,
        returns: pd.Series,
        cov: pd.DataFrame,
        **constraints,  # noqa: ARG002
    ) -> OptimizationResult:
        """Maximum Sharpe Ratio portfolio via CVXPY.

        Maximizes ``returns @ w - lambda * quad_form(w, cov)`` with lambda=1.0,
        long-only, weights summing to ≤ 1.

        Args:
            returns: Expected returns per asset as Series indexed by ticker.
            cov: Covariance matrix DataFrame.
            **constraints: Optional constraints (ignored, for API consistency).

        Returns:
            OptimizationResult with max-return-per-unit-risk weights.
        """
        tickers = list(cov.columns)
        n = len(tickers)
        returns_arr = returns.reindex(tickers).fillna(0.0).values.astype(float)
        cov_matrix = cov.values.astype(float)

        raw = self._solve_mean_variance(returns_arr, cov_matrix, n, tickers)
        weights = self._apply_constraints(raw, tickers)
        risk_contrib = self._compute_risk_contributions(weights, cov)
        logger.debug("mean_variance complete. tickers={}", tickers)
        return OptimizationResult(
            weights=weights, risk_contribution=risk_contrib, method="mean_variance"
        )

    def black_litterman(
        self,
        returns: pd.Series,
        cov: pd.DataFrame,
        views: dict,
        **constraints,  # noqa: ARG002
    ) -> OptimizationResult:
        """Black-Litterman portfolio using views to adjust expected returns.

        Views format: ``{ticker: relative_return_view}``
        e.g. ``{"AAPL": 0.02, "MSFT": -0.01}``.

        BL formula:
        - Prior: equilibrium returns proportional to ``cov @ eq_weights``
        - Posterior: blend prior with views using ``tau=0.025``
        - Posterior returns fed into mean_variance optimizer

        Args:
            returns: Market equilibrium returns (used as prior).
            cov: Covariance matrix.
            views: Dict of ticker → expected excess return view.
            **constraints: Optional (ignored, for API consistency).

        Returns:
            OptimizationResult with BL-adjusted weights.
        """
        tickers = list(cov.columns)
        n = len(tickers)
        cov_matrix = cov.values.astype(float)

        # Equilibrium weights (equal weight as market proxy)
        eq_weights = np.ones(n) / n
        pi = cov_matrix @ eq_weights  # CAPM equilibrium excess returns

        # Build view vector q from views dict (direct views on specific assets)
        view_tickers = [t for t in views if t in tickers]
        if view_tickers:
            k = len(view_tickers)
            p_matrix = np.zeros((k, n))
            q_vec = np.zeros(k)
            for i, ticker in enumerate(view_tickers):
                col_idx = tickers.index(ticker)
                p_matrix[i, col_idx] = 1.0
                q_vec[i] = float(views[ticker])

            tau = 0.025
            tau_sigma = tau * cov_matrix
            omega = tau * p_matrix @ cov_matrix @ p_matrix.T

            try:
                tau_sigma_inv = np.linalg.inv(tau_sigma)
                omega_inv = np.linalg.inv(omega)

                left = np.linalg.inv(tau_sigma_inv + p_matrix.T @ omega_inv @ p_matrix)
                right = tau_sigma_inv @ pi + p_matrix.T @ omega_inv @ q_vec
                mu_bl = left @ right
            except np.linalg.LinAlgError as exc:
                logger.warning("BL matrix inversion failed: {}. Using prior.", exc)
                mu_bl = pi
        else:
            logger.warning("No valid view tickers found. Using prior returns.")
            mu_bl = pi

        bl_returns = pd.Series(mu_bl, index=tickers)
        result = self.mean_variance(bl_returns, cov)
        result.method = "black_litterman"
        logger.debug("black_litterman complete. tickers={}", tickers)
        return result

    def hrp(
        self,
        cov: pd.DataFrame,
        **constraints,  # noqa: ARG002
    ) -> OptimizationResult:
        """Hierarchical Risk Parity via riskfolio-lib.

        Uses riskfolio HRPOpt to construct portfolio. Falls back to equal-weight
        if riskfolio is unavailable or the optimization fails.

        Args:
            cov: Covariance matrix DataFrame.
            **constraints: Optional (ignored, for API consistency).

        Returns:
            OptimizationResult with HRP weights.
        """
        tickers = list(cov.columns)
        n = len(tickers)

        try:
            import riskfolio as rp  # noqa: PLC0415

            # Generate synthetic returns that match the given covariance structure
            cov_array = cov.values.astype(float)
            # Add small regularization for numerical stability
            cov_reg = cov_array + 1e-8 * np.eye(n)
            L = np.linalg.cholesky(cov_reg)
            rng = np.random.default_rng(42)
            z = rng.standard_normal((252, n))
            synthetic_returns = pd.DataFrame(z @ L.T, columns=tickers)

            port = rp.HRPOpt(returns=synthetic_returns)
            w_df = port.optimization(
                model="HRP",
                codependence="pearson",
                covariance="hist",
                obj="MinRisk",
                rm="MV",
                rf=0,
                alpha=0.05,
                leaf_order=True,
            )
            raw = w_df["weights"].reindex(tickers).fillna(0.0).values.astype(float)
            logger.debug("HRP via riskfolio complete. tickers={}", tickers)
        except ImportError:
            logger.warning("riskfolio not available. Falling back to equal weight for HRP.")
            raw = np.ones(n) / n
        except Exception as exc:  # noqa: BLE001
            logger.warning("HRP optimization failed: {}. Falling back to equal weight.", exc)
            raw = np.ones(n) / n

        weights = self._apply_constraints(raw, tickers)
        risk_contrib = self._compute_risk_contributions(weights, cov)
        return OptimizationResult(weights=weights, risk_contribution=risk_contrib, method="hrp")

    def max_diversification(
        self,
        cov: pd.DataFrame,
        **constraints,  # noqa: ARG002
    ) -> OptimizationResult:
        """Maximum Diversification Ratio portfolio via CVXPY.

        Maximizes ``(w @ stds) / sqrt(w @ cov @ w)`` via a parametric
        reformulation: let ``y = w / (w @ stds)``, then maximize ``sum(y * stds)``
        subject to ``y @ cov @ y <= 1``, ``y >= 0``.

        Args:
            cov: Covariance matrix DataFrame.
            **constraints: Optional (ignored, for API consistency).

        Returns:
            OptimizationResult with max diversification weights.
        """
        tickers = list(cov.columns)
        n = len(tickers)
        cov_matrix = cov.values.astype(float)
        stds = np.sqrt(np.diag(cov_matrix))

        raw = self._solve_max_diversification(cov_matrix, stds, n, tickers)
        weights = self._apply_constraints(raw, tickers)
        risk_contrib = self._compute_risk_contributions(weights, cov)
        logger.debug("max_diversification complete. tickers={}", tickers)
        return OptimizationResult(
            weights=weights, risk_contribution=risk_contrib, method="max_diversification"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_constraints(
        self,
        weights_raw: np.ndarray,
        tickers: list[str],
    ) -> dict[str, float]:
        """Apply hard position-size constraints and normalize.

        Caps each weight at ``config.risk.max_position_weight / 100`` (default 10%),
        ensures total weight ≤ 0.95 (min 5% cash), long-only (clips negatives to 0).

        Args:
            weights_raw: Raw optimization weights array.
            tickers: Ticker names corresponding to weights.

        Returns:
            Dict of ticker → constrained weight (sums to ≤ 0.95).
        """
        max_weight = self._config.risk.max_position_weight / 100.0  # e.g. 0.10
        max_total = 1.0 - self._config.risk.min_cash_pct / 100.0    # e.g. 0.95

        w = np.clip(weights_raw, 0.0, None)  # long-only
        w = np.clip(w, 0.0, max_weight)      # cap per-asset

        total = w.sum()
        if total > 1e-9:
            w = w / total * min(total, 1.0)  # renormalize so sum ≤ 1
        else:
            # Degenerate: equal weight
            w = np.ones(len(tickers)) / len(tickers)

        # Enforce max total invested
        if w.sum() > max_total:
            w = w * max_total / w.sum()

        return {ticker: float(w[i]) for i, ticker in enumerate(tickers)}

    def _compute_risk_contributions(
        self,
        weights: dict[str, float],
        cov: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute marginal risk contribution of each asset as % of total risk.

        ``MRC_i = (cov @ w)_i * w_i / (w @ cov @ w)``

        Args:
            weights: Dict of ticker → weight.
            cov: Covariance matrix DataFrame.

        Returns:
            Dict of ticker → % of total portfolio risk (sums to ~100%).
        """
        tickers = list(cov.columns)
        w = np.array([weights.get(t, 0.0) for t in tickers])
        cov_matrix = cov.values.astype(float)

        portfolio_var = float(w @ cov_matrix @ w)
        if portfolio_var < 1e-12:
            n = len(tickers)
            return {t: 100.0 / n for t in tickers}

        mrc = (cov_matrix @ w) * w
        pct = mrc / portfolio_var * 100.0
        return {ticker: float(pct[i]) for i, ticker in enumerate(tickers)}

    def _solve_mean_variance(
        self,
        returns_arr: np.ndarray,
        cov_matrix: np.ndarray,
        n: int,
        tickers: list[str],
    ) -> np.ndarray:
        """Solve the mean-variance optimization problem.

        Attempts CLARABEL → OSQP → SCS solvers in order, falling back to
        equal weight if all fail.

        Args:
            returns_arr: Expected returns array.
            cov_matrix: Covariance matrix.
            n: Number of assets.
            tickers: Ticker names (for logging).

        Returns:
            Raw weight array (not yet constrained).
        """
        try:
            import cvxpy as cp  # noqa: PLC0415

            w = cp.Variable(n)
            lam = 1.0
            objective = cp.Maximize(returns_arr @ w - lam * cp.quad_form(w, cov_matrix))
            cons = [cp.sum(w) <= 1.0, w >= 0]
            prob = cp.Problem(objective, cons)

            for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
                try:
                    prob.solve(solver=solver, ignore_dpp=True)
                    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                        logger.debug("mean_variance solved with solver={}", solver)
                        return np.array(w.value)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Solver {} failed: {}", solver, exc)
                    continue

            logger.warning(
                "All CVXPY solvers failed for mean_variance on {}. Using equal weight.", tickers
            )
        except ImportError:
            logger.warning("cvxpy not available. Using equal weight for mean_variance.")

        return np.ones(n) / n

    def _solve_max_diversification(
        self,
        cov_matrix: np.ndarray,
        stds: np.ndarray,
        n: int,
        tickers: list[str],
    ) -> np.ndarray:
        """Solve the maximum diversification optimization problem.

        Parametric reformulation: let y = w / (w @ stds), then minimize
        ``quad_form(y, cov)`` subject to ``stds @ y == 1``, ``y >= 0``,
        then renormalize y to sum to 1.

        Args:
            cov_matrix: Covariance matrix.
            stds: Per-asset standard deviations.
            n: Number of assets.
            tickers: Ticker names (for logging).

        Returns:
            Raw weight array (not yet constrained).
        """
        try:
            import cvxpy as cp  # noqa: PLC0415

            y = cp.Variable(n)
            objective = cp.Minimize(cp.quad_form(y, cov_matrix))
            cons = [stds @ y == 1.0, y >= 0]
            prob = cp.Problem(objective, cons)

            for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
                try:
                    prob.solve(solver=solver, ignore_dpp=True)
                    if prob.status in ("optimal", "optimal_inaccurate") and y.value is not None:
                        raw = np.array(y.value)
                        total = raw.sum()
                        if total > 1e-9:
                            return raw / total
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Solver {} failed for max_diversification: {}", solver, exc)
                    continue

            logger.warning(
                "All CVXPY solvers failed for max_diversification on {}. Using equal weight.",
                tickers,
            )
        except ImportError:
            logger.warning("cvxpy not available. Using equal weight for max_diversification.")

        return np.ones(n) / n
