"""RD-Agent factor and model discovery — Python-native implementation.

When the ``rdagent`` package is not installed (the common case), all methods
run a pure-Python simulation that mines factors, optimises model hyper-
parameters, and maintains a persistent knowledge base in
``data/rd_knowledge_base/kb.json``.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.logger import get_logger
from src.utils.schemas import FactorDefinition

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Knowledge-base path (created at runtime)
# ---------------------------------------------------------------------------
_KB_DIR = Path("data/rd_knowledge_base")
_KB_PATH = _KB_DIR / "kb.json"

_EMPTY_KB: dict[str, Any] = {
    "tested_factors": [],
    "failed_factors": [],
    "tested_configs": [],
    "discoveries": [],
    "last_run_date": "never",
}

# ---------------------------------------------------------------------------
# Factor templates for proposal generation
# ---------------------------------------------------------------------------
_FACTOR_TEMPLATES: list[dict[str, str]] = [
    {
        "name": "momentum_20d",
        "expression": "close/close.shift(20) - 1",
        "category": "momentum",
    },
    {
        "name": "rsi_14",
        "expression": "ta.rsi(close, 14)",
        "category": "technical",
    },
    {
        "name": "rsi_7",
        "expression": "ta.rsi(close, 7)",
        "category": "technical",
    },
    {
        "name": "macd_signal",
        "expression": "ta.macd(close).macd_signal",
        "category": "technical",
    },
    {
        "name": "bb_pct",
        "expression": "(close - bb_lower) / (bb_upper - bb_lower)",
        "category": "technical",
    },
    {
        "name": "atr_pct",
        "expression": "ta.atr(high, low, close, 14) / close",
        "category": "volatility",
    },
    {
        "name": "obv_momentum",
        "expression": "ta.obv(close, volume).pct_change(20)",
        "category": "volume",
    },
    {
        "name": "momentum_5d",
        "expression": "close/close.shift(5) - 1",
        "category": "momentum",
    },
    {
        "name": "momentum_60d",
        "expression": "close/close.shift(60) - 1",
        "category": "momentum",
    },
    {
        "name": "volume_ratio",
        "expression": "volume / volume.rolling(20).mean()",
        "category": "volume",
    },
]

# LightGBM hyperparameter search grid
_PARAM_GRID: dict[str, list[int | float]] = {
    "n_estimators": [100, 200, 500, 1000],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "num_leaves": [31, 63, 127],
}


class RDAgentRunner:
    """Factor/model discovery runner with Python-native simulation fallback.

    When the ``rdagent`` package is importable the runner can delegate to it;
    otherwise every method executes a fast, deterministic simulation backed by
    the knowledge base at ``data/rd_knowledge_base/kb.json``.

    Attributes:
        config: Full application config including RD-Agent settings.
        _rd_agent_available: ``True`` when the ``rdagent`` package is importable.
    """

    def __init__(self, config: FullAppConfig | None = None) -> None:
        self.config = config or get_full_config()
        self._rd_agent_available = False
        try:
            import rdagent  # noqa: F401, PLC0415

            self._rd_agent_available = True
            logger.info("RD-Agent package detected.")
        except ImportError:
            logger.info("RD-Agent not installed. Using Python-native simulation mode.")

    # ------------------------------------------------------------------
    # Knowledge-base helpers
    # ------------------------------------------------------------------

    def _load_kb(self) -> dict[str, Any]:
        """Load the knowledge base from disk, returning an empty KB if missing.

        Returns:
            KB dict with keys: tested_factors, failed_factors, tested_configs,
            discoveries, last_run_date.
        """
        if not _KB_PATH.exists():
            return dict(_EMPTY_KB)  # shallow copy of defaults
        try:
            with _KB_PATH.open() as fh:
                kb = json.load(fh)
            # Ensure all expected keys exist (backward-compat)
            for key, default in _EMPTY_KB.items():
                kb.setdefault(key, default)
            return kb
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load KB ({}); starting fresh.", exc)
            return dict(_EMPTY_KB)

    def _save_kb(self, kb: dict[str, Any]) -> None:
        """Persist the knowledge base to disk.

        Args:
            kb: Knowledge-base dict to serialise.
        """
        _KB_DIR.mkdir(parents=True, exist_ok=True)
        with _KB_PATH.open("w") as fh:
            json.dump(kb, fh, indent=2)
        logger.debug("Knowledge base saved ({} entries).", _KB_PATH)

    # ------------------------------------------------------------------
    # Factor proposal helpers
    # ------------------------------------------------------------------

    def _propose_factors(self, n: int) -> list[dict[str, str]]:
        """Sample ``n`` factor proposals from the template library.

        Each proposal is a copy of a template, with a unique numeric suffix
        appended to avoid name collisions when multiple proposals share the
        same base name.

        Args:
            n: Number of proposals to generate.

        Returns:
            List of factor proposal dicts (name, expression, category).
        """
        proposals: list[dict[str, str]] = []
        seen: set[str] = set()
        pool = list(_FACTOR_TEMPLATES)
        random.shuffle(pool)

        for template in pool * ((n // len(pool)) + 2):
            if len(proposals) >= n:
                break
            proposal = dict(template)
            base_name = proposal["name"]
            # Make the name unique within this batch
            candidate = base_name
            suffix = 0
            while candidate in seen:
                suffix += 1
                candidate = f"{base_name}_v{suffix}"
            proposal["name"] = candidate
            seen.add(candidate)
            proposals.append(proposal)

        return proposals[:n]

    def _simulate_ic(self) -> float:
        """Simulate an IC value using a Gaussian draw clamped to [-0.15, 0.15].

        Returns:
            Simulated IC float.
        """
        raw = random.gauss(0, 0.05)
        return max(-0.15, min(0.15, raw))

    # ------------------------------------------------------------------
    # Legacy methods (kept for backward-compat)
    # ------------------------------------------------------------------

    def run_factor_search(self, n_iterations: int | None = None) -> list[FactorDefinition]:
        """Discover new alpha factors via RD-Agent.

        Args:
            n_iterations: Number of search iterations. Defaults to
                          ``config.rd_agent.factor_iterations``.

        Returns:
            List of discovered :class:`~src.utils.schemas.FactorDefinition` objects.
            Returns an empty list when RD-Agent is not installed.
        """
        if not self._rd_agent_available:
            logger.warning("RD-Agent not available. Skipping factor search.")
            return []

        iterations = n_iterations or self.config.rd_agent.factor_iterations
        logger.info("RD-Agent factor search: {} iterations (stub).", iterations)
        return []

    def run_model_search(self, n_iterations: int | None = None) -> dict[str, Any]:
        """Optimise model hyperparameters via RD-Agent.

        Args:
            n_iterations: Number of optimisation rounds. Defaults to
                          ``config.rd_agent.model_iterations``.

        Returns:
            Dict of optimised model parameters.
            Returns an empty dict when RD-Agent is not installed.
        """
        if not self._rd_agent_available:
            logger.warning("RD-Agent not available. Skipping model search.")
            return {}

        iterations = n_iterations or self.config.rd_agent.model_iterations
        logger.info("RD-Agent model search: {} iterations (stub).", iterations)
        return {}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_factor_library(self, factors: list[FactorDefinition]) -> None:
        """Persist discovered factors to ``outputs/factor_library.json``.

        Merges *factors* with any existing library on disk (deduplication by
        name, new factors win).

        Args:
            factors: List of :class:`~src.utils.schemas.FactorDefinition` to save.
        """
        output_path = Path(self.config.rd_agent.factor_library_dir) / "factor_library.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing library and merge
        existing: list[dict[str, Any]] = []
        if output_path.exists():
            try:
                with output_path.open() as fh:
                    existing = json.load(fh)
            except (json.JSONDecodeError, OSError):
                existing = []

        existing_by_name = {f["name"]: f for f in existing}
        for factor in factors:
            existing_by_name[factor.name] = factor.model_dump()

        merged = list(existing_by_name.values())
        with output_path.open("w") as fh:
            json.dump(merged, fh, indent=2)

        logger.info("Factor library saved to {}. ({} factors total)", output_path, len(merged))

    def save_model_config(self, config: dict[str, Any]) -> None:
        """Persist optimised model config to ``outputs/best_model_config.yaml``.

        Args:
            config: Dict of model hyperparameters to save.
        """
        output_path = Path(self.config.rd_agent.best_model_config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as fh:
            yaml.safe_dump(config, fh, default_flow_style=False)

        logger.info("Model config saved to {}.", output_path)

    # ------------------------------------------------------------------
    # New methods
    # ------------------------------------------------------------------

    def co_optimize(
        self,
        iterations: int = 10,
        budget: float = 1.0,
        traces: int = 1,
    ) -> dict[str, Any]:
        """Joint factor-model optimisation loop using Python-native simulation.

        For each iteration, one or more factor proposals are generated and
        evaluated against the knowledge base.  Factors that pass the minimum
        IC threshold are accepted into the factor library.

        Args:
            iterations: Number of factor-evaluation iterations.
            budget: Fractional budget multiplier (reserved for future use).
            traces: Number of independent traces to run (handled by
                    :meth:`multi_trace_co_optimize` when >1).

        Returns:
            Dict with keys:
                - ``factors_proposed``: total factor proposals generated.
                - ``factors_accepted``: factors that passed the IC threshold.
                - ``model_configs_tested``: number of model configs evaluated.
                - ``best_ic``: highest IC observed during the run.
        """
        logger.info(
            "co_optimize: {} iterations, budget={:.2f}, traces={}", iterations, budget, traces
        )

        kb = self._load_kb()
        tested_set: set[str] = set(kb.get("tested_factors", []))
        failed_set: set[str] = set(kb.get("failed_factors", []))

        proposals_count = 0
        accepted: list[FactorDefinition] = []
        best_ic = 0.0
        min_ic = self.config.rd_agent.min_ic

        for _ in range(iterations):
            proposals = self._propose_factors(n=3)
            for prop in proposals:
                proposals_count += 1
                name = prop["name"]

                # Skip already-tested or known-failed factors
                if name in tested_set or name in failed_set:
                    logger.debug("Skipping already-known factor: {}", name)
                    continue

                ic = self._simulate_ic()
                tested_set.add(name)

                if abs(ic) >= min_ic:
                    factor = FactorDefinition(
                        name=name,
                        expression=prop["expression"],
                        category=prop["category"],
                        ic_mean=ic,
                        icir=abs(ic) / 0.02,  # simulate ICIR
                        source="rd_agent_sim",
                    )
                    accepted.append(factor)
                    kb["discoveries"].append(
                        {"date": str(date.today()), "factor": name, "ic": round(ic, 6)}
                    )
                    if abs(ic) > abs(best_ic):
                        best_ic = ic
                    logger.info("Factor '{}' accepted (IC={:.4f}).", name, ic)
                else:
                    failed_set.add(name)
                    logger.debug("Factor '{}' rejected (IC={:.4f} < {}).", name, ic, min_ic)

        # Run a quick model config search
        model_result = self.optimize_model(iterations=max(1, iterations // 3), budget=budget)
        model_configs_tested = model_result.get("configs_tested", 0)

        # Update KB
        kb["tested_factors"] = sorted(tested_set)
        kb["failed_factors"] = sorted(failed_set)
        kb["last_run_date"] = str(date.today())
        self._save_kb(kb)

        # Persist accepted factors
        if accepted:
            self.save_factor_library(accepted)

        result = {
            "factors_proposed": proposals_count,
            "factors_accepted": len(accepted),
            "model_configs_tested": model_configs_tested,
            "best_ic": round(best_ic, 6),
        }
        logger.info("co_optimize complete: {}", result)
        return result

    def mine_factors(
        self,
        iterations: int = 10,
        min_ic: float | None = None,
    ) -> list[FactorDefinition]:
        """Factor-only evolution (no model optimisation).

        Args:
            iterations: Number of evaluation iterations.
            min_ic: Minimum absolute IC to accept a factor.  Uses
                    ``config.rd_agent.min_ic`` when ``None``.

        Returns:
            List of accepted :class:`~src.utils.schemas.FactorDefinition` objects.
        """
        effective_min_ic = min_ic if min_ic is not None else self.config.rd_agent.min_ic
        logger.info("mine_factors: {} iterations, min_ic={}", iterations, effective_min_ic)

        accepted: list[FactorDefinition] = []
        kb = self._load_kb()
        tested_set: set[str] = set(kb.get("tested_factors", []))
        failed_set: set[str] = set(kb.get("failed_factors", []))

        for _ in range(iterations):
            proposals = self._propose_factors(n=2)
            for prop in proposals:
                name = prop["name"]
                if name in tested_set or name in failed_set:
                    continue

                ic = self._simulate_ic()
                tested_set.add(name)

                if abs(ic) >= effective_min_ic:
                    factor = FactorDefinition(
                        name=name,
                        expression=prop["expression"],
                        category=prop["category"],
                        ic_mean=ic,
                        icir=abs(ic) / 0.02,
                        source="rd_agent_sim",
                    )
                    accepted.append(factor)
                    kb["discoveries"].append(
                        {"date": str(date.today()), "factor": name, "ic": round(ic, 6)}
                    )
                    logger.info("Factor '{}' mined (IC={:.4f}).", name, ic)
                else:
                    failed_set.add(name)

        kb["tested_factors"] = sorted(tested_set)
        kb["failed_factors"] = sorted(failed_set)
        kb["last_run_date"] = str(date.today())
        self._save_kb(kb)

        if accepted:
            self.save_factor_library(accepted)

        logger.info("mine_factors: {} factors accepted.", len(accepted))
        return accepted

    def optimize_model(
        self,
        iterations: int = 10,
        budget: float = 1.0,
    ) -> dict[str, Any]:
        """Hyperparameter search for LightGBM via random search.

        Args:
            iterations: Number of random configs to evaluate.
            budget: Fractional budget multiplier (caps ``iterations``).

        Returns:
            Dict with keys:
                - ``best_config``: best hyperparameter dict found.
                - ``best_sharpe``: simulated Sharpe ratio of the best config.
                - ``configs_tested``: number of distinct configs evaluated.
        """
        effective_iters = max(1, int(iterations * budget))
        logger.info("optimize_model: {} iterations (budget={:.2f})", effective_iters, budget)

        kb = self._load_kb()
        tested_configs: list[dict[str, Any]] = kb.get("tested_configs", [])
        tested_set: set[str] = {json.dumps(c, sort_keys=True) for c in tested_configs}

        best_config: dict[str, Any] = {}
        best_sharpe: float = -999.0
        configs_tested = 0

        for _ in range(effective_iters):
            candidate = {
                "n_estimators": random.choice(_PARAM_GRID["n_estimators"]),
                "learning_rate": random.choice(_PARAM_GRID["learning_rate"]),
                "max_depth": random.choice(_PARAM_GRID["max_depth"]),
                "num_leaves": random.choice(_PARAM_GRID["num_leaves"]),
            }
            key = json.dumps(candidate, sort_keys=True)
            if key in tested_set:
                logger.debug("Skipping already-tested config: {}", candidate)
                continue

            # Simulate Sharpe: centre around 1.0 with noise
            sharpe = random.gauss(1.0, 0.3)
            tested_set.add(key)
            tested_configs.append(candidate)
            configs_tested += 1

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_config = dict(candidate)
                logger.debug("New best config (Sharpe={:.3f}): {}", sharpe, best_config)

        # Persist
        kb["tested_configs"] = tested_configs
        kb["last_run_date"] = str(date.today())
        self._save_kb(kb)

        if best_config:
            self.save_model_config(best_config)

        result = {
            "best_config": best_config,
            "best_sharpe": round(best_sharpe, 4),
            "configs_tested": configs_tested,
        }
        logger.info("optimize_model complete: Sharpe={:.4f}", best_sharpe)
        return result

    def multi_trace_co_optimize(
        self,
        traces: int = 3,
        iterations: int = 5,
    ) -> dict[str, Any]:
        """Run ``traces`` independent co_optimize calls and merge results.

        Uses :class:`~concurrent.futures.ThreadPoolExecutor` for parallelism.

        Args:
            traces: Number of independent optimisation traces.
            iterations: Iterations per trace.

        Returns:
            Merged dict with keys:
                - ``all_factors``: deduplicated list of accepted factor dicts.
                - ``total_accepted``: total factors accepted across all traces.
                - ``trace_results``: list of per-trace result dicts.
        """
        logger.info("multi_trace_co_optimize: {} traces x {} iterations", traces, iterations)

        trace_results: list[dict[str, Any]] = []
        all_factors_by_name: dict[str, dict[str, Any]] = {}

        def _run_trace(trace_id: int) -> dict[str, Any]:
            logger.debug("Starting trace {}.", trace_id)
            return self.co_optimize(iterations=iterations, budget=1.0, traces=1)

        with ThreadPoolExecutor(max_workers=traces) as executor:
            futures = {executor.submit(_run_trace, i): i for i in range(traces)}
            for future in as_completed(futures):
                trace_id = futures[future]
                try:
                    result = future.result()
                    trace_results.append(result)
                    logger.debug("Trace {} finished: {}", trace_id, result)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Trace {} failed: {}", trace_id, exc)

        # Merge factor libraries from disk (saved by each trace)
        output_path = (
            Path(self.config.rd_agent.factor_library_dir) / "factor_library.json"
        )
        if output_path.exists():
            try:
                with output_path.open() as fh:
                    library = json.load(fh)
                all_factors_by_name = {f["name"]: f for f in library}
            except (json.JSONDecodeError, OSError):
                pass

        total_accepted = sum(r.get("factors_accepted", 0) for r in trace_results)

        merged = {
            "all_factors": list(all_factors_by_name.values()),
            "total_accepted": total_accepted,
            "trace_results": trace_results,
        }
        logger.info(
            "multi_trace_co_optimize complete: {} factors total, {} accepted.",
            len(all_factors_by_name),
            total_accepted,
        )
        return merged

    def implement_paper(self, source: str) -> FactorDefinition | None:
        """Derive a factor from a research paper or local text file.

        Args:
            source: URL (``http``/``arxiv``), arXiv identifier, or local file path.

        Returns:
            :class:`~src.utils.schemas.FactorDefinition` if a factor was
            successfully derived and passed the IC threshold, else ``None``.
        """
        logger.info("implement_paper: source='{}'", source)

        min_ic = self.config.rd_agent.min_ic

        # --- Remote sources: stub ---
        if source.startswith("http") or source.lower().startswith("arxiv"):
            logger.warning(
                "PDF/arXiv parsing not available without keys. "
                "Cannot fetch remote source: '{}'",
                source,
            )
            return None

        # --- Local file: simple keyword extraction ---
        file_path = Path(source)
        if not file_path.exists():
            logger.error("Source file not found: '{}'", source)
            return None

        try:
            text = file_path.read_text(errors="ignore").lower()
        except OSError as exc:
            logger.error("Failed to read source file '{}': {}", source, exc)
            return None

        # Map keywords to factor templates
        keyword_map = {
            "momentum": "momentum_20d",
            "rsi": "rsi_14",
            "macd": "macd_signal",
            "bollinger": "bb_pct",
            "atr": "atr_pct",
            "obv": "obv_momentum",
            "volume": "volume_ratio",
        }

        matched_template: dict[str, str] | None = None
        for keyword, template_name in keyword_map.items():
            if keyword in text:
                for tmpl in _FACTOR_TEMPLATES:
                    if tmpl["name"] == template_name:
                        matched_template = tmpl
                        break
                if matched_template:
                    break

        if matched_template is None:
            # Fall back to a random template
            matched_template = random.choice(_FACTOR_TEMPLATES)
            logger.debug("No keyword match found; using random template '{}'.", matched_template["name"])

        ic = self._simulate_ic()
        if abs(ic) < min_ic:
            logger.info(
                "Paper-derived factor '{}' rejected (IC={:.4f} < {}).",
                matched_template["name"],
                ic,
                min_ic,
            )
            return None

        factor = FactorDefinition(
            name=f"paper_{matched_template['name']}",
            expression=matched_template["expression"],
            category=matched_template["category"],
            ic_mean=ic,
            icir=abs(ic) / 0.02,
            source=f"paper:{file_path.name}",
            description=f"Derived from local file: {file_path.name}",
        )
        self.save_factor_library([factor])
        logger.info("implement_paper: factor '{}' accepted (IC={:.4f}).", factor.name, ic)
        return factor

    def library_status(self) -> dict[str, Any]:
        """Report the current state of the factor library and knowledge base.

        Returns:
            Dict with keys:
                - ``factor_count``: number of factors in the library.
                - ``avg_ic``: mean abs IC across all library factors.
                - ``last_run_date``: ISO date string from the KB, or ``"never"``.
                - ``kb_tested_count``: factors tested so far.
                - ``kb_failed_count``: factors that failed the IC threshold.
        """
        output_path = (
            Path(self.config.rd_agent.factor_library_dir) / "factor_library.json"
        )
        factors: list[dict[str, Any]] = []
        if output_path.exists():
            try:
                with output_path.open() as fh:
                    factors = json.load(fh)
            except (json.JSONDecodeError, OSError):
                factors = []

        avg_ic = (
            sum(abs(f.get("ic_mean", 0.0)) for f in factors) / len(factors) if factors else 0.0
        )

        kb = self._load_kb()

        status = {
            "factor_count": len(factors),
            "avg_ic": round(avg_ic, 6),
            "last_run_date": kb.get("last_run_date", "never"),
            "kb_tested_count": len(kb.get("tested_factors", [])),
            "kb_failed_count": len(kb.get("failed_factors", [])),
        }
        logger.info("library_status: {}", status)
        return status

    def validate_library(self) -> dict[str, Any]:
        """3-way comparison: Alpha158-only vs library-only vs combined.

        Sharpe ratios are simulated with small random noise around fixed
        reference values:
            - Alpha158-only: 0.8
            - Library-only:  0.6
            - Combined:      0.9

        Returns:
            Dict with keys: ``alpha158_sharpe``, ``library_sharpe``,
            ``combined_sharpe``, ``winner``.
        """
        logger.info("validate_library: running 3-way simulation.")

        alpha158_sharpe = round(0.8 + random.gauss(0, 0.05), 4)
        library_sharpe = round(0.6 + random.gauss(0, 0.05), 4)
        combined_sharpe = round(0.9 + random.gauss(0, 0.05), 4)

        sharpes = {
            "alpha158": alpha158_sharpe,
            "library": library_sharpe,
            "combined": combined_sharpe,
        }
        winner = max(sharpes, key=lambda k: sharpes[k])

        result = {
            "alpha158_sharpe": alpha158_sharpe,
            "library_sharpe": library_sharpe,
            "combined_sharpe": combined_sharpe,
            "winner": winner,
        }
        logger.info("validate_library: winner='{}', result={}", winner, result)
        return result

    def reset_knowledge(self) -> None:
        """Clear the knowledge base by writing an empty KB structure to disk.

        Returns:
            None
        """
        _KB_DIR.mkdir(parents=True, exist_ok=True)
        with _KB_PATH.open("w") as fh:
            json.dump(dict(_EMPTY_KB), fh, indent=2)
        logger.info("Knowledge base reset. All prior discoveries cleared.")
