"""RD-Agent factor and model discovery — Python-native implementation.

When the ``rdagent`` package is not installed (the common case), all methods
run a pure-Python simulation that mines factors, optimises model hyper-
parameters, and maintains a persistent knowledge base in
``data/rd_knowledge_base/kb.json``.
"""

from __future__ import annotations

import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.logger import get_logger
from src.utils.schemas import FactorDefinition, Strategy

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Knowledge-base path (created at runtime)
# ---------------------------------------------------------------------------
_KB_DIR = Path("data/rd_knowledge_base")
_KB_PATH = _KB_DIR / "kb.json"

# ---------------------------------------------------------------------------
# Strategy library path
# ---------------------------------------------------------------------------
_STRATEGY_DIR = Path("data/strategy_library")

_EMPTY_KB: dict[str, Any] = {
    "tested_factors": [],
    "failed_factors": [],
    "tested_configs": [],
    "discoveries": [],
    "last_run_date": "never",
    "tested_strategies": [],
    "discovered_strategies": [],
    "tested_regime_configs": [],
    "discovered_regime_improvements": [],
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
        self._file_lock = threading.Lock()
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

        with self._file_lock:
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
        with self._file_lock:
            self._save_kb(kb)

        # Persist accepted factors
        if accepted:
            with self._file_lock:
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
        with self._file_lock:
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
        with self._file_lock:
            self._save_kb(kb)

        if accepted:
            with self._file_lock:
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

        with self._file_lock:
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
        with self._file_lock:
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
        if source.startswith("http") or source.lower().startswith("arxiv") or re.match(r"^\d{4}\.\d+", source):
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

    def copilot_factor(self, description: str) -> dict[str, Any]:
        """Implement a factor from a natural language description.

        Scans the description for known keywords, maps them to factor templates,
        evaluates each matched template, and saves passing factors to the library.

        Args:
            description: Natural-language description of the factor hypothesis.

        Returns:
            Dict with keys:
                - ``description``: original description.
                - ``factors_evaluated``: total templates evaluated.
                - ``factors_accepted``: factors that passed IC threshold.
                - ``accepted``: list of accepted factor dicts.
                - ``best_ic``: best IC observed.
        """
        logger.info("copilot_factor: description='{}'", description)

        text = description.lower()
        min_ic = self.config.rd_agent.min_ic

        keyword_map = {
            "momentum": "momentum_20d",
            "rsi": "rsi_14",
            "macd": "macd_signal",
            "bollinger": "bb_pct",
            "atr": "atr_pct",
            "obv": "obv_momentum",
            "volume": "volume_ratio",
        }

        matched_templates: list[dict[str, str]] = []
        for keyword, template_name in keyword_map.items():
            if keyword in text:
                for tmpl in _FACTOR_TEMPLATES:
                    if tmpl["name"] == template_name:
                        matched_templates.append(tmpl)
                        break

        if not matched_templates:
            matched_templates = [random.choice(_FACTOR_TEMPLATES)]
            logger.debug("No keyword match; using random template '{}'.", matched_templates[0]["name"])

        with self._file_lock:
            kb = self._load_kb()

        accepted: list[FactorDefinition] = []
        best_ic = 0.0

        for tmpl in matched_templates:
            ic = self._simulate_ic()
            if abs(ic) >= min_ic:
                factor = FactorDefinition(
                    name=f"copilot_{tmpl['name']}",
                    expression=tmpl["expression"],
                    category=tmpl["category"],
                    ic_mean=ic,
                    icir=abs(ic) / 0.02,
                    source="copilot",
                    description=description,
                )
                accepted.append(factor)
                kb["discoveries"].append(
                    {"date": str(date.today()), "factor": factor.name, "ic": round(ic, 6)}
                )
                if abs(ic) > abs(best_ic):
                    best_ic = ic
                logger.info("copilot_factor: '{}' accepted (IC={:.4f}).", factor.name, ic)
            else:
                logger.debug(
                    "copilot_factor: '{}' rejected (IC={:.4f} < {}).", tmpl["name"], ic, min_ic
                )

        kb["last_run_date"] = str(date.today())
        with self._file_lock:
            self._save_kb(kb)

        if accepted:
            with self._file_lock:
                self.save_factor_library(accepted)

        result: dict[str, Any] = {
            "description": description,
            "factors_evaluated": len(matched_templates),
            "factors_accepted": len(accepted),
            "accepted": [f.model_dump() for f in accepted],
            "best_ic": round(best_ic, 6),
        }
        logger.info("copilot_factor complete: {}", result)
        return result

    def copilot_model(self, source: str) -> dict[str, Any]:
        """Build a model config from a paper/file description.

        Reads a local text file, detects model type via keywords, generates a
        hyperparameter config, simulates a Sharpe ratio, and saves if above threshold.

        Args:
            source: URL, arXiv ID, or local file path describing a model architecture.

        Returns:
            Dict with keys:
                - ``source``: original source string.
                - ``model_name``: detected model type.
                - ``model_params``: hyperparameter config dict.
                - ``simulated_sharpe``: simulated Sharpe ratio.
                - ``saved``: whether the config was persisted.
        """
        logger.info("copilot_model: source='{}'", source)

        _stub = {
            "source": source,
            "model_name": "LightGBM",
            "model_params": {},
            "simulated_sharpe": 0.0,
            "saved": False,
        }

        # Remote sources: stub (no API keys for fetching)
        if (
            source.startswith("http")
            or source.lower().startswith("arxiv")
            or re.match(r"^\d{4}\.\d+", source)
        ):
            logger.warning(
                "PDF/arXiv parsing not available without keys. Cannot fetch: '{}'", source
            )
            return _stub

        file_path = Path(source)
        if not file_path.exists():
            logger.error("Source file not found: '{}'", source)
            return _stub

        try:
            text = file_path.read_text(errors="ignore").lower()
        except OSError as exc:
            logger.error("Failed to read source file '{}': {}", source, exc)
            return _stub

        # Detect model type and build param config
        if "lightgbm" in text or "gradient boost" in text or " gbm" in text:
            model_name = "LightGBM"
            params: dict[str, Any] = {
                "n_estimators": random.choice(_PARAM_GRID["n_estimators"]),
                "learning_rate": random.choice(_PARAM_GRID["learning_rate"]),
                "max_depth": random.choice(_PARAM_GRID["max_depth"]),
                "num_leaves": random.choice(_PARAM_GRID["num_leaves"]),
            }
        elif "catboost" in text or "categorical" in text:
            model_name = "CatBoost"
            params = {
                "iterations": random.choice([100, 200, 500]),
                "learning_rate": random.choice(_PARAM_GRID["learning_rate"]),
                "depth": random.choice([4, 6, 8]),
            }
        elif "xgboost" in text or "extreme gradient" in text:
            model_name = "XGBoost"
            params = {
                "n_estimators": random.choice(_PARAM_GRID["n_estimators"]),
                "learning_rate": random.choice(_PARAM_GRID["learning_rate"]),
                "max_depth": random.choice(_PARAM_GRID["max_depth"]),
            }
        elif "lstm" in text or "recurrent" in text or " rnn" in text:
            model_name = "LSTM"
            params = {
                "hidden_size": random.choice([64, 128, 256]),
                "num_layers": random.choice([1, 2, 3]),
                "dropout": random.choice([0.1, 0.2, 0.3]),
            }
        elif "transformer" in text or "attention" in text:
            model_name = "Transformer"
            params = {
                "d_model": random.choice([64, 128, 256]),
                "nhead": random.choice([4, 8]),
                "num_layers": random.choice([2, 4, 6]),
            }
        else:
            model_name = "LightGBM"
            params = {
                "n_estimators": random.choice(_PARAM_GRID["n_estimators"]),
                "learning_rate": random.choice(_PARAM_GRID["learning_rate"]),
                "max_depth": random.choice(_PARAM_GRID["max_depth"]),
                "num_leaves": random.choice(_PARAM_GRID["num_leaves"]),
            }

        sharpe = random.gauss(1.0, 0.3)
        saved = False
        if sharpe > 0.5:
            self.save_model_config({"model": model_name, **params})
            saved = True

        result: dict[str, Any] = {
            "source": source,
            "model_name": model_name,
            "model_params": params,
            "simulated_sharpe": round(sharpe, 4),
            "saved": saved,
        }
        logger.info("copilot_model: model={}, sharpe={:.4f}, saved={}", model_name, sharpe, saved)
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

    # ------------------------------------------------------------------
    # Strategy library helpers
    # ------------------------------------------------------------------

    def _save_strategy(self, strategy: Strategy) -> Path:
        """Persist a strategy definition to the strategy library on disk.

        Args:
            strategy: :class:`~src.utils.schemas.Strategy` instance to save.

        Returns:
            Path where the strategy was written.
        """
        _STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _STRATEGY_DIR / f"{strategy.name}.json"
        with out_path.open("w") as fh:
            json.dump(strategy.model_dump(mode="json"), fh, indent=2)
        logger.debug("Strategy '{}' saved to {}.", strategy.name, out_path)
        return out_path

    def _simulate_backtest_sharpe(self) -> float:
        """Simulate a backtest Sharpe ratio using a Gaussian draw clamped to [-1.0, 3.0].

        Returns:
            Simulated Sharpe float.
        """
        raw = random.gauss(0.8, 0.5)
        return max(-1.0, min(3.0, raw))

    def _build_strategy_from_keywords(
        self,
        name: str,
        description: str,
    ) -> Strategy:
        """Build a :class:`~src.utils.schemas.Strategy` dict from keyword extraction.

        Parses keywords in *description* to select factor_set, entry_rules, and
        other fields. Used in stub/simulation mode when no LLM is available.

        Args:
            name: Unique strategy name.
            description: Natural language description to parse.

        Returns:
            A populated :class:`~src.utils.schemas.Strategy` instance.
        """
        desc_lower = description.lower()

        # Determine factor set from description keywords
        factor_set: list[str] = ["Alpha158"]
        if "momentum" in desc_lower:
            factor_set.append("momentum_20d")
        if "mean reversion" in desc_lower or "reversion" in desc_lower:
            factor_set.append("rsi_14")
        if "value" in desc_lower:
            factor_set.append("volume_ratio")
        if "crypto" in desc_lower:
            factor_set.append("funding_rate")

        # Determine universe from description keywords
        universe = "SP500"
        if "crypto" in desc_lower:
            universe = "crypto_top30"
        elif "large cap" in desc_lower or "large_cap" in desc_lower:
            universe = "large_cap"
        elif "small cap" in desc_lower or "small_cap" in desc_lower:
            universe = "small_cap"

        # Build entry rules from description keywords
        entry_rules: dict[str, float | int | str | bool] = {
            "rank_threshold": 0.2,
        }
        if "volatility filter" in desc_lower or "with filter" in desc_lower:
            entry_rules["atr_filter"] = True
            entry_rules["max_atr_pct"] = 0.05
        if "sma" in desc_lower or "moving average" in desc_lower:
            entry_rules["sma_cross"] = True

        # Build exit rules
        exit_rules: dict[str, float | int | str | bool] = {
            "stop_loss_pct": 0.08,
            "holding_period_days": 20,
        }
        if "momentum" in desc_lower:
            exit_rules["rank_drop_threshold"] = 0.5

        return Strategy(
            name=name,
            description=description,
            factor_set=factor_set,
            model="LightGBM",
            universe=universe,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            source="rd_agent_copilot_sim",
        )

    # ------------------------------------------------------------------
    # Strategy evolution and copilot
    # ------------------------------------------------------------------

    def evolve_strategies(
        self,
        iterations: int = 10,
        budget: float = 15.0,
        traces: int = 1,
    ) -> list[dict]:
        """Evolve trading strategies via LLM proposal + backtest validation loop.

        For each iteration, a strategy is proposed (via LLM when available, or
        synthetically in stub mode), backtested, and saved to the strategy
        library if it meets the minimum Sharpe threshold.

        Args:
            iterations: Number of strategy evolution iterations.
            budget: Maximum API budget in USD.
            traces: Number of parallel research threads.

        Returns:
            List of validated strategy dicts saved to data/strategy_library/.
        """
        if traces > 1:
            logger.warning("evolve_strategies: traces>1 not yet parallelised; running single trace.")
        logger.info(
            "Starting strategy evolution: {} iterations, budget ${:.1f}", iterations, budget
        )

        if not self._rd_agent_available:
            logger.warning(
                "RD-Agent not installed. Running Python-native strategy evolution simulation."
            )

        with self._file_lock:
            kb = self._load_kb()

        # Track which strategy names have already been tested (reuse KB)
        tested_strategies: set[str] = set(kb.get("tested_strategies", []))

        min_sharpe = getattr(self.config, "strategy", None)
        min_sharpe = min_sharpe.min_backtest_sharpe if min_sharpe is not None else 0.5

        saved: list[dict] = []

        _strategy_archetypes = [
            ("momentum", ["Alpha158", "momentum_20d", "momentum_60d"], "SP500"),
            ("mean_reversion", ["Alpha158", "rsi_14", "bb_pct"], "SP500"),
            ("value_quality", ["Alpha158", "volume_ratio"], "large_cap"),
            ("volatility_breakout", ["Alpha158", "atr_pct", "bb_pct"], "SP500"),
            ("crypto_momentum", ["Alpha158", "momentum_20d", "funding_rate"], "crypto_top30"),
        ]

        for i in range(iterations):
            archetype_name, factor_set, universe = _strategy_archetypes[i % len(_strategy_archetypes)]
            strategy_name = f"evolved_{archetype_name}_{i}"

            if strategy_name in tested_strategies:
                logger.debug("Skipping already-tested strategy: {}", strategy_name)
                continue

            tested_strategies.add(strategy_name)

            # Simulate backtest
            backtest_sharpe = self._simulate_backtest_sharpe()
            backtest_max_drawdown = round(random.uniform(-0.25, -0.05), 4)
            validated = backtest_sharpe >= min_sharpe

            strategy = Strategy(
                name=strategy_name,
                description=f"Evolved {archetype_name} strategy (iteration {i})",
                factor_set=factor_set,
                model="LightGBM",
                universe=universe,
                entry_rules={"rank_threshold": 0.2, "min_adv_usd": 1_000_000},
                exit_rules={"stop_loss_pct": 0.08, "holding_period_days": 20},
                backtest_sharpe=round(backtest_sharpe, 4),
                backtest_max_drawdown=backtest_max_drawdown,
                validated=validated,
                source="rd_agent_evolve_sim",
            )

            if validated:
                with self._file_lock:
                    self._save_strategy(strategy)
                saved.append(strategy.model_dump(mode="json"))
                kb.setdefault("discovered_strategies", []).append(
                    {
                        "date": str(date.today()),
                        "strategy": strategy_name,
                        "sharpe": round(backtest_sharpe, 4),
                    }
                )
                logger.info(
                    "Strategy '{}' accepted (Sharpe={:.4f}).", strategy_name, backtest_sharpe
                )
            else:
                logger.debug(
                    "Strategy '{}' rejected (Sharpe={:.4f} < {}).",
                    strategy_name,
                    backtest_sharpe,
                    min_sharpe,
                )

        # Persist updated knowledge base
        kb["tested_strategies"] = sorted(tested_strategies)
        kb["last_run_date"] = str(date.today())
        with self._file_lock:
            self._save_kb(kb)

        logger.info(
            "evolve_strategies complete: {}/{} strategies accepted.", len(saved), iterations
        )
        return saved

    def copilot_strategy(self, description: str) -> dict:
        """Convert English strategy description to a validated Strategy definition.

        Parses the natural language *description* into structured strategy
        parameters (factor_set, model, entry_rules, etc.), runs a simulated
        backtest, and returns the strategy dict with backtest results.

        Args:
            description: Natural language description of the desired strategy.

        Returns:
            Strategy dict with backtest results, or error dict on failure.
        """
        logger.info("Running strategy copilot for: {}", description)

        if not self._rd_agent_available:
            logger.warning(
                "RD-Agent not installed. Using keyword-based strategy parser."
            )

        try:
            # Derive a safe strategy name from description
            safe_name = re.sub(r"[^a-z0-9_]", "_", description.lower())[:40].strip("_")
            strategy_name = f"copilot_{safe_name}"

            strategy = self._build_strategy_from_keywords(
                name=strategy_name,
                description=description,
            )

            # Simulate backtest
            backtest_sharpe = self._simulate_backtest_sharpe()
            backtest_max_drawdown = round(random.uniform(-0.25, -0.05), 4)
            min_sharpe = getattr(self.config, "strategy", None)
            min_sharpe = min_sharpe.min_backtest_sharpe if min_sharpe is not None else 0.5
            validated = backtest_sharpe >= min_sharpe

            strategy.backtest_sharpe = round(backtest_sharpe, 4)
            strategy.backtest_max_drawdown = backtest_max_drawdown
            strategy.validated = validated

            # Save regardless of validation result (user explicitly requested it)
            with self._file_lock:
                self._save_strategy(strategy)

            result = strategy.model_dump(mode="json")
            logger.info(
                "copilot_strategy: strategy '{}' created (Sharpe={:.4f}, validated={}).",
                strategy_name,
                backtest_sharpe,
                validated,
            )
            return result

        except Exception as exc:  # noqa: BLE001
            logger.error("copilot_strategy failed: {}", exc)
            return {"error": str(exc), "description": description}

    # ------------------------------------------------------------------
    # Regime evolution and copilot
    # ------------------------------------------------------------------

    def evolve_regime(
        self,
        iterations: int = 10,
        budget: float = 10.0,
        traces: int = 1,
    ) -> list[dict]:
        """Evolve the HMM regime detector via simulated backtest comparison loop.

        For each iteration, one of 6 change archetypes is proposed (add signal,
        remove signal, change state count, adjust confidence threshold, or
        change normalisation window).  The change is evaluated by simulating
        baseline vs modified Sharpe ratios.  Accepted improvements are saved
        to the knowledge base.

        Args:
            iterations: Number of regime evolution iterations.
            budget: Maximum API budget in USD (reserved for future use).
            traces: Number of parallel research threads (reserved).

        Returns:
            List of accepted improvement dicts.
        """
        from src.utils.schemas import RegimeEvolutionResult

        if traces > 1:
            logger.warning("evolve_regime: traces>1 not yet parallelised; running single trace.")
        logger.info(
            "Starting regime evolution: {} iterations, budget ${:.1f}", iterations, budget
        )

        with self._file_lock:
            kb = self._load_kb()

        tested_configs: list[str] = [
            str(c) for c in kb.get("tested_regime_configs", [])
        ]
        tested_set: set[str] = set(tested_configs)

        # 6 change archetypes cycled deterministically
        _archetypes = [
            ("add_signal", {"signal": "put_call_ratio"}),
            ("remove_signal", {"signal": "dxy_roc_20d"}),
            ("change_states", {"n_states": 5}),
            ("change_states", {"n_states": 3}),
            ("change_threshold", {"confidence_threshold": 0.8}),
            ("change_window", {"normalization_window": 126}),
        ]

        accepted: list[dict] = []

        for i in range(iterations):
            change_type, change_params = _archetypes[i % len(_archetypes)]
            change_key = f"{change_type}:{json.dumps(change_params, sort_keys=True)}"

            if change_key in tested_set:
                logger.debug("Skipping already-tested regime config: {}", change_key)
                continue

            tested_set.add(change_key)

            # Simulate baseline and modified Sharpe
            baseline_sharpe = self._simulate_backtest_sharpe()
            # Small perturbation for modified
            modified_sharpe = self._simulate_backtest_sharpe()
            improvement = modified_sharpe - baseline_sharpe
            accepted_flag = modified_sharpe > baseline_sharpe

            # Determine signals_used and n_states for the result
            current_signals = self.config.macro_regime.signals if hasattr(self.config, "macro_regime") else [
                "vix_level", "vix_roc_10d", "yield_curve_10y2y", "sp500_breadth",
                "sp500_realized_vol_20d", "sp500_momentum_20d", "dxy_roc_20d", "credit_spread_proxy",
            ]
            signals_used = list(current_signals)
            n_states = 4

            if change_type == "add_signal":
                new_signal = change_params.get("signal", "")
                if new_signal and new_signal not in signals_used:
                    signals_used.append(new_signal)
            elif change_type == "remove_signal":
                rm_signal = change_params.get("signal", "")
                signals_used = [s for s in signals_used if s != rm_signal]
            elif change_type == "change_states":
                n_states = change_params.get("n_states", 4)

            change_description = f"{change_type}: {change_params}"

            result = RegimeEvolutionResult(
                iteration=i,
                change_description=change_description,
                baseline_sharpe=round(baseline_sharpe, 4),
                modified_sharpe=round(modified_sharpe, 4),
                improvement=round(improvement, 4),
                accepted=accepted_flag,
                signals_used=signals_used,
                n_states=n_states,
            )

            if accepted_flag:
                accepted.append(result.model_dump(mode="json"))
                kb.setdefault("discovered_regime_improvements", []).append({
                    "date": str(date.today()),
                    "change": change_description,
                    "improvement": round(improvement, 4),
                    "modified_sharpe": round(modified_sharpe, 4),
                })
                logger.info(
                    "Regime change '{}' accepted (Sharpe improvement={:.4f}).",
                    change_description,
                    improvement,
                )
            else:
                logger.debug(
                    "Regime change '{}' rejected (improvement={:.4f}).",
                    change_description,
                    improvement,
                )

        # Persist updated KB
        kb["tested_regime_configs"] = sorted(tested_set)
        kb["last_run_date"] = str(date.today())
        with self._file_lock:
            self._save_kb(kb)

        logger.info(
            "evolve_regime complete: {}/{} changes accepted.", len(accepted), iterations
        )
        return accepted

    def copilot_regime(self, description: str) -> dict:
        """Apply a user-described change to the regime detector and evaluate it.

        Parses natural language keywords to build a regime change dict
        (add/remove signal, try N states, adjust threshold), simulates a
        backtest, and returns the result.

        Args:
            description: Natural language description of the desired change.

        Returns:
            Dict with change details and simulated performance impact.
        """
        from src.utils.schemas import RegimeEvolutionResult

        logger.info("Running regime copilot for: {}", description)
        desc_lower = description.lower()

        # Parse keywords → change dict
        change_type = "add_signal"
        change_params: dict[str, str | int | float] = {}

        if "add" in desc_lower or "include" in desc_lower:
            change_type = "add_signal"
            # Extract signal name (everything after "add" or "include")
            for keyword in ["put_call", "put/call", "copper", "copper/gold", "credit"]:
                if keyword in desc_lower:
                    change_params["signal"] = keyword.replace("/", "_").replace(" ", "_")
                    break
            if not change_params:
                # Fallback: extract the last word in the description as signal name
                words = re.sub(r"[^a-z0-9_ ]", "", desc_lower).split()
                change_params["signal"] = words[-1] if words else "new_signal"

        elif "remove" in desc_lower or "drop" in desc_lower:
            change_type = "remove_signal"
            for sig in ["dxy", "vix_roc", "credit_spread", "sp500_breadth"]:
                if sig in desc_lower:
                    change_params["signal"] = sig
                    break
            if not change_params:
                change_params["signal"] = "dxy_roc_20d"

        elif "state" in desc_lower or "states" in desc_lower:
            change_type = "change_states"
            # Extract integer from description
            numbers = re.findall(r"\b(\d+)\b", description)
            n_states = int(numbers[0]) if numbers else 4
            change_params["n_states"] = n_states

        elif "threshold" in desc_lower or "confidence" in desc_lower:
            change_type = "change_threshold"
            numbers = re.findall(r"\b(0\.\d+)\b", description)
            threshold = float(numbers[0]) if numbers else 0.8
            change_params["confidence_threshold"] = threshold

        elif "window" in desc_lower or "normaliz" in desc_lower:
            change_type = "change_window"
            numbers = re.findall(r"\b(\d+)\b", description)
            window = int(numbers[0]) if numbers else 126
            change_params["normalization_window"] = window

        # Simulate evaluation
        baseline_sharpe = self._simulate_backtest_sharpe()
        modified_sharpe = self._simulate_backtest_sharpe()
        improvement = modified_sharpe - baseline_sharpe
        accepted = modified_sharpe > baseline_sharpe

        # Determine signals_used and n_states
        current_signals = self.config.macro_regime.signals if hasattr(self.config, "macro_regime") else [
            "vix_level", "vix_roc_10d", "yield_curve_10y2y", "sp500_breadth",
            "sp500_realized_vol_20d", "sp500_momentum_20d", "dxy_roc_20d", "credit_spread_proxy",
        ]
        signals_used = list(current_signals)
        n_states = 4

        if change_type == "add_signal":
            new_sig = str(change_params.get("signal", ""))
            if new_sig and new_sig not in signals_used:
                signals_used.append(new_sig)
        elif change_type == "remove_signal":
            rm_sig = str(change_params.get("signal", ""))
            signals_used = [s for s in signals_used if s != rm_sig]
        elif change_type == "change_states":
            n_states = int(change_params.get("n_states", 4))

        result = RegimeEvolutionResult(
            iteration=0,
            change_description=f"{change_type}: {change_params}",
            baseline_sharpe=round(baseline_sharpe, 4),
            modified_sharpe=round(modified_sharpe, 4),
            improvement=round(improvement, 4),
            accepted=accepted,
            signals_used=signals_used,
            n_states=n_states,
        )

        logger.info(
            "copilot_regime: change='{}', accepted={}, improvement={:.4f}",
            result.change_description,
            accepted,
            improvement,
        )
        return result.model_dump(mode="json")
