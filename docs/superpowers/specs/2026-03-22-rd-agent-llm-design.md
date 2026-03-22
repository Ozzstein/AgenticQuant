# RD-Agent LLM Integration Design

**Date:** 2026-03-22
**Status:** Approved
**Scope:** Tier 2A — Replace simulated factor/model/strategy/regime discovery with real LLM-driven proposals and genuine backtest validation.

---

## 1. Problem Statement

The existing `RDAgentRunner` (`src/core/rd_agent_runner.py`) simulates alpha discovery using `random.gauss()` for IC values and keyword matching for factor proposals. Every "discovered" factor has a randomly-drawn IC and no real predictive validity. The strategy, model, and regime copilot commands map natural language to templates via string matching rather than reasoning.

This design replaces all simulation and keyword matching with:
- **Claude Haiku** for structured proposal generation (factors, model configs, strategies, regime changes)
- **Real two-stage backtest validation** for IC/ICIR measurement
- **Claude Opus** for meta-analyst research memos that feed back into subsequent proposals

---

## 2. Architecture

### New files

```
src/core/
├── rd_agent_runner.py       # MODIFIED — thin orchestrator (~150 lines)
├── factor_proposer.py       # NEW — Haiku-backed proposal generation
├── factor_evaluator.py      # NEW — two-stage IC gate + walk-forward validation
└── research_analyst.py      # NEW — Opus-backed meta-analyst, research memos
```

### Data flow per iteration batch

```
ResearchAnalyst loads last memo from KB
        ↓
FactorProposer(Haiku) → batch of N candidate proposals
        ↓
FactorEvaluator → Stage 1: fast 90-day IC check (rejects ~70% cheaply)
        ↓  survivors only
FactorEvaluator → Stage 2: 2-fold walk-forward (real IC/ICIR/Sharpe)
        ↓
Results appended to KB
        ↓
ResearchAnalyst(Opus) → reads batch results + KB → writes research memo
        ↓
Next batch reads memo as context for proposals
```

### Fallback behavior

When `anthropic_api_key` is empty, `FactorProposer` and `ResearchAnalyst` fall back to the existing keyword/random logic. All unit tests pass without an API key.

---

## 3. Component Specifications

### 3.1 `FactorProposer` (`src/core/factor_proposer.py`)

**Responsibility:** Generate candidate proposals for all four domains using Claude Haiku.

**Model:** `config.agents.quick_think_model` (default: `claude-haiku-4-5-20251001`) via `langchain-anthropic`

**Public interface:**
```python
class ModelConfigProposal(BaseModel):
    model_type: str          # "LightGBM" | "CatBoost" | "XGBoost" | "Linear"
    n_estimators: int
    learning_rate: float
    max_depth: int
    num_leaves: int
    rationale: str           # 1-sentence justification

class ModelConfigProposalList(BaseModel):
    configs: list[ModelConfigProposal]

class FactorProposer:
    def __init__(self, config: FullAppConfig) -> None

    def propose_factors(
        self, n: int, memo: str, tested: list[str], description: str = ""
    ) -> list[FactorDefinition]
    # description: optional NL context (used by copilot_factor and implement_paper)

    def propose_model_config(
        self, n: int, memo: str, tested: list[dict]
    ) -> list[ModelConfigProposal]

    def propose_strategy(
        self, description: str, memo: str
    ) -> Strategy

    def propose_regime_change(
        self, description: str, memo: str
    ) -> dict
```

**Prompt design (factors):**
- System: "You are a quantitative researcher. Generate novel alpha factor expressions for US equities."
- User: includes current research memo, list of already-tested factor names to avoid, requested count N
- Output: structured JSON list via `with_structured_output(FactorProposalList)`

**Structured output schemas:**
```python
class FactorProposal(BaseModel):
    name: str
    expression: str       # pandas-compatible expression on close/volume/high/low
    category: str         # momentum | mean_reversion | volatility | volume | quality
    description: str      # 1-sentence rationale

class FactorProposalList(BaseModel):
    factors: list[FactorProposal]
```

**Fallback:** When no API key, falls back to `_keyword_fallback()` (existing `_FACTOR_TEMPLATES` logic).

**Retry:** 3x with exponential backoff via `tenacity`. On final failure, falls back silently.

---

### 3.2 `FactorEvaluator` (`src/core/factor_evaluator.py`)

**Responsibility:** Validate factor candidates with real market data via a two-stage gate.

**Public interface:**
```python
class EvalResult(BaseModel):   # Pydantic v2, lives in src/utils/schemas.py
    factor_name: str
    stage1_ic: float
    stage1_passed: bool
    stage2_ic: float | None = None
    stage2_icir: float | None = None
    passed: bool
    reason: str    # "passed" | "low_ic" | "no_data" | "eval_error"

class FactorEvaluator:
    def __init__(self, config: FullAppConfig) -> None

    def evaluate_factor(
        self, factor: FactorDefinition, tickers: list[str] | None = None
    ) -> EvalResult

    def evaluate_factors_batch(
        self, factors: list[FactorDefinition]
    ) -> list[EvalResult]
```

**Factor expression execution:**
Factor expressions (e.g., `"close / close.shift(20) - 1"`) are evaluated using `pandas.DataFrame.eval()` in a restricted namespace. Available variables: `close`, `high`, `low`, `open`, `volume` (all lowercase pandas Series from the OHLCV DataFrame). If `eval()` raises any exception, the factor is marked `reason="eval_error"` and skipped. No arbitrary Python execution — `eval()` is restricted to pandas expression syntax. LLM prompts instruct Haiku to use only these variable names.

**Default evaluation universe:**
When `tickers=None`, `FactorEvaluator` uses a fixed module-level constant `_EVAL_UNIVERSE` of 30 liquid large-cap US equities (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V, MA, UNH, JNJ, PG, HD, CVX, LLY, ABBV, MRK, PEP, KO, AVGO, COST, MCD, ACN, TMO, DHR, WMT, BAC, XOM). This universe is large enough for statistically meaningful cross-sectional IC and small enough for fast fetches.

**Stage 1 — Fast 90-day IC check:**
- Fetch last 90 trading days via `DataPipeline.yfinance_fallback(tickers=_EVAL_UNIVERSE)`
- Evaluate factor expression per ticker using `df.eval()` in the OHLCV namespace
- Compute Spearman IC of factor values vs. next-day returns across all tickers
- Threshold: `|IC| >= config.rd_agent.min_ic` (default 0.02)
- Rejection rate target: ~70% of candidates
- Runtime: < 10 seconds per factor

**Stage 2 — IC computation on walk-forward OOS windows:**
- Only for stage 1 survivors
- Data lookback: `stage2_lookback_days = 500` trading days (~2 years), hardcoded constant
- `DataPipeline.yfinance_fallback()` fetches the 500-day window
- `WalkForwardBacktester` splits this into 2 folds (`walk_forward_months=6, embargo_days=5`) — requires minimum ~12 months data, satisfied by 500-day window
- IC and ICIR are computed **independently** from the backtester: for each OOS fold window, compute per-ticker factor values, then Spearman IC vs. next-day returns; `ic_mean = mean(fold_ics)`, `icir = ic_mean / std(fold_ics)`
- The backtester itself is **not called** in Stage 2 — only its fold-splitting logic is reused to define consistent OOS windows. `WalkForwardBacktester._split_folds` will be promoted to a public `split_folds` module-level function in `backtester.py`. Both existing call sites (`backtester.py` and `walk_forward.py` line 80) will be updated to use the public function. The private method is removed (not shimmed).
- Threshold: `IC >= min_ic AND ICIR >= min_icir` (defaults: 0.02, 0.3)
- Stage 2 routing rule: only entered when `stage1_passed is True`. When `stage1_passed is False`, `stage2_ic` and `stage2_icir` are left as `None` and `passed=False`.
- Runtime: 30–120 seconds per factor

**Error handling:**
- `no_data`: yfinance returns empty DataFrame → skip, mark `reason="no_data"`
- `eval_error`: Stage 1 `df.eval()` or Stage 2 IC computation raises exception → mark `reason="eval_error"`, log warning, continue

---

### 3.3 `ResearchAnalyst` (`src/core/research_analyst.py`)

**Responsibility:** Synthesize batch evaluation results into a research memo that guides the next round of proposals.

**Model:** `config.agents.deep_think_model` (default: `claude-opus-4-20250514`) via `langchain-anthropic`

**Public interface:**
```python
class ResearchAnalyst:
    def __init__(self, config: FullAppConfig) -> None

    def write_memo(
        self, batch_results: list[EvalResult], kb: dict
    ) -> str

    def load_memo(self, kb: dict) -> str

    def save_memo(self, memo: str, kb: dict) -> None
```

**Memo prompt:**
- System: "You are a senior quantitative researcher reviewing alpha factor research. Write a concise research memo."
- User: previous memo (if any) + batch results table (name, category, stage1_IC, stage2_IC, ICIR, passed) + total KB stats
- Output: 200–400 word freeform memo covering: what worked, what failed, observed patterns, hypotheses for next batch
- `max_tokens=600` to bound Opus output cost and KB size

**KB storage:** `kb["research_memos"]` — rolling list of last 5 memos (oldest dropped when limit reached)

**Fallback:** When no API key, `write_memo()` returns a static placeholder string. `load_memo()` returns empty string.

---

### 3.4 `RDAgentRunner` (modified orchestrator)

**Responsibility:** Coordinate the three components, manage KB I/O, expose the same public API as before.

**Changes:**
- Constructor instantiates `FactorProposer`, `FactorEvaluator`, `ResearchAnalyst`
- `mine_factors()`: calls proposer → evaluator → analyst in loop
- `optimize_model()`: calls `propose_model_config()` to get LLM-proposed hyperparameter dicts, then scores each config using the existing random-Gaussian Sharpe simulation (`random.gauss(1.0, 0.3)`) as a cheap proxy. Real model evaluation requires training on labeled data and is deferred to a future task. In this iteration, Haiku improves the *proposal quality* (configs are reasoning-based rather than purely random) while scoring remains simulated.
- `copilot_factor(description)`: calls `propose_factors(n=3, memo=..., tested=..., description=description)` → evaluator → returns top passing result
- `copilot_model()`: calls `propose_model_config(n=1, ...)` → returns `ModelConfigProposal` dict
- `copilot_strategy()`: calls `propose_strategy()` → returns Strategy object
- `copilot_regime()`: calls `propose_regime_change()` → returns regime dict
- `implement_paper(source)`: reads file/text content → calls `propose_factors(n=3, memo=..., tested=..., description=paper_text)` — paper text passed via the `description` parameter as NL context for the proposer
- All other public methods (`library_status`, `validate_library`, `reset_knowledge`, etc.) unchanged

---

## 4. KB Schema Changes

Backward-compatible additions to `data/rd_knowledge_base/kb.json`:

```json
{
  "...existing fields preserved...",
  "research_memos": [],
  "eval_results": [
    {
      "factor_name": "mom_20d",
      "stage1_ic": 0.031,
      "stage2_ic": 0.028,
      "stage2_icir": 0.41,
      "passed": true,
      "date": "2026-03-22",
      "reason": "passed"
    }
  ]
}
```

Existing `tested_factors`, `failed_factors`, `discoveries` fields continue to be populated as before (for backward compatibility with `library_status()` and `validate_library()`).

---

## 5. Configuration

No new config fields required. Existing `RDAgentConfig` fields map cleanly:

| Field | Used by |
|-------|---------|
| `min_ic` | Stage 1 and Stage 2 threshold |
| `min_icir` | Stage 2 threshold |
| `factor_iterations` | `mine_factors()` loop count |
| `budget` | batch size per iteration |
| `model_iterations` | `optimize_model()` loop count |

LLM mode is activated when `AppConfig.anthropic_api_key` is non-empty; when empty, keyword/random fallback is used. Model identifiers are read from `AgentConfig.quick_think_model` (Haiku) and `AgentConfig.deep_think_model` (Opus).

---

## 6. Error Handling

| Scenario | Behavior |
|----------|----------|
| LLM API call fails | `tenacity` retries 3x with exponential backoff; on final failure, falls back to keyword/random for that candidate |
| Stage 1: no market data | Mark `reason="no_data"`, skip to next candidate |
| Stage 2: IC computation raises exception | Mark `reason="eval_error"`, count against budget, do not retry |
| Opus memo call fails | Log warning, store empty memo, continue |
| KB file corrupted | Re-initialize with `_EMPTY_KB`, log error |

---

## 7. Testing

| Component | Test approach |
|-----------|---------------|
| `FactorProposer` | Mock `langchain-anthropic` LLM with `mock_llm` fixture; test JSON parsing, fallback, dedup |
| `FactorEvaluator` | Mock `yfinance_fallback()` + `WalkForwardBacktester.run()`; test stage1 gate, stage2 routing, `EvalResult` fields |
| `ResearchAnalyst` | Mock Opus call; test memo parsing, KB save/load, rolling-5 window |
| `RDAgentRunner` | Mock all three components; test full orchestration loop, KB persistence |
| CLI commands | Existing CLI tests cover `mine-factors`, `copilot-factor`, `library-status` — no changes needed |

All tests pass without API keys (fallback mode activated automatically).

---

## 8. Improvement List (deferred)

- `FactorProposer`: accept `temperature` param per domain (lower for model configs, higher for novel factors)
- `EvalResult`: include regime-conditional IC (IC in `risk_on` vs. `risk_off`) for richer Opus feedback
- `ResearchAnalyst.write_memo()`: return structured object (hypotheses list + confidence scores) instead of freeform text
- `FactorEvaluator`: parallel batch evaluation using `ThreadPoolExecutor`
- Add `max_cost_usd` config cap for automated overnight runs

---

## 9. Thread Safety

The KB file lock (`threading.Lock`) remains owned exclusively by `RDAgentRunner`. The three new components (`FactorProposer`, `FactorEvaluator`, `ResearchAnalyst`) never read or write the KB directly — they only accept KB data as function arguments and return results. Only the orchestrator calls `_load_kb()` and `_save_kb()` while holding the lock. This ensures safety even when `evaluate_factors_batch` is later parallelized (Section 8 deferred list), since evaluation results are collected by the orchestrator before any KB write.

---

## 10. Files Changed

| File | Change |
|------|--------|
| `src/core/rd_agent_runner.py` | Refactored to thin orchestrator (~150 lines) |
| `src/core/factor_proposer.py` | New — Haiku-backed proposals for all four domains |
| `src/core/backtester.py` | Promote `_split_folds` to public `split_folds` module-level function; update internal call site |
| `src/core/walk_forward.py` | Update call site from `self._backtester._split_folds()` to `split_folds()` |
| `src/core/factor_evaluator.py` | New — two-stage IC gate, imports `split_folds` from backtester |
| `src/core/research_analyst.py` | New — Opus meta-analyst, research memo lifecycle |
| `tests/test_rd_agent.py` | New — unit tests for all 4 components (mocked LLM + data) |
| `src/utils/schemas.py` | Add `FactorProposal`, `FactorProposalList`, `ModelConfigProposal`, `ModelConfigProposalList`, `EvalResult` Pydantic v2 models |
