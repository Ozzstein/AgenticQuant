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

**Model:** `claude-haiku-4-5-20251001` via `langchain-anthropic`

**Public interface:**
```python
class FactorProposer:
    def __init__(self, config: FullAppConfig) -> None

    def propose_factors(
        self, n: int, memo: str, tested: list[str]
    ) -> list[FactorDefinition]

    def propose_model_config(
        self, n: int, memo: str, tested: list[dict]
    ) -> list[dict]

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
@dataclass
class EvalResult:
    factor: FactorDefinition
    stage1_ic: float
    stage1_passed: bool
    stage2_ic: float | None
    stage2_icir: float | None
    passed: bool
    reason: str    # "passed" | "low_ic" | "no_data" | "eval_error"

class FactorEvaluator:
    def __init__(self, config: AppConfig) -> None

    def evaluate_factor(
        self, factor: FactorDefinition, tickers: list[str] | None = None
    ) -> EvalResult

    def evaluate_factors_batch(
        self, factors: list[FactorDefinition]
    ) -> list[EvalResult]
```

**Stage 1 — Fast 90-day IC check:**
- Fetch last 90 trading days via `DataPipeline.yfinance_fallback()`
- Evaluate factor expression per ticker using `FactorEngine` (or inline pandas)
- Compute IC (Spearman correlation of factor values vs. next-day returns)
- Threshold: `|IC| >= config.rd_agent.min_ic` (default 0.02)
- Rejection rate target: ~70% of random candidates
- Runtime: < 10 seconds per factor

**Stage 2 — 2-fold walk-forward:**
- Only for stage 1 survivors
- Uses `WalkForwardBacktester` with `walk_forward_months=6, embargo_days=5`
- Extracts IC mean and ICIR from fold results
- Threshold: `IC >= min_ic AND ICIR >= min_icir` (defaults: 0.02, 0.3)
- Runtime: 30–120 seconds per factor

**Error handling:**
- `no_data`: yfinance returns empty DataFrame → skip, mark `reason="no_data"`
- `eval_error`: backtest raises exception → mark `reason="eval_error"`, log warning, continue

---

### 3.3 `ResearchAnalyst` (`src/core/research_analyst.py`)

**Responsibility:** Synthesize batch evaluation results into a research memo that guides the next round of proposals.

**Model:** `claude-opus-4-6` via `langchain-anthropic`

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

**KB storage:** `kb["research_memos"]` — rolling list of last 5 memos (oldest dropped when limit reached)

**Fallback:** When no API key, `write_memo()` returns a static placeholder string. `load_memo()` returns empty string.

---

### 3.4 `RDAgentRunner` (modified orchestrator)

**Responsibility:** Coordinate the three components, manage KB I/O, expose the same public API as before.

**Changes:**
- Constructor instantiates `FactorProposer`, `FactorEvaluator`, `ResearchAnalyst`
- `mine_factors()`: calls proposer → evaluator → analyst in loop
- `optimize_model()`: calls `propose_model_config()` → lightweight Sharpe simulation (no backtest for hyperparams — model selection is separate from factor IC)
- `copilot_factor()`: calls `propose_factors(n=3, description=...)` → evaluator → returns top result
- `copilot_model()`: calls `propose_model_config(n=1, ...)` → returns config
- `copilot_strategy()`: calls `propose_strategy()` → returns Strategy object
- `copilot_regime()`: calls `propose_regime_change()` → returns regime dict
- `implement_paper()`: reads file content → passes full text to `propose_factors()` as context
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

Model names are sourced from `AppConfig.anthropic_api_key` (present = use LLM, absent = fallback).

---

## 6. Error Handling

| Scenario | Behavior |
|----------|----------|
| LLM API call fails | `tenacity` retries 3x with exponential backoff; on final failure, falls back to keyword/random for that candidate |
| Stage 1: no market data | Mark `reason="no_data"`, skip to next candidate |
| Stage 2: `BacktestError` | Mark `reason="eval_error"`, count against budget, do not retry |
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

## 9. Files Changed

| File | Change |
|------|--------|
| `src/core/rd_agent_runner.py` | Refactored to thin orchestrator |
| `src/core/factor_proposer.py` | New |
| `src/core/factor_evaluator.py` | New |
| `src/core/research_analyst.py` | New |
| `tests/test_rd_agent.py` | New (unit tests for all 4 components) |
| `src/utils/schemas.py` | Add `FactorProposal`, `FactorProposalList`, `EvalResult` Pydantic models |
