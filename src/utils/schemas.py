"""All shared Pydantic v2 models for QuantAgentLab."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# --- Enums ---


class Decision(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    CRYPTO = "CRYPTO"
    ETF = "ETF"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class ValidationVerdict(str, Enum):
    APPROVED = "APPROVED"
    CAUTION = "CAUTION"
    REJECTED = "REJECTED"


# --- Agent Models ---


class AgentReport(BaseModel):
    """Individual agent analysis report."""

    agent_name: str
    decision: Decision
    confidence: float = Field(ge=0, le=100, description="Confidence 0-100")
    reasoning: str
    data_points: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Complete multi-agent analysis output."""

    ticker: str
    asset_class: AssetClass = AssetClass.EQUITY
    decision: Decision
    confidence: float = Field(ge=0, le=100)
    target_price: float | None = None
    stop_loss: float | None = None
    position_size_pct: float = Field(ge=0, le=100, default=0.0)
    time_horizon: str = "3M"
    reasoning: str = ""
    agent_reports: dict[str, AgentReport] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


# --- Trading Models ---


class Signal(BaseModel):
    """Trading signal from a strategy."""

    ticker: str
    direction: SignalDirection
    strength: float = Field(ge=0, le=1)
    source: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class Order(BaseModel):
    """Trade order."""

    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class Position(BaseModel):
    """Portfolio position."""

    ticker: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    weight_pct: float = 0.0
    asset_class: AssetClass = AssetClass.EQUITY


class Portfolio(BaseModel):
    """Portfolio snapshot."""

    positions: dict[str, Position] = Field(default_factory=dict)
    cash: float = 100_000.0
    nav: float = 100_000.0
    timestamp: datetime = Field(default_factory=datetime.now)


# --- Backtest Models ---


class WalkForwardFold(BaseModel):
    """Single walk-forward fold result."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    model_params: dict[str, float | int | str | bool] = Field(default_factory=dict)


class BacktestMetrics(BaseModel):
    """Aggregate backtest performance metrics."""

    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_trade_return: float = 0.0
    volatility: float = 0.0
    sortino_ratio: float = 0.0


class BacktestResult(BaseModel):
    """Full backtest output."""

    metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)
    daily_returns: list[float] = Field(default_factory=list)
    trades: list[dict[str, str | float]] = Field(default_factory=list)
    walk_forward_results: list[WalkForwardFold] = Field(default_factory=list)
    equity_curve: list[float] = Field(default_factory=list)
    timestamps: list[str] = Field(default_factory=list)


class ValidationCheck(BaseModel):
    """Single validation check result."""

    name: str
    passed: bool
    details: str = ""
    severity: str = "info"


class ValidationResult(BaseModel):
    """BacktestValidator output."""

    checks: list[ValidationCheck] = Field(default_factory=list)
    verdict: ValidationVerdict = ValidationVerdict.CAUTION
    summary: str = ""

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


class BenchmarkComparison(BaseModel):
    """Comparison against benchmarks."""

    strategy_sharpe: float = 0.0
    spy_sharpe: float = 0.0
    btc_sharpe: float = 0.0
    sixty_forty_sharpe: float = 0.0
    equal_weight_sharpe: float = 0.0
    monte_carlo_percentile: float = 0.0
    excess_sharpe_vs_spy: float = 0.0


class WalkForwardResult(BaseModel):
    """Enhanced walk-forward result with feature drift and rolling IC."""

    folds: list[WalkForwardFold]
    stitched_returns: list[float]
    stitched_timestamps: list[str]
    aggregate_metrics: BacktestMetrics
    feature_importance_drift: list[float]  # rank corr between consecutive fold importances
    rolling_ic: list[float]  # per-fold IC (predictions vs realized returns)
    model_name: str


# --- Factor Models ---


class FactorDefinition(BaseModel):
    """Factor metadata for the factor library."""

    name: str
    expression: str = ""
    category: str = "custom"
    ic_mean: float = 0.0
    icir: float = 0.0
    source: str = "builtin"
    description: str = ""


class ModelSpec(BaseModel):
    """Model configuration specification."""

    name: str
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)
    feature_set: list[str] = Field(default_factory=list)


# --- Strategy Models ---


class MacroRegime(str, Enum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    CRISIS = "crisis"


class SizingMethod(str, Enum):
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"
    SIGNAL_PROPORTIONAL = "signal_proportional"


class RebalanceFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class SelectionMethod(str, Enum):
    REGIME_BASED = "regime_based"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    BANDIT = "bandit"
    MANUAL = "manual"


class Strategy(BaseModel):
    """Complete strategy definition stored in data/strategy_library/."""

    name: str
    description: str = ""
    factor_set: list[str] = Field(default_factory=lambda: ["Alpha158"])
    model: str = "LightGBM"
    universe: str = "SP500"
    entry_rules: dict[str, float | int | str | bool] = Field(default_factory=dict)
    exit_rules: dict[str, float | int | str | bool] = Field(default_factory=dict)
    sizing_method: SizingMethod = SizingMethod.EQUAL_WEIGHT
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY
    regime_applicability: list[MacroRegime] = Field(default_factory=lambda: list(MacroRegime))
    risk_overrides: dict[str, float] = Field(default_factory=dict)
    backtest_sharpe: float = 0.0
    backtest_max_drawdown: float = 0.0
    validated: bool = False
    source: str = "builtin"


class StrategyAllocation(BaseModel):
    """Output of the strategy selector — maps strategy names to capital %."""

    allocations: dict[str, float] = Field(default_factory=dict)
    active_strategy: str = ""
    selection_method: SelectionMethod = SelectionMethod.REGIME_BASED
    regime: MacroRegime = MacroRegime.NEUTRAL
    timestamp: datetime = Field(default_factory=datetime.now)


class StrategyPerformance(BaseModel):
    """Per-strategy live performance tracking."""

    strategy_name: str
    rolling_sharpe_30d: float = 0.0
    rolling_sharpe_60d: float = 0.0
    rolling_sharpe_90d: float = 0.0
    backtest_sharpe: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: list[float] = Field(default_factory=list)
    turnover: float = 0.0
    drift_detected: bool = False
    last_updated: datetime = Field(default_factory=datetime.now)


# --- Validation Suite Models ---


class CheckVerdict(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class CheckResult(BaseModel):
    """Single validation check result."""

    name: str
    verdict: CheckVerdict
    explanation: str


class ValidationReport(BaseModel):
    """Enhanced validation report with RELIABLE/NOT RELIABLE verdict."""

    checks: list[CheckResult]
    overall: str  # "RELIABLE" or "NOT RELIABLE"
    summary: str
    benchmark: BenchmarkComparison | None = None
