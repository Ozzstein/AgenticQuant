"""Unit tests for src/execution/risk_controls.py (7 pre-trade risk checks)."""

from __future__ import annotations

import pytest

from src.execution.risk_controls import RiskCheckResult, check_order
from src.utils.config import RiskConfig
from src.utils.schemas import Order, OrderSide, OrderType, Portfolio, Position


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_portfolio() -> Portfolio:
    """Empty portfolio with $100K cash and NAV."""
    return Portfolio(
        positions={},
        cash=100_000.0,
        nav=100_000.0,
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    """Default RiskConfig values."""
    return RiskConfig()


def _buy(ticker: str, quantity: float) -> Order:
    return Order(ticker=ticker, side=OrderSide.BUY, quantity=quantity)


def _sell(ticker: str, quantity: float) -> Order:
    return Order(ticker=ticker, side=OrderSide.SELL, quantity=quantity)


# ---------------------------------------------------------------------------
# 1. Large buy blocked by position_size
# ---------------------------------------------------------------------------


def test_large_buy_blocked(base_portfolio: Portfolio, risk_config: RiskConfig) -> None:
    """$50K order in $100K portfolio (50% weight) should fail position_size check."""
    order = _buy("AAPL", 100)
    market_prices = {"AAPL": 500.0}  # 100 * 500 = $50K = 50% of NAV

    result = check_order(order, base_portfolio, market_prices, risk_config)

    assert isinstance(result, RiskCheckResult)
    assert not result.passed
    assert "position_size" in result.failed_checks


# ---------------------------------------------------------------------------
# 2. Small buy passes all checks
# ---------------------------------------------------------------------------


def test_small_buy_passes(base_portfolio: Portfolio, risk_config: RiskConfig) -> None:
    """$3K order in $100K portfolio (3% weight) should pass all applicable checks."""
    order = _buy("AAPL", 30)
    market_prices = {"AAPL": 100.0}  # 30 * 100 = $3K = 3% of NAV

    result = check_order(order, base_portfolio, market_prices, risk_config)

    assert isinstance(result, RiskCheckResult)
    assert result.passed
    assert result.failed_checks == []


# ---------------------------------------------------------------------------
# 3. Drawdown circuit-breaker blocks buys
# ---------------------------------------------------------------------------


def test_drawdown_blocks_buy(risk_config: RiskConfig) -> None:
    """Drawdown of -20% (nav=80K, peak=100K) should block BUY orders."""
    portfolio = Portfolio(positions={}, cash=80_000.0, nav=80_000.0)
    order = _buy("MSFT", 1)
    market_prices = {"MSFT": 100.0}

    result = check_order(
        order, portfolio, market_prices, risk_config, peak_nav=100_000.0
    )

    assert "drawdown_cb" in result.failed_checks
    assert not result.passed


# ---------------------------------------------------------------------------
# 4. Drawdown does NOT block sells
# ---------------------------------------------------------------------------


def test_drawdown_does_not_block_sell(risk_config: RiskConfig) -> None:
    """Drawdown check is skipped for SELL orders; sell should not fail drawdown_cb."""
    portfolio = Portfolio(
        positions={
            "MSFT": Position(
                ticker="MSFT", quantity=10, avg_cost=100.0, current_price=100.0
            )
        },
        cash=80_000.0,
        nav=80_000.0,
    )
    order = _sell("MSFT", 1)
    market_prices = {"MSFT": 100.0}

    result = check_order(
        order, portfolio, market_prices, risk_config, peak_nav=100_000.0
    )

    assert "drawdown_cb" not in result.failed_checks


# ---------------------------------------------------------------------------
# 5. Insufficient cash for BUY
# ---------------------------------------------------------------------------


def test_insufficient_cash(risk_config: RiskConfig) -> None:
    """Order cost $99K but cash is $50K should fail cash_buffer."""
    portfolio = Portfolio(positions={}, cash=50_000.0, nav=100_000.0)
    order = _buy("TSLA", 990)  # 990 * 100 = $99K
    market_prices = {"TSLA": 100.0}

    result = check_order(order, portfolio, market_prices, risk_config)

    assert not result.passed
    assert "cash_buffer" in result.failed_checks


# ---------------------------------------------------------------------------
# 6. Sell without sufficient shares
# ---------------------------------------------------------------------------


def test_sell_insufficient_shares(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """SELL 200 shares but portfolio has only 100 → fails sufficient_shares."""
    portfolio = Portfolio(
        positions={
            "GOOG": Position(
                ticker="GOOG", quantity=100, avg_cost=150.0, current_price=150.0
            )
        },
        cash=100_000.0,
        nav=100_000.0,
    )
    order = _sell("GOOG", 200)
    market_prices = {"GOOG": 150.0}

    result = check_order(order, portfolio, market_prices, risk_config)

    assert not result.passed
    assert "sufficient_shares" in result.failed_checks


# ---------------------------------------------------------------------------
# 7. Crypto gets tighter 5% position-size limit
# ---------------------------------------------------------------------------


def test_position_size_crypto(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """Crypto order worth 6% of NAV should fail position_size (max 5% for crypto)."""
    order = _buy("BTC-USD", 1)
    market_prices = {"BTC-USD": 6_000.0}  # $6K = 6% of $100K NAV

    result = check_order(
        order,
        base_portfolio,
        market_prices,
        risk_config,
        asset_class="CRYPTO",
    )

    assert not result.passed
    assert "position_size" in result.failed_checks


def test_position_size_crypto_passes(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """Crypto order worth 4% of NAV should pass position_size (max 5%)."""
    order = _buy("BTC-USD", 1)
    market_prices = {"BTC-USD": 4_000.0}  # $4K = 4% of $100K NAV

    result = check_order(
        order,
        base_portfolio,
        market_prices,
        risk_config,
        asset_class="CRYPTO",
    )

    assert "position_size" not in result.failed_checks


# ---------------------------------------------------------------------------
# 8. Daily turnover limit exceeded
# ---------------------------------------------------------------------------


def test_daily_turnover_limit(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """Already 20% turnover today; new 10% order pushes total to 30% > 25%."""
    already_traded = 0.20 * base_portfolio.nav  # $20K already traded
    order = _buy("NVDA", 100)
    market_prices = {"NVDA": 100.0}  # 100 * 100 = $10K = 10% more

    result = check_order(
        order,
        base_portfolio,
        market_prices,
        risk_config,
        daily_turnover_so_far=already_traded,
    )

    assert not result.passed
    assert "daily_turnover" in result.failed_checks


def test_daily_turnover_within_limit(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """Already 10% turnover; new 5% order = 15% total < 25%: should pass."""
    already_traded = 0.10 * base_portfolio.nav  # $10K
    order = _buy("NVDA", 50)
    market_prices = {"NVDA": 100.0}  # 50 * 100 = $5K = 5% more

    result = check_order(
        order,
        base_portfolio,
        market_prices,
        risk_config,
        daily_turnover_so_far=already_traded,
    )

    assert "daily_turnover" not in result.failed_checks


# ---------------------------------------------------------------------------
# 9. Failed check details are populated with info strings
# ---------------------------------------------------------------------------


def test_result_details_populated(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """When a check fails the details dict must contain the relevant key."""
    order = _buy("AAPL", 100)
    market_prices = {"AAPL": 500.0}  # 50% weight → position_size fails

    result = check_order(order, base_portfolio, market_prices, risk_config)

    assert "position_size" in result.details
    info = result.details["position_size"]
    assert "%" in info  # contains a percentage string
    assert ">" in info


def test_cash_buffer_details_populated(risk_config: RiskConfig) -> None:
    """cash_buffer failure details must mention order cost and available cash."""
    portfolio = Portfolio(positions={}, cash=50_000.0, nav=100_000.0)
    order = _buy("TSLA", 990)
    market_prices = {"TSLA": 100.0}

    result = check_order(order, portfolio, market_prices, risk_config)

    assert "cash_buffer" in result.details
    info = result.details["cash_buffer"]
    assert "order cost" in info
    assert "available" in info


# ---------------------------------------------------------------------------
# 10. Liquidity and market-impact checks skipped when ADV key absent
# ---------------------------------------------------------------------------


def test_liquidity_skipped_without_adv(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """If no {ticker}_adv key in market_prices, liquidity check is skipped."""
    order = _buy("XYZ", 5)
    market_prices = {"XYZ": 100.0}  # no XYZ_adv

    result = check_order(order, base_portfolio, market_prices, risk_config)

    assert "liquidity" not in result.failed_checks
    assert "market_impact" not in result.failed_checks


def test_liquidity_fails_low_adv(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """ADV of $500K is below the $1M minimum → liquidity check fails."""
    order = _buy("ILLIQUID", 1)
    market_prices = {"ILLIQUID": 50.0, "ILLIQUID_adv": 500_000.0}

    result = check_order(order, base_portfolio, market_prices, risk_config)

    assert "liquidity" in result.failed_checks
    assert "liquidity" in result.details


def test_market_impact_fails_large_order(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """Order of 10K shares when ADV only allows 1K shares at 1% → fails market_impact."""
    order = _buy("SMALL", 10_000)
    # ADV = $100M / $100 per share = 1M shares; 1% = 10K shares → order is at limit
    # Use 9M ADV so 1% = 90K shares but order is 10K to pass, then use tight ADV
    # ADV $1M => 10K shares per $100; 1% = 100 shares. Order 10K >> 100 → fail
    market_prices = {"SMALL": 100.0, "SMALL_adv": 1_000_000.0}

    result = check_order(order, base_portfolio, market_prices, risk_config)

    assert "market_impact" in result.failed_checks


# ---------------------------------------------------------------------------
# 11. Drawdown check skipped when peak_nav is None
# ---------------------------------------------------------------------------


def test_drawdown_skipped_when_no_peak_nav(
    base_portfolio: Portfolio, risk_config: RiskConfig
) -> None:
    """If peak_nav is not provided, drawdown check is not run."""
    portfolio = Portfolio(positions={}, cash=80_000.0, nav=80_000.0)
    order = _buy("SPY", 1)
    market_prices = {"SPY": 100.0}

    result = check_order(order, portfolio, market_prices, risk_config, peak_nav=None)

    assert "drawdown_cb" not in result.failed_checks


# ---------------------------------------------------------------------------
# 12. Sell with no position at all fails sufficient_shares
# ---------------------------------------------------------------------------


def test_sell_no_position(base_portfolio: Portfolio, risk_config: RiskConfig) -> None:
    """SELL when ticker not in positions → fails sufficient_shares."""
    order = _sell("AMZN", 10)
    market_prices = {"AMZN": 200.0}

    result = check_order(order, base_portfolio, market_prices, risk_config)

    assert not result.passed
    assert "sufficient_shares" in result.failed_checks
