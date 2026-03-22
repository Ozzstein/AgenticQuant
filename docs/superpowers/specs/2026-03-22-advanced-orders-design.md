# Advanced Orders Design

**Date:** 2026-03-22
**Feature:** Tier 3C — Bracket Orders, Trailing Stops, Time-in-Force
**Status:** Approved for implementation

---

## Goal

Add bracket orders (take-profit + stop-loss), percentage-based trailing stops, and full time-in-force (GTC/DAY/IOC/FOK) support across all three brokers: `PaperTrader`, `AlpacaTrader`, and `CcxtTrader`.

---

## Architecture

```
Order (extended schema)
  ├── time_in_force: TimeInForce (GTC|DAY|IOC|FOK)
  ├── take_profit_price: float | None   ← bracket TP leg
  ├── stop_loss_price: float | None     ← bracket SL leg
  └── trail_percent: float | None       ← trailing stop %

OrderType (extended enum)
  ├── MARKET, LIMIT (existing)
  ├── STOP          (new — market sell when price hits stop)
  └── TRAILING_STOP (new — stop ratchets up with price)

ContingentOrderSimulator (new utility in order_manager.py)
  ├── Used by PaperTrader and CcxtTrader
  └── Not used by AlpacaTrader (native API handles it)

PaperTrader    → compose ContingentOrderSimulator
AlpacaTrader   → Alpaca native bracket/trailing API
CcxtTrader     → compose ContingentOrderSimulator
BaseBroker     → UNCHANGED
```

---

## Schema Changes (`src/utils/schemas.py`)

### New: `TimeInForce` enum

```python
class TimeInForce(str, Enum):
    GTC = "GTC"   # good-till-cancelled (default)
    DAY = "DAY"   # expires at end of trading day
    IOC = "IOC"   # immediate-or-cancel (unfilled portion cancelled)
    FOK = "FOK"   # fill-or-kill (must fill entirely or cancel)
```

### Extended: `OrderType` enum

Add two new values:
- `STOP` — triggers a market sell when price falls to `stop_loss_price`
- `TRAILING_STOP` — stop price ratchets up with peak price, fires when price falls `trail_percent`% from peak

### Extended: `Order` model

Add four optional fields (all backward-compatible — default to current behaviour):

```python
time_in_force: TimeInForce = TimeInForce.GTC
take_profit_price: float | None = None
stop_loss_price: float | None = None
trail_percent: float | None = None
```

A bracket order is an `Order` with both `take_profit_price` and `stop_loss_price` set. After the parent BUY fills, the broker registers both contingent legs and monitors them until one triggers (OCO — one-cancels-other).

---

## New File: `ContingentOrderSimulator` (`src/execution/order_manager.py`)

Standalone utility — no broker interface. Tracks open bracket and trailing stop entries; evaluates price triggers each snapshot cycle.

### Internal state

```python
@dataclass
class ContingentEntry:
    ticker: str
    quantity: float
    take_profit_price: float | None   # LIMIT sell at this price
    stop_loss_price: float | None     # STOP sell at this price
    trail_percent: float | None       # trailing stop percentage
    trail_peak_price: float           # highest price seen (ratchets up, never down)
```

### Public API

```python
class ContingentOrderSimulator:
    def register(
        self,
        ticker: str,
        quantity: float,
        entry_price: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        trail_percent: float | None = None,
    ) -> None:
        """Register contingent legs after a BUY fills."""

    def update_trailing_peaks(self, market_prices: dict[str, float]) -> None:
        """Ratchet trail_peak_price upward as prices move higher. Call before evaluate()."""

    def evaluate(self, market_prices: dict[str, float]) -> list[Order]:
        """Check all entries against current prices. Return triggered SELL orders.
        Removes the entry (both legs) for each triggered ticker (OCO)."""

    def cancel(self, ticker: str) -> None:
        """Remove contingent entry for ticker (called on manual close)."""

    @property
    def open_tickers(self) -> set[str]:
        """Set of tickers currently being monitored."""
```

### Trigger logic (evaluated in order):

1. **Take-profit**: fires if `current_price >= take_profit_price` → LIMIT SELL at `take_profit_price`
2. **Stop-loss / trailing stop**: fires if `current_price <= effective_stop`
   - Fixed stop: `effective_stop = stop_loss_price`
   - Trailing stop: `effective_stop = trail_peak_price * (1 - trail_percent / 100)`
3. First trigger wins; both legs are removed immediately (OCO).
4. Triggered orders have `order_type=MARKET` (execute at market on next tick).

---

## `PaperTrader` Changes (`src/execution/paper_trader.py`)

### Composition
```python
self._contingent = ContingentOrderSimulator()
```

### After BUY fills
If the executed order has `take_profit_price`, `stop_loss_price`, or `trail_percent` set:
```python
self._contingent.register(
    ticker=order.ticker,
    quantity=filled.quantity,
    entry_price=filled.fill_price,
    take_profit_price=order.take_profit_price,
    stop_loss_price=order.stop_loss_price,
    trail_percent=order.trail_percent,
)
```

### In `snapshot()`
```python
self._contingent.update_trailing_peaks(market_prices)
triggered = self._contingent.evaluate(market_prices)
for contingent_order in triggered:
    self.execute_order(contingent_order, market_prices)
```

### On SELL (manual close)
Call `self._contingent.cancel(order.ticker)` to remove pending legs.

### TIF simulation
- `GTC` / `DAY`: paper orders fill instantly — treated identically (no real time concept in simulation)
- `IOC`: fills whatever quantity is available (in paper, always full qty)
- `FOK`: fills only if full quantity can be filled; rejects otherwise (check cash/position before fill)

### `STOP` order type
Executes as a market SELL when `market_prices[ticker] <= order.stop_loss_price`.
Evaluated directly in `execute_order()` — if price condition not met, order is rejected.

---

## `AlpacaTrader` Changes (`src/execution/alpaca_trader.py`)

No `ContingentOrderSimulator` — Alpaca monitors server-side.

### Bracket orders
In `_submit_order()`, if `order.take_profit_price` or `order.stop_loss_price` is set, build a bracket request:
```python
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest
)
request = MarketOrderRequest(
    symbol=symbol,
    qty=order.quantity,
    side=OrderSide.BUY,
    time_in_force=_map_tif(order.time_in_force),
    order_class="bracket",
    take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
    stop_loss=StopLossRequest(stop_price=order.stop_loss_price),
)
```

### Trailing stops
If `order.order_type == OrderType.TRAILING_STOP`:
```python
from alpaca.trading.requests import TrailingStopOrderRequest
request = TrailingStopOrderRequest(
    symbol=symbol,
    qty=order.quantity,
    side=OrderSide.SELL,
    time_in_force=_map_tif(order.time_in_force),
    trail_percent=order.trail_percent,
)
```

### TIF mapping
```python
def _map_tif(tif: TimeInForce) -> alpaca.TimeInForce:
    return {
        TimeInForce.GTC: alpaca.TimeInForce.GTC,
        TimeInForce.DAY: alpaca.TimeInForce.DAY,
        TimeInForce.IOC: alpaca.TimeInForce.IOC,
        TimeInForce.FOK: alpaca.TimeInForce.FOK,
    }[tif]
```

---

## `CcxtTrader` Changes (`src/execution/ccxt_trader.py`)

### Composition
```python
self._contingent = ContingentOrderSimulator()
```

Identical pattern to PaperTrader: register after BUY fills, evaluate in `snapshot()`, cancel on manual close.

### TIF passthrough
CCXT supports TIF on most exchanges via order params:
```python
params = {"timeInForce": order.time_in_force.value}
ex.create_order(symbol, order_type, side, qty, price, params=params)
```

### STOP / TRAILING_STOP
Routed through `ContingentOrderSimulator` (not native CCXT — support is exchange-specific and unreliable). The `order.stop_loss_price` and `order.trail_percent` fields are registered in the simulator after parent fill.

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `src/utils/schemas.py` | Modify | Add `TimeInForce`, extend `OrderType`, extend `Order` |
| `src/execution/order_manager.py` | **Create** | `ContingentOrderSimulator` + `ContingentEntry` |
| `src/execution/paper_trader.py` | Modify | Compose simulator, bracket/trailing/TIF/STOP support |
| `src/execution/alpaca_trader.py` | Modify | Native bracket/trailing/TIF via alpaca-py |
| `src/execution/ccxt_trader.py` | Modify | Compose simulator, TIF passthrough |
| `tests/test_order_manager.py` | **Create** | Simulator unit tests (12 tests) |
| `tests/test_advanced_orders.py` | **Create** | End-to-end tests across all brokers (20 tests) |

---

## Testing Strategy

### `tests/test_order_manager.py` (unit tests, no broker needed)
1. `test_register_bracket` — entry stored with correct TP/SL prices
2. `test_register_trailing_stop` — entry stored with trail_percent and trail_peak = entry_price
3. `test_tp_triggers_when_price_above` — evaluate returns SELL when price >= TP
4. `test_tp_does_not_trigger_below` — no trigger when price < TP
5. `test_sl_triggers_when_price_below` — evaluate returns SELL when price <= SL
6. `test_sl_does_not_trigger_above` — no trigger when price > SL
7. `test_trailing_stop_peak_ratchets_up` — update_trailing_peaks raises peak
8. `test_trailing_stop_peak_does_not_fall` — peak never decreases
9. `test_trailing_stop_triggers_from_peak` — fires when price falls trail_percent% from peak
10. `test_oco_both_legs_removed_on_tp_trigger` — after TP fires, SL entry is gone
11. `test_cancel_removes_entry` — cancel() removes ticker from open_tickers
12. `test_evaluate_multiple_tickers` — handles multiple concurrent positions

### `tests/test_advanced_orders.py` (per-broker tests, all mocked)
**PaperTrader (8 tests):**
1. `test_paper_bracket_tp_fills_on_snapshot`
2. `test_paper_bracket_sl_fills_on_snapshot`
3. `test_paper_trailing_stop_triggers_after_rally`
4. `test_paper_trailing_stop_does_not_trigger_during_rally`
5. `test_paper_stop_order_rejects_if_price_above_stop`
6. `test_paper_fok_rejects_if_insufficient_cash`
7. `test_paper_manual_sell_cancels_contingent`
8. `test_paper_no_double_fill_after_contingent_fires`

**AlpacaTrader (6 tests, all mocked):**
9. `test_alpaca_bracket_submits_native_bracket_request`
10. `test_alpaca_trailing_stop_submits_trailing_stop_request`
11. `test_alpaca_tif_gtc_mapped_correctly`
12. `test_alpaca_tif_day_mapped_correctly`
13. `test_alpaca_tif_ioc_mapped_correctly`
14. `test_alpaca_tif_fok_mapped_correctly`

**CcxtTrader (6 tests, all mocked):**
15. `test_ccxt_bracket_registers_contingent_after_fill`
16. `test_ccxt_bracket_tp_triggers_in_snapshot`
17. `test_ccxt_bracket_sl_triggers_in_snapshot`
18. `test_ccxt_trailing_stop_triggers`
19. `test_ccxt_tif_passed_as_param`
20. `test_ccxt_manual_sell_cancels_contingent`

---

## Constraints

- `BaseBroker` protocol: **unchanged** — no new required methods
- All new `Order` fields are optional with safe defaults — zero breaking changes to existing callers
- `signal_translator.py`: unchanged — continues producing plain MARKET orders
- `risk_controls.py`: unchanged — `check_order()` operates on the parent order only; contingent legs bypass risk checks (they are defensive exits)
- Contingent orders bypass `check_order()` — they are stop-outs, not new entries
