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
- `STOP` — triggers a market sell when price falls to `stop_loss_price` (requires `stop_loss_price` to be set)
- `TRAILING_STOP` — stop price ratchets up with peak price, fires when price falls `trail_percent`% from peak (requires `trail_percent` to be set)

### Extended: `Order` model

Add four optional fields (all backward-compatible — default to current behaviour):

```python
time_in_force: TimeInForce = TimeInForce.GTC
take_profit_price: float | None = None
stop_loss_price: float | None = None
trail_percent: float | None = None
```

**Model-level validation** — add a `@model_validator(mode='after')` with these rules:
- If `order_type == STOP`: `stop_loss_price` must not be `None`
- If `order_type == TRAILING_STOP`: `trail_percent` must not be `None` and must be > 0
- If both `take_profit_price` and `stop_loss_price` are set (bracket): `take_profit_price` must be > `stop_loss_price` (prevents inverted brackets that would fire immediately)

A bracket order is an `Order` with both `take_profit_price` and `stop_loss_price` set. After the parent BUY fills, the broker registers both contingent legs and monitors them until one triggers (OCO — one-cancels-other).

**Scope:** Bracket fields (`take_profit_price`, `stop_loss_price`, `trail_percent`) are only meaningful on BUY-side orders. Brokers must skip registration if `order.side != OrderSide.BUY`.

---

## New File: `ContingentOrderSimulator` (`src/execution/order_manager.py`)

Standalone utility — no broker interface. Tracks open bracket and trailing stop entries; evaluates price triggers each snapshot cycle.

### Internal state

```python
@dataclass
class ContingentEntry:
    ticker: str
    quantity: float
    take_profit_price: float | None   # sell at this price (fills at TP price, not market)
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
        """Register contingent legs after a BUY fills.

        If ticker is already tracked, the existing entry is overwritten with the
        new parameters (position additions replace prior bracket levels). Callers
        that need to preserve the original bracket must cancel() first.

        Raises ValueError if stop_loss_price >= take_profit_price (inverted bracket).

        Mutual exclusivity of stop types: `stop_loss_price` and `trail_percent` are
        mutually exclusive on a single entry. If both are provided, `trail_percent`
        takes precedence and `stop_loss_price` is ignored. Use `stop_loss_price`
        alone for a fixed stop; use `trail_percent` alone for a trailing stop.
        To combine a trailing stop with a take-profit, set `take_profit_price` and
        `trail_percent` (leave `stop_loss_price=None`).
        """

    def update_trailing_peaks(self, market_prices: dict[str, float]) -> None:
        """Ratchet trail_peak_price upward as prices move higher. Call before evaluate()."""

    def evaluate(self, market_prices: dict[str, float]) -> list[Order]:
        """Check all entries against current prices. Return triggered SELL orders.

        Each triggered Order has fill_price pre-set to the trigger price
        (take_profit_price for TP triggers, effective_stop for SL/trailing triggers)
        so brokers fill at the correct contingent price rather than raw market price.

        Always removes the entry for each triggered ticker regardless of whether
        the returned order is subsequently executed successfully (OCO — both legs gone).
        """

    def cancel(self, ticker: str) -> None:
        """Remove contingent entry for ticker (called on manual close)."""

    def cleanup_stale(self, held_tickers: set[str]) -> None:
        """Remove entries for tickers no longer held in the portfolio.
        Call at start of snapshot() to prevent repeated SELL attempts on
        already-closed positions."""

    @property
    def open_tickers(self) -> set[str]:
        """Set of tickers currently being monitored."""
```

### Trigger logic (evaluated in order):

1. **Take-profit**: fires if `current_price >= take_profit_price`
   → MARKET SELL with `fill_price=take_profit_price` (ensures correct simulated fill price)
2. **Stop-loss / trailing stop**: fires if `current_price <= effective_stop`
   - Fixed stop: `effective_stop = stop_loss_price`
   - Trailing stop: `effective_stop = trail_peak_price * (1 - trail_percent / 100)`
   → MARKET SELL with `fill_price=effective_stop`
3. First trigger wins; entry is removed immediately (OCO — even if downstream execute_order fails).
4. All triggered orders have `order_type=MARKET`.

**Stale entry cleanup:** Entries for tickers no longer in the portfolio (manually sold) are cleared by `cleanup_stale()`. PaperTrader and CcxtTrader call this at the start of each `snapshot()` to prevent repeated failed SELL attempts.

---

## `PaperTrader` Changes (`src/execution/paper_trader.py`)

### Composition
```python
self._contingent = ContingentOrderSimulator()
```

### After BUY fills — bracket/trailing registration (BUY-side only)
If `filled.side == OrderSide.BUY` and at least one of `take_profit_price`, `stop_loss_price`, `trail_percent` is set:
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
# 1. Clean up stale entries for positions no longer held
self._contingent.cleanup_stale(set(self._positions.keys()))
# 2. Ratchet trailing stop peaks
self._contingent.update_trailing_peaks(market_prices)
# 3. Evaluate triggers; execute each (fill_price already set by simulator)
triggered = self._contingent.evaluate(market_prices)
for contingent_order in triggered:
    self.execute_order(contingent_order, market_prices)
```

### On SELL (manual close)
Call `self._contingent.cancel(order.ticker)` after a manual SELL fills to remove pending legs.

### TIF simulation
- `GTC` / `DAY`: paper orders fill instantly — treated identically (no real time concept in simulation)
- `IOC`: fills the full requested quantity (paper always fills immediately); no special logic needed
- `FOK`: semantically equivalent to a standard MARKET order in paper simulation — the simulator always fills the full requested quantity or rejects entirely. The existing cash/position rejection paths handle this; no additional FOK-specific logic is required.

### `STOP` order type
Executes as a market SELL when `market_prices[ticker] <= order.stop_loss_price`.
Checked directly in `execute_order()` — if the price condition is not met, the order is rejected with status `REJECTED`.

---

## `AlpacaTrader` Changes (`src/execution/alpaca_trader.py`)

No `ContingentOrderSimulator` — Alpaca monitors server-side.

### TIF mapping (new private helper — applies to ALL order paths)

```python
def _map_tif(tif: TimeInForce) -> alpaca.TimeInForce:
    return {
        TimeInForce.GTC: alpaca.TimeInForce.GTC,
        TimeInForce.DAY: alpaca.TimeInForce.DAY,
        TimeInForce.IOC: alpaca.TimeInForce.IOC,
        TimeInForce.FOK: alpaca.TimeInForce.FOK,
    }[tif]
```

**Important:** The existing `_submit_order()` hardcodes `time_in_force=alpaca.TimeInForce.DAY`. Replace this in all existing MARKET and LIMIT paths with `_map_tif(order.time_in_force)`. This ensures GTC/IOC/FOK are honoured for non-bracket orders too.

### Bracket orders — three paths based on which legs are present

**Both TP and SL set** → Alpaca `order_class="bracket"` (requires both legs):
```python
request = MarketOrderRequest(
    symbol=symbol, qty=order.quantity, side=OrderSide.BUY,
    time_in_force=_map_tif(order.time_in_force),
    order_class="bracket",
    take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
    stop_loss=StopLossRequest(stop_price=order.stop_loss_price),
)
```

**SL only** → Alpaca `order_class="oto"` (one-triggers-other):
```python
request = MarketOrderRequest(
    symbol=symbol, qty=order.quantity, side=OrderSide.BUY,
    time_in_force=_map_tif(order.time_in_force),
    order_class="oto",
    stop_loss=StopLossRequest(stop_price=order.stop_loss_price),
)
```

**TP only** → Alpaca `order_class="oto"`:
```python
request = MarketOrderRequest(
    symbol=symbol, qty=order.quantity, side=OrderSide.BUY,
    time_in_force=_map_tif(order.time_in_force),
    order_class="oto",
    take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
)
```

### Trailing stops
If `order.order_type == OrderType.TRAILING_STOP`:
```python
# `side` is already mapped to AlpacaOrderSide earlier in _submit_order()
request = TrailingStopOrderRequest(
    symbol=symbol, qty=order.quantity, side=side,
    time_in_force=_map_tif(order.time_in_force),
    trail_percent=order.trail_percent,
)
```

---

## `CcxtTrader` Changes (`src/execution/ccxt_trader.py`)

### Composition
```python
self._contingent = ContingentOrderSimulator()
```

Identical pattern to PaperTrader: register after BUY fills (BUY-side only), call `cleanup_stale()` + evaluate in `snapshot()`, cancel on manual close.

### TIF passthrough — LIMIT orders only
CCXT supports TIF on most exchanges for LIMIT orders. MARKET orders must not include `timeInForce` (strict exchanges like Binance reject it):

```python
params: dict = {}
if order.order_type == OrderType.LIMIT:
    params["timeInForce"] = order.time_in_force.value
ccxt_order = ex.create_order(symbol, order_type, side, qty, price, params=params)
```

### STOP / TRAILING_STOP
Routed through `ContingentOrderSimulator` (not native CCXT — support is exchange-specific and unreliable). The `order.stop_loss_price` and `order.trail_percent` fields are registered in the simulator after the parent BUY fill; triggers execute as market SELLs via the existing `execute_order()` path.

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `src/utils/schemas.py` | Modify | Add `TimeInForce`, extend `OrderType`, extend `Order` with validator |
| `src/execution/order_manager.py` | **Create** | `ContingentOrderSimulator` + `ContingentEntry` |
| `src/execution/paper_trader.py` | Modify | Compose simulator, bracket/trailing/TIF/STOP support |
| `src/execution/alpaca_trader.py` | Modify | Native bracket/trailing/TIF, replace hardcoded TIF.DAY |
| `src/execution/ccxt_trader.py` | Modify | Compose simulator, TIF passthrough (LIMIT only) |
| `tests/test_order_manager.py` | **Create** | Simulator unit tests (13 tests) |
| `tests/test_advanced_orders.py` | **Create** | End-to-end tests across all brokers (21 tests) |

---

## Testing Strategy

### `tests/test_order_manager.py` (unit tests, no broker needed)
1. `test_register_bracket` — entry stored with correct TP/SL prices
2. `test_register_trailing_stop` — entry stored with trail_percent and trail_peak = entry_price
3. `test_register_overwrites_existing_ticker` — second register() on same ticker replaces first entry
4. `test_tp_triggers_when_price_above` — evaluate returns SELL with fill_price=take_profit_price
5. `test_tp_does_not_trigger_below` — no trigger when price < TP
6. `test_sl_triggers_when_price_below` — evaluate returns SELL with fill_price=stop_loss_price
7. `test_sl_does_not_trigger_above` — no trigger when price > SL
8. `test_trailing_stop_peak_ratchets_up` — update_trailing_peaks raises peak
9. `test_trailing_stop_peak_does_not_fall` — peak never decreases
10. `test_trailing_stop_triggers_from_peak` — fires when price falls trail_percent% from peak
11. `test_oco_both_legs_removed_on_tp_trigger` — after TP fires, ticker not in open_tickers
12. `test_cancel_removes_entry` — cancel() removes ticker from open_tickers
13. `test_evaluate_multiple_tickers` — handles multiple concurrent positions
14. `test_cleanup_stale_removes_unheld_tickers` — cleanup_stale() removes entries for gone positions

### `tests/test_advanced_orders.py` (per-broker tests, all mocked)

**PaperTrader (9 tests):**
1. `test_paper_bracket_tp_fills_on_snapshot`
2. `test_paper_bracket_sl_fills_on_snapshot`
3. `test_paper_trailing_stop_triggers_after_rally`
4. `test_paper_trailing_stop_does_not_trigger_during_rally`
5. `test_paper_stop_order_rejects_if_price_above_stop`
6. `test_paper_stop_order_fills_when_price_at_stop`
7. `test_paper_fok_rejects_if_insufficient_cash`
8. `test_paper_manual_sell_cancels_contingent`
9. `test_paper_no_double_fill_after_contingent_fires`

**AlpacaTrader (7 tests, all mocked):**
10. `test_alpaca_both_legs_submits_bracket_request`
11. `test_alpaca_sl_only_submits_oto_request`
12. `test_alpaca_tp_only_submits_oto_request`
13. `test_alpaca_trailing_stop_submits_trailing_stop_request`
14. `test_alpaca_tif_gtc_applied_to_non_bracket_order`
15. `test_alpaca_tif_day_mapped_correctly`
16. `test_alpaca_tif_ioc_fok_mapped_correctly`

**CcxtTrader (6 tests, all mocked):**
17. `test_ccxt_bracket_registers_contingent_after_fill`
18. `test_ccxt_bracket_tp_triggers_in_snapshot`
19. `test_ccxt_bracket_sl_triggers_in_snapshot`
20. `test_ccxt_trailing_stop_triggers`
21. `test_ccxt_tif_passed_only_for_limit_orders`
22. `test_ccxt_market_order_no_tif_param`

---

## Constraints and Scope

- `BaseBroker` protocol: **unchanged** — no new required methods
- All new `Order` fields are optional with safe defaults — zero breaking changes to existing callers
- `signal_translator.py`: unchanged — continues producing plain MARKET orders with default TIF=GTC
- `risk_controls.py`: unchanged — `check_order()` operates on the parent order only; contingent legs bypass risk checks (they are defensive exits, not new position entries)
- **Scope limitation:** Bracket fields on optimizer-generated orders are not supported in this iteration. The `portfolio_optimizer.py` → `signal_translator.py` → `target_weights_to_orders()` pipeline produces plain MARKET orders. Advanced order types (bracket, trailing) are only available when callers manually construct `Order` objects with the new fields. This is a known scope boundary for Tier 3C.
