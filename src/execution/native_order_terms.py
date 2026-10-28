"""Pure native order/intent checks shared by live parsing and archive audit."""

from src.execution.ledger import LedgerError


def require_alpaca_order(intent, client_order_id, order, symbol):
    from src.execution.alpaca_venue import _decimal

    if (
        not order.get("id")
        or order.get("extended_hours") is not False
        or order.get("order_class") not in ("", "simple")
        or order.get("client_order_id") != client_order_id
        or order.get("symbol") != symbol
        or order.get("side") != intent.side
        or _decimal(order.get("qty")) != intent.quantity
        or order.get("type") != intent.order_type
        or order.get("time_in_force", "").upper() != intent.tif
    ):
        raise LedgerError("remote acknowledgement differs from exact intended order")
    keys = (
        ("limit_price", "trail_price", "trail_percent")
        if intent.order_type == "trailing_stop"
        else ("limit_price", "stop_price", "trail_price", "trail_percent")
    )
    for key in keys:
        actual = order.get(key)
        if (None if actual is None else _decimal(actual)) != getattr(intent, key):
            raise LedgerError("remote acknowledgement price differs from intended order")
    if intent.order_type == "trailing_stop":
        for key in ("hwm", "stop_price"):
            if order.get(key) is not None and _decimal(order[key]) <= 0:
                raise LedgerError("native trailing observation must be positive or null")


def require_binance_order(intent, client_order_id, order, market):
    from src.execution.binance_venue import native_integer, order_type, trailing_delta
    from src.execution.ccxt_transport import decimal

    native_integer(order.get("orderId"))
    if (
        order.get("symbol") != market.symbol
        or order.get("clientOrderId") != client_order_id
        or order.get("side") != intent.side.upper()
        or order.get("type") != order_type(intent)
        or decimal(order["origQty"]) != intent.quantity
        or order.get("orderListId", -1) != -1
    ):
        raise LedgerError("native Binance order differs from exact submitted intent")
    if (
        order.get("timeInForce") != intent.tif
        or intent.order_type == "limit"
        and decimal(order["price"]) != intent.limit_price
        or intent.order_type == "stop"
        and decimal(order["stopPrice"]) != intent.stop_price
    ):
        raise LedgerError("native Binance order price/time-in-force differs")
    if intent.order_type == "trailing_stop":
        if native_integer(order.get("trailingDelta")) != trailing_delta(intent):
            raise LedgerError("native Binance trailing delta differs from original intent")
        if decimal(order.get("stopPrice", "0")) < 0:
            raise LedgerError("native Binance trailing stop observation is negative")
    elif order.get("trailingDelta") is not None:
        raise LedgerError("native Binance order unexpectedly has trailing terms")


def require_kraken_order(intent, client_id, order, market):
    from src.execution.ccxt_transport import decimal
    from src.execution.kraken_venue import order_type, require_trailing_price

    description = order["descr"]
    if (
        order.get("cl_ord_id") != client_id
        or description.get("pair") not in (market.native_id, market.symbol)
        or description.get("type") != intent.side
        or description.get("ordertype") != order_type(intent)
        or description.get("leverage") != "none"
        or description.get("close") not in (None, "")
        or decimal(order["vol"]) != intent.quantity
        or decimal(order.get("expiretm", 0))
    ):
        raise LedgerError("native Kraken order differs from exact spot intent")
    if set(filter(None, order.get("oflags", "").split(","))) - {"fciq", "fcib"}:
        raise LedgerError("unexpected native Kraken order flags")
    if intent.order_type == "trailing_stop":
        require_trailing_price(intent, description.get("price"))
        if order.get("trigger") != "last":
            raise LedgerError(
                "native Kraken trailing trigger requires the original last-price basis"
            )
        for key in ("stopprice", "limitprice"):
            if key in order and decimal(order[key]) < 0:
                raise LedgerError("native Kraken trailing observation is negative")
        if description.get("price2") is not None and decimal(description["price2"]):
            raise LedgerError("native Kraken market trailing order cannot have a limit price")
    elif intent.order_type != "market" and decimal(description["price"]) != (
        intent.limit_price if intent.order_type == "limit" else intent.stop_price
    ):
        raise LedgerError("native Kraken order price differs")
