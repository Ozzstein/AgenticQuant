"""Pure exact native submission terms and existing definite rejection predicates."""


def submission_payload(venue, intent, client, *, symbol, native_id=None):
    if venue == "alpaca":
        result = {
            "symbol": symbol,
            "qty": str(intent.quantity),
            "side": intent.side,
            "type": intent.order_type,
            "time_in_force": intent.tif.lower(),
            "client_order_id": client,
            "extended_hours": False,
            "order_class": "simple",
        }
        for name in ("limit_price", "stop_price", "trail_price", "trail_percent"):
            if getattr(intent, name) is not None:
                result[name] = str(getattr(intent, name))
        return result
    if venue == "binance":
        from src.execution.binance_venue import order_type, trailing_delta

        result = {
            "symbol": symbol,
            "side": intent.side.upper(),
            "type": order_type(intent),
            "quantity": str(intent.quantity),
            "newClientOrderId": client,
            "newOrderRespType": "RESULT",
        }
        if intent.order_type == "limit":
            result.update(price=str(intent.limit_price), timeInForce=intent.tif)
        if intent.order_type == "stop":
            result["stopPrice"] = str(intent.stop_price)
        elif intent.order_type == "trailing_stop":
            result["trailingDelta"] = trailing_delta(intent)
        return result
    if venue == "kraken":
        from src.execution.kraken_venue import order_type, trailing_price

        result = {
            "pair": native_id,
            "type": intent.side,
            "ordertype": order_type(intent),
            "volume": str(intent.quantity),
            "cl_ord_id": client,
            "oflags": "fciq",
            "timeinforce": intent.tif,
        }
        if intent.order_type == "trailing_stop":
            result.update(price=trailing_price(intent), trigger="last")
        elif intent.order_type != "market":
            result["price"] = str(
                intent.limit_price if intent.order_type == "limit" else intent.stop_price
            )
        return result
    raise ValueError("unsupported native submission venue")


def definitive_rejection(venue, status, value):
    code = value.get("code") if isinstance(value, dict) else None
    if venue == "alpaca":
        return status in (400, 401, 403, 422) and str(code) != "42210000"
    if venue == "binance":
        return status == 400 and code in {
            -1013,
            -1100,
            -1101,
            -1102,
            -1103,
            -1111,
            -1116,
            -1117,
            -1118,
            -1121,
            -1130,
            -2014,
            -2015,
        }
    if venue == "kraken":
        errors = value.get("error") if isinstance(value, dict) else None
        definite = {
            "EAPI:Invalid key",
            "EAPI:Invalid signature",
            "EAPI:Invalid nonce",
            "EGeneral:Permission denied",
            "EOrder:Insufficient funds",
            "EOrder:Order minimum not met",
            "EOrder:Cost minimum not met",
            "EOrder:Tick size check failed",
            "EOrder:Orders limit exceeded",
            "EOrder:Invalid price",
        }
        return (
            status == 200
            and isinstance(errors, list)
            and bool(errors)
            and all(
                isinstance(error, str)
                and (error in definite or error.startswith("EGeneral:Invalid arguments:"))
                for error in errors
            )
        )
    return False
