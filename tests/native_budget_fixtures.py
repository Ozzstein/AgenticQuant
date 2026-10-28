"""Synthetic public gateway protocol fixtures; no provider authority claims."""
import copy
import hashlib
from datetime import timedelta, timezone


def receipt(req, now):
    value = {
        "schema_version": "1.0",
        "run_id": req.run_id,
        "config_hash": req.gateway_config_hash,
        "ready": True,
        "provider_configured": True,
        "streaming": False,
        "blocked": False,
        "expires_at": (now + timedelta(days=2)).isoformat(),
        "models": {
            scope: {"provider": "openai", "model": model.removeprefix("openai/")}
            for scope, model in (("chat", req.chat_model), ("embedding", req.embedding_model))
        },
        "admission_available": {"chat": True, "embedding": True},
        "limits": req.gateway_limits.model_dump(mode="json", exclude={"schema_version"}),
        "day_limits": req.gateway_day_limits.model_dump(mode="json", exclude={"schema_version"}),
        "observed_models": [req.chat_model.removeprefix("openai/")],
        "charged": {"requests": 1, "tokens": 15, "cost_usd": "0.01"},
        "actual": {
            "requests_with_usage": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": "0.01",
        },
        "uncertain": {"requests": 0, "reserved_tokens": 0, "reserved_cost_usd": "0"},
    }

    scope = {
        "requests": 1, "pending_requests": 0,
        "known_input_tokens": 10, "known_output_tokens": 5,
        "known_total_tokens_lower_bound": 15, "known_cost_lower_bound_usd": "0.01",
        "actual_tokens": 15, "actual_cost_usd": "0.01",
        "usage_unknown_requests": 0, "unpriced_requests": 0,
        "conservative_tokens": 15, "conservative_cost_usd": "0.01",
        "uncertain_tokens": 0, "uncertain_cost_usd": "0",
        "breached": False, "breach_call_ids": [], "breach_reasons": [],
        "statuses": {"ok": 1}, "observed_models": value["observed_models"],
        "observed_model_hashes": [hashlib.sha256(value["observed_models"][0].encode()).hexdigest()],
    }
    value["accounting"] = {"schema_version": "2.0", "policy": "reservation-floor-2",
        "run_id": req.run_id, "as_of": now.isoformat(),
        "day_id": now.astimezone(timezone.utc).date().isoformat(),
        "run": scope, "day": copy.deepcopy(scope)}
    project(value)
    return value


def project(value):
    """Project a synthetic v2 protocol scope into its backward-compatible keys."""
    scope = value["accounting"]["run"]
    value.update(
        cost_semantics="verified_usage_with_reservation_floor",
        charged={"requests": scope["requests"], "tokens": scope["conservative_tokens"],
                 "cost_usd": scope["conservative_cost_usd"]},
        actual={"requests_with_usage": scope["requests"]-scope["unpriced_requests"],
                "input_tokens": scope["known_input_tokens"], "output_tokens": scope["known_output_tokens"],
                "cost_usd": scope["actual_cost_usd"]},
        uncertain={"requests": scope["unpriced_requests"], "reserved_tokens": scope["uncertain_tokens"],
                   "reserved_cost_usd": scope["uncertain_cost_usd"]},
        statuses=copy.deepcopy(scope["statuses"]), observed_models=list(scope["observed_models"]),
        blocked=scope["breached"] or value["accounting"]["day"]["breached"],
    )


def pending(value):
    scope = value["accounting"]["run"]
    scope.update(requests=2, pending_requests=1, usage_unknown_requests=1, unpriced_requests=1,
                 actual_tokens=None, actual_cost_usd=None, conservative_tokens=25,
                 conservative_cost_usd="0.02", uncertain_tokens=10, uncertain_cost_usd="0.01",
                 statuses={"ok": 1, "reserved": 1})
    value["accounting"]["day"] = copy.deepcopy(scope)
    project(value)
