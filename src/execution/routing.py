"""Canonical Decimal target sizing before risk, venue routing and sleeve netting."""

from __future__ import annotations

from datetime import timedelta
from decimal import ROUND_DOWN
from decimal import Decimal as D

from src.domain.portfolio import (
    FillPolicy,
    InstrumentRisk,
    OrderIntent,
    PortfolioSnapshot,
    PortfolioTarget,
    PriceMark,
)
from src.execution.ledger import LedgerError


def target_to_intents(
    target: PortfolioTarget,
    snapshot: PortfolioSnapshot,
    marks: dict[str, PriceMark],
    metadata: dict[str, InstrumentRisk],
    policy: FillPolicy,
    *,
    max_price_age_seconds: int,
) -> list[OrderIntent]:
    """Size against actual holdings. Pending sells never supply buying power.

    Caller cancels/safely reconciles superseded orders before this sizing step;
    every returned intent still passes the canonical aggregate execution risk.
    """
    at = target.decision_at
    if snapshot.as_of != at or snapshot.reserved_cash:
        raise LedgerError(
            "target sizing requires current snapshot and settled/cancelled prior intents"
        )
    held = {p.instrument.instrument_id: p.quantity for p in snapshot.positions}
    weights = {w.instrument.instrument_id: w.weight for w in target.weights}
    identities = {w.instrument.instrument_id: w.instrument for w in target.weights}
    identities.update({p.instrument.instrument_id: p.instrument for p in snapshot.positions})
    candidates = []
    for key in sorted(set(held) | set(weights)):
        if key not in marks or key not in metadata:
            raise LedgerError(f"target lacks observed price or risk metadata: {key}")
        mark, meta = marks[key], metadata[key]
        if mark.instrument != identities[key] or meta.instrument != identities[key]:
            raise LedgerError("target/mark/metadata instrument identity mismatch")
        if not 0 <= (at - mark.as_of).total_seconds() <= max_price_age_seconds:
            raise LedgerError(f"target uses stale/future price: {key}")
        desired = (
            snapshot.nav * weights.get(key, D(0)) / (mark.price * mark.fx_rate) / meta.lot_size
        ).to_integral_value(rounding=ROUND_DOWN) * meta.lot_size
        if key in target.preserve_quantities:
            if target.preserve_quantities[key] != held.get(key, D(0)):
                raise LedgerError("preserved quantity differs from actual current holdings")
            desired = target.preserve_quantities[key]
        delta = desired - held.get(key, D(0))
        if delta == 0:
            continue
        side, qty = ("buy" if delta > 0 else "sell"), abs(delta)
        price = mark.price * (1 + policy.slippage_bps / 10000 * (1 if side == "buy" else -1))
        if side == "buy":
            bps = (
                policy.crypto_cost_bps
                if identities[key].asset_class == "crypto"
                else policy.equity_cost_bps
            )
            budget = snapshot.nav * weights.get(key, D(0))
            qty = max(
                D(0),
                (
                    (budget - policy.minimum_commission)
                    / (price * mark.fx_rate * (1 + bps / 10000) + policy.commission_per_unit)
                    - held.get(key, D(0))
                )
                / meta.lot_size,
            )
            qty = qty.to_integral_value(rounding=ROUND_DOWN) * meta.lot_size
        if qty <= 0:
            continue
        candidates.append(
            OrderIntent(
                intent_id=target.target_id + ":" + key,
                account_id=snapshot.account_id,
                run_id=target.target_id,
                instrument=identities[key],
                side=side,
                quantity=qty,
                decision_at=at,
                eligible_at=at + timedelta(microseconds=1),
                reference_price=price,
                fx_rate=mark.fx_rate,
                estimated_fee=policy.fee(identities[key], qty, price, mark.fx_rate),
                strategy_id=target.strategy_id,
            )
        )
    return sorted(candidates, key=lambda o: (o.side != "sell", o.instrument.instrument_id))


def intent_for_delta(target, snapshot, mark, delta, policy):
    """Explicit post-netting quantity; risk still checks all projected exposures."""
    if not delta:
        raise LedgerError("zero quantity cannot create an intent")
    side = "buy" if delta > 0 else "sell"
    price = mark.price * (1 + policy.slippage_bps / 10000 * (1 if delta > 0 else -1))
    qty = abs(delta)
    return OrderIntent(
        intent_id=target.target_id + ":" + mark.instrument.instrument_id,
        account_id=snapshot.account_id,
        run_id=target.target_id,
        instrument=mark.instrument,
        side=side,
        quantity=qty,
        decision_at=target.decision_at,
        eligible_at=target.decision_at + timedelta(microseconds=1),
        reference_price=price,
        fx_rate=mark.fx_rate,
        estimated_fee=policy.fee(mark.instrument, qty, price, mark.fx_rate),
        strategy_id=target.strategy_id,
    )


def target_from_blended_frame(frame, context, instruments, *, target_id, current_snapshot, outputs):
    """Adapt D's numeric frame with actual actions and quantities at the boundary.

    Float contributions are converted explicitly through their decimal text.
    HOLD/ABSTAIN/VETO retain actual Decimal ownership, never a rounded weight
    reconstruction. The complete per-sleeve decision rows/provenance are hashed.
    """
    import hashlib
    import json

    import numpy as np
    import pandas as pd

    from src.domain.portfolio import TargetWeight

    attrs = frame.attrs
    required = {
        "context",
        "allocations",
        "data_snapshot_hash",
        "universe_version",
        "signal_at",
        "strategy_hashes",
        "model_artifact_hashes",
        "sleeve_provenance",
        "contributions",
        "vetoed",
    }
    if (
        required - attrs.keys()
        or attrs["context"] != context.model_dump(mode="json")
        or pd.Timestamp(attrs["signal_at"]).tzinfo is None
        or not pd.Timestamp(attrs["signal_at"]) <= context.as_of
    ):
        raise LedgerError(
            "blended target requires exact causal context and complete sleeve provenance"
        )
    if current_snapshot.account_id != context.account_id or current_snapshot.as_of != context.as_of:
        raise LedgerError("actual current account snapshot required for blended target")
    if attrs["data_snapshot_hash"] not in context.data_snapshot_ids:
        raise LedgerError("blended snapshot is outside RunContext evidence")
    names = set(attrs["contributions"])
    if (
        not names
        or names != set(outputs)
        or names != set(current_snapshot.sleeves)
        or any(
            names != set(attrs[k])
            for k in (
                "allocations",
                "strategy_hashes",
                "model_artifact_hashes",
                "sleeve_provenance",
            )
        )
    ):
        raise LedgerError("all actual sleeves, decisions and provenance must be represented")
    if not frame.index.is_unique or current_snapshot.nav <= 0:
        raise LedgerError("unique instrument rows and positive actual NAV required")
    projected = pd.DataFrame(attrs["contributions"]).fillna(0).sum(axis=1).reindex(frame.index)
    if not np.isfinite(frame.target_weight).all() or not np.array_equal(
        projected.to_numpy(), frame.target_weight.to_numpy()
    ):
        raise LedgerError("numeric aggregate frame differs from its actual sleeve contributions")
    identities = {i.instrument_id: i for i in instruments}
    contributions, preserve, sources = {}, {}, {}
    for name in sorted(names):
        output, provenance = outputs[name], attrs["sleeve_provenance"][name]
        if (
            output.attrs != provenance
            or provenance.get("strategy") != name
            or provenance.get("strategy_hash") != attrs["strategy_hashes"][name]
            or provenance.get("model_hash") != attrs["model_artifact_hashes"][name]
        ):
            raise LedgerError("per-sleeve output differs from bound provenance")
        if any(
            provenance.get(k) != attrs[k]
            for k in ("context", "signal_at", "data_snapshot_hash", "universe_version")
        ):
            raise LedgerError("per-sleeve causal evidence mismatch")
        if not output.index.is_unique or not {"target_weight", "action", "reason", "score"} <= set(
            output
        ):
            raise LedgerError("actual per-sleeve actions and decision rows required")
        contributions[name] = {
            key: D(str(value)) for key, value in attrs["contributions"][name].items()
        }
        held = {
            p.instrument.instrument_id: p
            for p in current_snapshot.sleeves[name].portfolio.positions
        }
        if set(held) - contributions[name].keys():
            raise LedgerError("complete actual held-position contribution coverage required")
        preserve[name] = {}
        for key, value in contributions[name].items():
            if (
                key not in identities
                or not value.is_finite()
                or value < 0
                or key not in output.index
            ):
                raise LedgerError("invalid sleeve contribution identity/weight")
            action = str(output.loc[key, "action"])
            if action in ("HOLD", "ABSTAIN", "VETO"):
                actual = held[key].market_value / current_snapshot.nav if key in held else D(0)
                if float(value) != float(actual):
                    raise LedgerError("preserve action differs from actual current contribution")
                preserve[name][key] = held[key].quantity if key in held else D(0)
            elif action == "EXIT" and value != 0:
                raise LedgerError("EXIT requires zero contribution")
            elif action == "REDUCE" and float(value) > float(
                held[key].market_value / current_snapshot.nav if key in held else D(0)
            ):
                raise LedgerError("REDUCE cannot add sleeve exposure")
            elif action not in ("BUY", "REDUCE", "EXIT"):
                raise LedgerError("unknown sleeve action")
        rows = json.loads(output.to_json(orient="index", date_format="iso"))
        source = {
            "provenance": provenance,
            "rows": rows,
            "allocation": attrs["allocations"][name],
            "actual_sleeve_source": current_snapshot.sleeves[name].source_hash,
        }
        sources[name] = hashlib.sha256(
            json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    keys = sorted({key for c in contributions.values() for key in c})
    weights = tuple(
        TargetWeight(
            instrument=identities[key],
            weight=sum((c.get(key, D(0)) for c in contributions.values()), D(0)),
        )
        for key in keys
    )
    artifact = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return PortfolioTarget(
        target_id=target_id,
        decision_at=context.as_of,
        universe_version=attrs["universe_version"],
        strategy_id="blend",
        artifact_version=artifact,
        data_snapshot_id=attrs["data_snapshot_hash"],
        weights=weights,
        sleeve_contributions=contributions,
        sleeve_sources=sources,
        sleeve_preserve_quantities=preserve,
    )


def target_from_strategy_frame(frame, context, instruments, *, target_id, current_snapshot):
    """Canonical single-strategy adapter, including exact passive ownership."""
    import hashlib
    import json

    import numpy as np
    import pandas as pd

    from src.domain.portfolio import TargetWeight

    attrs = frame.attrs
    required = {
        "strategy",
        "strategy_hash",
        "context",
        "signal_at",
        "data_snapshot_hash",
        "universe_version",
        "model_hash",
    }
    if (
        required - attrs.keys()
        or attrs["context"] != context.model_dump(mode="json")
        or pd.Timestamp(attrs["signal_at"]).tzinfo is None
        or not pd.Timestamp(attrs["signal_at"]) <= context.as_of
    ):
        raise LedgerError("strategy frame requires exact causal context and provenance")
    if (
        attrs["data_snapshot_hash"] not in context.data_snapshot_ids
        or current_snapshot.as_of != context.as_of
        or current_snapshot.account_id != context.account_id
    ):
        raise LedgerError("strategy frame data/account cutoff differs from actual snapshot")
    if current_snapshot.sleeves:
        raise LedgerError("use the blended adapter for an attributed multi-sleeve book")
    if (
        not frame.index.is_unique
        or not {"target_weight", "action", "reason", "score"} <= set(frame)
        or not np.isfinite(frame.target_weight).all()
        or (frame.target_weight < 0).any()
    ):
        raise LedgerError("actual finite strategy weights, actions and decision rows required")
    held = {p.instrument.instrument_id: p for p in current_snapshot.positions}
    identities = {i.instrument_id: i for i in instruments}
    if (
        set(held) - set(frame.index)
        or set(frame.index) - identities.keys()
        or current_snapshot.nav <= 0
    ):
        raise LedgerError(
            "complete actual held/target instrument coverage and positive NAV required"
        )
    weights, preserve = [], {}
    for key, row in frame.iterrows():
        action = str(row.action)
        weight = D(str(row.target_weight))
        current = held[key].market_value / current_snapshot.nav if key in held else D(0)
        if action in ("HOLD", "ABSTAIN", "VETO"):
            if float(weight) != float(current):
                raise LedgerError("preserve action does not match actual held weight")
            preserve[key] = held[key].quantity if key in held else D(0)
            weight = current
        elif action == "EXIT":
            if weight != 0:
                raise LedgerError("EXIT requires zero target")
        elif action == "REDUCE":
            if float(weight) > float(current):
                raise LedgerError("REDUCE cannot add exposure")
        elif action != "BUY":
            raise LedgerError("unknown strategy action")
        weights.append(TargetWeight(instrument=identities[key], weight=weight))
    source = {
        "provenance": attrs,
        "rows": json.loads(frame.to_json(orient="index", date_format="iso")),
        "snapshot_hash": current_snapshot.content_hash(),
    }
    artifact = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return PortfolioTarget(
        target_id=target_id,
        decision_at=context.as_of,
        universe_version=attrs["universe_version"],
        strategy_id=attrs["strategy"],
        artifact_version=artifact,
        data_snapshot_id=attrs["data_snapshot_hash"],
        weights=tuple(weights),
        preserve_quantities=preserve,
    )
