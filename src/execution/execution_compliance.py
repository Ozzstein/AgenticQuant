"""Read-only native execution compliance, independent of mutable application flags.

This module cannot book an execution. Missing retained authority is a failure;
the absence of a discrepancy index is never evidence of compliance.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from src.domain.common import _utc
from src.domain.portfolio import AccountSpec, Fill
from src.execution.discrepancies import (
    AdmissionViolation,
    ExecutionComplianceAudit,
    ExecutionDiscrepancy,
    ViolationBinding,
)
from src.execution.ledger import LedgerError
from src.execution.native_archive import NativeArchive, decode, digest
from src.execution.native_authority import reexport_report
from src.execution.venue import VenueOrderReport

_INVALID = (ValueError, KeyError, TypeError, IndexError, ArithmeticError, AttributeError)
_AUDIT_PREFIXES = (
    "execution-authority:",
    "execution-violation:",
    "execution-discrepancy:",
    "execution-assessment:",
)
_NATIVE = ("alpaca", "binance", "kraken")


@dataclass(frozen=True)
class ComplianceFacts:
    audit: ExecutionComplianceAudit
    authorities: tuple
    violations: tuple[AdmissionViolation, ...]


def _rows(archive, table):
    if not archive.table_exists(table):
        return []
    return [
        dict(row)
        for row in archive.ledger.db.execute(
            f'SELECT rowid AS retained_sequence,* FROM "{table}" WHERE account=? ORDER BY rowid',
            (archive.account_id,),
        )
    ]


def _lineage(archive, events, fills, reports, book):
    initial = archive.ledger.db.execute(
        "SELECT payload FROM journal WHERE account=? AND id='initial' AND kind='deposit'",
        (archive.account_id,),
    ).fetchone()
    if (
        archive.spec.account_id != archive.account_id
        or initial is None
        or AccountSpec.model_validate_json(initial[0]) != archive.spec
    ):
        raise LedgerError("original account identity missing/changed")
    native = any(
        key.startswith(
            tuple(
                venue + suffix
                for venue in _NATIVE
                for suffix in ("-http:", "-order:", "-native-trade:", "-submit-attempt:")
            )
        )
        or key in tuple("remote-binding:" + venue for venue in _NATIVE)
        or key.startswith("lean-native-")
        for key in events
    ) or any(fill.fill_id.startswith(tuple(venue + ":" for venue in _NATIVE)) for fill in fills)
    native |= any(row["venue"] in _NATIVE for row in reports)
    configuration = decode(book["configuration"]) if book else None
    venues = set(configuration["venues"]) if configuration else set()
    if book and book["owner"] == "lean" and "remote-binding:lean" not in events:
        raise LedgerError("original LEAN deployment authority unavailable")
    native |= bool(venues & set(_NATIVE))
    if "remote-binding:lean" in events:
        from src.execution.lean_venue import LeanDeployment

        deployment = LeanDeployment.model_validate(decode(events["remote-binding:lean"]["payload"]))
        if (
            not book
            or book["owner"] != "lean"
            or venues != {"lean"}
            or deployment.account_id != archive.account_id
            or deployment.mode != archive.spec.mode
        ):
            raise LedgerError("original LEAN execution lineage differs")
        if deployment.brokerage == "PaperBrokerage":
            if native or deployment.mode != "paper":
                raise LedgerError("native execution cannot acquire local LEAN lineage")
            return "lean-paper"
        native = True
    if native:
        if (
            not book
            or book["owner"] not in ("direct", "lean")
            or archive.spec.mode not in ("live", "paper")
        ):
            raise LedgerError("original native execution lineage unavailable")
        expected = (
            venues
            if book["owner"] == "direct"
            else {
                {
                    "AlpacaBrokerage": "alpaca",
                    "BinanceBrokerage": "binance",
                    "KrakenBrokerage": "kraken",
                }[deployment.brokerage]
            }
        )
        if not expected or not expected <= set(_NATIVE):
            raise LedgerError("unsupported mixed native execution lineage")
        for venue in expected:
            binding = decode(events["remote-binding:" + venue]["payload"])
            mode = binding.get("mode")
            if venue == "alpaca":
                mode = {
                    "https://paper-api.alpaca.markets": "paper",
                    "https://api.alpaca.markets": "live",
                }.get(binding.get("endpoint"))
            if mode != archive.spec.mode or not binding.get("broker_account_id"):
                raise LedgerError("native original account mode/identity differs")
        return "native"
    if book and (book["owner"] != "direct" or venues != {"paper"}):
        raise LedgerError("unknown canonical execution owner")
    if archive.spec.mode == "backtest":
        return "historical-replay"
    if archive.spec.mode == "paper":
        return "local-paper"
    raise LedgerError("live account has no original native execution authority")


def _same_fact(first, later):
    omitted = {"original_report", "first_complete_at", "sources", "source_manifest_hash"}
    return (
        first.model_dump(exclude=omitted) == later.model_dump(exclude=omitted)
        and first.first_complete_at <= later.first_complete_at
    )


def _violation(archive, authority):
    intent = archive.ledger.order(archive.account_id, authority.intent_id)["intent"]
    fill, reasons = authority.fill, []
    if fill.executed_at < intent.eligible_at:
        reasons.append("before_eligibility")
    if intent.expires_at is not None and fill.executed_at >= intent.expires_at:
        reasons.append("outside_expiry")
    if intent.limit_price is not None and (
        intent.side == "buy"
        and fill.price > intent.limit_price
        or intent.side == "sell"
        and fill.price < intent.limit_price
    ):
        reasons.append("limit_breach")
    original = VenueOrderReport.model_validate(archive.verify_reference(authority.original_report))
    if original.cumulative_quantity > intent.quantity:
        reasons.append("quantity_excess")
    if not reasons:
        return None
    return AdmissionViolation(
        case_id=digest([archive.account_id, authority.key.model_dump(mode="json")]),
        authority_hash=authority.content_hash(),
        intent_hash=intent.content_hash(),
        reason_codes=tuple(sorted(reasons)),
        executed_at=fill.executed_at,
        admitted_eligible_at=intent.eligible_at,
        admitted_expires_at=intent.expires_at,
        first_observed_at=authority.first_observed_at,
    )


def _raw_executions(archive, row):
    """Discover source members even when report/index/application rows were removed."""
    identity, header = row["id"], decode(row["payload"])
    venue = identity.split("-http:", 1)[0]
    path = (
        header.get("path") if venue == "alpaca" else header.get("endpoint", "").rsplit("/", 1)[-1]
    )
    relevant = (
        path
        == {"alpaca": "/v2/account/activities", "binance": "myTrades", "kraken": "QueryTrades"}[
            venue
        ]
    )
    if not relevant or header.get("status") != 200:
        return ()
    if venue == "alpaca" and (header.get("params") or {}).get("activity_types") != "FILL":
        return ()
    network = (
        ("alpaca-paper" if venue == "alpaca" else "binance-testnet")
        if archive.spec.mode == "paper"
        else "mainnet"
    )
    _, payload, _, _ = archive.raw(identity, venue, network)
    if venue == "alpaca":
        if not isinstance(payload, list):
            raise LedgerError("native execution activity coverage invalid")
        return tuple("alpaca:" + item["id"] for item in payload if item["activity_type"] == "FILL")
    if venue == "binance":
        if not isinstance(payload, list):
            raise LedgerError("native execution trade coverage invalid")
        return tuple("binance:" + item["symbol"] + ":" + str(item["id"]) for item in payload)
    from src.execution.kraken_venue import result

    return tuple("kraken:" + identity for identity in result(payload))


def _raw_orders(archive, row, verified):
    """Discover order-side counts even if individual report/index rows vanished."""
    from src.execution.native_order_terms import (
        require_alpaca_order,
        require_binance_order,
        require_kraken_order,
    )
    from src.execution.spot_venue import NativeSpotMarket

    header = decode(row["payload"])
    venue = row["id"].split("-http:", 1)[0]
    path = (
        header.get("path") if venue == "alpaca" else header.get("endpoint", "").rsplit("/", 1)[-1]
    )
    relevant = (
        path.startswith("/v2/orders")
        if venue == "alpaca" and isinstance(path, str)
        else path == "order"
        if venue == "binance"
        else path in ("QueryOrders", "OpenOrders", "ClosedOrders")
    )
    if not relevant or header.get("status") != 200:
        return
    network = (
        ("alpaca-paper" if venue == "alpaca" else "binance-testnet")
        if archive.spec.mode == "paper"
        else "mainnet"
    )
    _, value, _, _ = archive.raw(row["id"], venue, network)
    if venue == "kraken":
        from src.execution.kraken_venue import result

        entries = result(value)
        if path != "QueryOrders":
            entries = entries["open" if path == "OpenOrders" else "closed"]
    elif venue == "alpaca" and isinstance(value, list):
        entries = {item["id"]: item for item in value}
    else:
        entries = {str(value["id" if venue == "alpaca" else "orderId"]): value}
    for identity, native in entries.items():
        matches = [
            (original, exported)
            for original, exported in verified
            if exported.native_report.venue == venue
            and exported.native_report.venue_order_id == identity
        ]
        if not matches:
            raise LedgerError("unbound native order source")
        original, exported = matches[0]
        intent = archive.ledger.order(archive.account_id, original["intent"])["intent"]
        if venue == "alpaca":
            mapping, _, _ = archive.event(
                "remote-instrument:alpaca:" + intent.instrument.instrument_id
            )
            require_alpaca_order(
                intent, exported.native_report.client_order_id, native, mapping["symbol"]
            )
            cumulative = Decimal(native["filled_qty"])
        else:
            binding, _, _ = archive.event("remote-binding:" + venue)
            market = NativeSpotMarket.model_validate(
                binding["markets"][intent.instrument.instrument_id]
            )
            if venue == "binance":
                require_binance_order(
                    intent, exported.native_report.client_order_id, native, market
                )
                cumulative = Decimal(native["executedQty"])
            else:
                client = (
                    native.get("cl_ord_id")
                    if exported.report.venue == "lean"
                    else exported.native_report.client_order_id
                )
                require_kraken_order(intent, client, native, market)
                cumulative = Decimal(native["vol_exec"])
        if cumulative < 0 or not any(
            item.native_report.executions_complete
            and item.native_report.cumulative_quantity >= cumulative
            for _, item in matches
        ):
            raise LedgerError("native raw order execution count lacks complete source closure")


def collect(ledger, account_id):
    """One current retained view. Deliberately no historical as_of escape hatch."""
    with ledger._reading():
        archive = NativeArchive(ledger, account_id, as_of=datetime.now(timezone.utc))
        tables = {
            name: _rows(archive, name)
            for name in (
                "events",
                "execution_books",
                "venue_outbox",
                "venue_reports",
                "fills",
                "orders",
                "journal",
            )
        }
        events = {row["id"]: row for row in tables["events"]}
        retained = [row for row in tables["events"] if not row["id"].startswith(_AUDIT_PREFIXES)]
        prefix = digest(
            {**tables, "events": retained, "account_spec": archive.spec.model_dump(mode="json")}
        )
        blockers, unresolved, authorities, verified = set(), {}, {}, []

        def failed(code, row=None, *, report=False):
            blockers.add(code)
            if row is not None:
                ref = archive.reference(
                    "venue-report" if report else "ledger-event",
                    row["evidence"] if report else row["id"],
                    row["payload"],
                    intent_id=row["intent"] if report else None,
                )
                unresolved[ref.content_hash()] = ref

        fills = {}
        for row in tables["fills"]:
            try:
                value = Fill.model_validate_json(row["payload"])
                if (
                    value.account_id != account_id
                    or value.fill_id != row["id"]
                    or value.intent_id != row["intent"]
                ):
                    raise LedgerError("canonical fill row identity differs")
                fills[value.fill_id] = value
            except _INVALID:
                failed("invalid_canonical_fill")
        report_values = []
        for row in tables["venue_reports"]:
            try:
                value = VenueOrderReport.model_validate_json(row["payload"])
                report_values.append({"venue": value.venue})
            except _INVALID:
                failed("invalid_retained_report", row, report=True)
        book = tables["execution_books"][0] if tables["execution_books"] else None
        try:
            lineage = _lineage(archive, events, fills.values(), report_values, book)
        except _INVALID:
            lineage = "unverified"
            failed("unverified_execution_lineage")

        if lineage in ("native", "unverified"):
            for row in tables["venue_reports"]:
                try:
                    ref = archive.reference(
                        "venue-report", row["evidence"], row["payload"], intent_id=row["intent"]
                    )
                    exported = reexport_report(ledger, ref, as_of=archive.as_of)
                    verified.append((row, exported))
                    for authority in exported.authorities:
                        identity = authority.fill.fill_id
                        if identity in authorities and not _same_fact(
                            authorities[identity], authority
                        ):
                            raise LedgerError("conflicting native complete execution prefixes")
                        authorities.setdefault(identity, authority)
                except _INVALID as exc:
                    code = (
                        "unsupported_external_valuation_closure"
                        if "unsupported_external_valuation_closure" in str(exc)
                        else "invalid_native_source_authority"
                    )
                    failed(code, row, report=True)
            # All previous complete prefixes remain members of every later complete prefix.
            # An honestly incomplete receipt may only be closed by an actual later full prefix.
            for row, exported in verified:
                same = [
                    other
                    for other_row, other in verified
                    if other_row["intent"] == row["intent"]
                    and other.report.venue_order_id == exported.report.venue_order_id
                ]
                old = {fill.fill_id: fill for fill in exported.report.fills}

                def covers(other):
                    new = {fill.fill_id: fill for fill in other.report.fills}
                    return (
                        other.report.executions_complete
                        and other.report.observed_at >= exported.report.observed_at
                        and other.report.cumulative_quantity >= exported.report.cumulative_quantity
                        and all(new.get(key) == value for key, value in old.items())
                    )

                if not exported.report.executions_complete and not any(
                    covers(other) for other in same
                ):
                    failed("incomplete_native_execution_prefix", row, report=True)
                elif exported.report.executions_complete and any(
                    other.report.executions_complete
                    and other.report.observed_at >= exported.report.observed_at
                    and not covers(other)
                    for other in same
                ):
                    failed("conflicting_native_execution_prefix", row, report=True)

            known_sources = {
                reference.record_id for _, exported in verified for _, reference in exported.sources
            }
            known_sources.update(row["evidence"] for row, _ in verified)
            for row in retained:
                try:
                    if row["id"].startswith(tuple(venue + "-http:" for venue in _NATIVE)):
                        if any(
                            identity not in authorities
                            for identity in _raw_executions(archive, row)
                        ):
                            failed("unbound_native_execution", row)
                        _raw_orders(archive, row, verified)
                    elif (
                        row["id"].startswith(tuple(venue + "-native-trade:" for venue in _NATIVE))
                        or row["id"].startswith("alpaca-activity:")
                        and decode(row["payload"]).get("activity_type") == "FILL"
                    ):
                        if row["id"] not in known_sources:
                            failed("unbound_retained_execution", row)
                except _INVALID:
                    failed("invalid_retained_execution_source", row)
            journal_fills = {}
            for row in tables["journal"]:
                if row["kind"] != "fill":
                    continue
                try:
                    value = Fill.model_validate_json(row["payload"])
                    if row["id"] != "fill:" + value.fill_id or value.account_id != account_id:
                        raise LedgerError("canonical journal execution identity differs")
                    journal_fills[value.fill_id] = value
                except _INVALID:
                    failed("invalid_execution_journal")
            if journal_fills != fills:
                failed("execution_journal_fill_mismatch")
            for identity, fill in {**journal_fills, **fills}.items():
                if identity not in authorities or authorities[identity].fill != fill:
                    failed("unproved_booked_execution")
            if any(identity not in fills for identity in authorities):
                failed("unapplied_native_execution")
            if any(
                row["state"]
                in ("submitting", "ambiguous", "reconciliation_required", "cancel_pending")
                for row in tables["venue_outbox"]
            ):
                failed("unresolved_native_outbox")

        violations = {}
        for authority in authorities.values():
            try:
                value = _violation(archive, authority)
                if value is not None:
                    violations[value.case_id] = value
            except _INVALID:
                failed("invalid_original_execution_admission")
        # Metadata is an immutable duplicate of proved facts, never the discovery index.
        for row in tables["events"]:
            try:
                if row["id"].startswith("execution-violation:"):
                    value = AdmissionViolation.model_validate_json(row["payload"])
                    if (
                        row["id"] != "execution-violation:" + value.case_id
                        or violations.get(value.case_id) != value
                    ):
                        raise LedgerError(
                            "retained violation differs from independently proved fact"
                        )
                elif row["id"].startswith("execution-discrepancy:"):
                    value = ExecutionDiscrepancy.model_validate_json(row["payload"])
                    violation = violations[value.case_id]
                    authority = authorities[value.key.venue + ":" + value.key.native_execution_id]
                    if (
                        row["id"] != "execution-discrepancy:" + value.case_id
                        or value.key != authority.key
                        or value.authority_hash != authority.content_hash()
                        or value.original_report != authority.original_report
                        or value.first_observed_at != authority.first_observed_at
                        or value.reason_codes != violation.reason_codes
                        or value.first_detected_at > archive.as_of
                        or value.first_detected_at < authority.first_complete_at
                        or value.account_id != account_id
                    ):
                        raise LedgerError(
                            "retained discrepancy differs from independently proved fact"
                        )
                elif row["id"].startswith("execution-authority:"):
                    from src.execution.discrepancies import NativeExecutionAuthority

                    value = NativeExecutionAuthority.model_validate_json(row["payload"])
                    if (
                        row["id"] != "execution-authority:" + value.content_hash()
                        or authorities.get(value.fill.fill_id) != value
                    ):
                        raise LedgerError(
                            "retained authority differs from independently proved sources"
                        )
            except _INVALID:
                failed("invalid_retained_discrepancy_metadata", row)
        bindings = tuple(
            ViolationBinding(
                case_id=value.case_id,
                authority_hash=value.authority_hash,
                violation_hash=value.content_hash(),
            )
            for _, value in sorted(violations.items())
        )
        audit = ExecutionComplianceAudit(
            account_id=account_id,
            authority_prefix_hash=prefix,
            lineage=lineage,
            status="FAIL"
            if blockers or bindings or unresolved or lineage == "unverified"
            else "PASS",
            violations=bindings,
            unresolved_sources=tuple(value for _, value in sorted(unresolved.items())),
            blocking_codes=tuple(sorted(blockers)),
        )
        return ComplianceFacts(
            audit,
            tuple(authorities[key] for key in sorted(authorities)),
            tuple(violations[key] for key in sorted(violations)),
        )


def execution_compliance(ledger, account_id):
    return collect(ledger, account_id).audit


def detect_discrepancies(ledger, account_id, *, as_of):
    """Persist proved metadata only; detection time is measured, never caller supplied.

    The caller cutoff must include the complete known source prefix. Failed proof
    remains visible in execution_compliance and cannot be converted into a case
    using a caller-provided Fill, authority, or successful proof flag.
    """
    as_of = _utc(as_of)
    now = datetime.now(timezone.utc)
    if as_of > now:
        raise LedgerError("discrepancy cutoff cannot be in the future")
    with ledger.atomic():
        facts = collect(ledger, account_id)
        by_hash = {authority.content_hash(): authority for authority in facts.authorities}
        if any(authority.first_complete_at > as_of for authority in facts.authorities):
            raise LedgerError("discrepancy cutoff omits an already known complete execution")
        if "invalid_retained_discrepancy_metadata" in facts.audit.blocking_codes:
            raise LedgerError(
                "retained discrepancy metadata conflicts with actual source authority"
            )
        detected = []
        for violation in facts.violations:
            authority = by_hash[violation.authority_hash]
            identity = "execution-discrepancy:" + violation.case_id
            existing = ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?", (account_id, identity)
            ).fetchone()
            if existing:
                case = ExecutionDiscrepancy.model_validate_json(existing[0])
            else:
                case = ExecutionDiscrepancy(
                    case_id=violation.case_id,
                    account_id=account_id,
                    key=authority.key,
                    authority_hash=authority.content_hash(),
                    original_report=authority.original_report,
                    first_detected_at=datetime.now(timezone.utc),
                    first_observed_at=authority.first_observed_at,
                    reason_codes=violation.reason_codes,
                )
            ledger.record_event(
                account_id,
                "execution-authority:" + authority.content_hash(),
                authority.model_dump_json(),
            )
            ledger.record_event(
                account_id, "execution-violation:" + violation.case_id, violation.model_dump_json()
            )
            ledger.record_event(account_id, identity, case.model_dump_json())
            detected.append(case)
        return tuple(detected)
