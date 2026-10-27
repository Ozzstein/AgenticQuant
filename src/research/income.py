"""Historical income certification uses native source authority, never shape alone."""

from __future__ import annotations

from src.operations.income import _parsed_capture, require_coverage_union
from src.research.evidence import BoundArtifact
from src.research.registry import RegistryError

RESEARCH_INCOME_KINDS = ("cash_dividends", "dividend_distributions", "action_coverage")


def verify_research_income_provenance(registry, store, inputs, references):
    """Bind the full economic observation interval, independent of fitted splits.

    The current native parser proves gaps only. Retaining those observations is
    useful, but cannot certify empty income or complete effective-event coverage.
    Operational account postings never size a counterfactual historical book.
    """
    if inputs.income_policy != "source-confirmed-replay-distribution-1":
        raise RegistryError(
            "historical income policy/coverage missing; diagnostic replay is not certifiable"
        )
    if any(action.kind == "dividend" for action in inputs.actions):
        raise RegistryError("legacy immediate dividends cannot certify ordinary ex/pay income")
    checked, parsed = {}, {}
    for kind in RESEARCH_INCOME_KINDS:
        key = references.get(kind + "_id")
        if key is None:
            raise RegistryError("missing historical income source artifact: " + kind)
        artifact = BoundArtifact.model_validate_json(registry.blob(key))
        expected = [value.model_dump(mode="json") for value in getattr(inputs, kind)]
        if (
            artifact.kind != kind
            or artifact.inputs_hash != inputs.content_hash()
            or artifact.value() != expected
        ):
            raise RegistryError("historical income artifact differs from registered input: " + kind)
        actual = []
        for binding in artifact.sources:
            if binding.pointer != "/" + kind or binding.available_at > inputs.context.as_of:
                raise RegistryError(
                    "historical income source needs exact canonical array and eligible cutoff"
                )
            if binding.evidence_id not in parsed:
                parsed[binding.evidence_id] = _parsed_capture(
                    store, binding.evidence_id, as_of=inputs.context.as_of
                )
            source, contents = parsed[binding.evidence_id]
            if source.record.available_at != binding.available_at:
                raise RegistryError(
                    "historical income source eligibility differs from original capture"
                )
            actual.extend(contents[kind])
        if actual != expected:
            raise RegistryError("native income source does not substantiate historical economics")
        checked[kind + "_id"] = key
    instruments = {bar.instrument.instrument_id: bar.instrument for bar in inputs.bars}
    for instrument in instruments.values():
        if instrument.asset_class == "crypto":
            raise RegistryError(
                "historical crypto income/protocol coverage needs a supported native authority"
            )
        for event_type in ("cash_dividend_terms", "cash_dividend_distributions"):
            require_coverage_union(
                inputs.action_coverage,
                instrument_id=instrument.instrument_id,
                account_id=inputs.account.account_id,
                started_at=inputs.account.opened_at,
                ended_at=max(bar.closed_at for bar in inputs.bars),
                as_of=inputs.context.as_of,
                scope="effective-event-stream",
                event_type=event_type,
            )
    return checked
