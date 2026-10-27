"""Explicit source identities/units for the four required crypto context fields."""

from src.data.features.contextual import NumericEvidenceBinding
from src.domain.common import Instrument


def crypto_context_bindings(
    target: Instrument,
    *,
    perpetual: Instrument,
    benchmark: Instrument,
    open_interest_units: str,
    funding_max_age_seconds: int = 86400,
    oi_max_age_seconds: int = 7200,
    dominance_max_age_seconds: int = 86400,
    benchmark_max_age_seconds: int = 172800,
) -> tuple[NumericEvidenceBinding, ...]:
    """Use actual dated perpetual/global evidence and existing raw BTC spot bars.

    Source instruments remain unchanged and confer no execution eligibility.
    Benchmark quote currency must match; no stablecoin/USD parity is assumed.
    """
    if (
        target.asset_class != "crypto"
        or ":" in target.symbol
        or target.symbol.count("/") != 1
        or benchmark.asset_class != "crypto"
        or ":" in benchmark.symbol
        or benchmark.symbol.count("/") != 1
        or perpetual.asset_class != "crypto"
        or perpetual.symbol.count(":") != 1
        or perpetual.symbol.count("/") != 1
        or not all((target.venue, perpetual.venue, benchmark.venue))
    ):
        raise ValueError(
            "explicit spot target/BTC benchmark and separate perpetual source required"
        )
    base, quote = target.symbol.split("/")
    perp_base, rest = perpetual.symbol.split("/")
    perp_quote, settle = rest.split(":")
    if (
        not all((base, quote, perp_base, perp_quote, settle))
        or perp_base != base
        or benchmark.symbol != "BTC/" + quote
        or quote != target.quote_currency
        or perp_quote != perpetual.quote_currency
        or benchmark.quote_currency != quote
        or target.instrument_id == perpetual.instrument_id
        or benchmark.instrument_id == perpetual.instrument_id
        or open_interest_units not in ("base:" + base, "contracts")
    ):
        raise ValueError(
            "source base/quote/settlement/unit identities differ; explicit FX requires a separate source"
        )
    common = dict(
        target_instrument_id=target.instrument_id, max_age_seconds=funding_max_age_seconds
    )
    source_filters = (
        ("market_type", "perpetual"),
        ("base_currency", perp_base),
        ("quote_currency", perp_quote),
        ("settle_currency", settle),
    )
    funding = NumericEvidenceBinding(
        **common,
        name="funding_rate",
        source_instrument_id=perpetual.instrument_id,
        kind="funding",
        value_path=("value",),
        observation_date_path=("observed_at",),
        source_market="perpetual",
        filters=source_filters + (("units", "fraction-per-funding-event"),),
    )
    oi = NumericEvidenceBinding(
        **{**common, "max_age_seconds": oi_max_age_seconds},
        name="open_interest",
        source_instrument_id=perpetual.instrument_id,
        kind="open_interest",
        value_path=("value",),
        observation_date_path=("observed_at",),
        source_market="perpetual",
        filters=source_filters + (("units", open_interest_units),),
    )
    dominance = NumericEvidenceBinding(
        **{**common, "max_age_seconds": dominance_max_age_seconds},
        name="btc_dominance",
        source_instrument_id=None,
        kind="btc_dominance",
        value_path=("value",),
        observation_date_path=("observed_at",),
        source_market="global",
        filters=(
            ("market_type", "global"),
            ("units", "fraction-of-global-market-cap"),
            ("numerator_scope", "bitcoin"),
            ("denominator_scope", "coingecko-tracked-global-market"),
        ),
    )
    reference = NumericEvidenceBinding(
        **{**common, "max_age_seconds": benchmark_max_age_seconds},
        name="benchmark_close",
        source_instrument_id=benchmark.instrument_id,
        kind="bar",
        value_path=("close",),
        observation_date_path=("closed_at",),
        source_market="instrument",
        filters=(("market_type", "spot"), ("price_basis", "raw")),
    )
    return funding, oi, dominance, reference
