"""The desk consumes the quantitative HMM; it does not override its regime."""

from src.agents.contracts import HMMContext
from src.agents.desk import eligible_sources

VALID_REGIMES = frozenset(("risk_on", "neutral", "risk_off", "crisis"))


def market_context_node(state, llm=None):
    sources = eligible_sources(state, ("macro", "technical"))
    return {"market_context": "\n".join(s.text for s in sources)}


def macro_regime_node(state, llm=None):
    hmm = state.get("hmm_context")
    context = state.get("context")
    if (
        not isinstance(hmm, HMMContext)
        or context is None
        or hmm.status != "available"
        or hmm.as_of != context.as_of
    ):
        return {
            "macro_regime": "unavailable",
            "macro_confidence": None,
            "new_exposure_blocked": True,
        }
    # This is the HMM state probability, in its native 0..1 scale.
    return {"macro_regime": hmm.regime, "macro_confidence": hmm.state_probability}
