"""Source-grounded earnings specialist; synthesis remains generative."""

from src.agents.desk import structured_report, unavailable

_SYSTEM_PROMPT = "Evaluate documented earnings calendar, estimates, reported surprises and guidance. Date and numeric comparisons must already be supplied by deterministic data tools. Do not invent a beat probability."


def earnings_node(state, llm=None):
    if state.get("instrument") and state["instrument"].asset_class == "crypto":
        return unavailable(
            "earnings",
            "Corporate earnings transcripts do not apply to spot crypto",
            applicable=False,
        )
    return structured_report(
        state, "earnings", _SYSTEM_PROMPT, llm=llm, kinds=("fundamental", "transcript", "news")
    )
