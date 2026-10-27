"""Source-grounded fundamental specialist; synthesis remains generative."""

from src.agents.desk import structured_report

_SYSTEM_PROMPT = "Evaluate business quality, valuation, financial statements, leverage, insider/institutional activity and filings. For crypto evaluate tokenomics, protocol revenue/security, concentration and liquidity. Distinguish missing valuation evidence from fair value."


def fundamental_node(state, llm=None):
    return structured_report(
        state, "fundamental", _SYSTEM_PROMPT, llm=llm, kinds=("fundamental", "news")
    )
