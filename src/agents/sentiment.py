"""Source-grounded sentiment specialist; synthesis remains generative."""

from src.agents.desk import structured_report

_SYSTEM_PROMPT = "Synthesize source-linked news, analyst revisions, surprises and market sentiment. Separate factual developments from reactions and contrarian hypotheses. Keep mixed, unknown and genuinely neutral evidence distinct."


def sentiment_node(state, llm=None):
    return structured_report(
        state, "sentiment", _SYSTEM_PROMPT, llm=llm, kinds=("news", "fundamental")
    )
