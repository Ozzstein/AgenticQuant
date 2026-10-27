"""Source-grounded technical specialist; synthesis remains generative."""

from src.agents.desk import structured_report

_SYSTEM_PROMPT = "Evaluate observed price/history, indicators, support/resistance, relative strength, volume and liquidity. Quantitative arithmetic comes from source data; never infer prices or technical levels from sentiment."


def technical_node(state, llm=None):
    return structured_report(state, "technical", _SYSTEM_PROMPT, llm=llm, kinds=("technical",))
