"""Source-grounded transcript_nlp specialist; synthesis remains generative."""

from src.agents.desk import structured_report, unavailable

_SYSTEM_PROMPT = "Synthesize management commentary, guidance, hedging, Q&A responsiveness and specific operating risks. Distinguish prepared remarks from Q&A, quote exactly, retain contrary passages. Jev labels never substitute for your narrative reasoning."


def transcript_nlp_node(state, llm=None):
    if state.get("instrument") and state["instrument"].asset_class == "crypto":
        return unavailable(
            "transcript_nlp",
            "Corporate earnings transcripts do not apply to spot crypto",
            applicable=False,
        )
    return structured_report(
        state, "transcript_nlp", _SYSTEM_PROMPT, llm=llm, kinds=("transcript",)
    )
