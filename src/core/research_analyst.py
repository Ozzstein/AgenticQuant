"""Budgeted research memos from exact sources and recorded development outcomes."""

from __future__ import annotations

from src.research.copilot import CopilotLimits, MemoDocument, MemoOutcome, ResearchCopilot


class ResearchAnalyst:
    def __init__(
        self,
        config,
        *,
        registry,
        evidence_store,
        code_hash,
        environment_hash,
        llm_factory=None,
        limits: CopilotLimits | None = None,
    ):
        self.copilot = ResearchCopilot(
            config,
            registry=registry,
            evidence_store=evidence_store,
            code_hash=code_hash,
            environment_hash=environment_hash,
            llm_factory=llm_factory,
            limits=limits,
        )

    def write_memo(
        self, *, context, sources=(), development_trial_ids=(), prior_memo_hash=None
    ) -> MemoOutcome:
        """No holdout reader, caller-reported performance, or placeholder success."""
        return self.copilot.write_memo(
            context=context,
            sources=sources,
            development_trial_ids=development_trial_ids,
            prior_memo_hash=prior_memo_hash,
        )

    def load_memo(self, memo_hash: str, *, context) -> MemoDocument:
        """Load an immutable, as-of-eligible memo; raw mutable KB strings are unsupported."""
        return self.copilot.load_memo(memo_hash, context=context)
