"""Source-grounded, budgeted research proposal entry points.

These methods return explicit typed outcomes and only quarantine exact candidates.
They never return template/random candidates on missing credentials or failures.
"""

from __future__ import annotations

from src.research.copilot import CopilotLimits, ProposalOutcome, ResearchCopilot


class FactorProposer:
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

    def propose_factors(
        self,
        n,
        *,
        context,
        sources,
        description,
        tested=(),
        parent_ids=(),
        model_name="Linear",
        allowed_fields=("open", "high", "low", "close", "volume"),
        prior_memo_hash=None,
    ) -> ProposalOutcome:
        return self.copilot.propose(
            "factor",
            description,
            context=context,
            sources=sources,
            n=n,
            tested=tested,
            parent_ids=parent_ids,
            selected_model=model_name,
            allowed_fields=allowed_fields,
            prior_memo_hash=prior_memo_hash,
        )

    def propose_model_config(
        self,
        n,
        *,
        context,
        sources,
        description,
        tested=(),
        parent_ids=(),
        allowed_fields=("open", "high", "low", "close", "volume"),
        prior_memo_hash=None,
    ) -> ProposalOutcome:
        return self.copilot.propose(
            "model",
            description,
            context=context,
            sources=sources,
            n=n,
            tested=tested,
            parent_ids=parent_ids,
            allowed_fields=allowed_fields,
            prior_memo_hash=prior_memo_hash,
        )

    def propose_strategy(
        self,
        description,
        *,
        context,
        sources,
        parent_ids=(),
        allowed_fields=("open", "high", "low", "close", "volume"),
        prior_memo_hash=None,
    ) -> ProposalOutcome:
        return self.copilot.propose(
            "strategy",
            description,
            context=context,
            sources=sources,
            parent_ids=parent_ids,
            allowed_fields=allowed_fields,
            prior_memo_hash=prior_memo_hash,
        )

    def propose_regime_change(
        self, description, *, context, sources, parent_ids=(), prior_memo_hash=None
    ) -> ProposalOutcome:
        return self.copilot.propose(
            "regime",
            description,
            context=context,
            sources=sources,
            parent_ids=parent_ids,
            prior_memo_hash=prior_memo_hash,
        )

    def propose_text_feature(
        self, description, *, context, sources, tested=(), prior_memo_hash=None
    ) -> ProposalOutcome:
        return self.copilot.propose(
            "semantic_question",
            description,
            context=context,
            sources=sources,
            tested=tested,
            prior_memo_hash=prior_memo_hash,
        )

    def implement_paper(
        self, evidence_ids, *, context, kind, description, **options
    ) -> ProposalOutcome:
        """Propose from complete already-ingested paper documents, never URL keywords."""
        return self.copilot.propose_from_papers(
            evidence_ids, context=context, kind=kind, description=description, **options
        )
