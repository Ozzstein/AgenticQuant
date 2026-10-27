"""Operator-selected source authorities, kept outside candidate experiment inputs."""
from pathlib import Path

from pydantic import Field

from src.domain.common import NonEmptyString, VersionedContract


class AgentAuthorityPaths(VersionedContract):
    history_store: Path
    decision_store: Path
    budget_store: Path
    portfolio_ledgers: dict[NonEmptyString, Path] = Field(min_length=1)

    def open(self):
        from src.agents.history import DeskHistoryStore
        from src.decision.store import DecisionStore
        from src.research.strategy_experiments import AgentHistoryAuthorities
        # Never create empty authorities to make a missing history look usable.
        for path in (self.history_store,self.decision_store,self.budget_store,*self.portfolio_ledgers.values()):
            if not path.is_file():
                raise ValueError('configured agent history authority file is absent')
        return AgentHistoryAuthorities(DeskHistoryStore(self.history_store),DecisionStore(self.decision_store),
                                       self.budget_store,self.portfolio_ledgers)
