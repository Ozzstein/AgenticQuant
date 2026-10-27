"""Explicit compatibility entry to native and registered custom research.

The former random IC/Sharpe simulators have been removed. Native discoveries
remain quarantined; only registered ledger evaluations may support activation.
"""
from __future__ import annotations

from pathlib import Path

from src.data.evidence.store import EvidenceStore
from src.research.native import NativeRequest, NativeSupervisor
from src.research.registry import ResearchRegistry
from src.research.runner import ResearchRunner
from src.utils.config_loader import get_full_config


class RDAgentRunner(ResearchRunner):
    def __init__(self, config=None, *, registry_path=None, evidence_path=None):
        config=config or get_full_config()
        root=Path(config.output_dir)
        super().__init__(ResearchRegistry(registry_path or root/'research.sqlite3'),
                         EvidenceStore(evidence_path or root/'evidence.sqlite3'),config)

    def native(self, request: NativeRequest, *, gateway_token_path: str | Path, output_dir: str | Path):
        return NativeSupervisor(output_dir).run(request,gateway_token_path=gateway_token_path,
                                                registry=self.registry)

    def run_factor_search(self, *, trial_ids: list[str], output_dir: str | Path):
        return self._registered_search(trial_ids,output_dir,{'factor','semantic_question'})

    def run_model_search(self, *, trial_ids: list[str], output_dir: str | Path):
        return self._registered_search(trial_ids,output_dir,{'model'})

    def _registered_search(self, trial_ids, output_dir, kinds):
        if not trial_ids or len(set(trial_ids))!=len(trial_ids):
            raise ValueError('explicit unique preregistered trial identities required')
        for key in trial_ids:
            trial=self.registry.trial(key)
            if self.registry.candidate(trial.candidate_id).kind not in kinds:
                raise ValueError('candidate kind does not match research workflow')
        return [self.run(key,output_dir=Path(output_dir)/str(index)) for index,key in enumerate(trial_ids)]

    def mine_factors(self, *, trial_ids: list[str], output_dir: str | Path):
        return self.run_factor_search(trial_ids=trial_ids,output_dir=output_dir)

    def optimize_model(self, *, trial_ids: list[str], output_dir: str | Path):
        return self.run_model_search(trial_ids=trial_ids,output_dir=output_dir)

    def library_status(self):
        return self.registry.status()
