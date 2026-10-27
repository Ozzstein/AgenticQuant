"""Re-execute registered recipes and bind paired regime experiments to real books.

An accounting-valid target trace alone does not prove that a proposed strategy
produced those targets. Verification runs the registered implementation again
from frozen sources, fresh fits and actual holdings, then compares its output.
No supplied pickle or generated code is loaded to perform this verification.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from src.core.backtester import ReplayArtifact, ReplayResult, load_replay_artifact
from src.core.validation import ValidationSuite
from src.research.registry import RegistryError, canonical, content_hash


def _verified_opportunities(result: ReplayResult) -> bytes:
    reference = result.manifest['provenance']['opportunities_artifact']
    payload = Path(reference['path']).read_bytes()
    if content_hash(payload) != reference['sha256']:
        raise RegistryError('opportunity artifact bytes differ from the replay manifest')
    return payload


def require_reproduced(actual: ReplayResult, reproduced: ReplayResult) -> None:
    """Ignore only filesystem placement after verifying exact artifact contents."""
    if _verified_opportunities(actual) != _verified_opportunities(reproduced):
        raise RegistryError('independently reproduced opportunities differ')
    values = []
    for result in (actual, reproduced):
        value = result.model_dump(mode='json')
        value['manifest']['provenance']['opportunities_artifact']['path'] = '<verified-content>'
        values.append(value)
    if values[0] != values[1]:
        raise RegistryError('registered recipe does not reproduce the actual decisions, fills, NAV or folds')


def verify_execution(registry, store, trial_id: str, result: ReplayResult, *,
                     artifact_dir: str | Path, baseline: ReplayArtifact | None = None) -> dict:
    """Return archived verification evidence; never record or promote an outcome."""
    from src.research.regime_experiments import execute_registered_regime
    from src.research.strategy_experiments import execute_registered_strategy

    trial = registry.trial(trial_id)
    candidate = registry.candidate(trial.candidate_id)
    if candidate.kind not in ('strategy', 'regime'):
        raise RegistryError('recipe reproduction requires a registered strategy or regime')
    if (candidate.kind == 'regime') != (baseline is not None):
        raise RegistryError('regime evaluation requires exactly its preregistered paired baseline')
    execute = execute_registered_regime if candidate.kind == 'regime' else execute_registered_strategy
    with tempfile.TemporaryDirectory(prefix='quantagent-execution-proof-') as temporary:
        work = Path(temporary)
        expected = execute(registry, trial_id, store=store, ledger_path=work/'ledger.sqlite3',
                           artifact_dir=work/'artifacts')
        if expected.inputs_hash != trial.inputs_hash or expected.candidate_hash != candidate.content_hash():
            raise RegistryError('reproduced execution has a different admission identity')
        require_reproduced(result, expected.result)
        original = Path(artifact_dir)
        artifacts = {}
        for name, payload in expected.artifacts:
            if (original/name).read_bytes() != payload:
                raise RegistryError('independently fitted model or opportunity bytes differ')
            artifacts[name] = registry.archive(payload)
        proof = {'trial_id':trial_id, 'inputs_hash':trial.inputs_hash,
                 'candidate_hash':candidate.content_hash(), 'artifacts':artifacts,
                 'result_hash':registry.archive(canonical(result.model_dump(mode='json'))),
                 'implementation':'registered-recipe-reproduction-1'}
        paired = None
        if baseline is not None:
            comparisons = dict(expected.comparisons)
            if set(comparisons) != {'baseline'}:
                raise RegistryError('registered regime recipe must reproduce one actual baseline')
            actual_baseline = load_replay_artifact(baseline,verify_accounting=False)
            accounting = ValidationSuite().validate(actual_baseline,ledger_path=baseline.ledger_path,
                                                    _accounting_only=True)
            # Development inspection may reproduce a truncated final fold. Its
            # incompleteness remains an explicit certification failure below;
            # every economic verification must still pass without exceptions.
            failures = [c for c in accounting.checks if c.name != 'complete_run' and c.verdict.value != 'PASS']
            if failures or actual_baseline.status == 'failed':
                raise RegistryError('paired baseline failed independent accounting verification')
            require_reproduced(actual_baseline, comparisons['baseline'].result)
            if result.timestamps != actual_baseline.timestamps:
                raise RegistryError('paired candidate and baseline valuation intervals differ')
            if result.manifest['context'] != actual_baseline.manifest['context']:
                raise RegistryError('paired baseline belongs to a different run/context')
            left = [f.model_version for f in result.walk_forward_results]
            right = [f.model_version for f in actual_baseline.walk_forward_results]
            if not left or left != right:
                raise RegistryError('paired regime experiment must use identical actual numeric model bytes')
            excess = np.asarray(result.daily_returns)-np.asarray(actual_baseline.daily_returns)
            paired = {'trial_id':trial_id, 'inputs_hash':trial.inputs_hash,
                      'candidate_result_hash':proof['result_hash'],
                      'baseline_result_hash':registry.archive(canonical(actual_baseline.model_dump(mode='json'))),
                      'baseline_ledger_hash':registry.archive(Path(baseline.ledger_path).read_bytes()),
                      'baseline_reference':baseline.model_dump(mode='json'),
                      'timestamps':result.timestamps, 'excess_returns':excess.tolist(),
                      'mean_excess_return':float(excess.mean()) if len(excess) else None,
                      'comparison':'same-admission-fixed-strategy-paired-regime-1'}
            paired['candidate_status'],paired['baseline_status'] = result.status,actual_baseline.status
            # Independently reconstructed baseline model bytes are archived too;
            # the original baseline's full economic trace was read above.
            paired['artifacts'] = {name:registry.archive(payload)
                                   for name,payload in comparisons['baseline'].artifacts}
        return {'execution_hash':registry.archive(canonical(proof)), 'paired':paired}


def verified_paired_returns(registry, key: str) -> list[float]:
    """Restricted content check used by registry admission, not caller metrics."""
    proof = json.loads(registry.blob(key))
    candidate = ReplayResult.model_validate_json(registry.blob(proof['candidate_result_hash']))
    baseline = ReplayResult.model_validate_json(registry.blob(proof['baseline_result_hash']))
    registry.blob(proof['baseline_ledger_hash'])
    if candidate.timestamps != baseline.timestamps or proof['timestamps'] != candidate.timestamps:
        raise RegistryError('paired return calendar differs')
    values = (np.asarray(candidate.daily_returns)-np.asarray(baseline.daily_returns)).tolist()
    if values != proof['excess_returns'] or not values or float(np.mean(values)) != proof['mean_excess_return']:
        raise RegistryError('paired return contents differ')
    return values
