"""Verified read-only registry projections; no artifact or holdout payload export."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.research.registry import (
    Activation,
    MissingDevelopmentEvaluation,
    RegistryError,
    ResearchRegistry,
)

_RESULT_STATES = frozenset({
    'discovery_complete', 'timeout', 'output_limit', 'interrupted', 'failed',
    'cleanup_failed', 'audit_unavailable', 'artifacts_unavailable',
})
_EVENT_FIELDS = (
    'run_id', 'sequence', 'observed_at', 'observation', 'action', 'reason',
    'cost_state', 'exposure_retained', 'activation', 'internal_search_trials',
)


def _tables(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _native_view(registry, exposure_id, tables):
    exposure = registry.discovery_exposure(exposure_id)
    if not exposure_id.startswith('native:'):
        raise RegistryError('invalid native exposure identity')
    run_id = exposure_id[len('native:'):]
    launch = registry.native_launch(run_id) if 'native_launches' in tables else None
    binding = registry.native_container(run_id) if 'native_containers' in tables else None
    events = registry.native_recoveries(run_id) if 'native_recoveries' in tables else ()
    with registry._connection() as db:
        completed = ('discovery_results' in tables and db.execute(
            'SELECT 1 FROM discovery_results WHERE id=?', (exposure_id,)).fetchone() is not None)
    result_state = None
    if completed:
        result = registry.native_result(run_id)
        if (result.get('status') not in _RESULT_STATES
                or result.get('activation') != 'quarantined'
                or result.get('internal_search_trials') is not None):
            raise RegistryError('invalid native result state')
        result_state = result['status']
    last = {name: events[-1][name] for name in _EVENT_FIELDS} if events else None
    return {
        'run_id': run_id, 'status': 'verified',
        'admitted_at': exposure.admitted_at.isoformat(),
        'exposure_retained': True, 'activation': 'quarantined',
        'internal_search_trials': None,
        'launch_recorded': launch is not None, 'container_bound': binding is not None,
        'result_recorded': bool(completed), 'result_status': result_state,
        'latest_recovery': last,
    }


def research_snapshot(registry: ResearchRegistry, *, limit: int = 200) -> dict:
    """Read one database snapshot; current eligibility is separate from history.

    Recovery observations describe the last recorded inspection, not a fresh
    Docker observation. Absent launch/result records never imply success.
    """
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('monitoring limit must be an integer from 1 to 1000')
    with registry.read_snapshot(), registry._connection() as db:
        tables = _tables(db)
        candidates = []
        for (candidate_id,) in db.execute('SELECT id FROM candidates ORDER BY rowid DESC LIMIT ?', (limit,)):
            candidate = registry.candidate(candidate_id)
            for key in (candidate.payload_hash, candidate.code_hash,
                        candidate.environment_hash, *candidate.source_evidence_hashes):
                registry.blob(key)
            candidates.append(candidate.model_dump(mode='json'))

        activations = []
        for slot, generation, payload in db.execute(
                'SELECT slot,generation,payload FROM activations ORDER BY rowid DESC LIMIT ?', (limit,)):
            value = Activation.model_validate_json(payload)
            candidate = registry.candidate(value.candidate_id)
            if ((value.slot, value.generation) != (slot, generation)
                    or value.candidate_hash != candidate.content_hash()
                    or value.payload_hash != candidate.payload_hash):
                raise RegistryError('activation history identity differs')
            activations.append(value.model_dump(mode='json'))

        active, blocked = [], []
        now = datetime.now(timezone.utc)
        for slot, generation in db.execute('SELECT slot,generation FROM active ORDER BY slot'):
            row = db.execute('SELECT payload FROM activations WHERE slot=? AND generation=?',
                             (slot, generation)).fetchone()
            if row is None:
                raise RegistryError('active generation has no receipt')
            value = Activation.model_validate_json(row[0])
            if (value.slot, value.generation) != (slot, generation):
                raise RegistryError('active generation identity differs')
            try:
                actual = registry.resolve(slot, payload_hash=value.payload_hash, as_of=now)
                if actual != value:
                    raise RegistryError('active generation is not currently eligible')
            except RegistryError:
                blocked.append({'slot': slot, 'generation': generation,
                                'candidate_id': value.candidate_id,
                                'reason': 'current_eligibility_unverified'})
            else:
                active.append(value.model_dump(mode='json'))

        # Check the immutable trial admission BEFORE selecting an outcome.
        # Holdout status/counts are visible; its measurements are never returned.
        development = []
        for trial_id, payload in db.execute('SELECT id,payload FROM trials ORDER BY rowid DESC'):
            trial = registry.trial(trial_id)
            if json.loads(payload)['purpose'] != trial.purpose:
                raise RegistryError('trial purpose differs')
            if trial.purpose != 'development':
                continue
            try:
                evaluation = registry.development_evaluation(trial_id)
            except MissingDevelopmentEvaluation:
                continue
            development.append(evaluation.model_dump(mode='json'))
            if len(development) >= limit:
                break
        holdouts = [
            {'holdout_id': key, 'consumed': db.execute(
                'SELECT 1 FROM trials WHERE holdout=?', (key,)).fetchone() is not None}
            for (key,) in db.execute('SELECT id FROM holdouts ORDER BY rowid DESC LIMIT ?', (limit,))
        ]
        native = []
        if 'discovery_exposures' in tables:
            for (key,) in db.execute('SELECT id FROM discovery_exposures ORDER BY rowid DESC LIMIT ?', (limit,)):
                try:
                    native.append(_native_view(registry, key, tables))
                except (ValueError, KeyError, TypeError, OSError):
                    # Do not export exception text, paths or unverified blobs.
                    native.append({'run_id': key.removeprefix('native:'),
                                   'status': 'unverifiable', 'reason': 'native_authority_invalid',
                                   'activation': 'quarantined', 'internal_search_trials': None})
    return {'status': 'available', 'candidates': candidates, 'activations': activations,
            'active': active, 'blocked_activations': blocked,
            'development_outcomes': development, 'holdouts': holdouts,
            'native_runs': native, 'live_capital_authorized': False}
