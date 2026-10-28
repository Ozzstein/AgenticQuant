"""Supervise pinned native RD-Agent jobs in bounded, isolated containers.

Native outputs are quarantined discovery artifacts. They never authorize a
strategy/model or replace the application's ledger replay and holdout evaluator.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import Field, model_validator

from src.data.qlib_us import USQlibScenario, verify_bundle
from src.domain.common import NonEmptyString, Sha256, VersionedContract
from src.research.native_runtime.overlay import UPSTREAM_REVISION

COMMANDS = ('fin_factor', 'fin_model', 'fin_quant', 'fin_factor_report', 'general_model', 'ui', 'health_check')


def _usage_integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('invalid gateway usage count')
    return value


def _usage_cost(value):
    if not isinstance(value, str):
        raise ValueError('gateway costs require exact decimal strings')
    number = Decimal(value)
    if not number.is_finite() or number < 0:
        raise ValueError('invalid gateway usage cost')
    return number


def _usage_scope(scope):
    """Check public summary consistency, without recreating reservation pricing."""
    counts = ('requests','pending_requests','known_input_tokens','known_output_tokens',
              'known_total_tokens_lower_bound','usage_unknown_requests','unpriced_requests',
              'conservative_tokens','uncertain_tokens')
    costs = ('known_cost_lower_bound_usd','conservative_cost_usd','uncertain_cost_usd')
    extra = ('actual_tokens','actual_cost_usd','breached','breach_call_ids','breach_reasons',
             'statuses','observed_models','observed_model_hashes')
    if not isinstance(scope, dict) or set(scope) != set(counts+costs+extra):
        raise ValueError('closed v2 gateway usage scope required')
    for key in counts:
        _usage_integer(scope[key])
    money = {key:_usage_cost(scope[key]) for key in costs}
    n, unknown, unpriced = (scope[k] for k in ('requests','usage_unknown_requests','unpriced_requests'))
    if not 0 <= scope['pending_requests'] <= unknown <= unpriced <= n:
        raise ValueError('inconsistent gateway uncertainty counts')
    total = scope['known_input_tokens']+scope['known_output_tokens']
    if not total <= scope['known_total_tokens_lower_bound'] <= scope['conservative_tokens']:
        raise ValueError('gateway lower bound exceeds conservative tokens')
    if (money['known_cost_lower_bound_usd'] > money['conservative_cost_usd']
            or money['uncertain_cost_usd'] > money['conservative_cost_usd']
            or scope['uncertain_tokens'] > scope['conservative_tokens']):
        raise ValueError('gateway cost/uncertainty exceeds conservative charge')
    if unknown:
        if scope['actual_tokens'] is not None:
            raise ValueError('unknown usage cannot have exact tokens')
    elif (_usage_integer(scope['actual_tokens']) != total
          or total != scope['known_total_tokens_lower_bound']):
        raise ValueError('exact token total differs')
    if unpriced:
        if scope['actual_cost_usd'] is not None:
            raise ValueError('unpriced usage cannot have exact cost')
    elif (_usage_cost(scope['actual_cost_usd']) != money['known_cost_lower_bound_usd']
          or scope['uncertain_tokens'] or money['uncertain_cost_usd']):
        raise ValueError('exact cost or uncertainty differs')
    for key in ('breach_call_ids','breach_reasons','observed_models','observed_model_hashes'):
        values = scope[key]
        if (not isinstance(values, list) or any(not isinstance(v, str) or not v or len(v)>160 for v in values)
                or len(values) != len(set(values))):
            raise ValueError('invalid public usage identities')
    if (any(not re.fullmatch(r'[0-9a-f]{64}', v) for v in scope['observed_model_hashes'])
            or type(scope['breached']) is not bool
            or scope['breached'] != bool(scope['breach_call_ids'])
            or scope['breached'] != bool(scope['breach_reasons'])
            or len(scope['breach_call_ids']) > n):
        raise ValueError('gateway breach/model identity differs')
    if not n and (any(scope[k] for k in counts) or any(money.values())
                  or any(scope[k] for k in ('observed_models','observed_model_hashes'))):
        raise ValueError('empty gateway scope has nonempty usage')
    statuses = scope['statuses']
    if (not isinstance(statuses, dict)
            or any(not isinstance(k,str) or not re.fullmatch(r'[a-z_]{1,48}',k)
                   or not _usage_integer(v) for k,v in statuses.items())
            or sum(statuses.values()) != n or statuses.get('reserved',0) != scope['pending_requests']):
        raise ValueError('gateway status counts differ')


def _gateway_accounting(request, receipt, now):
    """Require the authenticated gateway's v2 scope; no legacy success downgrade."""
    value = receipt.get('accounting')
    if (not isinstance(value,dict)
            or set(value) != {'schema_version','policy','run_id','as_of','day_id','run','day'}
            or value['schema_version'] != '2.0' or value['policy'] != 'reservation-floor-2'
            or value['run_id'] != request.run_id
            or not isinstance(value['as_of'],str) or not isinstance(value['day_id'],str)
            or receipt.get('cost_semantics') != 'verified_usage_with_reservation_floor'):
        raise ValueError('verified v2 gateway accounting required')
    at = datetime.fromisoformat(value['as_of'].replace('Z','+00:00'))
    if (now.tzinfo is None or at.tzinfo is None or at > now
            or value['day_id'] != at.astimezone(timezone.utc).date().isoformat()
            or value['day_id'] != now.astimezone(timezone.utc).date().isoformat()):
        raise ValueError('gateway accounting clock/day differs')
    for name in ('run','day'):
        _usage_scope(value[name])
    run = value['run']
    expected = {
        'charged': {'requests':run['requests'],'tokens':run['conservative_tokens'],
                    'cost_usd':run['conservative_cost_usd']},
        'actual': {'requests_with_usage':run['requests']-run['unpriced_requests'],
                   'input_tokens':run['known_input_tokens'],'output_tokens':run['known_output_tokens'],
                   'cost_usd':run['actual_cost_usd']},
        'uncertain': {'requests':run['unpriced_requests'],'reserved_tokens':run['uncertain_tokens'],
                      'reserved_cost_usd':run['uncertain_cost_usd']},
    }
    if any(json.dumps(receipt.get(key),sort_keys=True,allow_nan=False)
           != json.dumps(item,sort_keys=True,allow_nan=False) for key,item in expected.items()):
        raise ValueError('gateway public counters differ from verified scope')
    if (receipt.get('observed_models') != run['observed_models']
            or receipt.get('statuses') != run['statuses']
            or type(receipt.get('blocked')) is not bool
            or receipt['blocked'] != (run['breached'] or value['day']['breached'])):
        raise ValueError('gateway public status differs from verified scope')
    # A copy keeps the caller's mutable response from changing an archived proof.
    return json.loads(json.dumps(value,allow_nan=False))


def _budget_audit_error(receipt):
    accounting = receipt['accounting']
    for name, limits_key in (('run','limits'),('day','day_limits')):
        scope, limits = accounting[name], receipt[limits_key]
        if (scope['breached'] or scope['requests'] > limits['requests']
                or scope['conservative_tokens'] > limits['tokens']
                or _usage_cost(scope['conservative_cost_usd']) > _usage_cost(limits['cost_usd'])):
            return 'budget_breached'
    if any(accounting[name][key] for name in ('run','day')
           for key in ('pending_requests','usage_unknown_requests','unpriced_requests')):
        return 'uncertain_usage_retained'
    return None


class NativeBudgetLimits(VersionedContract):
    requests: int = Field(gt=0)
    tokens: int = Field(gt=0)
    cost_usd: Decimal = Field(gt=0)


class NativeKnowledgeRef(VersionedContract):
    source_run: str = Field(pattern=r'^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$')
    source_result_hash: Sha256
    artifact: Literal['factor','model']
    artifact_hash: Sha256


class NativeRequest(VersionedContract):
    run_id: NonEmptyString
    command: Literal['fin_factor','fin_model','fin_quant','fin_factor_report','general_model','ui','health_check']
    scenario: USQlibScenario
    image: str = Field(pattern=r'^.+@sha256:[0-9a-f]{64}$')
    network: NonEmptyString
    gateway_url: NonEmptyString
    gateway_receipt_url: NonEmptyString
    gateway_config_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    chat_model: NonEmptyString
    embedding_model: NonEmptyString
    gateway_limits: NativeBudgetLimits
    gateway_day_limits: NativeBudgetLimits
    max_output_tokens: int = Field(default=4096, ge=1, le=32768)
    max_seconds: int = Field(default=3600, ge=1, le=86400)
    max_steps: int = Field(default=30, ge=1, le=10000)
    max_loops: int = Field(default=5, ge=1, le=1000)
    memory_mb: int = Field(default=8192, ge=512, le=65536)
    cpus: int = Field(default=2, ge=1, le=32)
    max_output_bytes: int = Field(default=1024*1024*1024, ge=1024)
    max_source_bytes: int = Field(default=128*1024*1024, ge=1, le=1024*1024*1024)
    max_source_files: int = Field(default=100, ge=1, le=1000)
    document: str | None = None
    ui_source_run: str | None = None
    ui_port: int = Field(default=19899, ge=1024, le=65535)
    knowledge: tuple[NativeKnowledgeRef,...] = ()

    @model_validator(mode='after')
    def explicit_bounds(self):
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}',self.run_id):
            raise ValueError('native run_id must be a bounded file/container identity')
        self.scenario.validate_splits()
        for endpoint in (self.gateway_url,self.gateway_receipt_url):
            parsed = urlparse(endpoint)
            if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('explicit credential-free HTTP gateway URL required')
        if (self.command in ('fin_factor_report','general_model')) != (self.document is not None):
            raise ValueError('paper/report commands require exactly one local source document/directory')
        if (self.command == 'ui') != (self.ui_source_run is not None):
            raise ValueError('UI requires an existing completed native run identity')
        if self.ui_source_run and not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}', self.ui_source_run):
            raise ValueError('invalid native UI source run identity')
        if any(not model.startswith('openai/') for model in (self.chat_model,self.embedding_model)):
            raise ValueError('native LiteLLM models require the explicit openai/ adapter prefix')
        if len({item.artifact for item in self.knowledge}) != len(self.knowledge):
            raise ValueError('one exact inherited knowledge artifact per coder required')
        if any(item.source_run == self.run_id for item in self.knowledge):
            raise ValueError('native knowledge must come from a distinct prior run')
        return self

    def native_args(self) -> list[str]:
        command = self.command
        if command in ('fin_factor','fin_model','fin_quant'):
            return [command,'--step-n',str(self.max_steps),'--loop-n',str(self.max_loops),'--no-checkout']
        if command == 'fin_factor_report':
            return [command,'--report-folder','/sources','--no-checkout']
        if command == 'general_model':
            return [command,'/sources/'+Path(self.document).name]
        if command == 'ui':
            return [command,'--port','19899','--log-dir','/work/log']
        return [command,'--check-env','--no-check-docker','--check-ports']


class NativeSupervisor:
    def __init__(self, root: str | Path):
        from src.research.native_mounts import canonical_path
        self.root = canonical_path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def reserve_mounts(self, request, registry, *, control_paths=()):
        from src.research.native_mounts import NativeMountAuthority
        from src.research.registry import ResearchRegistry
        if not isinstance(registry,ResearchRegistry):
            raise ValueError('native execution requires the authoritative research registry')
        protected=[registry.path,Path(str(registry.path)+'-wal'),Path(str(registry.path)+'-shm'),
                   *map(Path,control_paths)]
        inputs=[self.root/'.inputs'/request.run_id,Path(request.scenario.provider_uri)]
        if request.document:
            inputs.append(Path(request.document))
        return NativeMountAuthority().reserve(
            {'request_hash':request.content_hash(),'registry':str(registry.path.resolve()),
             'run_id':request.run_id},writable=self.root/request.run_id,inputs=tuple(inputs),controls=tuple(protected))

    @staticmethod
    def _docker(*args, timeout=30):
        try:
            result = subprocess.run(['docker',*args],capture_output=True,text=True,timeout=timeout,check=False)
        except (FileNotFoundError,subprocess.TimeoutExpired) as exc:
            raise RuntimeError('native RD execution requires an available Docker engine') from exc
        if result.returncode:
            raise RuntimeError('Docker operation failed: '+result.stderr.strip()[:1000])
        return result.stdout

    def preflight(self, request: NativeRequest, *, gateway_token_path: Path) -> dict:
        manifest = verify_bundle(request.scenario)
        self._docker('info','--format','{{.ServerVersion}}')
        image = json.loads(self._docker('image','inspect',request.image))[0]
        network = json.loads(self._docker('network','inspect',request.network))[0]
        if not network.get('Internal'):
            raise ValueError('native generated code requires an internal Docker network with no direct internet route')
        requested_at = datetime.now(timezone.utc)
        receipt = self.gateway_receipt(request, gateway_token_path)
        self.verify_receipt(request, receipt, now=datetime.now(timezone.utc), not_before=requested_at)
        return {'upstream_revision':UPSTREAM_REVISION,'image_id':image['Id'], 'network_id':network['Id'],
                'scenario_hash':request.scenario.content_hash(),'bundle_hash':request.scenario.bundle_hash,
                'data_snapshot_hash': hashlib.sha256(json.dumps(manifest['snapshot'],sort_keys=True).encode()).hexdigest(),
                'gateway_receipt':receipt}

    @staticmethod
    def gateway_receipt(request: NativeRequest, token_path: Path) -> dict:
        from src.research.gateway_deployment import private_secret
        token = private_secret(token_path)
        deadline = time.monotonic()+10
        with httpx.Client(timeout=10,follow_redirects=False,trust_env=False) as client:
            with client.stream('GET',request.gateway_receipt_url.rstrip('/')+'/v1/budget',
                               headers={'Authorization':'Bearer '+token}) as response:
                response.raise_for_status()
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw)>1024*1024 or time.monotonic()>deadline:
                        raise ValueError('gateway receipt exceeds its bounded size/deadline')
        return json.loads(raw)

    @staticmethod
    def verify_receipt(request: NativeRequest, receipt: dict, *, now: datetime, final: bool=False,
                       not_before: datetime | None=None) -> dict:
        if (not isinstance(receipt, dict) or receipt.get('schema_version') != '1.0'
                or any(type(receipt.get(k)) is not bool for k in
                       ('ready','provider_configured','streaming','blocked'))):
            raise ValueError('closed gateway schema and boolean status required')
        if receipt.get('run_id') != request.run_id or receipt.get('config_hash') != request.gateway_config_hash:
            raise ValueError('gateway admission does not match native run/config identity')
        if ((not final and not receipt.get('ready')) or not receipt.get('provider_configured')
                or receipt.get('streaming') is not False):
            raise ValueError('metered model gateway is not ready')
        expires = datetime.fromisoformat(receipt['expires_at'].replace('Z','+00:00'))
        if expires.tzinfo is None or (not final and expires < now+timedelta(seconds=request.max_seconds)):
            raise ValueError('gateway expiry does not cover the bounded native run')
        for scope, model in (('chat',request.chat_model),('embedding',request.embedding_model)):
            declared = receipt['models'][scope]
            if declared['model'] != model.removeprefix('openai/') or declared['provider'] != 'openai':
                raise ValueError('gateway model differs from the native request')
            if not final and receipt.get('admission_available',{}).get(scope) is not True:
                raise ValueError('gateway scope is exhausted')
        for name, expected in (('limits',request.gateway_limits),('day_limits',request.gateway_day_limits)):
            _usage_integer(receipt[name]['requests'])
            _usage_integer(receipt[name]['tokens'])
            _usage_cost(receipt[name]['cost_usd'])
            actual = NativeBudgetLimits.model_validate(receipt[name])
            if actual != expected:
                raise ValueError('gateway budget caps differ from explicit native bounds')
        allowed = {request.chat_model.removeprefix('openai/'),request.embedding_model.removeprefix('openai/')}
        if set(receipt.get('observed_models',[]))-allowed:
            raise ValueError('gateway reported a model substitution or fault')
        accounting = _gateway_accounting(request,receipt,now)
        if not_before is not None:
            at = datetime.fromisoformat(accounting['as_of'].replace('Z','+00:00'))
            if not_before.tzinfo is None or not not_before <= at <= now:
                raise ValueError('gateway accounting predates the requested fresh audit')
        if not final and _budget_audit_error(receipt):
            raise ValueError('gateway usage is breached or uncertain')
        return accounting

    def command(self, request: NativeRequest, work: Path, token: Path, *, frozen_root: Path | None=None) -> list[str]:
        data_root=frozen_root/'qlib' if frozen_root is not None else Path(request.scenario.provider_uri).resolve()
        if (request.document or request.knowledge or request.command=='ui') and frozen_root is None:
            raise ValueError('native sources require authoritative frozen admission before command construction')
        for path in (work,token,data_root):
            if ',' in str(path):
                raise ValueError('Docker mount paths cannot contain comma separators')
        command = ['docker','run','--rm','--name','quantagent-rd-'+request.run_id,
                   '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
                   '--pids-limit','256','--memory',f'{request.memory_mb}m','--cpus',str(request.cpus),
                   '--log-driver','none','--label','quantagent.native-run='+request.run_id,
                   '--label','quantagent.request='+request.content_hash(),
                   '--user',f'{os.getuid()}:{os.getgid()}', '--network',request.network,
                   '--tmpfs','/tmp:rw,nosuid,nodev,size=512m',
                   '--mount',f'type=bind,src={work},dst=/work',
                   '--mount',f'type=bind,src={data_root},dst=/data/qlib,readonly',
                   '--mount',f'type=bind,src={token.resolve()},dst=/run/gateway-token,readonly']
        for name in ('request.json','scenario.json','admission.json'):
            command += ['--mount',f'type=bind,src={work/name},dst=/work/{name},readonly']
        if request.knowledge:
            directory=frozen_root/'knowledge'
            command += ['--mount',f'type=bind,src={directory},dst=/work/knowledge-input,readonly']
        if request.document:
            command += ['--mount',f'type=bind,src={frozen_root}/sources,dst=/sources,readonly']
        if request.command == 'ui':
            command += ['--mount',f'type=bind,src={frozen_root}/log,dst=/work/log,readonly']
            command += ['--publish',f'127.0.0.1:{request.ui_port}:19899']
        return [*command,request.image]

    @staticmethod
    def _ordinary_bytes(path: Path, limit: int) -> bytes:
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('native source must not use symbolic links')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as handle:
            info=os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>limit:
                raise ValueError('native source exceeds ordinary bounded file contract')
            raw=handle.read(limit+1)
        if len(raw)>limit:
            raise ValueError('native source exceeds byte allowance')
        return raw

    def _prior_run(self, run_id, registry, *, seen=frozenset()):
        if run_id in seen:
            raise ValueError('native inherited source lineage contains a cycle')
        exposure=registry.discovery_exposure('native:'+run_id)
        authority=registry.native_result(run_id)
        root=self.root/run_id
        result_path=root/'result.json'
        raw=self._ordinary_bytes(result_path,16*1024*1024)
        result=json.loads(raw)
        if result!=authority or result.get('discovery_exposure_hash')!=exposure.content_hash():
            raise ValueError('native source result differs from authoritative registry completion')
        request_path=root/'request.json'
        request_raw=self._ordinary_bytes(request_path,16*1024*1024)
        if hashlib.sha256(request_raw).hexdigest()!=result['files'].get('request.json'):
            raise ValueError('native knowledge source request changed')
        payload=json.loads(request_raw)
        payload.pop('native_args',None)
        request=NativeRequest.model_validate(payload)
        if (result.get('run_id')!=run_id or result.get('status')!='discovery_complete'
                or result.get('request_hash')!=request.content_hash()
                or exposure.request_hash!=request.content_hash()
                or result.get('upstream_revision')!=UPSTREAM_REVISION
                or result.get('gateway_audit_error') or not result.get('gateway_final_receipt')):
            raise ValueError('native knowledge requires a complete audited source run')
        expected=[]
        for parent in request.knowledge:
            _,_,prior,parent_raw=self._prior_run(parent.source_run,registry,seen=seen|{run_id})
            if hashlib.sha256(parent_raw).hexdigest()!=parent.source_result_hash:
                raise ValueError('native inherited knowledge result changed')
            expected.append(prior.content_hash())
        if request.ui_source_run:
            expected.append(self._prior_run(request.ui_source_run,registry,seen=seen|{run_id})[2].content_hash())
        if tuple(sorted(expected))!=exposure.inherited_exposures:
            raise ValueError('native source exposure lineage differs from admitted sources')
        return request,result,exposure,raw

    def knowledge_bytes(self, reference: NativeKnowledgeRef, *, registry) -> bytes:
        """Verify authoritative prior admission/completion; never deserialize pickle."""
        _,result,_,raw=self._prior_run(reference.source_run,registry)
        if hashlib.sha256(raw).hexdigest()!=reference.source_result_hash:
            raise ValueError('native knowledge source result changed')
        root=self.root/reference.source_run
        name='knowledge/'+reference.artifact+'.pkl'
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('native knowledge artifact is absent or escaped its run')
        data=self._ordinary_bytes(path,1024*1024*1024)
        if hashlib.sha256(data).hexdigest()!=reference.artifact_hash or result['files'].get(name)!=reference.artifact_hash:
            raise ValueError('native knowledge artifact bytes changed')
        return data

    def admit_discovery(self, request: NativeRequest, registry):
        from src.research.registry import ResearchRegistry
        if not isinstance(registry,ResearchRegistry):
            raise ValueError('native execution requires the authoritative research registry')
        self.reserve_mounts(request,registry)
        root=Path(request.scenario.provider_uri).absolute()
        actual=set()
        if any(path.is_symlink() for path in (root,*root.parents)):
            raise ValueError('native data mount must not be a symlink')
        for path in root.rglob('*'):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError('native data mount contains unsupported links or special files')
            if path.is_file():
                if path.stat().st_nlink!=1:
                    raise ValueError('native data file must not be a hardlink')
                actual.add(str(path.relative_to(root)))
        manifest=verify_bundle(request.scenario)
        if actual != {*manifest['files'],'manifest.json'}:
            raise ValueError('native data mount contains unrecorded or missing files')
        # Hashes alone could bind binaries with observations absent from the
        # declared snapshot. Rebuild the real format and compare every file.
        from src.data.qlib_us import export_bundle
        from src.domain.market import MarketSnapshot
        from src.research.registry import RegistryError
        frozen=self.root/'.inputs'/request.run_id
        if frozen.exists():
            raise RegistryError('native discovery already admitted or staged; reconcile original run')
        if any(p.is_symlink() for p in (frozen,*frozen.parents)):
            raise ValueError('native frozen input path must not use symbolic links')
        frozen.mkdir(parents=True,mode=0o700)
        rebuilt=export_bundle(MarketSnapshot.model_validate(manifest['snapshot']),frozen/'qlib')
        rebuilt.pop('bundle_hash')
        if rebuilt != manifest:
            raise ValueError('native bundle bytes differ from the declared exposure snapshot')
        documents=None
        if request.document:
            source=Path(request.document).absolute()
            if any(p.is_symlink() for p in (source,*source.parents)):
                raise ValueError('native source document must not use symbolic links')
            paths=sorted(source.iterdir()) if source.is_dir() else [source]
            if not paths or len(paths)>request.max_source_files:
                raise ValueError('native source file allowance exceeded or source empty')
            (frozen/'sources').mkdir()
            documents={'scope':'unbounded','files':{}}
            remaining=request.max_source_bytes
            for path in paths:
                data=self._ordinary_bytes(path,remaining)
                remaining-=len(data)
                key=registry.archive(data)
                documents['files'][path.name]={'sha256':key,'bytes':len(data)}
                (frozen/'sources'/path.name).write_bytes(data)
        inherited=[]
        if request.knowledge:
            (frozen/'knowledge').mkdir()
            for reference in request.knowledge:
                data=self.knowledge_bytes(reference,registry=registry)
                inherited.append(registry.discovery_exposure('native:'+reference.source_run))
                (frozen/'knowledge'/(reference.artifact+'.pkl')).write_bytes(data)
        if request.command=='ui':
            _,result,prior,_=self._prior_run(request.ui_source_run,registry)
            inherited.append(prior)
            (frozen/'log').mkdir()
            remaining=request.max_source_bytes
            names=[name for name in result['files'] if name.startswith('log/')]
            if not names or len(names)>request.max_source_files:
                raise ValueError('native UI source logs empty or exceed file allowance')
            for name in names:
                source=self.root/request.ui_source_run/name
                if not source.resolve().is_relative_to((self.root/request.ui_source_run).resolve()):
                    raise ValueError('native UI source log escapes predecessor run')
                data=self._ordinary_bytes(source,remaining)
                remaining-=len(data)
                if hashlib.sha256(data).hexdigest()!=result['files'][name]:
                    raise ValueError('native UI source log bytes changed')
                target=frozen/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(data)
        for path in sorted(frozen.rglob('*'),reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        frozen.chmod(0o555)
        return registry.admit_native_discovery(request,manifest,at=datetime.now(timezone.utc),
                                               document_manifest=documents,inherited_exposures=tuple(inherited))

    def run(self, request: NativeRequest, *, gateway_token_path: str | Path, registry) -> dict:
        from src.research.native_recovery import (
            _artifacts,
            _budget_receipt,
            _load_launch,
            _publish_result,
            _result,
            bind_container,
            native_run_lock,
            record_launch,
        )
        from src.research.registry import ResearchRegistry
        if not isinstance(registry,ResearchRegistry):
            raise ValueError('native execution requires the authoritative research registry')
        token = Path(gateway_token_path).absolute()
        self.reserve_mounts(request,registry,control_paths=(token,))
        evidence = self.preflight(request,gateway_token_path=token)
        # All mounted rows are available to generated code regardless of Qlib's
        # configured split/universe. Reserve that exposure before any process.
        exposure=self.admit_discovery(request,registry)
        evidence={**evidence,'discovery_exposure':exposure.model_dump(mode='json'),
                  'discovery_exposure_hash':exposure.content_hash()}
        work = self.root/request.run_id
        work.mkdir()  # Run identities and discovery output directories are immutable.
        scenario = {**request.scenario.model_dump(mode='json'),'provider_uri':'/data/qlib'}
        (work/'scenario.json').write_text(json.dumps(scenario,indent=2))
        runtime = {**request.model_dump(mode='json'),'native_args':request.native_args()}
        (work/'request.json').write_text(json.dumps(runtime,indent=2))
        (work/'admission.json').write_text(json.dumps(evidence,indent=2))
        command = self.command(request,work,token,frozen_root=self.root/'.inputs'/request.run_id)
        started = time.monotonic()
        state, code = 'failed',None
        primary_error = None
        with (work/'stdout.log').open('wb') as stdout, (work/'stderr.log').open('wb') as stderr:
            with native_run_lock(registry,request.run_id):
                record_launch(self,request,registry,gateway_token_path=token,
                              preflight=evidence,exposure=exposure)
                if command[:2] != ['docker','run']:
                    raise ValueError('native launch requires the fixed Docker command')
                container_id = self._docker('create',*command[2:]).strip()
                if not re.fullmatch(r'[0-9a-f]{64}',container_id):
                    raise ValueError('Docker create did not return one exact container ID')
                bind_container(self,registry,request.run_id,container_id)
                process = subprocess.Popen(['docker','start','--attach',container_id],
                                           stdout=stdout,stderr=stderr)
            try:
                while process.poll() is None:
                    if time.monotonic()-started > request.max_seconds:
                        state = 'timeout'
                        break
                    if sum(p.stat().st_size for p in work.rglob('*') if p.is_file() and not p.is_symlink()) > request.max_output_bytes:
                        state = 'output_limit'
                        break
                    time.sleep(.25)
                else:
                    code = process.returncode
                    state = {0:'discovery_complete',124:'timeout',125:'output_limit',130:'interrupted'}.get(code,'failed')
            except BaseException as exc:
                primary_error = exc
                state = 'interrupted'
                raise
            finally:
                try:
                    cleanup_error = None
                    if process.poll() is None:
                        try:
                            self._docker('rm','--force',container_id)
                            process.wait(timeout=30)
                        except Exception as exc:
                            cleanup_error = type(exc).__name__
                            state = 'cleanup_failed'
                            try:
                                process.kill()
                                process.wait(timeout=10)
                            except Exception as kill_error:
                                cleanup_error += ':'+type(kill_error).__name__
                    code = process.returncode
                    final_receipt, audit_error = None, None
                    try:
                        audit_started_at = datetime.now(timezone.utc)
                        final_receipt = _budget_receipt(request,self.gateway_receipt(request,token),
                                                        datetime.now(timezone.utc), not_before=audit_started_at)
                        audit_error = _budget_audit_error(final_receipt)
                    except Exception as exc:
                        audit_error = type(exc).__name__
                    if audit_error and state == 'discovery_complete':
                        state = 'audit_unavailable'
                    # The original attached process and recovery share one
                    # authority. Neither may replace the other's publication.
                    with native_run_lock(registry,request.run_id):
                        retained = _result(registry,request.run_id)
                        if retained is not None:
                            result = retained
                        else:
                            files, artifact_error = {}, None
                            if code == 0 and cleanup_error is None:
                                try:
                                    artifacts = _artifacts(work,request)
                                    if _artifacts(work,request) != artifacts:
                                        raise ValueError('native outputs changed during finalization')
                                    _load_launch(self,registry,request,exposure)
                                    files = {f['path']:f['sha256'] for f in artifacts['files']}
                                except Exception as exc:
                                    artifact_error = type(exc).__name__
                                    if state == 'discovery_complete':
                                        state = 'artifacts_unavailable'
                            result = {'status':state,'returncode':code,'run_id':request.run_id,
                                      'elapsed_seconds':time.monotonic()-started,'request_hash':request.content_hash(),
                                      'upstream_revision':UPSTREAM_REVISION,'files':files,'activation':'quarantined',
                                      'discovery_exposure_hash':exposure.content_hash(),'internal_search_trials':None,
                                      'cleanup_error':cleanup_error,'gateway_final_receipt':final_receipt,
                                      'gateway_audit_error':audit_error,'artifact_error':artifact_error,
                                      'container_id':container_id}
                            registry.record_native_result(result)
                        _publish_result(work,result)
                except BaseException as final_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note('Native finalization also failed: '+type(final_error).__name__)
        return result
