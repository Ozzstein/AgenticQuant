"""Supervise pinned native RD-Agent jobs in bounded, isolated containers.

Native outputs are quarantined discovery artifacts. They never authorize a
strategy/model or replace the application's ledger replay and holdout evaluator.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
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
from src.domain.common import NonEmptyString, VersionedContract
from src.research.native_runtime.overlay import UPSTREAM_REVISION

COMMANDS = ('fin_factor', 'fin_model', 'fin_quant', 'fin_factor_report', 'general_model', 'ui', 'health_check')


class NativeBudgetLimits(VersionedContract):
    requests: int = Field(gt=0)
    tokens: int = Field(gt=0)
    cost_usd: Decimal = Field(gt=0)


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
    document: str | None = None
    ui_source_run: str | None = None
    ui_port: int = Field(default=19899, ge=1024, le=65535)

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
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

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
        token = gateway_token_path.read_text().strip()
        if not token:
            raise ValueError('scoped gateway bearer is empty')
        with httpx.Client(timeout=10,follow_redirects=False,trust_env=False) as client:
            response = client.get(request.gateway_receipt_url.rstrip('/')+'/v1/budget',
                                  headers={'Authorization':'Bearer '+token})
            response.raise_for_status()
            receipt = response.json()
        self.verify_receipt(request, receipt, now=datetime.now(timezone.utc))
        return {'upstream_revision':UPSTREAM_REVISION,'image_id':image['Id'], 'network_id':network['Id'],
                'scenario_hash':request.scenario.content_hash(),'bundle_hash':request.scenario.bundle_hash,
                'data_snapshot_hash': hashlib.sha256(json.dumps(manifest['snapshot'],sort_keys=True).encode()).hexdigest(),
                'gateway_receipt':receipt}

    @staticmethod
    def verify_receipt(request: NativeRequest, receipt: dict, *, now: datetime) -> None:
        if receipt.get('run_id') != request.run_id or receipt.get('config_hash') != request.gateway_config_hash:
            raise ValueError('gateway admission does not match native run/config identity')
        if not receipt.get('ready') or not receipt.get('provider_configured') or receipt.get('streaming') is not False:
            raise ValueError('metered model gateway is not ready')
        expires = datetime.fromisoformat(receipt['expires_at'].replace('Z','+00:00'))
        if expires.tzinfo is None or expires < now+timedelta(seconds=request.max_seconds):
            raise ValueError('gateway expiry does not cover the bounded native run')
        for scope, model in (('chat',request.chat_model),('embedding',request.embedding_model)):
            declared = receipt['models'][scope]
            if declared['model'] != model.removeprefix('openai/') or declared['provider'] != 'openai':
                raise ValueError('gateway model differs from the native request')
            if receipt.get('admission_available',{}).get(scope) is not True:
                raise ValueError('gateway scope is exhausted')
        for name, expected in (('limits',request.gateway_limits),('day_limits',request.gateway_day_limits)):
            actual = NativeBudgetLimits.model_validate(receipt[name])
            if actual != expected:
                raise ValueError('gateway budget caps differ from explicit native bounds')
        allowed = {request.chat_model.removeprefix('openai/'),request.embedding_model.removeprefix('openai/')}
        if set(receipt.get('observed_models',[]))-allowed or receipt.get('blocked'):
            raise ValueError('gateway reported a model substitution or fault')

    def command(self, request: NativeRequest, work: Path, token: Path) -> list[str]:
        for path in (work,token,Path(request.scenario.provider_uri).resolve()):
            if ',' in str(path):
                raise ValueError('Docker mount paths cannot contain comma separators')
        command = ['docker','run','--rm','--name','quantagent-rd-'+request.run_id,
                   '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
                   '--pids-limit','256','--memory',f'{request.memory_mb}m','--cpus',str(request.cpus),
                   '--user',f'{os.getuid()}:{os.getgid()}', '--network',request.network,
                   '--tmpfs','/tmp:rw,nosuid,nodev,size=512m',
                   '--mount',f'type=bind,src={work},dst=/work',
                   '--mount',f'type=bind,src={Path(request.scenario.provider_uri).resolve()},dst=/data/qlib,readonly',
                   '--mount',f'type=bind,src={token.resolve()},dst=/run/gateway-token,readonly']
        if request.document:
            source = Path(request.document).resolve()
            if not source.exists() or ',' in str(source):
                raise ValueError('native source document does not exist or has an invalid mount path')
            # Mount only a staged copy of the selected source, never its host parent.
            destination = work/'sources'
            destination.mkdir()
            import shutil
            if source.is_dir():
                for path in source.iterdir():
                    if path.is_symlink() or not path.is_file():
                        raise ValueError('report source directory must contain ordinary files only')
                    shutil.copyfile(path,destination/path.name)
            else:
                shutil.copyfile(source,destination/source.name)
            command += ['--mount',f'type=bind,src={destination},dst=/sources,readonly']
        if request.command == 'ui':
            source_run = self.root/request.ui_source_run
            result = json.loads((source_run/'result.json').read_text())
            if result['run_id'] != request.ui_source_run:
                raise ValueError('native UI source identity mismatch')
            import shutil
            for name, digest in result['files'].items():
                if not name.startswith('log/'):
                    continue
                source = source_run/name
                if source.is_symlink() or not source.resolve().is_relative_to(source_run.resolve()):
                    raise ValueError('UI source path escaped its recorded run')
                with source.open('rb') as handle:
                    if hashlib.file_digest(handle,'sha256').hexdigest() != digest:
                        raise ValueError('UI source log bytes changed')
                destination = work/name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,destination)
            command += ['--publish',f'127.0.0.1:{request.ui_port}:19899']
        return [*command,request.image]

    def run(self, request: NativeRequest, *, gateway_token_path: str | Path) -> dict:
        token = Path(gateway_token_path).resolve()
        evidence = self.preflight(request,gateway_token_path=token)
        work = self.root/request.run_id
        work.mkdir()  # Run identities and discovery output directories are immutable.
        scenario = {**request.scenario.model_dump(mode='json'),'provider_uri':'/data/qlib'}
        (work/'scenario.json').write_text(json.dumps(scenario,indent=2))
        runtime = {**request.model_dump(mode='json'),'native_args':request.native_args()}
        (work/'request.json').write_text(json.dumps(runtime,indent=2))
        (work/'admission.json').write_text(json.dumps(evidence,indent=2))
        command = self.command(request,work,token)
        started = time.monotonic()
        state, code = 'failed',None
        with (work/'stdout.log').open('wb') as stdout, (work/'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command,stdout=stdout,stderr=stderr)
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
                    state = 'discovery_complete' if code == 0 else 'failed'
            except BaseException:
                state = 'interrupted'
                raise
            finally:
                cleanup_error = None
                if process.poll() is None:
                    try:
                        self._docker('rm','--force','quantagent-rd-'+request.run_id)
                        process.wait(timeout=30)
                    except (RuntimeError,subprocess.TimeoutExpired) as exc:
                        cleanup_error = str(exc)
                        state = 'cleanup_failed'
                        process.kill()
                        process.wait(timeout=10)
                code = process.returncode
                files = {}
                for path in work.rglob('*'):
                    if path.is_file() and not path.is_symlink():
                        with path.open('rb') as handle:
                            files[str(path.relative_to(work))] = hashlib.file_digest(handle,'sha256').hexdigest()
                result = {'status':state,'returncode':code,'run_id':request.run_id,
                          'elapsed_seconds':time.monotonic()-started,'request_hash':request.content_hash(),
                          'upstream_revision':UPSTREAM_REVISION,'files':files,'activation':'quarantined',
                          'cleanup_error':cleanup_error}
                (work/'result.json').write_text(json.dumps(result,indent=2))
        return result
