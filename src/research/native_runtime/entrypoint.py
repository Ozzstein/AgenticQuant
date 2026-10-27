"""Container-only entrypoint. Generated native code never executes on the host."""
from __future__ import annotations

import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

from src.research.native_runtime.overlay import prepare_overlay


def worker():
    if not Path('/.dockerenv').exists() or os.environ.get('QUANTAGENT_NATIVE_SANDBOX') != '1':
        raise RuntimeError('native RD entrypoint requires the isolated supervisor container')
    work = Path('/work')
    request = json.loads((work/'request.json').read_text())
    scenario = json.loads((work/'scenario.json').read_text())
    package = importlib.util.find_spec('rdagent')
    locations = tuple(package.submodule_search_locations or ()) if package else ()
    if len(locations) != 1:
        raise RuntimeError('exactly one pinned RD-Agent package installation required')
    source = Path(locations[0])
    overlay = work/'overlay'
    overlay.mkdir(exist_ok=True)
    manifest = prepare_overlay(source, overlay/'rdagent', scenario)
    (work/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2))
    sys.path.insert(0,str(overlay))
    os.environ.update({
        'HOME': '/work/home', 'WORKSPACE_PATH': '/work/workspaces', 'LOG_TRACE_PATH': '/work/log',
        'MODEL_COSTEER_ENV_TYPE': 'conda', 'CONDA_DEFAULT_ENV': 'locked-native',
        'FACTOR_COSTEER_DATA_FOLDER': '/work/factor_data',
        'FACTOR_COSTEER_DATA_FOLDER_DEBUG': '/work/factor_debug',
        'LITELLM_CHAT_STREAM': 'false', 'LITELLM_MAX_RETRY': '0',
        'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'LITELLM_LOG_LLM_CHAT_CONTENT': 'false',
        'LITELLM_CHAT_MODEL': request['chat_model'], 'CHAT_MODEL': request['chat_model'],
        'EMBEDDING_MODEL': request['embedding_model'], 'BACKEND': 'rdagent.oai.backend.LiteLLMAPIBackend',
        'LITELLM_EMBEDDING_MODEL': request['embedding_model'],
        'LITELLM_CHAT_MAX_TOKENS': str(request['max_output_tokens']),
        'OPENAI_API_BASE': request['gateway_url'].rstrip('/')+'/v1',
        'OMP_NUM_THREADS': str(request['cpus']), 'OPENBLAS_NUM_THREADS': str(request['cpus']),
        'MKL_NUM_THREADS': str(request['cpus']), 'NUMEXPR_NUM_THREADS': str(request['cpus']),
    })
    for family in ('QLIB_FACTOR_', 'QLIB_MODEL_', 'QLIB_QUANT_'):
        for segment in ('train','valid','test'):
            for position, suffix in enumerate(('start','end')):
                os.environ[family+segment.upper()+'_'+suffix.upper()] = scenario[segment][position]
    token = Path('/run/gateway-token').read_text().strip() if Path('/run/gateway-token').exists() else ''
    if token:
        os.environ['OPENAI_API_KEY'] = token
    Path('/work/home').mkdir(exist_ok=True)
    os.chdir(work)
    # Native Conda/Docker helpers are local processes ONLY within this outer
    # container. No docker socket, host Python, package install or China download.
    import rdagent.utils.env as native_env
    def bin_path(self):
        self.bin_path = str(Path(sys.executable).parent)
    native_env.CondaConf._update_bin_path = bin_path
    native_env.QlibCondaEnv.prepare = lambda self: None
    class ContainedQlibEnv(native_env.LocalEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(conf=native_env.LocalConf(bin_path=str(Path(sys.executable).parent),
                                                     default_entry='qrun conf.yaml', retry_count=0))
    native_env.QTDockerEnv = ContainedQlibEnv
    # The settings object can contain the scoped bearer; never log it.
    from rdagent.oai.backend.litellm import LiteLLMAPIBackend
    LiteLLMAPIBackend._has_logged_settings = True
    # Upstream health_check logs failed provider probes but exits successfully.
    # Turn its actual probe failures into a nonzero supervised outcome.
    import rdagent.app.utils.health_check as health
    def checked(probe):
        def run(*args, **kwargs):
            if not probe(*args, **kwargs):
                raise RuntimeError('native authenticated health probe failed')
            return True
        return run
    health.test_chat = checked(health.test_chat)
    health.test_embedding = checked(health.test_embedding)
    sys.argv = ['rdagent', *request['native_args']]
    runpy.run_module('rdagent.app.cli', run_name='__main__')


def main():
    if not Path('/.dockerenv').exists() or os.environ.get('QUANTAGENT_NATIVE_SANDBOX') != '1':
        raise RuntimeError('native RD entrypoint requires the isolated supervisor container')
    if sys.argv[1:] == ['--worker']:
        return worker()
    if sys.argv[1:]:
        raise ValueError('unexpected native entrypoint arguments')
    from src.research.native_runtime.watchdog import supervise
    work = Path('/work')
    request = json.loads((work/'request.json').read_text())
    result = supervise([sys.executable, '-m', 'src.research.native_runtime.entrypoint', '--worker'],
                       work, max_seconds=request['max_seconds'], max_output_bytes=request['max_output_bytes'])
    raise SystemExit(result['exit_code'])


if __name__ == '__main__':
    main()
