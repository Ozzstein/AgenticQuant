"""Pinned native adaptation checks; Docker/provider execution is separate acceptance."""
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, StrictUndefined

from src.research.native import NativeKnowledgeRef, NativeRequest, NativeSupervisor
from src.research.native_runtime.entrypoint import main
from src.research.native_runtime.overlay import prepare_overlay


def scenario():
    return dict(provider_uri='/data/qlib',bundle_hash='1'*64,universe=['us_one','us_two'],benchmark='us_one',
                train=['2019-01-01','2019-12-27'],valid=['2020-01-02','2020-12-28'],test=['2021-01-04','2021-12-31'],
                price_basis='raw',calendar='XNYS',region='us',label_horizon=1,equity_cost_bps=10)


def request():
    return NativeRequest(run_id='native-test',command='fin_factor',scenario=scenario(),image='local/rd@sha256:'+'2'*64,
                         network='research-internal',gateway_url='http://gateway:8787',gateway_receipt_url='http://127.0.0.1:8787',
                         gateway_config_hash='3'*64,chat_model='openai/pinned-chat',embedding_model='openai/pinned-embedding',
                         gateway_limits=dict(requests=100,tokens=100000,cost_usd='5'),
                         gateway_day_limits=dict(requests=200,tokens=200000,cost_usd='10'))


def test_command_has_no_host_execution_or_provider_credentials(tmp_path):
    req=request()
    supervisor=NativeSupervisor(tmp_path/'jobs')
    args=supervisor.command(req,tmp_path/'work',tmp_path/'scoped-token')
    assert args[:2]==['docker','run']
    assert '--read-only' in args and '--cap-drop' in args and '--network' in args
    assert not any('docker.sock' in arg or 'API_KEY' in arg for arg in args)
    assert req.native_args()==['fin_factor','--step-n','30','--loop-n','5','--no-checkout']
    assert args[-1]==req.image


def test_host_native_execution_is_rejected(monkeypatch):
    monkeypatch.delenv('QUANTAGENT_NATIVE_SANDBOX',raising=False)
    with pytest.raises(RuntimeError,match='isolated supervisor container'):
        main()


def test_all_required_native_commands_and_source_paths():
    for command in ('fin_factor','fin_model','fin_quant','fin_factor_report','general_model','ui','health_check'):
        payload=request().model_dump(mode='json')
        payload.update(command=command,document='/tmp/paper.pdf' if command in ('fin_factor_report','general_model') else None)
        payload['ui_source_run'] = 'previous-run' if command == 'ui' else None
        req=NativeRequest.model_validate(payload)
        assert req.native_args()[0]==command
    with pytest.raises(ValueError):
        NativeRequest.model_validate({**request().model_dump(mode='json'),'run_id':'../outside'})


def test_real_pinned_templates_render_explicit_us_scenario(tmp_path):
    root=Path(__file__).parents[1]/'environments/rd-agent/.venv/lib/python3.11/site-packages/rdagent'
    if not root.exists():
        pytest.skip('separate pinned native environment is not installed')
    output=tmp_path/'rdagent'
    manifest=prepare_overlay(root,output,scenario())
    assert manifest['upstream_revision']=='484776c211e4fbbeef03e0ec00d6bbee7362a4f4'
    assert len(manifest['files'])==8
    context={f'{segment}_{position}':scenario()[segment][i] for segment in ('train','valid','test')
             for i,position in enumerate(('start','end'))}
    context.update(feature_expressions=['$close'],feature_names=['CLOSE'],model_code='',n_epochs=1,lr=.01,early_stop=1,batch_size=8,weight_decay=0,num_timesteps=None,num_features=1,step_len=None)
    for path in output.glob('scenarios/qlib/experiment/*template/*.yaml'):
        text=path.read_text()
        # Some templates have additional native placeholders; satisfy only those
        # used by the pinned version, with real parser-compatible local paths.
        template=Environment(undefined=StrictUndefined).from_string(text)
        content=template.render(**context)
        result=yaml.safe_load(content)
        assert result['qlib_init']['region']=='us'
        assert result['market']==scenario()['universe']
        assert result['benchmark']=='us_one'
        costs=result['port_analysis_config']['backtest']['exchange_kwargs']
        assert costs['limit_threshold'] is None and costs['deal_price']=='open'
        assert costs['open_cost']==costs['close_cost']==.001 and costs['min_cost']==0
        assert 'cn_data' not in content and 'csi300' not in content
    generated=(output/'scenarios/qlib/experiment/factor_data_template/generate.py').read_text()
    compile(generated,'generated-us-data.py','exec')
    assert 'scenario[\'universe\']' in generated


def test_receipt_enforces_models_caps_scope_expiry_and_faults():
    req = request()
    now = datetime.now(timezone.utc)
    receipt = dict(run_id=req.run_id,config_hash=req.gateway_config_hash,ready=True,provider_configured=True,
                   streaming=False,expires_at=(now+timedelta(hours=2)).isoformat(),
                   models={scope:dict(provider='openai',model=model.removeprefix('openai/'))
                           for scope,model in [('chat',req.chat_model),('embedding',req.embedding_model)]},
                   admission_available={'chat':True,'embedding':True},
                   limits=req.gateway_limits.model_dump(mode='json',exclude={'schema_version'}),
                   day_limits=req.gateway_day_limits.model_dump(mode='json',exclude={'schema_version'}),
                   observed_models=[],blocked=False)
    NativeSupervisor.verify_receipt(req,receipt,now=now)
    for path,value in [(('models','chat','model'),'other-model'),(('limits','requests'),101),
                       (('admission_available','embedding'),False), (('blocked',),True),
                       (('expires_at',),now.isoformat()),(('observed_models',),['substitute'])]:
        changed = copy.deepcopy(receipt)
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ValueError):
            NativeSupervisor.verify_receipt(req,changed,now=now)


def test_exact_native_knowledge_bytes_are_staged_without_host_deserialization(tmp_path):
    import pickle
    marker=tmp_path/'must-not-execute'
    class Untrusted:
        def __reduce__(self):
            return (eval,(f"__import__('pathlib').Path({str(marker)!r}).write_text('unsafe')",))
    payload=pickle.dumps(Untrusted())
    supervisor=NativeSupervisor(tmp_path/'jobs')
    source=supervisor.root/'prior'
    (source/'knowledge').mkdir(parents=True)
    (source/'knowledge'/'factor.pkl').write_bytes(payload)
    previous=request().model_copy(update={'run_id':'prior'})
    (source/'request.json').write_text(previous.model_dump_json())
    def h(raw):
        return hashlib.sha256(raw).hexdigest()
    result=dict(run_id='prior',status='discovery_complete',request_hash=previous.content_hash(),
        upstream_revision='484776c211e4fbbeef03e0ec00d6bbee7362a4f4',gateway_audit_error=None,
        gateway_final_receipt={'fixture':'offline'},files={'knowledge/factor.pkl':h(payload),
        'request.json':h((source/'request.json').read_bytes())})
    (source/'result.json').write_text(json.dumps(result))
    ref=NativeKnowledgeRef(source_run='prior',source_result_hash=h((source/'result.json').read_bytes()),
                           artifact='factor',artifact_hash=h(payload))
    req=request().model_copy(update={'knowledge':(ref,)})
    work=supervisor.root/'next'
    work.mkdir()
    command=supervisor.command(req,work,tmp_path/'bearer')
    assert (work/'knowledge-input'/'factor.pkl').read_bytes()==payload and not marker.exists()
    assert f'type=bind,src={work}/knowledge-input,dst=/work/knowledge-input,readonly' in command
    (source/'knowledge'/'factor.pkl').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='bytes changed'):
        supervisor.knowledge_bytes(ref)
