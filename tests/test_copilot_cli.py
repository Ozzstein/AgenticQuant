import json

from typer.testing import CliRunner

from scripts.run_research_copilot import MemoInput, ProposalInput, app
from src.research.registry import ResearchRegistry
from tests.test_research_copilot import research_env as research_env


def arguments(tmp_path,env,request):
    source=tmp_path/'request.json'
    source.write_text(request.model_dump_json())
    code,environment=tmp_path/'code.txt',tmp_path/'environment.txt'
    code.write_text('reviewed fixture execution code')
    environment.write_text('pinned fixture dependency set')
    output=tmp_path/'outcome.json'
    return ['--request',str(source),'--registry',str(env.registry.path),'--evidence',str(env.store.path),
            '--code',str(code),'--environment',str(environment),'--output',str(output)],output


def test_cli_missing_access_archives_unavailable_without_claiming_candidate(research_env,tmp_path,monkeypatch):
    env=research_env
    monkeypatch.setattr('scripts.run_research_copilot.get_config',lambda:env.config)
    request=ProposalInput(context=env.context,kind='factor',description='Research source hypothesis',sources=(env.source,))
    args,output=arguments(tmp_path,env,request)
    result=CliRunner().invoke(app,['propose',*args])
    assert result.exit_code==2,result.output
    outcome=json.loads(output.read_text())
    assert outcome['status']=='unavailable' and outcome['candidates']==[]
    assert ResearchRegistry(env.registry.path).status()['counts']['candidates']==0


def test_cli_memo_rejects_caller_metric_fields(research_env,tmp_path,monkeypatch):
    env=research_env
    monkeypatch.setattr('scripts.run_research_copilot.get_config',lambda:env.config)
    request=MemoInput(context=env.context,sources=(env.source,))
    args,output=arguments(tmp_path,env,request)
    path=tmp_path/'request.json'
    path.write_text(json.dumps({**request.model_dump(mode='json'),'sharpe':9}))
    result=CliRunner().invoke(app,['memo',*args])
    assert result.exit_code==1 and not output.exists()


def test_registered_cli_exposes_all_custom_plan_families_and_copilot():
    from scripts.run_rd_agent import app as research_app
    runner=CliRunner()
    result=runner.invoke(research_app,['register','--help'])
    assert result.exit_code==0 and 'strategy' in result.output and 'regime' in result.output
    assert runner.invoke(research_app,['copilot','--help']).exit_code==0
