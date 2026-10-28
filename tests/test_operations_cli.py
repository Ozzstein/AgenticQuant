import json

from typer.testing import CliRunner

from scripts.run_operations import app
from src.operations.refresh import RefreshPlan
from src.operations.store import OperationsStore
from tests.test_operations_refresh import context, task


def test_refresh_cli_missing_access_has_failure_receipt_and_nonzero_exit(tmp_path,monkeypatch):
    monkeypatch.delenv('FMP_API_KEY',raising=False)
    ctx=context()
    plan=RefreshPlan(context=ctx,scheduled_for=ctx.as_of,tasks=(task(),))
    source=tmp_path/'request.json'
    source.write_text(plan.model_dump_json())
    ops=tmp_path/'ops.db'
    response=CliRunner().invoke(app,['refresh','--input',str(source),'--store',str(ops),'--evidence',str(tmp_path/'data.db')])
    assert response.exit_code==1
    assert OperationsStore(ops,read_only=True).status(ctx.run_id)['status']=='failed'


def test_status_cli_serializes_only_real_missing_configured_stores(tmp_path,monkeypatch):
    from src.utils.config import AppConfig
    cfg=AppConfig(_env_file=None,output_dir=str(tmp_path/'outputs'),log_dir=str(tmp_path/'logs'))
    with monkeypatch.context() as patch:
        patch.setattr('src.utils.config.get_config',lambda:cfg)
        response=CliRunner().invoke(app,['status'])
    assert response.exit_code==0,response.output
    value=json.loads(response.output)
    assert all(section['status']=='missing' for name,section in value.items() if name not in ('decision_budgets','budgets'))
    assert value['decision_budgets'] == {'status':'unavailable','reason':'decision_store_missing'}
    assert value['budgets']['day_usage'] is None
    assert not (tmp_path/'outputs/paper-ledger.sqlite3').exists()


def test_canonical_cli_exposes_the_same_operating_service():
    from src.utils.cli import app as canonical
    response=CliRunner().invoke(canonical,['operations','--help'])
    assert response.exit_code==0
    assert 'refresh' in response.output and 'resolve-delivery' in response.output
