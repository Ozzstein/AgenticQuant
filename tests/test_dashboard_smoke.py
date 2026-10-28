"""Actual Streamlit rendering uses read-only views and missing-data states."""
from pathlib import Path

from src.operations.reporting import budget_view, system_view
from src.utils.config import AppConfig


def test_missing_stores_remain_unknown_and_are_not_created(tmp_path):
    config = AppConfig(output_dir=str(tmp_path/'missing'),_env_file=None)
    result = system_view(config)
    assert all(section['status']=='missing' for name,section in result.items() if name!='decision_budgets')
    assert result['decision_budgets'] == {'status':'unavailable','reason':'decision_store_missing'}
    assert result['budgets']['actual_cost_usd'] is None
    assert not (tmp_path/'missing').exists()


def test_dashboard_pages_render_without_fabricated_values(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.utils.config import reset_config
    monkeypatch.setenv('AIQUANT_OUTPUT_DIR',str(tmp_path/'empty'))
    reset_config()
    app = AppTest.from_file(str(Path('src/dashboard.py').resolve())).run(timeout=20)
    assert not app.exception
    for name in ('Agent Desk','Backtests & Research','Data & Evidence','Operations','Portfolio'):
        app.sidebar.radio[0].set_value(name).run(timeout=20)
        assert not app.exception
        assert not app.metric  # Absent stores never display measured $0 or NAV.
    assert not (tmp_path/'empty').exists()
    reset_config()


def test_budget_report_preserves_unknown_usage_reservation(tmp_path):
    import time
    from datetime import datetime, timezone

    from src.agents.budget import AgentCallBudget
    from src.utils.config import AgentConfig, ModelPricing
    config = AgentConfig(llm_provider='openai',budget_store_path=str(tmp_path/'budget.db'),
        pricing=[ModelPricing(provider='openai',model='test-pinned',price_version='fixture',source_uri='https://example.test',
            effective_at=datetime(2020,1,1,tzinfo=timezone.utc),input_usd_per_million=1,output_usd_per_million=2)])
    budget = AgentCallBudget(config,'run',deadline=time.monotonic()+30)
    budget.reserve_bound('test-pinned',100,100)
    value = budget_view(tmp_path/'budget.db')
    assert value['calls_with_unknown_usage']==1
    assert value['uncertain_reserved_cost_usd']=='0.0003'
    assert value['actual_cost_usd']=='0'
    assert value['charged_cost_usd']=='0.0003'
