"""Read-only dashboard over persisted ledger, evidence, budgets and operating receipts."""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
import streamlit as st

from src.operations.reporting import system_view
from src.operations.store import OperationsStore
from src.utils.config import get_config


def _unavailable(section: dict, name: str) -> bool:
    if section.get('status') not in ('available','no_valuations'):
        st.info(f"{name}: {section.get('status','unknown')}. No measured result is available.")
        if section.get('error_type'):
            st.error(section['error_type'])
        return True
    return False


def portfolio_page(view):
    st.title('Portfolio')
    if _unavailable(view,'Ledger'):
        return
    st.caption(f"{view['account_id']} · {view.get('base_currency','')} · paper execution")
    snapshot = view.get('snapshot')
    if snapshot is None:
        st.info('The account has no recorded valuation yet.')
        return
    st.caption(f"Last recorded valuation: {snapshot['as_of']}")
    cols = st.columns(4)
    for column,label,key in zip(cols,('NAV','Cash','Reserved cash','Fees'),('nav','cash','reserved_cash','fees'),strict=True):
        column.metric(label,f"{Decimal(snapshot[key]):,.2f}")
    history = pd.DataFrame(view['history'])
    if not history.empty:
        history['as_of'] = pd.to_datetime(history['as_of'],utc=True)
        history['nav'] = history.nav.astype(float)
        st.line_chart(history.set_index('as_of')[['nav']])
    st.subheader('Holdings')
    st.dataframe(pd.json_normalize(snapshot['positions']),hide_index=True)
    st.subheader('Pending orders')
    st.dataframe(pd.DataFrame(view['pending']),hide_index=True)
    st.subheader('Actual fills')
    st.dataframe(pd.DataFrame(view['fills']),hide_index=True)
    with st.expander('Accounting reconciliation'):
        st.json(view['reconciliation'])


def desk_page(config, operations):
    st.title('Agent desk')
    if _unavailable(operations,'Operating receipts'):
        return
    choices = [run for run in operations['runs'] if run['stages'].get('agent_desk',{}).get('state')=='completed']
    if not choices:
        st.info('No completed evidence-grounded desk receipt has been recorded.')
        return
    chosen = st.selectbox('Run',[run['run_id'] for run in choices])
    run = next(run for run in choices if run['run_id']==chosen)
    store = OperationsStore(config.operations.store_path,read_only=True)
    value = store.artifact(run['stages']['agent_desk']['output_hash'])
    for identity,entry in value.get('decisions',{}).items():
        decision = entry.get('decision',entry)
        with st.expander(f"{identity} · {decision.get('action','unavailable')}"):
            st.write(decision.get('reasoning',''))
            st.caption('Analyst conviction, evidence completeness and calibrated task accuracy retain separate meanings.')
            st.json(decision.get('confidence',{}))
            st.write('Reports and debate')
            st.json({key:decision.get(key) for key in ('reports','debate_transcript','judgments','risk_veto','new_exposure_blocked')})
            st.write('Source coverage')
            st.json(entry.get('coverage',{}))


def research_page(view):
    st.title('Backtests and research')
    if _unavailable(view,'Research registry'):
        return
    st.caption('Development outcomes are recorded ledger evaluations. Holdout results are excluded from this research view.')
    st.subheader('Development outcomes')
    rows = view.get('development_outcomes',[])
    if rows:
        st.dataframe(pd.DataFrame(rows)[['trial_id','status','validation_reliable','synthetic','observations','sharpe','ic','icir','unadjusted_p_value']],hide_index=True)
        selected = st.selectbox('Evaluation evidence',[row['trial_id'] for row in rows])
        st.json(next(row for row in rows if row['trial_id']==selected))
    else:
        st.info('No development outcomes recorded.')
    st.subheader('Candidates in quarantine')
    st.dataframe(pd.DataFrame(view.get('candidates',[])),hide_index=True)
    st.subheader('Versioned activations')
    st.dataframe(pd.DataFrame(view.get('activations',[])),hide_index=True)
    st.caption('Research activation does not authorize live capital.')
    with st.expander('Holdout reservations'):
        st.dataframe(pd.DataFrame(view.get('holdouts',[])),hide_index=True)


def sources_page(view):
    st.title('Data and evidence')
    if _unavailable(view,'Evidence archive'):
        return
    st.dataframe(pd.DataFrame(view.get('kinds',[])),hide_index=True)
    st.caption('Publication, requested cutoff and ingestion clocks remain distinct; source presence alone does not establish historical eligibility.')
    st.dataframe(pd.json_normalize(view.get('sources',[])),hide_index=True)


def operations_page(operations,budgets):
    st.title('Operations')
    if not _unavailable(operations,'Operating store'):
        rows = operations.get('runs',[])
        st.dataframe(pd.DataFrame([{key:row[key] for key in ('run_id','kind','account_id','as_of','status','manifest_hash')} for row in rows]),hide_index=True)
        if rows:
            selected = st.selectbox('Stage receipts',[row['run_id'] for row in rows])
            st.json(next(row for row in rows if row['run_id']==selected)['stages'])
        st.subheader('Schedules and recovery')
        st.dataframe(pd.DataFrame(operations.get('schedule_ticks',[])),hide_index=True)
        st.subheader('Alerts and delivery receipts')
        st.dataframe(pd.DataFrame(operations.get('alerts',[])),hide_index=True)
    st.subheader('Model usage and budget charges')
    if not _unavailable(budgets,'Durable model budget'):
        cols = st.columns(3)
        for column,label,key in zip(cols,('Observed cost','Budget charged','Unknown usage reserved'),
                                    ('actual_cost_usd','charged_cost_usd','uncertain_reserved_cost_usd'),strict=True):
            column.metric(label,f"USD {Decimal(budgets[key]):,.6f}")
        st.caption('Costs use reported tokens at configured rates; they are not provider invoice claims. Unknown usage retains its reservation.')
        st.dataframe(pd.DataFrame(budgets['calls']),hide_index=True)


def main():
    st.set_page_config(page_title='QuantAgentLab',layout='wide')
    config = get_config()
    view = system_view(config)
    page = st.sidebar.radio('Navigate',['Portfolio','Agent Desk','Backtests & Research','Data & Evidence','Operations'])
    st.sidebar.caption('Read-only · recorded results')
    if page == 'Portfolio':
        portfolio_page(view['portfolio'])
    elif page == 'Agent Desk':
        desk_page(config,view['operations'])
    elif page == 'Backtests & Research':
        research_page(view['research'])
    elif page == 'Data & Evidence':
        sources_page(view['evidence'])
    else:
        operations_page(view['operations'],view['budgets'])


if __name__=='__main__':
    main()
