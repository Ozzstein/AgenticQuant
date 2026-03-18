"""QuantAgentLab — 5-page Streamlit dashboard.

Run with:
    streamlit run src/dashboard.py --server.headless true
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Root path resolution (works whether launched from repo root or src/)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = _HERE.parent if (_HERE.parent / "outputs").exists() or (_HERE.parent / "data").exists() else _HERE
_OUTPUTS = _ROOT / "outputs"
_DATA = _ROOT / "data"

# ---------------------------------------------------------------------------
# Data-loading helpers (all cached, all fault-tolerant)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def _load_json(path: Path) -> dict | list:
    """Load JSON file, returning empty dict on any error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=60)
def _load_portfolio() -> dict:
    """Load paper portfolio JSON."""
    p = _OUTPUTS / "paper_portfolio.json"
    result = _load_json(p)
    return result if isinstance(result, dict) else {}


@st.cache_data(ttl=60)
def _load_audit_files() -> list[dict]:
    """Load up to 10 most-recent audit JSON files from data/audit/."""
    audit_dir = _DATA / "audit"
    if not audit_dir.exists():
        return []
    files = sorted(audit_dir.glob("*.json"), reverse=True)[:10]
    records: list[dict] = []
    for f in files:
        data = _load_json(f)
        if isinstance(data, dict):
            records.append(data)
    return records


@st.cache_data(ttl=60)
def _load_backtest_files() -> list[dict]:
    """Load all backtest result JSON files from outputs/."""
    if not _OUTPUTS.exists():
        return []
    records: list[dict] = []
    for f in sorted(_OUTPUTS.glob("backtest_*.json")):
        data = _load_json(f)
        if isinstance(data, dict):
            records.append(data)
    return records


@st.cache_data(ttl=60)
def _load_factor_library() -> dict:
    """Load factor library JSON."""
    p = _OUTPUTS / "factor_library.json"
    result = _load_json(p)
    return result if isinstance(result, dict) else {}


@st.cache_data(ttl=60)
def _load_scheduler_state() -> dict:
    """Load scheduler state JSON."""
    p = _DATA / "scheduler_state.json"
    result = _load_json(p)
    return result if isinstance(result, dict) else {}


@st.cache_data(ttl=60)
def _load_today_api_costs() -> dict:
    """Load today's API cost file from data/costs/."""
    today = date.today().isoformat()
    costs_dir = _DATA / "costs"
    if not costs_dir.exists():
        return {}
    candidates = list(costs_dir.glob(f"{today}*.json"))
    if not candidates:
        return {}
    result = _load_json(candidates[0])
    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# Page 1 — Portfolio
# ---------------------------------------------------------------------------


def _holdings_df(portfolio: dict) -> pd.DataFrame:
    """Extract holdings DataFrame from portfolio dict."""
    positions: dict[str, Any] = portfolio.get("positions", {})
    if not positions:
        return pd.DataFrame()
    rows = []
    for ticker, pos in positions.items():
        if isinstance(pos, dict):
            rows.append(
                {
                    "ticker": ticker,
                    "qty": pos.get("quantity", 0),
                    "avg_cost": pos.get("avg_cost", 0.0),
                    "current_price": pos.get("current_price", 0.0),
                    "unrealized_pnl": pos.get("unrealized_pnl", 0.0),
                    "weight_pct": pos.get("weight_pct", 0.0),
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _nav_history_df(portfolio: dict) -> pd.DataFrame:
    """Extract NAV history DataFrame."""
    history = portfolio.get("nav_history", portfolio.get("history", []))
    if not history:
        return pd.DataFrame()
    rows = []
    for entry in history:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            rows.append({"timestamp": entry[0], "nav": entry[1]})
        elif isinstance(entry, dict):
            rows.append({"timestamp": entry.get("ts", ""), "nav": entry.get("nav", 0.0)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def page_portfolio() -> None:
    """Render the Portfolio page."""
    st.title("Portfolio")

    portfolio = _load_portfolio()

    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total NAV", f"${portfolio.get('nav', 0):,.2f}")
    col2.metric("Cash", f"${portfolio.get('cash', 0):,.2f}")
    col3.metric("Daily P&L", f"${portfolio.get('daily_pnl', 0):+,.2f}")

    st.divider()

    # Holdings table
    st.subheader("Holdings")
    holdings = _holdings_df(portfolio)
    if holdings.empty:
        st.info("No open positions")
    else:
        st.dataframe(holdings, use_container_width=True)

    # Allocation pie
    st.subheader("Allocation")
    if holdings.empty:
        st.info("No holdings to display allocation")
    else:
        fig = px.pie(holdings, names="ticker", values="weight_pct", title="Portfolio Weights")
        st.plotly_chart(fig, use_container_width=True)

    # NAV history
    st.subheader("NAV History")
    nav_df = _nav_history_df(portfolio)
    if nav_df.empty:
        st.info("No NAV history available")
    else:
        fig2 = px.line(nav_df, x="timestamp", y="nav", title="Daily NAV")
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Agent Analysis
# ---------------------------------------------------------------------------

_DECISION_COLORS = {
    "STRONG_BUY": "success",
    "BUY": "success",
    "HOLD": "warning",
    "SELL": "error",
    "STRONG_SELL": "error",
}


def _render_analysis(analysis: dict) -> None:
    """Render a single analysis record."""
    ticker = analysis.get("ticker", "?")
    decision = str(analysis.get("decision", "HOLD")).upper()
    confidence_raw = analysis.get("confidence", 0)
    try:
        confidence = float(confidence_raw) / 100.0
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    st.markdown(f"### {ticker}")
    color = _DECISION_COLORS.get(decision, "info")
    getattr(st, color)(f"Decision: **{decision}**")
    st.progress(confidence, text=f"Confidence: {confidence:.0%}")

    reasoning = analysis.get("reasoning", "")
    if reasoning:
        with st.expander("Reasoning"):
            st.write(reasoning)

    agent_reports = analysis.get("agent_reports", {})
    if agent_reports:
        with st.expander("Agent Reports"):
            if isinstance(agent_reports, dict):
                rows = []
                for agent, report in agent_reports.items():
                    if isinstance(report, dict):
                        rows.append(
                            {
                                "Agent": agent,
                                "Decision": report.get("decision", ""),
                                "Confidence": report.get("confidence", ""),
                                "Reasoning": str(report.get("reasoning", ""))[:120],
                            }
                        )
                    else:
                        rows.append({"Agent": agent, "Decision": "", "Confidence": "", "Reasoning": str(report)[:120]})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    risk_flags = analysis.get("risk_flags", [])
    if risk_flags:
        st.markdown("**Risk Flags:**")
        for flag in risk_flags:
            st.markdown(f"- {flag}")

    catalysts = analysis.get("catalysts", [])
    if catalysts:
        st.markdown("**Catalysts:**")
        for catalyst in catalysts:
            st.markdown(f"- {catalyst}")

    st.divider()


def page_agent_analysis() -> None:
    """Render the Agent Analysis page."""
    st.title("Agent Analysis")

    ticker_filter = st.text_input("Filter by ticker", "").strip().upper()
    audit_files = _load_audit_files()

    if not audit_files:
        st.info("No audit files found in data/audit/")
        return

    analyses_found = 0
    for audit in audit_files:
        analyses: list[dict] = audit.get("analyses", [])
        for analysis in analyses:
            ticker = str(analysis.get("ticker", "")).upper()
            if ticker_filter and ticker_filter not in ticker:
                continue
            _render_analysis(analysis)
            analyses_found += 1

    if analyses_found == 0:
        st.info(f"No analyses found{f' for ticker {ticker_filter!r}' if ticker_filter else ''}.")


# ---------------------------------------------------------------------------
# Page 3 — Backtests
# ---------------------------------------------------------------------------


def _walk_forward_df(backtest: dict) -> pd.DataFrame:
    """Build walk-forward results DataFrame from backtest dict."""
    results = backtest.get("walk_forward_results", [])
    rows = []
    for fold in results:
        if isinstance(fold, dict):
            rows.append(
                {
                    "date": fold.get("test_start", ""),
                    "sharpe": fold.get("oos_sharpe", fold.get("sharpe", 0.0)),
                    "max_drawdown": fold.get("max_drawdown", 0.0),
                    "total_return": fold.get("total_return", 0.0),
                    "model": fold.get("model", backtest.get("model", "")),
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _validation_df(backtest: dict) -> pd.DataFrame | None:
    """Build validation checks DataFrame."""
    val = backtest.get("validation", backtest.get("validation_result", {}))
    if not val:
        return None
    checks = val.get("checks", [])
    if not checks:
        return None
    rows = [
        {
            "check": c.get("name", ""),
            "passed": "YES" if c.get("passed") else "NO",
            "severity": c.get("severity", "info"),
            "details": c.get("details", ""),
        }
        for c in checks
        if isinstance(c, dict)
    ]
    return pd.DataFrame(rows) if rows else None


def page_backtests() -> None:
    """Render the Backtests page."""
    st.title("Backtests")

    backtest_files = _load_backtest_files()
    if not backtest_files:
        st.info("No backtest results found in outputs/. Run a backtest first.")
        return

    for i, bt in enumerate(backtest_files):
        model_name = bt.get("model", f"Backtest {i + 1}")
        with st.expander(f"{model_name}", expanded=(i == 0)):
            # Walk-forward table
            st.markdown("**Walk-Forward Results**")
            wf_df = _walk_forward_df(bt)
            if wf_df.empty:
                st.info("No walk-forward folds in this result.")
            else:
                st.dataframe(wf_df, use_container_width=True)

            # Aggregate metrics
            metrics = bt.get("metrics", {})
            if metrics:
                st.markdown("**Aggregate Metrics**")
                m_cols = st.columns(4)
                m_cols[0].metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
                m_cols[1].metric("Max DD", f"{metrics.get('max_drawdown', 0):.1%}")
                m_cols[2].metric("Total Return", f"{metrics.get('total_return', 0):.1%}")
                m_cols[3].metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")

            # Validation
            val = bt.get("validation", bt.get("validation_result", {}))
            if val:
                verdict = val.get("verdict", "UNKNOWN")
                st.markdown("**Validation**")
                badge_fn = st.success if verdict == "APPROVED" else (st.warning if verdict == "CAUTION" else st.error)
                badge_fn(f"Verdict: **{verdict}**")
                val_df = _validation_df(bt)
                if val_df is not None:
                    st.dataframe(val_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4 — Factors
# ---------------------------------------------------------------------------


def _factors_df(library: dict) -> pd.DataFrame:
    """Build factors DataFrame from factor library dict."""
    factors: list | dict = library.get("factors", library)
    rows = []
    if isinstance(factors, list):
        for f in factors:
            if isinstance(f, dict):
                rows.append(
                    {
                        "name": f.get("name", ""),
                        "category": f.get("category", ""),
                        "ic_mean": f.get("ic_mean", 0.0),
                        "icir": f.get("icir", f.get("ic_ir", 0.0)),
                    }
                )
    elif isinstance(factors, dict):
        for name, meta in factors.items():
            if isinstance(meta, dict):
                rows.append(
                    {
                        "name": name,
                        "category": meta.get("category", ""),
                        "ic_mean": meta.get("ic_mean", 0.0),
                        "icir": meta.get("icir", meta.get("ic_ir", 0.0)),
                    }
                )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def page_factors() -> None:
    """Render the Factors page."""
    st.title("Factors")

    library = _load_factor_library()
    if not library:
        st.info("No factor library found in outputs/factor_library.json. Run RD-Agent factor discovery first.")
        return

    df = _factors_df(library)
    if df.empty:
        st.info("Factor library is empty or has an unexpected format.")
        return

    st.subheader("Factor Library")
    st.dataframe(df, use_container_width=True)

    # IC/ICIR bar chart
    if "ic_mean" in df.columns and "icir" in df.columns:
        st.subheader("IC / ICIR")
        chart_data = df.melt(id_vars="name", value_vars=["ic_mean", "icir"], var_name="metric", value_name="value")
        fig = px.bar(chart_data, x="name", y="value", color="metric", barmode="group", title="IC and ICIR by Factor")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 5 — System
# ---------------------------------------------------------------------------


def _pipeline_steps_df(scheduler_state: dict) -> pd.DataFrame:
    """Build pipeline step status DataFrame from scheduler state."""
    rows: list[dict] = []
    # Scheduler state: {date: {step: {status, ts}}}
    for date_key in sorted(scheduler_state.keys(), reverse=True)[:1]:
        day_data = scheduler_state[date_key]
        if isinstance(day_data, dict):
            for step, info in day_data.items():
                if isinstance(info, dict):
                    rows.append(
                        {
                            "step": step,
                            "status": info.get("status", ""),
                            "timestamp": info.get("ts", ""),
                        }
                    )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _last_run_info(audit_files: list[dict]) -> tuple[str, str]:
    """Return (last_run_timestamp, overall_status) from most recent audit."""
    if not audit_files:
        return "Never", "unknown"
    audit = audit_files[0]
    steps = audit.get("pipeline_steps", [])
    ts = audit.get("date", "")
    if steps:
        last_step = steps[-1] if isinstance(steps[-1], dict) else {}
        ts = last_step.get("ts", ts)
        statuses = [s.get("status", "") for s in steps if isinstance(s, dict)]
        overall = "ok" if all(s == "ok" for s in statuses) else "partial"
    else:
        overall = "unknown"
    return ts, overall


def page_system() -> None:
    """Render the System page."""
    st.title("System")

    scheduler_state = _load_scheduler_state()
    audit_files = _load_audit_files()
    costs = _load_today_api_costs()

    # Last run info
    last_ts, overall_status = _last_run_info(audit_files)
    st.subheader("Last Pipeline Run")
    col1, col2 = st.columns(2)
    col1.metric("Last Run", last_ts or "N/A")
    status_fn = st.success if overall_status == "ok" else (st.warning if overall_status == "partial" else st.info)
    with col2:
        status_fn(f"Status: **{overall_status}**")

    st.divider()

    # Pipeline step table
    st.subheader("Pipeline Steps (Latest Run)")
    if scheduler_state:
        steps_df = _pipeline_steps_df(scheduler_state)
        if not steps_df.empty:
            st.dataframe(steps_df, use_container_width=True)
        else:
            st.info("No step data found in scheduler state.")
    elif audit_files:
        # Fall back to audit pipeline_steps
        steps = audit_files[0].get("pipeline_steps", [])
        if steps and isinstance(steps[0], dict):
            rows = [{"step": s.get("step", ""), "status": s.get("status", ""), "timestamp": s.get("ts", "")} for s in steps]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No pipeline step data available.")
    else:
        st.info("No scheduler state or audit files found.")

    st.divider()

    # API costs
    st.subheader("Today's API Costs")
    if costs:
        total = costs.get("total_usd", 0.0)
        st.metric("Total Cost (USD)", f"${total:.4f}")
        calls = costs.get("calls", [])
        if calls:
            st.dataframe(pd.DataFrame(calls), use_container_width=True)
    else:
        st.info("No API cost data found for today.")


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(page_title="QuantAgentLab", layout="wide")

    page = st.sidebar.radio(
        "Navigate",
        ["Portfolio", "Agent Analysis", "Backtests", "Factors", "System"],
    )

    match page:
        case "Portfolio":
            page_portfolio()
        case "Agent Analysis":
            page_agent_analysis()
        case "Backtests":
            page_backtests()
        case "Factors":
            page_factors()
        case "System":
            page_system()


if __name__ == "__main__" or True:
    # Streamlit executes the module top-level; call main() conditionally.
    # The `or True` is intentional: Streamlit re-runs the whole file, so we
    # must call main() at module level, but only inside a Streamlit context.
    try:
        import streamlit.runtime.scriptrunner as _sr  # noqa: PLC0415

        if _sr.get_script_run_ctx() is not None:
            main()
    except Exception:
        pass
