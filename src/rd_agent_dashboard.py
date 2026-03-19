"""RD-Agent Streamlit dashboard — knowledge base, factor library, R&D loop results.

Run with:
    streamlit run src/rd_agent_dashboard.py --server.headless true
Or via CLI:
    python scripts/run_rd_agent.py ui --port 8080
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Root path resolution
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = (
    _HERE.parent
    if (_HERE.parent / "outputs").exists() or (_HERE.parent / "data").exists()
    else _HERE
)
_DATA = _ROOT / "data"
_OUTPUTS = _ROOT / "outputs"
_KB_PATH = _DATA / "rd_knowledge_base" / "kb.json"
_FACTOR_LIBRARY_PATH = _OUTPUTS / "factor_library" / "factor_library.json"
_BEST_MODEL_PATH = _OUTPUTS / "best_model_config.yaml"

# ---------------------------------------------------------------------------
# Data-loading helpers (cached, fault-tolerant)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def _load_kb() -> dict[str, Any]:
    """Load the knowledge base JSON, returning empty dict on error."""
    try:
        return json.loads(_KB_PATH.read_text())
    except Exception:
        return {
            "tested_factors": [],
            "failed_factors": [],
            "tested_configs": [],
            "discoveries": [],
            "last_run_date": "never",
        }


@st.cache_data(ttl=30)
def _load_factor_library() -> list[dict[str, Any]]:
    """Load factor library JSON, returning empty list on error."""
    try:
        data = json.loads(_FACTOR_LIBRARY_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


@st.cache_data(ttl=30)
def _load_best_model() -> dict[str, Any]:
    """Load best model config YAML, returning empty dict on error."""
    try:
        import yaml

        return yaml.safe_load(_BEST_MODEL_PATH.read_text()) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------


def _page_knowledge_base() -> None:
    st.header("Knowledge Base")

    kb = _load_kb()

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Run", kb.get("last_run_date", "never"))
    col2.metric("Factors Tested", len(kb.get("tested_factors", [])))
    col3.metric("Factors Failed", len(kb.get("failed_factors", [])))

    discoveries: list[dict] = kb.get("discoveries", [])
    if discoveries:
        st.subheader(f"Discoveries ({len(discoveries)})")
        df = pd.DataFrame(discoveries)
        st.dataframe(df, use_container_width=True)

        if "ic" in df.columns and not df.empty:
            import plotly.express as px

            fig = px.bar(
                df.tail(30),
                x="factor",
                y="ic",
                title="IC by Factor (last 30 discoveries)",
                color="ic",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No discoveries yet. Run `co-optimize` or `mine-factors` to populate.")


def _page_factor_library() -> None:
    st.header("Factor Library")

    factors = _load_factor_library()

    if not factors:
        st.info(
            f"Factor library is empty.\n\nExpected path: `{_FACTOR_LIBRARY_PATH}`\n\n"
            "Run `python scripts/run_rd_agent.py co-optimize --iterations 10` to populate."
        )
        return

    st.metric("Total Factors", len(factors))

    df = pd.DataFrame(factors)
    display_cols = [c for c in ["name", "category", "ic_mean", "icir", "source"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)

    if "ic_mean" in df.columns and not df.empty:
        import plotly.express as px

        fig = px.bar(
            df.sort_values("ic_mean", ascending=False).head(20),
            x="name",
            y="ic_mean",
            color="category",
            title="IC Mean by Factor (top 20)",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def _page_rd_loop_results() -> None:
    st.header("R&D Loop Results")

    kb = _load_kb()
    discoveries: list[dict] = kb.get("discoveries", [])

    if discoveries:
        # Group by date as "R&D loops"
        df = pd.DataFrame(discoveries)
        if "date" in df.columns:
            loops = df.groupby("date").agg(
                factors_discovered=("factor", "count"),
                avg_ic=("ic", "mean"),
                best_ic=("ic", lambda x: x.abs().max()),
            ).reset_index()
            loops.columns = ["Date", "Factors Discovered", "Avg IC", "Best |IC|"]
            st.subheader("R&D Loops by Date")
            st.dataframe(loops, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No R&D loop data found. Run co-optimize to generate results.")

    # Tested model configs
    tested_configs: list[dict] = kb.get("tested_configs", [])
    if tested_configs:
        st.subheader(f"Model Configs Tested ({len(tested_configs)})")
        cfg_df = pd.DataFrame(tested_configs)
        st.dataframe(cfg_df.head(20), use_container_width=True)

    # Best model config
    best_model = _load_best_model()
    if best_model:
        st.subheader("Best Model Config")
        st.json(best_model)
    else:
        st.info(
            f"No best model config found.\n\nExpected path: `{_BEST_MODEL_PATH}`\n\n"
            "Run `python scripts/run_rd_agent.py optimize-model` to generate one."
        )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RD-Agent Dashboard",
    page_icon="🔬",
    layout="wide",
)

st.title("RD-Agent Research Dashboard")

page = st.sidebar.radio(
    "Page",
    ["Knowledge Base", "Factor Library", "R&D Loop Results"],
)

if page == "Knowledge Base":
    _page_knowledge_base()
elif page == "Factor Library":
    _page_factor_library()
else:
    _page_rd_loop_results()
