"""Interactive AML Analyst Dashboard for Money Mule Ring Detection."""
from __future__ import annotations
import json
from pathlib import Path
import networkx as nx
import pandas as pd
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Mule Ring Detector | AML Capstone",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    .badge-cycle { background-color: #ff4b4b; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-fanin { background-color: #0083b8; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-fanout { background-color: #00a08b; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-scatter { background-color: #f39c12; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    reports_dir = Path("reports")
    rings_file = reports_dir / "rings.parquet"
    preds_file = reports_dir / "test_predictions.parquet"
    metrics_file = reports_dir / "metrics.json"

    rings_df = pd.read_parquet(rings_file) if rings_file.exists() else None
    preds_df = pd.read_parquet(preds_file) if preds_file.exists() else None
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else None

    return rings_df, preds_df, metrics


rings_df, preds_df, metrics = load_data()

# Sidebar
st.sidebar.title("🛡️ AML Ring Sentinel")
st.sidebar.caption("Graph-Based Money Mule Ring Detection")
st.sidebar.markdown("---")

if rings_df is None or preds_df is None:
    st.error("Missing pipeline artifacts. Please run the pipeline first:")
    st.code("make ingest features train-baseline rings evaluate")
    st.stop()

# Alert Budget & Filters
st.sidebar.subheader("🎯 Investigator Controls")
max_rings = len(rings_df)
alert_budget = st.sidebar.slider(
    "Alert Budget (Top K Rings)",
    min_value=1,
    max_value=max(max_rings, 1),
    value=min(10, max_rings),
    help="Target number of high-risk cases for AML analyst queue.",
)

pattern_filter = st.sidebar.selectbox(
    "Filter by Pattern",
    options=["All Patterns"] + sorted(list(rings_df["pattern_guess"].unique())),
)

min_score = st.sidebar.slider(
    "Minimum Ring Risk Score",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Capstone Evaluation Note:**\n"
    "Temporal train/val/test split applied with strict causal feature engineering to prevent data leakage."
)

# Header Section
st.title("🕸️ Money Mule Ring Detector")
st.markdown(
    "**Targeting organized mule networks rather than isolated transactions.** "
    "Surfacing high-risk structural topologies (cycles, fan-in, fan-out, scatter-gather) for compliance investigation."
)

# Overview KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Scored Transfers (Test)", f"{len(preds_df):,}")
kpi2.metric("Flagged High-Risk Edges", f"{int((preds_df['score'] >= 0.5).sum()):,}")
kpi3.metric("Candidate Rings Extracted", f"{len(rings_df):,}")
top_pattern = rings_df["pattern_guess"].mode().iloc[0] if not rings_df.empty else "N/A"
kpi4.metric("Dominant Topology", top_pattern.upper())

# Tabbed interface
tab_queue, tab_graph, tab_metrics = st.tabs([
    "📋 Ring Queue (Investigator View)",
    "🔍 Ring Explorer & Network Graph",
    "📈 Capstone Benchmark & Rigor",
])

# ----------------- TAB 1: Ring Queue -----------------
with tab_queue:
    st.subheader("🚨 Prioritized Case Queue")

    filtered = rings_df[rings_df["ring_score"] >= min_score]
    if pattern_filter != "All Patterns":
        filtered = filtered[filtered["pattern_guess"] == pattern_filter]

    ranked = filtered.sort_values("ring_score", ascending=False).head(alert_budget).reset_index(drop=True)

    if ranked.empty:
        st.warning("No candidate rings match the selected filters.")
    else:
        st.markdown(f"Showing **{len(ranked)}** candidate rings within current alert budget:")

        display_df = ranked[[
            "pred_ring_id", "pattern_guess", "ring_score", "n_accounts", "total_amount", "top_reasons"
        ]].copy()
        display_df.columns = ["Ring ID", "Topology", "Risk Score", "Accounts", "Total Volume ($)", "Investigation Summary"]
        display_df["Risk Score"] = display_df["Risk Score"].round(4)
        display_df["Total Volume ($)"] = display_df["Total Volume ($)"].apply(lambda x: f"${x:,.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        selected_id = st.selectbox(
            "Select a Ring ID to explore in Graph View:",
            options=ranked["pred_ring_id"].tolist(),
            format_func=lambda x: f"Ring #{x} — {ranked.loc[ranked['pred_ring_id'] == x, 'pattern_guess'].iloc[0].upper()} (${ranked.loc[ranked['pred_ring_id'] == x, 'total_amount'].iloc[0]:,.2f})"
        )
        st.session_state["selected_ring_id"] = selected_id

# ----------------- TAB 2: Ring Explorer -----------------
with tab_graph:
    cur_id = st.session_state.get("selected_ring_id", rings_df["pred_ring_id"].iloc[0] if not rings_df.empty else 0)
    ring_row = rings_df[rings_df["pred_ring_id"] == cur_id]

    if ring_row.empty:
        st.info("Select a ring from the Queue tab to view its graph.")
    else:
        ring = ring_row.iloc[0]
        st.subheader(f"🔍 Case Details: Ring #{ring['pred_ring_id']}")

        # Ring metadata card
        col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
        col_meta1.markdown(f"**Topology:** `{ring['pattern_guess'].upper()}`")
        col_meta2.markdown(f"**Mule Accounts:** `{ring['n_accounts']}`")
        col_meta3.markdown(f"**Total Volume:** `${ring['total_amount']:,.2f}`")
        duration_hrs = (pd.to_datetime(ring["end_ts"]) - pd.to_datetime(ring["start_ts"])).total_seconds() / 3600.0
        col_meta4.markdown(f"**Activity Span:** `{duration_hrs:.1f} hours`")

        st.info(f"**Analyst Brief:** {ring['top_reasons']}")

        # Subgraph edges
        edge_ids = set(ring["edge_ids"])
        member_edges = preds_df[preds_df["edge_id"].isin(edge_ids)].copy()

        if member_edges.empty:
            # Fallback to endpoints matching
            acct_set = set(ring["accounts"])
            member_edges = preds_df[preds_df["src"].isin(acct_set) & preds_df["dst"].isin(acct_set)].copy()

        # Build PyVis Network Graph
        net = Network(height="450px", width="100%", directed=True, bgcolor="#ffffff", font_color="#333333")
        net.force_atlas_2based()

        # Add Nodes
        nodes = set(member_edges["src"]).union(set(member_edges["dst"]))
        for node in nodes:
            net.add_node(
                node,
                label=str(node),
                title=f"Account: {node}",
                color="#e74c3c" if "CYC" in str(node) or "MULE" in str(node) or "BOSS" in str(node) else "#3498db",
                size=22,
            )

        # Add Edges
        for _, e in member_edges.iterrows():
            amt = float(e.get("amount_sent", 0.0))
            ts_str = str(e.get("ts", ""))
            net.add_edge(
                e["src"],
                e["dst"],
                title=f"Time: {ts_str}<br>Amount: ${amt:,.2f}<br>Risk Score: {e.get('score', 0):.3f}",
                value=max(1, int(amt / 1000)),
                color="#e67e22" if e.get("label", 0) == 1 else "#95a5a6",
                arrows="to",
            )

        # Render Graph HTML
        html_path = Path("reports/selected_ring.html")
        net.save_graph(str(html_path))
        with open(html_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=480)

        # Show Member Transfers Table
        st.markdown("#### Member Transactions in Ring")
        show_cols = [c for c in ["edge_id", "ts", "src", "dst", "amount_sent", "cur_sent", "score", "label"] if c in member_edges.columns]
        st.dataframe(member_edges[show_cols], use_container_width=True, hide_index=True)

# ----------------- TAB 3: Capstone Rigor & Metrics -----------------
with tab_metrics:
    st.subheader("📊 Capstone Evaluation & Benchmark")
    st.markdown("Rigorous dual-level evaluation: **Transaction-level** alert efficiency and **Ring-level** structural recovery.")

    if metrics:
        edge_res = metrics.get("edge_level", {})
        ring_res = metrics.get("ring_level", {})

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("### 1. Transaction-Level Performance")
            m_data = []
            for m_name, m_vals in edge_res.items():
                m_data.append({
                    "Model": "M1 (Raw Tabular)" if "m1" in m_name else "M2 (Graph-Augmented)",
                    "PR-AUC": f"{m_vals.get('pr_auc', 0):.4f}",
                    "Minority F1": f"{m_vals.get('f1', 0):.4f}",
                    "Precision": f"{m_vals.get('precision', 0):.4f}",
                    "Recall": f"{m_vals.get('recall', 0):.4f}",
                    "Precision @ 20": f"{m_vals.get('p_at_20', 0):.4f}",
                })
            st.table(pd.DataFrame(m_data))

        with col_m2:
            st.markdown("### 2. Ring-Level Performance")
            if ring_res:
                st.metric("Ring Recall (Jaccard ≥ 0.5)", f"{ring_res.get('jaccard_0.5', {}).get('ring_recall', 0):.1%}")
                st.metric("Ring Recall (Jaccard ≥ 0.3)", f"{ring_res.get('jaccard_0.3', {}).get('ring_recall', 0):.1%}")
                st.caption(f"Ground Truth Rings in Test: {ring_res.get('n_true_rings_test', 0)} | Extracted Rings: {ring_res.get('n_pred_rings', 0)}")

        st.markdown("---")
        st.markdown("### 3. Recall Breakdown by Laundering Topology")
        pat_data = []
        for ptype, pvals in ring_res.get("pattern_recall", {}).items():
            pat_data.append({
                "Laundering Pattern": ptype,
                "Detected": pvals.get("detected", 0),
                "Total Planted": pvals.get("total", 0),
                "Recall Rate": f"{pvals.get('recall', 0):.1%}",
            })
        if pat_data:
            st.table(pd.DataFrame(pat_data))

    st.markdown("---")
    st.markdown("""
    #### 🛡️ Capstone Methodological Guardrails
    1. **Chronological Splitting:** 60% Train, 20% Validation, 20% Test strictly by timestamp. Random splitting is prohibited to avoid future-leakage.
    2. **Causal Features by Construction:** All rolling counts, balances, and counterparties are calculated strictly on transactions *prior* to timestamp $t$. Verified with automated future-invariance unit testing.
    3. **Threshold Selection on Validation Only:** Decision boundary tuned exclusively on validation F1 before being frozen for test evaluation.
    """)
