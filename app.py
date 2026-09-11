"""
app.py — MITHYA Crypto-Forensic Intelligence Dashboard
========================================================

Production-grade Streamlit dashboard for the MITHYA crypto-forensic triage
engine (Smart India Hackathon 2026).  Features a 4-tab operational
architecture, dynamic sidebar controls, interactive PyVis graph, per-
transaction heuristic inspector, and court-ready export.

Run:
    streamlit run app.py

Dependencies:
    pip install streamlit pyvis pandas scikit-learn
"""

import subprocess
import sys
import tempfile
import os
from datetime import datetime
from pathlib import Path
from io import StringIO
from typing import Optional

import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import streamlit.components.v1 as components
# pyrefly: ignore [missing-import]
from pyvis.network import Network

# Import the AI engine and cross-chain adapter layer
from ml_engine import (
    run_pipeline_from_df,
    is_mixer_transaction,
    compute_peel_chain_disparity,
    compute_fan_in_out,
    compute_fee_rate_urgency,
    score_port_risk,
    _parse_pipe_amounts,
    _count_pipe_elements,
    load_institutional_whitelist,
)
from transaction_adapter import BitcoinCSVAdapter

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MITHYA — Crypto Forensic Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark mode + Professional Cybersecurity Theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* ── Import premium font ───────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global dark theme overrides ───────────────────────────── */
    .stApp {
        background: linear-gradient(145deg, #0a0a0f 0%, #0d1117 40%, #0f0b1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit header/footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Dashboard title ───────────────────────────────────────── */
    .dashboard-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .dashboard-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .dashboard-header p {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* ── Bento metric cards ────────────────────────────────────── */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        padding: 1rem 0;
    }
    .bento-card {
        background: linear-gradient(135deg, rgba(30, 32, 48, 0.9) 0%, rgba(20, 22, 35, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 1rem;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .bento-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
    }
    .bento-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 1rem 1rem 0 0;
    }
    .bento-card.card-blue::before   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .bento-card.card-green::before  { background: linear-gradient(90deg, #22c55e, #4ade80); }
    .bento-card.card-red::before    { background: linear-gradient(90deg, #ef4444, #f87171); }
    .bento-card.card-purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    .bento-card.card-amber::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .bento-card.card-cyan::before   { background: linear-gradient(90deg, #06b6d4, #22d3ee); }

    .bento-card .card-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
    .bento-card .card-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }
    .bento-card .card-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .bento-card.card-blue .card-value   { color: #60a5fa; }
    .bento-card.card-green .card-value  { color: #4ade80; }
    .bento-card.card-red .card-value    { color: #f87171; }
    .bento-card.card-purple .card-value { color: #a78bfa; }
    .bento-card.card-amber .card-value  { color: #fbbf24; }
    .bento-card.card-cyan .card-value   { color: #22d3ee; }
    .bento-card .card-sub {
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 0.4rem;
    }

    /* ── Section titles ────────────────────────────────────────── */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #e5e7eb;
        margin: 1.5rem 0 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title .icon { font-size: 1.3rem; }
    .section-subtitle {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
    }

    /* ── Graph container ───────────────────────────────────────── */
    .graph-container {
        background: linear-gradient(135deg, rgba(30, 32, 48, 0.7) 0%, rgba(15, 17, 28, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 1rem;
        padding: 0.5rem;
        overflow: hidden;
    }

    /* ── Investigation panel ───────────────────────────────────── */
    .investigation-panel {
        background: linear-gradient(135deg, rgba(30, 32, 48, 0.7) 0%, rgba(15, 17, 28, 0.9) 100%);
        border: 1px solid rgba(239, 68, 68, 0.15);
        border-radius: 1rem;
        padding: 1rem;
    }

    /* ── Heuristic telemetry table ─────────────────────────────── */
    .heuristic-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 0.75rem;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
    }
    .heuristic-table th {
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        padding: 0.7rem 1rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .heuristic-table td {
        padding: 0.6rem 1rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.08);
        color: #e5e7eb;
    }
    .heuristic-table tr:last-child td { border-bottom: none; }
    .heuristic-table tr:hover td { background: rgba(99, 102, 241, 0.05); }

    /* ── Inline badges ─────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-red      { background: rgba(239, 68, 68, 0.15); color: #f87171; }
    .badge-green    { background: rgba(34, 197, 94, 0.15);  color: #4ade80; }
    .badge-amber    { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .badge-purple   { background: rgba(139, 92, 246, 0.15); color: #a78bfa; }
    .badge-blue     { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
    .badge-cyan     { background: rgba(6, 182, 212, 0.15);  color: #22d3ee; }

    /* ── TX Detail card ────────────────────────────────────────── */
    .tx-detail-card {
        background: linear-gradient(135deg, rgba(30, 32, 48, 0.9) 0%, rgba(20, 22, 35, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .tx-detail-card .tx-hash {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #a5b4fc;
        word-break: break-all;
    }

    /* ── Divider ───────────────────────────────────────────────── */
    .fancy-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 1.2rem 0;
    }

    /* ── Live-pulse indicator ──────────────────────────────────── */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
    }

    /* ── Upload success banner ─────────────────────────────────── */
    .upload-banner {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 0.75rem;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        color: #4ade80;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* ── Sidebar styling ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #131720 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #9ca3af;
    }

    /* ── Tab styling overrides ─────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 32, 48, 0.5);
        border-radius: 0.5rem 0.5rem 0 0;
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-bottom: none;
        padding: 0.5rem 1.2rem;
        color: #9ca3af;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Streamlit dataframe overrides */
    .stDataFrame { border-radius: 0.75rem; overflow: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Required columns for validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
    "txid", "input_addresses", "output_addresses",
    "input_amounts", "output_amounts", "fee", "script_type", "geo_country",
]

# Path to the Python interpreter in our venv
PYTHON_BIN = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python3")
if not os.path.isfile(PYTHON_BIN):
    PYTHON_BIN = sys.executable  # fallback


# ---------------------------------------------------------------------------
# Sidebar — Dynamic Threat Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 0.8rem 0;">
            <span style="font-size: 2rem;">🛡️</span>
            <h3 style="color: #e5e7eb; margin: 0.3rem 0 0 0; font-family: Inter, sans-serif;">
                MITHYA Control Panel
            </h3>
            <p style="color: #6b7280; font-size: 0.75rem; margin-top: 0.2rem;">
                Crypto-Forensic Triage Engine
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Section 1: Data Source ──
    st.markdown("##### 📂 Data Source")

    uploaded_file = st.file_uploader(
        "Upload Bitcoin Traffic CSV",
        type=["csv"],
        help="Upload a raw CSV with columns: timestamp, src_ip, dst_ip, "
             "src_port, dst_port, txid, input_addresses, output_addresses, "
             "input_amounts, output_amounts, fee, script_type, geo_country",
        key="csv_upload",
    )

    uploaded_whitelist = st.file_uploader(
        "Upload Custom Whitelist CSV",
        type=["csv"],
        help="CSV with columns: address, entity_name, risk_override",
        key="whitelist_upload",
    )

    st.markdown("---")

    # ── Section 2: Model Controls ──
    st.markdown("##### 🎛️ Model Controls")

    contamination = st.slider(
        "Anomaly Sensitivity",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        help="Expected fraction of anomalous transactions. "
             "Higher = more flags, lower = stricter.",
    )

    st.markdown("---")

    # ── Section 3: Live Data Generator ──
    st.markdown("##### ⚡ Synthetic Data Generator")

    gen_records = st.number_input(
        "Total Records",
        min_value=500,
        max_value=5000,
        value=1500,
        step=500,
        help="Number of synthetic transactions to generate.",
    )

    gen_ratio = st.slider(
        "Suspicious Ratio",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
        help="Fraction of transactions that are threat vectors.",
    )

    if st.button("🔄 Regenerate Data", use_container_width=True, type="primary"):
        with st.spinner("Generating synthetic data..."):
            result = subprocess.run(
                [
                    PYTHON_BIN, "generate_data.py",
                    "--total-records", str(gen_records),
                    "--suspicious-ratio", str(gen_ratio),
                ],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__) or ".",
            )
            if result.returncode == 0:
                st.success("Data regenerated!")
                st.rerun()
            else:
                st.error(f"Generator failed:\n{result.stderr}")

    st.markdown("---")

    # ── Section 4: Whitelist Viewer ──
    with st.expander("🏛️ Active Institutional Whitelist"):
        try:
            wl_addrs, wl_labels = load_institutional_whitelist()
            if wl_addrs:
                wl_display = pd.DataFrame([
                    {"Address": addr, "Institution": wl_labels.get(addr, "Unknown")}
                    for addr in sorted(wl_addrs)
                ])
                st.dataframe(wl_display, hide_index=True, use_container_width=True)
            else:
                st.info("No whitelist loaded.")
        except Exception:
            st.info("Whitelist file not found.")

    # ── Section 5: Info ──
    st.markdown(
        """
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2);
                    border-radius: 0.5rem; padding: 0.7rem; margin-top: 0.5rem;">
            <p style="color: #a5b4fc; font-size: 0.7rem; margin: 0; font-weight: 600;">
                ℹ️ ARCHITECTURE
            </p>
            <p style="color: #9ca3af; font-size: 0.68rem; margin: 0.3rem 0 0 0; line-height: 1.4;">
                CSV → Adapter → Graph Builder → Mixer Bypass →
                Change-Address Linking → Wallet Clustering →
                18-Feature Extraction → IsolationForest →
                Institutional Whitelist → XAI Explanations
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loading & pipeline execution
# ---------------------------------------------------------------------------

_adapter = BitcoinCSVAdapter()

# Handle custom whitelist upload
if uploaded_whitelist is not None:
    custom_wl = pd.read_csv(uploaded_whitelist)
    custom_wl.to_csv("institutional_whitelist.csv", index=False)


def run_ai_on_upload(raw_df: pd.DataFrame, cont: float):
    """Run the full AI pipeline via the polymorphic adapter."""
    enriched_df, _model, features = _adapter.run_pipeline_from_df(
        raw_df, contamination=cont,
    )
    return enriched_df, features


# Determine data source and load
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
        if missing:
            st.error(f"❌ **Missing columns:** {', '.join(missing)}")
            st.stop()

        with st.spinner("🧠 AI Engine running — detecting anomalies..."):
            df, features_df = run_ai_on_upload(raw_df, contamination)
            data_source = "uploaded"

        with st.sidebar:
            st.markdown(
                f'<div class="upload-banner">'
                f"✅ Analysed <b>{len(raw_df)}</b> transactions · "
                f"<b>{(df['is_anomaly'].sum())}</b> flagged"
                f"</div>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        import traceback
        st.error(f"❌ **Error:** {e}\n\n```python\n{traceback.format_exc()}\n```")
        st.stop()
elif os.path.isfile("bitcoin_traffic.csv"):
    try:
        raw_df = pd.read_csv("bitcoin_traffic.csv")
        with st.spinner("🧠 AI Engine running on default dataset..."):
            df, features_df = run_ai_on_upload(raw_df, contamination)
            data_source = "default"
    except Exception as e:
        import traceback
        st.error(f"❌ **Error:** {e}\n\n```python\n{traceback.format_exc()}\n```")
        st.stop()
else:
    st.warning("📁 Please upload a CSV or generate synthetic data using the sidebar.")
    st.stop()


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

total_tx = len(df)
flagged_df = df[df["is_anomaly"] == True].copy()
total_flagged = len(flagged_df)
max_risk = df["risk_score"].max() if total_flagged > 0 else 0.0

# Count specific detections
whitelisted_count = df["explanation"].str.contains("Regulated", na=False).sum()
mixer_count = df["attack_type"].eq("CoinJoin_Mixer").sum() if "attack_type" in df.columns else 0
peel_count = df["attack_type"].eq("Peel_Chain").sum() if "attack_type" in df.columns else 0
fanout_count = df["attack_type"].eq("FanOut_Dispersal").sum() if "attack_type" in df.columns else 0
feespike_count = df["attack_type"].eq("Fee_Spike").sum() if "attack_type" in df.columns else 0


# ---------------------------------------------------------------------------
# Dashboard header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="dashboard-header">
        <h1>🛡️ MITHYA — Crypto Forensic Intelligence</h1>
        <p><span class="live-dot"></span>Graph-Aware Anomaly Detection Engine &nbsp;·&nbsp; SIH 2026 Prototype</p>
    </div>
    """,
    unsafe_allow_html=True,
)

source_label = (
    f"📂 Live analysis of uploaded file ({total_tx} tx)"
    if data_source == "uploaded"
    else f"💾 Default dataset: bitcoin_traffic.csv ({total_tx} tx)"
)
source_color = "#4ade80" if data_source == "uploaded" else "#60a5fa"

st.markdown(
    f'<p style="text-align:center; color:{source_color}; font-size:0.78rem; '
    f'font-weight:500; margin-top:-0.5rem;">{source_label}</p>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-TAB ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Overview & Triage",
    "📋 Entity Attribution",
    "🕸️ Graph & Clusters",
    "🔬 Heuristics Inspector",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW & TRIAGE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    # ── Row 1: Core metrics (3 cards) ──
    anomaly_pct = (total_flagged / total_tx * 100) if total_tx > 0 else 0.0

    st.markdown(
        f"""
        <div class="bento-grid">
            <div class="bento-card card-blue">
                <div class="card-icon">📡</div>
                <div class="card-label">Total Transactions</div>
                <div class="card-value">{total_tx:,}</div>
                <div class="card-sub">Bitcoin network traffic analysed</div>
            </div>
            <div class="bento-card card-green">
                <div class="card-icon">🏛️</div>
                <div class="card-label">Whitelisted Entities</div>
                <div class="card-value">{whitelisted_count}</div>
                <div class="card-sub">Institutional transfers cleared</div>
            </div>
            <div class="bento-card card-red">
                <div class="card-icon">🚨</div>
                <div class="card-label">ML Anomalies Flagged</div>
                <div class="card-value">{total_flagged}</div>
                <div class="card-sub">{anomaly_pct:.1f}% anomaly detection rate</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Row 2: Threat vector metrics (3 cards) ──
    st.markdown(
        f"""
        <div class="bento-grid">
            <div class="bento-card card-purple">
                <div class="card-icon">🔀</div>
                <div class="card-label">CoinJoin Mixers</div>
                <div class="card-value">{mixer_count}</div>
                <div class="card-sub">Equal-output anonymity sets detected</div>
            </div>
            <div class="bento-card card-amber">
                <div class="card-icon">⛓️</div>
                <div class="card-label">Peel Chains</div>
                <div class="card-value">{peel_count}</div>
                <div class="card-sub">Micro-peel layering structures</div>
            </div>
            <div class="bento-card card-cyan">
                <div class="card-icon">⚠️</div>
                <div class="card-label">Max Risk Score</div>
                <div class="card-value">{max_risk:.1f}%</div>
                <div class="card-sub">Highest threat confidence level</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Pipeline summary ──
    st.markdown(
        f"""
        <div style="background: rgba(99, 102, 241, 0.06); border: 1px solid rgba(99, 102, 241, 0.15);
                    border-radius: 0.75rem; padding: 1.2rem; color: #d1d5db; font-size: 0.85rem; line-height: 1.6;">
            <b style="color: #a5b4fc;">📊 Pipeline Summary</b><br>
            The engine processed <b style="color: #60a5fa;">{total_tx:,}</b> transactions,
            bypassed <b style="color: #a78bfa;">{mixer_count}</b> mixer anonymity sets,
            detected <b style="color: #fbbf24;">{peel_count}</b> peel-chain structures,
            identified <b style="color: #22d3ee;">{fanout_count}</b> fan-out dispersals and
            <b style="color: #f87171;">{feespike_count}</b> fee-spike urgency hops,
            and cleared <b style="color: #4ade80;">{whitelisted_count}</b> institutional transfers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Threat vector breakdown chart ──
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(
            '<div class="section-title"><span class="icon">📊</span> Threat Vector Distribution</div>',
            unsafe_allow_html=True,
        )
        if "attack_type" in df.columns:
            attack_counts = df["attack_type"].value_counts()
            st.bar_chart(attack_counts, color="#6366f1")

    with col_chart2:
        st.markdown(
            '<div class="section-title"><span class="icon">📈</span> Risk Score Distribution</div>',
            unsafe_allow_html=True,
        )
        if total_flagged > 0:
            st.bar_chart(flagged_df["risk_score"].value_counts(bins=10).sort_index(), color="#ef4444")
        else:
            st.info("No anomalies to display.")

    # ── Export buttons ──
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        csv_export = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Results CSV",
            data=csv_export,
            file_name="mithya_full_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        # Court-ready report
        report_lines = [
            "=" * 70,
            "  MITHYA — Crypto Forensic Triage Report",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Engine: IsolationForest (n_estimators=200, contamination={contamination})",
            "=" * 70,
            "",
            f"  Total Transactions Analysed:  {total_tx:,}",
            f"  Institutional Transfers Cleared: {whitelisted_count}",
            f"  ML Anomalies Flagged:        {total_flagged}",
            f"  CoinJoin Mixers Detected:    {mixer_count}",
            f"  Peel Chains Detected:        {peel_count}",
            f"  Fan-Out Dispersals:          {fanout_count}",
            f"  Fee-Spike Urgency Hops:      {feespike_count}",
            f"  Max Risk Score:              {max_risk:.1f}%",
            "",
            "-" * 70,
            "  FLAGGED TRANSACTIONS",
            "-" * 70,
            "",
        ]

        if total_flagged > 0:
            for _, row in flagged_df.sort_values("risk_score", ascending=False).iterrows():
                report_lines.append(f"  TXID: {row['txid']}")
                report_lines.append(f"  Risk: {row['risk_score']:.1f}%  |  Entity: {row.get('entity_id', 'N/A')}")
                report_lines.append(f"  Source IP: {row['src_ip']}  |  Port: {row['src_port']}")
                report_lines.append(f"  Amount: {row.get('total_amount_btc', 'N/A')} BTC  |  Fee: {row['fee']}")
                report_lines.append(f"  {row['explanation']}")
                report_lines.append("")
        else:
            report_lines.append("  No anomalies detected.")

        report_lines.append("=" * 70)
        report_lines.append("  END OF REPORT")
        report_lines.append("=" * 70)

        report_text = "\n".join(report_lines)
        st.download_button(
            label="📄 Download Court-Ready Report",
            data=report_text.encode("utf-8"),
            file_name="mithya_forensic_report.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: ENTITY ATTRIBUTION & TABLE VIEWER
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        '<div class="section-title"><span class="icon">📋</span> Entity Attribution & Transaction Viewer</div>',
        unsafe_allow_html=True,
    )

    # ── Filters ──
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

    with filter_col1:
        attack_types = ["All"] + sorted(df["attack_type"].unique().tolist()) if "attack_type" in df.columns else ["All"]
        selected_type = st.selectbox("Filter by Attack Type", attack_types)

    with filter_col2:
        risk_range = st.slider(
            "Risk Score Range",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 100.0),
            step=1.0,
        )

    with filter_col3:
        anomalies_only = st.toggle("Anomalies Only", value=False)

    # ── Apply filters ──
    view_df = df.copy()
    if selected_type != "All" and "attack_type" in view_df.columns:
        view_df = view_df[view_df["attack_type"] == selected_type]
    view_df = view_df[
        (view_df["risk_score"] >= risk_range[0]) &
        (view_df["risk_score"] <= risk_range[1])
    ]
    if anomalies_only:
        view_df = view_df[view_df["is_anomaly"] == True]

    st.markdown(
        f'<p class="section-subtitle">Showing {len(view_df):,} of {total_tx:,} transactions</p>',
        unsafe_allow_html=True,
    )

    # ── Build display table ──
    display_cols = [
        "txid", "risk_score", "entity_id", "explanation",
        "src_port", "dst_port", "total_amount_btc", "fee",
        "script_type", "geo_country",
    ]
    if "attack_type" in view_df.columns:
        display_cols.insert(3, "attack_type")

    display_df = view_df[[c for c in display_cols if c in view_df.columns]].copy()

    # Truncate txid for display
    display_df["txid"] = display_df["txid"].apply(
        lambda x: f"{str(x)[:8]}…{str(x)[-6:]}" if len(str(x)) > 16 else str(x)
    )

    display_df = display_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Rename columns for display
    rename_map = {
        "txid": "TX ID",
        "risk_score": "Risk %",
        "entity_id": "Entity",
        "attack_type": "Attack Type",
        "explanation": "XAI Explanation",
        "src_port": "Src Port",
        "dst_port": "Dst Port",
        "total_amount_btc": "Amount (BTC)",
        "fee": "Fee",
        "script_type": "Script",
        "geo_country": "Propagation Node",
    }
    display_df.rename(columns=rename_map, inplace=True)

    # ── Render ──
    st.markdown('<div class="investigation-panel">', unsafe_allow_html=True)

    st.dataframe(
        display_df.style.background_gradient(
            subset=["Risk %"], cmap="YlOrRd", vmin=0, vmax=100
        ).format({"Amount (BTC)": "{:.6f}", "Risk %": "{:.1f}", "Fee": "{:.8f}"}),
        use_container_width=True,
        height=520,
        hide_index=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Download filtered view ──
    if len(view_df) > 0:
        st.download_button(
            label=f"📥 Download Filtered View ({len(view_df)} rows)",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name="mithya_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: GRAPH & CLUSTER VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        """
        <div class="section-title">
            <span class="icon">🕸️</span> Entity / Transaction Network Graph
        </div>
        <p class="section-subtitle">
            Fund flow visualization &nbsp;·&nbsp;
            <span style="color:#ef4444; font-weight:600;">● Red = flagged anomalies</span> &nbsp;·&nbsp;
            <span style="color:#f59e0b; font-weight:600;">◆ Amber = transactions</span> &nbsp;·&nbsp;
            <span style="color:#9b59b6; font-weight:600;">■ Purple = wallet entities</span> &nbsp;·&nbsp;
            <span style="color:#4ade80; font-weight:600;">● Green = regulated</span> &nbsp;·&nbsp;
            <span style="color:#3b82f6; font-weight:600;">● Blue = context nodes</span>
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Graph mode toggle ──
    toggle_col1, toggle_col2 = st.columns([1, 3])
    with toggle_col1:
        threats_only = st.toggle(
            "🔍 Threat-Centric Mode",
            value=True,
            help="ON = show only flagged threats and their direct connections. "
                 "OFF = include surrounding normal traffic for full context.",
        )
    with toggle_col2:
        if threats_only:
            st.markdown(
                '<p style="color:#f87171; font-size:0.82rem; margin-top:0.5rem; font-weight:500;">'
                '🚨 Showing <b>flagged anomalies only</b> — threats and their immediate neighbours</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="color:#60a5fa; font-size:0.82rem; margin-top:0.5rem; font-weight:500;">'
                '📡 Showing <b>all traffic context</b> — threats within broader network</p>',
                unsafe_allow_html=True,
            )

    MAX_CONTEXT_NORMAL = 50
    MAX_CLEAN_SAMPLE = 40

    def build_network_graph(graph_df: pd.DataFrame, threats_only_mode: bool = True) -> str:
        """Build a focused, intelligence-style tripartite network graph."""
        net = Network(
            height="580px",
            width="100%",
            bgcolor="#0d1117",
            font_color="#e5e7eb",
            directed=True,
            notebook=False,
        )

        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -120,
                    "centralGravity": 0.005,
                    "springLength": 280,
                    "springConstant": 0.02,
                    "damping": 0.6,
                    "avoidOverlap": 0.8
                },
                "solver": "forceAtlas2Based",
                "stabilization": { "iterations": 200 },
                "minVelocity": 0.75
            },
            "nodes": {
                "font": { "size": 11, "face": "Inter, sans-serif" },
                "borderWidth": 2,
                "borderWidthSelected": 3
            },
            "edges": {
                "smooth": { "type": "curvedCW", "roundness": 0.12 },
                "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
                "width": 1
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "zoomView": true,
                "dragView": true
            }
        }
        """)

        anom_rows = graph_df[graph_df["is_anomaly"] == True]
        regulated_rows = graph_df[graph_df["explanation"].str.contains("Regulated", na=False)]

        anom_ips = set(anom_rows["src_ip"].unique())
        anom_entities = set()
        if "entity_id" in anom_rows.columns:
            anom_entities = set(anom_rows["entity_id"].unique())

        normal_rows = graph_df[(graph_df["is_anomaly"] == False) & ~graph_df["explanation"].str.contains("Regulated", na=False)]
        ip_mask = normal_rows["src_ip"].isin(anom_ips)
        entity_mask = (
            normal_rows["entity_id"].isin(anom_entities)
            if "entity_id" in normal_rows.columns
            else pd.Series(False, index=normal_rows.index)
        )
        context_candidates = normal_rows[ip_mask | entity_mask]
        context_rows = context_candidates.sort_values("risk_score", ascending=False).head(MAX_CONTEXT_NORMAL)

        if not threats_only_mode:
            already_selected = set(context_rows.index)
            clean_pool = normal_rows[~normal_rows.index.isin(already_selected)]
            clean_sample = clean_pool.sample(n=min(MAX_CLEAN_SAMPLE, len(clean_pool)), random_state=42)
            plot_df = pd.concat([anom_rows, regulated_rows.head(10), context_rows, clean_sample]).drop_duplicates()
        else:
            plot_df = pd.concat([anom_rows, context_rows]).drop_duplicates()

        added_nodes: set = set()

        for _, row in plot_df.iterrows():
            is_anom = bool(row["is_anomaly"])
            is_regulated = "Regulated" in str(row.get("explanation", ""))
            src_ip = str(row["src_ip"])
            txid = str(row["txid"])
            entity = str(row.get("entity_id", "Unknown"))
            risk = row["risk_score"]
            country = row.get("geo_country", "")
            amount = row.get("total_amount_btc", 0)

            # ─── IP node ───
            if src_ip not in added_nodes:
                if is_anom or src_ip in anom_ips:
                    ip_color, ip_size = "#ef4444", 24
                    ip_title = f"⚠️ SUSPICIOUS IP: {src_ip}"
                elif is_regulated:
                    ip_color, ip_size = "#4ade80", 18
                    ip_title = f"✅ REGULATED IP: {src_ip}"
                else:
                    ip_color, ip_size = "#3b82f6", 14
                    ip_title = f"IP: {src_ip}"
                net.add_node(src_ip, label=src_ip, color=ip_color, size=ip_size, shape="dot", title=ip_title)
                added_nodes.add(src_ip)

            # ─── TX node ───
            if txid not in added_nodes:
                tx_label = f"{txid[:6]}…{txid[-4:]}"
                if is_anom:
                    tx_color, tx_size = "#f59e0b", 16
                    tx_title = f"🚨 FLAGGED TX: {txid}\nAmount: {amount} BTC\nRisk: {risk}%"
                elif is_regulated:
                    tx_color, tx_size = "#4ade80", 12
                    tx_title = f"✅ REGULATED TX: {txid}\nAmount: {amount} BTC"
                else:
                    tx_color, tx_size = "#6b7280", 8
                    tx_title = f"TX: {txid}\nAmount: {amount} BTC"
                net.add_node(txid, label=tx_label, color=tx_color, size=tx_size, shape="diamond", title=tx_title)
                added_nodes.add(txid)

            # ─── Entity node ───
            if entity not in added_nodes:
                if is_regulated:
                    ent_color, ent_size = "#4ade80", 28
                    ent_title = f"✅ REGULATED ENTITY: {entity}"
                elif is_anom or entity in anom_entities:
                    ent_color, ent_size = "#9b59b6", 30
                    ent_title = f"⚠️ HIGH-RISK ENTITY: {entity}"
                else:
                    ent_color, ent_size = "#9b59b6", 22
                    ent_title = f"Entity Cluster: {entity}"
                net.add_node(
                    entity, label=entity, color=ent_color, size=ent_size, shape="square",
                    title=ent_title, borderWidth=3,
                    font={"size": 14, "color": "#ffffff", "bold": True},
                )
                added_nodes.add(entity)

            # ─── Edges ───
            if is_anom:
                edge_color, edge_width = "#ef4444", 2.5
            elif is_regulated:
                edge_color, edge_width = "rgba(74, 222, 128, 0.4)", 1.5
            else:
                edge_color, edge_width = "rgba(107, 114, 128, 0.35)", 0.7

            net.add_edge(src_ip, txid, color=edge_color, width=edge_width, title=f"Node: {country}")
            net.add_edge(entity, txid, color=edge_color, width=edge_width, title="Inputs owned by entity")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w")
        net.save_graph(tmp.name)
        with open(tmp.name, "r") as f:
            html_content = f.read()
        return html_content

    st.markdown('<div class="graph-container">', unsafe_allow_html=True)
    graph_html = build_network_graph(df, threats_only_mode=threats_only)
    components.html(graph_html, height=600, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: ADVANCED HEURISTICS INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown(
        '<div class="section-title"><span class="icon">🔬</span> Advanced Heuristics Inspector</div>'
        '<p class="section-subtitle">Select any transaction to inspect its full heuristic telemetry</p>',
        unsafe_allow_html=True,
    )

    # ── Transaction selector ──
    sorted_df = df.sort_values("risk_score", ascending=False)
    tx_options = [
        f"{row['txid'][:12]}… | Risk: {row['risk_score']:.1f}% | {row.get('attack_type', 'N/A')}"
        for _, row in sorted_df.iterrows()
    ]
    txid_map = {opt: row["txid"] for opt, (_, row) in zip(tx_options, sorted_df.iterrows())}

    selected_tx_label = st.selectbox(
        "Select Transaction",
        tx_options[:200],  # limit dropdown for performance
        help="Sorted by risk score (highest first). Select any transaction to inspect.",
    )

    if selected_tx_label:
        selected_txid = txid_map[selected_tx_label]
        tx_row = df[df["txid"] == selected_txid].iloc[0]
        tx_idx = df[df["txid"] == selected_txid].index[0]

        # Get feature values for this transaction
        feat_row = features_df.iloc[tx_idx]

        # ── Transaction detail card ──
        risk_val = tx_row["risk_score"]
        if risk_val >= 70:
            risk_badge = '<span class="badge badge-red">CRITICAL</span>'
        elif risk_val >= 40:
            risk_badge = '<span class="badge badge-amber">ELEVATED</span>'
        elif risk_val > 0:
            risk_badge = '<span class="badge badge-blue">LOW</span>'
        else:
            risk_badge = '<span class="badge badge-green">CLEAR</span>'

        attack_type = tx_row.get("attack_type", "Unknown")
        attack_badge_map = {
            "CoinJoin_Mixer": "badge-purple",
            "Peel_Chain": "badge-amber",
            "FanOut_Dispersal": "badge-cyan",
            "Fee_Spike": "badge-red",
            "Normal_P2P": "badge-blue",
            "Whitelisted_Institutional": "badge-green",
        }
        attack_badge_cls = attack_badge_map.get(attack_type, "badge-blue")

        st.markdown(
            f"""
            <div class="tx-detail-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div>
                        <div style="color: #9ca3af; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;">
                            Transaction Hash
                        </div>
                        <div class="tx-hash">{selected_txid}</div>
                    </div>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        {risk_badge}
                        <span class="badge {attack_badge_cls}">{attack_type}</span>
                        <span style="color: #e5e7eb; font-size: 1.4rem; font-weight: 800;">{risk_val:.1f}%</span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; font-size: 0.82rem;">
                    <div>
                        <div style="color: #9ca3af; font-size: 0.68rem; text-transform: uppercase;">Source IP</div>
                        <div style="color: #e5e7eb; font-family: 'JetBrains Mono', monospace;">{tx_row['src_ip']}</div>
                    </div>
                    <div>
                        <div style="color: #9ca3af; font-size: 0.68rem; text-transform: uppercase;">Ports</div>
                        <div style="color: #e5e7eb;">{tx_row['src_port']} → {tx_row['dst_port']}</div>
                    </div>
                    <div>
                        <div style="color: #9ca3af; font-size: 0.68rem; text-transform: uppercase;">Amount</div>
                        <div style="color: #e5e7eb;">{tx_row.get('total_amount_btc', 0):.6f} BTC</div>
                    </div>
                    <div>
                        <div style="color: #9ca3af; font-size: 0.68rem; text-transform: uppercase;">Entity</div>
                        <div style="color: #e5e7eb;">{tx_row.get('entity_id', 'N/A')}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── XAI Explanation ──
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.15);
                        border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem;">
                <div style="color: #f87171; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                            letter-spacing: 0.06em; margin-bottom: 0.3rem;">
                    🧠 AI Explanation
                </div>
                <div style="color: #d1d5db; font-size: 0.85rem; line-height: 1.5;">
                    {tx_row['explanation']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Heuristic Telemetry Table ──
        st.markdown(
            '<div class="section-title"><span class="icon">📐</span> Heuristic Telemetry</div>',
            unsafe_allow_html=True,
        )

        # Compute per-transaction heuristic values
        peel_val = feat_row.get("peel_chain_disparity", 0)
        fan_in_val = int(feat_row.get("fan_in", 0))
        fan_out_val = int(feat_row.get("fan_out", 0))
        fan_ratio_val = feat_row.get("fan_ratio", 0)
        fee_urgency_val = feat_row.get("fee_rate_urgency", 0)
        zscore_val = feat_row.get("value_zscore", 0)
        zscore_abs = feat_row.get("value_zscore_abs", 0)
        port_risk_val = feat_row.get("port_risk_combined", 0)
        entity_ip_div = feat_row.get("entity_ip_diversity", 0)
        is_micro = int(feat_row.get("is_micro_tx", 0))
        is_mixer = is_mixer_transaction(tx_row)

        def _check(triggered: bool) -> str:
            return '<span style="color: #f87171; font-weight: 700;">✅ TRIGGERED</span>' if triggered else '<span style="color: #6b7280;">—</span>'

        heuristic_rows = f"""
        <tr>
            <td><b>Mixer Signature</b></td>
            <td style="font-family: 'JetBrains Mono', monospace;">{'True' if is_mixer else 'False'}</td>
            <td>equal-output ratio ≥ 50%</td>
            <td>{_check(is_mixer)}</td>
        </tr>
        <tr>
            <td><b>Peel Chain Disparity</b></td>
            <td style="font-family: 'JetBrains Mono', monospace;">{peel_val:.4f}</td>
            <td>&gt; 0.0</td>
            <td>{_check(peel_val > 0)}</td>
        </tr>
        <tr>
            <td><b>Fan-In / Fan-Out</b></td>
            <td style="font-family: 'JetBrains Mono', monospace;">{fan_in_val} → {fan_out_val} (ratio: {fan_ratio_val:.4f})</td>
            <td>ratio &lt; 0.2 &amp; fan_out &gt; 5</td>
            <td>{_check(fan_ratio_val < 0.2 and fan_out_val > 5)}</td>
        </tr>
        <tr>
            <td><b>Fee Rate Urgency</b></td>
            <td style="font-family: 'JetBrains Mono', monospace;">{fee_urgency_val:.6f}</td>
            <td>&gt; 0.05</td>
            <td>{_check(fee_urgency_val > 0.05)}</td>
        </tr>
        <tr>
            <td><b>Value Z-Score</b></td>
            <td style="font-family: 'JetBrains Mono', monospace;">{zscore_val:.4f} (abs: {zscore_abs:.4f})</td>
            <td>|Z| &gt; 2.5</td>
            <td>{_check(zscore_abs > 2.5)}</td>
        </tr>
        <tr>
            <td><b>Port Risk Combined</b></td>
            <td style="font-family: 'JetBrains Mono', monospace;">{port_risk_val:.4f}</td>
            <td>≥ 0.6</td>
            <td>{_check(port_risk_val >= 0.6)}</td>
        </tr>
        <tr>
            <td><b>Entity IP Diversity</b></td>
            <td style="font-family: 'JetBrains Mono', monospace;">{int(entity_ip_div)}</td>
            <td>&gt; 2 IPs</td>
            <td>{_check(entity_ip_div > 2)}</td>
        </tr>
        <tr>
            <td><b>Micro Transaction</b></td>
            <td style="font-family: 'JetBrains Mono', monospace;">{'Yes' if is_micro else 'No'}</td>
            <td>amount &lt; 0.006 BTC</td>
            <td>{_check(is_micro == 1)}</td>
        </tr>
        """

        st.markdown(
            f"""
            <table class="heuristic-table">
                <thead>
                    <tr>
                        <th>Heuristic</th>
                        <th>Value</th>
                        <th>Threshold</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {heuristic_rows}
                </tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

        # ── Address Flow Detail ──
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title"><span class="icon">💰</span> Address & Amount Flow</div>',
            unsafe_allow_html=True,
        )

        addr_col1, addr_col2 = st.columns(2)

        with addr_col1:
            st.markdown("**Input Addresses & Amounts**")
            in_addrs = str(tx_row.get("input_addresses", "")).split("|")
            in_amounts = _parse_pipe_amounts(tx_row.get("input_amounts"))
            in_data = []
            for i, addr in enumerate(in_addrs):
                amt = in_amounts[i] if i < len(in_amounts) else 0.0
                in_data.append({"Address": addr.strip(), "Amount (BTC)": f"{amt:.8f}"})
            if in_data:
                st.dataframe(pd.DataFrame(in_data), hide_index=True, use_container_width=True)

        with addr_col2:
            st.markdown("**Output Addresses & Amounts**")
            out_addrs = str(tx_row.get("output_addresses", "")).split("|")
            out_amounts = _parse_pipe_amounts(tx_row.get("output_amounts"))
            out_data = []
            for i, addr in enumerate(out_addrs):
                amt = out_amounts[i] if i < len(out_amounts) else 0.0
                out_data.append({"Address": addr.strip(), "Amount (BTC)": f"{amt:.8f}"})
            if out_data:
                st.dataframe(pd.DataFrame(out_data), hide_index=True, use_container_width=True)

        st.markdown(
            f"""
            <div style="text-align: center; color: #6b7280; font-size: 0.78rem; margin-top: 0.5rem;">
                Fee: <b style="color: #fbbf24;">{tx_row['fee']:.8f} BTC</b> &nbsp;·&nbsp;
                Script: <b style="color: #a5b4fc;">{tx_row['script_type']}</b> &nbsp;·&nbsp;
                First-Seen Node: <b style="color: #60a5fa;">{tx_row.get('geo_country', 'N/A')}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="fancy-divider"></div>
    <div style="text-align: center; padding: 0.8rem 0 1.5rem 0;">
        <p style="color: #4b5563; font-size: 0.75rem; font-weight: 400;">
            MITHYA — Crypto Forensic Intelligence Engine &nbsp;·&nbsp;
            Graph-Aware IsolationForest &nbsp;·&nbsp;
            18-Feature Behavioral Analysis &nbsp;·&nbsp;
            Built for Smart India Hackathon 2026
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
