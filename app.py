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
import json
import socket
import hashlib
from datetime import datetime
from pathlib import Path
import io
from io import StringIO
from typing import Optional

import dossier

import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import streamlit.components.v1 as components
# pyrefly: ignore [missing-import]
from pyvis.network import Network
# pyrefly: ignore [missing-import]
import altair as alt

# ---------------------------------------------------------------------------
# AIR-GAP SANITIZATION (Strict Offline Enforcement)
# ---------------------------------------------------------------------------
_original_socket = socket.socket
class AirGapSocket(_original_socket):
    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise PermissionError(f"Air-Gap Violation: Outbound connection to {host} blocked.")
        return super().connect(address)
socket.socket = AirGapSocket

# Offline Geo-ASN enrichment (Phase 4)
try:
    from geo_asn import enrich_dataframe as _geo_enrich_df
    _HAS_GEO_ASN = True
except ImportError:
    _HAS_GEO_ASN = False

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
    get_adapter,
)
from transaction_adapter import BitcoinCSVAdapter, correlate_layers

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
# Theme Palettes & CSS — Light / Dark with CSS Variables
# ---------------------------------------------------------------------------

THEME_LIGHT = {
    "bg": "#F8FAFC",
    "bg2": "#FFFFFF",
    "card": "#FFFFFF",
    "card_border": "#E2E8F0",
    "text": "#0F172A",
    "muted": "#475569",
    "accent": "#0F172A",
    "accent2": "#0D9488",
    "card_bg_grad1": "rgba(255,255,255,0.95)",
    "card_bg_grad2": "rgba(248,250,252,1)",
    "sidebar_bg1": "#F1F5F9",
    "sidebar_bg2": "#E2E8F0",
    "sidebar_border": "#CBD5E1",
    "tab_bg": "rgba(241,245,249,0.8)",
    "tab_border": "#CBD5E1",
    "tab_active_bg": "rgba(13,148,136,0.1)",
    "tab_active_text": "#0D9488",
    "tab_active_border": "#0D9488",
    "tab_text": "#334155",
    "divider": "rgba(15,23,42,0.12)",
    "badge_alpha": "0.12",
    "graph_bgcolor": "#F8FAFC",
    "graph_fontcolor": "#0F172A",
    "chart_bg": "#FFFFFF",
    "chart_text": "#0F172A",
    "chart_grid": "#E2E8F0",
    "heuristic_th_bg": "rgba(13,148,136,0.08)",
    "heuristic_th_color": "#0D9488",
    "heuristic_td_color": "#0F172A",
    "heuristic_td_border": "rgba(13,148,136,0.1)",
    "df_bg": "#FFFFFF",
    "df_text": "#0F172A",
}

THEME_DARK = {
    "bg": "#0D1117",
    "bg2": "#0D1117",
    "card": "#161B22",
    "card_border": "#30363D",
    "text": "#E6EDF3",
    "muted": "#9DA7B3",
    "accent": "#E6EDF3",
    "accent2": "#2DD4BF",
    "card_bg_grad1": "rgba(22,27,34,0.95)",
    "card_bg_grad2": "rgba(13,17,23,1)",
    "sidebar_bg1": "#0D1117",
    "sidebar_bg2": "#131720",
    "sidebar_border": "#21262D",
    "tab_bg": "rgba(22,27,34,0.7)",
    "tab_border": "#30363D",
    "tab_active_bg": "rgba(45,212,191,0.1)",
    "tab_active_text": "#2DD4BF",
    "tab_active_border": "#2DD4BF",
    "tab_text": "#9DA7B3",
    "divider": "rgba(45,212,191,0.2)",
    "badge_alpha": "0.15",
    "graph_bgcolor": "#0D1117",
    "graph_fontcolor": "#E6EDF3",
    "chart_bg": "#161B22",
    "chart_text": "#E6EDF3",
    "chart_grid": "#30363D",
    "heuristic_th_bg": "rgba(45,212,191,0.12)",
    "heuristic_th_color": "#2DD4BF",
    "heuristic_td_color": "#E6EDF3",
    "heuristic_td_border": "rgba(45,212,191,0.08)",
    "df_bg": "#161B22",
    "df_text": "#E6EDF3",
}

# Initialise theme state (default = light)
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

def toggle_theme():
    st.session_state["dark_mode"] = st.session_state["theme_switch"]


def _get_palette() -> dict:
    return THEME_DARK if st.session_state.get("dark_mode", False) else THEME_LIGHT


def _build_css(p: dict) -> str:
    """Return the full <style> block using CSS variables from the chosen palette."""
    return f"""
<style>
    /* ── CSS custom properties (theme tokens) ────────────────── */
    :root {{
        --bg: {p['bg']};
        --bg2: {p['bg2']};
        --card: {p['card']};
        --card-border: {p['card_border']};
        --text: {p['text']};
        --muted: {p['muted']};
        --accent: {p['accent']};
        --accent2: {p['accent2']};
    }}

    /* ── Global theme ──────────────────────────────────────────── */
    .stApp {{
        background: var(--bg) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: var(--text);
    }}

    /* Hide Streamlit chrome — keep header visible for sidebar button */
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}

    /* Sidebar expand/collapse always reachable */
    button[data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        pointer-events: auto !important;
    }}

    /* ── Bento metric cards ────────────────────────────────────── */
    .bento-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        padding: 1rem 0;
    }}
    .bento-card {{
        background: linear-gradient(135deg, {p['card_bg_grad1']} 0%, {p['card_bg_grad2']} 100%);
        border: 1px solid var(--card-border);
        border-radius: 1rem;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .bento-card:hover {{
        border-color: var(--accent2);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(13,148,136,0.10);
    }}
    .bento-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 1rem 1rem 0 0;
    }}
    .bento-card.card-blue::before   {{ background: linear-gradient(90deg, #3b82f6, #60a5fa); }}
    .bento-card.card-green::before  {{ background: linear-gradient(90deg, #22c55e, #4ade80); }}
    .bento-card.card-red::before    {{ background: linear-gradient(90deg, #ef4444, #f87171); }}
    .bento-card.card-purple::before {{ background: linear-gradient(90deg, #8b5cf6, #a78bfa); }}
    .bento-card.card-amber::before  {{ background: linear-gradient(90deg, #f59e0b, #fbbf24); }}
    .bento-card.card-cyan::before   {{ background: linear-gradient(90deg, #06b6d4, #22d3ee); }}

    .bento-card .card-icon {{ font-size: 1.4rem; margin-bottom: 0.4rem; }}
    .bento-card .card-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.3rem;
    }}
    .bento-card .card-value {{
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1;
    }}
    .bento-card.card-blue .card-value   {{ color: #3b82f6; }}
    .bento-card.card-green .card-value  {{ color: #22c55e; }}
    .bento-card.card-red .card-value    {{ color: #ef4444; }}
    .bento-card.card-purple .card-value {{ color: #8b5cf6; }}
    .bento-card.card-amber .card-value  {{ color: #f59e0b; }}
    .bento-card.card-cyan .card-value   {{ color: #06b6d4; }}
    .bento-card .card-sub {{
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 0.4rem;
    }}

    /* ── Section titles ────────────────────────────────────────── */
    .section-title {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text);
        margin: 1.5rem 0 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .section-title .icon {{ font-size: 1.3rem; }}
    .section-subtitle {{
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
    }}

    /* ── Graph container ───────────────────────────────────────── */
    .graph-container {{
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 1rem;
        padding: 0.5rem;
        overflow: hidden;
    }}

    /* ── Investigation panel ───────────────────────────────────── */
    .investigation-panel {{
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 1rem;
        padding: 1rem;
    }}

    /* ── Heuristic telemetry table ─────────────────────────────── */
    .heuristic-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 0.75rem;
        overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 0.82rem;
    }}
    .heuristic-table th {{
        background: {p['heuristic_th_bg']};
        color: {p['heuristic_th_color']};
        padding: 0.7rem 1rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .heuristic-table td {{
        padding: 0.6rem 1rem;
        border-bottom: 1px solid {p['heuristic_td_border']};
        color: {p['heuristic_td_color']};
    }}
    .heuristic-table tr:last-child td {{ border-bottom: none; }}
    .heuristic-table tr:hover td {{ background: rgba(13,148,136,0.05); }}

    /* ── Inline badges ─────────────────────────────────────────── */
    .badge {{
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .badge-red      {{ background: rgba(239, 68, 68, {p['badge_alpha']}); color: #ef4444; }}
    .badge-green    {{ background: rgba(34, 197, 94, {p['badge_alpha']});  color: #22c55e; }}
    .badge-amber    {{ background: rgba(245, 158, 11, {p['badge_alpha']}); color: #f59e0b; }}
    .badge-purple   {{ background: rgba(139, 92, 246, {p['badge_alpha']}); color: #8b5cf6; }}
    .badge-blue     {{ background: rgba(59, 130, 246, {p['badge_alpha']}); color: #3b82f6; }}
    .badge-cyan     {{ background: rgba(6, 182, 212, {p['badge_alpha']});  color: #06b6d4; }}

    /* ── TX Detail card ────────────────────────────────────────── */
    .tx-detail-card {{
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}
    .tx-detail-card .tx-hash {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
        font-size: 0.82rem;
        color: var(--accent2);
        word-break: break-all;
    }}

    /* ── Divider ───────────────────────────────────────────────── */
    .fancy-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {p['divider']}, transparent);
        margin: 1.2rem 0;
    }}

    /* ── Live-pulse indicator ──────────────────────────────────── */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}
    .live-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
    }}

    /* ── Upload success banner ─────────────────────────────────── */
    .upload-banner {{
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 0.75rem;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        color: #22c55e;
        font-size: 0.82rem;
        font-weight: 500;
    }}

    /* ── Sidebar styling ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {p['sidebar_bg1']} 0%, {p['sidebar_bg2']} 100%);
        border-right: 1px solid {p['sidebar_border']};
    }}
    section[data-testid="stSidebar"] .stMarkdown p {{
        color: var(--muted);
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stFileUploader label,
    section[data-testid="stSidebar"] .stToggle label,
    section[data-testid="stSidebar"] summary {{
        color: var(--text) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
        background: var(--card) !important;
        border-color: var(--card-border) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section span,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section small {{
        color: var(--muted) !important;
    }}

    /* ── Tab styling overrides ─────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {p['tab_bg']};
        border-radius: 0.5rem 0.5rem 0 0;
        border: 1px solid {p['tab_border']};
        border-bottom: none;
        padding: 0.5rem 1.2rem;
        color: {p['tab_text']};
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {p['tab_active_bg']};
        color: {p['tab_active_text']};
        border-color: {p['tab_active_border']};
    }}

    /* Streamlit dataframe overrides */
    .stDataFrame {{ border-radius: 0.75rem; overflow: hidden; }}

    /* ── Streamlit widget text colour overrides ────────────────── */
    .stSelectbox label, .stMultiSelect label, .stTextInput label,
    .stTextArea label, .stDateInput label, .stTimeInput label {{
        color: var(--text) !important;
    }}
    .stMarkdown, .stMarkdown p {{ color: var(--text); }}

    /* ── Download / action buttons ─────────────────────────────── */
    .stDownloadButton button, .stButton button {{
        background: var(--card) !important;
        color: var(--text) !important;
        border: 1px solid var(--card-border) !important;
        transition: all 0.2s ease;
    }}
    .stDownloadButton button:hover, .stButton button:hover {{
        border-color: var(--accent2) !important;
        box-shadow: 0 2px 12px rgba(13,148,136,0.15);
    }}
    /* Primary buttons keep teal */
    .stButton button[data-testid="stBaseButton-primary"] {{
        background: var(--accent2) !important;
        color: #FFFFFF !important;
        border-color: var(--accent2) !important;
    }}

    /* ── MITHYA professional header ────────────────────────────── */
    .mithya-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 1.5rem;
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 1rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
        gap: 1rem;
    }}
    .mithya-header-left {{
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    .mithya-header-title {{
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: var(--accent);
        line-height: 1;
        margin: 0;
    }}
    .mithya-header-tagline {{
        font-size: 0.82rem;
        color: var(--muted);
        margin: 0.2rem 0 0 0;
        font-weight: 400;
    }}
    .mithya-header-badges {{
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }}
    .mithya-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid var(--card-border);
        color: var(--muted);
        background: var(--bg);
    }}
    @media (max-width: 640px) {{
        .mithya-header {{
            flex-direction: column;
            align-items: flex-start;
        }}
    }}
</style>
"""


def _inject_theme_css():
    """Inject theme CSS based on session state."""
    palette = _get_palette()
    st.markdown(_build_css(palette), unsafe_allow_html=True)


_inject_theme_css()


# ---------------------------------------------------------------------------
# Required columns for validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
    "txid", "input_addresses", "output_addresses",
    "input_amounts", "output_amounts", "fee", "script_type", "geo_country",
]
# ASN is enriched offline after pipeline run (Phase 4)
ASN_COLUMN = "asn"

# Path to the Python interpreter in our venv
PYTHON_BIN = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python3")
if not os.path.isfile(PYTHON_BIN):
    PYTHON_BIN = sys.executable  # fallback

# Default dataset — generate_data.py writes to this file
DEFAULT_DATASET = "synthetic_transactions.csv"
_FALLBACK_DATASET = "bitcoin_traffic.csv"


def _display_geo(value, kind="geo"):
    """Return 'Unknown' for sentinel / missing geo-country or ASN values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    s = str(value).strip()
    if kind == "geo" and s in ("", "XX", "N/A"):
        return "Unknown"
    if kind == "asn" and s in ("", "AS0", "N/A"):
        return "Unknown"
    return s


# ---------------------------------------------------------------------------
# Sidebar — Dynamic Threat Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    # Dark mode toggle — very first sidebar item
    dark_mode = st.toggle(
        "🌙 Dark Mode" if st.session_state["dark_mode"] else "☀️ Light Mode",
        value=st.session_state["dark_mode"],
        key="theme_switch",
        on_change=toggle_theme
    )
    # Re-inject CSS after toggle change
    _inject_theme_css()
    p = _get_palette()

    st.markdown(
        f"""
        <div style="text-align: center; padding: 0.8rem 0;">
            <span style="font-size: 2rem;">🛡️</span>
            <h3 style="color: {p['text']}; margin: 0.3rem 0 0 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                MITHYA Control Panel
            </h3>
            <p style="color: {p['muted']}; font-size: 0.75rem; margin-top: 0.2rem;">
                Crypto-Forensic Triage Engine
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Section 1: Data Source ──
    st.markdown("##### 📂 Data Source")

    upload_mode = st.radio(
        "Upload mode",
        ["Single merged file", "Separate network + blockchain files"],
        index=0,
        key="upload_mode",
        help="Choose whether to upload a single pre-merged transaction file or separate network-telemetry and blockchain CSV files that will be correlated on txid.",
    )

    uploaded_file = None
    uploaded_net_file = None
    uploaded_chain_file = None

    if upload_mode == "Single merged file":
        uploaded_file = st.file_uploader(
            "Upload Transaction Data",
            type=["csv", "json", "xml"],
            help="Upload raw transactions. The appropriate adapter will be automatically selected based on file extension.",
            key="data_upload",
        )
        if uploaded_file:
            st.info(f"✅ Currently using: **{uploaded_file.name}**")
        else:
            st.info("📄 Currently using default: **synthetic_transactions.csv**")
    else:
        uploaded_net_file = st.file_uploader(
            "Upload Network Telemetry CSV",
            type=["csv"],
            help="CSV with columns: timestamp, src_ip, dst_ip, src_port, dst_port, txid",
            key="net_upload",
        )
        uploaded_chain_file = st.file_uploader(
            "Upload Blockchain Transactions CSV",
            type=["csv"],
            help="CSV with columns: txid, timestamp, input_addresses, output_addresses, input_amounts, output_amounts, fee, script_type",
            key="chain_upload",
        )
        if uploaded_net_file and uploaded_chain_file:
            st.info(f"✅ Net: **{uploaded_net_file.name}** · Chain: **{uploaded_chain_file.name}**")
        elif uploaded_net_file or uploaded_chain_file:
            st.warning("⚠️ Upload both files for correlation.")
        else:
            st.info("📄 Using defaults: **sample_data/network_telemetry.csv** + **sample_data/blockchain_tx.csv**")

    uploaded_whitelist = st.file_uploader(
        "Upload Custom Whitelist CSV",
        type=["csv"],
        help="CSV with columns: address, entity_name, risk_override",
        key="whitelist_upload",
    )
    if uploaded_whitelist:
        st.info(f"✅ Currently using: **{uploaded_whitelist.name}**")
    else:
        st.info("📄 Currently using default: **institutional_whitelist.csv**")

    uploaded_watchlist = st.file_uploader(
        "Upload Watchlist CSV",
        type=["csv"],
        help="CSV with columns: indicator, type, label, severity",
        key="watchlist_upload",
    )
    if uploaded_watchlist:
        st.info(f"✅ Currently using: **{uploaded_watchlist.name}**")
    else:
        st.info("📄 Currently using default: **sample_data/watchlist.csv**")

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

    if st.button("🔄 Regenerate Data", width="stretch", type="primary"):
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

    # ── Section 4: Whitelist & Watchlist Viewers ──
    with st.expander("🏛️ Active Institutional Whitelist"):
        try:
            wl_addrs, wl_labels = load_institutional_whitelist()
            if wl_addrs:
                wl_display = pd.DataFrame([
                    {"Address": addr, "Institution": wl_labels.get(addr, "Unknown")}
                    for addr in sorted(wl_addrs)
                ])
                st.dataframe(wl_display, hide_index=True, width="stretch")
            else:
                st.info("No whitelist loaded.")
        except Exception:
            st.info("Whitelist file not found.")

    with st.expander("🚫 Active Watchlist"):
        try:
            from ml_engine import load_watchlist
            wl_data = load_watchlist()
            all_entries = []
            for t, entries in wl_data.items():
                for e in entries:
                    all_entries.append({
                        "Indicator": e.get("indicator", ""),
                        "Type": t,
                        "Label": e.get("label", ""),
                        "Severity": e.get("severity", 0.0),
                    })
            if all_entries:
                st.dataframe(pd.DataFrame(all_entries), hide_index=True, width="stretch")
            else:
                st.info("No watchlist loaded.")
        except Exception:
            st.info("Watchlist file not found.")

    # ── Section 5: Info ──
    st.markdown(
        f"""
        <div style="background: rgba(13,148,136,0.06); border: 1px solid rgba(13,148,136,0.2);
                    border-radius: 0.5rem; padding: 0.7rem; margin-top: 0.5rem;">
            <p style="color: {p['accent2']}; font-size: 0.7rem; margin: 0; font-weight: 600;">
                ℹ️ ARCHITECTURE
            </p>
            <p style="color: {p['muted']}; font-size: 0.68rem; margin: 0.3rem 0 0 0; line-height: 1.4;">
                CSV → Adapter → Graph Builder → Mixer Bypass →
                Change-Address Linking → Wallet Clustering →
                21-Feature Extraction → IsolationForest →
                Institutional Whitelist → XAI Explanations
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 6: Geo DB Status Footer ──
    try:
        import geo_asn
        _g_stat = geo_asn.db_status()
        _geo_db_label = _g_stat.get("database_in_use")
        if _geo_db_label and _geo_db_label.lower() != "none":
            _geo_footer_txt = f"Geo DB: {_geo_db_label}"
        else:
            _geo_footer_txt = "Geo DB: none"
    except Exception:
        _geo_footer_txt = "Geo DB: none"

    st.markdown(
        f'<div style="text-align: center; margin-top: 0.8rem; font-size: 0.74rem; color: {p["muted"]};">'
        f'{_geo_footer_txt}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loading & pipeline execution
# ---------------------------------------------------------------------------

# Handle custom whitelist upload
if uploaded_whitelist is not None:
    custom_wl = pd.read_csv(uploaded_whitelist)
    custom_wl.to_csv("institutional_whitelist.csv", index=False)

# Handle custom watchlist upload
if uploaded_watchlist is not None:
    custom_wl_bad = pd.read_csv(uploaded_watchlist)
    os.makedirs("sample_data", exist_ok=True)
    custom_wl_bad.to_csv(os.path.join("sample_data", "watchlist.csv"), index=False)


def run_ai_on_upload(adapter, raw_df: pd.DataFrame, cont: float):
    """Run the full AI pipeline via the polymorphic adapter, then enrich with offline Geo-ASN."""
    import time
    _t_start = time.perf_counter()

    if "geo_country" not in raw_df.columns:
        raw_df["geo_country"] = "XX"
    if hasattr(adapter, "REQUIRED_COLUMNS"):
        adapter.REQUIRED_COLUMNS = [c for c in adapter.REQUIRED_COLUMNS if c != "geo_country"]

    enriched_df, _model, features = adapter.run_pipeline_from_df(
        raw_df, contamination=cont,
    )
    # Phase 4: offline ASN enrichment (no network I/O)
    if _HAS_GEO_ASN and "src_ip" in enriched_df.columns:
        enriched_df = _geo_enrich_df(enriched_df, ip_col="src_ip", inplace=False)
    elif "asn" not in enriched_df.columns:
        enriched_df["asn"] = "AS0"

    # Ensure behavioural clustering & graph communities are present
    if "behaviour_cluster" not in enriched_df.columns or "graph_community" not in enriched_df.columns:
        try:
            import clustering
            enriched_df, _ = clustering.apply_clustering_to_transactions(enriched_df, features)
        except Exception as _cl_err:
            pass

    _elapsed = time.perf_counter() - _t_start
    st.session_state["pipeline_elapsed_s"] = _elapsed
    st.session_state["pipeline_tx_count"] = len(enriched_df)

    return enriched_df, features


import hashlib

# Determine data source and load
file_hash = "N/A"
ingest_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
correlation_stats = None   # Populated only in separate-file mode
_default_path = DEFAULT_DATASET if os.path.isfile(DEFAULT_DATASET) else _FALLBACK_DATASET

# ── Helper: load separate files and correlate ──
def _load_separate_files(net_source, chain_source, is_upload=False):
    """Load network + blockchain CSVs, correlate on txid, and return (merged_df, adapter, hash, corr_stats)."""
    if is_upload:
        net_bytes = net_source.getvalue()
        chain_bytes = chain_source.getvalue()
        combined_hash = hashlib.sha256(net_bytes + chain_bytes).hexdigest()
        net_df = pd.read_csv(io.BytesIO(net_bytes))
        chain_df = pd.read_csv(io.BytesIO(chain_bytes))
    else:
        with open(net_source, "rb") as f1, open(chain_source, "rb") as f2:
            b1, b2 = f1.read(), f2.read()
        combined_hash = hashlib.sha256(b1 + b2).hexdigest()
        net_df = pd.read_csv(net_source)
        chain_df = pd.read_csv(chain_source)

    merged, corr_stats = correlate_layers(net_df, chain_df)
    if "geo_country" not in merged.columns:
        merged["geo_country"] = "XX"

    adapter = BitcoinCSVAdapter()
    if hasattr(adapter, "REQUIRED_COLUMNS"):
        adapter.REQUIRED_COLUMNS = [c for c in adapter.REQUIRED_COLUMNS if c != "geo_country"]

    # Run validation/report on the merged result
    merged = adapter._validate_and_build_report(merged, source=None, fmt="CSV (correlated)")
    if "geo_country" not in merged.columns:
        merged["geo_country"] = "XX"

    adapter.last_report["sha256"] = combined_hash
    adapter.last_report["correlation"] = corr_stats
    return merged, adapter, combined_hash, corr_stats


if upload_mode == "Separate network + blockchain files" and (uploaded_net_file or uploaded_chain_file):
    # ── Separate-file mode ──
    if uploaded_net_file and uploaded_chain_file:
        try:
            raw_df, adapter, file_hash, correlation_stats = _load_separate_files(
                uploaded_net_file, uploaded_chain_file, is_upload=True,
            )
            with st.spinner("🧠 AI Engine running — detecting anomalies (correlated mode)..."):
                df, features_df = run_ai_on_upload(adapter, raw_df, contamination)
                data_source = "uploaded_separate"

            with st.sidebar:
                _p = _get_palette()
                st.markdown(
                    f'<div class="upload-banner">'
                    f"✅ Correlated <b>{correlation_stats['matched']}</b> txids · "
                    f"<b>{(df['is_anomaly'].sum())}</b> flagged"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            import traceback
            st.error(f"❌ **Error:** {e}\n\n```python\n{traceback.format_exc()}\n```")
            st.stop()
    else:
        st.warning("⚠️ Please upload **both** the network telemetry and blockchain files.")
        st.stop()

elif uploaded_file is not None:
    try:
        # Calculate SHA-256 of uploaded file for data provenance
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Save to a temporary file so the adapter can determine format by extension
        ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            adapter = get_adapter(tmp_path)
            raw_df = adapter.load(tmp_path)
        finally:
            os.remove(tmp_path)

        with st.spinner("🧠 AI Engine running — detecting anomalies..."):
            df, features_df = run_ai_on_upload(adapter, raw_df, contamination)
            data_source = "uploaded"

        with st.sidebar:
            _p = _get_palette()
            st.markdown(
                f'<div class="upload-banner">'
                f"✅ Analysed <b>{len(raw_df)}</b> transactions · "
                f"<b>{(df['is_anomaly'].sum())}</b> flagged"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:0.75rem; color:{_p["muted"]}; padding:10px; background:{_p["card"]}; border-radius:4px; margin-bottom:10px; border:1px solid {_p["card_border"]};">'
                f'<b>🔒 SHA-256 Source Hash:</b><br/><code>{file_hash[:16]}...{file_hash[-16:]}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        import traceback
        st.error(f"❌ **Error:** {e}\n\n```python\n{traceback.format_exc()}\n```")
        st.stop()
else:
    # ── Default dataset loading ──
    # In separate mode without uploads, try the split files first
    _NET_DEFAULT = os.path.join("sample_data", "network_telemetry.csv")
    _CHAIN_DEFAULT = os.path.join("sample_data", "blockchain_tx.csv")

    if upload_mode == "Separate network + blockchain files" and os.path.isfile(_NET_DEFAULT) and os.path.isfile(_CHAIN_DEFAULT):
        try:
            raw_df, adapter, file_hash, correlation_stats = _load_separate_files(
                _NET_DEFAULT, _CHAIN_DEFAULT, is_upload=False,
            )
            with st.spinner("🧠 AI Engine running on default split dataset (correlated)..."):
                df, features_df = run_ai_on_upload(adapter, raw_df, contamination)
                data_source = "default_separate"

            with st.sidebar:
                _p = _get_palette()
                st.markdown(
                    f'<div style="font-size:0.75rem; color:{_p["muted"]}; padding:10px; background:{_p["card"]}; border-radius:4px; margin-bottom:10px; border:1px solid {_p["card_border"]};">'
                    f'<b>🔒 SHA-256 Source Hash (Correlated):</b><br/><code>{file_hash[:16]}...{file_hash[-16:]}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            import traceback
            st.error(f"❌ **Error:** {e}\n\n```python\n{traceback.format_exc()}\n```")
            st.stop()
    else:
        # Standard single-file default path
        _default_path = DEFAULT_DATASET if os.path.isfile(DEFAULT_DATASET) else _FALLBACK_DATASET
        if os.path.isfile(_default_path):
            try:
                # Calculate SHA-256 of default dataset
                with open(_default_path, "rb") as f:
                    file_bytes = f.read()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()

                adapter = get_adapter(_default_path)
                raw_df = adapter.load(_default_path)
                with st.spinner("🧠 AI Engine running on default dataset..."):
                    df, features_df = run_ai_on_upload(adapter, raw_df, contamination)
                    data_source = "default"

                with st.sidebar:
                    _p = _get_palette()
                    st.markdown(
                        f'<div style="font-size:0.75rem; color:{_p["muted"]}; padding:10px; background:{_p["card"]}; border-radius:4px; margin-bottom:10px; border:1px solid {_p["card_border"]};">'
                        f'<b>🔒 SHA-256 Source Hash (Default CSV):</b><br/><code>{file_hash[:16]}...{file_hash[-16:]}</code>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                import traceback
                st.error(f"❌ **Error:** {e}\n\n```python\n{traceback.format_exc()}\n```")
                st.stop()
        else:
            st.warning("📁 Please upload a CSV or generate synthetic data using the sidebar.")
            st.stop()
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
mixer_count = df["detected_type"].eq("CoinJoin_Mixer").sum() if "detected_type" in df.columns else 0
peel_count = df["detected_type"].eq("Peel_Chain").sum() if "detected_type" in df.columns else 0
fanout_count = df["detected_type"].eq("FanOut_Dispersal").sum() if "detected_type" in df.columns else 0
feespike_count = df["detected_type"].eq("Fee_Spike").sum() if "detected_type" in df.columns else 0

# Actionable-leads funnel
cleared_count = int(((df["is_anomaly"] == True) & (df.get("whitelisted_side", pd.Series(dtype="object")).notna())).sum()) if "whitelisted_side" in df.columns else 0
actionable_count = int(((df["is_anomaly"] == True) & (~df.get("whitelisted_side", pd.Series(dtype="object")).notna())).sum()) if "whitelisted_side" in df.columns else total_flagged

# Risk tier counts
tier_critical = int(df["risk_tier"].eq("Critical").sum()) if "risk_tier" in df.columns else 0
tier_high = int(df["risk_tier"].eq("High").sum()) if "risk_tier" in df.columns else 0
tier_medium = int(df["risk_tier"].eq("Medium").sum()) if "risk_tier" in df.columns else 0
tier_low = int(df["risk_tier"].eq("Low").sum()) if "risk_tier" in df.columns else 0


# ---------------------------------------------------------------------------
# Dashboard header — Professional SVG logo + badge bar
# ---------------------------------------------------------------------------

_hp = _get_palette()

_MITHYA_SVG_LOGO = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="56" height="56" viewBox="0 0 56 56" fill="none">
  <!-- Shield body -->
  <path d="M28 2 L50 14 L50 32 C50 42 40 50 28 54 C16 50 6 42 6 32 L6 14 Z"
        fill="{_hp['accent2']}" fill-opacity="0.12" stroke="{_hp['accent2']}" stroke-width="2"/>
  <!-- Blockchain nodes -->
  <circle cx="28" cy="18" r="4" fill="{_hp['accent2']}"/>
  <circle cx="18" cy="34" r="4" fill="{_hp['accent']}"/>
  <circle cx="38" cy="34" r="4" fill="{_hp['accent']}"/>
  <!-- Connecting lines -->
  <line x1="28" y1="22" x2="18" y2="30" stroke="{_hp['accent2']}" stroke-width="1.8" stroke-linecap="round"/>
  <line x1="28" y1="22" x2="38" y2="30" stroke="{_hp['accent2']}" stroke-width="1.8" stroke-linecap="round"/>
  <line x1="18" y1="34" x2="38" y2="34" stroke="{_hp['accent2']}" stroke-width="1.8" stroke-linecap="round" stroke-dasharray="3 2"/>
</svg>
"""

_proc_seconds = float(st.session_state.get("pipeline_elapsed_s", 1.2))
_proc_n = int(st.session_state.get("pipeline_tx_count", total_tx))
_proc_text = f"Processed {_proc_n} tx in {_proc_seconds:.1f}s"

st.markdown(
    f"""
<div class="mithya-header">
<div class="mithya-header-left">
{_MITHYA_SVG_LOGO}
<div>
<div class="mithya-header-title">MITHYA</div>
<p class="mithya-header-tagline">Offline AI Forensics for Bitcoin Transaction Traffic</p>
</div>
</div>
<div class="mithya-header-badges">
<span class="mithya-pill">🟢 Offline</span>
<span class="mithya-pill">🔒 Air-Gapped</span>
<span class="mithya-pill">{_proc_text}</span>
<span class="mithya-pill">SIH 2026</span>
</div>
</div>
    """,
    unsafe_allow_html=True,
)

if data_source == "uploaded_separate":
    source_label = f"📂 Live analysis of uploaded separate files (telemetry + blockchain) — {_proc_text}"
    source_color = "#22c55e"
elif data_source == "uploaded":
    source_label = f"📂 Live analysis of uploaded file — {_proc_text}"
    source_color = "#22c55e"
elif data_source == "default_separate":
    source_label = f"💾 Default correlated dataset: network_telemetry.csv + blockchain_tx.csv — {_proc_text}"
    source_color = "#3b82f6"
else:
    source_label = f"💾 Default dataset: {_default_path} — {_proc_text}"
    source_color = "#3b82f6"

st.markdown(
    f'<p style="text-align:center; color:{source_color}; font-size:0.78rem; '
    f'font-weight:500; margin-top:-0.5rem;">{source_label}</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Ingestion Validation Report Expander
# ---------------------------------------------------------------------------
_ingest_report = getattr(adapter, "last_report", None) or {}
with st.expander("🧾 Ingestion Report", expanded=False):
    _r_read = _ingest_report.get("rows_read", len(df))
    _r_valid = _ingest_report.get("rows_valid", len(df))
    _r_rejected = _ingest_report.get("rows_rejected", 0)
    _r_dups = _ingest_report.get("duplicate_txids", 0)
    _r_range = _ingest_report.get("timestamp_range", [None, None])
    _r_sha = _ingest_report.get("sha256") or file_hash
    if not _r_sha or _r_sha == "N/A":
        _r_sha = file_hash

    # Small metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Rows Read", f"{_r_read:,}")
    m_col2.metric("Valid Rows", f"{_r_valid:,}")
    m_col3.metric("Rejected Rows", f"{_r_rejected:,}")
    m_col4.metric("Duplicates", f"{_r_dups:,}")

    # Time range and SHA-256
    _ts_str = f"{_r_range[0]}  ➔  {_r_range[1]}" if _r_range and all(_r_range) else "N/A"
    st.markdown(
        f'<div style="background:{_hp["card"]}; border:1px solid {_hp["card_border"]}; border-radius:6px; padding:10px 14px; margin:10px 0; font-size:0.82rem;">'
        f'<div><b>⏱️ Time Range:</b> <span style="font-family:monospace; color:{_hp["accent"]};">{_ts_str}</span></div>'
        f'<div style="margin-top:4px;"><b>🔒 SHA-256 Provenance:</b> <code style="color:{_hp["text"]}; word-break:break-all;">{_r_sha}</code></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Cross-layer correlation metrics (when in separate upload / correlated mode)
    _corr = correlation_stats or _ingest_report.get("correlation")
    if _corr:
        _ingest_report["correlation"] = _corr
        st.markdown(
            f'<div style="margin-top:12px; margin-bottom:6px; font-weight:700; font-size:0.85rem; color:{_hp["text"]};">'
            f'🔗 Cross-Layer Correlation (Network Telemetry × Blockchain)'
            f'</div>',
            unsafe_allow_html=True,
        )
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric("Matched txids", f"{_corr.get('matched', 0):,}")
        c_col2.metric("Match Rate", f"{_corr.get('match_rate', 0.0) * 100:.1f}%")
        c_col3.metric("Telemetry Only", f"{_corr.get('telemetry_only', 0):,}")
        c_col4.metric("Blockchain Only", f"{_corr.get('chain_only', 0):,}")

    # Reasons table
    _reasons = _ingest_report.get("reject_reasons", {})
    if _reasons and any(v > 0 for v in _reasons.values()):
        reasons_list = [
            {"Reason": k.replace("_", " ").title(), "Count": int(v)}
            for k, v in _reasons.items() if v > 0
        ]
        st.dataframe(pd.DataFrame(reasons_list), width="stretch", hide_index=True)
    else:
        st.caption("✅ All rows passed schema and structural integrity checks (0 rejections).")

    # Download report button
    report_json = json.dumps(_ingest_report, indent=2, default=str)
    st.download_button(
        label="Download report (JSON)",
        data=report_json,
        file_name="ingestion_validation_report.json",
        mime="application/json",
        key="btn_download_ingestion_report",
    )


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-TAB ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🚨 Suspicious Transactions",
    "🕸️ Network Graph",
    "🔍 Why Flagged?",
    "📈 Model Performance",
    "👥 Clusters & Communities",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    _tab_p = _get_palette()
    st.markdown(f'<p style="color:{_tab_p["muted"]};font-size:0.88rem;margin-bottom:0.2rem;">Summary of the whole dataset: how many transactions were scanned and how many look suspicious.</p>', unsafe_allow_html=True)
    with st.expander("ℹ️ How to read this"):
        st.markdown("The top cards count what the engine found: total transactions, whitelisted exchange/mining-pool transfers, AI-flagged anomalies, CoinJoin mixers, peel chains, and the highest Investigative Priority Index (0–100). The charts show how threats are distributed by type and by risk score.")

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
                <div class="card-sub">{total_flagged} flagged → {cleared_count} cleared by whitelist → {actionable_count} actionable</div>
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
                <div class="card-label">Max Investigative Priority Index</div>
                <div class="card-value">{max_risk:.1f}%</div>
                <div class="card-sub">Highest anomaly deviation from baseline</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Risk tier badges row ──
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; gap:1rem; margin:0.6rem 0 0.2rem 0; flex-wrap:wrap;">
            <span style="background:rgba(239,68,68,0.15); color:#ef4444; padding:0.3rem 0.9rem; border-radius:999px; font-size:0.78rem; font-weight:700; border:1px solid rgba(239,68,68,0.3);">🔴 Critical: {tier_critical}</span>
            <span style="background:rgba(249,115,22,0.15); color:#f97316; padding:0.3rem 0.9rem; border-radius:999px; font-size:0.78rem; font-weight:700; border:1px solid rgba(249,115,22,0.3);">🟠 High: {tier_high}</span>
            <span style="background:rgba(234,179,8,0.15); color:#eab308; padding:0.3rem 0.9rem; border-radius:999px; font-size:0.78rem; font-weight:700; border:1px solid rgba(234,179,8,0.3);">🟡 Medium: {tier_medium}</span>
            <span style="background:rgba(107,114,128,0.15); color:#9ca3af; padding:0.3rem 0.9rem; border-radius:999px; font-size:0.78rem; font-weight:700; border:1px solid rgba(107,114,128,0.3);">⚪ Low: {tier_low}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Pipeline summary ──
    _sp = _get_palette()
    st.markdown(
        f"""
        <div style="background: rgba(13,148,136,0.05); border: 1px solid rgba(13,148,136,0.15);
                    border-radius: 0.75rem; padding: 1.2rem; color: {_sp['text']}; font-size: 0.85rem; line-height: 1.6;">
            <b style="color: {_sp['accent2']};">📊 Pipeline Summary</b><br>
            The engine processed <b style="color: #3b82f6;">{total_tx:,}</b> transactions,
            bypassed <b style="color: #8b5cf6;">{mixer_count}</b> mixer anonymity sets,
            detected <b style="color: #f59e0b;">{peel_count}</b> peel-chain structures,
            identified <b style="color: #06b6d4;">{fanout_count}</b> fan-out dispersals and
            <b style="color: #ef4444;">{feespike_count}</b> fee-spike urgency hops,
            and cleared <b style="color: #22c55e;">{whitelisted_count}</b> institutional transfers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Threat vector breakdown chart ──
    col_chart1, col_chart2 = st.columns(2)

    _cp = _get_palette()
    with col_chart1:
        st.markdown(
            '<div class="section-title"><span class="icon">📊</span> Threat Vector Distribution</div>',
            unsafe_allow_html=True,
        )
        if "detected_type" in df.columns:
            attack_counts = df["detected_type"].value_counts().reset_index()
            attack_counts.columns = ["Detected Pattern", "Count"]
            chart1 = alt.Chart(attack_counts).mark_bar(
                cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=_cp["accent2"]
            ).encode(
                x=alt.X("Detected Pattern:N", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], labelAngle=-30)),
                y=alt.Y("Count:Q", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], gridColor=_cp["chart_grid"])),
            ).properties(height=300).configure(background=_cp["chart_bg"]).configure_view(strokeWidth=0)
            st.altair_chart(chart1, width="stretch")

    with col_chart2:
        st.markdown(
            '<div class="section-title"><span class="icon">📈</span> Risk Score Distribution</div>',
            unsafe_allow_html=True,
        )
        if total_tx > 0:
            _bins = list(range(0, 101, 10))
            # Build per-group histograms
            _hist_rows = []
            for _, _r in df.iterrows():
                _sc = _r["risk_score"]
                _grp = "Whitelisted" if ("whitelisted_side" in df.columns and pd.notna(_r.get("whitelisted_side"))) else "Non-whitelisted"
                _hist_rows.append({"score": _sc, "Group": _grp})
            _hist_df = pd.DataFrame(_hist_rows)
            _hist_df["Risk Bin"] = pd.cut(_hist_df["score"], bins=_bins, include_lowest=True, right=True)
            _hist_df["Risk Bin"] = _hist_df["Risk Bin"].apply(lambda b: f"{b.left:.0f}–{b.right:.0f}" if pd.notna(b) else "100")
            _hist_agg = _hist_df.groupby(["Risk Bin", "Group"], as_index=False).size()
            _hist_agg.columns = ["Risk Bin", "Group", "Count"]
            _color_scale = alt.Scale(
                domain=["Non-whitelisted", "Whitelisted"],
                range=["#ef4444", "#6b7280"],
            )
            chart2 = alt.Chart(_hist_agg).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
            ).encode(
                x=alt.X("Risk Bin:N", sort=_bins, axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], labelAngle=-30)),
                y=alt.Y("Count:Q", stack=True, axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], gridColor=_cp["chart_grid"])),
                color=alt.Color("Group:N", scale=_color_scale, legend=alt.Legend(title="Group", labelColor=_cp["chart_text"], titleColor=_cp["chart_text"])),
                order=alt.Order("Group:N", sort="descending"),
            ).properties(height=300).configure(background=_cp["chart_bg"]).configure_view(strokeWidth=0)
            st.altair_chart(chart2, width="stretch")
        else:
            st.info("No data to display.")

    # ── Geographic Distribution Section ──
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title"><span class="icon">🌍</span> Geographic Distribution</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{_cp["muted"]};font-size:0.88rem;margin-bottom:0.75rem;">'
        'Offline geolocation and Autonomous System (ASN) telemetry breakdown of observed transaction traffic.'
        '</p>',
        unsafe_allow_html=True,
    )

    _geo_world_path = os.path.join("data", "geo", "world-110m.json")
    _iso_codes_path = os.path.join("data", "geo", "iso_codes.csv")

    _iso_lookup_df = None
    if os.path.isfile(_iso_codes_path):
        try:
            _iso_lookup_df = pd.read_csv(_iso_codes_path)
        except Exception:
            _iso_lookup_df = None

    # Country aggregations
    if "geo_country" in df.columns and total_tx > 0:
        _country_grp = df.groupby("geo_country").agg(
            total=("txid", "count"),
            flagged=("is_anomaly", lambda s: int(s.sum())),
            avg_risk=("risk_score", "mean"),
        ).reset_index()

        if _iso_lookup_df is not None:
            _country_stats = _country_grp.merge(_iso_lookup_df, left_on="geo_country", right_on="iso2", how="inner")
        else:
            _country_stats = _country_grp.copy()
            _country_stats["country_name"] = _country_stats["geo_country"]
            _country_stats["id"] = 0
    else:
        _country_stats = pd.DataFrame(columns=["geo_country", "total", "flagged", "avg_risk", "country_name", "id"])

    # Top 10 Countries Bar Chart data
    _valid_countries = df[~df["geo_country"].isin(["XX", "", "Unknown", "nan", None])].copy() if "geo_country" in df.columns else pd.DataFrame()
    if not _valid_countries.empty:
        _c_top = _valid_countries.groupby("geo_country").agg(
            flagged=("is_anomaly", lambda s: int(s.sum())),
            total=("txid", "count"),
            avg_risk=("risk_score", "mean"),
        ).reset_index()
        if _iso_lookup_df is not None:
            _c_top = _c_top.merge(_iso_lookup_df, left_on="geo_country", right_on="iso2", how="left")
            _c_top["Country"] = _c_top["country_name"].fillna(_c_top["geo_country"])
        else:
            _c_top["Country"] = _c_top["geo_country"]
        _c_top10 = _c_top.sort_values(["flagged", "total"], ascending=[False, False]).head(10)
    else:
        _c_top10 = pd.DataFrame(columns=["Country", "flagged", "total", "avg_risk"])

    # Top 10 ASN Table data
    if "asn" in df.columns and total_flagged > 0:
        _asn_flagged = df[df["is_anomaly"] == True].copy()
        if not _asn_flagged.empty:
            _asn_top = _asn_flagged.groupby(["asn", "asn_org"], as_index=False).size()
            _asn_top.columns = ["ASN", "Organisation", "Flagged Count"]
            _asn_top10 = _asn_top.sort_values("Flagged Count", ascending=False).head(10)
        else:
            _asn_top10 = pd.DataFrame(columns=["ASN", "Organisation", "Flagged Count"])
    else:
        _asn_top10 = pd.DataFrame(columns=["ASN", "Organisation", "Flagged Count"])

    # Render: Map (if available) + Bar Chart & ASN Table
    _map_available = os.path.isfile(_geo_world_path) and (_iso_lookup_df is not None)

    if _map_available:
        geo_col1, geo_col2 = st.columns([3, 2])
        with geo_col1:
            st.markdown(
                '<div style="font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;">🗺️ Global Threat Distribution (Choropleth)</div>',
                unsafe_allow_html=True,
            )
            try:
                with open(_geo_world_path, "r", encoding="utf-8") as _wf:
                    _world_topo = json.load(_wf)

                _source = alt.InlineData(
                    values=_world_topo,
                    format=alt.DataFormat(type="topojson", feature="countries"),
                )

                _map_chart = alt.Chart(_source).mark_geoshape(
                    stroke=_cp.get("chart_grid", "#334155"),
                    strokeWidth=0.4,
                ).transform_lookup(
                    lookup="id",
                    from_=alt.LookupData(
                        data=_country_stats,
                        key="id",
                        fields=["country_name", "flagged", "total", "avg_risk"],
                    ),
                ).encode(
                    color=alt.Color(
                        "flagged:Q",
                        scale=alt.Scale(scheme="reds"),
                        legend=alt.Legend(
                            title="Flagged Tx",
                            labelColor=_cp["chart_text"],
                            titleColor=_cp["chart_text"],
                            orient="bottom",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("country_name:N", title="Country"),
                        alt.Tooltip("total:Q", title="Total Tx"),
                        alt.Tooltip("flagged:Q", title="Flagged Tx"),
                        alt.Tooltip("avg_risk:Q", title="Average Risk", format=".1f"),
                    ],
                ).project(
                    type="equalEarth",
                ).properties(
                    height=360,
                ).configure(
                    background=_cp["chart_bg"],
                ).configure_view(
                    strokeWidth=0,
                )

                st.altair_chart(_map_chart, width="stretch")
            except Exception as _map_err:
                st.warning(f"Could not render map: {_map_err}")

            st.markdown(
                f'<div style="font-size:0.75rem; color:{_cp["muted"]}; margin-top:2px; margin-bottom:0.5rem;">'
                'IP Geolocation by DB-IP (CC BY 4.0)'
                '</div>',
                unsafe_allow_html=True,
            )

        with geo_col2:
            st.markdown(
                '<div style="font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;">🚩 Top 10 Countries by Flagged Tx</div>',
                unsafe_allow_html=True,
            )
            if not _c_top10.empty and _c_top10["flagged"].sum() > 0:
                _c_bar = alt.Chart(_c_top10).mark_bar(
                    cornerRadiusTopRight=4,
                    cornerRadiusBottomRight=4,
                    color=_cp["accent2"],
                ).encode(
                    y=alt.Y("Country:N", sort="-x", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"])),
                    x=alt.X("flagged:Q", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], gridColor=_cp["chart_grid"]), title="Flagged Tx"),
                    tooltip=[
                        alt.Tooltip("Country:N", title="Country"),
                        alt.Tooltip("flagged:Q", title="Flagged Tx"),
                        alt.Tooltip("total:Q", title="Total Tx"),
                        alt.Tooltip("avg_risk:Q", title="Average Risk", format=".1f"),
                    ],
                ).properties(height=180).configure(background=_cp["chart_bg"]).configure_view(strokeWidth=0)
                st.altair_chart(_c_bar, width="stretch")
            else:
                st.info("No flagged country anomalies.")

            st.markdown(
                '<div style="font-weight:600; font-size:0.95rem; margin-top:0.6rem; margin-bottom:0.4rem;">🌐 Top 10 Autonomous Systems (ASNs)</div>',
                unsafe_allow_html=True,
            )
            if not _asn_top10.empty:
                st.dataframe(_asn_top10, hide_index=True, width="stretch")
            else:
                st.info("No ASN telemetry recorded.")

    else:
        # Fallback when map file is missing: show only bar chart and ASN table
        geo_col1, geo_col2 = st.columns(2)
        with geo_col1:
            st.markdown(
                '<div style="font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;">🚩 Top 10 Countries by Flagged Tx</div>',
                unsafe_allow_html=True,
            )
            if not _c_top10.empty and _c_top10["flagged"].sum() > 0:
                _c_bar = alt.Chart(_c_top10).mark_bar(
                    cornerRadiusTopRight=4,
                    cornerRadiusBottomRight=4,
                    color=_cp["accent2"],
                ).encode(
                    y=alt.Y("Country:N", sort="-x", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"])),
                    x=alt.X("flagged:Q", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], gridColor=_cp["chart_grid"]), title="Flagged Tx"),
                    tooltip=[
                        alt.Tooltip("Country:N", title="Country"),
                        alt.Tooltip("flagged:Q", title="Flagged Tx"),
                        alt.Tooltip("total:Q", title="Total Tx"),
                        alt.Tooltip("avg_risk:Q", title="Average Risk", format=".1f"),
                    ],
                ).properties(height=240).configure(background=_cp["chart_bg"]).configure_view(strokeWidth=0)
                st.altair_chart(_c_bar, width="stretch")
            else:
                st.info("No flagged country anomalies.")

        with geo_col2:
            st.markdown(
                '<div style="font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;">🌐 Top 10 Autonomous Systems (ASNs)</div>',
                unsafe_allow_html=True,
            )
            if not _asn_top10.empty:
                st.dataframe(_asn_top10, hide_index=True, width="stretch")
            else:
                st.info("No ASN telemetry recorded.")

    # ── Export buttons & Evidence Dossier ──
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title"><span class="icon">📑</span> Evidence Dossier & Forensic Case Export</div>',
        unsafe_allow_html=True,
    )
    _cp = _get_palette()
    st.markdown(
        f'<p style="color:{_cp["muted"]};font-size:0.88rem;margin-bottom:0.75rem;">'
        'Generate court-ready, multi-page evidentiary dossiers with multi-signal score decompositions, '
        'hop-by-hop fund traces, cryptographic provenance hashes, and methodology limitations.'
        '</p>',
        unsafe_allow_html=True,
    )

    # UI Controls: Lead Forensic Analyst & Case Reference
    if "case_id" not in st.session_state:
        st.session_state["case_id"] = f"MITHYA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    case_id = st.session_state["case_id"]

    dossier_col1, dossier_col2 = st.columns([1.5, 2.5])
    with dossier_col1:
        analyst_name = st.text_input(
            "Lead Forensic Analyst",
            value="Analyst",
            key="dossier_analyst_name",
            help="Name or badge number to appear on the official dossier cover page and signature block.",
        )
    with dossier_col2:
        st.text_input(
            "Active Case Identification",
            value=case_id,
            disabled=True,
            help="Unique evidence tracking identifier (MITHYA-YYYYMMDD-HHMMSS).",
        )

    # 1. Existing TXT Forensic Report
    import platform
    report_lines = [
        "=" * 70,
        "  MITHYA — Crypto Forensic Triage Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Engine: IsolationForest (n_estimators=200, contamination={contamination})",
        "=" * 70,
        "",
        "  DATA PROVENANCE & CONFIGURATION",
        "  -------------------------------",
        f"  Ingestion Time:   {ingest_time}",
        f"  Source File Hash: {file_hash} (SHA-256)",
        f"  Model Version:    v1.4.0 (Graph-Aware Anomaly Detection)",
        f"  System Config:    {platform.platform()} / Python {platform.python_version()}",
        "",
        "  ANALYSIS SUMMARY",
        "  ----------------",
        f"  Total Transactions Analysed:  {total_tx:,}",
        f"  Institutional Transfers Cleared: {whitelisted_count}",
        f"  ML Anomalies Flagged:        {total_flagged}",
        f"  CoinJoin Mixers Detected:    {mixer_count}",
        f"  Peel Chains Detected:        {peel_count}",
        f"  Fan-Out Dispersals:          {fanout_count}",
        f"  Fee-Spike Urgency Hops:      {feespike_count}",
        f"  Max Investigative Priority Index: {max_risk:.1f}%",
        "",
        "-" * 70,
        "  FLAGGED TRANSACTIONS",
        "-" * 70,
        "",
    ]

    if total_flagged > 0:
        for _, row in flagged_df.sort_values("risk_score", ascending=False).iterrows():
            report_lines.append(f"  TXID: {row['txid']}")
            report_lines.append(f"  Priority: {row['risk_score']:.1f}%  |  Entity: {row.get('entity_id', 'N/A')}")
            report_lines.append(f"  Source IP: {row['src_ip']}  |  Port: {row['src_port']}")
            report_lines.append(f"  Amount: {row.get('total_amount_btc', 'N/A')} BTC  |  Fee: {row['fee']}")
            report_lines.append(f"  Geo: {_display_geo(row.get('geo_country'), 'geo')}  |  ASN: {_display_geo(row.get('asn'), 'asn')}")
            report_lines.append(f"  {row['explanation']}")
            report_lines.append("")
    else:
        report_lines.append("  No anomalies detected.")

    report_lines.append("=" * 70)
    report_lines.append("  END OF REPORT")
    report_lines.append("=" * 70)
    report_text = "\n".join(report_lines)

    # 2. Build Metadata for PDF and JSON Dossiers
    def _safe_sha256(filepath: str) -> str:
        if os.path.isfile(filepath):
            try:
                with open(filepath, "rb") as _f:
                    return hashlib.sha256(_f.read()).hexdigest()
            except Exception:
                pass
        return "N/A"

    wl_sha = _safe_sha256("institutional_whitelist.csv")
    if wl_sha == "N/A":
        wl_sha = _safe_sha256("sample_data/institutional_whitelist.csv")

    wt_sha = _safe_sha256("sample_data/watchlist.csv")
    if wt_sha == "N/A":
        wt_sha = _safe_sha256("watchlist.csv")

    rf_hash = "N/A"
    if os.path.isfile("models/rf_model.sha256"):
        try:
            with open("models/rf_model.sha256", "r") as _rf_f:
                rf_hash = _rf_f.read().strip()
        except Exception:
            pass
    if rf_hash == "N/A" and os.path.isfile("models/rf_model.joblib"):
        rf_hash = _safe_sha256("models/rf_model.joblib")

    # Dynamic fusion weights and thresholds
    weights = getattr(adapter, "WEIGHTS", None)
    if not isinstance(weights, dict):
        from anomaly_engine import WEIGHTS as DEFAULT_WEIGHTS
        weights = DEFAULT_WEIGHTS

    detector_thresholds = {
        "CoinJoin Min Inputs": "3 inputs",
        "Fan-Out Min Dispersal": "5 outputs",
        "Peel Disparity Ratio": "0.70",
        "Port Risk Threshold": "40.0 pts",
    }

    funnel_numbers = {
        "total_ingested": total_tx,
        "whitelisted_cleared": whitelisted_count,
        "flagged_anomalies": total_flagged,
        "mixers_detected": mixer_count,
        "peel_chains_detected": peel_count,
        "fanouts_detected": fanout_count,
        "feespikes_detected": feespike_count,
    }

    tier_counts = {
        "Critical": int((df["risk_tier"] == "Critical").sum()) if "risk_tier" in df.columns else 0,
        "High": int((df["risk_tier"] == "High").sum()) if "risk_tier" in df.columns else 0,
        "Medium": int((df["risk_tier"] == "Medium").sum()) if "risk_tier" in df.columns else 0,
        "Low": int((df["risk_tier"] == "Low").sum()) if "risk_tier" in df.columns else 0,
    }

    dossier_meta = {
        "case_id": case_id,
        "analyst_name": analyst_name.strip() or "Analyst",
        "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file_sha256": file_hash,
        "whitelist_sha256": wl_sha,
        "watchlist_sha256": wt_sha,
        "model_version": "v2.2.0 (Fused Multi-Signal Anomaly Engine)",
        "rf_model_hash": rf_hash,
        "contamination": contamination,
        "weights": weights,
        "detector_thresholds": detector_thresholds,
        "funnel_numbers": funnel_numbers,
        "tier_counts": tier_counts,
    }

    # 3. Build PDF and JSON Dossiers (with caching in session_state)
    dossier_cache_key = f"{file_hash}_{dossier_meta['analyst_name']}_{contamination}_{len(df)}"
    if st.session_state.get("_dossier_cache_key") == dossier_cache_key:
        pdf_bytes = st.session_state["_dossier_pdf_bytes"]
        json_bytes = st.session_state["_dossier_json_bytes"]
        pdf_sha256 = st.session_state["_dossier_pdf_sha256"]
    else:
        with st.spinner("Generating court-ready multi-page PDF & JSON evidence dossiers..."):
            pdf_bytes = dossier.build_pdf(df, dossier_meta)
            json_str = dossier.build_json(df, dossier_meta)
            json_bytes = json_str.encode("utf-8")
            pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()

            st.session_state["_dossier_cache_key"] = dossier_cache_key
            st.session_state["_dossier_pdf_bytes"] = pdf_bytes
            st.session_state["_dossier_json_bytes"] = json_bytes
            st.session_state["_dossier_pdf_sha256"] = pdf_sha256

    # 4. Display PDF Provenance SHA-256
    st.markdown("<b>🔒 Generated PDF SHA-256 Provenance Hash:</b>", unsafe_allow_html=True)
    st.code(pdf_sha256, language=None)

    # 5. Download Buttons: PDF, JSON, TXT, and .sha256 Sidecar
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        st.download_button(
            label="📕 Download Dossier (PDF)",
            data=pdf_bytes,
            file_name=f"mithya_forensic_dossier_{case_id}.pdf",
            mime="application/pdf",
            width="stretch",
        )

    with btn_col2:
        st.download_button(
            label="📋 Download Evidence (JSON)",
            data=json_bytes,
            file_name=f"mithya_forensic_dossier_{case_id}.json",
            mime="application/json",
            width="stretch",
        )

    with btn_col3:
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report_text.encode("utf-8"),
            file_name=f"mithya_forensic_report_{case_id}.txt",
            mime="text/plain",
            width="stretch",
        )

    with btn_col4:
        sidecar_text = f"{pdf_sha256}  mithya_forensic_dossier_{case_id}.pdf\n"
        st.download_button(
            label="🔐 Download .sha256 Sidecar",
            data=sidecar_text.encode("utf-8"),
            file_name=f"mithya_forensic_dossier_{case_id}.pdf.sha256",
            mime="text/plain",
            width="stretch",
        )

    # Full Results CSV
    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Results CSV",
        data=csv_export,
        file_name="mithya_full_results.csv",
        mime="text/csv",
        width="stretch",
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: SUSPICIOUS TRANSACTIONS
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        '<div class="section-title"><span class="icon">🚨</span> Suspicious Transactions — Ranked by Investigative Priority</div>',
        unsafe_allow_html=True,
    )
    _tab_p = _get_palette()
    st.markdown(f'<p style="color:{_tab_p["muted"]};font-size:0.88rem;margin-bottom:0.2rem;">Highest-risk transactions first, each with its reason.</p>', unsafe_allow_html=True)
    with st.expander("ℹ️ How to read this"):
        st.markdown("Each row is one transaction. Investigative Priority % is the AI risk score (higher = more suspicious). Cluster confidence shows how reliable the wallet grouping is: High-Confidence, Mixer-Affected, or Heuristic/Inferred. Use the filters to narrow by attack type or risk range.")

    # ── Filters ──
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 1.5, 1])

    with filter_col1:
        detected_types = ["All"] + sorted(df["detected_type"].unique().tolist()) if "detected_type" in df.columns else ["All"]
        selected_type = st.selectbox("Filter by Detected Pattern", detected_types)

    with filter_col2:
        risk_range = st.slider(
            "Risk Score Range",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 100.0),
            step=1.0,
        )

    with filter_col3:
        _all_tiers = ["Critical", "High", "Medium", "Low"]
        _available_tiers = sorted(df["risk_tier"].unique().tolist(), key=lambda t: _all_tiers.index(t) if t in _all_tiers else 99) if "risk_tier" in df.columns else _all_tiers
        selected_tiers = st.multiselect("Risk Tier", _available_tiers, default=_available_tiers)

    with filter_col4:
        anomalies_only = st.toggle("Anomalies Only", value=False)

    filter_clust_col1, filter_clust_col2 = st.columns(2)
    with filter_clust_col1:
        _all_clusters = ["All"] + sorted(df["behaviour_cluster"].dropna().unique().tolist()) if "behaviour_cluster" in df.columns else ["All"]
        selected_cluster = st.selectbox("Filter by Behaviour Cluster", _all_clusters)
    with filter_clust_col2:
        _all_comms = ["All"] + sorted(df["graph_community"].dropna().unique().tolist()) if "graph_community" in df.columns else ["All"]
        selected_comm = st.selectbox("Filter by Graph Community", _all_comms)

    # ── Apply filters ──
    view_df = df.copy()
    if selected_type != "All" and "detected_type" in view_df.columns:
        view_df = view_df[view_df["detected_type"] == selected_type]
    if "risk_tier" in view_df.columns and selected_tiers:
        view_df = view_df[view_df["risk_tier"].isin(selected_tiers)]
    if selected_cluster != "All" and "behaviour_cluster" in view_df.columns:
        view_df = view_df[view_df["behaviour_cluster"] == selected_cluster]
    if selected_comm != "All" and "graph_community" in view_df.columns:
        view_df = view_df[view_df["graph_community"] == selected_comm]
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
    # Prepare Watchlist, Taint %, and Hops from known-bad
    view_df["Watchlist"] = view_df.apply(
        lambda r: str(r.get("watchlist_label", "")) if r.get("watchlist_hit") and pd.notna(r.get("watchlist_label")) else "",
        axis=1,
    )
    view_df["Taint %"] = view_df["taint_score"].apply(
        lambda v: round(float(v), 1) if pd.notna(v) and str(v) != "nan" else 0.0
    ) if "taint_score" in view_df.columns else 0.0

    view_df["Hops from known-bad"] = view_df["taint_hops"].apply(
        lambda v: str(int(v)) if pd.notna(v) and v is not None and str(v) not in ("nan", "None", "") else ""
    ) if "taint_hops" in view_df.columns else ""

    display_cols = [
        "txid", "risk_score", "entity_id", "cluster_confidence", "explanation",
        "src_port", "dst_port", "total_amount_btc", "fee",
        "script_type", "geo_country", "asn",
    ]
    if "risk_tier" in view_df.columns:
        display_cols.insert(2, "risk_tier")
    if "detected_type" in view_df.columns:
        display_cols.insert(3, "detected_type")

    ins_idx = 4 if "detected_type" in view_df.columns else 3
    display_cols.insert(ins_idx, "Watchlist")
    display_cols.insert(ins_idx + 1, "Taint %")
    display_cols.insert(ins_idx + 2, "Hops from known-bad")
    if "behaviour_cluster" in view_df.columns:
        display_cols.insert(ins_idx + 3, "behaviour_cluster")
    if "graph_community" in view_df.columns:
        display_cols.insert(ins_idx + 4, "graph_community")

    display_df = view_df[[c for c in display_cols if c in view_df.columns]].copy()

    # Truncate txid for display
    display_df["txid"] = display_df["txid"].apply(
        lambda x: f"{str(x)[:8]}…{str(x)[-6:]}" if len(str(x)) > 16 else str(x)
    )

    # Clean up sentinel geo/ASN values for display
    if "geo_country" in display_df.columns:
        display_df["geo_country"] = display_df["geo_country"].apply(lambda v: _display_geo(v, "geo"))
    if "asn" in display_df.columns:
        display_df["asn"] = display_df["asn"].apply(lambda v: _display_geo(v, "asn"))

    display_df = display_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Rename columns for display
    rename_map = {
        "txid": "TX ID",
        "risk_score": "Investigative Priority %",
        "risk_tier": "Risk Tier",
        "entity_id": "Entity",
        "detected_type": "Detected Pattern",
        "behaviour_cluster": "Behaviour Cluster",
        "graph_community": "Graph Community",
        "explanation": "XAI Explanation",
        "src_port": "Src Port",
        "dst_port": "Dst Port",
        "total_amount_btc": "Amount (BTC)",
        "fee": "Fee",
        "script_type": "Script",
        "geo_country": "Propagation Node",
        "asn": "ASN",
    }
    display_df.rename(columns=rename_map, inplace=True)

    # ── Render ──
    # Phase 3 disclaimer: network telemetry is NOT absolute identity attribution
    st.markdown(
        """
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3);
                    border-radius: 0.6rem; padding: 0.7rem 1rem; margin-bottom: 0.8rem;">
            <span style="color: #fbbf24; font-size: 0.78rem; font-weight: 600;">
                &#9888;&#65039; Network observation correlation &mdash; not absolute identity attribution
                (Subject to VPN / NAT / Tor limits).
                ASN and geo-country are resolved offline via a local MaxMind GeoLite2 database; unmatched IPs are shown as Unknown.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="investigation-panel">', unsafe_allow_html=True)

    prio_col = "Investigative Priority %" if "Investigative Priority %" in display_df.columns else "Risk %"
    _tp = _get_palette()
    # Tier colour styling function
    _tier_colors = {
        "Critical": "rgba(239, 68, 68, 0.25)",
        "High": "rgba(249, 115, 22, 0.25)",
        "Medium": "rgba(234, 179, 8, 0.20)",
        "Low": "rgba(107, 114, 128, 0.15)",
    }
    _tier_text_colors = {
        "Critical": "#ef4444",
        "High": "#f97316",
        "Medium": "#eab308",
        "Low": "#9ca3af",
    }
    def _style_risk_tier(val):
        bg = _tier_colors.get(val, "")
        fg = _tier_text_colors.get(val, _tp["df_text"])
        return f"background-color: {bg}; color: {fg}; font-weight: 700" if bg else ""

    _fmt = {"Amount (BTC)": "{:.6f}", prio_col: "{:.1f}", "Fee": "{:.8f}"}
    if "Taint %" in display_df.columns:
        _fmt["Taint %"] = "{:.1f}"

    _styler = (
        display_df.style
        .background_gradient(subset=[prio_col], cmap="YlOrRd", vmin=0, vmax=100)
        .format(_fmt)
        .set_properties(**{"background-color": _tp["df_bg"], "color": _tp["df_text"]})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", _tp["card"]), ("color", _tp["text"])]},
        ])
    )
    if "Risk Tier" in display_df.columns:
        _styler = _styler.map(_style_risk_tier, subset=["Risk Tier"])
    styled_df = _styler
    st.dataframe(
        styled_df,
        width="stretch",
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
            width="stretch",
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: NETWORK GRAPH
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        """
        <div class="section-title">
            <span class="icon">🕸️</span> Network Graph
        </div>
        <p class="section-subtitle">
            Fund flow visualization &nbsp;·&nbsp;
            <span style="color:#ef4444; font-weight:600;">● Red = flagged anomalies</span> &nbsp;·&nbsp;
            <span style="color:#f59e0b; font-weight:600;">◆ Amber = transactions</span> &nbsp;·&nbsp;
            <span style="color:#9b59b6; font-weight:600;">■ Purple = wallet entities</span> &nbsp;·&nbsp;
            <span style="color:#4ade80; font-weight:600;">● Green = regulated</span> &nbsp;·&nbsp;
            <span style="color:#3b82f6; font-weight:600;">● Blue = context nodes</span> &nbsp;·&nbsp;
            <span style="color:#ef4444; font-weight:600;">★ = watchlist (known-bad)</span>
        </p>
        """,
        unsafe_allow_html=True,
    )
    _tab_p = _get_palette()
    st.markdown(f'<p style="color:{_tab_p["muted"]};font-size:0.88rem;margin-bottom:0.2rem;">See which wallets, IPs, and transactions are connected.</p>', unsafe_allow_html=True)
    with st.expander("ℹ️ How to read this"):
        st.markdown("Nodes are wallets, IP addresses and transactions; lines show how funds and observations connect. Red = flagged anomalies, amber = transactions, purple = wallet entities, green = regulated/whitelisted, blue = context nodes, ★ star = watchlist (known-bad). Pick a target entity to isolate its neighbourhood. Drag, zoom and hover to explore.")

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

    # ── Target Entity Isolation ──
    sorted_df = df.sort_values("risk_score", ascending=False)
    unique_entities = sorted_df["entity_id"].dropna().unique().tolist()
    entity_options = ["View Full Graph"] + unique_entities

    target_entity = st.selectbox(
        "🎯 Isolate Target Entity (Wallet Address)",
        entity_options,
        help="Select a specific entity to view its localized subgraph (ego network) up to 2 hops away."
    )

    MAX_CONTEXT_NORMAL = 50
    MAX_CLEAN_SAMPLE = 40

    _gp = _get_palette()

    def build_network_graph(graph_df: pd.DataFrame, threats_only_mode: bool = True, target_entity: str = "View Full Graph") -> str:
        """Build a focused, intelligence-style tripartite network graph."""
        net = Network(
            height="580px",
            width="100%",
            bgcolor=_gp["graph_bgcolor"],
            font_color=_gp["graph_fontcolor"],
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

        # Load known-bad watchlist
        try:
            from ml_engine import load_watchlist
            _wl_dict = load_watchlist()
            _wl_addrs = {e["indicator"].lower(): e for e in _wl_dict.get("address", [])}
            _wl_ips = {e["indicator"].lower(): e for e in _wl_dict.get("ip", [])}
        except Exception:
            _wl_addrs = {}
            _wl_ips = {}

        _wl_hit_entities = set()
        for _, r in graph_df.iterrows():
            if r.get("watchlist_hit"):
                if "entity_id" in r and pd.notna(r["entity_id"]):
                    _wl_hit_entities.add(str(r["entity_id"]))

        anom_rows = graph_df[graph_df["is_anomaly"] == True]
        wl_rows = graph_df[graph_df["watchlist_hit"] == True] if "watchlist_hit" in graph_df.columns else pd.DataFrame()
        regulated_rows = graph_df[graph_df["explanation"].str.contains("Regulated", na=False)]

        anom_ips = set(anom_rows["src_ip"].unique())
        anom_entities = set()
        if "entity_id" in anom_rows.columns:
            anom_entities = set(anom_rows["entity_id"].unique())

        if target_entity == "View Full Graph":
            plot_df = graph_df.sort_values("risk_score", ascending=False).head(400)
        else:
            ent_rows = graph_df[graph_df["entity_id"] == target_entity]
            ent_ips = set(ent_rows["src_ip"].dropna().unique())
            rel_rows = graph_df[graph_df["src_ip"].isin(ent_ips) | (graph_df["entity_id"] == target_entity)]
            plot_df = rel_rows.sort_values("risk_score", ascending=False).head(400)

        anom_rows = graph_df[graph_df["is_anomaly"] == True]
        anom_ips = set(anom_rows["src_ip"].unique())
        anom_entities = set()
        if "entity_id" in anom_rows.columns:
            anom_entities = set(anom_rows["entity_id"].dropna().unique())

        import networkx as nx
        temp_nx = nx.DiGraph()
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
                if src_ip.lower() in _wl_ips:
                    ip_color = {"background": "black", "border": "#ef4444"}
                    ip_size = 28
                    ip_shape = "star"
                    ip_lbl = _wl_ips[src_ip.lower()].get("label", "Threat Indicator")
                    ip_title = f"★ WATCHLIST IP ({ip_lbl}): {src_ip}"
                    temp_nx.add_node(
                        src_ip, label=f"★ {src_ip}", color=ip_color, size=ip_size,
                        shape=ip_shape, borderWidth=3, title=ip_title,
                        font={"size": 12, "color": "#ef4444", "bold": True},
                    )
                else:
                    if is_anom or src_ip in anom_ips:
                        ip_color, ip_size = "#ef4444", 24
                        ip_title = f"⚠️ SUSPICIOUS IP: {src_ip}"
                    elif is_regulated:
                        ip_color, ip_size = "#4ade80", 18
                        ip_title = f"✅ REGULATED IP: {src_ip}"
                    else:
                        ip_color, ip_size = "#3b82f6", 14
                        ip_title = f"IP: {src_ip}"
                    temp_nx.add_node(src_ip, label=src_ip, color=ip_color, size=ip_size, shape="dot", title=ip_title)
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
                temp_nx.add_node(txid, label=tx_label, color=tx_color, size=tx_size, shape="diamond", title=tx_title)
                added_nodes.add(txid)

            # ─── Entity node ───
            if entity not in added_nodes:
                is_wl_ent = (entity.lower() in _wl_addrs) or (entity in _wl_hit_entities)
                if is_wl_ent:
                    ent_color = {"background": "black", "border": "#ef4444"}
                    ent_size = 32
                    ent_shape = "star"
                    ent_lbl = _wl_addrs.get(entity.lower(), {}).get("label", "Known-Bad Wallet")
                    ent_title = f"★ WATCHLIST ENTITY ({ent_lbl}): {entity}"
                    display_label = f"★ {entity}"
                    temp_nx.add_node(
                        entity, label=display_label, color=ent_color, size=ent_size, shape=ent_shape,
                        title=ent_title, borderWidth=3,
                        font={"size": 14, "color": "#ffffff", "bold": True},
                    )
                else:
                    # Phase 3: cluster confidence labels (not all common-inputs are
                    # absolute facts — label based on evidence strength)
                    is_mixer_ent = "CoinJoin" in str(row.get("detected_type", ""))
                    if is_regulated:
                        ent_color, ent_size = "#10b981", 28  # Emerald Green
                        ent_title = f"✅ REGULATED ENTITY: {entity}"
                        cluster_label = "Whitelisted cluster"
                        display_label = f"{entity} 🛡️"
                    elif is_mixer_ent:
                        ent_color, ent_size = "#a78bfa", 26
                        ent_title = (
                            f"⚠️ MIXER-AFFECTED ENTITY: {entity}\n"
                            f"Cluster Confidence: LOWER (CoinJoin may merge unrelated wallets)"
                        )
                        cluster_label = "Mixer-affected cluster"
                        display_label = entity
                    elif is_anom or entity in anom_entities:
                        ent_color, ent_size = "#dc2626", 30  # Crimson
                        ent_title = (
                            f"⚠️ HIGH-RISK ENTITY: {entity}\n"
                            f"Cluster Confidence: HIGH (multi-hop peel chain linkage)"
                        )
                        cluster_label = "High-confidence cluster"
                        display_label = entity
                    else:
                        ent_color, ent_size = "#9b59b6", 22
                        ent_title = (
                            f"Entity Cluster: {entity}\n"
                            f"Cluster Confidence: HEURISTIC (common-input-ownership)"
                        )
                        cluster_label = "Heuristic cluster"
                        display_label = entity
                    temp_nx.add_node(
                        entity, label=display_label, color=ent_color, size=ent_size, shape="square",
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

            temp_nx.add_edge(src_ip, txid, color=edge_color, width=edge_width, title=f"Node: {country}")
            temp_nx.add_edge(entity, txid, color=edge_color, width=edge_width, title="Inputs owned by entity")

        # ── Filter using ego_graph if a target is selected ──
        if target_entity != "View Full Graph" and target_entity in temp_nx.nodes:
            filtered_nx = nx.ego_graph(temp_nx, target_entity, radius=2, undirected=True)
        else:
            filtered_nx = temp_nx

        # ── Populate PyVis Network from the filtered subgraph ──
        for node_id, node_attrs in filtered_nx.nodes(data=True):
            net.add_node(node_id, **node_attrs)
        for source, target, edge_attrs in filtered_nx.edges(data=True):
            net.add_edge(source, target, **edge_attrs)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w")
        net.save_graph(tmp.name)
        with open(tmp.name, "r") as f:
            html_content = f.read()
        return html_content

    if target_entity == "View Full Graph":
        st.info(f"Showing top 400 of {len(df)}; use Isolate Target Entity for more")

    st.markdown('<div class="graph-container">', unsafe_allow_html=True)
    graph_html = build_network_graph(df, threats_only_mode=threats_only, target_entity=target_entity)
    components.html(graph_html, height=600, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: WHY FLAGGED?
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown(
        '<div class="section-title"><span class="icon">🔍</span> Why Flagged?</div>'
        '<p class="section-subtitle">Select any transaction to inspect its dynamic feature percentiles and statistical deviations.</p>',
        unsafe_allow_html=True,
    )
    _tab_p = _get_palette()
    st.markdown(f'<p style="color:{_tab_p["muted"]};font-size:0.88rem;margin-bottom:0.2rem;">Pick one transaction and see exactly why the AI flagged it.</p>', unsafe_allow_html=True)
    with st.expander("ℹ️ How to read this"):
        st.markdown("Select a transaction to see its details and the plain-English reasons it was flagged. The feature table compares each value to the rest of the dataset using percentiles (e.g. '98th percentile' = higher than 98% of transactions). IP/ASN data shows where traffic was observed, not the sender's identity.")

    # Phase 3: network telemetry disclaimer (persistent, visible)
    st.markdown(
        """
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3);
                    border-radius: 0.6rem; padding: 0.7rem 1rem; margin-bottom: 0.8rem;">
            <span style="color: #fbbf24; font-size: 0.78rem; font-weight: 600;">
                &#9888;&#65039; Network observation correlation &mdash; not absolute identity attribution
                (Subject to VPN / NAT / Tor limits). IP telemetry identifies observation nodes,
                NOT sender identities. ASN data resolved offline (no external API).
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Transaction selector ──
    sorted_df = df.sort_values("risk_score", ascending=False)
    tx_options = [
        f"{row['txid'][:12]}... | Anomaly Score: {row['risk_score']:.1f}% | {row.get('detected_type', 'N/A')}"
        for _, row in sorted_df.iterrows()
    ]
    txid_map = {opt: row["txid"] for opt, (_, row) in zip(tx_options, sorted_df.iterrows())}

    selected_tx_label = st.selectbox(
        "Select Transaction",
        tx_options[:200],
        help="Sorted by Investigative Priority Index. Select any transaction to inspect.",
    )

    if selected_tx_label:
        selected_txid = txid_map[selected_tx_label]
        tx_row = df[df["txid"] == selected_txid].iloc[0]
        tx_idx = df[df["txid"] == selected_txid].index[0]
        feat_row = features_df.iloc[tx_idx]

        # ── Transaction detail card ──
        risk_val = tx_row["risk_score"]
        _row_tier = tx_row.get("risk_tier", "Low") if "risk_tier" in df.columns else ("Critical" if risk_val >= 80 else "High" if risk_val >= 60 else "Medium" if risk_val >= 40 else "Low")
        _tier_badge_map = {
            "Critical": ("badge-red", "CRITICAL"),
            "High": ("badge-amber", "HIGH"),
            "Medium": ("badge-blue", "MEDIUM"),
            "Low": ("badge-green", "LOW"),
        }
        _tb_cls, _tb_label = _tier_badge_map.get(_row_tier, ("badge-blue", "LOW"))
        risk_badge = f'<span class="badge {_tb_cls}">{_tb_label}</span>'

        detected_type = tx_row.get("detected_type", "Unknown")
        attack_badge_map = {
            "CoinJoin_Mixer": "badge-purple",
            "Peel_Chain": "badge-amber",
            "FanOut_Dispersal": "badge-cyan",
            "Fee_Spike": "badge-red",
            "Normal_P2P": "badge-blue",
            "Whitelisted_Institutional": "badge-green",
        }
        attack_badge_cls = attack_badge_map.get(detected_type, "badge-blue")

        _dp = _get_palette()
        st.markdown(
            f"""
            <div class="tx-detail-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
                    <div>
                        <div style="color: {_dp['muted']}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;">
                            Transaction Hash
                        </div>
                        <div class="tx-hash">{selected_txid}</div>
                    </div>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        {risk_badge}
                        <span class="badge {attack_badge_cls}">{detected_type}</span>
                        <span style="color: {_dp['muted']}; font-size: 0.72rem; margin-right: 4px;">Anomaly Score:</span>
                        <span style="color: {_dp['text']}; font-size: 1.4rem; font-weight: 800;">{risk_val:.1f}%</span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; font-size: 0.82rem;">
                    <div>
                        <div style="color: {_dp['muted']}; font-size: 0.68rem; text-transform: uppercase;">Source IP</div>
                        <div style="color: {_dp['text']}; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">{tx_row['src_ip']}</div>
                    </div>
                    <div>
                        <div style="color: {_dp['muted']}; font-size: 0.68rem; text-transform: uppercase;">Ports</div>
                        <div style="color: {_dp['text']};">{tx_row['src_port']} &rarr; {tx_row['dst_port']}</div>
                    </div>
                    <div>
                        <div style="color: {_dp['muted']}; font-size: 0.68rem; text-transform: uppercase;">Amount</div>
                        <div style="color: {_dp['text']};">{tx_row.get('total_amount_btc', 0):.6f} BTC</div>
                    </div>
                    <div>
                        <div style="color: {_dp['muted']}; font-size: 0.68rem; text-transform: uppercase;">Entity</div>
                        <div style="color: {_dp['text']};">{tx_row.get('entity_id', 'N/A')}</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.82rem; margin-top: 0.8rem;
                            background: rgba(59,130,246,0.06); border: 1px solid rgba(59,130,246,0.2); border-radius: 0.5rem; padding: 0.6rem 1rem;">
                    <div>
                        <div style="color: {_dp['muted']}; font-size: 0.68rem; text-transform: uppercase; font-weight: 600;">📡 Telemetry Observations</div>
                        <div style="color: #60a5fa; font-weight: 700; font-size: 0.95rem;">
                            {int(tx_row.get('ip_observation_count', 1))} <span style="font-size:0.75rem; font-weight: normal; color:{_dp['muted']};">observation(s)</span>
                        </div>
                    </div>
                    <div>
                        <div style="color: {_dp['muted']}; font-size: 0.68rem; text-transform: uppercase; font-weight: 600;">⏱️ Observation Spread</div>
                        <div style="color: #60a5fa; font-weight: 700; font-size: 0.95rem;">
                            {float(tx_row.get('observation_spread_s', 0.0)):.2f}s <span style="font-size:0.75rem; font-weight: normal; color:{_dp['muted']};">propagation window</span>
                        </div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.82rem; margin-top: 0.8rem;
                            background: rgba(245,158,11,0.05); border: 1px solid rgba(245,158,11,0.15); border-radius: 0.5rem; padding: 0.6rem 1rem;">
                    <div>
                        <div style="color: {_dp['muted']}; font-size: 0.68rem; text-transform: uppercase;">Geo (Offline ASN)</div>
                        <div style="color: #f59e0b;">{_display_geo(tx_row.get('geo_country'), 'geo')} &nbsp;&middot;&nbsp; {_display_geo(tx_row.get('asn'), 'asn')}</div>
                    </div>
                    <div>
                        <div style="color: {_dp['muted']}; font-size: 0.68rem; text-transform: uppercase;">Attribution Caveat</div>
                        <div style="color: #f59e0b; font-size: 0.72rem;">&#9888; Subject to VPN/NAT/Tor limits</div>
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
                <div style="color: #ef4444; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                            letter-spacing: 0.06em; margin-bottom: 0.3rem;">
                    🧠 AI Explanation
                </div>
                <div style="color: {_dp['text']}; font-size: 0.85rem; line-height: 1.5;">
                    {tx_row.get('explanation', 'No explanation available.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── 🧮 Score Breakdown ──
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title"><span class="icon">🧮</span> Score Breakdown</div>',
            unsafe_allow_html=True,
        )

        _sb_ml  = float(tx_row.get("score_ml", risk_val))
        _sb_pat = float(tx_row.get("score_pattern", 0))
        _sb_net = float(tx_row.get("score_network", 0))
        _sb_gra = float(tx_row.get("score_graph", 0))
        _sb_wlf = float(tx_row.get("whitelist_factor", 1.0))
        _sb_final = float(tx_row.get("risk_score", risk_val))

        from anomaly_engine import WEIGHTS as _W
        _contrib_ml  = round(_W["ml"]      * _sb_ml,  1)
        _contrib_pat = round(_W["pattern"] * _sb_pat, 1)
        _contrib_net = round(_W["network"] * _sb_net, 1)
        _contrib_gra = round(_W["graph"]   * _sb_gra, 1)
        _raw_total = _contrib_ml + _contrib_pat + _contrib_net + _contrib_gra
        _wl_discount = round(_sb_final - _raw_total, 1) if _sb_wlf < 1.0 else 0.0

        _breakdown_rows = [
            {"Component": f"ML anomaly ({_W['ml']:.0%})", "Value": _contrib_ml, "Color": "component"},
            {"Component": f"Pattern detectors ({_W['pattern']:.0%})", "Value": _contrib_pat, "Color": "component"},
            {"Component": f"Network risk ({_W['network']:.0%})", "Value": _contrib_net, "Color": "component"},
            {"Component": f"Graph / taint ({_W['graph']:.0%})", "Value": _contrib_gra, "Color": "component"},
        ]
        if _sb_wlf < 1.0:
            _breakdown_rows.append(
                {"Component": "Whitelist discount", "Value": _wl_discount, "Color": "discount"}
            )
        _breakdown_rows.append(
            {"Component": "Final priority", "Value": _sb_final, "Color": "final"}
        )
        _bd_df = pd.DataFrame(_breakdown_rows)
        _bd_order = [r["Component"] for r in _breakdown_rows]

        _bd_color_scale = alt.Scale(
            domain=["component", "discount", "final"],
            range=[_dp["accent2"], "#6b7280", "#ef4444"],
        )
        _bd_chart = alt.Chart(_bd_df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Component:N", sort=_bd_order, axis=alt.Axis(labelColor=_dp["text"], titleColor=_dp["text"])),
            x=alt.X("Value:Q", axis=alt.Axis(labelColor=_dp["text"], titleColor=_dp["text"], gridColor=_dp.get("chart_grid", "rgba(255,255,255,0.06)"))),
            color=alt.Color("Color:N", scale=_bd_color_scale, legend=None),
        ).properties(height=max(len(_breakdown_rows) * 36, 180))

        # Add labels
        _bd_text = alt.Chart(_bd_df).mark_text(
            align="left", dx=4, fontSize=11, color=_dp["text"],
        ).encode(
            y=alt.Y("Component:N", sort=_bd_order),
            x=alt.X("Value:Q"),
            text=alt.Text("Value:Q", format=".1f"),
        )

        st.altair_chart(
            (_bd_chart + _bd_text).configure(
                background=_dp.get("chart_bg", "transparent")
            ).configure_view(strokeWidth=0),
            width="stretch",
        )

        # ── Top Feature Deviations chart ──
        telemetry_str_bd = tx_row.get("telemetry", "")
        if isinstance(telemetry_str_bd, str) and telemetry_str_bd.strip():
            try:
                _telem_data = json.loads(telemetry_str_bd)
                if _telem_data:
                    _dev_rows = []
                    for _t in _telem_data[:6]:
                        _fname = _t.get("label", _t.get("feature_name", ""))
                        _pctile = float(_t.get("percentile_rank", 50))
                        _dev_rows.append({"Feature": _fname, "Deviation": round(_pctile - 50, 1)})
                    if _dev_rows:
                        _dev_df = pd.DataFrame(_dev_rows)
                        _dev_order = [r["Feature"] for r in _dev_rows]
                        _dev_chart = alt.Chart(_dev_df).mark_bar(cornerRadiusEnd=3).encode(
                            y=alt.Y("Feature:N", sort=_dev_order, axis=alt.Axis(labelColor=_dp["text"], titleColor=_dp["text"])),
                            x=alt.X("Deviation:Q", axis=alt.Axis(labelColor=_dp["text"], titleColor=_dp["text"], gridColor=_dp.get("chart_grid", "rgba(255,255,255,0.06)"))),
                            color=alt.condition(
                                alt.datum.Deviation > 0,
                                alt.value("#ef4444"),
                                alt.value("#22c55e"),
                            ),
                        ).properties(height=max(len(_dev_rows) * 34, 150), title=alt.TitleParams(
                            text="Top feature deviations (percentile − 50)",
                            color=_dp["text"], fontSize=13,
                        ))

                        _dev_text = alt.Chart(_dev_df).mark_text(
                            fontSize=10, color=_dp["text"],
                        ).encode(
                            y=alt.Y("Feature:N", sort=_dev_order),
                            x=alt.X("Deviation:Q"),
                            text=alt.Text("Deviation:Q", format=".1f"),
                        ).transform_calculate(
                            align="datum.Deviation >= 0 ? 'left' : 'right'",
                            dx="datum.Deviation >= 0 ? 4 : -4",
                        )

                        st.altair_chart(
                            (_dev_chart + _dev_text).configure(
                                background=_dp.get("chart_bg", "transparent")
                            ).configure_view(strokeWidth=0),
                            width="stretch",
                        )
            except Exception:
                pass  # Fall through to telemetry table

        # ── Telemetry Table ──
        telemetry_str = tx_row.get("telemetry", "")
        if isinstance(telemetry_str, str) and telemetry_str.strip():
            try:
                telemetry_data = json.loads(telemetry_str)
                if telemetry_data:
                    _rows_html = []
                    for item in telemetry_data:
                        fname = item.get("label", item.get("feature_name", ""))
                        obs = item.get("observed_value", 0.0)
                        p_rank = item.get("percentile_rank", 0.0)
                        med = item.get("dataset_median", 0.0)
                        reason = item.get("audit_reason", "")
                        _rows_html.append(
                            f'<tr><td><b>{fname}</b></td>'
                            f'<td style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;">{obs:.4f}</td>'
                            f'<td>{p_rank:.1f}th Percentile (Med: {med:.4f})</td>'
                            f'<td style="color: #f87171;">{reason}</td></tr>'
                        )
                    _table_html = (
                        '<table class="heuristic-table">'
                        '<thead><tr><th>Feature</th><th>Value</th><th>Statistical Deviation</th><th>Audit Reason</th></tr></thead>'
                        '<tbody>' + ''.join(_rows_html) + '</tbody>'
                        '</table>'
                    )
                    st.markdown(_table_html, unsafe_allow_html=True)
                else:
                    st.info("No extreme statistical deviations detected for this transaction.")
            except Exception as e:
                st.error("Error parsing telemetry data.")
        else:
            st.info("No structured telemetry data available for this transaction.")

        # ── 🧭 Follow the Money ──
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title"><span class="icon">🧭</span> Follow the Money</div>',
            unsafe_allow_html=True,
        )

        import flow_tracer

        _ftm_col1, _ftm_col2 = st.columns([1, 1])
        with _ftm_col1:
            trace_direction = st.radio(
                "Trace Direction",
                ["Forward", "Backward"],
                horizontal=True,
                key=f"ftm_dir_{selected_txid}",
            )
        with _ftm_col2:
            trace_hops = st.slider(
                "Hops",
                min_value=1,
                max_value=6,
                value=4,
                key=f"ftm_hops_{selected_txid}",
            )

        if trace_direction == "Forward":
            hops_data = flow_tracer.trace_forward(df, selected_txid, max_hops=trace_hops)
        else:
            hops_data = flow_tracer.trace_backward(df, selected_txid, max_hops=trace_hops)

        if not hops_data:
            st.info("No transaction hops found for this identifier.")
        else:
            # ── Next Step Box ──
            last_hop = hops_data[-1]
            last_role = last_hop.get("role")
            if last_role == "Exchange endpoint":
                _ent = last_hop.get("whitelisted_entity") or "regulated exchange"
                st.success(f"Funds reached {_ent}. Next step: lawful KYC request to this VASP.")
            elif last_role == "Mixer":
                st.warning("Trail enters a mixer; further tracing has low confidence.")

            # ── Altair Timeline ──
            # x = timestamp, y = BTC moving at each hop, points coloured by role, tooltip with txid, delta_t and retained_pct
            _tl_rows = []
            for h in hops_data:
                _tl_rows.append({
                    "Hop": h["hop"],
                    "txid": h["txid"],
                    "timestamp": pd.to_datetime(h["timestamp"]) if pd.notna(h.get("timestamp")) else pd.Timestamp.now(),
                    "btc_in": h["btc_in"],
                    "role": h["role"],
                    "delta_t": h["delta_t_minutes"],
                    "retained_pct": h["retained_pct"],
                })
            _tl_df = pd.DataFrame(_tl_rows)

            _role_domain = [
                "Source",
                "Layering (peel)",
                "Mixer",
                "Exchange endpoint",
                "Dispersal",
                "Consolidation",
                "Intermediary",
            ]
            _role_range = [
                "#3B82F6",
                "#F59E0B",
                "#EF4444",
                "#10B981",
                "#8B5CF6",
                "#EC4899",
                "#64748B",
            ]

            _tl_points = alt.Chart(_tl_df).mark_circle(size=140).encode(
                x=alt.X("timestamp:T", title="Timeline (Timestamp)"),
                y=alt.Y("btc_in:Q", title="BTC Moving at Hop"),
                color=alt.Color(
                    "role:N",
                    scale=alt.Scale(domain=_role_domain, range=_role_range),
                    legend=alt.Legend(title="Role", orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("txid:N", title="TxID"),
                    alt.Tooltip("role:N", title="Role"),
                    alt.Tooltip("delta_t:Q", title="delta_t (min)", format=".1f"),
                    alt.Tooltip("retained_pct:Q", title="retained (%)", format=".2f"),
                    alt.Tooltip("btc_in:Q", title="BTC Moving", format=".4f"),
                ],
            )
            _tl_line = alt.Chart(_tl_df).mark_line(
                strokeDash=[3, 3], color="#94A3B8", strokeWidth=1.5
            ).encode(
                x="timestamp:T",
                y="btc_in:Q",
            )
            _timeline_chart = (_tl_line + _tl_points).properties(
                height=260,
                title=alt.TitleParams(
                    text=f"Fund Flow Timeline ({trace_direction} Trace, {len(hops_data)} Hops)",
                    color=_dp["text"],
                    fontSize=12,
                ),
            )
            st.altair_chart(
                _timeline_chart.configure(
                    background=_dp.get("chart_bg", "transparent")
                ).configure_view(strokeWidth=0),
                width="stretch",
            )

            # ── Hop Table ──
            _hop_table_rows = [
                {
                    "Hop": h["hop"],
                    "Role": h["role"],
                    "TxID": f"{h['txid'][:14]}...",
                    "Timestamp": str(h["timestamp"]),
                    "Δt (min)": h["delta_t_minutes"],
                    "BTC In": f"{h['btc_in']:.4f}",
                    "Largest Output (BTC)": f"{h['largest_output_btc']:.4f}",
                    "Retained %": f"{h['retained_pct']:.2f}%",
                    "Peeled (BTC)": f"{h['peeled_btc']:.4f}",
                }
                for h in hops_data
            ]
            st.dataframe(pd.DataFrame(_hop_table_rows), width="stretch", hide_index=True)

            # ── Small PyVis Graph of Only the Path ──
            # with the selected tx in red and the endpoint in green
            try:
                path_net = Network(
                    height="280px",
                    width="100%",
                    bgcolor=_dp.get("chart_bg", "#0f172a"),
                    font_color=_dp["text"],
                    directed=True,
                )

                for i, h in enumerate(hops_data):
                    is_start = (i == 0)
                    is_end = (i == len(hops_data) - 1 and len(hops_data) > 1)

                    if is_start:
                        n_color = "#ef4444"  # Red for selected tx
                    elif is_end:
                        n_color = "#22c55e"  # Green for endpoint
                    else:
                        n_color = "#3b82f6"  # Blue for intermediate

                    lbl = f"Hop {h['hop']}: {h['role']}\n{h['txid'][:8]}...\n{h['btc_in']:.2f} BTC"
                    title_tip = (
                        f"TxID: {h['txid']}\n"
                        f"Role: {h['role']}\n"
                        f"BTC In: {h['btc_in']:.4f}\n"
                        f"Retained: {h['retained_pct']:.2f}%\n"
                        f"Peeled: {h['peeled_btc']:.4f} BTC\n"
                        f"Δt: {h['delta_t_minutes']:.1f} min\n"
                        f"Time: {h['timestamp']}"
                    )
                    path_net.add_node(
                        h["txid"],
                        label=lbl,
                        title=title_tip,
                        color=n_color,
                        shape="box",
                        font={"size": 11, "color": "#ffffff"},
                    )

                for i in range(len(hops_data) - 1):
                    src_h = hops_data[i]
                    dst_h = hops_data[i + 1]
                    if trace_direction == "Forward":
                        edge_src = src_h["txid"]
                        edge_dst = dst_h["txid"]
                    else:
                        edge_src = dst_h["txid"]
                        edge_dst = src_h["txid"]

                    edge_lbl = f"{dst_h['btc_in']:.2f} BTC ({dst_h['delta_t_minutes']:.0f}m)"
                    path_net.add_edge(
                        edge_src,
                        edge_dst,
                        label=edge_lbl,
                        color={"color": "#94a3b8", "highlight": "#38bdf8"},
                        arrows="to",
                        font={"size": 10, "color": _dp["muted"]},
                    )

                path_net.set_options("""
                {
                  "physics": {
                    "hierarchicalRepulsion": { "nodeDistance": 130 },
                    "solver": "hierarchicalRepulsion"
                  },
                  "layout": {
                    "hierarchical": { "enabled": true, "direction": "LR", "sortMethod": "directed" }
                  }
                }
                """)
                _path_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w")
                path_net.save_graph(_path_tmp.name)
                with open(_path_tmp.name, "r") as _pf:
                    _path_html = _pf.read()
                components.html(_path_html, height=290, scrolling=False)
            except Exception as _p_err:
                st.caption(f"Path visualizer note: {_p_err}")

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
                st.dataframe(pd.DataFrame(in_data), hide_index=True, width="stretch")

        with addr_col2:
            st.markdown("**Output Addresses & Amounts**")
            out_addrs = str(tx_row.get("output_addresses", "")).split("|")
            out_amounts = _parse_pipe_amounts(tx_row.get("output_amounts"))
            out_data = []
            for i, addr in enumerate(out_addrs):
                amt = out_amounts[i] if i < len(out_amounts) else 0.0
                out_data.append({"Address": addr.strip(), "Amount (BTC)": f"{amt:.8f}"})
            if out_data:
                st.dataframe(pd.DataFrame(out_data), hide_index=True, width="stretch")

        st.markdown(
            f"""
            <div style="text-align: center; color: {_dp['muted']}; font-size: 0.78rem; margin-top: 0.5rem;">
                Fee: <b style="color: #f59e0b;">{tx_row['fee']:.8f} BTC</b> &nbsp;&middot;&nbsp;
                Script: <b style="color: {_dp['accent2']};">{tx_row['script_type']}</b> &nbsp;&middot;&nbsp;
                First-Seen Node: <b style="color: #3b82f6;">{_display_geo(tx_row.get('geo_country'), 'geo')}</b>
                &nbsp;&middot;&nbsp;
                ASN: <b style="color: #22c55e;">{_display_geo(tx_row.get('asn'), 'asn')}</b>
            </div>
            <div style="text-align: center; color: #92400e; font-size: 0.70rem; margin-top: 0.3rem;
                        background: rgba(245,158,11,0.06); border-radius: 0.4rem; padding: 0.3rem;">
                &#9888; Network observation correlation &mdash; not absolute identity attribution
                (Subject to VPN / NAT / Tor limits)
            </div>
            """,
            unsafe_allow_html=True,
        )



# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    _p5 = _get_palette()
    st.markdown(
        f'<p style="color:{_p5["muted"]};font-size:0.88rem;margin-bottom:0.6rem;">'
        "Supervised Calibrated Random Forest evaluation and benchmark against baseline detectors."
        "</p>",
        unsafe_allow_html=True,
    )

    # Check if dataset has labels (synthetic data)
    if "attack_type" not in df.columns:
        _sha256 = ""
        _sha_path = "models/rf_model.sha256"
        if os.path.isfile(_sha_path):
            try:
                with open(_sha_path, "r") as _sf:
                    _sha256 = _sf.read().strip()
            except Exception:
                _sha256 = "unknown"
        elif os.path.isfile("models/rf_model.joblib"):
            try:
                import hashlib
                with open("models/rf_model.joblib", "rb") as _jf:
                    _sha256 = hashlib.sha256(_jf.read()).hexdigest()
            except Exception:
                _sha256 = "unknown"
        else:
            _sha256 = "unknown"

        st.info(f"Upload labelled data to evaluate. Using the pre-trained model {_sha256[:12]}.")
    else:
        st.caption(
            "Measured on held-out synthetic data from MITHYA's own generator. "
            "Real-world performance will be lower and must be validated on labelled case data."
        )

        import model_eval

        # Cache evaluation in session_state so tab switching is instant
        _eval_key = f"perf_eval_{len(df)}_{file_hash}"
        if _eval_key not in st.session_state:
            with st.spinner("🧠 Evaluating model performance and calibration..."):
                st.session_state[_eval_key] = model_eval.run_evaluation_and_training(
                    df, features_df, save_model=True, run_gen_test=True
                )

        _eval_res = st.session_state[_eval_key]
        if _eval_res is not None:
            _ho = _eval_res["holdout"]
            _rf = _ho["rf"]

            # ── 1. Metric cards: ROC-AUC, PR-AUC, Precision@25, F1, ECE ──
            st.markdown(
                f"""
                <div class="bento-grid" style="grid-template-columns: repeat(5, 1fr); margin-bottom: 1.2rem;">
                    <div class="bento-card card-blue">
                        <div class="card-icon">🎯</div>
                        <div class="card-label">ROC-AUC</div>
                        <div class="card-value">{_rf['roc_auc']:.4f}</div>
                        <div class="card-sub">Discrimination curve</div>
                    </div>
                    <div class="bento-card card-purple">
                        <div class="card-icon">⚖️</div>
                        <div class="card-label">PR-AUC</div>
                        <div class="card-value">{_rf['pr_auc']:.4f}</div>
                        <div class="card-sub">Precision-Recall trade-off</div>
                    </div>
                    <div class="bento-card card-red">
                        <div class="card-icon">🔝</div>
                        <div class="card-label">Precision@25</div>
                        <div class="card-value">{_rf['precision_at_25']:.4f}</div>
                        <div class="card-sub">Top-25 ranked accuracy</div>
                    </div>
                    <div class="bento-card card-orange">
                        <div class="card-icon">🏆</div>
                        <div class="card-label">F1-Score</div>
                        <div class="card-value">{_rf['f1']:.4f}</div>
                        <div class="card-sub">Harmonic mean P & R</div>
                    </div>
                    <div class="bento-card card-green">
                        <div class="card-icon">📉</div>
                        <div class="card-label">ECE (10 Bins)</div>
                        <div class="card-value">{_ho['ece_after']:.4f}</div>
                        <div class="card-sub">Before: {_ho['ece_before']:.4f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── 2. Hold-out Benchmark Table (4 Methods) ──
            st.markdown(f'<h4 style="color:{_p5["accent"]};margin-top:1rem;margin-bottom:0.4rem;">Hold-Out Evaluation (70/30 Stratified Split)</h4>', unsafe_allow_html=True)
            st.dataframe(_ho["table"], width="stretch", hide_index=True)

            # ── 3. Altair ROC curve and PR curve with the 4 methods as coloured lines ──
            _roc_pr_col1, _roc_pr_col2 = st.columns(2)

            _all_methods = [_ho["rf"], _ho["isolation_forest"], _ho["rules"], _ho["fused"]]

            # Build ROC DataFrame
            _roc_rows = []
            for _m in _all_methods:
                for _fpr, _tpr in zip(_m["fpr_curve"], _m["tpr_curve"]):
                    _roc_rows.append({"FPR": _fpr, "TPR": _tpr, "Method": _m["method"]})
            _roc_df = pd.DataFrame(_roc_rows)

            _diag_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
            _diag_chart = alt.Chart(_diag_df).mark_line(
                strokeDash=[4, 4], color="#94A3B8"
            ).encode(x="x:Q", y="y:Q")

            _color_domain = ["Supervised RF (Calibrated)", "Isolation Forest only", "Rules only", "Fused Score"]
            _color_range = ["#2DD4BF", "#F59E0B", "#8B5CF6", "#3B82F6"]

            with _roc_pr_col1:
                st.markdown(f'<p style="font-weight:600;font-size:0.95rem;color:{_p5["text"]};">Receiver Operating Characteristic (ROC)</p>', unsafe_allow_html=True)
                _roc_line = alt.Chart(_roc_df).mark_line(strokeWidth=2.2).encode(
                    x=alt.X("FPR:Q", title="False Positive Rate (FPR)", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("TPR:Q", title="True Positive Rate (TPR / Recall)", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Method:N", scale=alt.Scale(domain=_color_domain, range=_color_range), legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Method:N", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
                )
                _roc_chart = (_roc_line + _diag_chart).properties(height=300)
                st.altair_chart(_roc_chart, width="stretch")

            # Build PR DataFrame
            _pr_rows = []
            for _m in _all_methods:
                for _rec, _prec in zip(_m["pr_recall_curve"], _m["pr_precision_curve"]):
                    _pr_rows.append({"Recall": _rec, "Precision": _prec, "Method": _m["method"]})
            _pr_df = pd.DataFrame(_pr_rows)

            with _roc_pr_col2:
                st.markdown(f'<p style="font-weight:600;font-size:0.95rem;color:{_p5["text"]};">Precision-Recall Curves</p>', unsafe_allow_html=True)
                _pr_line = alt.Chart(_pr_df).mark_line(strokeWidth=2.2).encode(
                    x=alt.X("Recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Method:N", scale=alt.Scale(domain=_color_domain, range=_color_range), legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Method:N", alt.Tooltip("Recall:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")],
                )
                st.altair_chart(_pr_line.properties(height=300), width="stretch")

            # ── 4. Confusion matrix heatmap and Calibration plot ──
            _cm_cal_col1, _cm_cal_col2 = st.columns(2)

            with _cm_cal_col1:
                st.markdown(f'<p style="font-weight:600;font-size:0.95rem;color:{_p5["text"]};">Confusion Matrix (Calibrated RF)</p>', unsafe_allow_html=True)
                _cm = _ho["confusion_matrix"]
                _cm_data = pd.DataFrame([
                    {"Actual": "Normal (0)", "Predicted": "Normal (0)", "Count": _cm[0][0]},
                    {"Actual": "Normal (0)", "Predicted": "Threat (1)", "Count": _cm[0][1]},
                    {"Actual": "Threat (1)", "Predicted": "Normal (0)", "Count": _cm[1][0]},
                    {"Actual": "Threat (1)", "Predicted": "Threat (1)", "Count": _cm[1][1]},
                ])
                _max_cnt = max([row["Count"] for row in _cm_data.to_dict("records")] or [1])
                _rect = alt.Chart(_cm_data).mark_rect().encode(
                    x=alt.X("Predicted:N", title="Predicted Class", sort=["Normal (0)", "Threat (1)"]),
                    y=alt.Y("Actual:N", title="Actual Class", sort=["Threat (1)", "Normal (0)"]),
                    color=alt.Color("Count:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
                    tooltip=["Actual", "Predicted", "Count"],
                )
                _text = alt.Chart(_cm_data).mark_text(baseline="middle", fontSize=18, fontWeight="bold").encode(
                    x="Predicted:N",
                    y=alt.Y("Actual:N", sort=["Threat (1)", "Normal (0)"]),
                    text=alt.Text("Count:Q"),
                    color=alt.condition(alt.datum.Count > _max_cnt / 2, alt.value("white"), alt.value("black")),
                )
                st.altair_chart((_rect + _text).properties(height=280), width="stretch")

            with _cm_cal_col2:
                st.markdown(f'<p style="font-weight:600;font-size:0.95rem;color:{_p5["text"]};">Calibration Reliability Curve (ECE: {_ho["ece_after"]:.4f})</p>', unsafe_allow_html=True)
                _cal_cal = _ho["calibration_curve_cal"]
                _cal_uncal = _ho["calibration_curve_uncal"]
                _cal_rows = []
                for _pred_p, _true_p in zip(_cal_cal["prob_pred"], _cal_cal["prob_true"]):
                    _cal_rows.append({"Predicted": _pred_p, "Actual": _true_p, "Calibration": "Calibrated (Isotonic)"})
                for _pred_p, _true_p in zip(_cal_uncal["prob_pred"], _cal_uncal["prob_true"]):
                    _cal_rows.append({"Predicted": _pred_p, "Actual": _true_p, "Calibration": "Uncalibrated RF"})
                _cal_df = pd.DataFrame(_cal_rows)

                _cal_line = alt.Chart(_cal_df).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X("Predicted:Q", title="Mean Predicted Probability", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Actual:Q", title="Fraction of Positives", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Calibration:N", scale=alt.Scale(domain=["Calibrated (Isotonic)", "Uncalibrated RF"], range=["#2DD4BF", "#F59E0B"]), legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Calibration:N", alt.Tooltip("Predicted:Q", format=".3f"), alt.Tooltip("Actual:Q", format=".3f")],
                )
                _cal_diag = alt.Chart(_diag_df).mark_line(
                    strokeDash=[4, 4], color="#94A3B8"
                ).encode(x="x:Q", y="y:Q")
                st.altair_chart((_cal_line + _cal_diag).properties(height=280), width="stretch")

            # ── 5. Per-Attack-Type Recall and Generalisation Results ──
            _tab5_bottom_col1, _tab5_bottom_col2 = st.columns(2)

            with _tab5_bottom_col1:
                st.markdown(f'<p style="font-weight:600;font-size:0.95rem;color:{_p5["text"]};">Per-Attack-Type Recall</p>', unsafe_allow_html=True)
                _ptr = _ho["per_type_recall"]
                _ptr_rows = [
                    {
                        "Threat Vector": _atk,
                        "Recall Rate": f"{_rec * 100:.1f}%",
                        "Status": "✅ Optimal" if _rec >= 0.90 else "⚠️ Flagged",
                    }
                    for _atk, _rec in _ptr.items()
                ]
                st.dataframe(pd.DataFrame(_ptr_rows), width="stretch", hide_index=True)

            with _tab5_bottom_col2:
                st.markdown(f'<p style="font-weight:600;font-size:0.95rem;color:{_p5["text"]};">Generalisation Test (Unseen 1500-Record Dataset, Seed=999)</p>', unsafe_allow_html=True)
                if _eval_res.get("generalisation") and "table" in _eval_res["generalisation"]:
                    st.dataframe(_eval_res["generalisation"]["table"], width="stretch", hide_index=True)
                else:
                    st.info("Generalisation test pending.")

            st.caption(
                "Measured on held-out synthetic data from MITHYA's own generator. "
                "Real-world performance will be lower and must be validated on labelled case data."
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: CLUSTERS & COMMUNITIES
# ═══════════════════════════════════════════════════════════════════════════

with tab6:
    _p6 = _get_palette()
    st.markdown(
        f'<p style="color:{_p6["muted"]};font-size:0.88rem;margin-bottom:0.6rem;">'
        "Entity-level behavioural clustering via HDBSCAN and structural community detection via Louvain modularity."
        "</p>",
        unsafe_allow_html=True,
    )

    import clustering

    _clustering_key = f"clustering_profile_{len(df)}_{file_hash}"
    if _clustering_key not in st.session_state:
        with st.spinner("👥 Computing behavioural clusters and graph communities..."):
            _, _profile_df = clustering.apply_clustering_to_transactions(df, features_df)
            st.session_state[_clustering_key] = _profile_df
    _profile_df = st.session_state[_clustering_key]

    # Metrics overview cards
    _n_entities = len(_profile_df)
    _n_clusters = _profile_df["behaviour_cluster"].nunique() if "behaviour_cluster" in _profile_df.columns else 0
    _n_outliers = int((_profile_df["behaviour_cluster"] == "Behavioural outlier").sum()) if "behaviour_cluster" in _profile_df.columns else 0
    _n_comms = _profile_df["graph_community"].nunique() if "graph_community" in _profile_df.columns else 0

    st.markdown(
        f"""
        <div class="bento-grid" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 1.2rem;">
            <div class="bento-card card-blue">
                <div class="card-icon">👥</div>
                <div class="card-label">Profiled Entities</div>
                <div class="card-value">{_n_entities:,}</div>
                <div class="card-sub">Common-Input clustered wallets</div>
            </div>
            <div class="bento-card card-purple">
                <div class="card-icon">🧩</div>
                <div class="card-label">HDBSCAN Clusters</div>
                <div class="card-value">{_n_clusters}</div>
                <div class="card-sub">Density-based behavioural groupings</div>
            </div>
            <div class="bento-card card-red">
                <div class="card-icon">🚨</div>
                <div class="card-label">Outlier Entities</div>
                <div class="card-value">{_n_outliers}</div>
                <div class="card-sub">Atypical behaviour profiles</div>
            </div>
            <div class="bento-card card-green">
                <div class="card-icon">🕸️</div>
                <div class="card-label">Graph Communities</div>
                <div class="card-value">{_n_comms}</div>
                <div class="card-sub">Louvain modularity partitions</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 1. Altair Scatter of PCA x/y ──
    # colour = cluster, size = mean_risk_score, tooltip = entity details
    st.markdown(f'<h4 style="color:{_p6["accent"]};margin-top:0.8rem;margin-bottom:0.4rem;">Entity Behavioural Space (2D PCA Projection)</h4>', unsafe_allow_html=True)

    _pca_chart = alt.Chart(_profile_df).mark_circle().encode(
        x=alt.X("pca_x:Q", title="PCA Dimension 1 (Variance Axis 1)"),
        y=alt.Y("pca_y:Q", title="PCA Dimension 2 (Variance Axis 2)"),
        color=alt.Color(
            "behaviour_cluster:N",
            scale=alt.Scale(scheme="tableau20"),
            legend=alt.Legend(title="Behaviour Cluster", orient="top", columns=4),
        ),
        size=alt.Size(
            "mean_risk_score:Q",
            title="Avg Risk Score",
            scale=alt.Scale(domain=[0, 100], range=[40, 360]),
            legend=alt.Legend(title="Avg Risk", orient="right"),
        ),
        tooltip=[
            alt.Tooltip("entity_id:N", title="Entity ID"),
            alt.Tooltip("behaviour_cluster:N", title="Cluster"),
            alt.Tooltip("graph_community:N", title="Community"),
            alt.Tooltip("tx_count:Q", title="Transaction Count"),
            alt.Tooltip("total_btc:Q", title="Total BTC", format=".2f"),
            alt.Tooltip("mean_risk_score:Q", title="Mean Risk Score", format=".1f"),
            alt.Tooltip("night_share:Q", title="Night Share (00-05h)", format=".1%"),
            alt.Tooltip("distinct_src_ips:Q", title="Distinct IPs"),
            alt.Tooltip("mean_fan_out:Q", title="Mean Fan-Out", format=".1f"),
            alt.Tooltip("mean_port_risk:Q", title="Mean Port Risk", format=".1f"),
        ],
    ).properties(height=380)

    st.altair_chart(
        _pca_chart.configure(
            background=_p6.get("chart_bg", "transparent")
        ).configure_view(strokeWidth=0),
        width="stretch",
    )

    # ── 2. Cluster Table and Community Table ──
    _tbl_col1, _tbl_col2 = st.columns(2)

    _cluster_summary = clustering.build_cluster_summary_table(df, _profile_df)
    _comm_summary = clustering.build_community_summary_table(df, _profile_df)

    with _tbl_col1:
        st.markdown(f'<h5 style="color:{_p6["text"]};margin-bottom:0.4rem;">Behavioural Clusters Summary</h5>', unsafe_allow_html=True)
        st.dataframe(_cluster_summary, width="stretch", hide_index=True)
        st.download_button(
            "📥 Download Cluster Table (CSV)",
            data=_cluster_summary.to_csv(index=False),
            file_name="behaviour_clusters_summary.csv",
            mime="text/csv",
            key="btn_download_cluster_csv",
        )

    with _tbl_col2:
        st.markdown(f'<h5 style="color:{_p6["text"]};margin-bottom:0.4rem;">Graph Communities Summary (Louvain)</h5>', unsafe_allow_html=True)
        st.dataframe(_comm_summary, width="stretch", hide_index=True)
        st.download_button(
            "📥 Download Community Table (CSV)",
            data=_comm_summary.to_csv(index=False),
            file_name="graph_communities_summary.csv",
            mime="text/csv",
            key="btn_download_comm_csv",
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

_fp = _get_palette()
st.markdown(
    f"""
    <div class="fancy-divider"></div>
    <div style="text-align: center; padding: 0.8rem 0 1.5rem 0;">
        <p style="color: {_fp['muted']}; font-size: 0.75rem; font-weight: 400;">
            MITHYA — Crypto Forensic Intelligence Engine &nbsp;·&nbsp;
            Graph-Aware IsolationForest &nbsp;·&nbsp;
            21-Feature Behavioral Analysis &nbsp;·&nbsp;
            Built for Smart India Hackathon 2026
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
