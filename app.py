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
    st.session_state.dark_mode = False


def _get_palette() -> dict:
    return THEME_DARK if st.session_state.dark_mode else THEME_LIGHT


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
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")
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


# ---------------------------------------------------------------------------
# Data loading & pipeline execution
# ---------------------------------------------------------------------------

# Handle custom whitelist upload
if uploaded_whitelist is not None:
    custom_wl = pd.read_csv(uploaded_whitelist)
    custom_wl.to_csv("institutional_whitelist.csv", index=False)


def run_ai_on_upload(adapter, raw_df: pd.DataFrame, cont: float):
    """Run the full AI pipeline via the polymorphic adapter, then enrich with offline Geo-ASN."""
    enriched_df, _model, features = adapter.run_pipeline_from_df(
        raw_df, contamination=cont,
    )
    # Phase 4: offline ASN enrichment (no network I/O)
    if _HAS_GEO_ASN and "src_ip" in enriched_df.columns:
        enriched_df = _geo_enrich_df(enriched_df, ip_col="src_ip", inplace=False)
    elif "asn" not in enriched_df.columns:
        enriched_df["asn"] = "AS0"
    return enriched_df, features


import hashlib

# Determine data source and load
file_hash = "N/A"
ingest_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if uploaded_file is not None:
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
    # Determine default dataset: prefer synthetic_transactions.csv, fall back to bitcoin_traffic.csv
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
mixer_count = df["attack_type"].eq("CoinJoin_Mixer").sum() if "attack_type" in df.columns else 0
peel_count = df["attack_type"].eq("Peel_Chain").sum() if "attack_type" in df.columns else 0
fanout_count = df["attack_type"].eq("FanOut_Dispersal").sum() if "attack_type" in df.columns else 0
feespike_count = df["attack_type"].eq("Fee_Spike").sum() if "attack_type" in df.columns else 0


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
<span class="mithya-pill">SIH 2026</span>
</div>
</div>
    """,
    unsafe_allow_html=True,
)

source_label = (
    f"📂 Live analysis of uploaded file ({total_tx} tx)"
    if data_source == "uploaded"
    else f"💾 Default dataset: {_default_path} ({total_tx} tx)"
)
source_color = "#22c55e" if data_source == "uploaded" else "#3b82f6"

st.markdown(
    f'<p style="text-align:center; color:{source_color}; font-size:0.78rem; '
    f'font-weight:500; margin-top:-0.5rem;">{source_label}</p>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-TAB ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🚨 Suspicious Transactions",
    "🕸️ Network Graph",
    "🔍 Why Flagged?",
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
                <div class="card-label">Max Investigative Priority Index</div>
                <div class="card-value">{max_risk:.1f}%</div>
                <div class="card-sub">Highest anomaly deviation from baseline</div>
            </div>
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
        if "attack_type" in df.columns:
            attack_counts = df["attack_type"].value_counts().reset_index()
            attack_counts.columns = ["Attack Type", "Count"]
            chart1 = alt.Chart(attack_counts).mark_bar(
                cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=_cp["accent2"]
            ).encode(
                x=alt.X("Attack Type:N", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], labelAngle=-30)),
                y=alt.Y("Count:Q", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], gridColor=_cp["chart_grid"])),
            ).properties(height=300).configure(background=_cp["chart_bg"]).configure_view(strokeWidth=0)
            st.altair_chart(chart1, use_container_width=True)

    with col_chart2:
        st.markdown(
            '<div class="section-title"><span class="icon">📈</span> Risk Score Distribution</div>',
            unsafe_allow_html=True,
        )
        if total_flagged > 0:
            _bins = range(0, 101, 10)
            _cut = pd.cut(flagged_df["risk_score"], bins=_bins, include_lowest=True, right=True)
            risk_hist = _cut.value_counts().sort_index().reset_index()
            risk_hist.columns = ["Risk Bin", "Count"]
            risk_hist["Risk Bin"] = [f"{b.left:.0f}–{b.right:.0f}" for b in risk_hist["Risk Bin"]]
            chart2 = alt.Chart(risk_hist).mark_bar(
                cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#ef4444"
            ).encode(
                x=alt.X("Risk Bin:N", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], labelAngle=-30)),
                y=alt.Y("Count:Q", axis=alt.Axis(labelColor=_cp["chart_text"], titleColor=_cp["chart_text"], gridColor=_cp["chart_grid"])),
            ).properties(height=300).configure(background=_cp["chart_bg"]).configure_view(strokeWidth=0)
            st.altair_chart(chart2, use_container_width=True)
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
        st.download_button(
            label="📄 Generate Evidence Dossier",
            data=report_text.encode("utf-8"),
            file_name=f"mithya_forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
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
        "txid", "risk_score", "entity_id", "cluster_confidence", "explanation",
        "src_port", "dst_port", "total_amount_btc", "fee",
        "script_type", "geo_country", "asn",
    ]
    if "attack_type" in view_df.columns:
        display_cols.insert(3, "attack_type")

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
        "entity_id": "Entity",
        "attack_type": "Attack Type",
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
    styled_df = (
        display_df.style
        .background_gradient(subset=[prio_col], cmap="YlOrRd", vmin=0, vmax=100)
        .format({"Amount (BTC)": "{:.6f}", prio_col: "{:.1f}", "Fee": "{:.8f}"})
        .set_properties(**{"background-color": _tp["df_bg"], "color": _tp["df_text"]})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", _tp["card"]), ("color", _tp["text"])]},
        ])
    )
    st.dataframe(
        styled_df,
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
            <span style="color:#3b82f6; font-weight:600;">● Blue = context nodes</span>
        </p>
        """,
        unsafe_allow_html=True,
    )
    _tab_p = _get_palette()
    st.markdown(f'<p style="color:{_tab_p["muted"]};font-size:0.88rem;margin-bottom:0.2rem;">See which wallets, IPs, and transactions are connected.</p>', unsafe_allow_html=True)
    with st.expander("ℹ️ How to read this"):
        st.markdown("Nodes are wallets, IP addresses and transactions; lines show how funds and observations connect. Red = flagged anomalies, amber = transactions, purple = wallet entities, green = regulated/whitelisted, blue = context nodes. Pick a target entity to isolate its neighbourhood. Drag, zoom and hover to explore.")

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
                # Phase 3: cluster confidence labels (not all common-inputs are
                # absolute facts — label based on evidence strength)
                is_mixer_ent = "CoinJoin" in str(row.get("attack_type", ""))
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
        f"{row['txid'][:12]}... | Anomaly Score: {row['risk_score']:.1f}% | {row.get('attack_type', 'N/A')}"
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
        if risk_val >= 80:
            risk_badge = '<span class="badge badge-red">CRITICAL</span>'
        elif risk_val >= 50:
            risk_badge = '<span class="badge badge-amber">HIGH</span>'
        elif risk_val > 0:
            risk_badge = '<span class="badge badge-blue">NORMAL</span>'
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
                        <span class="badge {attack_badge_cls}">{attack_type}</span>
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
