"""
dossier.py — Forensic Evidence Dossier Generator (PDF & JSON)
=============================================================

Generates court-ready, evidence-grade forensic investigation dossiers:
1. build_pdf(df, meta) -> bytes: Full multi-page PDF with Cover, Executive
   Summary, Top 25 Ranked Leads, Top 10 Detailed Forensic Profiles with
   Matplotlib Score Breakdown Bar Charts, Hop-by-Hop Flow Traces, Feature
   Deviations, Threat-Intel Taint, and Methodological Boundaries / Limitations.
2. build_json(df, meta) -> str: Structured JSON evidence package with
   the complete digital forensic dossier schema.
"""

from __future__ import annotations

import io
import json
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas

import flow_tracer
import explainability
from anomaly_engine import WEIGHTS as DEFAULT_WEIGHTS


# ═══════════════════════════════════════════════════════════════════════════
# NUMBERED CANVAS (RUNNING FOOTER: "Page N — MITHYA — Offline")
# ═══════════════════════════════════════════════════════════════════════════

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to stamp running footer with total page count.
    Footer format: 'Page N — MITHYA — Offline'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_footer(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Footer divider line
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 32, letter[0] - 36, 32)

        # Centered footer
        footer_text = f"Page {self._pageNumber} — MITHYA — Offline"
        self.drawCentredString(letter[0] / 2.0, 20, footer_text)

        # Subtle confidentiality tag
        self.drawRightString(letter[0] - 36, 20, "CONFIDENTIAL // LAW ENFORCEMENT & COMPLIANCE")
        self.drawString(36, 20, "CRYPTO FORENSIC TRIAGE")

        self.restoreState()


# ═══════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CHART GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def _generate_score_breakdown_chart(row: pd.Series, weights: Dict[str, float]) -> io.BytesIO:
    """Generate high-resolution horizontal bar chart of score components."""
    risk_val = float(row.get("risk_score", 0.0))
    sb_ml = float(row.get("score_ml", risk_val))
    sb_pat = float(row.get("score_pattern", 0.0))
    sb_net = float(row.get("score_network", 0.0))
    sb_gra = float(row.get("score_graph", 0.0))
    sb_wlf = float(row.get("whitelist_factor", 1.0))
    sb_final = float(row.get("risk_score", risk_val))

    w_ml = weights.get("ml", 0.50)
    w_pat = weights.get("pattern", 0.25)
    w_net = weights.get("network", 0.15)
    w_gra = weights.get("graph", 0.10)

    c_ml = round(w_ml * sb_ml, 1)
    c_pat = round(w_pat * sb_pat, 1)
    c_net = round(w_net * sb_net, 1)
    c_gra = round(w_gra * sb_gra, 1)
    raw_tot = c_ml + c_pat + c_net + c_gra
    c_wl = round(sb_final - raw_tot, 1) if sb_wlf < 1.0 else 0.0

    labels = [
        f"ML Anomaly ({w_ml:.0%})",
        f"Pattern ({w_pat:.0%})",
        f"Network ({w_net:.0%})",
        f"Graph/Taint ({w_gra:.0%})",
    ]
    vals = [c_ml, c_pat, c_net, c_gra]
    bar_colors = ["#3b82f6", "#8b5cf6", "#06b6d4", "#f59e0b"]

    if sb_wlf < 1.0:
        labels.append("WL Discount")
        vals.append(c_wl)
        bar_colors.append("#6b7280")

    labels.append("Final Priority")
    vals.append(sb_final)
    bar_colors.append("#ef4444" if sb_final >= 60 else "#f59e0b" if sb_final >= 40 else "#10b981")

    # Invert for top-to-bottom reading in matplotlib horizontal bars
    labels = labels[::-1]
    vals = vals[::-1]
    bar_colors = bar_colors[::-1]

    fig, ax = plt.subplots(figsize=(6.2, 1.8), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, vals, align="center", color=bar_colors, height=0.55, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5, color="#1e293b", fontweight="bold")
    ax.set_xlim(min(0, min(vals) * 1.1), max(100, max(vals) * 1.15))
    ax.set_xlabel("Contribution / Score Points", fontsize=7, color="#64748b")
    ax.tick_params(axis="x", labelsize=7, colors="#64748b")
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    for bar, val in zip(bars, vals):
        w = bar.get_width()
        offset = 1.5 if w >= 0 else -1.5
        ha = "left" if w >= 0 else "right"
        ax.text(
            w + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            ha=ha,
            fontsize=7,
            fontweight="bold",
            color="#0f172a",
        )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════════
# JSON DOSSIER BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_json(df: pd.DataFrame, meta: Dict[str, Any]) -> str:
    """
    Build structured JSON evidence package with identical forensic content.
    """
    ranked_df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    top25 = ranked_df.head(25)

    # Top 25 serialized records
    top25_records = []
    for idx, row in top25.iterrows():
        top25_records.append({
            "rank": idx + 1,
            "txid": str(row.get("txid", "")),
            "risk_score": round(float(row.get("risk_score", 0.0)), 2),
            "risk_tier": str(row.get("risk_tier", "Low")),
            "detected_type": str(row.get("detected_type", "Unknown")),
            "amount_btc": round(float(row.get("total_amount_btc", row.get("amount_btc", 0.0))), 6),
            "fee_btc": round(float(row.get("fee", 0.0)), 8),
            "src_ip": str(row.get("src_ip", "")),
            "dst_ip": str(row.get("dst_ip", "")),
            "src_port": int(row.get("src_port", 0)),
            "dst_port": int(row.get("dst_port", 0)),
            "timestamp": str(row.get("timestamp", "")),
            "geo_country": str(row.get("geo_country", "XX")),
            "asn": str(row.get("asn", "AS0")),
            "ip_observation_count": int(row.get("ip_observation_count", 1)),
            "observation_spread_s": round(float(row.get("observation_spread_s", 0.0)), 2),
            "watchlist_match": bool(row.get("watchlist_match", False)),
            "taint_score": round(float(row.get("taint_score", 0.0)), 4),
        })

    # Top 10 detailed profiles
    top10 = ranked_df.head(10)
    top10_details = []
    for idx, row in top10.iterrows():
        txid = str(row.get("txid", ""))

        # Deviations
        telem_raw = row.get("telemetry", "")
        deviations = []
        if isinstance(telem_raw, str) and telem_raw.strip():
            try:
                deviations = json.loads(telem_raw)
            except Exception:
                deviations = []
        elif isinstance(telem_raw, list):
            deviations = telem_raw

        # Flow trace
        try:
            hops = flow_tracer.trace_forward(df, txid, max_hops=4)
        except Exception:
            hops = []

        top10_details.append({
            "rank": idx + 1,
            "txid": txid,
            "timestamp": str(row.get("timestamp", "")),
            "scores": {
                "final_priority": round(float(row.get("risk_score", 0.0)), 2),
                "score_ml": round(float(row.get("score_ml", row.get("risk_score", 0.0))), 2),
                "score_pattern": round(float(row.get("score_pattern", 0.0)), 2),
                "score_network": round(float(row.get("score_network", 0.0)), 2),
                "score_graph": round(float(row.get("score_graph", 0.0)), 2),
                "whitelist_factor": round(float(row.get("whitelist_factor", 1.0)), 2),
            },
            "network_telemetry": {
                "src_ip": str(row.get("src_ip", "")),
                "dst_ip": str(row.get("dst_ip", "")),
                "src_port": int(row.get("src_port", 0)),
                "dst_port": int(row.get("dst_port", 0)),
                "observations": int(row.get("ip_observation_count", 1)),
                "spread_seconds": round(float(row.get("observation_spread_s", 0.0)), 2),
            },
            "blockchain": {
                "amount_btc": round(float(row.get("total_amount_btc", 0.0)), 6),
                "fee_btc": round(float(row.get("fee", 0.0)), 8),
                "detected_type": str(row.get("detected_type", "")),
                "entity_id": str(row.get("entity_id", "N/A")),
            },
            "threat_intel": {
                "watchlist_match": bool(row.get("watchlist_match", False)),
                "watchlist_type": str(row.get("watchlist_type", "")),
                "taint_score": round(float(row.get("taint_score", 0.0)), 4),
                "taint_hops": int(row.get("taint_hops", 0)) if pd.notna(row.get("taint_hops")) else 0,
            },
            "explanation": str(row.get("explanation", "")),
            "top_deviations": deviations,
            "forward_flow_trace": hops,
        })

    package = {
        "metadata": {
            "dossier_format": "MITHYA_FORENSIC_DOSSIER_V2",
            "case_id": meta.get("case_id", f"MITHYA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            "analyst_name": meta.get("analyst_name", "Forensic Analyst"),
            "generated_time": meta.get("generated_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "source_file_sha256": meta.get("source_file_sha256", "N/A"),
            "whitelist_sha256": meta.get("whitelist_sha256", "N/A"),
            "watchlist_sha256": meta.get("watchlist_sha256", "N/A"),
            "model_version": meta.get("model_version", "v2.2.0 (Fused Multi-Signal Anomaly Engine)"),
            "rf_model_hash": meta.get("rf_model_hash", "N/A"),
            "contamination": meta.get("contamination", 0.05),
            "fusion_weights": meta.get("weights", DEFAULT_WEIGHTS),
            "detector_thresholds": meta.get("detector_thresholds", {
                "coinjoin_min_outputs": 3,
                "fanout_min_outputs": 5,
                "peel_disparity_ratio": 0.70,
                "fee_urgency_ratio": 0.005,
            }),
        },
        "executive_summary": {
            "funnel_numbers": meta.get("funnel_numbers", {
                "total_ingested": len(df),
                "whitelisted_cleared": int((df.get("whitelist_factor", 1.0) == 0.0).sum()) if "whitelist_factor" in df.columns else 0,
                "flagged_anomalies": int(df.get("is_anomaly", False).sum()) if "is_anomaly" in df.columns else 0,
            }),
            "tier_counts": meta.get("tier_counts", {
                "Critical": int((df.get("risk_tier") == "Critical").sum()) if "risk_tier" in df.columns else 0,
                "High": int((df.get("risk_tier") == "High").sum()) if "risk_tier" in df.columns else 0,
                "Medium": int((df.get("risk_tier") == "Medium").sum()) if "risk_tier" in df.columns else 0,
                "Low": int((df.get("risk_tier") == "Low").sum()) if "risk_tier" in df.columns else 0,
            }),
        },
        "ranked_leads_top25": top25_records,
        "top_10_detailed_profiles": top10_details,
        "limitations": {
            "synthetic_data_note": (
                "This evidence dossier was prepared using synthetic or simulated blockchain transaction "
                "and network telemetry datasets. While behavioral patterns, graph topologies, and cryptographic "
                "attributes emulate real-world adversarial behavior, synthetic traces should not be conflated "
                "with real-world criminal liability."
            ),
            "network_attribution_boundaries": (
                "IP telemetry attributes observation nodes and relay points across the P2P broadcast network, "
                "NOT absolute end-user identities. Network telemetry is subject to inherent attribution boundaries "
                "imposed by VPN tunnels, NAT gateways, Tor exit relays, dynamic mobile IPs, and proxy infrastructure."
            ),
            "investigative_lead_vs_verdict": (
                "MITHYA is an investigative triage prioritization instrument, not an adjudicative verdict. "
                "Flagged transactions represent statistical, network, and graph anomalies designed to guide "
                "compliance officers, forensic auditors, and law enforcement analysts toward high-priority "
                "investigative leads. Every lead must be verified through lawful process, KYC records, and "
                "corroborating evidence."
            ),
        },
    }

    return json.dumps(package, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# PDF DOSSIER BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_pdf(df: pd.DataFrame, meta: Dict[str, Any]) -> bytes:
    """
    Build a comprehensive, evidence-grade PDF Forensic Dossier.
    """
    case_id = meta.get("case_id", f"MITHYA-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    analyst_name = meta.get("analyst_name", "Forensic Analyst")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=45,
        title=f"MITHYA Forensic Dossier — {case_id}",
        author=analyst_name,
        subject=f"Offline Forensic Triage Evidence Case {case_id}",
    )

    styles = getSampleStyleSheet()

    # Custom typography & styles
    c_primary = colors.HexColor("#0f172a")
    c_accent = colors.HexColor("#2563eb")
    c_muted = colors.HexColor("#475569")
    c_border = colors.HexColor("#cbd5e1")
    c_light = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=c_primary,
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=c_accent,
        alignment=0,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
    )
    meta_label = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=c_primary,
    )
    meta_val = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1e293b"),
    )
    table_hdr = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.white,
    )

    story = []

    # ─────────────────────────────────────────────────────────────
    # PAGE 1: COVER PAGE
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("MITHYA", title_style))
    story.append(Paragraph("FORENSIC EVIDENCE DOSSIER & TRIAGE REPORT", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=c_accent, spaceBefore=4, spaceAfter=14))

    case_id = meta.get("case_id", f"MITHYA-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    analyst_name = meta.get("analyst_name", "Forensic Analyst")
    gen_time = meta.get("generated_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    src_sha = meta.get("source_file_sha256", "N/A")
    wl_sha = meta.get("whitelist_sha256", "N/A")
    wt_sha = meta.get("watchlist_sha256", "N/A")
    mod_ver = meta.get("model_version", "v2.2.0 (Fused Multi-Signal Anomaly Engine)")
    rf_hash = meta.get("rf_model_hash", "N/A")
    cont = meta.get("contamination", 0.05)
    weights = meta.get("weights", DEFAULT_WEIGHTS)
    thresh = meta.get("detector_thresholds", {
        "CoinJoin Min Inputs": "3",
        "Fan-Out Min Dispersal": "5 outputs",
        "Peel Disparity Ratio": "0.70",
        "Port Risk Threshold": "40.0 pts",
    })

    weights_str = " | ".join(f"{k.upper()}: {v:.0%}" for k, v in weights.items())
    thresh_str = " | ".join(f"{k}: {v}" for k, v in thresh.items())

    cover_meta = [
        [Paragraph("Case Identification", meta_label), Paragraph(f"<b>{case_id}</b>", meta_val)],
        [Paragraph("Lead Forensic Analyst", meta_label), Paragraph(f"<b>{analyst_name}</b>", meta_val)],
        [Paragraph("Execution Timestamp", meta_label), Paragraph(gen_time, meta_val)],
        [Paragraph("Operational Mode", meta_label), Paragraph("OFFLINE AIR-GAPPED FORENSICS (Zero External APIs)", meta_val)],
        [Paragraph("Source Dataset SHA-256", meta_label), Paragraph(src_sha, meta_val)],
        [Paragraph("Institutional Whitelist SHA", meta_label), Paragraph(wl_sha, meta_val)],
        [Paragraph("Threat-Intel Watchlist SHA", meta_label), Paragraph(wt_sha, meta_val)],
        [Paragraph("Forensic Model Engine", meta_label), Paragraph(mod_ver, meta_val)],
        [Paragraph("Supervised RF Model SHA", meta_label), Paragraph(rf_hash, meta_val)],
        [Paragraph("Contamination Baseline", meta_label), Paragraph(f"{cont:.1%} ({cont})", meta_val)],
        [Paragraph("Score Fusion Weights", meta_label), Paragraph(weights_str, meta_val)],
        [Paragraph("Structural Thresholds", meta_label), Paragraph(thresh_str, meta_val)],
    ]

    t_cover = Table(cover_meta, colWidths=[150, 390])
    t_cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#ffffff")),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_cover)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Chain of Custody & Procedural Assurance", h2_style))
    story.append(Paragraph(
        "This dossier compiles automated forensic extraction, probabilistic anomaly scoring, "
        "and topological graph taint tracing conducted in a strictly air-gapped, offline environment. "
        "All source hashes and detector weights are recorded above to guarantee evidentiary reproducibility "
        "under Federal Rule of Evidence 902(13)/(14) standards for electronic forensic records.",
        body_style,
    ))

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # PAGE 2: EXECUTIVE SUMMARY & TRIAGE FUNNEL
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary & Triage Funnel", h1_style))
    story.append(Paragraph(
        "The MITHYA triage engine applies multi-stage heuristic filtering, unsupervised Isolation Forest "
        "outlier scoring, rule-based threat-vector identification, and graph-based taint propagation to distill "
        "large volumes of raw transactions into prioritized, actionable investigative leads.",
        body_style,
    ))
    story.append(Spacer(1, 8))

    funnel = meta.get("funnel_numbers", {})
    tot_tx = funnel.get("total_ingested", len(df))
    wl_cleared = funnel.get("whitelisted_cleared", int((df.get("whitelist_factor", 1.0) == 0.0).sum()) if "whitelist_factor" in df.columns else 0)
    flagged = funnel.get("flagged_anomalies", int(df.get("is_anomaly", False).sum()) if "is_anomaly" in df.columns else 0)
    mixers = funnel.get("mixers_detected", int((df.get("detected_type") == "CoinJoin_Mixer").sum()) if "detected_type" in df.columns else 0)
    peels = funnel.get("peel_chains_detected", int((df.get("detected_type") == "Peel_Chain").sum()) if "detected_type" in df.columns else 0)
    fanouts = funnel.get("fanouts_detected", int((df.get("detected_type") == "FanOut_Dispersal").sum()) if "detected_type" in df.columns else 0)
    feespikes = funnel.get("feespikes_detected", int((df.get("detected_type") == "Fee_Spike").sum()) if "detected_type" in df.columns else 0)
    wl_matches = int((df.get("watchlist_match") == True).sum()) if "watchlist_match" in df.columns else 0
    tainted = int((df.get("taint_score", 0.0) > 0.0).sum()) if "taint_score" in df.columns else 0

    funnel_table_data = [
        [Paragraph("Pipeline Metric", table_hdr), Paragraph("Count", table_hdr), Paragraph("% of Dataset", table_hdr), Paragraph("Forensic Significance", table_hdr)],
        [Paragraph("Total Transactions Ingested", table_cell), Paragraph(f"{tot_tx:,}", table_cell), Paragraph("100.0%", table_cell), Paragraph("Base volume evaluated", table_cell)],
        [Paragraph("Institutional Cleared (Whitelist)", table_cell), Paragraph(f"{wl_cleared:,}", table_cell), Paragraph(f"{(wl_cleared/tot_tx*100) if tot_tx else 0:.1f}%", table_cell), Paragraph("Zero-risk regulated VASP endpoints", table_cell)],
        [Paragraph("Total Flagged Anomalies", table_cell), Paragraph(f"{flagged:,}", table_cell), Paragraph(f"{(flagged/tot_tx*100) if tot_tx else 0:.1f}%", table_cell), Paragraph("Investigative leads exceeding baseline threshold", table_cell)],
        [Paragraph("↳ CoinJoin Mixers", table_cell), Paragraph(f"{mixers:,}", table_cell), Paragraph(f"{(mixers/tot_tx*100) if tot_tx else 0:.1f}%", table_cell), Paragraph("Equal-denomination multi-party mixing", table_cell)],
        [Paragraph("↳ Multi-Hop Peel Chains", table_cell), Paragraph(f"{peels:,}", table_cell), Paragraph(f"{(peels/tot_tx*100) if tot_tx else 0:.1f}%", table_cell), Paragraph("Sequential layering & change peeling", table_cell)],
        [Paragraph("↳ Rapid Fan-Out Dispersals", table_cell), Paragraph(f"{fanouts:,}", table_cell), Paragraph(f"{(fanouts/tot_tx*100) if tot_tx else 0:.1f}%", table_cell), Paragraph("1-to-many rapid fund dispersal", table_cell)],
        [Paragraph("↳ Fee-Spike Urgency Hops", table_cell), Paragraph(f"{feespikes:,}", table_cell), Paragraph(f"{(feespikes/tot_tx*100) if tot_tx else 0:.1f}%", table_cell), Paragraph("Time-critical overpayment hops", table_cell)],
        [Paragraph("Threat-Intel Watchlist Matches", table_cell), Paragraph(f"{wl_matches:,}", table_cell), Paragraph(f"{(wl_matches/tot_tx*100) if tot_tx else 0:.1f}%", table_cell), Paragraph("Direct hits against known bad indicators", table_cell)],
        [Paragraph("Tainted Downstream Nodes", table_cell), Paragraph(f"{tainted:,}", table_cell), Paragraph(f"{(tainted/tot_tx*100) if tot_tx else 0:.1f}%", table_cell), Paragraph("Graph-propagated exposure to threat vectors", table_cell)],
    ]

    t_funnel = Table(funnel_table_data, colWidths=[160, 65, 75, 240])
    t_funnel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_funnel)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Risk Tier Distribution", h2_style))

    tiers = meta.get("tier_counts", {})
    t_crit = tiers.get("Critical", int((df.get("risk_tier") == "Critical").sum()) if "risk_tier" in df.columns else 0)
    t_high = tiers.get("High", int((df.get("risk_tier") == "High").sum()) if "risk_tier" in df.columns else 0)
    t_med = tiers.get("Medium", int((df.get("risk_tier") == "Medium").sum()) if "risk_tier" in df.columns else 0)
    t_low = tiers.get("Low", int((df.get("risk_tier") == "Low").sum()) if "risk_tier" in df.columns else 0)

    tier_table_data = [
        [Paragraph("Risk Tier", table_hdr), Paragraph("Score Band", table_hdr), Paragraph("Count", table_hdr), Paragraph("Action Protocol", table_hdr)],
        [Paragraph("<font color='#dc2626'><b>CRITICAL</b></font>", table_cell), Paragraph("80.0% – 100.0%", table_cell), Paragraph(f"{t_crit:,}", table_cell), Paragraph("Immediate investigative assignment & expedited SAR/KYC subpoena", table_cell)],
        [Paragraph("<font color='#d97706'><b>HIGH</b></font>", table_cell), Paragraph("60.0% – 79.9%", table_cell), Paragraph(f"{t_high:,}", table_cell), Paragraph("Active forensic review & forward hop tracing", table_cell)],
        [Paragraph("<font color='#2563eb'><b>MEDIUM</b></font>", table_cell), Paragraph("40.0% – 59.9%", table_cell), Paragraph(f"{t_med:,}", table_cell), Paragraph("Routine surveillance & behavioral cluster correlation", table_cell)],
        [Paragraph("<font color='#16a34a'><b>LOW</b></font>", table_cell), Paragraph("0.0% – 39.9%", table_cell), Paragraph(f"{t_low:,}", table_cell), Paragraph("Normal P2P baseline / pre-cleared institutional volume", table_cell)],
    ]
    t_tier = Table(tier_table_data, colWidths=[90, 85, 65, 300])
    t_tier.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_accent),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_tier)

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # PAGE 3: RANKED LEADS TABLE (TOP 25)
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("2. Ranked Investigative Leads (Top 25)", h1_style))
    story.append(Paragraph(
        "Ranked strictly by Investigative Priority Index. Prioritized by composite multi-signal fusion "
        "(Isolation Forest anomaly distance + structural attack signatures + network telemetry risk + graph taint).",
        body_style,
    ))
    story.append(Spacer(1, 6))

    ranked_df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    top25 = ranked_df.head(25)

    lead_headers = [
        Paragraph("#", table_hdr),
        Paragraph("TXID", table_hdr),
        Paragraph("Priority", table_hdr),
        Paragraph("Tier", table_hdr),
        Paragraph("Detected Pattern", table_hdr),
        Paragraph("Amount (BTC)", table_hdr),
        Paragraph("Source IP", table_hdr),
        Paragraph("Watchlist/Taint", table_hdr),
    ]
    lead_rows = [lead_headers]

    for idx, row in top25.iterrows():
        tx_short = str(row.get("txid", ""))[:12] + "…"
        sc = float(row.get("risk_score", 0.0))
        tier_val = str(row.get("risk_tier", "Low"))
        tier_color = "#dc2626" if tier_val == "Critical" else ("#d97706" if tier_val == "High" else "#2563eb")
        tier_html = f"<font color='{tier_color}'><b>{tier_val}</b></font>"

        pat = str(row.get("detected_type", "Normal"))
        amt = float(row.get("total_amount_btc", row.get("amount_btc", 0.0)))
        src = str(row.get("src_ip", "Unknown"))
        if len(src) > 15:
            src = src[:14] + "…"

        wl_hit = bool(row.get("watchlist_match", False))
        t_sc = float(row.get("taint_score", 0.0))
        if wl_hit:
            wl_str = "<font color='#dc2626'><b>DIRECT HIT</b></font>"
        elif t_sc > 0:
            wl_str = f"<font color='#d97706'>Taint: {t_sc:.2f}</font>"
        else:
            wl_str = "<font color='#64748b'>Clean</font>"

        lead_rows.append([
            Paragraph(str(idx + 1), table_cell),
            Paragraph(f"<font name='Courier'>{tx_short}</font>", table_cell),
            Paragraph(f"<b>{sc:.1f}%</b>", table_cell),
            Paragraph(tier_html, table_cell),
            Paragraph(pat, table_cell),
            Paragraph(f"{amt:.4f}", table_cell),
            Paragraph(src, table_cell),
            Paragraph(wl_str, table_cell),
        ])

    t_leads = Table(lead_rows, colWidths=[20, 85, 45, 45, 110, 65, 85, 85])
    t_leads.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
    ]))
    story.append(t_leads)

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # PAGES 4+: DETAIL PROFILES FOR TOP 10 LEADS
    # ─────────────────────────────────────────────────────────────
    top10 = ranked_df.head(10)

    story.append(Paragraph("3. Detailed Lead Inspection (Top 10)", h1_style))
    story.append(Paragraph(
        "Individual forensic case profiles for the top 10 highest-priority investigative leads, "
        "including multi-signal score breakdown graphs, statistical percentiles, hop-by-hop fund flows, "
        "and contextual explainability indicators.",
        body_style,
    ))
    story.append(Spacer(1, 6))

    for idx, row in top10.iterrows():
        txid = str(row.get("txid", ""))
        rank_num = idx + 1
        sc = float(row.get("risk_score", 0.0))
        tier_val = str(row.get("risk_tier", "Low"))
        tier_color = "#dc2626" if tier_val == "Critical" else ("#d97706" if tier_val == "High" else "#2563eb")

        # Lead Header Card
        story.append(Paragraph(
            f"<b>LEAD #{rank_num} &mdash; PRIORITY: {sc:.1f}% (<font color='{tier_color}'>{tier_val.upper()}</font>)</b>",
            h2_style,
        ))
        story.append(Paragraph(f"<font name='Courier' size=8>TXID: <b>{txid}</b></font>", body_style))
        story.append(Spacer(1, 4))

        # Core Metadata Grid
        amt = float(row.get("total_amount_btc", row.get("amount_btc", 0.0)))
        fee = float(row.get("fee", 0.0))
        src_ip = str(row.get("src_ip", "Unknown"))
        dst_ip = str(row.get("dst_ip", "Unknown"))
        src_port = int(row.get("src_port", 0))
        dst_port = int(row.get("dst_port", 0))
        pat = str(row.get("detected_type", "Unknown"))
        ent = str(row.get("entity_id", "N/A"))
        geo = str(row.get("geo_country", "XX"))
        asn = str(row.get("asn", "AS0"))
        obs = int(row.get("ip_observation_count", 1))
        spread = float(row.get("observation_spread_s", 0.0))

        lead_meta_table = [
            [
                Paragraph("<b>Timestamp:</b>", meta_label), Paragraph(str(row.get("timestamp", "")), table_cell),
                Paragraph("<b>Amount / Fee:</b>", meta_label), Paragraph(f"{amt:.6f} BTC (Fee: {fee:.6f})", table_cell),
            ],
            [
                Paragraph("<b>Source IP/Port:</b>", meta_label), Paragraph(f"{src_ip}:{src_port}", table_cell),
                Paragraph("<b>Destination:</b>", meta_label), Paragraph(f"{dst_ip}:{dst_port}", table_cell),
            ],
            [
                Paragraph("<b>Pattern:</b>", meta_label), Paragraph(f"<b>{pat}</b>", table_cell),
                Paragraph("<b>Wallet Entity:</b>", meta_label), Paragraph(ent, table_cell),
            ],
            [
                Paragraph("<b>Offline Geo/ASN:</b>", meta_label), Paragraph(f"{geo} &middot; {asn}", table_cell),
                Paragraph("<b>Telemetry Obs:</b>", meta_label), Paragraph(f"{obs} relay(s) &middot; {spread:.2f}s spread", table_cell),
            ],
        ]
        t_lmeta = Table(lead_meta_table, colWidths=[95, 175, 95, 175])
        t_lmeta.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(t_lmeta)
        story.append(Spacer(1, 6))

        # Chart: Score Breakdown
        chart_buf = _generate_score_breakdown_chart(row, weights)
        chart_img = Image(chart_buf, width=540, height=135)
        story.append(chart_img)
        story.append(Spacer(1, 4))

        # Deviations Table
        telem_raw = row.get("telemetry", "")
        deviations = []
        if isinstance(telem_raw, str) and telem_raw.strip():
            try:
                deviations = json.loads(telem_raw)
            except Exception:
                deviations = []
        elif isinstance(telem_raw, list):
            deviations = telem_raw

        if deviations:
            dev_rows = [
                [Paragraph("Feature Deviation", table_hdr), Paragraph("Observed", table_hdr), Paragraph("Percentile", table_hdr), Paragraph("Statistical Audit Reason", table_hdr)]
            ]
            for d in deviations[:4]:
                fname = d.get("label", d.get("feature_name", ""))
                oval = float(d.get("observed_value", 0.0))
                prank = float(d.get("percentile_rank", 0.0))
                dreason = d.get("audit_reason", "")
                dev_rows.append([
                    Paragraph(f"<b>{fname}</b>", table_cell),
                    Paragraph(f"{oval:.4f}", table_cell),
                    Paragraph(f"{prank:.1f}th", table_cell),
                    Paragraph(dreason, table_cell),
                ])
            t_dev = Table(dev_rows, colWidths=[120, 60, 55, 305])
            t_dev.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(t_dev)
            story.append(Spacer(1, 4))

        # Flow Trace Forward Table
        try:
            hops = flow_tracer.trace_forward(df, txid, max_hops=4)
        except Exception:
            hops = []

        if hops and len(hops) > 1:
            flow_rows = [
                [Paragraph("Hop", table_hdr), Paragraph("Transaction ID", table_hdr), Paragraph("Role", table_hdr), Paragraph("Δt (min)", table_hdr), Paragraph("BTC In", table_hdr), Paragraph("Largest Out", table_hdr), Paragraph("Peeled BTC", table_hdr)]
            ]
            for h in hops[:5]:
                htx = str(h.get("txid", ""))[:12] + "…"
                hrole = str(h.get("role", "Intermediary"))
                hdt = float(h.get("delta_t_minutes", 0.0))
                hbin = float(h.get("btc_in", 0.0))
                hlout = float(h.get("largest_output_btc", 0.0))
                hpeel = float(h.get("peeled_btc", 0.0))
                flow_rows.append([
                    Paragraph(str(h.get("hop", 0)), table_cell),
                    Paragraph(f"<font name='Courier'>{htx}</font>", table_cell),
                    Paragraph(hrole, table_cell),
                    Paragraph(f"{hdt:.1f}", table_cell),
                    Paragraph(f"{hbin:.4f}", table_cell),
                    Paragraph(f"{hlout:.4f}", table_cell),
                    Paragraph(f"{hpeel:.4f}", table_cell),
                ])
            t_flow = Table(flow_rows, colWidths=[25, 95, 95, 55, 90, 90, 90])
            t_flow.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(t_flow)
            story.append(Spacer(1, 4))

        # Watchlist & Taint status
        wl_hit = bool(row.get("watchlist_match", False))
        t_sc = float(row.get("taint_score", 0.0))
        if wl_hit or t_sc > 0:
            wl_text = (
                f"<b>Threat-Intel Alert:</b> Direct Watchlist Hit ({row.get('watchlist_type', 'Known Bad')}). "
                f"Taint Score: {t_sc:.4f} across {row.get('taint_hops', 0)} hops."
                if wl_hit else
                f"<b>Taint Propagation Warning:</b> Taint Score: {t_sc:.4f} (direct exposure to monitored threat cluster)."
            )
            story.append(Paragraph(f"<font color='#dc2626'>{wl_text}</font>", body_style))
            story.append(Spacer(1, 3))

        # AI Explanation
        exp_text = str(row.get("explanation", "Anomalous multi-dimensional deviation flagged."))
        story.append(Paragraph(f"<b>Forensic AI Explanation:</b> {exp_text}", body_style))

        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceBefore=4, spaceAfter=8))
        story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # FINAL PAGE: METHODOLOGICAL BOUNDARY & LIMITATIONS
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Methodological Scope & Limitations", h1_style))
    story.append(Paragraph(
        "To ensure compliance with evidentiary standards and protect against investigative misattribution, "
        "analysts must review and observe the following mandatory constraints.",
        body_style,
    ))
    story.append(Spacer(1, 10))

    limits_data = [
        [
            Paragraph("<b>1. Synthetic Simulation Baseline</b>", h2_style),
            Paragraph(
                "This evidence dossier was prepared using synthetic or simulated blockchain transaction "
                "and network telemetry datasets. While behavioral patterns, graph topologies, and cryptographic "
                "attributes emulate real-world adversarial behavior, synthetic traces should not be conflated "
                "with real-world criminal liability.",
                body_style,
            ),
        ],
        [
            Paragraph("<b>2. Network Attribution Caveat</b>", h2_style),
            Paragraph(
                "IP telemetry attributes observation nodes and relay points across the P2P broadcast network, "
                "NOT absolute end-user identities. Network telemetry is subject to inherent attribution boundaries "
                "imposed by VPN tunnels, NAT gateways, Tor exit relays, dynamic mobile IPs, and proxy infrastructure.",
                body_style,
            ),
        ],
        [
            Paragraph("<b>3. Investigative Lead vs. Verdict</b>", h2_style),
            Paragraph(
                "MITHYA is an investigative triage prioritization instrument, not an adjudicative verdict. "
                "Flagged transactions represent statistical, network, and graph anomalies designed to guide "
                "compliance officers, forensic auditors, and law enforcement analysts toward high-priority "
                "investigative leads. Every lead must be verified through lawful process, KYC records, and "
                "corroborating evidence.",
                body_style,
            ),
        ],
    ]

    t_limits = Table(limits_data, colWidths=[150, 390])
    t_limits.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (1, 0), (1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_limits)

    story.append(Spacer(1, 35))

    # Signature Block
    sig_data = [
        [
            Paragraph(f"<b>Analyst Certification:</b><br/><br/>_____________________________________<br/><b>{analyst_name}</b><br/>Lead Forensic Examiner", body_style),
            Paragraph(f"<b>Verification & Chain of Custody:</b><br/><br/>_____________________________________<br/><b>Supervising Special Agent / Officer</b><br/>Date: {gen_time[:10]}", body_style),
        ]
    ]
    t_sig = Table(sig_data, colWidths=[270, 270])
    t_sig.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t_sig)

    # Build the document using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()
