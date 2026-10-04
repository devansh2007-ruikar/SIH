"""
tests/test_dossier.py — Unit and integration tests for evidence dossier generation
=================================================================================
"""

import json
import re
import pytest
import pandas as pd
import numpy as np

import dossier


@pytest.fixture
def sample_triage_df():
    """Create a representative dataframe of scored transactions."""
    rows = []
    for i in range(30):
        rows.append({
            "txid": f"tx_{i:04d}_{'a' * 16}",
            "timestamp": f"2026-10-04 12:{i:02d}:00",
            "total_amount_btc": float(round(1.5 + i * 0.25, 4)),
            "amount_btc": float(round(1.5 + i * 0.25, 4)),
            "fee": 0.00015,
            "src_ip": f"192.168.1.{10 + i}",
            "dst_ip": f"10.0.0.{20 + i}",
            "src_port": 8333,
            "dst_port": 8333,
            "geo_country": "US" if i % 2 == 0 else "DE",
            "asn": "AS15169" if i % 2 == 0 else "AS24940",
            "ip_observation_count": 1 + (i % 3),
            "observation_spread_s": float(i * 1.5),
            "risk_score": float(100.0 - (i * 2.5)),
            "score_ml": float(95.0 - (i * 2.0)),
            "score_pattern": float(80.0 if i < 10 else 20.0),
            "score_network": float(70.0 if i < 5 else 15.0),
            "score_graph": float(60.0 if i < 8 else 5.0),
            "whitelist_factor": 1.0 if i > 2 else 0.0,
            "risk_tier": "Critical" if (100.0 - i * 2.5) >= 80 else "High" if (100.0 - i * 2.5) >= 60 else "Medium",
            "detected_type": "Peel_Chain" if i % 3 == 0 else "CoinJoin_Mixer" if i % 3 == 1 else "FanOut_Dispersal",
            "is_anomaly": True if (100.0 - i * 2.5) >= 50 else False,
            "entity_id": f"entity_{i % 5}",
            "watchlist_match": True if i == 0 else False,
            "watchlist_type": "Known Ransomware Wallet" if i == 0 else "",
            "taint_score": 0.85 if i == 0 else (0.35 if i == 1 else 0.0),
            "taint_hops": 1 if i == 0 else (2 if i == 1 else 0),
            "explanation": f"Statistical anomaly detected at step {i}.",
            "telemetry": json.dumps([
                {
                    "label": "High Fan-Out",
                    "observed_value": 7.0,
                    "percentile_rank": 97.5,
                    "audit_reason": "Output count substantially exceeds baseline distribution",
                }
            ]),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_meta():
    return {
        "case_id": "MITHYA-20261004-213000",
        "analyst_name": "Special Agent Jane Doe",
        "generated_time": "2026-10-04 21:30:00",
        "source_file_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "whitelist_sha256": "1f410855b245761f5fa4448b8e9c34b8fd83bfe6e943a9e5b11bf52e8cd87745",
        "watchlist_sha256": "d06bfed0ba8effed4b7c1ed16f31eca2df028c1165dbd391a9d61ac7e3c928ac",
        "model_version": "v2.2.0 (Fused Multi-Signal Anomaly Engine)",
        "rf_model_hash": "e0e8483d31598a0ad3939dd95e1d9947aa38246acf8455fec683c3a99d905a2a",
        "contamination": 0.05,
        "weights": {"ml": 0.50, "pattern": 0.25, "network": 0.15, "graph": 0.10},
        "detector_thresholds": {
            "CoinJoin Equal Outputs": "≥ 3 outputs",
            "Fan-Out Dispersal": "≥ 5 outputs",
            "Peel Disparity Ratio": "≥ 0.70",
            "Fee-Spike Urgency": "≥ 0.005 BTC/KB",
        },
        "funnel_numbers": {
            "total_ingested": 30,
            "whitelisted_cleared": 3,
            "flagged_anomalies": 21,
            "mixers_detected": 10,
            "peel_chains_detected": 10,
            "fanouts_detected": 10,
            "feespikes_detected": 0,
        },
        "tier_counts": {
            "Critical": 9,
            "High": 8,
            "Medium": 13,
            "Low": 0,
        },
    }


def test_build_json_structure(sample_triage_df, sample_meta):
    """Verify build_json produces valid JSON conforming to the dossier schema."""
    json_str = dossier.build_json(sample_triage_df, sample_meta)
    data = json.loads(json_str)

    assert "metadata" in data
    assert "executive_summary" in data
    assert "ranked_leads_top25" in data
    assert "top_10_detailed_profiles" in data
    assert "limitations" in data

    # Verify metadata fields
    meta_sec = data["metadata"]
    assert meta_sec["case_id"] == "MITHYA-20261004-213000"
    assert meta_sec["analyst_name"] == "Special Agent Jane Doe"
    assert meta_sec["source_file_sha256"] == sample_meta["source_file_sha256"]
    assert meta_sec["whitelist_sha256"] == sample_meta["whitelist_sha256"]
    assert meta_sec["watchlist_sha256"] == sample_meta["watchlist_sha256"]
    assert meta_sec["rf_model_hash"] == sample_meta["rf_model_hash"]
    assert meta_sec["contamination"] == 0.05
    assert meta_sec["fusion_weights"] == sample_meta["weights"]
    assert meta_sec["detector_thresholds"] == sample_meta["detector_thresholds"]

    # Verify executive summary
    exec_sec = data["executive_summary"]
    assert exec_sec["funnel_numbers"]["total_ingested"] == 30
    assert exec_sec["tier_counts"]["Critical"] == 9

    # Verify ranked top 25
    top25 = data["ranked_leads_top25"]
    assert len(top25) == 25
    assert top25[0]["rank"] == 1
    assert top25[0]["risk_score"] >= top25[1]["risk_score"]

    # Verify top 10 profiles
    top10 = data["top_10_detailed_profiles"]
    assert len(top10) == 10
    first = top10[0]
    assert "scores" in first
    assert "network_telemetry" in first
    assert "blockchain" in first
    assert "threat_intel" in first
    assert "explanation" in first
    assert "top_deviations" in first
    assert "forward_flow_trace" in first

    # Verify limitations
    limits = data["limitations"]
    assert "synthetic_data_note" in limits
    assert "network_attribution_boundaries" in limits
    assert "investigative_lead_vs_verdict" in limits


def test_build_pdf_generation(sample_triage_df, sample_meta):
    """Verify build_pdf generates a valid, multi-page binary PDF."""
    pdf_bytes = dossier.build_pdf(sample_triage_df, sample_meta)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 50000
    assert pdf_bytes.startswith(b"%PDF-")

    # Verify page count is 14
    page_objs = re.findall(rb'/Type\s*/Page\b', pdf_bytes)
    assert len(page_objs) == 14

    # Check for MITHYA and Offline strings in binary
    assert b"MITHYA" in pdf_bytes
    assert b"Offline" in pdf_bytes


def test_dossier_small_dataset(sample_meta):
    """Verify dossier builds cleanly with fewer than 10 rows."""
    small_df = pd.DataFrame([
        {
            "txid": f"tx_small_{i}",
            "timestamp": "2026-10-04 12:00:00",
            "total_amount_btc": 2.0,
            "amount_btc": 2.0,
            "fee": 0.0001,
            "src_ip": "1.2.3.4",
            "dst_ip": "5.6.7.8",
            "src_port": 8333,
            "dst_port": 8333,
            "geo_country": "US",
            "asn": "AS15169",
            "ip_observation_count": 1,
            "observation_spread_s": 0.0,
            "risk_score": 90.0 - i * 10,
            "score_ml": 85.0,
            "score_pattern": 50.0,
            "score_network": 40.0,
            "score_graph": 10.0,
            "whitelist_factor": 1.0,
            "risk_tier": "Critical",
            "detected_type": "CoinJoin_Mixer",
            "is_anomaly": True,
            "entity_id": "ent_0",
            "watchlist_match": False,
            "taint_score": 0.0,
            "explanation": "Test explanation",
            "telemetry": "[]",
        }
        for i in range(3)
    ])

    pdf_bytes = dossier.build_pdf(small_df, sample_meta)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")

    json_str = dossier.build_json(small_df, sample_meta)
    data = json.loads(json_str)
    assert len(data["ranked_leads_top25"]) == 3
    assert len(data["top_10_detailed_profiles"]) == 3
