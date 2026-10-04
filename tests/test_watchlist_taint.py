"""
Tests for Known-Bad Watchlist and Taint Propagation (MITHYA Intelligence).

Verifies:
1. A watchlist address gets watchlist_hit True, matching label & indicator.
2. A tx one hop away has taint_hops == 1 and a higher taint_score than an unconnected tx.
3. Watchlist IP and ASN matches.
4. Forensic explanation prepending for direct matches and 1–3 hop taints.
5. Watchlist hits get score_graph == 100 in fused scoring.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml_engine import load_watchlist, apply_watchlist, compute_taint
from anomaly_engine import compute_fused_priority, assign_risk_tier


MOCK_WATCHLIST = {
    "address": [
        {"indicator": "bc1qbadactor111111111111111111111111111111", "label": "Ransomware_Payout", "severity": 1.0},
        {"indicator": "1SanctionedMixerUserAddress11111111111", "label": "Sanctioned_Mixer_User", "severity": 0.8},
    ],
    "ip": [
        {"indicator": "198.51.100.25", "label": "Tor_Exit_Node", "severity": 0.6},
    ],
    "asn": [
        {"indicator": "AS99999", "label": "Bulletproof_Hosting", "severity": 0.5},
    ],
}


class TestWatchlistDetection:
    """Test direct known-bad watchlist matching."""

    def test_watchlist_address_gets_hit_true(self):
        """A transaction with a watchlist address gets watchlist_hit == True."""
        df = pd.DataFrame([
            {
                "txid": "tx001",
                "input_addresses": "bc1qbadactor111111111111111111111111111111|1cleanaddress",
                "output_addresses": "1anothercleanaddress",
                "src_ip": "1.2.3.4",
                "asn": "AS12345",
            },
            {
                "txid": "tx002",
                "input_addresses": "1cleanaddress",
                "output_addresses": "1yetanotherclean",
                "src_ip": "5.6.7.8",
                "asn": "AS12345",
            }
        ])

        result = apply_watchlist(df, watchlist=MOCK_WATCHLIST)

        assert result.at[0, "watchlist_hit"] == True
        assert result.at[0, "watchlist_label"] == "Ransomware_Payout"
        assert result.at[0, "watchlist_indicator"] == "bc1qbadactor111111111111111111111111111111"

        assert result.at[1, "watchlist_hit"] == False
        assert result.at[1, "watchlist_label"] is None

    def test_watchlist_ip_match(self):
        """A transaction with a watchlist IP gets watchlist_hit == True."""
        df = pd.DataFrame([
            {
                "txid": "tx_ip",
                "input_addresses": "1cleanaddr",
                "output_addresses": "1cleanaddr2",
                "src_ip": "198.51.100.25",
                "asn": "AS12345",
            }
        ])
        result = apply_watchlist(df, watchlist=MOCK_WATCHLIST)
        assert result.at[0, "watchlist_hit"] == True
        assert result.at[0, "watchlist_label"] == "Tor_Exit_Node"
        assert result.at[0, "watchlist_indicator"] == "198.51.100.25"

    def test_watchlist_asn_match(self):
        """A transaction with a watchlist ASN gets watchlist_hit == True."""
        df = pd.DataFrame([
            {
                "txid": "tx_asn",
                "input_addresses": "1cleanaddr",
                "output_addresses": "1cleanaddr2",
                "src_ip": "1.2.3.4",
                "asn": "AS99999",
            }
        ])
        result = apply_watchlist(df, watchlist=MOCK_WATCHLIST)
        assert result.at[0, "watchlist_hit"] == True
        assert result.at[0, "watchlist_label"] == "Bulletproof_Hosting"


class TestTaintPropagation:
    """Test bipartite graph taint propagation and hops calculation."""

    def test_tx_one_hop_away_has_hops_1_and_higher_taint(self):
        """
        Requirement 7:
        A tx one hop away has taint_hops == 1 and a higher taint_score
        than an unconnected tx.
        """
        # tx1 has the watchlist address (1 hop to addr:bc1qbadactor...)
        # tx2 is in an unconnected component
        df = pd.DataFrame([
            {
                "txid": "tx_near",
                "input_addresses": "bc1qbadactor111111111111111111111111111111",
                "output_addresses": "1someoutput",
                "src_ip": "10.0.0.1",
            },
            {
                "txid": "tx_unconnected",
                "input_addresses": "1unconnected_in",
                "output_addresses": "1unconnected_out",
                "src_ip": "10.0.0.2",
            }
        ])

        apply_watchlist(df, watchlist=MOCK_WATCHLIST)
        compute_taint(df, watchlist=MOCK_WATCHLIST)

        # tx_near is 1 hop from the watchlist node (tx_near ↔ addr:bc1qbadactor...)
        assert df.at[0, "taint_hops"] == 1
        assert df.at[0, "taint_score"] > 0.0

        # tx_unconnected has no path to any watchlist node
        assert df.at[1, "taint_hops"] is None or pd.isna(df.at[1, "taint_hops"])
        assert df.at[1, "taint_score"] == 0.0

        # tx_near has strictly higher taint score than unconnected tx
        assert df.at[0, "taint_score"] > df.at[1, "taint_score"]


class TestForensicExplanations:
    """Test explanation prepending for watchlist hits and taint hops."""

    def test_direct_watchlist_explanation_prepending(self):
        """Direct match prepends: Direct watchlist match: {label} ({indicator})."""
        df = pd.DataFrame([{
            "txid": "tx_exp1",
            "input_addresses": "bc1qbadactor111111111111111111111111111111",
            "output_addresses": "1clean",
            "src_ip": "10.0.0.1",
            "explanation": "Anomaly indicators: port 9050 high.",
        }])
        apply_watchlist(df, watchlist=MOCK_WATCHLIST)
        compute_taint(df, watchlist=MOCK_WATCHLIST)

        expl = df.at[0, "explanation"]
        assert expl.startswith("Direct watchlist match: Ransomware_Payout (bc1qbadactor111111111111111111111111111111)")
        assert "Anomaly indicators:" in expl

    def test_taint_hops_explanation_prepending(self):
        """Connected tx (not direct hit) prepends: {hops} hop(s) from known-bad indicator ({label})."""
        # tx_bad has the watchlist addr and a shared addr
        # tx_hop2 has the shared addr (not direct match, but 3 hops away in bipartite graph)
        df = pd.DataFrame([
            {
                "txid": "tx_bad",
                "input_addresses": "bc1qbadactor111111111111111111111111111111",
                "output_addresses": "1shared_intermediary",
                "src_ip": "10.0.0.1",
                "explanation": "Normal.",
            },
            {
                "txid": "tx_hop",
                "input_addresses": "1shared_intermediary",
                "output_addresses": "1recipient",
                "src_ip": "10.0.0.2",
                "explanation": "Anomaly indicators: peel chain.",
            }
        ])
        apply_watchlist(df, watchlist=MOCK_WATCHLIST)
        compute_taint(df, watchlist=MOCK_WATCHLIST)

        # tx_hop is not a direct hit
        assert df.at[1, "watchlist_hit"] == False
        hops = df.at[1, "taint_hops"]
        assert hops in (1, 2, 3)

        expl = df.at[1, "explanation"]
        assert f"{hops} hop(s) from known-bad indicator (Ransomware_Payout)" in expl


class TestFusedPriorityScoreGraph:
    """Test that score_graph is set to 100 for watchlist hits."""

    def test_watchlist_hit_gets_score_graph_100(self):
        df = pd.DataFrame([{
            "txid": "tx_fused",
            "risk_score": 50.0,
            "detected_type": "Normal_P2P",
            "watchlist_hit": True,
            "taint_score": 10.0,
            "whitelisted_side": None,
        }])
        features = pd.DataFrame([{
            "port_risk_combined": 0.0,
        }])

        compute_fused_priority(df, features)
        assert df.at[0, "score_graph"] == 100.0
