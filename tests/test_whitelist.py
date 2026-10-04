"""Tests for ml_engine.apply_institutional_whitelist.

Covers the four key scenarios:
1. Both sender and receiver are whitelisted → risk_score zeroed, is_anomaly False.
2. Only one side matches the whitelist → risk_score discounted (×0.15), anomaly flag preserved.
3. Neither side matches → no change.
4. Empty / None whitelist → no-op, no crash.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml_engine import apply_institutional_whitelist

# ── Fixtures ────────────────────────────────────────────────────────────────

WHITELIST_ADDRS = {
    "1binancecoldwallet",
    "1krakenhotwallet",
}

LABELS = {
    "1binancecoldwallet": "Binance",
    "1krakenhotwallet": "Kraken",
}


def _make_df(input_addresses, output_addresses, risk_scores, is_anomaly=None):
    """Build a minimal DataFrame matching the columns apply_institutional_whitelist expects."""
    n = len(risk_scores)
    if is_anomaly is None:
        is_anomaly = [True] * n
    return pd.DataFrame({
        "input_addresses": input_addresses,
        "output_addresses": output_addresses,
        "risk_score": risk_scores,
        "is_anomaly": is_anomaly,
        "explanation": ["Suspicious pattern detected."] * n,
    })


# ── Test cases ──────────────────────────────────────────────────────────────

class TestBothSidesWhitelisted:
    """When both input and output are whitelisted institutional entities."""

    def test_risk_score_zeroed(self):
        df = _make_df(
            input_addresses=["1BinanceColdWallet"],
            output_addresses=["1KrakenHotWallet"],
            risk_scores=[92.5],
        )
        result = apply_institutional_whitelist(df, whitelist=WHITELIST_ADDRS, labels=LABELS)

        assert result.at[0, "risk_score"] == 0.0
        assert result.at[0, "is_anomaly"] is False or result.at[0, "is_anomaly"] == False
        assert result.at[0, "whitelisted_side"] == "both"
        assert "Binance" in result.at[0, "whitelisted_entity"]
        assert "Kraken" in result.at[0, "whitelisted_entity"]

    def test_multiple_rows_both_cleared(self):
        df = _make_df(
            input_addresses=["1BinanceColdWallet", "1KrakenHotWallet"],
            output_addresses=["1KrakenHotWallet", "1BinanceColdWallet"],
            risk_scores=[80.0, 95.0],
        )
        result = apply_institutional_whitelist(df, whitelist=WHITELIST_ADDRS, labels=LABELS)

        np.testing.assert_array_equal(result["risk_score"].values, [0.0, 0.0])
        assert all(result["is_anomaly"] == False)
        assert all(result["whitelisted_side"] == "both")


class TestOneSideWhitelisted:
    """When only one side (input OR output) is whitelisted."""

    def test_input_only_discounted(self):
        original_risk = 80.0
        df = _make_df(
            input_addresses=["1BinanceColdWallet"],
            output_addresses=["1UnknownWalletXYZ"],
            risk_scores=[original_risk],
        )
        result = apply_institutional_whitelist(df, whitelist=WHITELIST_ADDRS, labels=LABELS)

        # risk_score is no longer modified here — discount is applied
        # downstream by compute_fused_priority via whitelist_factor.
        assert result.at[0, "risk_score"] == original_risk
        # Anomaly flag is preserved — the unverified side is still suspicious
        assert result.at[0, "is_anomaly"] == True
        assert result.at[0, "whitelisted_side"] == "input"
        assert result.at[0, "whitelisted_entity"] == "Binance"

    def test_output_only_discounted(self):
        original_risk = 60.0
        df = _make_df(
            input_addresses=["1SuspiciousSender"],
            output_addresses=["1KrakenHotWallet"],
            risk_scores=[original_risk],
        )
        result = apply_institutional_whitelist(df, whitelist=WHITELIST_ADDRS, labels=LABELS)

        # risk_score preserved — discount via whitelist_factor downstream
        assert result.at[0, "risk_score"] == original_risk
        assert result.at[0, "is_anomaly"] == True
        assert result.at[0, "whitelisted_side"] == "output"
        assert result.at[0, "whitelisted_entity"] == "Kraken"

    def test_pipe_delimited_addresses(self):
        """Addresses may be pipe-delimited; a match on any sub-address counts."""
        original_risk = 50.0
        df = _make_df(
            input_addresses=["1randomaddr|1BinanceColdWallet|1otheraddr"],
            output_addresses=["1completelyunknown"],
            risk_scores=[original_risk],
        )
        result = apply_institutional_whitelist(df, whitelist=WHITELIST_ADDRS, labels=LABELS)

        # risk_score preserved — discount via whitelist_factor downstream
        assert result.at[0, "risk_score"] == original_risk
        assert result.at[0, "whitelisted_side"] == "input"


class TestNeitherSideWhitelisted:
    """When neither side matches the whitelist — scores must be unchanged."""

    def test_risk_unchanged(self):
        original_risks = [80.0, 95.0, 50.0]
        df = _make_df(
            input_addresses=["1abc", "1def", "1ghi"],
            output_addresses=["1xyz", "1uvw", "1rst"],
            risk_scores=original_risks,
        )
        result = apply_institutional_whitelist(df, whitelist=WHITELIST_ADDRS, labels=LABELS)

        np.testing.assert_array_equal(result["risk_score"].values, original_risks)
        assert all(result["is_anomaly"] == True)
        assert all(result["whitelisted_side"].isna())


class TestEmptyOrNoneWhitelist:
    """Edge cases: empty set or None whitelist should be a no-op."""

    def test_empty_set_whitelist(self):
        original_risks = [70.0, 30.0]
        df = _make_df(
            input_addresses=["1BinanceColdWallet", "1abc"],
            output_addresses=["1KrakenHotWallet", "1xyz"],
            risk_scores=original_risks,
        )
        result = apply_institutional_whitelist(df, whitelist=set(), labels={})

        np.testing.assert_array_equal(result["risk_score"].values, original_risks)
        assert "whitelisted_side" in result.columns
        assert "whitelisted_entity" in result.columns

    def test_none_whitelist_no_csv(self, tmp_path):
        """Passing whitelist=None with a non-existent CSV should gracefully no-op."""
        original_risks = [55.0]
        df = _make_df(
            input_addresses=["1abc"],
            output_addresses=["1xyz"],
            risk_scores=original_risks,
        )
        fake_csv = str(tmp_path / "does_not_exist.csv")
        result = apply_institutional_whitelist(df, whitelist=None, labels=None, whitelist_csv=fake_csv)

        np.testing.assert_array_equal(result["risk_score"].values, original_risks)
        assert "whitelisted_side" in result.columns
