"""
tests/test_detected_type.py — Validate the detector-based detected_type column
================================================================================

Ensures that the pipeline:
  1. Produces a non-null ``detected_type`` for every row.
  2. Covers at least CoinJoin_Mixer, Peel_Chain, FanOut_Dispersal, Fee_Spike.
  3. Achieves ≥ 98 % agreement with the synthetic ground-truth ``attack_type``.
  4. Generates CoinJoin-related explanations for mixer rows.
"""

import os
import sys

import pandas as pd
import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transaction_adapter import BitcoinCSVAdapter

SYNTHETIC_CSV = os.path.join(_PROJECT_ROOT, "synthetic_transactions.csv")


@pytest.fixture(scope="module")
def pipeline_results_no_labels():
    """Load the CSV *without* ground-truth labels and run the full pipeline."""
    df_raw = pd.read_csv(SYNTHETIC_CSV, parse_dates=["timestamp"])
    # Drop the label-leakage columns
    df_raw = df_raw.drop(columns=["attack_type", "is_labeled_suspicious"], errors="ignore")
    adapter = BitcoinCSVAdapter()
    enriched_df, _model, _features = adapter.run_pipeline(df_raw, contamination=0.05)
    return enriched_df


@pytest.fixture(scope="module")
def pipeline_results_with_labels():
    """Load the CSV *with* ground-truth labels and run the full pipeline."""
    df_raw = pd.read_csv(SYNTHETIC_CSV, parse_dates=["timestamp"])
    # Keep attack_type for comparison, but ensure pipeline doesn't use it
    adapter = BitcoinCSVAdapter()
    enriched_df, _model, _features = adapter.run_pipeline(df_raw, contamination=0.05)
    return enriched_df


# ── Test 1: No nulls in detected_type ───────────────────────────────────

def test_detected_type_no_nulls(pipeline_results_no_labels):
    """detected_type must be populated for every row, with no NaN/None."""
    df = pipeline_results_no_labels
    assert "detected_type" in df.columns, "detected_type column missing"
    assert df["detected_type"].notna().all(), (
        f"Found {df['detected_type'].isna().sum()} null values in detected_type"
    )


# ── Test 2: Coverage of expected types ──────────────────────────────────

def test_detected_type_coverage(pipeline_results_no_labels):
    """Pipeline must detect at least the four main structural patterns."""
    df = pipeline_results_no_labels
    detected_values = set(df["detected_type"].unique())
    required = {"CoinJoin_Mixer", "Peel_Chain", "FanOut_Dispersal", "Fee_Spike"}
    missing = required - detected_values
    assert not missing, (
        f"detected_type is missing expected values: {missing}. "
        f"Found: {detected_values}"
    )


# ── Test 3: Agreement with ground-truth ≥ 98 % ─────────────────────────

def test_detected_type_agreement(pipeline_results_with_labels):
    """detected_type should agree with attack_type ≥ 98 % of the time."""
    df = pipeline_results_with_labels
    assert "attack_type" in df.columns, (
        "attack_type column missing — cannot compare"
    )
    assert "detected_type" in df.columns, "detected_type column missing"

    agreement = (df["detected_type"] == df["attack_type"]).mean()
    assert agreement >= 0.98, (
        f"Agreement between detected_type and attack_type is only "
        f"{agreement:.4f} (expected ≥ 0.98)"
    )


# ── Test 4: CoinJoin explanation text ───────────────────────────────────

def test_coinjoin_explanation_text(pipeline_results_no_labels):
    """At least one CoinJoin_Mixer row's explanation must mention 'CoinJoin'."""
    df = pipeline_results_no_labels
    mixer_rows = df[df["detected_type"] == "CoinJoin_Mixer"]
    assert len(mixer_rows) > 0, "No CoinJoin_Mixer rows detected"

    has_coinjoin_in_explanation = mixer_rows["explanation"].str.contains(
        "CoinJoin", case=False, na=False,
    ).any()
    assert has_coinjoin_in_explanation, (
        "None of the CoinJoin_Mixer rows mention 'CoinJoin' in their explanation"
    )
