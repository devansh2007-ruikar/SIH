import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from anomaly_engine import compute_investigative_priority

def test_whitelist_discount():
    """Asserts counterparty risk is preserved (discounted 0.15x) when interacting with whitelisted exchange nodes."""
    # Test base anomaly score without any mitigation
    base_scores = pd.Series([80.0, 95.0, 50.0])
    
    # Test with full institutional clearance (both sides whitelisted) -> risk should be 0.0
    cleared_scores = compute_investigative_priority(
        base_scores, 
        pd.Series([1, 1, 1]), # is_cleared_institutional
        pd.Series([0, 0, 0])  # is_mitigated_institutional
    )
    assert np.all(cleared_scores == 0.0)

    # Test with partial mitigation (one side whitelisted) -> risk should be discounted by 0.15x
    mitigated_scores = compute_investigative_priority(
        base_scores,
        pd.Series([0, 0, 0]), # is_cleared_institutional
        pd.Series([1, 1, 1])  # is_mitigated_institutional
    )
    
    expected_mitigated = base_scores * 0.15
    np.testing.assert_array_almost_equal(mitigated_scores, expected_mitigated)

    # Test standard risk (no whitelist involvement)
    standard_scores = compute_investigative_priority(
        base_scores,
        pd.Series([0, 0, 0]),
        pd.Series([0, 0, 0])
    )
    np.testing.assert_array_almost_equal(standard_scores, base_scores)
