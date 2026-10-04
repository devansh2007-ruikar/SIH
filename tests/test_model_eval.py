"""
tests/test_model_eval.py — Unit Tests for Supervised RF & Model Evaluation
==========================================================================

Tests:
1. model_eval only runs on DataFrames with 'attack_type'.
2. y mapping: 1 for threats, 0 for Normal_P2P and Whitelisted_Institutional.
3. Feature leakage assertion: no label columns in X.
4. Model calibration and honest scoring (StratifiedKFold cross_val_predict).
5. Hold-out report metrics (ROC-AUC, PR-AUC, Precision@25, Recall, F1, ECE, CM, per-type recall).
6. Baselines (IF, rules, fused score) evaluated on the same test rows.
7. Generalisation test with independent seed.
8. Artifact persistence: models/rf_model.joblib and models/rf_model.sha256.
9. Fused scoring in anomaly_engine.py with ml_probability and fallback paths.
"""

import os
import hashlib
import tempfile
import numpy as np
import pandas as pd
import pytest

import anomaly_engine
import generate_data
import ml_engine
import model_eval


@pytest.fixture
def synthetic_sample_df():
    """Generate a small synthetic dataset in memory."""
    return generate_data.generate_transactions_df(
        total_records=200,
        suspicious_ratio=0.25,
        seed=42,
    )


def test_model_eval_requires_attack_type():
    """model_eval must return None when the DataFrame has no 'attack_type' column."""
    df_no_labels = pd.DataFrame({
        "txid": ["tx1", "tx2"],
        "input_addresses": ["addrA", "addrB"],
        "output_addresses": ["addrC", "addrD"],
        "input_amounts": ["1.0", "2.0"],
        "output_amounts": ["0.99", "1.99"],
        "fee": [0.01, 0.01],
        "src_port": [8333, 8333],
    })
    res = model_eval.run_evaluation_and_training(df_no_labels)
    assert res is None


def test_label_mapping_and_no_leakage(synthetic_sample_df):
    """Verify y = 1 for threats, 0 for Normal_P2P / Whitelisted_Institutional, and no label columns in X."""
    df = synthetic_sample_df.copy()
    features, df = ml_engine.engineer_features(df)
    X = ml_engine.select_informative_features(features)

    # Check labels
    y = (~df["attack_type"].isin({"Normal_P2P", "Whitelisted_Institutional"})).astype(int)
    for atype, label in zip(df["attack_type"], y):
        if atype in {"Normal_P2P", "Whitelisted_Institutional"}:
            assert label == 0
        else:
            assert label == 1

    # Check leakage assertion
    for col in model_eval.PROHIBITED_LABEL_COLUMNS:
        assert col not in X.columns


def test_ece_and_precision_at_k():
    """Verify compute_ece and compute_precision_at_k edge cases and correctness."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # Perfectly confident and accurate
    y_prob_perf = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ece_perf = model_eval.compute_ece(y_true, y_prob_perf, n_bins=10)
    assert ece_perf == pytest.approx(0.0, abs=1e-5)

    # Precision@3 should be 1.0, Precision@10 should be 0.3
    assert model_eval.compute_precision_at_k(y_true, y_prob_perf, k=3) == 1.0
    assert model_eval.compute_precision_at_k(y_true, y_prob_perf, k=10) == 0.3


def test_run_evaluation_and_training(synthetic_sample_df):
    """Full execution of model_eval pipeline: honest scoring, hold-out report, and artifacts."""
    df = synthetic_sample_df.copy()
    features, df = ml_engine.engineer_features(df)

    results = model_eval.run_evaluation_and_training(
        df,
        features=features,
        save_model=True,
        run_gen_test=True,
    )

    assert results is not None
    assert "model_sha256" in results
    assert len(results["model_sha256"]) == 64

    # Verify honest scoring added ml_probability to df in [0, 100]
    assert "ml_probability" in df.columns
    assert df["ml_probability"].min() >= 0.0
    assert df["ml_probability"].max() <= 100.0

    # Hold-out report metrics
    ho = results["holdout"]
    rf = ho["rf"]
    assert 0.0 <= rf["roc_auc"] <= 1.0
    assert 0.0 <= rf["pr_auc"] <= 1.0
    assert 0.0 <= rf["precision_at_25"] <= 1.0
    assert 0.0 <= rf["recall"] <= 1.0
    assert 0.0 <= rf["f1"] <= 1.0
    assert 0.0 <= ho["ece_after"] <= 1.0

    # Baselines present
    assert "isolation_forest" in ho
    assert "rules" in ho
    assert "fused" in ho

    # Check per-attack-type recall
    assert len(ho["per_type_recall"]) > 0
    for atype, rec in ho["per_type_recall"].items():
        assert 0.0 <= rec <= 1.0

    # Check confusion matrix format [[TN, FP], [FN, TP]]
    cm = ho["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2

    # Check generalisation results
    gen = results["generalisation"]
    assert gen is not None
    assert "table" in gen
    assert len(gen["table"]) == 4  # 4 methods

    # Check persisted files
    assert os.path.isfile("models/rf_model.joblib")
    assert os.path.isfile("models/rf_model.sha256")
    with open("models/rf_model.sha256") as f:
        file_hash = f.read().strip()
    with open("models/rf_model.joblib", "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == actual_hash


def test_fused_scoring_with_ml_probability():
    """Verify compute_fused_priority fuses score_ml = 0.5 * IF + 0.5 * ml_probability."""
    df = pd.DataFrame({
        "txid": ["tx1", "tx2"],
        "risk_score": [80.0, 20.0],
        "ml_probability": [90.0, 10.0],
        "detected_type": ["CoinJoin_Mixer", "Normal"],
        "src_port": [8333, 8333],
        "dst_port": [8333, 8333],
    })
    features = pd.DataFrame(index=df.index)
    features["port_risk_combined"] = [0.0, 0.0]

    anomaly_engine.compute_fused_priority(df, features)

    # Expected score_ml = 0.5 * 80 + 0.5 * 90 = 85.0
    assert df.loc[0, "score_ml"] == pytest.approx(85.0, abs=0.1)
    # Expected score_ml = 0.5 * 20 + 0.5 * 10 = 15.0
    assert df.loc[1, "score_ml"] == pytest.approx(15.0, abs=0.1)


def test_fused_scoring_unlabelled_data_loads_model():
    """When ml_probability does not exist, compute_fused_priority loads models/rf_model.joblib."""
    assert os.path.isfile("models/rf_model.joblib"), "Pre-trained RF model must exist"

    df = pd.DataFrame({
        "txid": ["tx1"],
        "risk_score": [60.0],
        "detected_type": ["Normal"],
        "src_port": [8333],
        "dst_port": [8333],
    })
    features = pd.DataFrame(index=df.index)
    features["port_risk_combined"] = [0.0]

    anomaly_engine.compute_fused_priority(df, features)

    assert "ml_probability" in df.columns
    assert "score_ml" in df.columns
    # score_ml must be 0.5 * 60 + 0.5 * ml_probability
    expected = 0.5 * 60.0 + 0.5 * df.loc[0, "ml_probability"]
    assert df.loc[0, "score_ml"] == pytest.approx(expected, abs=0.1)
