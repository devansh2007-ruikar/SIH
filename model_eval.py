"""
model_eval.py — Supervised Random Forest Evaluation & Calibration
==================================================================

Provides an honest evaluation harness for supervised threat classification:
- Calibrated Random Forest (Isotonic regression, 3-fold CV)
- Honest 5-fold Stratified cross-validation scoring for synthetic data
- 70/30 stratified train/test hold-out report
- Precision@25, ROC-AUC, PR-AUC, Recall, F1, Confusion Matrix, Per-Attack-Type Recall
- Expected Calibration Error (ECE, 10 bins) before and after calibration
- Benchmarks against 3 baselines: Isolation Forest only, Rules only, and Fused Score
- Generalisation test on an unseen, independently seeded synthetic dataset
- Model persistence to models/rf_model.joblib and models/rf_model.sha256
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import anomaly_engine
import generate_data
import ml_engine

# Explicit set of prohibited label/target columns to prevent any label leakage
PROHIBITED_LABEL_COLUMNS = {
    "attack_type",
    "is_labeled_suspicious",
    "detected_type",
    "label",
    "is_anomaly",
    "watchlist_label",
    "watchlist_hit",
    "watchlist_indicator",
    "risk_score",
    "score_ml",
    "score_pattern",
    "score_network",
    "score_graph",
    "whitelist_factor",
    "ml_probability",
    "y",
    "target",
}


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) across equally spaced probability bins.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels in {0, 1}.
    y_prob : np.ndarray
        Predicted probabilities in [0.0, 1.0].
    n_bins : int
        Number of bins (default: 10).

    Returns
    -------
    float
        ECE score in [0.0, 1.0]. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 25) -> float:
    """
    Calculate Precision@K (proportion of actual threats in the top-K highest scored transactions).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = min(k, len(y_true))
    if k == 0:
        return 0.0
    top_indices = np.argsort(-scores)[:k]
    return float(np.mean(y_true[top_indices]))


def compute_method_metrics(
    y_true: np.ndarray,
    scores_or_proba: np.ndarray,
    attack_types: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    method_name: str = "",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification and ranking metrics for a score / probability vector.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores_or_proba, dtype=float)

    # Normalize if values are on 0-100 scale
    if scores.max() > 1.0:
        norm_scores = np.clip(scores / 100.0, 0.0, 1.0)
    else:
        norm_scores = np.clip(scores, 0.0, 1.0)

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, norm_scores))
    except Exception:
        roc_auc = 0.5

    # PR-AUC
    try:
        pr_auc = float(average_precision_score(y_true, norm_scores))
    except Exception:
        pr_auc = float(np.mean(y_true))

    # Precision@25
    p25 = compute_precision_at_k(y_true, norm_scores, k=25)

    # Binary decision metrics
    y_pred = (norm_scores >= threshold).astype(int)
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    ece = compute_ece(y_true, norm_scores, n_bins=10)

    # Per-attack-type recall
    per_type_rec: Dict[str, float] = {}
    if attack_types is not None:
        at_series = pd.Series(attack_types)
        threat_mask = y_true == 1
        for atype in sorted(at_series[threat_mask].unique()):
            mask = (at_series == atype).values
            if mask.sum() > 0:
                per_type_rec[str(atype)] = float(np.mean(y_pred[mask] == 1))

    # ROC & PR curves for Altair
    try:
        fpr, tpr, _ = roc_curve(y_true, norm_scores)
    except Exception:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    try:
        pr_p, pr_r, _ = precision_recall_curve(y_true, norm_scores)
    except Exception:
        pr_p, pr_r = np.array([1, 0]), np.array([0, 1])

    return {
        "method": method_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_at_25": p25,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "ece": ece,
        "confusion_matrix": cm,
        "per_type_recall": per_type_rec,
        "fpr_curve": fpr.tolist(),
        "tpr_curve": tpr.tolist(),
        "pr_precision_curve": pr_p.tolist(),
        "pr_recall_curve": pr_r.tolist(),
    }


def run_evaluation_and_training(
    df: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    save_model: bool = True,
    run_gen_test: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Execute the supervised Random Forest training, honest cross-val scoring,
    hold-out evaluation, and generalisation test.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame. Must contain 'attack_type' to proceed.
    features : pd.DataFrame | None
        Precomputed engineered features. If None, computes via ml_engine.engineer_features.
    save_model : bool
        Whether to dump models/rf_model.joblib and models/rf_model.sha256.
    run_gen_test : bool
        Whether to generate and evaluate on an unseen 1500-record dataset.

    Returns
    -------
    dict | None
        Dictionary of all metrics and artifacts, or None if df has no 'attack_type'.
    """
    if "attack_type" not in df.columns:
        return None

    # 1. Label formulation: 1 if attack_type is not Normal_P2P or Whitelisted_Institutional, else 0
    y = (~df["attack_type"].isin({"Normal_P2P", "Whitelisted_Institutional"})).astype(int).values

    # 2. Informative features X (same as Isolation Forest uses)
    if features is None:
        features, df = ml_engine.engineer_features(df)

    X = ml_engine.select_informative_features(features)

    # Assert no label columns from rule 3 are in X
    for col in PROHIBITED_LABEL_COLUMNS:
        assert col not in X.columns, f"[LEAKAGE ERROR] Label column '{col}' found in feature matrix X!"

    # 3. Model specification:
    # CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=1), method="isotonic", cv=3)
    base_rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=base_rf,
        method="isotonic",
        cv=3,
    )

    # 4. Honest scoring via 5-fold cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_proba = cross_val_predict(
        calibrated_model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    df["ml_probability"] = (cv_proba[:, 1] * 100.0).round(2)

    # 5. Hold-out report: a separate 70/30 stratified split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.30,
        stratify=y,
        random_state=42,
    )
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # Train uncalibrated RF on 70% train set
    rf_uncal = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    rf_uncal.fit(X_train, y_train)
    uncal_test_proba = rf_uncal.predict_proba(X_test)[:, 1]

    # Train calibrated RF on 70% train set
    rf_cal = CalibratedClassifierCV(estimator=rf_uncal, method="isotonic", cv=3)
    rf_cal.fit(X_train, y_train)
    cal_test_proba = rf_cal.predict_proba(X_test)[:, 1]

    # Hold-out RF metrics
    rf_metrics = compute_method_metrics(
        y_test,
        cal_test_proba,
        attack_types=df_test["attack_type"].values,
        threshold=0.5,
        method_name="Supervised RF (Calibrated)",
    )
    ece_before = compute_ece(y_test, uncal_test_proba, n_bins=10)
    ece_after = compute_ece(y_test, cal_test_proba, n_bins=10)
    rf_metrics["ece_before"] = ece_before
    rf_metrics["ece_after"] = ece_after

    # Compute baseline signals on the SAME test rows:
    # Isolation Forest only (score_ml), rules only (score_pattern), and fused risk_score
    test_features = features.iloc[test_idx].copy()

    # 1) Isolation Forest baseline (pure IF anomaly deviation)
    if_model = ml_engine.train_model(features)[0]
    if_cols = list(getattr(if_model, "feature_names_in_", features.columns))
    aligned_test_if = pd.DataFrame(index=test_features.index)
    for c in if_cols:
        aligned_test_if[c] = test_features[c] if c in test_features.columns else 0.0
    raw_if = if_model.decision_function(aligned_test_if)
    raw_if_scores = anomaly_engine.compute_anomaly_deviation_scores(raw_if)

    # Compute rules
    df_test["detected_type"] = ml_engine.classify_structural_type(df_test, test_features)
    ml_engine.finalize_detected_type(df_test)
    df_test = ml_engine.apply_institutional_whitelist(df_test)
    _wl = ml_engine.load_watchlist()
    ml_engine.apply_watchlist(df_test, _wl)
    ml_engine.compute_taint(df_test, _wl)

    # Set risk_score to raw IF score so compute_fused_priority can fuse it
    df_test["risk_score"] = raw_if_scores.copy()
    anomaly_engine.compute_fused_priority(df_test, test_features)

    # 1) Isolation Forest baseline
    if_metrics = compute_method_metrics(
        y_test,
        raw_if_scores,
        attack_types=df_test["attack_type"].values,
        threshold=0.40,
        method_name="Isolation Forest only",
    )

    # 2) Rules only baseline
    rules_scores = df_test["score_pattern"].values
    rules_metrics = compute_method_metrics(
        y_test,
        rules_scores,
        attack_types=df_test["attack_type"].values,
        threshold=0.40,
        method_name="Rules only",
    )

    # 3) Fused risk score baseline
    fused_scores = df_test["risk_score"].values
    fused_metrics = compute_method_metrics(
        y_test,
        fused_scores,
        attack_types=df_test["attack_type"].values,
        threshold=0.40,
        method_name="Fused Score",
    )

    # Calibration curve points for calibration plot
    cal_prob_true, cal_prob_pred = calibration_curve(y_test, cal_test_proba, n_bins=10)
    uncal_prob_true, uncal_prob_pred = calibration_curve(y_test, uncal_test_proba, n_bins=10)

    # 6. Fit final model on full dataset and save to models/rf_model.joblib
    calibrated_model.fit(X, y)
    model_sha256 = ""
    if save_model:
        os.makedirs("models", exist_ok=True)
        joblib.dump(calibrated_model, "models/rf_model.joblib")
        with open("models/rf_model.joblib", "rb") as f:
            model_sha256 = hashlib.sha256(f.read()).hexdigest()
        with open("models/rf_model.sha256", "w") as f:
            f.write(model_sha256)
    else:
        sha_path = "models/rf_model.sha256"
        if os.path.isfile(sha_path):
            with open(sha_path, "r") as f:
                model_sha256 = f.read().strip()

    # 7. Generalisation test: generate a second dataset in memory with different random seed (1500 rows, ratio 0.2)
    gen_results = None
    if run_gen_test:
        gen_df = generate_data.generate_transactions_df(
            total_records=1500,
            suspicious_ratio=0.2,
            seed=999,
        )
        gen_features, gen_df = ml_engine.engineer_features(gen_df)
        gen_y = (~gen_df["attack_type"].isin({"Normal_P2P", "Whitelisted_Institutional"})).astype(int).values

        # Align features to X columns
        gen_X = pd.DataFrame(index=gen_features.index)
        for c in X.columns:
            gen_X[c] = gen_features[c] if c in gen_features.columns else 0.0

        gen_rf_proba = calibrated_model.predict_proba(gen_X)[:, 1]
        gen_rf_metrics = compute_method_metrics(
            gen_y,
            gen_rf_proba,
            attack_types=gen_df["attack_type"].values,
            threshold=0.5,
            method_name="Supervised RF (Calibrated)",
        )

        # Baseline evaluation on gen_df
        gen_df["detected_type"] = ml_engine.classify_structural_type(gen_df, gen_features)
        ml_engine.finalize_detected_type(gen_df)
        gen_df = ml_engine.apply_institutional_whitelist(gen_df)
        _wl = ml_engine.load_watchlist()
        ml_engine.apply_watchlist(gen_df, _wl)
        ml_engine.compute_taint(gen_df, _wl)

        # IF scoring on gen_df
        if_model_gen = ml_engine.train_model(gen_features)[0]
        gen_if_cols = list(getattr(if_model_gen, "feature_names_in_", gen_features.columns))
        aligned_gen_if = pd.DataFrame(index=gen_features.index)
        for c in gen_if_cols:
            aligned_gen_if[c] = gen_features[c] if c in gen_features.columns else 0.0
        gen_raw_if = if_model_gen.decision_function(aligned_gen_if)
        gen_raw_if_scores = anomaly_engine.compute_anomaly_deviation_scores(gen_raw_if)
        gen_df["risk_score"] = gen_raw_if_scores.copy()

        # Compute fused priority
        anomaly_engine.compute_fused_priority(gen_df, gen_features)

        gen_if_metrics = compute_method_metrics(
            gen_y,
            gen_df["score_ml"].values,
            attack_types=gen_df["attack_type"].values,
            threshold=0.40,
            method_name="Isolation Forest only",
        )
        gen_rules_metrics = compute_method_metrics(
            gen_y,
            gen_df["score_pattern"].values,
            attack_types=gen_df["attack_type"].values,
            threshold=0.40,
            method_name="Rules only",
        )
        gen_fused_metrics = compute_method_metrics(
            gen_y,
            gen_df["risk_score"].values,
            attack_types=gen_df["attack_type"].values,
            threshold=0.40,
            method_name="Fused Score",
        )

        gen_summary_rows = [
            {
                "Method": m["method"],
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "PR-AUC": f"{m['pr_auc']:.4f}",
                "Precision@25": f"{m['precision_at_25']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1": f"{m['f1']:.4f}",
                "ECE": f"{m['ece']:.4f}",
            }
            for m in [gen_rf_metrics, gen_if_metrics, gen_rules_metrics, gen_fused_metrics]
        ]
        gen_summary_df = pd.DataFrame(gen_summary_rows)

        gen_results = {
            "rf": gen_rf_metrics,
            "isolation_forest": gen_if_metrics,
            "rules": gen_rules_metrics,
            "fused": gen_fused_metrics,
            "table": gen_summary_df,
        }

    # Format hold-out summary table
    holdout_summary_rows = [
        {
            "Method": m["method"],
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f}",
            "Precision@25": f"{m['precision_at_25']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "ECE": f"{m['ece']:.4f}",
        }
        for m in [rf_metrics, if_metrics, rules_metrics, fused_metrics]
    ]
    holdout_summary_df = pd.DataFrame(holdout_summary_rows)

    return {
        "model_sha256": model_sha256,
        "holdout": {
            "rf": rf_metrics,
            "isolation_forest": if_metrics,
            "rules": rules_metrics,
            "fused": fused_metrics,
            "table": holdout_summary_df,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "calibration_curve_cal": {
                "prob_pred": cal_prob_pred.tolist(),
                "prob_true": cal_prob_true.tolist(),
            },
            "calibration_curve_uncal": {
                "prob_pred": uncal_prob_pred.tolist(),
                "prob_true": uncal_prob_true.tolist(),
            },
            "confusion_matrix": rf_metrics["confusion_matrix"],
            "per_type_recall": rf_metrics["per_type_recall"],
        },
        "generalisation": gen_results,
    }


def main():
    """CLI runner to execute evaluation on default synthetic dataset."""
    default_csv = "synthetic_transactions.csv"
    if not os.path.isfile(default_csv):
        print(f"[*] {default_csv} not found, generating sample...")
        generate_data.main()

    print(f"[*] Loading {default_csv}...")
    df = pd.read_csv(default_csv)
    results = run_evaluation_and_training(df, save_model=True, run_gen_test=True)

    if results is None:
        print("[!] No 'attack_type' column found. Evaluation skipped.")
        return

    ho = results["holdout"]
    rf = ho["rf"]
    print("\n" + "=" * 70)
    print("  SUPERVISED RANDOM FOREST EVALUATION REPORT")
    print("=" * 70)
    print(f"  Model SHA-256 : {results['model_sha256']}")
    print(f"  ROC-AUC       : {rf['roc_auc']:.4f}")
    print(f"  PR-AUC        : {rf['pr_auc']:.4f}")
    print(f"  Precision@25  : {rf['precision_at_25']:.4f}")
    print(f"  Recall        : {rf['recall']:.4f}")
    print(f"  F1-Score      : {rf['f1']:.4f}")
    print(f"  ECE Before    : {ho['ece_before']:.4f}")
    print(f"  ECE After     : {ho['ece_after']:.4f}")
    print("-" * 70)
    print("  Hold-Out Performance (70/30 Stratified Split):")
    print(ho["table"].to_string(index=False))
    print("-" * 70)
    print("  Per-Attack-Type Recall:")
    for k, v in ho["per_type_recall"].items():
        print(f"    {k:<28}: {v * 100:.1f}%")
    print("-" * 70)
    print("  Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"    {ho['confusion_matrix']}")
    print("=" * 70)

    if results.get("generalisation"):
        print("\n" + "=" * 70)
        print("  GENERALISATION TEST (Unseen Dataset: 1500 rows, ratio 0.2, seed=999)")
        print("=" * 70)
        print(results["generalisation"]["table"].to_string(index=False))
        print("=" * 70)


if __name__ == "__main__":
    main()
