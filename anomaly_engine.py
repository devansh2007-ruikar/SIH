"""
anomaly_engine.py — Anomaly Deviation Scoring
==============================================

Extracts and renames the anomaly scoring pipeline from ``ml_engine.py``
with principled terminology that reflects what the IsolationForest model
actually measures.

Key Concepts
------------
- **Anomaly Deviation Score (0–100)**: Measures unsupervised distance
  from standard transaction distributions via IsolationForest's
  ``decision_function()``.  This is NOT a calibrated probability of
  criminal activity, guilt, or confidence.  It represents how
  statistically unusual a transaction is relative to the learned
  distribution of normal behavior.

- **Investigative Priority Index (0–100)**: A composite score that
  combines the deviation score with feature-level severity signals
  to help analysts prioritize which transactions to examine first.

Scoring Methodology
-------------------
IsolationForest produces a raw ``decision_function`` score where:
- **Positive values** → normal (close to the learned distribution)
- **Negative values** → anomalous (far from the distribution)
- **More negative** → more anomalous

We invert and min-max scale to [0, 100] so that:
- **0** → perfectly normal (closest to distribution center)
- **100** → maximum deviation (furthest from distribution)

Usage
-----
::

    from anomaly_engine import compute_anomaly_deviation_scores

    raw_scores = model.decision_function(features)
    deviation_scores = compute_anomaly_deviation_scores(raw_scores)
"""

import numpy as np
import pandas as pd


def compute_anomaly_deviation_scores(raw_scores: np.ndarray) -> np.ndarray:
    """
    Convert IsolationForest raw decision_function scores into
    Anomaly Deviation Scores on a [0, 100] scale.

    This score represents **unsupervised statistical distance** from
    the learned transaction distribution.  It is NOT:

    - A calibrated probability of criminal activity
    - A confidence score or guilt index
    - A classifier posterior probability

    It should be interpreted as: "How far does this transaction
    deviate from the baseline behavior learned by the model?"

    Parameters
    ----------
    raw_scores : np.ndarray
        Raw output from ``IsolationForest.decision_function()``.
        More negative values indicate greater anomaly.

    Returns
    -------
    np.ndarray
        Anomaly Deviation Scores in [0.0, 100.0], rounded to 1
        decimal place.  Higher values = greater deviation from
        baseline.
    """
    # Invert: IsolationForest uses negative = anomalous
    inverted = -raw_scores
    min_score = inverted.min()
    max_score = inverted.max()

    if max_score > min_score:
        scaled = (inverted - min_score) / (max_score - min_score)
        deviation_pct = scaled * 100.0
    else:
        deviation_pct = np.zeros_like(inverted)

    return np.clip(deviation_pct, 0, 100).round(1)


def compute_investigative_priority(
    deviation_scores: np.ndarray,
    features_df: pd.DataFrame,
    deviation_weight: float = 0.70,
    feature_weight: float = 0.30,
) -> np.ndarray:
    """
    Compute an Investigative Priority Index (0–100) by combining
    anomaly deviation scores with feature-level severity.

    This composite metric helps analysts prioritize transactions
    for manual review.  The feature-severity component rewards
    transactions that trigger multiple behavioral heuristics
    simultaneously (peel chains, fan-out, fee urgency, etc.).

    Parameters
    ----------
    deviation_scores : np.ndarray
        Output from :func:`compute_anomaly_deviation_scores`.
    features_df : pd.DataFrame
        The engineered feature matrix.
    deviation_weight : float
        Weight for the deviation score component (default 0.70).
    feature_weight : float
        Weight for the feature severity component (default 0.30).

    Returns
    -------
    np.ndarray
        Investigative Priority Index in [0.0, 100.0].
    """
    # Compute feature severity: count how many behavioral signals are
    # elevated (above their respective dataset medians)
    severity_features = [
        "peel_chain_disparity",
        "fan_out",
        "fee_rate_urgency",
        "port_risk_combined",
        "entity_ip_diversity",
        "value_zscore_abs",
        "peel_chain_length",
    ]

    available = [f for f in severity_features if f in features_df.columns]

    if not available:
        return deviation_scores.copy()

    # For each feature, compute whether it exceeds the 75th percentile
    severity_counts = np.zeros(len(features_df))
    for feat in available:
        col = features_df[feat]
        p75 = col.quantile(0.75)
        severity_counts += (col > p75).astype(float).values

    # Normalize severity to [0, 100]
    max_possible = len(available)
    if max_possible > 0:
        severity_pct = (severity_counts / max_possible) * 100.0
    else:
        severity_pct = np.zeros(len(features_df))

    # Weighted combination
    priority = (
        deviation_weight * deviation_scores
        + feature_weight * severity_pct
    )

    return np.clip(priority, 0, 100).round(1)


# ── Backward-compatible alias ────────────────────────────────────────
compute_risk_scores = compute_anomaly_deviation_scores
