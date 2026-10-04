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

import os
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# FUSED PRIORITY WEIGHTS (visible in every dossier / report)
# ═══════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    "ml":       0.50,   # IsolationForest anomaly deviation
    "pattern":  0.25,   # Structural pattern detector (CoinJoin, Peel, …)
    "network":  0.15,   # Port / network risk signal
    "graph":    0.10,   # Graph taint propagation (when available)
}

# Known anonymity-network ports (mirrors explainability._TOR_PORTS / _I2P_PORTS)
_ANON_PORTS = {9050, 9051, 9150, 4444, 4445}



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


def compute_fused_priority(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a transparent, multi-signal fused priority score.

    Adds the following columns (all 0–100 before whitelist factor):

    * ``score_ml`` — IsolationForest deviation (already in ``risk_score``
      at call time, before any whitelist discount).
    * ``score_pattern`` — structural-type detector bonus.
    * ``score_network`` — port-risk signal, forced to 100 for Tor/I2P.
    * ``score_graph`` — taint propagation (``taint_score`` column, or 0).
    * ``whitelist_factor`` — 0.0 / 0.15 / 1.0 based on ``whitelisted_side``.
    * ``risk_score`` — final fused score (overwrites the old ML-only value).

    The original ML score is preserved in ``score_ml``.

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain ``risk_score`` (the raw ML score) and
        ``detected_type``.  ``whitelisted_side`` is optional.
    features : pd.DataFrame
        Engineered feature matrix (same row order as *df*).

    Returns
    -------
    pd.DataFrame
        *df* mutated in-place with the new columns.
    """
    w = WEIGHTS

    # ── score_ml: preserve raw IF deviation or fuse with ml_probability ─
    iso_score = df["risk_score"].copy()
    if "ml_probability" in df.columns:
        df["score_ml"] = (0.5 * iso_score + 0.5 * df["ml_probability"]).round(1)
    else:
        rf_path = "models/rf_model.joblib"
        if os.path.isfile(rf_path):
            try:
                import joblib
                rf_model = joblib.load(rf_path)
                if hasattr(rf_model, "feature_names_in_"):
                    expected_cols = list(rf_model.feature_names_in_)
                else:
                    expected_cols = list(features.columns)
                X_rf = pd.DataFrame(index=features.index)
                for c in expected_cols:
                    X_rf[c] = features[c] if c in features.columns else 0.0
                probas = rf_model.predict_proba(X_rf)[:, 1] * 100.0
                df["ml_probability"] = probas.round(2)
                df["score_ml"] = (0.5 * iso_score + 0.5 * df["ml_probability"]).round(1)
            except Exception:
                df["score_ml"] = iso_score
        else:
            df["score_ml"] = iso_score

    # ── score_pattern ────────────────────────────────────────────────
    _pattern_map = {
        "CoinJoin_Mixer": 100.0,
        "Peel_Chain": 100.0,
        "FanOut_Dispersal": 70.0,
        "Fee_Spike": 70.0,
    }
    df["score_pattern"] = df["detected_type"].map(_pattern_map).fillna(0.0)

    # ── score_network ────────────────────────────────────────────────
    if "port_risk_combined" in features.columns:
        prc = features["port_risk_combined"].values.astype(float)
        prc_min, prc_max = prc.min(), prc.max()
        if prc_max > prc_min:
            net_scaled = (prc - prc_min) / (prc_max - prc_min) * 100.0
        else:
            net_scaled = np.zeros(len(prc))
        df["score_network"] = np.clip(net_scaled, 0, 100).round(1)
    else:
        df["score_network"] = 0.0

    # Force 100 when src_port or dst_port is a known Tor / I2P port
    for col in ("src_port", "dst_port"):
        if col in df.columns:
            mask = df[col].astype(float).isin(_ANON_PORTS)
            df.loc[mask, "score_network"] = 100.0

    # ── score_graph ──────────────────────────────────────────────────
    if "taint_score" in df.columns:
        df["score_graph"] = df["taint_score"].fillna(0.0).clip(0, 100)
    else:
        df["score_graph"] = 0.0

    # Direct watchlist matches always get maximum graph taint
    if "watchlist_hit" in df.columns:
        df.loc[df["watchlist_hit"] == True, "score_graph"] = 100.0

    # ── whitelist_factor ─────────────────────────────────────────────
    def _wl_factor(side):
        if side == "both":
            return 0.0
        if side in ("input", "output"):
            return 0.15
        return 1.0

    if "whitelisted_side" in df.columns:
        df["whitelist_factor"] = df["whitelisted_side"].apply(
            lambda s: _wl_factor(s) if pd.notna(s) else 1.0
        )
    else:
        df["whitelist_factor"] = 1.0

    # ── Fused score ──────────────────────────────────────────────────
    raw_fused = (
        w["ml"]      * df["score_ml"]
        + w["pattern"] * df["score_pattern"]
        + w["network"] * df["score_network"]
        + w["graph"]   * df["score_graph"]
    )
    df["risk_score"] = (raw_fused * df["whitelist_factor"]).round(1)

    return df


def assign_risk_tier(score: float) -> str:
    """
    Map an Anomaly Deviation Score to a categorical risk tier.

    Parameters
    ----------
    score : float
        Risk score in [0, 100].

    Returns
    -------
    str
        One of ``"Critical"``, ``"High"``, ``"Medium"``, or ``"Low"``.
    """
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


# ── Backward-compatible alias ────────────────────────────────────────
compute_risk_scores = compute_anomaly_deviation_scores
