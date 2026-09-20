"""
explainability.py — Statistical Percentile Explainability
=========================================================

Replaces arbitrary manual thresholds in the MITHYA XAI pipeline with
dynamic statistical profiling relative to the ingested dataset.

Instead of hardcoded rules like ``if fan_out > 5``, this module
computes the actual distribution (median, P75, P95, P99) of each
behavioral feature across the full dataset, then explains each
flagged transaction in terms of its **percentile rank** against
that distribution.

Key Components
--------------
1. **DatasetProfile** — per-feature statistics computed once from the
   full ingested dataset (median, percentiles, std).

2. **Percentile ranking** — locates each transaction's feature value
   within the dataset distribution (e.g., "99.2th percentile").

3. **Structured telemetry** — machine-readable dicts containing
   ``feature_name``, ``observed_value``, ``dataset_median``,
   ``percentile_rank``, and ``heuristic_flags``.

4. **Human-readable explanations** — audit-grade text comparing the
   transaction against the baseline population.

Design Principle
----------------
All thresholds derive from the **95th percentile** of the actual
ingested data, not from pre-selected magic numbers.  This ensures
the system adapts to datasets of different scales, time windows,
and transaction profiles.

Usage
-----
::

    from explainability import build_dataset_profile, generate_all_explanations

    profile = build_dataset_profile(features_df)
    enriched_df = generate_all_explanations(
        df, features_df, predictions, deviation_scores, profile,
    )
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Features to profile ─────────────────────────────────────────────
# Each entry maps a feature name to a human-readable label and the
# direction of "suspiciousness" (higher = more suspicious by default).
PROFILED_FEATURES = {
    "fan_in": {
        "label": "Input address count",
        "direction": "high",
        "heuristic": "mass_consolidation",
    },
    "fan_out": {
        "label": "Output address count",
        "direction": "high",
        "heuristic": "rapid_dispersal",
    },
    "fan_ratio": {
        "label": "Fan-in / fan-out ratio",
        "direction": "low",       # low ratio = dispersal
        "heuristic": "fan_ratio_anomaly",
    },
    "peel_chain_disparity": {
        "label": "Peel-pattern disparity",
        "direction": "high",
        "heuristic": "peel_pattern_candidate",
    },
    "fee_rate_urgency": {
        "label": "Fee rate urgency",
        "direction": "high",
        "heuristic": "fee_spike_urgency",
    },
    "value_zscore_abs": {
        "label": "Transaction value |Z-score|",
        "direction": "high",
        "heuristic": "volume_outlier",
    },
    "entity_ip_diversity": {
        "label": "Entity IP diversity",
        "direction": "high",
        "heuristic": "ip_anonymization",
    },
    "ip_entity_diversity": {
        "label": "IP entity diversity",
        "direction": "high",
        "heuristic": "shared_infrastructure",
    },
    "port_risk_combined": {
        "label": "Port risk score",
        "direction": "high",
        "heuristic": "proxy_tunneling",
    },
    "peel_chain_length": {
        "label": "Multi-hop peel chain length",
        "direction": "high",
        "heuristic": "multihop_peel_chain",
    },
    "fund_diminishment_ratio": {
        "label": "Fund diminishment ratio",
        "direction": "high",
        "heuristic": "progressive_peeling",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DATASET PROFILE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureStats:
    """Per-feature distribution statistics."""
    median: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0


@dataclass
class DatasetProfile:
    """
    Aggregate statistics for all profiled features.

    Built once from the full ingested dataset via
    :func:`build_dataset_profile`.
    """
    stats: Dict[str, FeatureStats] = field(default_factory=dict)
    series_cache: Dict[str, np.ndarray] = field(
        default_factory=dict, repr=False,
    )
    total_records: int = 0


def build_dataset_profile(features_df: pd.DataFrame) -> DatasetProfile:
    """
    Compute per-feature distribution statistics across the full dataset.

    Parameters
    ----------
    features_df : pd.DataFrame
        The engineered feature matrix (output of ``engineer_features()``).

    Returns
    -------
    DatasetProfile
        Contains median, P75, P95, P99, mean, std, min, max for each
        profiled feature that exists in the DataFrame.
    """
    profile = DatasetProfile(total_records=len(features_df))

    for feat_name in PROFILED_FEATURES:
        if feat_name not in features_df.columns:
            continue

        col = features_df[feat_name].dropna()
        if len(col) == 0:
            continue

        profile.stats[feat_name] = FeatureStats(
            median=float(col.median()),
            mean=float(col.mean()),
            std=float(col.std()) if len(col) > 1 else 0.0,
            p75=float(col.quantile(0.75)),
            p95=float(col.quantile(0.95)),
            p99=float(col.quantile(0.99)),
            min=float(col.min()),
            max=float(col.max()),
        )
        # Cache sorted values for fast percentile lookups
        profile.series_cache[feat_name] = np.sort(col.values)

    return profile


# ═══════════════════════════════════════════════════════════════════════════
# PERCENTILE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_percentile_rank(
    value: float,
    sorted_values: np.ndarray,
) -> float:
    """
    Compute the percentile rank of *value* within a sorted array.

    Uses the "weak" method: percentage of values that are strictly
    less than or equal to *value*.

    Parameters
    ----------
    value : float
        The observed value to rank.
    sorted_values : np.ndarray
        Pre-sorted array of all values in the dataset for this feature.

    Returns
    -------
    float
        Percentile rank in [0.0, 100.0], rounded to 1 decimal.
    """
    if len(sorted_values) == 0:
        return 0.0
    count_le = np.searchsorted(sorted_values, value, side="right")
    return round((count_le / len(sorted_values)) * 100.0, 1)


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURED TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════

def generate_telemetry(
    row_features: pd.Series,
    profile: DatasetProfile,
    percentile_threshold: float = 95.0,
) -> List[Dict[str, Any]]:
    """
    Generate structured telemetry dicts for a single transaction.

    For each profiled feature, computes the percentile rank of the
    observed value against the dataset distribution.  Features that
    exceed the ``percentile_threshold`` are flagged.

    Parameters
    ----------
    row_features : pd.Series
        Feature values for a single transaction.
    profile : DatasetProfile
        Pre-computed dataset-level statistics.
    percentile_threshold : float
        Minimum percentile rank to consider a feature noteworthy
        (default: 95.0 = 95th percentile).

    Returns
    -------
    list[dict]
        One dict per flagged feature, containing:

        - ``feature_name`` — canonical feature name
        - ``label`` — human-readable feature label
        - ``observed_value`` — the transaction's value
        - ``dataset_median`` — median of the dataset
        - ``dataset_p95`` — 95th percentile of the dataset
        - ``percentile_rank`` — where this value falls (0–100)
        - ``heuristic_flags`` — list of heuristic tags
        - ``audit_reason`` — human-readable explanation string
    """
    telemetry: List[Dict[str, Any]] = []

    for feat_name, meta in PROFILED_FEATURES.items():
        if feat_name not in profile.stats:
            continue

        observed = row_features.get(feat_name)
        if observed is None or (isinstance(observed, float) and np.isnan(observed)):
            continue

        observed = float(observed)
        stats = profile.stats[feat_name]
        sorted_vals = profile.series_cache.get(feat_name, np.array([]))
        pct_rank = compute_percentile_rank(observed, sorted_vals)

        # Determine if this feature is noteworthy
        direction = meta["direction"]
        is_flagged = False

        if direction == "high" and pct_rank >= percentile_threshold:
            is_flagged = True
        elif direction == "low" and pct_rank <= (100.0 - percentile_threshold):
            is_flagged = True

        # Special cases: binary/categorical flags always report if non-zero
        if feat_name == "peel_chain_disparity" and observed > 0:
            is_flagged = True
        if feat_name == "peel_chain_length" and observed >= 2:
            is_flagged = True

        if not is_flagged:
            continue

        # Build audit reason
        audit_reason = _build_audit_reason(
            feat_name, meta, observed, stats, pct_rank,
        )

        telemetry.append({
            "feature_name": feat_name,
            "label": meta["label"],
            "observed_value": round(observed, 6),
            "dataset_median": round(stats.median, 6),
            "dataset_p95": round(stats.p95, 6),
            "percentile_rank": pct_rank,
            "heuristic_flags": [meta["heuristic"]],
            "audit_reason": audit_reason,
        })

    return telemetry


def _build_audit_reason(
    feat_name: str,
    meta: dict,
    observed: float,
    stats: FeatureStats,
    pct_rank: float,
) -> str:
    """Build a human-readable audit reason for a flagged feature."""

    label = meta["label"]
    obs_str = _format_value(observed, feat_name)
    med_str = _format_value(stats.median, feat_name)
    p95_str = _format_value(stats.p95, feat_name)

    # Core percentile statement
    reason = (
        f"{label} of {obs_str} places this transaction in the "
        f"{pct_rank}th percentile; dataset median is {med_str}"
    )

    # Add contextual interpretation
    if feat_name == "fan_out" and observed > stats.p95:
        reason += " — rapid 1-to-many dispersal pattern"
    elif feat_name == "fan_in" and observed > stats.p95:
        reason += " — mass input consolidation pattern"
    elif feat_name == "fee_rate_urgency" and observed > stats.p95:
        reason += " — miner-priority overpayment suggesting time-sensitive hop"
    elif feat_name == "peel_chain_disparity" and observed > 0:
        reason += " — single-split output asymmetry (peel-pattern candidate)"
    elif feat_name == "peel_chain_length" and observed >= 2:
        reason += f" — multi-hop peel chain trajectory ({int(observed)} linked hops)"
    elif feat_name == "entity_ip_diversity" and observed > stats.p95:
        reason += " — entity broadcasts from anomalously many distinct IPs"
    elif feat_name == "port_risk_combined" and observed > stats.p95:
        reason += " — possible proxy/Tor/tunneling infrastructure"
    elif feat_name == "value_zscore_abs" and observed > stats.p95:
        reason += " — statistically anomalous transaction volume"

    return reason


def _format_value(value: float, feat_name: str) -> str:
    """Format a numeric value for human display."""
    if feat_name in ("fan_in", "fan_out", "peel_chain_length"):
        return str(int(value))
    elif feat_name in ("fee_rate_urgency", "peel_chain_disparity"):
        return f"{value:.6f}"
    elif feat_name in ("percentile_rank",):
        return f"{value:.1f}"
    else:
        return f"{value:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# EXPLANATION GENERATOR (Percentile-Based)
# ═══════════════════════════════════════════════════════════════════════════

def generate_explanation(
    row_features: pd.Series,
    deviation_score: float,
    original_row: pd.Series,
    profile: DatasetProfile,
) -> str:
    """
    Generate a human-readable explanation for a single transaction
    using dataset-relative percentile profiling.

    Unlike the legacy ``ml_engine.generate_explanation()`` which used
    hardcoded thresholds (e.g., ``fan_out > 5``), this function
    derives all trigger conditions from the **95th percentile** of
    the actual ingested dataset distribution.

    Parameters
    ----------
    row_features : pd.Series
        Feature values for this transaction.
    deviation_score : float
        Anomaly Deviation Score (0–100) from
        :func:`anomaly_engine.compute_anomaly_deviation_scores`.
    original_row : pd.Series
        The original transaction row (for display fields like
        ``src_ip``, ``src_port``, ``total_amount_btc``).
    profile : DatasetProfile
        Pre-computed dataset-level statistics.

    Returns
    -------
    str
        Human-readable explanation string.
    """
    if deviation_score < 50.0:
        return "Normal network traffic behavior."

    telemetry = generate_telemetry(row_features, profile)

    if not telemetry:
        return (
            "Anomalous deviation from baseline transaction distribution "
            "(structural graph relationships flagged by unsupervised model)."
        )

    reasons = [entry["audit_reason"] for entry in telemetry]

    # Check for micro-transaction (binary feature, not percentile-based)
    if row_features.get("is_micro_tx", 0) == 1:
        amount = original_row.get("total_amount_btc", 0)
        reasons.append(
            f"Micro-transaction detected ({amount:.4f} BTC)"
        )

    return "Anomaly indicators: " + "; ".join(reasons) + "."


def generate_all_explanations(
    df: pd.DataFrame,
    features: pd.DataFrame,
    predictions: np.ndarray,
    deviation_scores: np.ndarray,
    profile: DatasetProfile,
) -> pd.DataFrame:
    """
    Batch-generate explanations and structured telemetry for all
    transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Original transaction DataFrame.
    features : pd.DataFrame
        Engineered feature matrix.
    predictions : np.ndarray
        IsolationForest predictions (-1 = anomaly, 1 = normal).
    deviation_scores : np.ndarray
        Output from :func:`anomaly_engine.compute_anomaly_deviation_scores`.
    profile : DatasetProfile
        Pre-computed dataset-level statistics from
        :func:`build_dataset_profile`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with added columns:

        - ``is_anomaly`` — boolean anomaly flag
        - ``risk_score`` — Anomaly Deviation Score (backward-compat name)
        - ``anomaly_deviation_score`` — same value, canonical name
        - ``explanation`` — human-readable explanation string
        - ``telemetry`` — JSON string of structured telemetry dicts
    """
    result = df.copy()
    result["is_anomaly"] = (predictions == -1)
    result["risk_score"] = deviation_scores             # backward compat
    result["anomaly_deviation_score"] = deviation_scores # canonical name

    explanations = []
    telemetry_col = []

    for idx in range(len(result)):
        row = result.iloc[idx]
        if row["is_anomaly"]:
            feat_row = features.iloc[idx]
            score = float(row["risk_score"])

            exp = generate_explanation(feat_row, score, row, profile)
            explanations.append(exp)

            telem = generate_telemetry(feat_row, profile)
            telemetry_col.append(json.dumps(telem, default=str))
        else:
            explanations.append("Normal.")
            telemetry_col.append(None)

    result["explanation"] = explanations
    result["telemetry"] = telemetry_col
    return result
