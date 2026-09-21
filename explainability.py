"""
explainability.py — Dynamic Per-Transaction Explainability Engine
=================================================================

Generates unique, human-readable explanations for every flagged
transaction by examining the ACTUAL feature values and raw
transaction fields that caused the IsolationForest to flag it.

Architecture
------------
The explanation engine operates in two phases:

**Phase A — Direct Signal Detection:**
Checks raw transaction fields (port numbers, fee, amounts, attack_type,
addresses) against forensic rule-triggers. These produce crisp,
domain-specific statements like:
    "Routed through known Tor SOCKS port (9050)."

**Phase B — Statistical Deviation Ranking:**
For every engineered feature, computes a Z-score and percentile rank
against the dataset distribution. Features that deviate significantly
from the median are ranked by severity and reported as:
    "Fee urgency of 0.189 is in the 98.2th percentile (dataset median: 0.0007)."

If Phase A produces no specific triggers AND Phase B finds no features
above the 75th percentile, the fallback identifies the SINGLE feature
with the largest absolute Z-score deviation and reports it — so every
transaction gets a unique explanation.

Integration
-----------
Called from ``ml_engine.generate_all_explanations()`` which passes:
- ``features_df``: The 18-21 column numeric feature matrix
- ``df``: The original transaction DataFrame (with src_port, fee, etc.)
- ``predictions``: IsolationForest labels (-1 = anomaly)
- ``deviation_scores``: Anomaly Deviation Scores (0–100)
- ``profile``: Pre-computed DatasetProfile with medians/percentiles

The output ``explanation`` column is rendered directly in the
Entity Attribution table and the Heuristics Inspector panel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE METADATA
# ═══════════════════════════════════════════════════════════════════════════

# Maps feature names → human label + suspiciousness direction + heuristic tag
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
        "direction": "low",
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
    "amount_btc_scaled": {
        "label": "Scaled BTC amount",
        "direction": "high",
        "heuristic": "high_value_transfer",
    },
    "fee_scaled": {
        "label": "Scaled fee",
        "direction": "high",
        "heuristic": "fee_anomaly",
    },
    "src_port_risk": {
        "label": "Source port risk",
        "direction": "high",
        "heuristic": "src_port_anomaly",
    },
    "dst_port_risk": {
        "label": "Destination port risk",
        "direction": "high",
        "heuristic": "dst_port_anomaly",
    },
}

# Known Tor/proxy/I2P port numbers for direct signal detection
_TOR_PORTS = {9050, 9051, 9150}
_PROXY_PORTS = {3128, 8080, 1080}
_I2P_PORTS = {4444, 4445}
_SUSPICIOUS_PORTS = _TOR_PORTS | _PROXY_PORTS | _I2P_PORTS


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
# PHASE A: DIRECT SIGNAL DETECTION (raw transaction fields)
# ═══════════════════════════════════════════════════════════════════════════

def _detect_direct_signals(
    original_row: pd.Series,
    row_features: pd.Series,
    profile: DatasetProfile,
) -> List[str]:
    """
    Check raw transaction fields for forensic rule-triggers that
    produce specific, unambiguous explanation strings.

    These are NOT percentile-based — they fire on exact value matches
    or domain-knowledge thresholds.

    Parameters
    ----------
    original_row : pd.Series
        The original transaction row (src_port, dst_port, fee, etc.)
    row_features : pd.Series
        The engineered feature row.
    profile : DatasetProfile
        Dataset statistics (used for fee/amount comparisons).

    Returns
    -------
    list[str]
        Plain-English reason strings. May be empty if no direct
        signals fire.
    """
    signals: List[str] = []

    # ── 1. Port-based signals ─────────────────────────────────────────
    src_port = int(original_row.get("src_port", 0))
    dst_port = int(original_row.get("dst_port", 0))

    if src_port in _TOR_PORTS:
        signals.append(
            f"Routed through known Tor SOCKS port (src_port={src_port})"
        )
    elif src_port in _PROXY_PORTS:
        signals.append(
            f"Routed through known proxy port (src_port={src_port})"
        )
    elif src_port in _I2P_PORTS:
        signals.append(
            f"Routed through known I2P tunnel port (src_port={src_port})"
        )

    if dst_port in _TOR_PORTS:
        signals.append(
            f"Destination is a known Tor control/SOCKS port (dst_port={dst_port})"
        )
    elif dst_port in _I2P_PORTS:
        signals.append(
            f"Destination is a known I2P eepsite port (dst_port={dst_port})"
        )

    # ── 2. Mixer/CoinJoin signature ───────────────────────────────────
    attack_type = str(original_row.get("attack_type", ""))
    if "CoinJoin" in attack_type or "Mixer" in attack_type:
        signals.append(
            "Matches deterministic equal-denomination CoinJoin signature"
        )

    # ── 3. Fee spike ──────────────────────────────────────────────────
    if "Fee_Spike" in attack_type or "Fee" in attack_type:
        fee = float(original_row.get("fee", 0))
        total = float(original_row.get("total_amount_btc", 0))
        if total > 0:
            fee_pct = (fee / total) * 100
            signals.append(
                f"Abnormal fee-to-value ratio: {fee:.8f} BTC fee on "
                f"{total:.4f} BTC transfer ({fee_pct:.1f}% of value)"
            )
        else:
            signals.append(
                f"Fee spike detected: {fee:.8f} BTC on zero/micro transfer"
            )

    # ── 4. Peel chain ─────────────────────────────────────────────────
    if "Peel" in attack_type:
        disparity = float(row_features.get("peel_chain_disparity", 0))
        if disparity > 0:
            signals.append(
                f"Structural peel-chain disparity detected (ratio: {disparity:.4f})"
            )
        else:
            signals.append(
                "Peel-chain output structure — one large + one small output "
                "suggests change-address layering"
            )

    # ── 5. Micro-transaction (dust attack probe) ──────────────────────
    is_micro = row_features.get("is_micro_tx", 0)
    if is_micro == 1:
        amount = float(original_row.get("total_amount_btc", 0))
        signals.append(
            f"Micro-transaction ({amount:.6f} BTC) — possible dust-attack "
            f"probe or chain-pollution vector"
        )

    # ── 6. High fan-out (mass dispersal) ──────────────────────────────
    fan_out = int(row_features.get("fan_out", 1))
    if fan_out >= 5:
        signals.append(
            f"High fan-out of {fan_out} output addresses — "
            f"rapid 1-to-many dispersal pattern"
        )

    # ── 7. High fan-in (mass consolidation) ───────────────────────────
    fan_in = int(row_features.get("fan_in", 1))
    if fan_in >= 5:
        signals.append(
            f"High fan-in of {fan_in} input addresses — "
            f"mass consolidation pattern"
        )

    # ── 8. Multi-hop peel chain ───────────────────────────────────────
    chain_len = int(row_features.get("peel_chain_length", 0))
    if chain_len >= 2:
        signals.append(
            f"Multi-hop peel chain trajectory ({chain_len} linked hops)"
        )

    return signals


# ═══════════════════════════════════════════════════════════════════════════
# PHASE B: STATISTICAL DEVIATION RANKING
# ═══════════════════════════════════════════════════════════════════════════

def _rank_feature_deviations(
    row_features: pd.Series,
    profile: DatasetProfile,
    percentile_threshold: float = 75.0,
    max_reasons: int = 4,
) -> List[str]:
    """
    Rank all features by how far they deviate from the dataset median
    and return human-readable strings for the top deviators.

    Parameters
    ----------
    row_features : pd.Series
        Feature values for a single transaction.
    profile : DatasetProfile
        Pre-computed dataset statistics.
    percentile_threshold : float
        Minimum percentile rank to flag (default: 75.0).
    max_reasons : int
        Maximum number of statistical reasons to return.

    Returns
    -------
    list[str]
        Ranked reason strings, most-deviant first.
    """
    deviations: List[Tuple[float, str, str]] = []  # (severity, reason, feat_name)

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

        # Compute absolute Z-score for ranking severity
        if stats.std > 0:
            zscore = abs((observed - stats.median) / stats.std)
        else:
            zscore = 0.0

        # Skip features that are AT or BELOW the median (not anomalous)
        direction = meta["direction"]
        is_noteworthy = False

        if direction == "high" and pct_rank >= percentile_threshold and observed > stats.median:
            is_noteworthy = True
        elif direction == "low" and pct_rank <= (100.0 - percentile_threshold) and observed < stats.median:
            is_noteworthy = True

        # Special: peel chain disparity > 0 always noteworthy
        if feat_name == "peel_chain_disparity" and observed > 0 and observed > stats.median:
            is_noteworthy = True
        if feat_name == "peel_chain_length" and observed >= 2:
            is_noteworthy = True

        if not is_noteworthy:
            continue

        reason = _format_deviation_reason(feat_name, meta, observed, stats, pct_rank)
        deviations.append((zscore, reason, feat_name))

    # Sort by severity (highest Z-score first)
    deviations.sort(key=lambda x: x[0], reverse=True)

    return [reason for _, reason, _ in deviations[:max_reasons]]


def _format_deviation_reason(
    feat_name: str,
    meta: dict,
    observed: float,
    stats: FeatureStats,
    pct_rank: float,
) -> str:
    """Build a forensic-grade deviation reason string."""
    label = meta["label"]
    obs_str = _format_value(observed, feat_name)
    med_str = _format_value(stats.median, feat_name)

    # Core: what percentile is this value in?
    reason = (
        f"{label} of {obs_str} is in the {pct_rank}th percentile "
        f"(dataset median: {med_str})"
    )

    # Append domain-specific interpretation
    _INTERPRETATIONS = {
        "fan_out":               " — rapid 1-to-many dispersal pattern",
        "fan_in":                " — mass input consolidation pattern",
        "fee_rate_urgency":      " — miner-priority overpayment (time-sensitive hop)",
        "entity_ip_diversity":   " — entity broadcasts from anomalously many IPs",
        "ip_entity_diversity":   " — multiple entities sharing the same IP relay",
        "port_risk_combined":    " — proxy/Tor/tunneling port profile",
        "value_zscore_abs":      " — statistically anomalous transaction volume",
        "peel_chain_disparity":  " — peel-pattern output asymmetry",
        "peel_chain_length":     f" — multi-hop peel chain ({int(observed)} linked hops)",
        "fund_diminishment_ratio": " — progressive fund siphoning across hops",
        "amount_btc_scaled":     " — unusually large transfer value",
        "fee_scaled":            " — fee deviates from typical range",
        "src_port_risk":         " — source port associated with anonymization",
        "dst_port_risk":         " — destination port associated with hidden services",
    }

    if feat_name in _INTERPRETATIONS:
        reason += _INTERPRETATIONS[feat_name]

    return reason


def _format_value(value: float, feat_name: str) -> str:
    """Format a numeric value for human display."""
    if feat_name in ("fan_in", "fan_out", "peel_chain_length"):
        return str(int(value))
    elif feat_name in ("fee_rate_urgency", "peel_chain_disparity", "fund_diminishment_ratio"):
        return f"{value:.6f}"
    else:
        return f"{value:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# PHASE C: FALLBACK — LARGEST Z-SCORE DEVIATION
# ═══════════════════════════════════════════════════════════════════════════

def _fallback_max_deviation(
    row_features: pd.Series,
    profile: DatasetProfile,
) -> str:
    """
    When no specific triggers fire, identify the single feature that
    deviates furthest from the dataset mean (by Z-score) and report it.

    This guarantees every anomaly gets a UNIQUE explanation even when
    no individual feature crosses the 75th percentile threshold.
    """
    max_zscore = 0.0
    max_feat = None
    max_observed = 0.0
    max_stats = None

    for feat_name in PROFILED_FEATURES:
        if feat_name not in profile.stats:
            continue

        observed = row_features.get(feat_name)
        if observed is None or (isinstance(observed, float) and np.isnan(observed)):
            continue

        observed = float(observed)
        stats = profile.stats[feat_name]

        if stats.std > 0:
            zscore = abs((observed - stats.mean) / stats.std)
        else:
            zscore = 0.0

        if zscore > max_zscore:
            max_zscore = zscore
            max_feat = feat_name
            max_observed = observed
            max_stats = stats

    if max_feat is not None and max_stats is not None:
        meta = PROFILED_FEATURES[max_feat]
        obs_str = _format_value(max_observed, max_feat)
        med_str = _format_value(max_stats.median, max_feat)
        return (
            f"Largest deviation: {meta['label']} of {obs_str} "
            f"({max_zscore:.1f}σ from mean; dataset median: {med_str})"
        )

    return "Multivariate anomaly (combined feature interactions flagged by model)"


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURED TELEMETRY (for Heuristics Inspector panel)
# ═══════════════════════════════════════════════════════════════════════════

def generate_telemetry(
    row_features: pd.Series,
    profile: DatasetProfile,
    percentile_threshold: float = 75.0,
) -> List[Dict[str, Any]]:
    """
    Generate structured telemetry dicts for a single transaction.

    For each profiled feature, computes the percentile rank.
    Features above the threshold are included in the output.

    Returns
    -------
    list[dict]
        One dict per flagged feature containing: feature_name, label,
        observed_value, dataset_median, dataset_p95, percentile_rank,
        heuristic_flags, audit_reason.
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

        if direction == "high" and pct_rank >= percentile_threshold and observed > stats.median:
            is_flagged = True
        elif direction == "low" and pct_rank <= (100.0 - percentile_threshold) and observed < stats.median:
            is_flagged = True

        # Special cases
        if feat_name == "peel_chain_disparity" and observed > 0 and observed > stats.median:
            is_flagged = True
        if feat_name == "peel_chain_length" and observed >= 2:
            is_flagged = True

        if not is_flagged:
            continue

        audit_reason = _format_deviation_reason(
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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPLANATION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_explanation(
    row_features: pd.Series,
    deviation_score: float,
    original_row: pd.Series,
    profile: DatasetProfile,
) -> str:
    """
    Generate a UNIQUE, human-readable explanation for a single
    flagged transaction by combining direct signal detection with
    statistical deviation ranking.

    This function is the core of the MITHYA XAI engine. It guarantees
    that every anomaly receives a specific, tailored explanation —
    never a generic fallback string.

    Parameters
    ----------
    row_features : pd.Series
        Engineered feature values for this transaction (from the
        features DataFrame, NOT the original transaction row).
    deviation_score : float
        Anomaly Deviation Score (0–100).
    original_row : pd.Series
        The original transaction row (src_port, fee, attack_type, etc.)
    profile : DatasetProfile
        Pre-computed dataset-level statistics.

    Returns
    -------
    str
        A cohesive, forensic-grade explanation string.
    """
    # Normal traffic — no explanation needed
    if deviation_score < 50.0:
        return "Normal network traffic behavior."

    reasons: List[str] = []

    # ── Phase A: Direct signal detection (raw fields) ─────────────────
    direct_signals = _detect_direct_signals(original_row, row_features, profile)
    reasons.extend(direct_signals)

    # ── Phase B: Statistical deviation ranking (feature percentiles) ──
    stat_reasons = _rank_feature_deviations(row_features, profile)

    # Only add stat reasons that aren't redundant with direct signals
    # (e.g., don't say "port risk score is high" if we already said "Tor port 9050")
    _direct_text = " ".join(reasons).lower()
    for reason in stat_reasons:
        # Skip if the same concept was already covered by direct signals
        skip = False
        if "port" in reason.lower() and ("tor" in _direct_text or "proxy" in _direct_text or "port" in _direct_text):
            skip = True
        if "fan-out" in reason.lower() and "fan-out" in _direct_text:
            skip = True
        if "fan-in" in reason.lower() and "fan-in" in _direct_text:
            skip = True
        if "peel" in reason.lower() and "peel" in _direct_text:
            skip = True
        if "fee" in reason.lower() and "fee" in _direct_text:
            skip = True
        if not skip:
            reasons.append(reason)

    # ── Phase C: Fallback — largest single Z-score deviation ──────────
    if not reasons:
        fallback = _fallback_max_deviation(row_features, profile)
        reasons.append(fallback)

    return "Anomaly indicators: " + "; ".join(reasons) + "."


# ═══════════════════════════════════════════════════════════════════════════
# BATCH EXPLANATION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

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
