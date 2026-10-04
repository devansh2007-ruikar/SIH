"""
clustering.py -- Behavioural Clustering & Graph Communities
============================================================

Entity-level profiling, HDBSCAN behavioural clustering, PCA projection,
and NetworkX Louvain graph community detection for MITHYA.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _parse_amounts(val: Any) -> List[float]:
    """Parse pipe-separated or scalar amounts into floats."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, (list, tuple)):
        res = []
        for x in val:
            try:
                if pd.notna(x):
                    res.append(float(x))
            except (ValueError, TypeError):
                pass
        return res
    if isinstance(val, (int, float)):
        return [float(val)]
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    out = []
    for p in s.split("|"):
        p = p.strip()
        if p:
            try:
                out.append(float(p))
            except (ValueError, TypeError):
                pass
    return out


def entity_profile(
    df: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build an entity-level behavioural profile: one row per entity_id.

    Columns produced:
      - tx_count: transaction count
      - total_btc: total BTC transferred
      - mean_btc: average transaction value in BTC
      - mean_fan_out: average fan-out degree
      - mean_fan_in: average fan-in degree
      - mean_fee_urgency: average fee urgency
      - distinct_src_ips: unique source IP count
      - mean_port_risk: average port risk
      - night_share: fraction of transactions between 00:00 and 05:00
      - mean_risk_score: average forensic risk score
      - entity_id: entity identifier
    """
    if "entity_id" not in df.columns:
        df = df.copy()
        if "input_addresses" in df.columns:
            df["entity_id"] = df["input_addresses"].apply(
                lambda x: str(x).split("|")[0].strip() if pd.notna(x) and str(x).strip() else "Unknown"
            )
        else:
            df["entity_id"] = [f"Entity_{i}" for i in range(len(df))]

    # 1. Transaction count
    tx_count = df.groupby("entity_id", sort=False).size()
    entity_ids = tx_count.index

    # 2. Amounts (total_btc, mean_btc)
    if "amount_btc" in df.columns:
        btc_series = pd.to_numeric(df["amount_btc"], errors="coerce").fillna(0.0)
    elif "input_amounts" in df.columns:
        btc_series = df["input_amounts"].apply(
            lambda v: sum(_parse_amounts(v)) if pd.notna(v) else 0.0
        )
    elif "output_amounts" in df.columns:
        btc_series = df["output_amounts"].apply(
            lambda v: sum(_parse_amounts(v)) if pd.notna(v) else 0.0
        )
    else:
        btc_series = pd.Series(0.0, index=df.index)

    total_btc = btc_series.groupby(df["entity_id"]).sum()
    mean_btc = btc_series.groupby(df["entity_id"]).mean()

    # 3. Fan-out
    if features is not None and "fan_out" in features.columns:
        fan_out_series = pd.to_numeric(features["fan_out"], errors="coerce").fillna(1.0)
    elif "fan_out" in df.columns:
        fan_out_series = pd.to_numeric(df["fan_out"], errors="coerce").fillna(1.0)
    elif "output_addresses" in df.columns:
        fan_out_series = df["output_addresses"].apply(
            lambda v: len(str(v).split("|")) if pd.notna(v) and str(v).strip() else 1.0
        )
    else:
        fan_out_series = pd.Series(1.0, index=df.index)
    mean_fan_out = fan_out_series.groupby(df["entity_id"]).mean()

    # 4. Fan-in
    if features is not None and "fan_in" in features.columns:
        fan_in_series = pd.to_numeric(features["fan_in"], errors="coerce").fillna(1.0)
    elif "fan_in" in df.columns:
        fan_in_series = pd.to_numeric(df["fan_in"], errors="coerce").fillna(1.0)
    elif "input_addresses" in df.columns:
        fan_in_series = df["input_addresses"].apply(
            lambda v: len(str(v).split("|")) if pd.notna(v) and str(v).strip() else 1.0
        )
    else:
        fan_in_series = pd.Series(1.0, index=df.index)
    mean_fan_in = fan_in_series.groupby(df["entity_id"]).mean()

    # 5. Fee urgency
    if features is not None and "fee_rate_urgency" in features.columns:
        urgency_series = pd.to_numeric(features["fee_rate_urgency"], errors="coerce").fillna(0.0)
    elif "fee_rate_urgency" in df.columns:
        urgency_series = pd.to_numeric(df["fee_rate_urgency"], errors="coerce").fillna(0.0)
    elif "fee" in df.columns:
        urgency_series = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)
    else:
        urgency_series = pd.Series(0.0, index=df.index)
    mean_fee_urgency = urgency_series.groupby(df["entity_id"]).mean()

    # 6. Distinct IPs
    if "src_ip" in df.columns:
        distinct_src_ips = df.groupby("entity_id")["src_ip"].nunique()
    else:
        distinct_src_ips = pd.Series(1, index=entity_ids)

    # 7. Port risk
    if features is not None and "port_risk_combined" in features.columns:
        port_series = pd.to_numeric(features["port_risk_combined"], errors="coerce").fillna(0.0)
    elif "port_risk_combined" in df.columns:
        port_series = pd.to_numeric(df["port_risk_combined"], errors="coerce").fillna(0.0)
    elif "score_network" in df.columns:
        port_series = pd.to_numeric(df["score_network"], errors="coerce").fillna(0.0)
    else:
        port_series = pd.Series(0.0, index=df.index)
    mean_port_risk = port_series.groupby(df["entity_id"]).mean()

    # 8. Night share (transactions between 00:00 and 05:00)
    if "timestamp" in df.columns:
        try:
            dt = pd.to_datetime(df["timestamp"], errors="coerce")
            night_bool = ((dt.dt.hour >= 0) & (dt.dt.hour < 5)).astype(float)
        except Exception:
            night_bool = pd.Series(0.0, index=df.index)
    else:
        night_bool = pd.Series(0.0, index=df.index)
    night_share = night_bool.groupby(df["entity_id"]).mean()

    # 9. Risk score
    if "risk_score" in df.columns:
        risk_series = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
    elif "is_anomaly" in df.columns:
        risk_series = df["is_anomaly"].astype(float) * 100.0
    else:
        risk_series = pd.Series(0.0, index=df.index)
    mean_risk_score = risk_series.groupby(df["entity_id"]).mean()

    # Assemble profile DataFrame
    profile = pd.DataFrame(
        {
            "tx_count": tx_count,
            "total_btc": total_btc.reindex(entity_ids).fillna(0.0),
            "mean_btc": mean_btc.reindex(entity_ids).fillna(0.0),
            "mean_fan_out": mean_fan_out.reindex(entity_ids).fillna(1.0),
            "mean_fan_in": mean_fan_in.reindex(entity_ids).fillna(1.0),
            "mean_fee_urgency": mean_fee_urgency.reindex(entity_ids).fillna(0.0),
            "distinct_src_ips": distinct_src_ips.reindex(entity_ids).fillna(1),
            "mean_port_risk": mean_port_risk.reindex(entity_ids).fillna(0.0),
            "night_share": night_share.reindex(entity_ids).fillna(0.0),
            "mean_risk_score": mean_risk_score.reindex(entity_ids).fillna(0.0),
        },
        index=entity_ids,
    )
    profile["entity_id"] = profile.index.astype(str)
    return profile


def behaviour_clusters(profile: pd.DataFrame) -> pd.DataFrame:
    """
    StandardScaler + HDBSCAN(min_cluster_size=10).
    Label -1 = "Behavioural outlier", others = "Cluster {n}".
    Also computes 2D PCA projection coordinates (pca_x, pca_y).

    Returns
    -------
    pd.DataFrame
        profile DataFrame with new columns: behaviour_cluster, pca_x, pca_y.
    """
    profile_out = profile.copy()
    feature_cols = [
        "tx_count",
        "total_btc",
        "mean_btc",
        "mean_fan_out",
        "mean_fan_in",
        "mean_fee_urgency",
        "distinct_src_ips",
        "mean_port_risk",
        "night_share",
        "mean_risk_score",
    ]

    for c in feature_cols:
        if c not in profile_out.columns:
            profile_out[c] = 0.0

    X = profile_out[feature_cols].fillna(0.0).values
    n_samples = len(X)

    if n_samples == 0:
        profile_out["behaviour_cluster"] = []
        profile_out["pca_x"] = []
        profile_out["pca_y"] = []
        return profile_out

    # 1. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. HDBSCAN clustering (min_cluster_size=10, or smaller if dataset < 10)
    min_size = min(10, max(2, n_samples))
    try:
        clusterer = HDBSCAN(min_cluster_size=min_size, copy=True)
        labels = clusterer.fit_predict(X_scaled)
    except Exception:
        labels = np.zeros(n_samples, dtype=int)

    cluster_labels = [
        "Behavioural outlier" if lbl == -1 else f"Cluster {lbl}"
        for lbl in labels
    ]
    profile_out["behaviour_cluster"] = cluster_labels

    # 3. 2D PCA for plotting
    n_pca_components = min(2, X_scaled.shape[1], max(1, n_samples))
    try:
        pca = PCA(n_components=n_pca_components, random_state=42)
        coords = pca.fit_transform(X_scaled)
        if coords.shape[1] >= 2:
            profile_out["pca_x"] = coords[:, 0]
            profile_out["pca_y"] = coords[:, 1]
        elif coords.shape[1] == 1:
            profile_out["pca_x"] = coords[:, 0]
            profile_out["pca_y"] = 0.0
        else:
            profile_out["pca_x"] = 0.0
            profile_out["pca_y"] = 0.0
    except Exception:
        profile_out["pca_x"] = 0.0
        profile_out["pca_y"] = 0.0

    return profile_out


def graph_communities(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build an undirected NetworkX graph of entity, src_ip, and txid nodes,
    then apply nx.community.louvain_communities(G, seed=42).
    Maps each entity to "Community {n}".

    Returns
    -------
    Dict[str, str]
        Mapping of entity_id -> "Community {n}".
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        ent = row.get("entity_id")
        txid = row.get("txid")
        ip = row.get("src_ip")

        if pd.notna(ent) and str(ent).strip():
            e_node = f"entity:{ent}"
            G.add_node(e_node, node_type="entity", entity_id=str(ent))

            if pd.notna(txid) and str(txid).strip():
                t_node = f"tx:{txid}"
                G.add_node(t_node, node_type="tx")
                G.add_edge(e_node, t_node)

            if pd.notna(ip) and str(ip).strip():
                i_node = f"ip:{ip}"
                G.add_node(i_node, node_type="ip")
                if pd.notna(txid) and str(txid).strip():
                    G.add_edge(i_node, f"tx:{txid}")
                else:
                    G.add_edge(e_node, i_node)

    if len(G.nodes) == 0:
        return {}

    try:
        communities = nx.community.louvain_communities(G, seed=42)
        # Sort communities descending by size
        communities = sorted(communities, key=len, reverse=True)
    except Exception:
        communities = [set(G.nodes)]

    entity_community_map: Dict[str, str] = {}
    for comm_idx, comm in enumerate(communities):
        comm_name = f"Community {comm_idx}"
        for node in comm:
            if str(node).startswith("entity:"):
                actual_ent = str(node)[len("entity:"):]
                entity_community_map[actual_ent] = comm_name

    # Ensure all entities present in df have a mapping
    if "entity_id" in df.columns:
        for ent in df["entity_id"].dropna().unique():
            s_ent = str(ent)
            if s_ent not in entity_community_map:
                entity_community_map[s_ent] = "Community 0"

    return entity_community_map


def apply_clustering_to_transactions(
    df: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run entity profiling, behavioural HDBSCAN clustering, Louvain community detection,
    and map 'behaviour_cluster' and 'graph_community' back to df via entity_id.
    Also appends the HDBSCAN outlier explanation where applicable.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (enriched_df, profile_df)
    """
    # 1. Profile
    profile = entity_profile(df, features)

    # 2. HDBSCAN & PCA
    profile = behaviour_clusters(profile)

    # 3. Louvain Communities
    comm_map = graph_communities(df)
    profile["graph_community"] = profile["entity_id"].map(comm_map).fillna("Community 0")

    # 4. Map onto transaction dataframe
    cluster_map = dict(zip(profile["entity_id"], profile["behaviour_cluster"]))
    df["behaviour_cluster"] = df["entity_id"].astype(str).map(cluster_map).fillna("Cluster 0")
    df["graph_community"] = df["entity_id"].astype(str).map(comm_map).fillna("Community 0")

    # 5. Explanations: if the entity is a behavioural outlier, append explanation
    outlier_text = "Entity behaves unlike every known group (HDBSCAN outlier)"
    if "explanation" in df.columns:
        outlier_mask = df["behaviour_cluster"] == "Behavioural outlier"
        for idx in df[outlier_mask].index:
            curr_exp = str(df.at[idx, "explanation"]) if pd.notna(df.at[idx, "explanation"]) else ""
            if outlier_text not in curr_exp:
                if curr_exp and curr_exp.strip() not in ("Normal.", "Normal"):
                    if not curr_exp.endswith("."):
                        curr_exp += "."
                    df.at[idx, "explanation"] = f"{curr_exp} {outlier_text}."
                else:
                    df.at[idx, "explanation"] = f"{outlier_text}."

    return df, profile


def build_cluster_summary_table(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary table for behavioural clusters:
    size, avg risk, flagged count, dominant detected_type.
    """
    records = []
    clusters = sorted(df["behaviour_cluster"].dropna().unique().tolist())
    # Sort with Behavioural outlier first or last
    if "Behavioural outlier" in clusters:
        clusters.remove("Behavioural outlier")
        clusters = ["Behavioural outlier"] + clusters

    for c in clusters:
        c_df = df[df["behaviour_cluster"] == c]
        c_prof = profile[profile["behaviour_cluster"] == c] if "behaviour_cluster" in profile.columns else pd.DataFrame()

        n_entities = len(c_prof) if not c_prof.empty else c_df["entity_id"].nunique()
        n_tx = len(c_df)
        avg_risk = c_df["risk_score"].mean() if "risk_score" in c_df.columns and not c_df.empty else 0.0
        flagged_count = int(c_df["is_anomaly"].sum()) if "is_anomaly" in c_df.columns else int((c_df["risk_score"] >= 40).sum())

        if "detected_type" in c_df.columns and not c_df["detected_type"].dropna().empty:
            mode_series = c_df["detected_type"].mode()
            dom_type = mode_series.iloc[0] if not mode_series.empty else "Normal_P2P"
        else:
            dom_type = "Normal_P2P"

        records.append({
            "Cluster": c,
            "Entities": n_entities,
            "Transactions": n_tx,
            "Avg Risk": round(avg_risk, 1),
            "Flagged Count": flagged_count,
            "Dominant Pattern": dom_type,
        })

    return pd.DataFrame(records)


def build_community_summary_table(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary table for graph communities:
    size, avg risk, flagged count, number of IPs.
    """
    records = []
    communities = sorted(df["graph_community"].dropna().unique().tolist())

    for comm in communities:
        comm_df = df[df["graph_community"] == comm]
        comm_prof = profile[profile["graph_community"] == comm] if "graph_community" in profile.columns else pd.DataFrame()

        n_entities = len(comm_prof) if not comm_prof.empty else comm_df["entity_id"].nunique()
        n_tx = len(comm_df)
        avg_risk = comm_df["risk_score"].mean() if "risk_score" in comm_df.columns and not comm_df.empty else 0.0
        flagged_count = int(comm_df["is_anomaly"].sum()) if "is_anomaly" in comm_df.columns else int((comm_df["risk_score"] >= 40).sum())
        n_ips = comm_df["src_ip"].nunique() if "src_ip" in comm_df.columns else 0

        records.append({
            "Community": comm,
            "Entities": n_entities,
            "Transactions": n_tx,
            "Avg Risk": round(avg_risk, 1),
            "Flagged Count": flagged_count,
            "Number of IPs": n_ips,
        })

    # Sort descending by Flagged Count, then Avg Risk
    out_df = pd.DataFrame(records)
    if not out_df.empty:
        out_df = out_df.sort_values(by=["Flagged Count", "Avg Risk"], ascending=[False, False]).reset_index(drop=True)
    return out_df
