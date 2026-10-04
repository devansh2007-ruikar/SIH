"""
tests/test_clustering.py -- Tests for behavioural clustering and graph communities
==================================================================================

Validates:
1. entity_profile: one row per entity_id with all 10 required behavioral metrics.
2. behaviour_clusters: StandardScaler, HDBSCAN(min_cluster_size=10), label mapping,
   and 2D PCA coordinates.
3. graph_communities: Louvain community detection on undirected graph of entity,
   src_ip, and txid nodes.
4. Mapping of behaviour_cluster and graph_community to transaction dataframe.
5. Outlier explanation appending: 'Entity behaves unlike every known group (HDBSCAN outlier)'.
"""

import pandas as pd
import numpy as np
import pytest

import clustering


@pytest.fixture
def sample_tx_data():
    """Create a controlled sample dataset with multiple entities and IPs."""
    records = []
    # Entity 1: Normal repetitive traffic (Daytime)
    for i in range(12):
        records.append({
            "txid": f"tx_e1_{i}",
            "entity_id": "Entity_Alpha",
            "timestamp": "2026-03-15T14:30:00",
            "src_ip": "192.168.1.10",
            "amount_btc": 1.5,
            "fee": 0.0001,
            "fan_out": 2,
            "fan_in": 1,
            "fee_rate_urgency": 1.0,
            "port_risk_combined": 0.0,
            "risk_score": 10.0,
            "is_anomaly": False,
            "detected_type": "Normal_P2P",
            "explanation": "Normal traffic.",
        })

    # Entity 2: Nighttime, high fee urgency, mixer pattern
    for i in range(12):
        records.append({
            "txid": f"tx_e2_{i}",
            "entity_id": "Entity_Beta",
            "timestamp": "2026-03-15T02:15:00",  # Night
            "src_ip": "10.0.0.5",
            "amount_btc": 25.0,
            "fee": 0.05,
            "fan_out": 8,
            "fan_in": 8,
            "fee_rate_urgency": 9.5,
            "port_risk_combined": 80.0,
            "risk_score": 85.0,
            "is_anomaly": True,
            "detected_type": "CoinJoin_Mixer",
            "explanation": "Mixer transaction.",
        })

    # Entity 3: Diverse IPs, peel chain pattern
    for i in range(12):
        records.append({
            "txid": f"tx_e3_{i}",
            "entity_id": "Entity_Gamma",
            "timestamp": "2026-03-15T03:00:00",  # Night
            "src_ip": f"172.16.0.{i % 4}",
            "amount_btc": 50.0,
            "fee": 0.001,
            "fan_out": 2,
            "fan_in": 1,
            "fee_rate_urgency": 2.0,
            "port_risk_combined": 20.0,
            "risk_score": 75.0,
            "is_anomaly": True,
            "detected_type": "Peel_Chain",
            "explanation": "Peel chain layering.",
        })

    return pd.DataFrame(records)


def test_entity_profile(sample_tx_data):
    features = sample_tx_data[["fan_out", "fan_in", "fee_rate_urgency", "port_risk_combined"]]
    profile = clustering.entity_profile(sample_tx_data, features)

    expected_cols = [
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
        "entity_id",
    ]
    for col in expected_cols:
        assert col in profile.columns, f"Missing column {col} in entity profile"

    assert len(profile) == 3
    assert set(profile["entity_id"]) == {"Entity_Alpha", "Entity_Beta", "Entity_Gamma"}

    # Entity_Alpha has 0.0 night_share (hour 14), Entity_Beta has 1.0 (hour 2)
    alpha_prof = profile[profile["entity_id"] == "Entity_Alpha"].iloc[0]
    beta_prof = profile[profile["entity_id"] == "Entity_Beta"].iloc[0]
    assert alpha_prof["night_share"] == 0.0
    assert beta_prof["night_share"] == 1.0
    assert alpha_prof["tx_count"] == 12


def test_behaviour_clusters(sample_tx_data):
    features = sample_tx_data[["fan_out", "fan_in", "fee_rate_urgency", "port_risk_combined"]]
    profile = clustering.entity_profile(sample_tx_data, features)
    clustered = clustering.behaviour_clusters(profile)

    assert "behaviour_cluster" in clustered.columns
    assert "pca_x" in clustered.columns
    assert "pca_y" in clustered.columns

    for c in clustered["behaviour_cluster"]:
        assert c == "Behavioural outlier" or c.startswith("Cluster ")


def test_graph_communities(sample_tx_data):
    comm_map = clustering.graph_communities(sample_tx_data)
    assert isinstance(comm_map, dict)
    assert "Entity_Alpha" in comm_map
    assert "Entity_Beta" in comm_map
    assert "Entity_Gamma" in comm_map
    for comm in comm_map.values():
        assert comm.startswith("Community ")


def test_apply_clustering_to_transactions(sample_tx_data):
    df, prof = clustering.apply_clustering_to_transactions(sample_tx_data.copy())
    assert "behaviour_cluster" in df.columns
    assert "graph_community" in df.columns
    assert len(df) == len(sample_tx_data)

    # Check outlier explanation appending rule
    outlier_df = df[df["behaviour_cluster"] == "Behavioural outlier"]
    for _, row in outlier_df.iterrows():
        assert "Entity behaves unlike every known group (HDBSCAN outlier)" in row["explanation"]


def test_summary_tables(sample_tx_data):
    df, prof = clustering.apply_clustering_to_transactions(sample_tx_data.copy())
    cluster_tbl = clustering.build_cluster_summary_table(df, prof)
    comm_tbl = clustering.build_community_summary_table(df, prof)

    assert "Cluster" in cluster_tbl.columns
    assert "Entities" in cluster_tbl.columns
    assert "Avg Risk" in cluster_tbl.columns
    assert "Dominant Pattern" in cluster_tbl.columns

    assert "Community" in comm_tbl.columns
    assert "Number of IPs" in comm_tbl.columns
