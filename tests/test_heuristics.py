import os
import sys
import pandas as pd
import networkx as nx

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from features import extract_network_features
from ml_engine import score_port_risk

def test_isolation_forest_novelty():
    """Confirm novelty tracking (IsolationForest graph setup) does not pre-populate outputs."""
    # Build a dummy dataset
    df = pd.DataFrame({
        "txid": ["tx1", "tx2"],
        "src_ip": ["1.1.1.1", "2.2.2.2"],
        "entity_id": ["E1", "E2"],
        "input_amounts": ["1.0", "2.0"],
        "output_amounts": ["0.9", "1.9"],
        "fee": [0.1, 0.1],
        "geo_country": ["US", "UK"]
    })
    # Features shouldn't have anomaly columns yet
    G = nx.DiGraph()
    features = extract_network_features(df, G)
    assert "anomaly_score" not in features.columns
    assert "is_anomaly" not in features.columns

def test_port_risk_scoring():
    """Checks Tor vs standard ports risk scoring."""
    assert score_port_risk(9050, 8333) == 1.0  # Tor default port -> high risk
    assert score_port_risk(9150, 8333) == 1.0  # Tor browser port -> high risk
    assert score_port_risk(8333, 8333) == 0.0  # Standard Bitcoin ports -> low risk
    assert score_port_risk(443, 80) == 0.0     # Standard Web ports -> low risk
    assert score_port_risk(6667, 8333) == 0.6  # IRC port -> elevated risk
