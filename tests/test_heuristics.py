import os
import sys
import pandas as pd
import networkx as nx

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml_engine import engineer_features, score_port_risk


def test_isolation_forest_novelty():
    """Confirm novelty tracking (IsolationForest graph setup) does not pre-populate outputs."""
    # Build a dummy dataset
    df = pd.DataFrame({
        "txid": ["tx1", "tx2"],
        "src_ip": ["1.1.1.1", "2.2.2.2"],
        "input_addresses": ["addr1", "addr2"],
        "output_addresses": ["addr3|addr4", "addr5|addr6"],
        "input_amounts": ["1.0", "2.0"],
        "output_amounts": ["0.1|0.8", "0.2|1.7"],
        "fee": [0.1, 0.1],
        "src_port": [8333, 8333],
        "dst_port": [8333, 8333],
        "script_type": ["P2PKH", "P2PKH"],
        "timestamp": ["2026-10-04 12:00:00", "2026-10-04 12:01:00"],
        "geo_country": ["US", "UK"]
    })
    _, features = engineer_features(df)
    assert "anomaly_score" not in features.columns
    assert "is_anomaly" not in features.columns


def test_port_risk_scoring():
    """Checks Tor vs standard ports risk scoring."""
    assert score_port_risk(9050, 8333) == 1.0  # Tor default port -> high risk
    assert score_port_risk(9150, 8333) == 1.0  # Tor browser port -> high risk
    assert score_port_risk(8333, 8333) == 0.0  # Standard Bitcoin ports -> low risk
    assert score_port_risk(443, 80) == 0.0     # Standard Web ports -> low risk
    assert score_port_risk(6667, 8333) == 0.6  # IRC port -> elevated risk
