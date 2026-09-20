# MITHYA Model Card: Graph-Aware IsolationForest

## Model Details
- **Architecture**: `sklearn.ensemble.IsolationForest`
- **Version**: 1.0 (Smart India Hackathon 2026 Release)
- **Hyperparameters**: 
  - `n_estimators`: 200
  - `contamination`: 0.05 (Default, user adjustable via UI)
  - `random_state`: 42

## Intended Use
- **Primary Use Case**: Unsupervised anomaly detection for tracing illicit financial flows, identifying crypto-mixer interactions, and unraveling multi-hop peel chains on Bitcoin networks.
- **Out of Scope**: Not intended for direct probabilistic guilt assertion. Scores represent statistical deviation from baseline distributions, not calibrated legal probability.

## Factors
- **Demographic/Geographic**: Model operates on IP/Port and blockchain address data; geographic location (`geo_country`) is contextual but not a primary feature for the IsolationForest algorithm.

## Metrics
- **Performance**: Capable of processing 1,500+ transactions under 500ms on standard CPUs.
- **Explainability**: Percentile-based feature profiling mapping top deviations to `audit_reasons`.

## Features Engineered (18 core metrics)
- `entity_ip_diversity`, `ip_entity_diversity`
- `entity_tx_freq`, `src_ip_freq`
- `fee_rate_urgency`, `value_zscore`, `value_zscore_abs`
- `peel_chain_disparity`, `fan_in`, `fan_out`, `fan_ratio`
- `port_risk_combined`
- Note: Mixers are pre-emptively bypassed before IF training to preserve their inherent anonymity sets from skewing the tree structures.

## Limitations
- Graph analysis requires `NetworkX` which currently scales $O(V+E)$. For datasets $>10^6$ nodes, this architecture must be migrated to a distributed graph engine.
