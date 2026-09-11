# MITHYA — Enterprise-Grade Crypto-Forensics & Triage Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hackathon](https://img.shields.io/badge/SIH-2026-ff69b4.svg)](https://sih.gov.in/)

**MITHYA** is an offline, graph-aware cryptocurrency forensics and triage engine built for the **Smart India Hackathon 2026**. It ingests raw network/blockchain traffic, clusters wallet entities using Common-Input-Ownership rules, and scores transactions for illicit behavior using unsupervised machine learning. Designed for air-gapped forensic environments, it solves the critical challenges of polluted entity graphs from CoinJoins, easily spoofed Geo-IPs, and alert fatigue from institutional traffic.

---

## 1. Problem Statement & Motivation

Bitcoin's pseudonymous design lets criminal actors move, layer, and cash out illicit funds while evading traditional financial surveillance. Law enforcement needs tools that can operate entirely offline while providing AI-grade detection and graph analysis capabilities.

MITHYA addresses the core challenges of crypto-investigations:
1. **The Mixer Loophole:** Standard heuristics blindly group innocent users with criminals when they interact with CoinJoins.
2. **IP/VPN Falsification:** Physical IP mapping is unreliable against state-sponsored actors routing through Tor/proxies.
3. **Analyst Fatigue:** High-volume legal traffic constantly triggers anomaly alerts.

---

## 2. Key Features

| Feature | Description |
|---|---|
| **Mixer/CoinJoin Bypass** | Detects equal-output signatures (`equal_output_ratio=0.50`) and blocks false entity aggregation. |
| **Change-Address Detection** | 5-factor heuristic (script-type match, unrounded remainder, decimal precision, novelty, asymmetry) links change outputs back to sender entity. |
| **Port Risk Telemetry** | Scores network ports (`0.0` to `1.0`) to flag proxies and ephemeral routing, discarding static Geo-IP. |
| **Behavioral ML** | 18 engineered features (e.g., Peel Chain Disparity, Fan-In/Out) fed into an Isolation Forest model. |
| **Institutional Whitelist** | Pre-clears known safe addresses via `institutional_whitelist.csv` to eliminate false positives. |
| **Abstract Ingestion Adapter** | Implements the Adapter pattern (`BaseTransactionAdapter`) ensuring cross-chain compatibility. |
| **Advanced 4-Tab UI** | Streamlit dashboard with Overview, Entity Attribution, PyVis Graph, and Advanced Heuristics Inspector. |
| **Court-Ready Exports** | One-click export of a full forensic summary report detailing ML alerts and heuristic reasoning. |

---

## 3. System Architecture & Data Flow

```text
┌─────────────────────┐       ┌──────────────────────┐       ┌───────────────────────┐
│       STAGE 1       │       │       STAGE 2        │       │       STAGE 3         │
│   DATA INGESTION    │──────▶│     AI/ML ENGINE     │──────▶│      DASHBOARD        │
│                     │       │                      │       │                       │
│ transaction_adapter │       │     ml_engine.py     │       │       app.py          │
│ .py                 │       │                      │       │                       │
│ • BitcoinCSVAdapter │       │ • Mixer bypass       │       │ • Pyvis network graph │
│ • Schema validation │       │ • Change-addr detect │       │ • Metric cards        │
│                     │       │ • Wallet clustering  │       │ • Heuristics Inspector│
│                     │       │ • IsolationForest    │       │ • Data regenerator    │
│                     │       │ • Whitelist filter   │       │                       │
│          ▲          │       │          ▼           │       │                       │
└──────────┼──────────┘       └──────────┼───────────┘       └───────────────────────┘
           │                             │
    [ bitcoin_traffic.csv ]   [ flagged_transactions.csv ]
```

**Data Flow:**
1. `transaction_adapter.py` ingests `bitcoin_traffic.csv` via the `BitcoinCSVAdapter` and normalizes the pipe-delimited addresses.
2. `ml_engine.py` builds a NetworkX graph, filters out mixers, detects change addresses and links them back to sender entities, clusters wallets via Union-Find, engineers 18 features, trains an `IsolationForest`, flags anomalies, applies the `institutional_whitelist.csv`, and returns a 3-tuple `(enriched_df, model, features_df)`.
3. `app.py` runs a Streamlit dashboard that allows CSV upload, invokes the ML pipeline via the polymorphic adapter, and visualizes the results.

---

## 4. Tech Stack & Design Rationale

| Library | Role | Rationale |
|---|---|---|
| **Python** | Core Language | Standard for data science; natively supports offline execution. |
| **Pandas / NumPy** | Data Manipulation | High-performance memory structures for transaction matrices. |
| **Scikit-Learn** | ML Engine | Utilized for `IsolationForest` (unsupervised anomaly detection) and `MinMaxScaler`. |
| **NetworkX** | Graph Clustering | Essential for the Union-Find algorithm underpinning wallet ownership grouping. |
| **Streamlit** | Dashboard UI | Enables rapid frontend development strictly in Python. |
| **PyVis** | Visualization | Renders physics-based (`forceAtlas2Based`) interactive graphs in-browser. |
| **Faker / Requests**| Data Generation | Used exclusively in `generate_data.py` to synthesize realistic P2P traffic. |
| **GeoIP2** | Legacy Telemetry | Depreciated in ML features, but still imported in `generate_data.py` for display metadata. |

---

## 5. Dataset Schema

The `BitcoinCSVAdapter` strictly enforces this schema. Examples drawn from `bitcoin_traffic.csv`:

| Column | Type | Example |
|---|---|---|
| `timestamp` | String | `2026-07-27 12:38:43` |
| `src_ip` | String | `161.247.254.91` |
| `dst_ip` | String | `77.50.222.230` |
| `src_port` | Int64 | `64832` |
| `dst_port` | Int64 | `8334` |
| `txid` | String | `15eb...bd2dd585658` |
| `input_addresses` | String (Pipe-delimited) | `bc1q9nBNB6...` |
| `output_addresses` | String (Pipe-delimited) | `3Sto6RVk2O...` |
| `input_amounts` | Float64 / String | `1.81296791` |
| `output_amounts` | Float64 / String | `1.8122841` |
| `fee` | Float64 | `0.00068381` |
| `script_type` | String | `P2SH` |
| `geo_country` | String (Display metadata) | `US` |

---

## 6. AI/ML Pipeline Deep Dive

### IsolationForest Configuration
*   **Model:** `sklearn.ensemble.IsolationForest`
*   **n_estimators:** 200
*   **Contamination Rate:** Default `0.05` (adjustable via UI).

### Engineered Features (18 total)
The pipeline computes these features in `engineer_features()`:
1. `entity_ip_diversity`: Unique `src_ip` count per entity.
2. `ip_entity_diversity`: Unique `entity_id` count per `src_ip`.
3. `entity_tx_freq`: Frequency encoding of `entity_id`.
4. `src_ip_freq`: Frequency encoding of `src_ip`.
5. `amount_btc_scaled`: MinMax-scaled total transaction volume.
6. `fee_scaled`: MinMax-scaled fee.
7. `is_micro_tx`: Binary flag (total volume < 0.006 BTC).
8. `script_type_freq`: Frequency encoding of `script_type`.
9. `src_port_risk`: Port risk tier (0.0 to 1.0).
10. `dst_port_risk`: Port risk tier (0.0 to 1.0).
11. `port_risk_combined`: `src_port_risk * 0.6 + dst_port_risk * 0.4`.
12. `peel_chain_disparity`: `max_share - min_share` for 2-output TXs.
13. `fan_in`: Count of input addresses.
14. `fan_out`: Count of output addresses.
15. `fan_ratio`: `fan_in / fan_out`.
16. `fee_rate_urgency`: `fee / sum(input_amounts)`.
17. `value_zscore`: Z-Score of total volume.
18. `value_zscore_abs`: Absolute Z-Score of total volume.

---

## 7. Explainable AI / Output Logic

The `generate_explanation()` function constructs human-readable reasons by evaluating the feature vector against specific thresholds:

*   **Port Anomaly (`port_risk_combined >= 0.6`):** `"Anomalous port profile (src:{} / dst:{}) — possible proxy/Tor/tunneling"`
*   **Peel Chain (`peel_chain_disparity > 0`):** `"Peel-chain output structure detected (disparity={}) — classic layering signature"`
*   **Mass Consolidation (`fan_ratio > 5`):** `"Mass consolidation ({} inputs → {} outputs, ratio={})"`
*   **Rapid Dispersal (`fan_ratio < 0.2` and `fan_out > 5`):** `"Rapid dispersal ({} inputs → {} outputs, ratio={})"`
*   **Fee Urgency (`fee_rate_urgency > 0.05`):** `"Fee-rate urgency ({}) — miner-priority overpayment suggesting time-sensitive hop"`
*   **Volume Outlier (`value_zscore_abs > 2.5`):** `"Volume outlier (Z-score={}) — statistically anomalous transaction value"`
*   **IP Anonymization (`entity_ip_diversity > 2`):** `"Entity uses multiple distinct IPs ({} IPs), suggesting anonymization"`
*   **Botnet Relay (`ip_entity_diversity > 2`):** `"IP {} broadcasts for multiple distinct entities, acting as a high-traffic node"`

---

## 8. Dashboard & Visualization Guide

The Streamlit interface (`app.py`) provides a robust 4-tab intelligence architecture:

*   **Dynamic Sidebar:**
    *   File uploaders for `.csv` and custom whitelists.
    *   🎚️ Anomaly Sensitivity Slider: Adjusts the IsolationForest contamination rate.
    *   ⚡ Synthetic Data Generator: Live re-generation of data with custom parameters (`total-records`, `suspicious-ratio`).
    *   🏛️ Active Institutional Whitelist viewer.
*   **Tab 1 - 🎯 Overview & Triage:**
    *   6 Bento Metric Cards displaying Volume, Flagged Threats, Mixers, Peel Chains, etc.
    *   Threat Vector Distribution Pie Chart and Risk Score Histogram.
    *   📥 Court-Ready Report Export.
*   **Tab 2 - 📋 Entity Attribution Viewer:**
    *   Sortable and filterable DataFrame displaying flagged transactions.
    *   Color-coded badges for Risk, Attack Types, and Ports.
*   **Tab 3 - 🕸️ Graph & Clusters:**
    *   PyVis Network Graph with `forceAtlas2Based` physics.
    *   🔍 Threat-Centric Mode Toggle: View only flagged threats or broader network activity.
    *   Color-coded nodes: 🔴 Suspicious, 🔵 Normal, 🟢 Regulated, 🟣 Entity Clusters.
*   **Tab 4 - 🔬 Heuristics Inspector:**
    *   Interactive drill-down panel for selected transactions.
    *   Displays exactly which of the 18 mathematical heuristics triggered for the chosen entity.

---

## 9. Project Structure

```text
SIH-2026-2/
├── generate_data.py            # Generates synthetic P2P traffic dataset (CLI arguments supported)
├── ml_engine.py                # Core ML pipeline, graph logic, and XAI generator
├── transaction_adapter.py      # BaseTransactionAdapter and BitcoinCSVAdapter
├── app.py                      # Streamlit 4-tab frontend UI and PyVis rendering
├── bitcoin_traffic.csv         # Raw synthetic dataset
├── institutional_whitelist.csv # Offline CSV mapped to safe entity addresses
├── requirements.txt            # Python pip dependencies
└── README.md                   # This documentation file
```

---

## 10. Installation & Usage

1. **Clone & Setup Environment:**
   ```bash
   git clone <repo_url>
   cd SIH-2026-2
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data:**
   ```bash
   python generate_data.py --total-records 2000 --suspicious-ratio 0.25
   ```
   *(By default, generates 1,500 records with a 20% suspicious ratio including CoinJoins, Peel Chains, Fan-Outs, and Fee Spikes.)*

3. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *Access via `http://localhost:8501`.*

4. **Run Pipeline Headless (CLI):**
   ```bash
   python ml_engine.py
   ```

---

## 11. SIH Compliance Matrix

| Requirement | Implementation | Status |
|---|---|---|
| Offline Linux Operation | Python, Sklearn, NetworkX | ✅ Completed |
| Ingest bulk CSV metadata | `transaction_adapter.py` | ✅ Completed |
| Entity/transaction graph | `build_entity_graph()` in `ml_engine.py` | ✅ Completed |
| AI anomaly detection | `IsolationForest` on 18 features | ✅ Completed |
| Wallet clustering | Union-Find via `nx.connected_components` | ✅ Completed |
| Change-address detection | `detect_change_address()` 5-factor heuristic | ✅ Completed |
| Explainable alerts | `generate_explanation()` logic | ✅ Completed |
| Interactive UI | `app.py` Streamlit + PyVis | ✅ Completed |
| Actionable Intelligence | Advanced Heuristics Inspector & Exports | ✅ Completed |

---

## 12. Future Roadmap

*   **Ethereum Integration:** Implement an `EthereumAdapter` to support account/nonce-based anomaly tracing.
*   **Real-time Mempool Ingestion:** Shift from CSV batch-processing to subscribing to a live Bitcoin Core RPC node.
*   **Temporal Velocity Features:** Detect burst-sending patterns via sliding-window analysis over transaction timestamps.
