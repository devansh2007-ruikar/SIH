# MITHYA — Enterprise-Grade Crypto-Forensics & Triage Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hackathon](https://img.shields.io/badge/SIH-2026-ff69b4.svg)](https://sih.gov.in/)
[![Security](https://img.shields.io/badge/XML-defusedxml%20XXE--safe-green.svg)](https://pypi.org/project/defusedxml/)
[![Air-Gap](https://img.shields.io/badge/Air--Gap-Verified-brightgreen.svg)](#phase-2b-air-gap-proof--scriptstest_airgapsh)
[![ML Validated](https://img.shields.io/badge/ML-Benchmarked%20Precision%2FRecall%2FF1%2FFPR-blue.svg)](#phase-1-ml-validation--benchmarking--evaluate_modelpy)

**MITHYA** is an offline, graph-aware cryptocurrency forensics and triage engine built for **Smart India Hackathon 2026 — Problem Statement 26146**. It ingests raw network/blockchain traffic, clusters wallet entities using Common-Input-Ownership rules, and scores transactions for illicit behaviour using unsupervised machine learning — fully air-gapped, with no external API or CDN dependencies.

> **Security Hardening Notice (v2.0):** This release adds ML benchmarking, XXE-safe XML parsing, air-gap proof, forensic terminology compliance, cluster confidence labelling, and offline Geo-ASN enrichment. See [Hardening Changelog](#hardening-changelog-v20) for details.

---

## 1. Problem Statement & Motivation

Bitcoin's pseudonymous design lets criminal actors move, layer, and cash out illicit funds while evading traditional financial surveillance. Law enforcement needs tools that can operate entirely **offline** while providing AI-grade detection and graph analysis.

MITHYA addresses the core challenges of crypto-investigations:

1. **The Mixer Loophole** — Standard heuristics blindly group innocent users with criminals when they interact with CoinJoins.
2. **IP/VPN Falsification** — Physical IP mapping is unreliable against state-sponsored actors routing through Tor/proxies.
3. **Analyst Fatigue** — High-volume legal traffic constantly triggers anomaly alerts.
4. **Cluster Over-Confidence** — Common-input-ownership is a heuristic, not a fact; mixers can pollute cluster boundaries.

---

## 2. Key Features

| Feature | Description |
|---|---|
| **Mixer/CoinJoin Bypass** | Detects equal-output signatures (`equal_output_ratio=0.50`) and blocks false entity aggregation. |
| **Change-Address Detection** | 5-factor heuristic (script-type match, unrounded remainder, decimal precision, novelty, asymmetry) links change outputs back to sender entity. |
| **Port Risk Telemetry** | Scores network ports to flag Tor (9050/9150) and I2P (4444) proxies; standard ports carry zero risk. |
| **Behavioral ML** | 18 engineered features fed into `IsolationForest` to generate an objective **Anomaly Deviation Score** and **Investigative Priority Index**. |
| **ML Validation Suite** | `evaluate_model.py` reports Precision, Recall, F1, FPR, latency (us), and peak memory across 3 synthetic dataset categories. |
| **Institutional Whitelist** | Zeroes out risk for specific institutional nodes while preserving anomaly scores of counterparties. |
| **XXE-Safe XML Ingestion** | `defusedxml` replaces stdlib `xml.etree.ElementTree` — prevents billion-laughs, XXE injection, and DTD-based DoS. |
| **Offline Geo-ASN Enrichment** | `geo_asn.py` resolves any IP to `{geo_country, asn, asn_org}` entirely locally — no external API, no .mmdb file needed. |
| **Air-Gap Proof Script** | `scripts/test_airgap.sh` uses Linux `unshare -n` network namespaces to prove 100% offline execution. |
| **Cluster Confidence Labels** | Graph nodes are labelled `High-confidence cluster`, `Mixer-affected cluster`, or `Heuristic cluster` — not treated as absolute facts. |
| **Forensic Disclaimers** | Persistent warning banners on all network-telemetry views: "Not absolute identity attribution (Subject to VPN/NAT/Tor limits)." |
| **Abstract Ingestion Adapter** | `BaseTransactionAdapter` pattern ensures cross-chain compatibility for `.csv`, `.json`, and `.xml`. |
| **Advanced 4-Tab UI** | Streamlit dashboard: Overview, Entity Attribution (+ASN column), PyVis Graph, Analytical Explainability Panel. |
| **Court-Ready Exports** | One-click forensic summary report with all ML alerts and heuristic reasoning. |

---

## 3. System Architecture & Data Flow

```
+---------------------+       +----------------------+       +-----------------------+
|       STAGE 1       |       |       STAGE 2        |       |       STAGE 3         |
|   DATA INGESTION    |------>|     AI/ML ENGINE     |------>|      DASHBOARD        |
|                     |       |                      |       |                       |
| transaction_adapter |       |     ml_engine.py     |       |       app.py          |
| .py (defusedxml)    |       |                      |       |                       |
| * BitcoinCSVAdapter |       | * Mixer bypass       |       | * Pyvis network graph |
| * BitcoinJSONAdapter|       | * Change-addr detect |       | * Cluster labels      |
| * BitcoinXMLAdapter |       | * Wallet clustering  |       | * ASN column          |
|   (XXE-safe)        |       | * IsolationForest    |       | * Forensic disclaimers|
|                     |       | * Whitelist filter   |       | * Metric cards        |
+---------------------+       +----------------------+       +-----------------------+
         |                            |
  [ bitcoin_traffic.csv ]  [ flagged_transactions.csv ]
         |                            |
         +----------------------------+
                      |
           +------------------+     +----------------------+
           |   geo_asn.py     |     |  evaluate_model.py   |
           | Offline Geo-ASN  |     |  ML Validation Suite |
           | (no MaxMind API) |     |  Precision/Recall/   |
           +------------------+     |  F1 / FPR / Memory   |
                                    +----------------------+
```

**Data Flow:**
1. `transaction_adapter.py` ingests files via the appropriate adapter; XML is parsed with `defusedxml` (XXE-safe).
2. `ml_engine.py` builds a NetworkX graph, filters mixers, detects change addresses, clusters wallets, engineers 18 features, trains `IsolationForest`, applies the whitelist, and returns `(enriched_df, model, features_df)`.
3. `geo_asn.py` enriches the result with offline `geo_country` + `asn` fields (zero network I/O).
4. `app.py` displays results with forensic disclaimers, cluster confidence labels, and the `asn` column.
5. `evaluate_model.py` independently benchmarks the model across 3 threat categories.

---

## 4. Tech Stack & Design Rationale

| Library | Role | Rationale |
|---|---|---|
| **Python 3.10+** | Core Language | Standard for data science; natively supports offline execution. |
| **Pandas / NumPy** | Data Manipulation | High-performance memory structures for transaction matrices. |
| **Scikit-Learn** | ML Engine | `IsolationForest` (unsupervised anomaly detection) + `MinMaxScaler`. |
| **NetworkX** | Graph Clustering | Union-Find algorithm underpinning wallet ownership grouping. |
| **defusedxml** | XML Security | Replaces stdlib XML parser to prevent XXE injection and billion-laughs attacks. |
| **Streamlit** | Dashboard UI | Rapid Python-native frontend; no CDN calls in air-gapped mode. |
| **PyVis** | Visualization | Physics-based (`forceAtlas2Based`) interactive graphs rendered in-browser. |
| **Faker** | Data Generation | Used in `generate_data.py` and `evaluate_model.py` for synthetic traffic. |
| **ipaddress** (stdlib) | Geo-ASN Resolution | CIDR matching in `geo_asn.py` — zero external dependencies. |
| **tracemalloc** (stdlib) | Memory Profiling | Peak memory measurement in `evaluate_model.py`. |

---

## 5. Dataset Schema

All adapters normalise to this canonical schema:

| Column | Type | Example | Notes |
|---|---|---|---|
| `timestamp` | String | `2026-07-27 12:38:43` | |
| `src_ip` | String | `161.247.254.91` | Observation node — subject to VPN/NAT/Tor |
| `dst_ip` | String | `77.50.222.230` | |
| `src_port` | Int64 | `9050` | Tor SOCKS — high port risk |
| `dst_port` | Int64 | `8333` | Bitcoin P2P — zero risk |
| `txid` | String | `15eb...8d` | |
| `input_addresses` | String (pipe-delimited) | `bc1q9nB...\|3Jk...` | |
| `output_addresses` | String (pipe-delimited) | `3Sto6...\|bc1q...` | |
| `input_amounts` | Float/String | `1.81296791` | |
| `output_amounts` | Float/String | `1.812...\|0.0006...` | |
| `fee` | Float64 | `0.00068381` | |
| `script_type` | String | `P2SH` | |
| `geo_country` | String | `US` | ISO 3166-1 alpha-2; resolved offline |
| `asn` | String | `AS13335` | **New** — resolved offline by `geo_asn.py` |

> **Forensic caveat:** `src_ip`, `geo_country`, and `asn` identify the **network observation node only**. They are NOT the sender's physical identity. Results are subject to VPN/NAT/Tor limits.

---

## 6. AI/ML Pipeline Deep Dive

### IsolationForest Configuration
- **Model:** `sklearn.ensemble.IsolationForest`
- **n_estimators:** 200
- **Contamination Rate:** Default `0.05` (adjustable via UI slider)
- **Scoring:** `decision_function()` inverted and scaled to `[0, 100]` as **Anomaly Deviation Score**

### Investigative Priority Index
A composite score combining the Anomaly Deviation Score (70% weight) with a feature severity component (30% weight). This is an **investigative aid**, not a guilt indicator.

### Engineered Features (18 total)

| # | Feature | Description |
|---|---|---|
| 1 | `entity_ip_diversity` | Unique `src_ip` count per entity |
| 2 | `ip_entity_diversity` | Unique `entity_id` count per `src_ip` |
| 3 | `entity_tx_freq` | Frequency encoding of `entity_id` |
| 4 | `src_ip_freq` | Frequency encoding of `src_ip` |
| 5 | `amount_btc_scaled` | MinMax-scaled total transaction volume |
| 6 | `fee_scaled` | MinMax-scaled fee |
| 7 | `is_micro_tx` | Binary flag (total volume < 0.006 BTC) |
| 8 | `script_type_freq` | Frequency encoding of `script_type` |
| 9 | `src_port_risk` | Port risk tier (0.0–1.0) |
| 10 | `dst_port_risk` | Port risk tier (0.0–1.0) |
| 11 | `port_risk_combined` | `src_port_risk x 0.6 + dst_port_risk x 0.4` |
| 12 | `peel_chain_disparity` | `max_share - min_share` for 2-output TXs |
| 13 | `fan_in` | Count of input addresses |
| 14 | `fan_out` | Count of output addresses |
| 15 | `fan_ratio` | `fan_in / fan_out` |
| 16 | `fee_rate_urgency` | `fee / sum(input_amounts)` |
| 17 | `value_zscore` | Z-Score of total volume |
| 18 | `value_zscore_abs` | Absolute Z-Score of total volume |

---

## 7. Hardening Changelog (v2.0)

### Phase 1: ML Validation & Benchmarking — `evaluate_model.py`

A dedicated evaluation harness producing forensic-grade metrics:

```bash
python evaluate_model.py --records 300 --verbose
```

| Metric | Threshold | Description |
|---|---|---|
| **Precision** | >= 0.70 | Flagged threats that are true threats |
| **Recall** | >= 0.65 | True threats caught by the model |
| **F1-Score** | >= 0.65 | Harmonic mean of precision + recall |
| **False-Positive Rate (FPR)** | <= 0.20 | False alarms on benign traffic |
| **Median Latency** | (info) | Per-transaction inference time in microseconds |
| **Peak Memory** | (info) | tracemalloc peak across all datasets |

Three synthetic dataset categories:

- **`benign_traffic`** — Exchange/institutional transfers on standard ports; expected very low FPR.
- **`known_threats`** — Multi-hop peel chains (Tor ports), CoinJoin mixers, 30% fee spikes.
- **`novel_threats`** — Micro-dust fan-out (50+ outputs), I2P consolidation sweeps, consolidate-and-split. This is the **holdout set** that tests model generalisation to unseen patterns.

The script falls back to a lightweight 8-feature extractor if `ml_engine` is unavailable, ensuring it always runs standalone.

---

### Phase 2a: XML Security Hardening — `transaction_adapter.py`

**Risk addressed:** Malicious XML uploads could exploit `xml.etree.ElementTree`'s lack of protection against:
- **XXE (XML External Entity) injection** — exfiltrate local files via `SYSTEM` entities
- **Billion-laughs / exponential entity expansion** — crash the process
- **DTD-based denial of service**

**Fix:** `defusedxml.ElementTree` is now the primary XML parser with a graceful fallback and `RuntimeWarning`.

```bash
pip install defusedxml
```

---

### Phase 2b: Air-Gap Proof — `scripts/test_airgap.sh`

Proves 100% offline operation using a Linux network namespace:

```bash
bash scripts/test_airgap.sh
bash scripts/test_airgap.sh --verbose
bash scripts/test_airgap.sh --skip-streamlit
```

Steps inside the isolated namespace:
1. `curl https://8.8.8.8` — verified to **FAIL** (confirms air-gap)
2. All `pytest` suites run inside the namespace
3. `evaluate_model.py --records 100` runs offline
4. `geo_asn.py` resolver tested offline
5. Streamlit import + `AirGapSocket` override verified
6. Prints a signed **AIR-GAP PROOF CERTIFICATE** on success

---

### Phase 3: UI Terminology & Forensic Disclaimers — `app.py`

**Terminology changes:**

| Old Term | New Term |
|---|---|
| `Risk %` (table column) | `Investigative Priority %` |
| *(unlabelled score value)* | `Anomaly Score:` explicit label |

**Persistent disclaimers** at 3 locations in the UI:
```
Warning: Network observation correlation -- not absolute identity attribution
(Subject to VPN / NAT / Tor limits).
```

**Cluster confidence labels** in the graph node tooltips:

| Label | Trigger | Meaning |
|---|---|---|
| `High-confidence cluster` | Peel chain linkage | Strong sequential address reuse |
| `Mixer-affected cluster` | CoinJoin attack type | Lower confidence — unrelated wallets may be merged |
| `Heuristic cluster` | Common-input-ownership only | Statistical inference, not deterministic fact |
| `Whitelisted cluster` | Institutional whitelist | Cleared entity |

---

### Phase 4: Geo-ASN Offline Support — `geo_asn.py`

A fully offline IP resolver — no MaxMind API key, no network I/O, no external database file:

```python
from geo_asn import resolve_ip, resolve_ips_batch, enrich_dataframe

resolve_ip("1.1.1.1")
# {"geo_country": "AU", "asn": "AS13335", "asn_org": "Cloudflare Inc."}

resolve_ip("185.220.100.1")
# {"geo_country": "DE", "asn": "AS205100", "asn_org": "Tor Exit (ORG-TF5)"}

resolve_ip("10.0.0.1")
# {"geo_country": "PRIVATE", "asn": "AS0", "asn_org": "RFC-1918 Private Network"}
```

Coverage (50+ CIDR entries):
- RFC-1918 / loopback / link-local
- Tor exit node blocks (185.220.100.0/22, 198.98.50.0/24)
- Cloud ASNs: Cloudflare AS13335, AWS AS16509, Google AS15169, Azure AS8075
- Bitcoin node hosts: Hetzner AS24940, OVH AS16276, DigitalOcean AS14061
- Indian ISPs: BSNL AS9829, Jio AS55836, MTNL AS17813
- Russian, Chinese, Eastern European ranges

The `asn` column now appears in the Entity Attribution table and Transaction Detail cards.

Self-test: `python geo_asn.py`

---

## 8. Explainable AI / Output Logic

The `explainability.py` module generates human-readable reasons by evaluating feature vectors against **dataset percentiles** — no hardcoded thresholds. It produces a statistical **Investigative Priority Index** and **Anomaly Deviation Score**, eliminating subjective terms like "Guilt Probability" or "Confidence Score."

- **Port Anomaly:** `"Anomalous port profile -- possible proxy/Tor/tunneling"`
- **True Multi-Hop Peel Chain:** `"Peel-chain output structure detected across sequential hops -- classic layering signature"`
- **Mass Consolidation:** Triggers based on percentile-ranked fan-out ratios.
- **Fee Urgency:** `"Fee-rate urgency -- miner-priority overpayment suggesting time-sensitive hop"`
- **Volume Outlier:** Statistically anomalous values evaluated against baseline medians.
- **IP Anonymization / Botnet Relay:** Flagged when `entity_ip_diversity` or `ip_entity_diversity` exceeds 95th percentile.

---

## 9. Dashboard & Visualization Guide

- **Dynamic Sidebar:** File uploaders (.csv, .json, .xml), whitelist upload, sensitivity slider, synthetic data generator.

- **Tab 1 — Overview & Triage:** 6 Bento Metric Cards, Threat Vector chart, Anomaly Score histogram, court-ready export.

- **Tab 2 — Entity Attribution Viewer:** `Investigative Priority %`, `Entity`, `ASN`, `Propagation Node` columns. Persistent VPN/NAT/Tor disclaimer.

- **Tab 3 — Graph & Clusters:** PyVis graph with `forceAtlas2Based` physics. Cluster confidence labels in node tooltips. Threat-centric mode toggle. Ego-network isolation (2 hops). Persistent IP-attribution disclaimer.

- **Tab 4 — Analytical Explainability Panel:** Drill-down by Anomaly Score with percentile deviations. Geo-country + ASN panel with attribution caveat per transaction.

---

## 10. Project Structure

```text
SIH-2026-2/
├── generate_data.py            # Synthetic P2P traffic generator (CLI args)
├── ml_engine.py                # Core ML pipeline, graph logic, XAI
├── transaction_adapter.py      # Cross-format adapter (CSV, JSON, XML via defusedxml)
├── anomaly_engine.py           # Anomaly Deviation Score + Investigative Priority Index
├── features.py                 # Peel-chain and pipe-parsing feature extraction
├── explainability.py           # Percentile-based XAI report generation
├── app.py                      # Streamlit 4-tab UI (forensic disclaimers, ASN column)
├── geo_asn.py                  # [NEW] Offline Geo-ASN IP resolver (no external API)
├── evaluate_model.py           # [NEW] ML validation: Precision/Recall/F1/FPR/Latency
├── requirements.txt            # pip dependencies (includes defusedxml)
├── requirements-lock.txt       # Strict locked dependencies for offline reproducibility
├── Dockerfile                  # Container for air-gapped deployment
├── scripts/
│   └── test_airgap.sh          # [NEW] Air-gap proof via Linux unshare -n
├── tests/                      # Pytest suite (adapters, heuristics, airgap, whitelist)
├── evaluation/
│   └── benchmark.py            # Runtime speed + memory benchmarking
├── sample_data/                # Pre-built fixtures (.csv, .json, .xml, whitelist)
├── docs/                       # MODEL_CARD.md and DATASET_CARD.md
└── README.md                   # This file
```

---

## 11. Installation & Usage

### 1. Clone & Setup

```bash
git clone <repo_url>
cd SIH-2026-2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python generate_data.py --total-records 2000 --suspicious-ratio 0.25
```

### 3. Run the Dashboard

```bash
streamlit run app.py
# Access: http://localhost:8501
```

### 4. Run ML Validation

```bash
python evaluate_model.py --records 300 --verbose
```

### 5. Test Offline Geo-ASN Resolver

```bash
python geo_asn.py
```

### 6. Run Air-Gap Proof

```bash
bash scripts/test_airgap.sh --verbose
```

### 7. Run Headless ML Pipeline

```bash
python ml_engine.py
```

### 8. Run Test Suite

```bash
pytest tests/ -v
```

---

## 12. SIH Compliance Matrix

| Requirement | Implementation | Status |
|---|---|---|
| Offline Air-Gapped Operation | `unshare -n` proof + AirGapSocket override + no CDN | ✅ Verified |
| Ingest bulk multi-format metadata | `transaction_adapter.py` (CSV, JSON, XXE-safe XML) | ✅ Completed |
| Entity/transaction graph | `build_entity_graph()` in `ml_engine.py` | ✅ Completed |
| AI anomaly detection | `IsolationForest` on 18 features | ✅ Completed |
| ML Validation metrics | Precision / Recall / F1 / FPR / Latency / Memory | ✅ **New** |
| Wallet clustering | Union-Find via `nx.connected_components` | ✅ Completed |
| True Multi-Hop Peel Chains | Multi-step tracing across hops | ✅ Completed |
| Entity-Level Whitelisting | Zero-risk for institutional nodes | ✅ Completed |
| Explainable alerts | Percentile-based statistical telemetry profiling | ✅ Completed |
| Offline Geo enrichment | CIDR-based IP -> Country + ASN (no MaxMind API) | ✅ **New** |
| XML security hardening | `defusedxml` — XXE / billion-laughs protection | ✅ **New** |
| Forensic terminology | Anomaly Score / Investigative Priority (no "Confidence" / "Criminality") | ✅ **New** |
| Cluster confidence disclosure | High-confidence / Mixer-affected / Heuristic labels | ✅ **New** |
| Network attribution caveat | VPN/NAT/Tor disclaimer on all telemetry views | ✅ **New** |
| Interactive UI | `app.py` Streamlit + PyVis | ✅ Completed |
| Actionable intelligence | Analytical Explainability Panel & exports | ✅ Completed |

---

## 13. Security Considerations

| Threat | Mitigation |
|---|---|
| **XXE / Billion-Laughs** | `defusedxml.ElementTree` used for all XML parsing |
| **External data exfiltration** | `AirGapSocket` blocks all non-loopback TCP connections at the Python socket layer |
| **Network namespace escape** | `scripts/test_airgap.sh` verifies air-gap at the OS level via `unshare -n` |
| **IP identity over-attribution** | Persistent warning disclaimer on all network-telemetry views |
| **Cluster false positives** | Mixer-affected clusters explicitly labelled; common-input-ownership caveat shown in tooltips |
| **Model over-confidence** | Output termed "Investigative Priority" / "Anomaly Score" — not probability of guilt |

---

## 14. Future Roadmap

- **Ethereum Integration** — `EthereumAdapter` for account/nonce-based anomaly tracing.
- **Real-time Mempool Ingestion** — Subscribe to Bitcoin Core RPC node for live analysis.
- **Temporal Velocity Features** — Burst-sending pattern detection via sliding-window analysis.
- **Bundled GeoLite2 MMDB** — Ship an offline `.mmdb` file to upgrade `geo_asn.py` to city-level resolution.
- **SHAP Integration** — Replace percentile explanations with SHAP values for IsolationForest feature contributions.
- **Monero Adapter** — Ring-signature transaction parsing and stealth address decoding.
