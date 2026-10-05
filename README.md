# MITHYA — Enterprise-Grade Crypto-Forensics & Triage Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hackathon](https://img.shields.io/badge/SIH-2026-ff69b4.svg)](https://sih.gov.in/)
[![Security](https://img.shields.io/badge/XML-defusedxml%20XXE--safe-green.svg)](https://pypi.org/project/defusedxml/)
[![Air-Gap](https://img.shields.io/badge/Air--Gap-Strict%20Enforcement-blue.svg)](#12-air-gapped--offline-setup-guide)
[![Tests](https://img.shields.io/badge/Pytest-87%20passed-brightgreen.svg)](#11-test-suite--validation)
[![Throughput](https://img.shields.io/badge/Throughput-1%2C500%2B%20tx%2Fs-orange.svg)](#7-scalability-benchmarks--performance)
[![Offline Geo-ASN](https://img.shields.io/badge/Geo--ASN-Offline%20MMDB%20%2B%20TopoJSON-purple.svg)](#2-key-features)

**MITHYA** is an offline, graph-aware cryptocurrency forensics and triage engine built for **Smart India Hackathon 2026 — Problem Statement 26146**. It ingests raw network and blockchain telemetry, clusters wallet entities using Evidence-Based Confidence Tiers, traces multi-hop fund flows, computes threat-intel taint propagation, and scores transactions for illicit behavior using multi-signal machine learning — fully air-gapped with zero external API, CDN, or tile dependencies.

> **Forensic Evidentiary Assurance:** Built to Federal Rule of Evidence 902(13)/(14) electronic record standards: cryptographic SHA-256 data provenance hashing, court-ready 14-page PDF and JSON evidence dossiers with `.sha256` sidecar verification, XXE-safe fail-closed parsing, and verified Linux network namespace air-gap isolation.

---

## 1. Problem Statement & Motivation

Bitcoin's pseudonymous design allows illicit actors to move, layer, and cash out criminal proceeds while evading standard financial surveillance. Law enforcement and compliance officers need tools that operate entirely **offline** on sensitive evidentiary machines while providing AI-assisted forensic triage, topological fund tracing, and graph analytics.

MITHYA addresses the core challenges of crypto-investigations:

1. **The Mixer Loophole** — Standard heuristics blindly group innocent users with criminals when interacting with CoinJoins.
2. **IP/VPN Falsification** — Physical IP mapping is unreliable against actors routing through Tor relays, VPNs, and proxies.
3. **Analyst Fatigue** — High-volume institutional traffic constantly triggers false-positive anomaly alerts.
4. **Cluster Over-Confidence** — Common-input-ownership is a heuristic, not a fact; mixers pollute cluster boundaries.
5. **Multi-Hop Layering** — Complex peel chains and dispersal fan-outs obscure the ultimate beneficiaries across hops.

---

## 2. Key Features

| Feature | Description |
|---|---|
| **Mixer/CoinJoin Bypass** | Detects equal-output signatures (`equal_output_ratio=0.50`) and blocks false entity aggregation. |
| **Change-Address Detection** | 5-factor heuristic (script-type match, unrounded remainder, decimal precision, novelty, asymmetry) links change outputs back to sender entity. |
| **Fused Multi-Signal Anomaly Engine** | Composite risk scoring combining unsupervised Isolation Forest, structural pattern heuristics, network telemetry risk, and graph taint propagation. |
| **Hop-by-Hop Fund Flow Tracer** | Forward and backward breadth-first search (BFS) fund flow traversal with automated role classification (`Mixer`, `Exchange endpoint`, `Dispersal`, `Consolidation`, `Layering`, `Source`, `Intermediary`), delta-t ($\Delta t$), retained capital tracking, and interactive hierarchical path visualization. |
| **Behavioural Clustering & Communities** | HDBSCAN unsupervised behavioural clustering on entity feature profiles, 2D PCA projection, and NetworkX Louvain graph community partitioning. |
| **Dual Telemetry Ingestion** | Supports single merged transaction files (CSV, JSON, XML) or separate network telemetry (`network_telemetry.csv`) and blockchain files (`blockchain_tx.csv`) joined on `txid` with first-seen IP, observation counts, and relay spread analysis. |
| **Offline Global Threat Choropleth** | Zero-network geographic heatmap powered by local TopoJSON (`static/world-110m.json`) and ISO 3166-1 country mapping (`data/geo/iso_codes.csv`) with zero tile/CDN dependencies. |
| **Court-Ready Evidence Dossiers** | 14-page PDF dossiers via ReportLab featuring cover page provenance, executive summary triage funnel, top 25 ranked leads, top 10 detailed profiles with Matplotlib score breakdown bar charts, flow traces, statistical deviations, official examiner certification, and signature block. |
| **JSON Evidence Packages & Sidecars** | Structured JSON dossier export plus cryptographic `.sha256` sidecar checksum files for automated chain of custody verification. |
| **Cryptographic Provenance** | Source dataset SHA-256 hash, whitelist SHA-256, watchlist SHA-256, and RF model hash tracked and displayed with `st.code`. |
| **XXE-Safe & DoS-Resilient Ingestion** | `defusedxml` fail-closed architecture prevents billion-laughs and XML External Entity injection attacks; enforces strict file size (100 MB) and row count (250,000) safety limits. |
| **Offline Geo-ASN Enrichment** | Local offline DB-IP / MaxMind Lite integration (`geo_asn.py`) resolves IPs to real Autonomous System Numbers with no external network calls and strict unknown sentinels (`XX`, `AS0`, `Unknown`). |
| **Air-Gap Socket Interceptor** | Runtime socket interceptor (`AirGapSocket`) strictly blocks outbound network calls, allowing only local IPC, verified by `scripts/test_airgap.sh`. |

---

## 3. Operational 6-Tab Dashboard Architecture

The MITHYA dashboard (`app.py`) provides an operational 6-tab forensic workstation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MITHYA 🛡️  |  🟢 Offline  |  🔒 Air-Gapped  |  ⚡ Processed 10,004 tx in 6.9s  │
├──────────────┬──────────────┬────────────┬─────────────┬───────────┬────────┤
│ 1. Overview  │ 2. Suspicious│ 3. Network │ 4. Why      │ 5. Model  │ 6.     │
│ & Triage     │ Transactions │ Graph      │ Flagged?    │ Evaluation│ Clust  │
└──────────────┴──────────────┴────────────┴─────────────┴───────────┴────────┘
```

### 1. Tab 1: 📊 Overview & Triage Funnel
- **Executive Bento Metrics:** Live operational counters: Total Analyzed, Institutional Cleared, Flagged Anomalies, Peel Chains, CoinJoin Mixers, and Maximum Priority Index.
- **Offline Global Threat Telemetry:**
  - **Choropleth World Heatmap:** Fully offline Altair/Vega TopoJSON world map (`static/world-110m.json`) colored by flagged anomaly density per country.
  - **Top 10 Countries Bar Chart:** Breakdown of flagged vs. total observed transactions by jurisdiction.
  - **Top 10 ASNs Table:** Ranking observed Autonomous Systems by threat incidence, sorting unresolved IPs to the bottom.
- **Attack Signature Distribution & Anomaly Score Histograms:** Visual breakdown of threat vectors and score distribution across the ingested corpus.
- **Forensic Case Export Center & Ingestion Audit:**
  - Analyst name input and auto-generated Case ID (`MITHYA-YYYYMMDD-HHMMSS`).
  - Cryptographic provenance display: Dataset SHA-256, Whitelist SHA-256, Watchlist SHA-256, and Calibrated RF Model SHA-256 displayed via `st.code`.
  - **Court-Ready PDF Dossier (14 pages):** Generated on the fly with ReportLab and rendered with sidecar checksums.
  - **JSON Evidence Package:** Comprehensive structured machine-readable evidence export.
  - **Legacy TXT Report:** ASCII text audit log for command-line and air-gap terminals.
  - **`.sha256` Sidecar File:** GNU `sha256sum`-compatible tamper-proof sidecar file.
  - **Enriched Tabular CSV:** Full enriched dataset export with all engineered features, risk scores, and forensic tags.
  - **Ingestion Audit Report:** Detailed audit of accepted, rejected, and sanitized records.

### 2. Tab 2: 🚨 Suspicious Transactions
- **Investigative Priority Ranking:** Transactions ranked strictly by composite Investigative Priority Index (0–100%).
- **Confidence Tiers:** Displays evidence-based clustering confidence badges (`High-Confidence`, `Mixer-Affected`, or `Heuristic/Inferred`).
- **Interactive Multi-Filter System:** Filter by attack signature (Peel Chain, CoinJoin Mixer, Fan-Out, Fee Spike, High-Volume Layering), risk score range slider, and search by TXID, Address, IP, or ASN.
- **Forensic Badges & Plain-English Explanations:** Displays offline Geo-country, ASN organization, threat-intel taint status, and plain-English rationale for every flagged item.

### 3. Tab 3: 🕸️ Interactive Network Graph
- **Tripartite Entity Graph:** Interactive graph linking Transactions $\leftrightarrow$ Addresses/Entities $\leftrightarrow$ Observation IPs rendered using PyVis and `forceAtlas2Based` physics.
- **Color Semantics:**
  - 🔴 **Red:** Flagged anomalous transactions.
  - 🟠 **Amber:** Standard transactions.
  - 🟣 **Purple:** Clustered wallet entities.
  - 🟢 **Green:** Regulated institutional/exchange entities.
  - 🔵 **Blue:** Network observation context nodes (IPs).
  - ★ **Red Star:** Threat-intel watchlist match (known-bad).
- **Scalability Safeguard:** In "View Full Graph" mode, automatically capped at the 400 highest-risk transactions plus their direct neighbors to avoid browser DOM collapse on 10k–100k datasets, with an active notification.
- **Ego-Network Isolation:** Allows selecting any target entity or IP and isolating its 1-hop or 2-hop transaction neighborhood.
- **Stabilization & Physics Controls:** Interactive freeze toggle allowing analysts to inspect complex subgraphs without node drift.

### 4. Tab 4: 🔍 Why Flagged? (Analytical XAI & Fund Flow Tracer)
- **Granular Lead Inspector:** Inspect any of the top 200 ranked leads with deep contextual metadata (timestamps, input/output counts, total volume, miner fee, script types, offline ASN, and observation spread).
- **Persistent Network Disclaimer:** Emphasizes that P2P IP observations reflect broadcast relay nodes and are subject to VPN/NAT/Tor limits, not absolute identity attribution.
- **Risk Score Decomposition Gauge:** Visual bar breakdown of ML anomaly score, structural pattern heuristic, network telemetry risk, and graph taint propagation.
- **Statistical Feature Deviation Audit Table:** Compares observed values against dataset median and computes exact percentile ranks (e.g., 99.8th percentile) with plain-English audit reasons.
- **🧭 Follow the Money (Hop-by-Hop Fund Flow Tracer):**
  - **Bi-Directional Traversal:** Upstream backward provenance tracing and downstream forward dispersal tracing up to 6 hops or 200 nodes.
  - **Automated Role Attribution:** Tags each hop node as `Mixer`, `Exchange endpoint`, `Dispersal`, `Consolidation`, `Layering`, `Source`, or `Intermediary`.
  - **Flow Metrics:** Computes time delta ($\Delta t$ minutes), retained capital %, peeled BTC amount, and hop distance.
  - **Interactive Hierarchical Path Visualizer:** Left-to-right (LR) directed Vis.js graph mapping fund trajectory, colored by forensic role, with edge labels indicating transferred BTC amounts.
  - **Address Flow Breakdown Table:** Explicit address-level inputs and outputs per hop.

### 5. Tab 5: 📈 Model Performance & Evaluation
- **Supervised Calibrated Classifier:** Pre-trained Random Forest (300 estimators, balanced class weights) calibrated via Isotonic Regression (`CalibratedClassifierCV`).
- **Provenance Verification:** Verifies model integrity against `models/rf_model.sha256`. Automatically trains a 5-fold stratified cross-validated model if new labeled data is provided.
- **Multi-Threshold ROC & Precision-Recall Curves:** Interactive Plotly curves displaying Area Under Curve (ROC-AUC and PR-AUC) across operating thresholds.
- **Probability Calibration Diagnostics:** Reliability curve (calibration plot) with Brier score comparing raw uncalibrated probabilities against isotonic-calibrated probabilities.
- **Confusion Matrix & Metrics:** Precision, Recall, F1-Score, and False Positive / False Negative counts.
- **Generalization Gap Analysis:** Train-vs-Test score comparison to guarantee the model does not suffer from overfitting.

### 6. Tab 6: 👥 Clusters & Communities
- **HDBSCAN Behavioural Clustering:** Unsupervised spatial clustering on high-dimensional entity behavioural profiles (volume, velocity, fan-in/fan-out ratios, fee generosity).
- **2D PCA Projection:** Interactive scatter plot projecting multi-dimensional entity profiles into 2D space, color-coded by behavioural cluster with point size scaled by risk score.
- **NetworkX Louvain Community Partitioning:** Graph modularity optimization detecting co-operating wallet subgraphs and financial syndicates.
- **Cluster Profile Summary Table:** Cluster ID, entity count, mean risk score, dominant attack pattern, and primary ASN.

---

## 4. Multi-Layer Telemetry & Ingestion Architecture

MITHYA correlates blockchain state with peer-to-peer network broadcast telemetry:

```
[ P2P Network Telemetry ]                  [ Blockchain Ledgers ]
 (timestamp, src_ip, dst_ip,                (txid, timestamp, inputs,
  src_port, dst_port, txid)                  outputs, amounts, fee)
            │                                         │
            └─────────────────┬───────────────────────┘
                              ▼
               [ correlate_layers() on txid ]
                              │
     ┌────────────────────────┴────────────────────────┐
     ▼                                                 ▼
[ first_seen_ip ]                             [ Observation Spread ]
Earliest broadcasting relay                   Multi-relay spread (s) &
node on P2P network                           observation count
```

- **Single Merged File Mode:** Ingests unified CSV, JSON, or XML transaction streams.
- **Separate Files Mode:** Ingests `network_telemetry.csv` and `blockchain_tx.csv`, resolving multi-relay spreads, earliest seen broadcast IPs, and P2P observation counts.
- **Ingestion Sanitization & Validation:** Enforces non-negative amounts, valid pipe-delimited structures, DoS file size limits (100 MB max), row count caps (250,000 max), and defused XML parsing.

---

## 5. Evidence Dossier Specification (`dossier.py`)

Court-ready dossiers are generated in both binary PDF and structured JSON formats conforming to evidentiary standards:

- **Page 1 — Evidentiary Cover Page:** Case ID (`MITHYA-YYYYMMDD-HHMMSS`), Lead Forensic Analyst name, timestamp, air-gapped mode certification, dataset SHA-256, institutional whitelist SHA-256, watchlist SHA-256, model engine version, RF model SHA-256, contamination rate, fusion weights, and structural thresholds.
- **Page 2 — Executive Summary & Triage Funnel:** Pipeline ingestion counts, whitelist clearance metrics, detected vector breakdowns, and Critical / High / Medium / Low risk tier counts.
- **Page 3 — Ranked Investigative Leads Table:** Top 25 priority leads with rank, TXID, priority %, tier, detected pattern, BTC volume, source IP, and threat-intel taint status.
- **Pages 4–13 — Detailed Forensic Profiles (Top 10 Leads):** Dedicated pages per lead featuring:
  - High-resolution Matplotlib horizontal score breakdown bar chart.
  - Core metadata grid (timestamps, amounts, fees, source/destination IPs and ports, offline Geo/ASN, observation spread).
  - Statistical feature deviations table with observed values and percentile ranks.
  - Forward hop-by-hop flow trace table with role attribution and peeled capital.
  - Watchlist alerts, taint propagation warnings, and forensic AI explanation rationale.
- **Page 14 — Methodological Boundaries & Limitations:** Synthetic simulation baseline note, network attribution caveats (VPN, NAT, Tor), and "lead not verdict" guidance, followed by an official Examiner Certification and Signature block.
- **Running Footer:** Two-pass `NumberedCanvas` rendering `"Page N of 14 — MITHYA — Offline"` on every page.
- **Sidecar Checksums:** Companion `.sha256` sidecar file generated alongside dossiers for chain-of-custody verification.

---

## 6. Dataset Schema

All input adapters normalise data to the canonical forensic schema:

| Column | Type | Example | Description |
|---|---|---|---|
| `txid` | String | `15eb...8d` | Unique transaction identifier |
| `timestamp` | Datetime | `2026-10-04 12:38:43` | Transaction creation or broadcast time |
| `src_ip` | String | `161.247.254.91` | First-seen P2P broadcast observation node |
| `dst_ip` | String | `77.50.222.230` | Receiving peer node |
| `src_port` | Int64 | `9050` | Source port (Tor / standard) |
| `dst_port` | Int64 | `8333` | Destination port (Bitcoin P2P) |
| `input_addresses` | String | `bc1q9nB...\|3Jk...` | Pipe-delimited sender addresses |
| `output_addresses` | String | `3Sto6...\|bc1q...` | Pipe-delimited recipient addresses |
| `input_amounts` | String | `1.81296791` | Pipe-delimited input amounts (BTC) |
| `output_amounts` | String | `1.812...\|0.0006...` | Pipe-delimited output amounts (BTC) |
| `fee` | Float64 | `0.00068381` | Miner fee (BTC) |
| `script_type` | String | `P2WPKH` | Dominant script format |
| `geo_country` | String | `US` | ISO country code resolved offline |
| `asn` | String | `AS13335` | Autonomous System Number resolved offline |
| `asn_org` | String | `Cloudflare, Inc.` | Autonomous System Organization name |
| `ip_observation_count` | Int64 | `3` | P2P relay observations for this transaction |
| `observation_spread_s` | Float64 | `12.4` | Relay spread latency across nodes (seconds) |

---

## 7. Scalability Benchmarks & Performance

Empirical pipeline scalability benchmarks generated via `scripts/benchmark.py` and recorded in `evaluation/benchmark.csv`:

### Pipeline Stage Timing & Throughput

| Scale | Records | Ingestion & Norm | Feature Eng | Model Train | Scoring & Expl | Whitelist & Taint | Fusion | Total Pipeline | Throughput | Peak Memory |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10K** | 10,004 | 0.02s | 2.86s | 0.41s | 1.24s | 2.09s | 0.25s | **6.88s** | **1,454 tx/s** | 460 MB |
| **50K** | 50,000 | 0.07s | 14.95s | 1.21s | 6.37s | 11.15s | 0.38s | **34.13s** | **1,465 tx/s** | 588 MB |
| **100K** | 100,003 | 0.13s | 31.38s | 2.19s | 13.53s | 17.67s | 0.90s | **65.81s** | **1,519 tx/s** | 761 MB |

*Tested on Linux x86_64, Python 3.14.7, zero GPU requirements.*

To reproduce benchmarks:
```bash
python scripts/benchmark.py --scales 10k,50k,100k --output evaluation/benchmark.csv
```

---

## 8. ML Model Evaluation & Multi-Signal Fusion

MITHYA benchmarks unsupervised anomaly detection and supervised classifiers using a strict Train/Validation/Test setup (`evaluate_model.py` and `model_eval.py`):

### Model Capabilities & Baseline Comparisons

1. **Unsupervised Isolation Forest:**
   - Detects novel, zero-day threat patterns without labeled data.
   - High sensitivity on statistical outliers and multi-hop structure anomalies.
   - Mixers are bypassed before training to prevent skewed contamination bounds.
2. **Supervised Random Forest Classifier:**
   - 300 decision trees, balanced class weights, cross-validated and probability-calibrated using isotonic regression (`CalibratedClassifierCV`).
   - Cached model weights and SHA-256 hash persisted to `models/rf_model.joblib` and `models/rf_model.sha256`.
3. **Multi-Signal Score Fusion:**
   $$\text{Priority} = w_{\text{ml}} \cdot S_{\text{ml}} + w_{\text{pat}} \cdot S_{\text{pattern}} + w_{\text{net}} \cdot S_{\text{network}} + w_{\text{graph}} \cdot S_{\text{graph}} - \text{Discount}_{\text{whitelist}}$$
   - **Fusion weights:** ML Anomaly / Probability (50%), Pattern Signatures (25%), Network Risk (15%), Graph Taint (10%).
   - Institutional Whitelist Discount applied dynamically (up to 100% discount if both sides are regulated entities).

---

## 9. Tech Stack & Dependencies

| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.10+ | Scientific computing standard; runs natively offline. |
| **Data Processing** | Pandas, NumPy | High-performance in-memory tabular structures. |
| **Machine Learning** | Scikit-Learn | `IsolationForest`, `RandomForestClassifier`, `CalibratedClassifierCV`. |
| **Graph Analytics** | NetworkX | Union-Find address clustering, Louvain communities, PageRank taint. |
| **Clustering & PCA** | Scikit-Learn (HDBSCAN, PCA) | High-density spatial behavioural clustering. |
| **Document Generation**| ReportLab, Matplotlib | Evidentiary PDF layout engine with two-pass running footers and charts. |
| **Security & Provenance** | defusedxml, hashlib | Fail-closed XXE parsing, billion-laughs mitigation, and SHA-256 provenance hashing. |
| **Offline Geo-ASN** | maxminddb, geoip2 | Local MMDB reader (`data/geo/`) with strict unknown sentinels (no DNS / HTTP queries). |
| **Choropleth Heatmap** | Altair, TopoJSON | Zero-network global threat map using local `static/world-110m.json`. |
| **Frontend UI** | Streamlit | Rapid, responsive air-gapped web interface. |
| **Network Visualization**| PyVis, Vis.js | Interactive physics-based HTML canvas rendering. |

---

## 10. Project Directory Layout

```text
SIH-2026-2/
├── app.py                             # Main Streamlit dashboard (6-tab architecture)
├── dossier.py                         # Court-ready 14-page PDF and JSON evidence dossier generator
├── flow_tracer.py                     # BFS hop-by-hop fund flow tracer & role attribution
├── clustering.py                      # HDBSCAN behavioural clusters & Louvain communities
├── ml_engine.py                       # Core ML pipeline, Union-Find clustering, taint propagation
├── transaction_adapter.py             # Multi-format adapter (CSV, JSON, XML) & telemetry correlation
├── anomaly_engine.py                  # Anomaly scoring and score fusion logic
├── features.py                        # Structural peel chain and pipe-delimited feature extraction
├── explainability.py                  # Percentile-based statistical explainability
├── geo_asn.py                         # Offline DB-IP / MaxMind ASN and country resolver
├── generate_data.py                   # Synthetic Bitcoin transaction & telemetry generator
├── evaluate_model.py                  # ML validation harness (Train/Val/Test splits)
├── model_eval.py                      # Supervised model training, calibration, and caching
├── pytest.ini                         # Pytest configuration (pythonpath = .)
├── models/
│   ├── rf_model.joblib                # Calibrated Random Forest model artifact
│   └── rf_model.sha256                # Model SHA-256 cryptographic provenance hash
├── data/
│   └── geo/
│       ├── dbip-country-lite.mmdb     # Local offline country MMDB
│       ├── dbip-asn-lite.mmdb         # Local offline ASN MMDB
│       └── iso_codes.csv              # ISO 3166-1 country lookup table
├── static/
│   └── world-110m.json                # Local TopoJSON world atlas for offline choropleth
├── evaluation/
│   ├── benchmark.py                   # Scalability benchmarking harness
│   └── benchmark.csv                  # Recorded scalability results (10k, 50k, 100k)
├── sample_data/
│   ├── network_telemetry.csv          # Standalone P2P broadcast telemetry sample
│   ├── blockchain_tx.csv              # Standalone blockchain ledger sample
│   ├── institutional_whitelist.csv    # Regulated exchange and institutional addresses
│   ├── watchlist.csv                  # Threat-intel indicators (addresses, IPs, ASNs)
│   ├── synthetic_transactions_sample.json # Nested JSON sample
│   └── synthetic_transactions_sample.xml  # Nested XML sample
├── scripts/
│   ├── benchmark.py                   # Multi-scale benchmark execution script
│   └── test_airgap.sh                 # Linux network namespace air-gap verification
├── tests/                             # Comprehensive test suite (87 tests)
│   ├── test_adapters.py               # Ingestion, normalization, XXE, billion laughs (27 tests)
│   ├── test_airgap.py                 # Socket-level air-gap enforcement (1 test)
│   ├── test_clustering.py             # HDBSCAN and Louvain clustering (5 tests)
│   ├── test_detected_type.py          # Threat pattern classification (4 tests)
│   ├── test_dossier.py                # PDF & JSON dossier verification (3 tests)
│   ├── test_flow_tracer.py            # Hop-by-hop traversal & role attribution (4 tests)
│   ├── test_geo_asn.py                # Offline IP and ASN resolution (4 tests)
│   ├── test_heuristics.py             # Isolation Forest & port risk scoring (2 tests)
│   ├── test_ingestion_report.py       # Ingestion audit and rejections (3 tests)
│   ├── test_model_eval.py             # Supervised calibration & model caching (6 tests)
│   ├── test_security.py               # File size, row limits, DoS guards (10 tests)
│   ├── test_watchlist_taint.py        # Threat-intel matching & graph taint (6 tests)
│   └── test_whitelist.py              # Institutional clearance & discounts (12 tests)
├── requirements.txt                   # Production Python dependencies
├── requirements-lock.txt              # Pinned dependency versions
├── Dockerfile                         # Container definition for air-gapped staging
└── README.md                          # Comprehensive documentation
```

---

## 11. Test Suite & Validation

The test suite covers unit and integration tests across 13 test modules:

```bash
# Run entire test suite (using pytest.ini)
pytest -q
```

**Status:**
- **87 passed** in ~29 seconds (0 failures, 0 warnings, 0 skipped).

| Test Module | Tests | Scope |
|---|:---:|---|
| `test_adapters.py` | 27 | CSV/JSON/XML ingestion, multi-layer correlation, XXE mitigation, billion laughs defense. |
| `test_airgap.py` | 1 | Interception of outbound socket calls by `AirGapSocket`. |
| `test_clustering.py` | 5 | Entity profiling, HDBSCAN clusters, Louvain communities, summary tables. |
| `test_detected_type.py` | 4 | Threat pattern taxonomy agreement, null safety, CoinJoin explanations. |
| `test_dossier.py` | 3 | JSON dossier structure, ReportLab 14-page PDF layout, small-dataset resilience. |
| `test_flow_tracer.py` | 4 | Spend index, forward peel chain monotonic decrease, exchange/mixer termination, backward provenance. |
| `test_geo_asn.py` | 4 | Public IP resolution, RFC-1918 private ranges, DataFrame enrichment, sentinel fallback. |
| `test_heuristics.py` | 2 | Isolation Forest novelty scoring, port risk evaluation. |
| `test_ingestion_report.py`| 3 | Rejected rows audit, negative amounts / invalid structure handling, pipeline execution metrics. |
| `test_model_eval.py` | 6 | Supervised model training, label mapping, leakage absence, ECE / precision@k, cached inference. |
| `test_security.py` | 10 | File size validation (100 MB max), row count boundaries (250,000 max), DoS safeguards. |
| `test_watchlist_taint.py` | 6 | Address/IP/ASN watchlist matching, graph taint propagation, forensic explanation prepending. |
| `test_whitelist.py` | 12 | Institutional whitelist clearance, discounts for input/output/both sides. |

---

## 12. Air-Gapped & Offline Setup Guide

MITHYA is engineered to run in strictly isolated, air-gapped government and forensic environments.

### Step 1: Clone Repository
```bash
git clone https://github.com/devansh2007-ruikar/SIH.git
cd SIH
```

### Step 2: Create Offline Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Offline Dependencies
On an internet-connected staging machine:
```bash
pip download -r requirements.txt -d ./offline_wheels
```
On the air-gapped machine:
```bash
pip install --no-index --find-links=./offline_wheels -r requirements.txt
```

### Step 4: Verify Offline Geo-ASN Resolver
```bash
python3 geo_asn.py
```
*Confirms offline MaxMind/DB-IP MMDB loads with zero network queries.*

### Step 5: Run Air-Gap Verification Script
```bash
bash scripts/test_airgap.sh
```
*Executes the system inside an unshared Linux network namespace (`unshare -r -n`) verifying that all functions run with external network access blocked.*

### Step 6: Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Outbound socket calls are intercepted and blocked by `AirGapSocket`, guaranteeing complete privacy.

---

## 13. License & Disclaimer

Built for **Smart India Hackathon (SIH 2026) — Problem Statement 26146**.

**Forensic Disclaimer:** MITHYA is an investigative triage instrument, not an adjudicative verdict. Flagged transactions and entity clusters represent statistical, network, and graph anomalies designed to guide compliance officers, forensic auditors, and law enforcement analysts toward high-priority leads. Every lead must be verified through lawful process, KYC records, and corroborating evidence.
