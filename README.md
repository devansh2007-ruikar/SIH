# MITHYA — Enterprise-Grade Crypto-Forensics & Triage Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hackathon](https://img.shields.io/badge/SIH-2026-ff69b4.svg)](https://sih.gov.in/)
[![Security](https://img.shields.io/badge/XML-defusedxml%20XXE--safe-green.svg)](https://pypi.org/project/defusedxml/)
[![Air-Gap](https://img.shields.io/badge/Air--Gap-Strict%20Enforcement-blue.svg)](#12-air-gapped--offline-setup-guide)
[![Tests](https://img.shields.io/badge/Pytest-83%20passed-brightgreen.svg)](#11-test-suite--validation)
[![Throughput](https://img.shields.io/badge/Throughput-1%2C500%2B%20tx%2Fs-orange.svg)](#7-scalability-benchmarks--performance)

**MITHYA** is an offline, graph-aware cryptocurrency forensics and triage engine built for **Smart India Hackathon 2026 — Problem Statement 26146**. It ingests raw network and blockchain telemetry, clusters wallet entities using Evidence-Based Confidence Tiers, traces multi-hop fund flows, computes threat-intel taint propagation, and scores transactions for illicit behavior using multi-signal machine learning — fully air-gapped with zero external API or CDN dependencies.

> **Forensic Evidentiary Assurance:** Built to Federal Rule of Evidence 902(13)/(14) electronic record standards: cryptographic SHA-256 data provenance hashing, court-ready PDF and JSON evidence dossiers with `.sha256` sidecar verification, XXE-safe fail-closed parsing, and verified Linux network namespace air-gap isolation.

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
| **Hop-by-Hop Fund Flow Tracer** | Forward and backward breadth-first search (BFS) fund flow traversal with automated role classification (`Mixer`, `Exchange endpoint`, `Dispersal`, `Consolidation`, `Layering`, `Source`, `Intermediary`), delta-t, and retained capital tracking. |
| **Behavioural Clustering & Communities** | HDBSCAN unsupervised behavioural clustering on entity feature profiles, 2D PCA projection, and NetworkX Louvain graph community partitioning. |
| **Dual Telemetry Ingestion** | Supports single merged transaction files or separate network telemetry (`network_telemetry.csv`) and blockchain files (`blockchain_tx.csv`) joined on `txid` with first-seen IP, observation counts, and relay spread analysis. |
| **Court-Ready Evidence Dossiers** | Multi-page PDF dossiers via ReportLab featuring cover page provenance, executive summary triage funnel, top 25 ranked leads, top 10 detailed profiles with matplotlib score breakdown bar charts, flow traces, statistical deviations, and limitations. |
| **JSON Evidence Packages & Sidecars** | Structured JSON dossier export plus cryptographic `.sha256` sidecar checksum files for automated chain of custody verification. |
| **Cryptographic Provenance** | Source dataset SHA-256 hash, whitelist SHA-256, watchlist SHA-256, and RF model hash tracked and displayed with `st.code`. |
| **XXE-Safe XML Ingestion** | `defusedxml` fail-closed architecture prevents billion-laughs and XML External Entity injection attacks. |
| **Offline Geo-ASN Enrichment** | Local offline DB-IP / MaxMind Lite integration (`geo_asn.py`) resolves IPs to real Autonomous System Numbers with no external network calls. |
| **Air-Gap Socket Interceptor** | Runtime socket interceptor (`AirGapSocket`) strictly blocks outbound network calls, allowing only local IPC. |

---

## 3. Operational 7-Tab Dashboard Architecture

The MITHYA dashboard (`app.py`) provides an operational 7-tab interface:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MITHYA 🛡️  |  🟢 Offline  |  🔒 Air-Gapped  |  ⚡ Processed 10,004 tx in 6.9s  │
├─────────┬─────────────┬────────────┬───────────┬─────────────┬───────────┬──────┤
│ 1. Triage│ 2. Priority │ 3. Fund    │ 4. Network│ 5. Why      │ 6. Model  │ 7.   │
│ Overview│ Leads       │ Flow Tracer│ Graph     │ Flagged?    │ Evaluation│ Clust│
└─────────┴─────────────┴────────────┴───────────┴─────────────┴───────────┴──────┘
```

1. **Tab 1: Overview & Triage Funnel**
   - Executive Bento metrics: Total Analyzed, Institutional Cleared, Flagged Anomalies, Mixers, Peel Chains, Fan-Outs, Fee Spikes.
   - Attack signature distribution and Anomaly Score histograms.
   - **Forensic Case Export Center:** Analyst name input, auto-generated Case ID (`MITHYA-YYYYMMDD-HHMMSS`), PDF SHA-256 display via `st.code`, and 4 export downloads:
     - 📕 **Download Dossier (PDF)**: Multi-page court-ready dossier.
     - 📋 **Download Evidence (JSON)**: Structured JSON forensic evidence package.
     - 📄 **Download Report (TXT)**: Legacy ASCII court-ready triage report.
     - 🔐 **Download .sha256 Sidecar**: GNU-compatible checksum file for tamper-proofing.
     - 📥 **Download Full Results CSV**: Complete enriched tabular dataset.
2. **Tab 2: Suspicious Transactions**
   - Transactions ranked strictly by composite Investigative Priority Index.
   - Filter by detected pattern (Peel Chain, CoinJoin Mixer, Fan-Out, Fee Spike) and risk score range.
   - Displays cluster confidence, ASN, Geo-country, and forensic explanation reasons.
3. **Tab 3: Fund Flow Tracer**
   - Hop-by-hop forward and backward fund trajectory tracking up to 6 hops or 200 nodes.
   - Automated node role attribution (`Mixer`, `Exchange endpoint`, `Dispersal`, `Consolidation`, `Layering`, `Source`, `Intermediary`).
   - Calculates time delta ($\Delta t$ minutes), retained capital %, peeled BTC, and hop positions.
4. **Tab 4: Interactive Network Graph**
   - Tripartite graph (Transactions $\leftrightarrow$ Addresses/Entities $\leftrightarrow$ IPs) rendered using PyVis and `forceAtlas2Based` physics.
   - **Large-Scale Capping**: In "View Full Graph" mode, capped at the 400 highest-risk transactions plus their direct neighbours with the indicator: `"Showing top 400 of N; use Isolate Target Entity for more"`.
   - **Ego-Network Isolation**: Isolate any target wallet entity up to 2 hops away.
5. **Tab 5: Why Flagged? (Analytical XAI)**
   - Per-transaction explainability inspector showing exact percentile rankings across features.
   - Statistical feature deviation audit tables displaying observed values vs. baseline distributions.
6. **Tab 6: ML Model Evaluation & Performance**
   - Multi-threshold ROC curves and Precision-Recall curves.
   - Probability calibration diagnostics (reliability diagrams with Brier score).
   - Confusion matrices and train-vs-test generalisation gap analysis.
7. **Tab 7: 👥 Clusters & Communities**
   - Unsupervised HDBSCAN clustering on entity behavioural profiles.
   - 2D PCA projection scatter plot color-coded by behavioural cluster with risk score scaling.
   - NetworkX Louvain community detection identifying co-operating wallet subgraphs.

---

## 4. Multi-Layer Telemetry & Correlation Architecture

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

- **Single Merged File Mode**: Ingests unified CSV, JSON, or XML transaction streams.
- **Separate Files Mode**: Ingests `network_telemetry.csv` and `blockchain_tx.csv`, resolving multi-relay spreads, earliest seen broadcast IPs, and P2P observation counts.

---

## 5. Evidence Dossier Specification (`dossier.py`)

Court-ready dossiers are generated in both binary PDF and JSON formats:

- **Page 1 — Evidentiary Cover Page**: Case ID (`MITHYA-YYYYMMDD-HHMMSS`), Lead Forensic Analyst name, timestamp, air-gapped mode certification, dataset SHA-256, institutional whitelist SHA-256, watchlist SHA-256, model engine version, RF model SHA-256, contamination rate, fusion weights, and structural thresholds.
- **Page 2 — Executive Summary & Triage Funnel**: Pipeline ingestion counts, whitelist clearance metrics, detected vector breakdowns, and Critical / High / Medium / Low risk tier counts.
- **Page 3 — Ranked Investigative Leads Table**: Top 25 priority leads with rank, TXID, priority %, tier, detected pattern, BTC volume, source IP, and threat-intel taint status.
- **Pages 4–13 — Detailed Forensic Profiles (Top 10 Leads)**: Dedicated pages per lead featuring:
  - High-resolution Matplotlib horizontal score breakdown bar chart.
  - Core metadata grid (timestamps, amounts, fees, source/destination IPs and ports, offline Geo/ASN, observation spread).
  - Statistical feature deviations table with observed values and percentile ranks.
  - Forward hop-by-hop flow trace table with role attribution and peeled capital.
  - Watchlist alerts, taint propagation warnings, and forensic AI explanation rationale.
- **Page 14 — Methodological Boundaries & Limitations**: Synthetic simulation baseline note, network attribution caveats (VPN, NAT, Tor), and "lead not verdict" guidance, followed by an official Examiner Certification and Signature block.
- **Running Footer**: Two-pass `NumberedCanvas` rendering `"Page N — MITHYA — Offline"` on every page.

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

## 8. ML Model Evaluation & Honest Metrics

MITHYA benchmarks unsupervised anomaly detection and supervised classifiers using a strict Train/Validation/Test setup (`evaluate_model.py` and `model_eval.py`):

### Model Capabilities & Baseline Comparisons

1. **Unsupervised Isolation Forest**:
   - Detects novel, zero-day threat patterns without labeled data.
   - High sensitivity on statistical outliers and multi-hop structure anomalies.
   - Fused composite score incorporates domain heuristics to mitigate false-positive rates on complex legal traffic.
2. **Supervised Random Forest Classifier**:
   - Cross-validated and probability-calibrated using isotonic regression (`CalibratedClassifierCV`).
   - Model weights and SHA-256 hash persisted to `models/rf_model.joblib` and `models/rf_model.sha256`.
3. **Multi-Signal Score Fusion**:
   - $\text{Risk Score} = w_{\text{ml}} \cdot S_{\text{ml}} + w_{\text{pat}} \cdot S_{\text{pattern}} + w_{\text{net}} \cdot S_{\text{network}} + w_{\text{graph}} \cdot S_{\text{graph}} - \text{Discount}_{\text{whitelist}}$
   - Fusion weights: ML Anomaly (50%), Pattern Signatures (25%), Network Risk (15%), Graph Taint (10%).

---

## 9. Tech Stack & Dependencies

| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.10+ | Scientific computing standard; runs natively offline. |
| **Data Processing** | Pandas, NumPy | High-performance in-memory tabular structures. |
| **Machine Learning** | Scikit-Learn | `IsolationForest`, `RandomForestClassifier`, `CalibratedClassifierCV`. |
| **Graph Analytics** | NetworkX | Union-Find address clustering, Louvain communities, PageRank taint. |
| **Clustering & PCA** | Scikit-Learn (HDBSCAN, PCA) | High-density spatial behavioural clustering. |
| **Document Generation** | ReportLab | Evidentiary PDF layout engine with two-pass running footers. |
| **Data Security** | defusedxml, hashlib | Fail-closed XXE parsing and SHA-256 provenance hashing. |
| **Geo-ASN Resolution** | maxminddb, geoip2 | Offline local MMDB lookup (no DNS / HTTP queries). |
| **Frontend UI** | Streamlit | Rapid air-gapped web interface. |
| **Network Visualization**| PyVis | Interactive physics-based HTML canvas rendering. |

---

## 10. Project Directory Layout

```text
SIH-2026-2/
├── app.py                      # Main Streamlit dashboard (7-tab architecture)
├── dossier.py                  # Court-ready PDF and JSON evidence dossier generator
├── flow_tracer.py              # BFS hop-by-hop fund flow tracer
├── clustering.py               # HDBSCAN behavioural clusters & Louvain communities
├── ml_engine.py                # Core ML pipeline, Union-Find clustering, taint engine
├── transaction_adapter.py      # Multi-format adapter (CSV, JSON, XML) & telemetry correlation
├── anomaly_engine.py           # Anomaly scoring and score fusion logic
├── features.py                 # Structural peel chain and pipe-delimited feature extraction
├── explainability.py           # Percentile-based statistical explainability
├── geo_asn.py                  # Offline DB-IP / MaxMind ASN and country resolver
├── generate_data.py            # Synthetic Bitcoin transaction & telemetry generator
├── evaluate_model.py           # ML validation harness (Train/Val/Test splits)
├── model_eval.py               # Supervised model training and calibration
├── models/
│   ├── rf_model.joblib         # Calibrated Random Forest model artifact
│   └── rf_model.sha256         # Model SHA-256 cryptographic provenance hash
├── evaluation/
│   └── benchmark.csv           # Scalability benchmark results (10k, 50k, 100k)
├── sample_data/
│   ├── network_telemetry.csv   # Standalone P2P broadcast telemetry sample
│   ├── blockchain_tx.csv       # Standalone blockchain ledger sample
│   ├── institutional_whitelist.csv # Regulated exchange and institutional addresses
│   └── watchlist.csv           # Threat-intel indicators (addresses, IPs, ASNs)
├── scripts/
│   ├── benchmark.py            # Automated multi-scale benchmark script
│   └── test_airgap.sh          # Linux network namespace air-gap verification
├── tests/                      # Full Pytest test suite (83 tests)
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

---

## 11. Test Suite & Validation

The test suite covers unit and integration tests across all modules:

```bash
# Run entire test suite
PYTHONPATH=. pytest tests/ -v

# Run with concise summary
PYTHONPATH=. pytest -q
```

**Status:**
- **83 passed** in 35 seconds (0 failures, 0 warnings, 0 skipped).
- Tests verify: XML XXE safety, billion-laughs mitigation, air-gap socket enforcement, Union-Find clustering, peel-chain heuristics, flow tracing, behavioural clustering, dossier PDF/JSON generation, and watchlist taint propagation.

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
./.venv/bin/streamlit run app.py
```
Open your browser at `http://localhost:8501`. Outbound socket calls are intercepted and blocked by `AirGapSocket`, guaranteeing complete privacy.

---

## 13. License & Disclaimer

Built for **Smart India Hackathon (SIH 2026) — Problem Statement 26146**.

**Forensic Disclaimer:** MITHYA is an investigative triage instrument, not an adjudicative verdict. Flagged transactions and entity clusters represent statistical, network, and graph anomalies designed to guide compliance officers, forensic auditors, and law enforcement analysts toward high-priority leads. Every lead must be verified through lawful process, KYC records, and corroborating evidence.
