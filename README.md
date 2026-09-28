# MITHYA — Enterprise-Grade Crypto-Forensics & Triage Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hackathon](https://img.shields.io/badge/SIH-2026-ff69b4.svg)](https://sih.gov.in/)
[![Security](https://img.shields.io/badge/XML-defusedxml%20XXE--safe-green.svg)](https://pypi.org/project/defusedxml/)
[![Air-Gap](https://img.shields.io/badge/Air--Gap-Verified-brightgreen.svg)](#phase-2b-air-gap-proof--scriptstest_airgapsh)
[![ML Validated](https://img.shields.io/badge/ML-Benchmarked%20Precision%2FRecall%2FF1%2FFPR-blue.svg)](#phase-1-ml-validation--benchmarking--evaluate_modelpy)

**MITHYA** is an offline, graph-aware cryptocurrency forensics and triage engine built for **Smart India Hackathon 2026 — Problem Statement 26146**. It ingests raw network/blockchain traffic, clusters wallet entities using Evidence-Based Confidence Tiers, and scores transactions for illicit behaviour using unsupervised machine learning — fully air-gapped, with no external API or CDN dependencies.

> **Security & Forensic Hardening Notice (v2.0):** This release implements major enterprise-grade security and ML upgrades based on expert forensic feedback, including cryptographic data provenance (SHA-256 hashing), XXE-safe fail-closed XML parsing, air-gap namespace verification, True offline Geo-IP/ASN resolution (DB-IP/MaxMind), ML benchmarking with hard negatives, and evidence-based clustering tiers. See [Hardening Changelog](#7-hardening-changelog-v20) for details.

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
| **Behavioral ML** | 18 engineered features fed into `IsolationForest` to generate an objective **Anomaly Deviation Score** and **Investigative Priority Index**. |
| **Evidence-Based Clustering** | NetworkX nodes are grouped into Evidence-Based Confidence Tiers (*High-Confidence*, *Mixer-Affected*, *Heuristic/Inferred*). |
| **ML Validation Suite** | Dedicated `evaluate_model.py` with strict Train/Validation/Test splits benchmarking against rules-only baselines and hard negatives. |
| **Cryptographic Provenance** | Ingested files are hashed in-memory (SHA-256). Exported dossiers include the source file hash, ingestion timestamp, and model version. |
| **XXE-Safe XML Ingestion** | `defusedxml` replaces stdlib `xml.etree` — fail-closed architecture prevents billion-laughs and XXE injection attacks. |
| **Air-Gap Verification** | `scripts/test_airgap.sh` cryptographically isolates and proves 100% offline execution via Linux network namespaces (`unshare -r -n`). |
| **Offline Geo-ASN Enrichment** | Local offline DB-IP / MaxMind Lite integration (`geo_asn.py`) resolves IPs to real Autonomous System Numbers with no data fabrication. |
| **Forensic Disclaimers** | Persistent warning banners on all network-telemetry views: "Not absolute identity attribution (Subject to VPN/NAT/Tor limits)." |
| **Advanced 4-Tab UI** | Streamlit dashboard: Overview, Entity Attribution (+ASN column), PyVis Graph, Analytical Explainability Panel. |

---

## 3. System Architecture & Data Flow

```
+---------------------+       +----------------------+       +-----------------------+
|       STAGE 1       |       |       STAGE 2        |       |       STAGE 3         |
|   DATA INGESTION    |------>|     AI/ML ENGINE     |------>|      DASHBOARD        |
|                     |       |                      |       |                       |
| transaction_adapter |       |     ml_engine.py     |       |       app.py          |
| .py (defusedxml)    |       |                      |       |                       |
| * Fail-Closed XML   |       | * Mixer bypass       |       | * Pyvis network graph |
| * SHA-256 Hashing   |       | * Change-addr detect |       | * Cluster tiers       |
| * Memory buffering  |       | * Tiered clustering  |       | * Forensic export     |
|                     |       | * IsolationForest    |       | * Metric cards        |
+---------------------+       +----------------------+       +-----------------------+
         |                            |
  [ bitcoin_traffic.csv ]  [ flagged_transactions.csv ]
         |                            |
         +----------------------------+
                      |
           +------------------+     +----------------------+
           |   geo_asn.py     |     |  evaluate_model.py   |
           | Offline DB-IP /  |     |  ML Validation Suite |
           | MaxMind Lite     |     |  Precision/Recall/   |
           +------------------+     |  F1 / FPR / Memory   |
                                    +----------------------+
```

**Data Flow:**
1. `transaction_adapter.py` ingests files via the appropriate adapter; XML is parsed with `defusedxml` (XXE-safe). Ingested files are hashed in memory using `hashlib.sha256`.
2. `ml_engine.py` builds a NetworkX graph, filters mixers, clusters wallets into Evidence-Based Tiers, engineers 18 features, trains `IsolationForest`, and returns `(enriched_df, model, features_df)`.
3. `geo_asn.py` enriches the result with offline DB-IP `geo_country` + `asn` fields (zero network I/O, strictly offline).
4. `app.py` displays results with forensic disclaimers, cluster confidence labels, the `asn` column, and Court-Ready Evidence Dossiers.
5. `evaluate_model.py` independently benchmarks the model against baselines using a strict Train/Validation/Test setup.

---

## 4. Tech Stack & Design Rationale

| Library | Role | Rationale |
|---|---|---|
| **Python 3.10+** | Core Language | Standard for data science; natively supports offline execution. |
| **Pandas / NumPy** | Data Manipulation | High-performance memory structures for transaction matrices. |
| **Scikit-Learn** | ML Engine | `IsolationForest` (unsupervised anomaly detection) + `MinMaxScaler`. |
| **NetworkX** | Graph Clustering | Union-Find algorithm underpinning wallet ownership grouping. |
| **defusedxml** | XML Security | Fail-closed XML parser preventing XXE injection and billion-laughs attacks. |
| **maxminddb** | IP Enrichment | Real, true offline Autonomous System Number and Country-level resolution. |
| **Streamlit** | Dashboard UI | Rapid Python-native frontend; no CDN calls in air-gapped mode. |
| **PyVis** | Visualization | Physics-based (`forceAtlas2Based`) interactive graphs rendered in-browser. |

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
| `geo_country` | String | `US` | ISO 3166-1 alpha-2; resolved via MMDB |
| `asn` | String | `AS13335` | Resolved offline by `geo_asn.py` (MaxMind) |

> **Forensic caveat:** `src_ip`, `geo_country`, and `asn` identify the **network observation node only**. They are NOT the sender's physical identity. Results are subject to VPN/NAT/Tor limits.

---

## 6. AI/ML Pipeline Deep Dive

### IsolationForest Configuration
- **Model:** `sklearn.ensemble.IsolationForest`
- **n_estimators:** 200
- **Contamination Rate:** Default `0.05` (adjustable via UI slider)
- **Scoring:** `decision_function()` inverted and scaled to `[0, 100]` as **Anomaly Deviation Score**

### Investigative Priority Index
A composite score combining the Anomaly Deviation Score (70% weight) with a feature severity component (30% weight). **This is strictly an investigative aid, not a subjective "probability of criminality".**

---

## 7. Hardening Changelog (v2.0)

### Phase 1: ML Validation & Benchmarking — `evaluate_model.py`
A dedicated evaluation harness producing forensic-grade metrics:
- Implements strict **Train/Validation/Test split** (60/20/20).
- Evaluates against **three distinct datasets**: `benign_traffic`, `known_threats`, and `novel_threats`.
- The `benign_traffic` generator incorporates **hard negatives** (verified exchanges, mining pools, multi-sig treasuries) to robustly stress-test the model's false positive rate against institutional patterns.
- Output metrics include **hardware telemetry** (Median Processing Time, Peak Memory Usage) tailored for a 16GB RAM environment.
- F1-Score benchmarking is measured against a deterministic **rules-only baseline classifier**.

### Phase 2: Security & Air-Gap Proof
- **XML Parsing Hardening (`transaction_adapter.py`)**: `BitcoinXMLAdapter` now strictly uses `defusedxml` in a fail-closed architecture, completely removing fallbacks to standard `xml.etree`. This protects against XXE (XML External Entity) injections and Billion-Laughs attacks.
- **Air-Gap Verification (`scripts/test_airgap.sh`)**: We cryptographically isolate and prove 100% offline execution utilizing Linux network namespaces (`unshare -r -n`), demonstrating zero external dependencies or telemetry callbacks during analysis.

### Phase 3: Cryptographic Data Provenance (`app.py`)
- Ingested files (.csv, .json, .xml) are dynamically hashed in memory upon upload using `hashlib.sha256`.
- The UI exposes this file hash.
- **Court-Ready Evidence Dossiers** embed the source file hash, exact ingestion timestamp, host system configuration, and model version, enforcing strict data provenance and auditability for law enforcement standards.

### Phase 4: Evidence-Based Clustering & True Geo-ASN (`ml_engine.py` / `geo_asn.py`)
- **Evidence-Based Clustering**: NetworkX component clusters are assigned Confidence Tiers:
  - `High-Confidence`: Dense clusters (>3 transactions / 24h direct linkage).
  - `Mixer-Affected`: Overlaps with flagged CoinJoin mixer nodes.
  - `Heuristic/Inferred`: Standard common-input-ownership groupings.
- **Geo-IP/ASN Hardening**: `geo_asn.py` uses a local, offline DB-IP / MaxMind Lite MMDB reader (`maxminddb`). It enforces a strict fallback (`XX`, `AS0`, `Unknown`) for unmatched IPs, absolutely preventing the fabrication of plausible locations.

---

## 8. Explainable AI / Output Logic

The `explainability.py` module generates human-readable reasons by evaluating feature vectors against **dataset percentiles** — no hardcoded thresholds. It produces a statistical **Investigative Priority Index** and **Anomaly Deviation Score**, eliminating subjective terms like "Guilt Probability" or "Confidence Score."

- **Port Anomaly:** `"Anomalous port profile -- possible proxy/Tor/tunneling"`
- **True Multi-Hop Peel Chain:** `"Peel-chain output structure detected across sequential hops -- classic layering signature"`
- **Mass Consolidation:** Triggers based on percentile-ranked fan-out ratios.
- **Fee Urgency:** `"Fee-rate urgency -- miner-priority overpayment suggesting time-sensitive hop"`
- **IP Anonymization / Botnet Relay:** Flagged when `entity_ip_diversity` or `ip_entity_diversity` exceeds 95th percentile.

---

## 9. Dashboard & Visualization Guide

- **Dynamic Sidebar:** File uploaders (.csv, .json, .xml), whitelist upload, sensitivity slider, synthetic data generator, **SHA-256 Source Hash Display**.

- **Tab 1 — Overview & Triage:** 6 Bento Metric Cards, Threat Vector chart, Anomaly Score histogram, **Court-Ready Evidence Dossier Export**.

- **Tab 2 — Entity Attribution Viewer:** `Investigative Priority %`, `Entity`, `Cluster Confidence Tier`, `ASN`.

- **Tab 3 — Graph & Clusters:** PyVis graph with `forceAtlas2Based` physics. **Evidence-Based Confidence Labels** in node tooltips. Threat-centric mode toggle. Ego-network isolation (2 hops). 

- **Tab 4 — Analytical Explainability Panel:** Drill-down by Anomaly Score with percentile deviations. Geo-country + ASN panel with attribution caveat per transaction.

---

## 10. Project Structure

```text
SIH-2026-2/
├── generate_data.py            # Synthetic P2P traffic generator (CLI args)
├── ml_engine.py                # Core ML pipeline, tiered clustering, XAI
├── transaction_adapter.py      # Cross-format adapter (CSV, JSON, XML via defusedxml)
├── anomaly_engine.py           # Anomaly Deviation Score + Investigative Priority Index
├── features.py                 # Peel-chain and pipe-parsing feature extraction
├── explainability.py           # Percentile-based XAI report generation
├── app.py                      # Streamlit 4-tab UI, Provenance hashing, Evidence Dossier
├── geo_asn.py                  # Offline DB-IP / MaxMind IP resolver (maxminddb)
├── evaluate_model.py           # ML validation harness (Train/Val/Test, Hardware metrics)
├── requirements.txt            # pip dependencies (includes defusedxml, maxminddb)
├── requirements-lock.txt       # Strict locked dependencies for offline reproducibility
├── Dockerfile                  # Container for air-gapped deployment
├── scripts/
│   ├── test_airgap.sh          # Air-gap proof via Linux unshare -r -n
│   └── run_demo.sh             # [NEW] One-click demonstration runner
├── tests/                      # Pytest suite (XXE, Billion-Laughs, heuristics, airgap)
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

### 2. Run the One-Click Demo Dashboard

```bash
./run_demo.sh
# Access: http://localhost:8501
```

### 3. Run ML Validation

```bash
python evaluate_model.py --records 500 --verbose
```

### 4. Test Offline Geo-ASN Resolver

```bash
python geo_asn.py
```

### 5. Run Air-Gap Proof

```bash
bash scripts/test_airgap.sh --verbose
```

### 6. Run Test Suite (Security & Heuristics)

```bash
pytest tests/ -v
```

---

## 12. Future Roadmap

- **Ethereum Integration** — `EthereumAdapter` for account/nonce-based anomaly tracing.
- **Real-time Mempool Ingestion** — Subscribe to Bitcoin Core RPC node for live analysis.
- **Temporal Velocity Features** — Burst-sending pattern detection via sliding-window analysis.
- **SHAP Integration** — Replace percentile explanations with SHAP values for IsolationForest feature contributions.
- **Monero Adapter** — Ring-signature transaction parsing and stealth address decoding.
