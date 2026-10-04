"""
scripts/benchmark.py — Scalability Benchmark for MITHYA Crypto-Forensic Pipeline
================================================================================

Generates 10k, 50k, and 100k synthetic transaction datasets, profiles execution
latency for each distinct pipeline stage, and exports performance metrics to
evaluation/benchmark.csv.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --scales 10k,50k,100k
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from generate_data import generate_transactions_df
from transaction_adapter import BitcoinCSVAdapter
import ml_engine


def benchmark_pipeline_scale(
    target_records: int,
    scale_label: str,
    seed: int = 42,
) -> Dict[str, float | str | int]:
    """
    Generate dataset of target size and time every stage of the MITHYA triage pipeline.
    """
    print(f"\n{'=' * 70}", flush=True)
    print(f"[*] BENCHMARK SCALE: {scale_label} ({target_records:,} transactions)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Baseline memory
    mem_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # ── Stage 0: Synthetic Data Generation ──
    print(f"[*] [1/7] Generating {target_records:,} synthetic transactions...", flush=True)
    t0 = time.perf_counter()
    df = generate_transactions_df(total_records=target_records, seed=seed)
    gen_time = time.perf_counter() - t0
    actual_records = len(df)
    print(f"    ↳ Generated {actual_records:,} records in {gen_time:.2f}s", flush=True)

    # ── Stage 1: Ingestion & Normalisation ──
    print("[*] [2/7] Address normalisation and timestamp parsing...", flush=True)
    adapter = BitcoinCSVAdapter()
    t0 = time.perf_counter()
    df = adapter.normalize_addresses(df)
    if "geo_country" not in df.columns:
        df["geo_country"] = "XX"
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    norm_time = time.perf_counter() - t0
    print(f"    ↳ Normalised in {norm_time:.2f}s", flush=True)

    # ── Stage 2: Feature Engineering & Structural Classification ──
    print("[*] [3/7] Feature engineering and structural pattern classification...", flush=True)
    t0 = time.perf_counter()
    features, df = adapter.extract_features(df)
    df["detected_type"] = ml_engine.classify_structural_type(df, features)
    feat_time = time.perf_counter() - t0
    print(f"    ↳ Engineered {features.shape[1]} features in {feat_time:.2f}s", flush=True)

    # ── Stage 3: Unsupervised Isolation Forest Training ──
    print("[*] [4/7] Training Isolation Forest model...", flush=True)
    t0 = time.perf_counter()
    model, predictions, raw_scores = ml_engine.train_model(features, contamination=0.05)
    train_time = time.perf_counter() - t0
    print(f"    ↳ Model trained in {train_time:.2f}s", flush=True)

    # ── Stage 4: Risk Scoring & Percentile Explanations ──
    print("[*] [5/7] Computing anomaly risk scores and forensic explainability...", flush=True)
    t0 = time.perf_counter()
    risk_scores = ml_engine.compute_risk_scores(raw_scores)
    dataset_profile = ml_engine._build_dataset_profile(features)
    enriched_df = ml_engine.generate_all_explanations(
        df, features, predictions, risk_scores, dataset_profile,
    )
    score_time = time.perf_counter() - t0
    print(f"    ↳ Scored & explained in {score_time:.2f}s", flush=True)

    # ── Stage 5: Whitelist Clearance & Watchlist Taint Propagation ──
    print("[*] [6/7] Applying institutional whitelist and graph taint propagation...", flush=True)
    t0 = time.perf_counter()
    enriched_df = ml_engine.apply_institutional_whitelist(enriched_df)
    ml_engine.finalize_detected_type(enriched_df)
    watchlist_dict = ml_engine.load_watchlist()
    ml_engine.apply_watchlist(enriched_df, watchlist_dict)
    ml_engine.compute_taint(enriched_df, watchlist_dict)
    taint_time = time.perf_counter() - t0
    print(f"    ↳ Whitelist & taint propagation in {taint_time:.2f}s", flush=True)

    # ── Stage 6: Multi-Signal Score Fusion ──
    print("[*] [7/7] Computing fused investigative priority...", flush=True)
    t0 = time.perf_counter()
    ml_engine.compute_fused_priority(enriched_df, features)
    fuse_time = time.perf_counter() - t0
    print(f"    ↳ Fused score computed in {fuse_time:.2f}s", flush=True)

    mem_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    peak_mb = round(mem_end, 2)

    total_pipeline_s = round(norm_time + feat_time + train_time + score_time + taint_time + fuse_time, 2)
    throughput = round(actual_records / total_pipeline_s, 1) if total_pipeline_s > 0 else 0.0

    print(f"\n[✓] {scale_label} COMPLETED:", flush=True)
    print(f"    Total Pipeline Duration: {total_pipeline_s:.2f}s", flush=True)
    print(f"    Throughput:              {throughput:,} tx/s", flush=True)
    print(f"    Peak Memory:             {peak_mb:.2f} MB", flush=True)

    return {
        "scale": scale_label,
        "records": actual_records,
        "generation_s": round(gen_time, 2),
        "normalisation_s": round(norm_time, 2),
        "feature_engineering_s": round(feat_time, 2),
        "model_training_s": round(train_time, 2),
        "scoring_expl_s": round(score_time, 2),
        "whitelist_taint_s": round(taint_time, 2),
        "fusion_s": round(fuse_time, 2),
        "total_pipeline_s": total_pipeline_s,
        "throughput_tx_per_s": throughput,
        "peak_memory_mb": peak_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="MITHYA Scalability Benchmark")
    parser.add_argument(
        "--scales",
        type=str,
        default="10k,50k,100k",
        help="Comma-separated scale targets (e.g. 10k,50k,100k)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/benchmark.csv",
        help="Output CSV filepath",
    )
    args = parser.parse_args()

    scale_map = {
        "10k": 10_000,
        "50k": 50_000,
        "100k": 100_000,
    }

    scale_keys = [s.strip().lower() for s in args.scales.split(",") if s.strip()]
    results: List[Dict[str, float | str | int]] = []

    for key in scale_keys:
        count = scale_map.get(key)
        if count is None:
            try:
                count = int(key.replace("k", "000").replace("m", "000000"))
            except ValueError:
                print(f"[!] Unrecognized scale '{key}', skipping.", flush=True)
                continue

        result = benchmark_pipeline_scale(count, key.upper())
        results.append(result)

    out_df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n[★] Benchmark results written to {out_path.resolve()}", flush=True)
    print("\nSummary Results:", flush=True)
    print(out_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
