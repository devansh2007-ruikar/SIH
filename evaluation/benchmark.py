import os
import sys
import time
import tracemalloc
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transaction_adapter import get_adapter

def run_benchmark():
    """Benchmarks runtime execution speed and peak memory footprint of the core ML pipeline."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bitcoin_traffic.csv")
    
    if not os.path.exists(data_path):
        print(f"[!] Benchmark dataset not found at {data_path}. Please generate it first.")
        return

    print(f"[*] Loading dataset: {data_path}")
    adapter = get_adapter(data_path)
    raw_df = adapter.load(data_path)
    
    total_records = len(raw_df)
    print(f"[*] Total Records: {total_records}")
    
    print("[*] Starting benchmark...")
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Run pipeline
    enriched_df, model, features_df = adapter.run_pipeline_from_df(raw_df, contamination=0.05)
    
    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    exec_time = end_time - start_time
    peak_mem_mb = peak_mem / 1024 / 1024
    
    flagged = int(enriched_df["is_anomaly"].sum())
    
    print("-" * 50)
    print("BENCHMARK RESULTS")
    print("-" * 50)
    print(f"Total Transactions : {total_records}")
    print(f"Anomalies Flagged  : {flagged} ({(flagged/total_records)*100:.1f}%)")
    print(f"Execution Time     : {exec_time:.4f} seconds")
    print(f"Throughput         : {total_records / exec_time:.2f} tx/sec")
    print(f"Peak Memory Usage  : {peak_mem_mb:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    run_benchmark()
