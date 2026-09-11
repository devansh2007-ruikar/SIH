"""
generate_data.py — Synthetic Crypto-Forensic Data Generator
===========================================================

Generates realistic Bitcoin transaction datasets for the MITHYA triage engine.
Simulates distinct threat vectors (CoinJoins, Peel Chains, Fan-Out Dispersals,
Fee Spikes) and high-volume institutional traffic to test anomaly detection
and whitelisting.

Usage:
  python generate_data.py                                      # defaults
  python generate_data.py --total-records 2000 --suspicious-ratio 0.30

Outputs:
  - bitcoin_traffic.csv (Transaction graph data)
  - institutional_whitelist.csv (Pre-cleared entities)
"""

import argparse
import hashlib
import random
import string
import os
from datetime import datetime, timedelta

import pandas as pd
# pyrefly: ignore [missing-import]
from faker import Faker

# ---------------------------------------------------------------------------
# Configurable Hyperparameters
# ---------------------------------------------------------------------------
TOTAL_RECORDS = 1500
SUSPICIOUS_RATIO = 0.20

NUM_SUSPICIOUS = int(TOTAL_RECORDS * SUSPICIOUS_RATIO)
NUM_NORMAL = TOTAL_RECORDS - NUM_SUSPICIOUS

OUTPUT_CSV = "bitcoin_traffic.csv"
WHITELIST_CSV = "institutional_whitelist.csv"

SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

END_TIME = datetime(2026, 8, 26, 12, 0, 0)
START_TIME = END_TIME - timedelta(days=30)

# Port profiles
STANDARD_PORTS = [8333, 18333, 8332, 18332]
PROXY_PORTS = [9050, 9150, 1080, 3128, 443]  # Tor, SOCKS5, HTTP

# ---------------------------------------------------------------------------
# Whitelist Configuration
# ---------------------------------------------------------------------------
INSTITUTIONAL_ENTITIES = [
    {"name": "Binance_HotWallet", "addresses": ["1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s", "1BinanceHotXyz"]},
    {"name": "Coinbase_ColdStorage", "addresses": ["3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLsy", "3CoinbaseCold123"]},
    {"name": "F2Pool_Mining", "addresses": ["1F2PoolMiningXyz", "bc1qF2PoolReward"]},
    {"name": "Kraken_Sweep", "addresses": ["bc1qKrakenSweepXyz", "3KrakenDeposit"]},
]

def generate_whitelist_csv():
    """Generates the offline institutional whitelist."""
    rows = []
    for inst in INSTITUTIONAL_ENTITIES:
        for addr in inst["addresses"]:
            rows.append({"address": addr, "entity_name": inst["name"], "risk_override": 0.0})
    
    df = pd.DataFrame(rows)
    df.to_csv(WHITELIST_CSV, index=False)
    print(f"[*] Generated {WHITELIST_CSV} with {len(df)} cleared addresses.")

# ---------------------------------------------------------------------------
# Generators & Helpers
# ---------------------------------------------------------------------------
def random_btc_address(script_type=None) -> str:
    """Generates a random realistic Bitcoin address."""
    if not script_type:
        script_type = random.choice(["P2PKH", "P2SH", "P2WPKH"])
        
    prefix = "1" if script_type == "P2PKH" else "3" if script_type == "P2SH" else "bc1q"
    length = 33 if prefix in ("1", "3") else 38
    body = "".join(random.choices(string.ascii_letters + string.digits, k=length))
    return f"{prefix}{body}"

def random_timestamp() -> str:
    offset = random.random() * (END_TIME - START_TIME).total_seconds()
    return (START_TIME + timedelta(seconds=offset)).strftime("%Y-%m-%d %H:%M:%S")

def random_txid() -> str:
    return fake.sha256()

def format_amounts(amounts: list) -> str:
    return "|".join(f"{a:.8f}".rstrip("0").rstrip(".") if "." in f"{a:.8f}" else f"{a}" for a in amounts)

# ---------------------------------------------------------------------------
# Threat Vector Profiles
# ---------------------------------------------------------------------------

def create_coinjoin_mixer() -> dict:
    """
    Threat Vector 1: CoinJoin / Mixer
    Signature: High participant count, identical output amounts, Proxy/Tor ports.
    """
    participants = random.randint(10, 20)
    
    inputs = [random_btc_address() for _ in range(participants)]
    outputs = [random_btc_address() for _ in range(participants)]
    
    # Everyone puts in a random amount slightly above the mix denomination
    mix_denomination = random.choice([0.1, 0.5, 1.0, 5.0])
    fee = round(random.uniform(0.0005, 0.002), 8)
    
    input_amounts = [round(mix_denomination + fee + random.uniform(0.001, 0.05), 8) for _ in range(participants)]
    
    # The outputs are identical (the mix denomination)
    output_amounts = [mix_denomination for _ in range(participants)]
    
    # Add one random 'fee/change' collector output to balance the math
    remainder = sum(input_amounts) - sum(output_amounts) - fee
    if remainder > 0:
        outputs.append(random_btc_address())
        output_amounts.append(round(remainder, 8))
        
    return {
        "timestamp": random_timestamp(),
        "src_ip": fake.ipv4_public(),
        "dst_ip": fake.ipv4_public(),
        "src_port": random.choice(PROXY_PORTS),  # Tor/SOCKS5 indicator
        "dst_port": random.choice(STANDARD_PORTS),
        "txid": random_txid(),
        "input_addresses": "|".join(inputs),
        "output_addresses": "|".join(outputs),
        "input_amounts": format_amounts(input_amounts),
        "output_amounts": format_amounts(output_amounts),
        "fee": fee,
        "script_type": "P2WPKH",
        "geo_country": fake.country_code(),
        "is_labeled_suspicious": True,
        "attack_type": "CoinJoin_Mixer"
    }

def create_peel_chain() -> dict:
    """
    Threat Vector 2: Peel Chain
    Signature: Single input, one tiny peel output, one massive messy change output.
    """
    total_input = round(random.uniform(10.0, 50.0), 8)
    fee = round(random.uniform(0.0001, 0.0005), 8)
    
    # Peel a micro-fraction (< 1%)
    peel_amount = round(total_input * random.uniform(0.001, 0.005), 8)
    change_amount = round(total_input - peel_amount - fee, 8)
    
    script = random.choice(["P2PKH", "P2WPKH"])
    
    return {
        "timestamp": random_timestamp(),
        "src_ip": fake.ipv4_public(),
        "dst_ip": fake.ipv4_public(),
        "src_port": random.choice(STANDARD_PORTS + PROXY_PORTS),
        "dst_port": random.choice(STANDARD_PORTS),
        "txid": random_txid(),
        "input_addresses": random_btc_address(script),
        "output_addresses": f"{random_btc_address(script)}|{random_btc_address(script)}",
        "input_amounts": format_amounts([total_input]),
        "output_amounts": format_amounts([peel_amount, change_amount]),
        "fee": fee,
        "script_type": script,
        "geo_country": fake.country_code(),
        "is_labeled_suspicious": True,
        "attack_type": "Peel_Chain"
    }

def create_institutional_transfer() -> dict:
    """
    Normal Vector: Whitelisted Exchange Traffic
    Signature: High volume, uses known exchange addresses, normal ports.
    """
    institution = random.choice(INSTITUTIONAL_ENTITIES)
    inst_addr = random.choice(institution["addresses"])
    
    is_deposit = random.choice([True, False])
    amount = round(random.uniform(50.0, 500.0), 8)
    fee = round(random.uniform(0.0001, 0.001), 8)
    
    if is_deposit:
        # User -> Exchange
        inputs = [random_btc_address()]
        outputs = [inst_addr]
    else:
        # Exchange -> User
        inputs = [inst_addr]
        outputs = [random_btc_address()]

    return {
        "timestamp": random_timestamp(),
        "src_ip": fake.ipv4_public(),
        "dst_ip": fake.ipv4_public(),
        "src_port": random.choice(STANDARD_PORTS),
        "dst_port": random.choice(STANDARD_PORTS),
        "txid": random_txid(),
        "input_addresses": "|".join(inputs),
        "output_addresses": "|".join(outputs),
        "input_amounts": format_amounts([amount + fee]),
        "output_amounts": format_amounts([amount]),
        "fee": fee,
        "script_type": "P2WPKH",
        "geo_country": fake.country_code(),
        "is_labeled_suspicious": False,
        "attack_type": "Whitelisted_Institutional"
    }

def create_normal_traffic() -> dict:
    """Standard user-to-user P2P transaction."""
    amount = round(random.uniform(0.01, 2.0), 8)
    fee = round(random.uniform(0.00005, 0.0005), 8)
    
    num_inputs = random.randint(1, 3)
    input_amounts = []
    remaining = amount + fee
    for _ in range(num_inputs - 1):
        chunk = round(random.uniform(0.001, remaining / 2), 8)
        input_amounts.append(chunk)
        remaining -= chunk
    input_amounts.append(round(remaining, 8))
    
    return {
        "timestamp": random_timestamp(),
        "src_ip": fake.ipv4_public(),
        "dst_ip": fake.ipv4_public(),
        "src_port": random.choice(STANDARD_PORTS + list(range(49152, 65535))),
        "dst_port": random.choice(STANDARD_PORTS),
        "txid": random_txid(),
        "input_addresses": "|".join([random_btc_address() for _ in range(num_inputs)]),
        "output_addresses": random_btc_address(),
        "input_amounts": format_amounts(input_amounts),
        "output_amounts": format_amounts([amount]),
        "fee": fee,
        "script_type": random.choice(["P2PKH", "P2SH", "P2WPKH"]),
        "geo_country": fake.country_code(),
        "is_labeled_suspicious": False,
        "attack_type": "Normal_P2P"
    }

def create_fanout_dispersal() -> dict:
    """
    Threat Vector 3: Fan-Out Dispersal (Rapid Distribution)
    Signature: Single input rapidly split across many outputs.
    Triggers fan_ratio < 0.2 and fan_out > 5.
    """
    total_input = round(random.uniform(5.0, 30.0), 8)
    fee = round(random.uniform(0.001, 0.005), 8)
    num_outputs = random.randint(8, 15)

    outputs = [random_btc_address() for _ in range(num_outputs)]
    total_output = total_input - fee
    output_amounts = []
    remaining = total_output
    for _ in range(num_outputs - 1):
        amt = round(random.uniform(0.001, remaining / num_outputs * 2), 8)
        output_amounts.append(amt)
        remaining -= amt
    output_amounts.append(round(max(0.0, remaining), 8))

    return {
        "timestamp": random_timestamp(),
        "src_ip": fake.ipv4_public(),
        "dst_ip": fake.ipv4_public(),
        "src_port": random.choice(PROXY_PORTS + list(range(49152, 65535))),
        "dst_port": random.choice(STANDARD_PORTS),
        "txid": random_txid(),
        "input_addresses": random_btc_address(),
        "output_addresses": "|".join(outputs),
        "input_amounts": format_amounts([total_input]),
        "output_amounts": format_amounts(output_amounts),
        "fee": fee,
        "script_type": "P2WPKH",
        "geo_country": fake.country_code(),
        "is_labeled_suspicious": True,
        "attack_type": "FanOut_Dispersal"
    }

def create_fee_spike() -> dict:
    """
    Threat Vector 4: Fee-Spike Urgency
    Signature: Disproportionately high fee relative to input value.
    Triggers fee_rate_urgency > 0.05 (fee = 8-20% of input).
    """
    total_input = round(random.uniform(0.01, 0.5), 8)
    fee_ratio = random.uniform(0.08, 0.20)  # 8-20% fee rate
    fee = round(total_input * fee_ratio, 8)
    output_amount = round(total_input - fee, 8)

    return {
        "timestamp": random_timestamp(),
        "src_ip": fake.ipv4_public(),
        "dst_ip": fake.ipv4_public(),
        "src_port": random.choice(PROXY_PORTS + STANDARD_PORTS),
        "dst_port": random.choice(STANDARD_PORTS),
        "txid": random_txid(),
        "input_addresses": random_btc_address(),
        "output_addresses": random_btc_address(),
        "input_amounts": format_amounts([total_input]),
        "output_amounts": format_amounts([output_amount]),
        "fee": fee,
        "script_type": random.choice(["P2PKH", "P2WPKH"]),
        "geo_country": fake.country_code(),
        "is_labeled_suspicious": True,
        "attack_type": "Fee_Spike"
    }

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------
def main(total_records: int = TOTAL_RECORDS, suspicious_ratio: float = SUSPICIOUS_RATIO):
    num_suspicious = int(total_records * suspicious_ratio)
    num_normal = total_records - num_suspicious

    print(f"[*] Initializing Synthetic Data Generator (Records: {total_records})")
    
    # 1. Generate Whitelist
    generate_whitelist_csv()
    
    # 2. Generate Transactions
    transactions = []
    
    # Generate Suspicious — distribute across all 4 threat vectors
    for _ in range(num_suspicious):
        roll = random.random()
        if roll < 0.25:
            transactions.append(create_coinjoin_mixer())
        elif roll < 0.50:
            transactions.append(create_peel_chain())
        elif roll < 0.75:
            transactions.append(create_fanout_dispersal())
        else:
            transactions.append(create_fee_spike())
            
    # Generate Normal (Standard and Institutional)
    for _ in range(num_normal):
        if random.random() < 0.15:
            transactions.append(create_institutional_transfer())
        else:
            transactions.append(create_normal_traffic())
            
    # 3. Shuffle and Save
    random.shuffle(transactions)
    df = pd.DataFrame(transactions)
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"[*] Generated {len(df)} transactions -> Saved to '{OUTPUT_CSV}'.")
    print(f"[*] Threat Vectors injected: {num_suspicious} ({suspicious_ratio*100:.1f}%)")
    
    breakdown = df["attack_type"].value_counts()
    print("\n--- Breakdown ---")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")
    print("-----------------\n[✓] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MITHYA Synthetic Data Generator")
    parser.add_argument("--total-records", type=int, default=TOTAL_RECORDS,
                        help=f"Total number of transactions to generate (default: {TOTAL_RECORDS})")
    parser.add_argument("--suspicious-ratio", type=float, default=SUSPICIOUS_RATIO,
                        help=f"Fraction of suspicious transactions (default: {SUSPICIOUS_RATIO})")
    args = parser.parse_args()
    main(total_records=args.total_records, suspicious_ratio=args.suspicious_ratio)
