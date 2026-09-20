"""
generate_data.py — Synthetic Crypto-Forensic Data Generator
===========================================================

Generates realistic Bitcoin transaction datasets for the MITHYA triage engine.
Simulates distinct threat vectors:

- **CoinJoin / Mixer** — high participant count, identical outputs, Tor ports
- **Multi-Hop Peel Chains** — coordinated 3–5 hop sequences with address
  linkability, timestamp continuity, and progressive fund diminishment
- **Fan-Out Dispersal** — rapid 1-to-many distribution
- **Fee Spikes** — disproportionately high miner fees

Also generates high-volume institutional traffic for whitelist testing.

Usage
-----
::

    python generate_data.py                                         # defaults
    python generate_data.py --total-records 2000 --suspicious-ratio 0.30
    python generate_data.py --peel-depth-min 4 --peel-depth-max 6

Outputs
-------
- ``synthetic_transactions.csv``       — full transaction dataset
- ``synthetic_transactions_sample.json``  — first 50 records as JSON
- ``synthetic_transactions_sample.xml``   — first 50 records as XML
- ``institutional_whitelist.csv``       — pre-cleared entity addresses
"""

import argparse
import hashlib
import json
import os
import random
import string
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom

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

OUTPUT_CSV = "synthetic_transactions.csv"
OUTPUT_JSON = "synthetic_transactions_sample.json"
OUTPUT_XML = "synthetic_transactions_sample.xml"
WHITELIST_CSV = "institutional_whitelist.csv"

# Multi-hop peel chain depth bounds
PEEL_CHAIN_DEPTH_MIN = 3
PEEL_CHAIN_DEPTH_MAX = 5

SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

END_TIME = datetime(2026, 8, 26, 12, 0, 0)
START_TIME = END_TIME - timedelta(days=30)

# Port profiles
STANDARD_PORTS = [8333, 18333, 8332, 18332]
PROXY_PORTS = [9050, 9150, 4444]  # Tor SOCKS, I2P (aligned with ml_engine)

# ---------------------------------------------------------------------------
# Whitelist Configuration
# ---------------------------------------------------------------------------
INSTITUTIONAL_ENTITIES = [
    {"name": "Binance_HotWallet",    "addresses": ["1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s", "1BinanceHotXyz"]},
    {"name": "Coinbase_ColdStorage", "addresses": ["3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLsy", "3CoinbaseCold123"]},
    {"name": "F2Pool_Mining",        "addresses": ["1F2PoolMiningXyz", "bc1qF2PoolReward"]},
    {"name": "Kraken_Sweep",         "addresses": ["bc1qKrakenSweepXyz", "3KrakenDeposit"]},
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
    return "|".join(
        f"{a:.8f}".rstrip("0").rstrip(".") if "." in f"{a:.8f}" else f"{a}"
        for a in amounts
    )


def _sequential_timestamp(base: datetime, hop_index: int) -> Tuple[str, datetime]:
    """
    Generate a timestamp for hop *hop_index* that is 1–30 minutes
    after the base time.  Returns both the formatted string and the
    new datetime for chaining.

    Uses cumulative offset to guarantee strict monotonic ordering.
    """
    offset_seconds = random.randint(60, 1800)
    ts = base + timedelta(seconds=offset_seconds)
    return ts.strftime("%Y-%m-%d %H:%M:%S"), ts


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
        "src_port": random.choice(PROXY_PORTS),
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
        "attack_type": "CoinJoin_Mixer",
        "chain_id": None,
        "chain_hop": None,
        "remaining_balance": None,
        "change_address": None,
        "peel_amount": None,
        "peel_destination_type": None,
    }


def create_peel_chain_sequence(
    depth_min: int = PEEL_CHAIN_DEPTH_MIN,
    depth_max: int = PEEL_CHAIN_DEPTH_MAX,
    terminal_to_exchange: bool = True,
) -> List[dict]:
    """
    Threat Vector 2: Multi-Hop Peel Chain
    ======================================

    Synthesizes a coordinated sequence of 3–5 linked transactions that
    model a realistic peel-chain laundering pattern:

    ::

        TX_1: Stolen Fund (25 BTC)
              → Peel₁ (0.2 BTC to cashout)  +  Change Addr A (24.799 BTC)
        TX_2: Change Addr A
              → Peel₂ (0.3 BTC to mixer)    +  Change Addr B (24.498 BTC)
        TX_3: Change Addr B
              → Terminal transfer to institutional exchange

    **Invariants maintained**:
    - Change address of TX_n is the sole input of TX_{n+1}
    - Timestamps are strictly monotonic (1–30 min between hops)
    - Same script type throughout the chain
    - Progressive fund diminishment across hops
    - Unique ``chain_id`` tags all transactions in the sequence

    Parameters
    ----------
    depth_min : int
        Minimum number of hops (default 3).
    depth_max : int
        Maximum number of hops (default 5).
    terminal_to_exchange : bool
        If True, the final hop sends remaining funds to a random
        institutional exchange address.

    Returns
    -------
    list[dict]
        A list of transaction dicts (one per hop).
    """
    depth = random.randint(depth_min, depth_max)
    chain_id = f"peel_{uuid.uuid4().hex[:12]}"
    script = random.choice(["P2PKH", "P2WPKH"])

    # Initial stolen fund: 10–50 BTC
    initial_amount = round(random.uniform(10.0, 50.0), 8)

    # Pick a random base time within the dataset window
    base_offset = random.random() * (END_TIME - START_TIME).total_seconds()
    # Leave room for the chain to complete (max ~2.5 hours for 5 hops)
    base_offset = min(base_offset, (END_TIME - START_TIME).total_seconds() - 10800)
    base_time = START_TIME + timedelta(seconds=max(0, base_offset))

    # The initial input address (the "thief's" wallet)
    current_input_addr = random_btc_address(script)
    current_balance = initial_amount

    # Persistent IP for the sender (realistic: same machine across hops)
    sender_ip = fake.ipv4_public()

    transactions: List[dict] = []

    # Peel destinations for intermediate hops
    peel_destinations = [
        "cashout_p2p",
        "mixer_service",
        "gambling_site",
        "darknet_market",
        "privacy_wallet",
    ]

    for hop in range(depth):
        is_final_hop = (hop == depth - 1)
        fee = round(random.uniform(0.0001, 0.001), 8)

        # Cumulative timestamp: each hop advances base_time
        ts_str, base_time = _sequential_timestamp(base_time, hop)

        if is_final_hop:
            # ── Terminal hop: send remaining balance to exchange ──
            if terminal_to_exchange:
                institution = random.choice(INSTITUTIONAL_ENTITIES)
                terminal_addr = random.choice(institution["addresses"])
            else:
                terminal_addr = random_btc_address(script)

            output_amount = round(current_balance - fee, 8)
            if output_amount <= 0:
                break  # balance exhausted

            tx = {
                "timestamp": ts_str,
                "src_ip": sender_ip,
                "dst_ip": fake.ipv4_public(),
                "src_port": random.choice(STANDARD_PORTS + PROXY_PORTS),
                "dst_port": random.choice(STANDARD_PORTS),
                "txid": random_txid(),
                "input_addresses": current_input_addr,
                "output_addresses": terminal_addr,
                "input_amounts": format_amounts([current_balance]),
                "output_amounts": format_amounts([output_amount]),
                "fee": fee,
                "script_type": script,
                "geo_country": fake.country_code(),
                "is_labeled_suspicious": True,
                "attack_type": "Peel_Chain",
                "chain_id": chain_id,
                "chain_hop": hop + 1,
                "remaining_balance": 0.0,
                "change_address": None,
                "peel_amount": round(output_amount, 8),
                "peel_destination_type": "institutional_exchange",
            }
            transactions.append(tx)

        else:
            # ── Intermediate hop: peel off a small amount ────────
            # Peel amount: 0.5%–3% of current balance
            peel_fraction = random.uniform(0.005, 0.03)
            peel_amount = round(current_balance * peel_fraction, 8)
            change_amount = round(current_balance - peel_amount - fee, 8)

            if change_amount <= 0:
                break  # balance exhausted

            peel_addr = random_btc_address(script)
            change_addr = random_btc_address(script)

            peel_dest = random.choice(peel_destinations)

            tx = {
                "timestamp": ts_str,
                "src_ip": sender_ip,
                "dst_ip": fake.ipv4_public(),
                "src_port": random.choice(STANDARD_PORTS + PROXY_PORTS),
                "dst_port": random.choice(STANDARD_PORTS),
                "txid": random_txid(),
                "input_addresses": current_input_addr,
                "output_addresses": f"{peel_addr}|{change_addr}",
                "input_amounts": format_amounts([current_balance]),
                "output_amounts": format_amounts([peel_amount, change_amount]),
                "fee": fee,
                "script_type": script,
                "geo_country": fake.country_code(),
                "is_labeled_suspicious": True,
                "attack_type": "Peel_Chain",
                "chain_id": chain_id,
                "chain_hop": hop + 1,
                "remaining_balance": round(change_amount, 8),
                "change_address": change_addr,
                "peel_amount": round(peel_amount, 8),
                "peel_destination_type": peel_dest,
            }
            transactions.append(tx)

            # Chain linkage: change address becomes next input
            current_input_addr = change_addr
            current_balance = change_amount

    return transactions


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
        "attack_type": "Whitelisted_Institutional",
        "chain_id": None,
        "chain_hop": None,
        "remaining_balance": None,
        "change_address": None,
        "peel_amount": None,
        "peel_destination_type": None,
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
        "attack_type": "Normal_P2P",
        "chain_id": None,
        "chain_hop": None,
        "remaining_balance": None,
        "change_address": None,
        "peel_amount": None,
        "peel_destination_type": None,
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
        "attack_type": "FanOut_Dispersal",
        "chain_id": None,
        "chain_hop": None,
        "remaining_balance": None,
        "change_address": None,
        "peel_amount": None,
        "peel_destination_type": None,
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
        "attack_type": "Fee_Spike",
        "chain_id": None,
        "chain_hop": None,
        "remaining_balance": None,
        "change_address": None,
        "peel_amount": None,
        "peel_destination_type": None,
    }


# ---------------------------------------------------------------------------
# Export Helpers
# ---------------------------------------------------------------------------

def export_json_sample(df: pd.DataFrame, filepath: str, n: int = 50):
    """Export the first *n* records as a JSON array of objects."""
    sample = df.head(n).copy()
    # Convert NaN chain_id / chain_hop to None for clean JSON
    sample["chain_id"] = sample["chain_id"].where(sample["chain_id"].notna(), None)
    sample["chain_hop"] = sample["chain_hop"].where(sample["chain_hop"].notna(), None)

    records = sample.to_dict(orient="records")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"[*] Exported {len(records)} sample records → '{filepath}'")


def export_xml_sample(df: pd.DataFrame, filepath: str, n: int = 50):
    """Export the first *n* records as XML."""
    sample = df.head(n)

    root = ET.Element("transactions")

    for _, row in sample.iterrows():
        tx = ET.SubElement(root, "transaction")
        for col in sample.columns:
            el = ET.SubElement(tx, col)
            value = row[col]
            if pd.isna(value):
                el.text = ""
            else:
                el.text = str(value)

    # Pretty-print with minidom
    rough = ET.tostring(root, encoding="unicode")
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent="  ", encoding="UTF-8")

    with open(filepath, "wb") as f:
        f.write(pretty)
    print(f"[*] Exported {len(sample)} sample records → '{filepath}'")


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------
def main(
    total_records: int = TOTAL_RECORDS,
    suspicious_ratio: float = SUSPICIOUS_RATIO,
    peel_depth_min: int = PEEL_CHAIN_DEPTH_MIN,
    peel_depth_max: int = PEEL_CHAIN_DEPTH_MAX,
    output_csv: str = OUTPUT_CSV,
    terminal_to_exchange: bool = True,
):
    num_suspicious = int(total_records * suspicious_ratio)
    num_normal = total_records - num_suspicious

    print(f"[*] Initializing Synthetic Data Generator (Records: {total_records})")
    print(f"    Peel chain depth: {peel_depth_min}–{peel_depth_max} hops")

    # 1. Generate Whitelist
    generate_whitelist_csv()

    # 2. Generate Transactions
    transactions: List[dict] = []
    peel_chains_generated = 0

    # ── Generate Suspicious — distribute across 4 threat vectors ──
    suspicious_generated = 0
    while suspicious_generated < num_suspicious:
        roll = random.random()

        if roll < 0.25:
            transactions.append(create_coinjoin_mixer())
            suspicious_generated += 1

        elif roll < 0.50:
            # Multi-hop peel chain: generates multiple linked records
            chain = create_peel_chain_sequence(
                depth_min=peel_depth_min,
                depth_max=peel_depth_max,
                terminal_to_exchange=terminal_to_exchange,
            )
            transactions.extend(chain)
            suspicious_generated += len(chain)
            peel_chains_generated += 1

        elif roll < 0.75:
            transactions.append(create_fanout_dispersal())
            suspicious_generated += 1

        else:
            transactions.append(create_fee_spike())
            suspicious_generated += 1

    # ── Generate Normal (Standard and Institutional) ──
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

    df.to_csv(output_csv, index=False)

    print(f"\n[*] Generated {len(df)} transactions → Saved to '{output_csv}'.")
    print(f"[*] Peel chains: {peel_chains_generated} chains")
    print(f"[*] Threat Vectors injected: {suspicious_generated} ({suspicious_ratio*100:.1f}% target)")

    breakdown = df["attack_type"].value_counts()
    print("\n--- Breakdown ---")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")

    # ── Chain linkage statistics ──
    chain_df = df[df["chain_id"].notna()]
    if len(chain_df) > 0:
        chain_sizes = chain_df.groupby("chain_id").size()
        print(f"\n--- Peel Chain Statistics ---")
        print(f"  Total chains: {len(chain_sizes)}")
        print(f"  Total chain txs: {len(chain_df)}")
        print(f"  Avg chain depth: {chain_sizes.mean():.1f}")
        print(f"  Min/Max depth: {chain_sizes.min()}/{chain_sizes.max()}")

    # 4. Export JSON & XML samples
    json_path = os.path.splitext(output_csv)[0] + "_sample.json"
    xml_path = os.path.splitext(output_csv)[0] + "_sample.xml"
    export_json_sample(df, json_path)
    export_xml_sample(df, xml_path)

    print(f"\n-----------------\n[✓] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MITHYA Synthetic Data Generator")
    parser.add_argument("--total-records", type=int, default=TOTAL_RECORDS,
                        help=f"Total number of transactions to generate (default: {TOTAL_RECORDS})")
    parser.add_argument("--suspicious-ratio", type=float, default=SUSPICIOUS_RATIO,
                        help=f"Fraction of suspicious transactions (default: {SUSPICIOUS_RATIO})")
    parser.add_argument("--peel-depth-min", type=int, default=PEEL_CHAIN_DEPTH_MIN,
                        help=f"Minimum peel chain depth (default: {PEEL_CHAIN_DEPTH_MIN})")
    parser.add_argument("--peel-depth-max", type=int, default=PEEL_CHAIN_DEPTH_MAX,
                        help=f"Maximum peel chain depth (default: {PEEL_CHAIN_DEPTH_MAX})")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    parser.add_argument("--no-terminal-exchange", action="store_true",
                        help="Don't route final hop to institutional exchange")
    args = parser.parse_args()
    main(
        total_records=args.total_records,
        suspicious_ratio=args.suspicious_ratio,
        peel_depth_min=args.peel_depth_min,
        peel_depth_max=args.peel_depth_max,
        output_csv=args.output,
        terminal_to_exchange=not args.no_terminal_exchange,
    )
