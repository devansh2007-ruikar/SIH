"""
ml_engine.py — Graph-Aware AI Anomaly Detection Engine
======================================================

Implements a full data pipeline:
1. Constructs an Entity/Transaction graph using NetworkX.
2. Performs Wallet Clustering via Common-Input-Ownership (Union-Find).
3. Extracts graph-level and network-layer correlation features.
4. Trains an IsolationForest to detect anomalous graph behavior (layering).
5. Generates human-readable explanations (XAI) for flagged threats.

Can be run standalone via CLI, imported into Streamlit, or invoked
through a :class:`~transaction_adapter.BaseTransactionAdapter` for
cross-chain polymorphic usage.
"""

import os
import warnings
from typing import Tuple, List, Dict, Set

import pandas as pd
import numpy as np
import networkx as nx
# pyrefly: ignore [missing-import]
from sklearn.ensemble import IsolationForest
# pyrefly: ignore [missing-import]
from sklearn.preprocessing import MinMaxScaler

# ── Peel-chain feature extraction (delegated to features module) ────────
from features import (
    compute_peel_pattern_disparity,
    detect_multihop_peel_chains,
    parse_pipe_amounts as _parse_pipe_amounts_impl,
    parse_pipe_addresses as _parse_pipe_addresses_impl,
    count_pipe_elements as _count_pipe_elements_impl,
)

# ── Anomaly scoring (delegated to anomaly_engine module) ────────────────
from anomaly_engine import (
    compute_anomaly_deviation_scores,
    compute_investigative_priority,
)

# ── Percentile-based explainability (delegated to explainability module) ─
from explainability import (
    build_dataset_profile as _build_dataset_profile,
    generate_explanation as _generate_explanation_percentile,
    generate_all_explanations as _generate_all_explanations_percentile,
)

# Backward-compatible alias — app.py imports this name from ml_engine
compute_peel_chain_disparity = compute_peel_pattern_disparity

warnings.filterwarnings("ignore", category=UserWarning)

import builtins as _builtins
_print = _builtins.print  # capture real print before any overrides

def _log(msg: str) -> None:
    """Safe print wrapper — silently ignores I/O errors (e.g. inside Streamlit)."""
    try:
        _print(msg)
    except OSError:
        pass

INPUT_CSV = "bitcoin_traffic.csv"
OUTPUT_CSV = "flagged_transactions.csv"
WHITELIST_CSV = os.path.join("sample_data", "institutional_whitelist.csv")
CONTAMINATION = 0.05

MICRO_TX_THRESHOLD = 0.006

# ── Port-risk scoring constants ─────────────────────────────────────────
# Standard Bitcoin P2P ports (mainnet + testnet) → baseline risk = 0.0
STANDARD_P2P_PORTS = {8332, 8333, 8334, 18332, 18333, 18444, 38332, 38333}

# Confirmed anonymisation / proxy software ports — high forensic signal
# Only ports with unambiguous criminal-tooling association belong here.
ANON_PROXY_PORTS = {
    9050, 9051, 9150,          # Tor SOCKS / Tor control
    4444,                       # I2P HTTP proxy
}

# Zero-risk ports: standard Bitcoin P2P + common legitimate services
# Port 443 (HTTPS) and 80 (HTTP) are ubiquitous and must NOT be penalised.
ZERO_RISK_PORTS = STANDARD_P2P_PORTS | {443, 80}

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(filepath: str = INPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    _log(f"[*] Loaded {len(df)} transactions from {filepath}")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# 2a. COINJOIN / MIXER DETECTION (pre-clustering filter)
# ═══════════════════════════════════════════════════════════════════════════

def is_mixer_transaction(
    row,
    min_participants: int = 10,
    equal_output_ratio: float = 0.50,
    rel_tolerance: float = 0.01,
) -> bool:
    """
    Detect CoinJoin / Mixer transactions before wallet clustering.

    A mixer transaction is characterised by:
      1. A high fan-in **and** fan-out (>= *min_participants* unique
         inputs **or** outputs).
      2. A large share of outputs having *near-identical* values
         (>= *equal_output_ratio* of all outputs).

    The "near-identical" check uses *rel_tolerance* (default 1 %)
    so that rounding artefacts in BTC amounts don't defeat the
    heuristic.

    Parameters
    ----------
    row : pd.Series or dict-like
        A single transaction row containing at minimum the keys
        ``input_amounts`` and ``output_amounts`` (pipe-delimited
        strings of float values).  ``input_addresses`` and
        ``output_addresses`` are also accepted for participant
        counting.
    min_participants : int, default 10
        Minimum number of distinct inputs **or** outputs required
        to even consider the transaction a potential mix.
    equal_output_ratio : float, default 0.50
        Fraction of outputs that must share the same (within
        tolerance) value to classify the tx as a mix.
    rel_tolerance : float, default 0.01
        Maximum relative difference between two output values for
        them to be considered "equal".  0.01 → ±1 %.

    Returns
    -------
    bool
        ``True`` if the transaction exhibits mixer/CoinJoin
        characteristics; ``False`` otherwise.

    Notes
    -----
    This function is intentionally conservative: both the
    participant-count gate **and** the equal-output gate must
    trigger.  Normal transactions with many UTXOs but diverse
    output values will **not** be flagged.
    """
    # ------------------------------------------------------------------
    # Helper: safely split a pipe-delimited string into a list of floats
    # ------------------------------------------------------------------
    def _parse_amounts(raw) -> list:
        if raw is None:
            return []
        text = str(raw).strip()
        if text in ("", "nan", "None"):
            return []
        amounts = []
        for token in text.split("|"):
            token = token.strip()
            if not token:
                continue
            try:
                amounts.append(float(token))
            except (ValueError, TypeError):
                continue          # skip unparseable fragments
        return amounts

    def _count_participants(raw) -> int:
        """Count pipe-delimited addresses (or amounts as fallback)."""
        if raw is None:
            return 0
        text = str(raw).strip()
        if text in ("", "nan", "None"):
            return 0
        return len([t for t in text.split("|") if t.strip()])

    # ------------------------------------------------------------------
    # 1. Parse amounts
    # ------------------------------------------------------------------
    try:
        input_amounts  = _parse_amounts(row.get("input_amounts"))
        output_amounts = _parse_amounts(row.get("output_amounts"))
    except Exception:
        return False              # corrupt / missing row → safe default

    if not output_amounts:
        return False              # nothing to analyse

    # ------------------------------------------------------------------
    # 2. Participant-count gate
    # ------------------------------------------------------------------
    #    Prefer address columns when available; fall back to amounts.
    n_inputs  = _count_participants(row.get("input_addresses")) or len(input_amounts)
    n_outputs = _count_participants(row.get("output_addresses")) or len(output_amounts)

    if max(n_inputs, n_outputs) < min_participants:
        return False              # too few participants → normal tx

    # ------------------------------------------------------------------
    # 3. Equal-output pattern detection
    # ------------------------------------------------------------------
    #    Bucket outputs with relative tolerance: two values v1, v2 are
    #    "equal" iff  |v1 − v2| / max(|v1|, |v2|) <= rel_tolerance.
    #
    #    Implementation: sort outputs, then greedily assign each value
    #    to the current bucket if it is within tolerance of the bucket
    #    representative; otherwise start a new bucket.  Finally, the
    #    largest bucket determines the dominant equal-output cluster.
    sorted_outputs = sorted(output_amounts)
    buckets: list = []            # list of (representative, count)
    for val in sorted_outputs:
        if buckets:
            rep, cnt = buckets[-1]
            denominator = max(abs(rep), abs(val), 1e-12)  # avoid div-zero
            if abs(val - rep) / denominator <= rel_tolerance:
                buckets[-1] = (rep, cnt + 1)
                continue
        buckets.append((val, 1))

    largest_bucket = max(cnt for _, cnt in buckets)
    dominant_ratio = largest_bucket / len(output_amounts)

    return dominant_ratio >= equal_output_ratio


# ═══════════════════════════════════════════════════════════════════════════
# 2a-ii. CHANGE-ADDRESS DETECTION HEURISTIC
# ═══════════════════════════════════════════════════════════════════════════

# Bitcoin address prefix → script type mapping
_PREFIX_TO_SCRIPT: Dict[str, str] = {
    "1":    "P2PKH",       # Legacy Pay-to-Public-Key-Hash
    "3":    "P2SH",        # Pay-to-Script-Hash (incl. wrapped SegWit)
    "bc1q": "P2WPKH",     # Native SegWit v0
    "bc1p": "P2TR",        # Taproot (SegWit v1)
}


def _infer_script_type(address: str) -> str:
    """Infer the Bitcoin script type from an address prefix."""
    address = address.strip()
    for prefix, script in _PREFIX_TO_SCRIPT.items():
        if address.lower().startswith(prefix.lower()):
            return script
    return "UNKNOWN"


def _count_decimal_places(value: float) -> int:
    """
    Count the significant decimal digits of a float.

    For change outputs, Bitcoin wallets typically produce values with
    many trailing digits (e.g., ``1.79328419``), whereas human-chosen
    payment amounts tend to be round (e.g., ``0.5``, ``1.0``, ``0.01``).
    """
    s = f"{value:.8f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def _is_round_amount(value: float, max_decimals: int = 3) -> bool:
    """Return ``True`` if the amount looks like a human-chosen round number."""
    return _count_decimal_places(value) <= max_decimals


def _parse_pipe_addresses(raw) -> List[str]:
    """Safely split a pipe-delimited address string into a list.

    Delegates to :func:`features.parse_pipe_addresses`.
    """
    return _parse_pipe_addresses_impl(raw)


def detect_change_address(
    row,
    known_addresses: set = None,
) -> str | None:
    """
    Identify the most likely **change address** among a transaction's
    outputs using multi-factor heuristics.

    Heuristics Applied (each scored independently)
    -----------------------------------------------
    1. **Script-Type Matching** — Change outputs typically reuse the
       same script type as the inputs (e.g., if inputs are ``bc1q…``
       the change will also be ``bc1q…``).  Score: +3 if matching,
       −1 if mismatching.

    2. **Unrounded Remainder** — Payment amounts are usually round
       numbers chosen by humans (0.5 BTC, 0.01 BTC).  The change
       output is whatever remains after the payment + fee, producing
       a "messy" decimal.  Score: +2 if unround (>3 decimals) while
       at least one sibling is round.

    3. **Decimal Precision Profiling** — Change outputs typically
       have *more* decimal places than payment outputs.  Score: +1
       for the output with the highest decimal precision.

    4. **Novel Address Bias** — If an output address has never
       appeared in prior transactions (not in ``known_addresses``),
       it is more likely to be a fresh change address generated by
       the wallet.  Score: +1 if novel.

    5. **Amount Asymmetry** — In a 2-output peel chain the
       non-payment side is often the larger amount (bulk returning
       to the sender).  Score: +1 for the larger output when
       disparity > 10×.

    Parameters
    ----------
    row : dict-like
        A transaction row with ``input_addresses``,
        ``output_addresses``, ``output_amounts``, and optionally
        ``script_type``.
    known_addresses : set | None
        Set of previously seen addresses for the novelty heuristic.
        If ``None``, heuristic 4 is skipped.

    Returns
    -------
    str | None
        The address identified as the most likely change output,
        or ``None`` if the transaction has ≤ 1 outputs or all
        heuristics are inconclusive.
    """
    # ── Parse outputs ─────────────────────────────────────────────
    out_addrs  = _parse_pipe_addresses(row.get("output_addresses"))
    out_amounts = _parse_pipe_amounts(row.get("output_amounts"))

    # Guard: change detection only applies to multi-output txs
    if len(out_addrs) < 2 or len(out_amounts) < 2:
        return None

    # Truncate to shortest list if lengths mismatch
    n = min(len(out_addrs), len(out_amounts))
    out_addrs  = out_addrs[:n]
    out_amounts = out_amounts[:n]

    # ── Parse input script types ──────────────────────────────────
    in_addrs = _parse_pipe_addresses(row.get("input_addresses"))
    input_scripts = set(_infer_script_type(a) for a in in_addrs) - {"UNKNOWN"}

    # If the row has an explicit script_type column, include it
    explicit_script = str(row.get("script_type", "")).strip()
    if explicit_script and explicit_script != "nan":
        input_scripts.add(explicit_script)

    # ── Compute per-output scores ─────────────────────────────────
    scores = [0] * n
    decimals = [_count_decimal_places(a) for a in out_amounts]
    is_round = [_is_round_amount(a) for a in out_amounts]
    max_decimals = max(decimals) if decimals else 0
    any_round = any(is_round)

    for i in range(n):
        addr = out_addrs[i]
        amt  = out_amounts[i]

        # Heuristic 1: Script-type matching
        output_script = _infer_script_type(addr)
        if output_script != "UNKNOWN" and input_scripts:
            if output_script in input_scripts:
                scores[i] += 3       # same script → likely change
            else:
                scores[i] -= 1       # different script → likely payment

        # Heuristic 2: Unrounded remainder
        if not is_round[i] and any_round:
            scores[i] += 2

        # Heuristic 3: Highest decimal precision
        if decimals[i] == max_decimals and decimals[i] > 3:
            scores[i] += 1

        # Heuristic 4: Novel address
        if known_addresses is not None and addr not in known_addresses:
            scores[i] += 1

        # Heuristic 5: Amount asymmetry (only for 2-output txs)
        if n == 2:
            other_amt = out_amounts[1 - i]
            if other_amt > 0 and amt / other_amt > 10:
                scores[i] += 1

    # ── Select winner ─────────────────────────────────────────────
    max_score = max(scores)
    if max_score <= 0:
        return None                   # all inconclusive

    # Tie-break: choose the output with the highest score;
    # if still tied, pick the one with more decimal places
    best_idx = -1
    best_dec = -1
    for i in range(n):
        if scores[i] == max_score:
            if decimals[i] > best_dec:
                best_dec = decimals[i]
                best_idx = i

    return out_addrs[best_idx] if best_idx >= 0 else None


# ═══════════════════════════════════════════════════════════════════════════
# 2b. GRAPH CONSTRUCTION & WALLET CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

def build_entity_graph(df: pd.DataFrame) -> Tuple[nx.Graph, Dict[str, str]]:
    """
    Builds the Entity-Transaction graph and clusters wallets using the
    Common-Input-Ownership heuristic, with a **CoinJoin / Mixer bypass**
    and **Change-Address graph refinement**.

    For each transaction row the function:

    1. Calls :func:`is_mixer_transaction`.  If the row is a mixer, all
       input addresses are added as **isolated nodes** — no edges.
    2. Otherwise, applies Common-Input-Ownership (edges between all
       co-input addresses).
    3. Runs :func:`detect_change_address` on the outputs.  If a change
       address is identified, it is linked *back* to the first input
       address (i.e., the originating wallet), correctly attributing
       the change output to the sender's entity cluster rather than
       creating a false external link.

    Returns
    -------
    address_graph : nx.Graph
        The co-input address graph (with change-address edges).
    entity_map : dict[str, str]
        Mapping ``{address: "Entity_<id>"}``.
    """
    _log("[*] Building Entity/Transaction graph with NetworkX ...")

    address_graph = nx.Graph()

    mixer_bypassed   = 0       # counter for filtered CoinJoin / Mixer txs
    change_detected  = 0       # counter for change-address refinements
    total_rows       = 0
    known_addresses: set = set()   # running set for novelty heuristic

    for idx, row in df.iterrows():
        total_rows += 1
        inputs = str(row["input_addresses"]).split("|")
        inputs = [a.strip() for a in inputs if a.strip()]
        outputs = _parse_pipe_addresses(row.get("output_addresses"))

        # ── CRITICAL: snapshot known_addresses BEFORE this tx ────────
        # The novelty heuristic in detect_change_address() must see
        # only addresses from *prior* transactions — not the current
        # transaction's own outputs.  Adding them first invalidated
        # Heuristic 4 (Novel Address Bias) entirely.
        historical_addresses = known_addresses.copy()

        # ── Mixer bypass check ──────────────────────────────────────
        if is_mixer_transaction(row):
            # Add every input as an isolated node so it still appears
            # in the graph (important for downstream feature lookups)
            # but do NOT draw edges — this breaks the false link.
            for addr in inputs:
                address_graph.add_node(addr)
            mixer_bypassed += 1
            # Update known_addresses AFTER analysis (even for mixers)
            known_addresses.update(inputs)
            known_addresses.update(outputs)
            continue
        # ─────────────────────────────────────────────────────────────

        # Standard Common-Input-Ownership: link all co-inputs
        for i in range(len(inputs)):
            address_graph.add_node(inputs[i])
            for j in range(i + 1, len(inputs)):
                address_graph.add_edge(inputs[i], inputs[j])

        # ── Change-Address Refinement ────────────────────────────────
        # Identify the change output and link it to the sender's
        # entity cluster (first input address).  This prevents the
        # change address from being treated as a separate wallet.
        # Uses the HISTORICAL snapshot so the novelty check is valid.
        if inputs:
            change_addr = detect_change_address(
                row, known_addresses=historical_addresses,
            )
            if change_addr is not None:
                address_graph.add_node(change_addr)
                address_graph.add_edge(inputs[0], change_addr)
                change_detected += 1
        # ─────────────────────────────────────────────────────────────

        # Update known_addresses AFTER all analysis for this tx
        known_addresses.update(inputs)
        known_addresses.update(outputs)

    _log(
        f"[*] Mixer bypass: {mixer_bypassed}/{total_rows} transactions "
        f"identified as CoinJoin/Mixer and excluded from clustering."
    )
    _log(
        f"[*] Change-address refinement: {change_detected}/{total_rows} "
        f"transactions had a change output linked back to sender entity."
    )

    # Union Find (Connected Components) for entity clustering
    _log("[*] Running Union-Find wallet clustering (Common-Input-Ownership) ...")
    entity_map = {}
    components = list(nx.connected_components(address_graph))
    for entity_id, comp in enumerate(components):
        for addr in comp:
            entity_map[addr] = f"Entity_{entity_id}"

    _log(f"[*] Clustered {len(address_graph.nodes)} addresses into {len(components)} entities.")
    return address_graph, entity_map

def _frequency_encode(series: pd.Series) -> pd.Series:
    freq_map = series.value_counts(normalize=True)
    return series.map(freq_map)

# ═══════════════════════════════════════════════════════════════════════════
# 2c. PORT-RISK SCORING
# ═══════════════════════════════════════════════════════════════════════════

def _compute_port_frequency(df: pd.DataFrame) -> Dict[int, float]:
    """
    Compute the relative frequency of each port across the dataset.

    Combines ``src_port`` and ``dst_port`` into a single frequency
    distribution.  Used to derive a *rarity index* — ports seen
    rarely across the dataset are more suspicious than common ones.

    Returns
    -------
    dict[int, float]
        Mapping ``{port: relative_frequency}`` where frequencies
        sum to 1.0.
    """
    all_ports = pd.concat(
        [df["src_port"], df["dst_port"]], ignore_index=True,
    ).dropna()
    all_ports = pd.to_numeric(all_ports, errors="coerce").dropna().astype(int)
    freq = all_ports.value_counts(normalize=True)
    return freq.to_dict()


def score_port_risk(port, port_freq: Dict[int, float] | None = None) -> float:
    """
    Return a risk score in [0.0, 1.0] for a network port.

    Scoring Strategy
    ----------------
    * **0.0** — Zero-risk: standard Bitcoin P2P ports (8333 etc.),
      HTTPS (443), and HTTP (80).
    * **1.0** — Confirmed anonymisation/proxy: Tor SOCKS (9050, 9051,
      9150) and I2P (4444).
    * **Rarity index** — All other ports are scored based on their
      frequency in the dataset.  Rare ports → higher risk, common
      ports → lower risk.  Formula: ``1 - freq`` capped at [0, 0.8].

    Parameters
    ----------
    port : int-like
        The port number to score.
    port_freq : dict[int, float] | None
        Pre-computed port frequency distribution from
        :func:`_compute_port_frequency`.  When ``None``, falls back
        to a neutral 0.1 for non-special ports (backward compatible).

    Returns 0.1 (low-neutral) for unparseable or missing values.
    """
    try:
        p = int(port)
    except (ValueError, TypeError):
        return 0.1               # missing / corrupt → low-neutral

    if p in ZERO_RISK_PORTS:
        return 0.0               # baseline — expected traffic
    if p in ANON_PROXY_PORTS:
        return 1.0               # confirmed anonymisation tooling

    # Rarity-based scoring for all other ports
    if port_freq is not None:
        freq = port_freq.get(p, 0.0)
        # Invert: rare ports (low freq) → high risk
        # Cap at 0.8 to keep Tor/I2P at the top of the scale
        return round(min(1.0 - freq, 0.8), 4)

    return 0.1                    # fallback when no frequency data


# ═══════════════════════════════════════════════════════════════════════════
# 3a. BEHAVIORAL FEATURE EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════

def _parse_pipe_amounts(raw) -> List[float]:
    """Safely split a pipe-delimited string into a list of floats.

    Delegates to :func:`features.parse_pipe_amounts`.
    """
    return _parse_pipe_amounts_impl(raw)


def _count_pipe_elements(raw) -> int:
    """Count non-empty elements in a pipe-delimited string.

    Delegates to :func:`features.count_pipe_elements`.
    """
    return _count_pipe_elements_impl(raw)


def compute_peel_chain_disparity(output_amounts_str) -> float:
    """
    Detect peel-chain laundering structure (single-split candidate).

    .. deprecated::
        Use :func:`features.compute_peel_pattern_disparity` directly.
        This wrapper is kept for backward compatibility with consumers
        that import ``compute_peel_chain_disparity`` from ``ml_engine``.

    Delegates to :func:`features.compute_peel_pattern_disparity`.
    """
    return compute_peel_pattern_disparity(output_amounts_str)



def compute_fan_in_out(row) -> Tuple[int, int, float]:
    """
    Compute fan-in, fan-out, and their ratio for a transaction row.

    Returns
    -------
    fan_in : int
        Number of distinct input addresses.
    fan_out : int
        Number of distinct output addresses.
    fan_ratio : float
        ``fan_in / fan_out`` (clamped to 0.0 if fan_out == 0).
        * **ratio >> 1** → mass consolidation (many inputs → few outputs)
        * **ratio << 1** → rapid dispersal  (few inputs → many outputs)
        * **ratio ≈ 1**  → balanced / normal
    """
    fan_in  = _count_pipe_elements(row.get("input_addresses"))
    fan_out = _count_pipe_elements(row.get("output_addresses"))
    fan_ratio = (fan_in / fan_out) if fan_out > 0 else 0.0
    return fan_in, fan_out, round(fan_ratio, 4)


def compute_fee_rate_urgency(row) -> float:
    """
    Calculate ``fee / total_input_amount``.

    A disproportionately high fee relative to the transaction value
    signals urgency (the sender is overpaying miners to get fast
    confirmation), which is a common pattern in time-sensitive
    laundering hops.

    Returns 0.0 when fee or input amounts are missing / zero.
    """
    try:
        fee = float(row.get("fee", 0))
    except (ValueError, TypeError):
        return 0.0

    total_input = sum(_parse_pipe_amounts(row.get("input_amounts")))
    if total_input <= 0 or fee <= 0:
        return 0.0

    return round(fee / total_input, 8)


# ═══════════════════════════════════════════════════════════════════════════
# 3a-ii. MULTI-HOP PEEL CHAIN DETECTION
# ═══════════════════════════════════════════════════════════════════════════
#
# The canonical implementation now lives in features.py and uses a
# NetworkX DiGraph for directed graph traversal.  The function is
# imported at the top of this module:
#
#     from features import detect_multihop_peel_chains
#
# The import is used directly in engineer_features() below.


# ═══════════════════════════════════════════════════════════════════════════
# 3b. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform raw transaction data into graph-aware numeric features.
    """
    address_graph, entity_map = build_entity_graph(df)
    
    # Assign Entity ID to each transaction (based on first input)
    entities = []
    for inputs in df["input_addresses"].str.split("|"):
        if len(inputs) > 0 and inputs[0] in entity_map:
            entities.append(entity_map[inputs[0]])
        else:
            entities.append("UnknownEntity")
    df["entity_id"] = entities

    features = pd.DataFrame(index=df.index)

    # --- Graph Correlation Features ---
    # For each entity, how many distinct IPs does it use? (High = anonymization)
    entity_ip_counts = df.groupby("entity_id")["src_ip"].nunique()
    features["entity_ip_diversity"] = df["entity_id"].map(entity_ip_counts).fillna(1)
    
    # For each IP, how many distinct entities is it broadcasting for?
    ip_entity_counts = df.groupby("src_ip")["entity_id"].nunique()
    features["ip_entity_diversity"] = df["src_ip"].map(ip_entity_counts).fillna(1)

    features["entity_tx_freq"] = _frequency_encode(df["entity_id"])

    # --- Standard Network Features ---
    #     NOTE: geo_country is kept on the DataFrame for display only;
    #     it is NOT fed into the ML feature matrix because physical
    #     locations are trivially spoofed via VPN / proxy.
    features["src_ip_freq"] = _frequency_encode(df["src_ip"])

    # Calculate total input amount for each transaction
    def calc_total(amount_str):
        try:
            return sum(float(a) for a in str(amount_str).split("|") if a.strip())
        except:
            return 0.0

    df["total_amount_btc"] = df["input_amounts"].apply(calc_total)

    # --- Amount Features ---
    scaler = MinMaxScaler()
    features["amount_btc_scaled"] = scaler.fit_transform(df[["total_amount_btc"]])
    if "fee" in df.columns:
        features["fee_scaled"] = scaler.fit_transform(df[["fee"]])
    features["is_micro_tx"] = (df["total_amount_btc"] < MICRO_TX_THRESHOLD).astype(int)

    # Script type frequency encoding
    if "script_type" in df.columns:
        features["script_type_freq"] = _frequency_encode(df["script_type"])

    # --- Port-Risk Features (rarity-aware, replaces binary src_port_is_std) ---
    port_freq = _compute_port_frequency(df)
    features["src_port_risk"] = df["src_port"].apply(
        lambda p: score_port_risk(p, port_freq)
    )
    features["dst_port_risk"] = df["dst_port"].apply(
        lambda p: score_port_risk(p, port_freq)
    )
    features["port_risk_combined"] = (
        features["src_port_risk"] * 0.6 + features["dst_port_risk"] * 0.4
    ).round(4)

    # --- Behavioral Crime-Detection Features ---

    # 1) Peel-Pattern Candidate Disparity (renamed from peel_chain_disparity)
    features["peel_chain_disparity"] = df["output_amounts"].apply(
        compute_peel_pattern_disparity
    )

    # 2) Fan-In / Fan-Out Ratios
    fan_data = df.apply(compute_fan_in_out, axis=1, result_type="expand")
    fan_data.columns = ["fan_in", "fan_out", "fan_ratio"]
    features["fan_in"]    = fan_data["fan_in"]
    features["fan_out"]   = fan_data["fan_out"]
    features["fan_ratio"] = fan_data["fan_ratio"]

    # 3) Fee Rate Urgency
    features["fee_rate_urgency"] = df.apply(compute_fee_rate_urgency, axis=1)

    # 4) Transaction Value Z-Score
    mean_val = df["total_amount_btc"].mean()
    std_val  = df["total_amount_btc"].std()
    if std_val > 0:
        features["value_zscore"] = (
            (df["total_amount_btc"] - mean_val) / std_val
        ).round(4)
    else:
        features["value_zscore"] = 0.0

    # Take absolute Z-score — both extremes are suspicious
    features["value_zscore_abs"] = features["value_zscore"].abs()

    # --- Multi-Hop Peel Chain Detection (NetworkX DiGraph) ---
    chain_data = detect_multihop_peel_chains(df)
    features["peel_chain_length"] = chain_data["chain_length"]
    features["peel_chain_hop_position"] = chain_data["hop_position"]
    features["fund_diminishment_ratio"] = chain_data["fund_diminishment_ratio"]
    df["peel_chain_id"] = chain_data["chain_id"]

    _log(f"[*] Engineered {features.shape[1]} features: {list(features.columns)}")
    return features, df

# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING & PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def train_model(features: pd.DataFrame, contamination: float = CONTAMINATION) -> Tuple[IsolationForest, np.ndarray, np.ndarray]:
    _log(f"[*] Training IsolationForest (n_estimators=200, contamination={contamination}) …")
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=42,
        n_jobs=1
    )
    model.fit(features)
    predictions = model.predict(features)
    raw_scores = model.decision_function(features)
    
    num_anomalies = (predictions == -1).sum()
    _log(f"[*] Flagged {num_anomalies} transactions as anomalies ({num_anomalies / len(features) * 100:.1f}%)")
    return model, predictions, raw_scores

def compute_risk_scores(raw_scores: np.ndarray) -> np.ndarray:
    """
    Compute Anomaly Deviation Scores (0–100).

    .. deprecated::
        Use :func:`anomaly_engine.compute_anomaly_deviation_scores`
        directly.  This wrapper is kept for backward compatibility.

    This score represents **unsupervised statistical distance** from
    the learned transaction distribution via IsolationForest.  It is
    NOT a calibrated probability of guilt, criminal activity, or
    confidence.  It should be interpreted as: "How far does this
    transaction deviate from the baseline behavior learned by the
    model?"

    Parameters
    ----------
    raw_scores : np.ndarray
        Raw output from ``IsolationForest.decision_function()``.

    Returns
    -------
    np.ndarray
        Anomaly Deviation Scores in [0.0, 100.0].
    """
    return compute_anomaly_deviation_scores(raw_scores)

# ═══════════════════════════════════════════════════════════════════════════
# 5. EXPLAINABLE AI (XAI)
# ═══════════════════════════════════════════════════════════════════════════

def generate_explanation(row_features, risk_score, original_row, dataset_profile=None):
    """
    Generate a human-readable explanation for a flagged transaction.

    .. deprecated::
        Use :func:`explainability.generate_explanation` directly for
        percentile-based explanations.

    When *dataset_profile* is provided, delegates to the new
    percentile-based explainability module.  Otherwise falls back
    to a minimal explanation.
    """
    if dataset_profile is not None:
        return _generate_explanation_percentile(
            row_features, risk_score, original_row, dataset_profile,
        )
    # Fallback when no profile is available
    if risk_score < 50.0:
        return "Normal network traffic behavior."
    return (
        "Anomalous deviation from baseline transaction distribution "
        "(structural graph relationships flagged by unsupervised model)."
    )


def generate_all_explanations(
    df: pd.DataFrame,
    features: pd.DataFrame,
    predictions: np.ndarray,
    risk_scores: np.ndarray,
    dataset_profile=None,
) -> pd.DataFrame:
    """
    Batch-generate explanations for all transactions.

    When *dataset_profile* is provided (the new path), delegates to
    :func:`explainability.generate_all_explanations` which produces
    percentile-based audit reasons and structured telemetry.

    Parameters
    ----------
    dataset_profile : DatasetProfile | None
        Pre-computed dataset statistics.  Pass ``None`` to use the
        legacy (hardcoded-threshold) fallback.
    """
    _log("[*] Generating percentile-aware risk explanations …")

    if dataset_profile is not None:
        return _generate_all_explanations_percentile(
            df, features, predictions, risk_scores, dataset_profile,
        )

    # Legacy fallback (no profile)
    result = df.copy()
    result["is_anomaly"] = (predictions == -1)
    result["risk_score"] = risk_scores
    explanations = []
    for idx, row in result.iterrows():
        if row["is_anomaly"]:
            feat_row = features.iloc[idx]
            exp = generate_explanation(feat_row, row["risk_score"], row)
            explanations.append(exp)
        else:
            explanations.append("Normal.")
    result["explanation"] = explanations
    return result

# ═══════════════════════════════════════════════════════════════════════════
# 5a. INSTITUTIONAL WHITELIST FILTER (post-scoring, pre-export)
# ═══════════════════════════════════════════════════════════════════════════

def load_institutional_whitelist(
    filepath: str = WHITELIST_CSV,
) -> Tuple[Set[str], Dict[str, str]]:
    """
    Load known institutional wallet addresses from a CSV file.

    The CSV must have at least an ``address`` column.  An optional
    ``institution`` column is used to label matched transactions.

    Parameters
    ----------
    filepath : str
        Path to the whitelist CSV.  If the file is missing or empty
        an empty set is returned — the pipeline continues without
        filtering.

    Returns
    -------
    addresses : set[str]
        The set of whitelisted addresses (lowercased for
        case-insensitive matching).
    labels : dict[str, str]
        Mapping ``{address_lower: institution_name}``.
    """
    if not os.path.isfile(filepath):
        _log(f"[!] Whitelist file '{filepath}' not found — skipping institutional filter.")
        return set(), {}

    try:
        wl = pd.read_csv(filepath)
    except Exception as exc:
        _log(f"[!] Failed to read whitelist '{filepath}': {exc} — skipping.")
        return set(), {}

    if "address" not in wl.columns:
        _log("[!] Whitelist CSV missing 'address' column — skipping.")
        return set(), {}

    # Normalise: strip whitespace, lowercase for case-insensitive matching
    wl["address"] = wl["address"].astype(str).str.strip().str.lower()
    wl = wl[wl["address"].ne("") & wl["address"].ne("nan")]

    labels = {}
    # Support both column names: 'institution' (legacy) and 'entity_name' (generate_data.py)
    label_col = None
    if "institution" in wl.columns:
        label_col = "institution"
    elif "entity_name" in wl.columns:
        label_col = "entity_name"

    if label_col is not None:
        labels = dict(zip(wl["address"], wl[label_col].astype(str)))

    addresses = set(wl["address"])
    _log(f"[*] Loaded {len(addresses)} institutional addresses from '{filepath}'.")
    return addresses, labels


def apply_institutional_whitelist(
    df: pd.DataFrame,
    whitelist: Set[str] | None = None,
    labels: Dict[str, str] | None = None,
    whitelist_csv: str = WHITELIST_CSV,
) -> pd.DataFrame:
    """
    Post-scoring filter: apply entity-level whitelist annotations.

    **Scoping**: Whitelisting is applied to individual entity *nodes*,
    not entire transactions.  This prevents criminal deposit trails
    (e.g., laundered funds sent *to* Binance) from being concealed.

    Behaviour
    ---------
    For each transaction, check input and output sides independently:

    * **Both sides whitelisted** → full override:
      ``risk_score → 0.0``, ``is_anomaly → False``.
    * **One side whitelisted** → risk *discount* (×0.15) but anomaly
      flag and scoring on the unverified counterparty are retained.
      Explanation is updated to indicate which side is regulated.
    * **Neither side whitelisted** → no change.

    New columns added:

    * ``whitelisted_side`` — ``"input"``, ``"output"``, ``"both"``,
      or ``None``.
    * ``whitelisted_entity`` — institution name(s) or ``None``.

    Parameters
    ----------
    df : pd.DataFrame
        The enriched DataFrame (output of ``generate_all_explanations``).
        Must contain ``risk_score``, ``is_anomaly``, and ``explanation``.
    whitelist : set[str] | None
        Pre-loaded whitelist addresses.  If ``None`` the whitelist is
        loaded fresh from *whitelist_csv*.
    labels : dict[str, str] | None
        Address → institution name mapping.  Loaded alongside
        *whitelist* when ``None``.
    whitelist_csv : str
        Path used when *whitelist* is ``None``.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with per-side whitelist annotations.
    """
    if whitelist is None:
        whitelist, labels = load_institutional_whitelist(whitelist_csv)
    if labels is None:
        labels = {}

    if not whitelist:
        # Ensure columns exist even when whitelist is empty
        df["whitelisted_side"] = None
        df["whitelisted_entity"] = None
        return df

    # Risk discount factor when only one side is institutional.
    # The regulated side lowers overall suspicion but does NOT
    # eliminate scoring on the unverified counterparty.
    _SINGLE_SIDE_DISCOUNT = 0.15

    def _check_side(raw) -> Tuple[bool, str]:
        """Check if any pipe-delimited address is in the whitelist."""
        if raw is None:
            return False, ""
        text = str(raw).strip()
        if text in ("", "nan", "None"):
            return False, ""
        for addr in text.split("|"):
            addr_clean = addr.strip().lower()
            if addr_clean in whitelist:
                return True, labels.get(addr_clean, "Unknown Institution")
        return False, ""

    both_cleared = 0
    single_discounted = 0

    # Pre-populate annotation columns
    df["whitelisted_side"] = None
    df["whitelisted_entity"] = None

    for idx in df.index:
        input_match, input_inst = _check_side(
            df.at[idx, "input_addresses"]
        )
        output_match, output_inst = _check_side(
            df.at[idx, "output_addresses"]
        )

        if input_match and output_match:
            # ── Both sides are regulated → full override ─────────
            institutions = ", ".join(
                filter(None, dict.fromkeys([input_inst, output_inst]))
            )
            df.at[idx, "risk_score"]  = 0.0
            df.at[idx, "is_anomaly"] = False
            df.at[idx, "explanation"] = (
                f"Regulated Entity ({institutions}) — both sides institutional."
            )
            if "entity_id" in df.columns:
                df.at[idx, "entity_id"] = "Regulated Entity"
            df.at[idx, "whitelisted_side"] = "both"
            df.at[idx, "whitelisted_entity"] = institutions
            both_cleared += 1

        elif input_match:
            # ── Only input side is regulated ──────────────────────
            # Discount risk but DO NOT zero it — the output side
            # (recipient) is unverified and could be criminal.
            original_risk = float(df.at[idx, "risk_score"])
            discounted = round(original_risk * _SINGLE_SIDE_DISCOUNT, 2)
            df.at[idx, "risk_score"] = discounted
            # Preserve is_anomaly — anomaly status reflects the
            # unverified counterparty, not the regulated side.
            existing_expl = str(df.at[idx, "explanation"])
            df.at[idx, "explanation"] = (
                f"Regulated Entity ({input_inst}) on input side; "
                f"output counterparty unverified. {existing_expl}"
            )
            df.at[idx, "whitelisted_side"] = "input"
            df.at[idx, "whitelisted_entity"] = input_inst
            single_discounted += 1

        elif output_match:
            # ── Only output side is regulated ─────────────────────
            # This is the critical case: a potentially criminal
            # sender depositing to a regulated exchange.  Risk must
            # NOT be zeroed — the sender is the suspect.
            original_risk = float(df.at[idx, "risk_score"])
            discounted = round(original_risk * _SINGLE_SIDE_DISCOUNT, 2)
            df.at[idx, "risk_score"] = discounted
            existing_expl = str(df.at[idx, "explanation"])
            df.at[idx, "explanation"] = (
                f"Regulated Entity ({output_inst}) on output side; "
                f"input counterparty unverified. {existing_expl}"
            )
            df.at[idx, "whitelisted_side"] = "output"
            df.at[idx, "whitelisted_entity"] = output_inst
            single_discounted += 1

    _log(
        f"[*] Institutional whitelist: {both_cleared}/{len(df)} transactions "
        f"fully cleared (both sides institutional, risk → 0.0). "
        f"{single_discounted}/{len(df)} transactions discounted "
        f"(single-side match, risk × {_SINGLE_SIDE_DISCOUNT})."
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 6. EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_results(df: pd.DataFrame, filepath: str = OUTPUT_CSV) -> str:
    df.to_csv(filepath, index=False)
    _log(f"[*] Results saved to {filepath}")
    return filepath

# ═══════════════════════════════════════════════════════════════════════════
# 7. FULL PIPELINE ENTRIES
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
    contamination: float = CONTAMINATION,
    adapter=None,
) -> Tuple[pd.DataFrame, IsolationForest, pd.DataFrame]:
    """
    Full detection pipeline.

    Parameters
    ----------
    adapter : BaseTransactionAdapter | None
        If provided, the adapter handles load / normalise / extract.
        Otherwise the legacy code path is used (backward-compatible).
    """
    _log("=" * 60)
    _log("  Bitcoin Graph Anomaly Detection Engine")
    _log("=" * 60)

    if adapter is not None:
        return adapter.run_pipeline(
            input_csv, contamination=contamination, output_csv=output_csv,
        )

    # ── Legacy path (no adapter) ─────────────────────────────────
    df = load_data(input_csv)
    features, df = engineer_features(df)
    model, predictions, raw_scores = train_model(features, contamination=contamination)
    risk_scores = compute_anomaly_deviation_scores(raw_scores)
    dataset_profile = _build_dataset_profile(features)
    _log(f"[*] Built dataset profile: {len(dataset_profile.stats)} features profiled")
    enriched_df = generate_all_explanations(df, features, predictions, risk_scores, dataset_profile)

    # ── Institutional Whitelist Override ──
    enriched_df = apply_institutional_whitelist(enriched_df)

    export_results(enriched_df, output_csv)

    anomaly_df = enriched_df[enriched_df["is_anomaly"]]
    _log(f"\n{'─' * 60}")
    _log(f"  Summary")
    _log(f"{'─' * 60}")
    _log(f"  Total transactions:     {len(enriched_df)}")
    _log(f"  Flagged anomalies:      {len(anomaly_df)}")
    if len(anomaly_df) > 0:
        _log(f"  Avg risk (anomalies):   {anomaly_df['risk_score'].mean():.1f}%")
        _log(f"  Max risk score:         {anomaly_df['risk_score'].max():.1f}%")
    else:
        _log(f"  Avg risk (anomalies):   N/A")
        _log(f"  Max risk score:         N/A")
    _log(f"  Output:                 {output_csv}")
    _log(f"{'─' * 60}")
    _log(f"  [✓] Pipeline complete.\n")

    return enriched_df, model, features

def run_pipeline_from_df(
    df: pd.DataFrame,
    contamination: float = CONTAMINATION,
    adapter=None,
) -> Tuple[pd.DataFrame, IsolationForest, pd.DataFrame]:
    """
    Run the pipeline on an already-loaded DataFrame.

    Returns
    -------
    enriched_df : pd.DataFrame
        Transaction DataFrame with risk scores, anomaly flags, and explanations.
    model : IsolationForest
        The trained model instance.
    features : pd.DataFrame
        The numeric feature matrix used for scoring (for the Heuristics Inspector).

    Parameters
    ----------
    adapter : BaseTransactionAdapter | None
        If provided, the adapter handles normalise / extract.
        Otherwise the legacy code path is used (backward-compatible).
    """
    if adapter is not None:
        return adapter.run_pipeline(df, contamination=contamination)

    # ── Legacy path (no adapter) ──
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    features, df = engineer_features(df)
    model, predictions, raw_scores = train_model(features, contamination=contamination)
    risk_scores = compute_anomaly_deviation_scores(raw_scores)
    dataset_profile = _build_dataset_profile(features)
    enriched_df = generate_all_explanations(df, features, predictions, risk_scores, dataset_profile)

    # ── Institutional Whitelist Override ──
    enriched_df = apply_institutional_whitelist(enriched_df)

    return enriched_df, model, features


# ═══════════════════════════════════════════════════════════════════════════
# 8. ADAPTER FACTORY (convenience re-export)
# ═══════════════════════════════════════════════════════════════════════════

def get_adapter(chain: str = "bitcoin"):
    """
    Convenience re-export of :func:`transaction_adapter.get_adapter`.

    >>> adapter = get_adapter("bitcoin")
    >>> enriched_df, model = adapter.run_pipeline("data.csv")
    """
    from transaction_adapter import get_adapter as _get_adapter
    return _get_adapter(chain)


if __name__ == "__main__":
    run_pipeline()
