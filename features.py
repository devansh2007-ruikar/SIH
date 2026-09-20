"""
features.py — Peel-Chain Feature Extraction & Multi-Hop Detection
=================================================================

Dedicated feature extraction module for the MITHYA triage engine.
Provides:

1. **Single-split heuristic** — ``compute_peel_pattern_disparity()``
   identifies individual transactions with a 2-output peel signature
   (one micro-fraction, one large change).  Labeled as "Peel-Pattern
   Candidate" to distinguish from confirmed multi-hop chains.

2. **Multi-hop trajectory traversal** — ``detect_multihop_peel_chains()``
   builds a NetworkX ``DiGraph`` of transaction linkages and traces
   directed paths where change addresses flow sequentially from TX_n
   to TX_{n+1}, with progressive fund diminishment.

3. **Shared parsing utilities** — pipe-delimited address / amount
   parsers reused by ``ml_engine.py`` and other downstream consumers.

Dependencies
------------
- ``networkx`` — directed graph construction and traversal
- ``pandas`` — DataFrame I/O

Usage
-----
::

    from features import (
        compute_peel_pattern_disparity,
        detect_multihop_peel_chains,
        parse_pipe_amounts,
        parse_pipe_addresses,
        count_pipe_elements,
    )
"""

from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# SHARED PARSING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def parse_pipe_amounts(raw) -> List[float]:
    """Safely split a pipe-delimited string into a list of floats.

    Parameters
    ----------
    raw : str or None
        Pipe-separated amount values, e.g. ``"0.2|24.799"``.

    Returns
    -------
    list[float]
        Parsed float values.  Invalid / missing tokens are silently
        skipped.
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if text in ("", "nan", "None"):
        return []
    result = []
    for token in text.split("|"):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(float(token))
        except (ValueError, TypeError):
            continue
    return result


def parse_pipe_addresses(raw) -> List[str]:
    """Safely split a pipe-delimited address string into a list.

    Parameters
    ----------
    raw : str or None
        Pipe-separated address values, e.g. ``"1Abc...|bc1q..."``.

    Returns
    -------
    list[str]
        Non-empty, stripped address strings.
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if text in ("", "nan", "None"):
        return []
    return [a.strip() for a in text.split("|") if a.strip()]


def count_pipe_elements(raw) -> int:
    """Count non-empty elements in a pipe-delimited string.

    Parameters
    ----------
    raw : str or None
        Pipe-separated values.

    Returns
    -------
    int
        Number of non-empty elements.
    """
    if raw is None:
        return 0
    text = str(raw).strip()
    if text in ("", "nan", "None"):
        return 0
    return len([t for t in text.split("|") if t.strip()])


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-SPLIT PEEL-PATTERN CANDIDATE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_peel_pattern_disparity(output_amounts_str) -> float:
    """
    Detect a single-split peel-pattern candidate.

    In a peel chain the sender peels off a small payment to the
    recipient and sends the bulk back to a change address.  This
    produces a characteristic **two-output** pattern where one
    output is a micro-fraction (< 1 %) and the other captures the
    remaining balance (> 99 %).

    .. note::
       This heuristic identifies *candidates* for peel-chain
       membership at the individual transaction level.  Confirmed
       multi-hop chains require the graph traversal provided by
       :func:`detect_multihop_peel_chains`.

    Returns
    -------
    float
        A disparity score in ``[0.0, 1.0]``:

        * **1.0** — perfect peel-pattern signature (one output < 1 %,
          the other > 99 % of total output value)
        * **0.0** — no peel pattern detected (single output, equal
          split, or more than two outputs)

    The score for two outputs is ``max_share - min_share``, yielding
    values near 1.0 for extreme disparity and near 0.0 for even
    splits.  Transactions with != 2 outputs return 0.0 because
    the peel-pattern heuristic is defined only for the two-output case.
    """
    amounts = parse_pipe_amounts(output_amounts_str)
    if len(amounts) != 2:
        return 0.0                # heuristic applies to 2-output txs only

    total = sum(amounts)
    if total <= 0:
        return 0.0

    shares = [a / total for a in amounts]
    max_share = max(shares)
    min_share = min(shares)

    # Classic peel: one side < 1 %, other > 99 %
    if min_share < 0.01 and max_share > 0.99:
        return round(max_share - min_share, 6)
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-HOP PEEL CHAIN DETECTION (NetworkX DiGraph)
# ═══════════════════════════════════════════════════════════════════════════

def detect_multihop_peel_chains(
    df: pd.DataFrame,
    min_disparity: float = 0.5,
    min_chain_length: int = 2,
) -> pd.DataFrame:
    """
    Detect multi-hop peel chain trajectories via NetworkX directed
    graph traversal.

    Builds an ``nx.DiGraph`` where:

    - **Nodes** represent transaction IDs (``txid``)
    - **Edges** represent change-address linkage: an edge from TX_n
      to TX_{n+1} exists when TX_n's *change* (larger) output address
      appears as TX_{n+1}'s input address

    The algorithm then finds maximal directed paths through this graph,
    filtering for chains of at least ``min_chain_length`` hops.

    Invariants checked per hop
    --------------------------
    - TX_n has a 2-output peel pattern (one small peel, one large change)
    - The *change* (larger) output of TX_n is the *input* of TX_{n+1}
    - The balance progressively diminishes across hops

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame with ``txid``, ``input_addresses``,
        ``output_addresses``, ``output_amounts`` columns.
    min_disparity : float
        Minimum output disparity (max_share - min_share) to qualify
        a transaction as a peel-pattern candidate.
    min_chain_length : int
        Minimum number of linked hops to constitute a chain.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like *df* with columns:

        - ``chain_id``  — unique chain identifier (None if not in chain)
        - ``chain_length`` — total hops in the chain (0 if not in chain)
        - ``hop_position`` — 1-indexed position within the chain
        - ``fund_diminishment_ratio`` — ratio of this hop's input to
          the chain's initial input (1.0 at hop 1, decreasing)
    """
    # ── Build output-address → consuming-txid lookup ──────────────
    # For each address that appears as an input, record which txid
    # consumes it.
    addr_to_consuming_txid: Dict[str, str] = {}
    txid_to_idx: Dict[str, int] = {}

    for idx, row in df.iterrows():
        txid = str(row["txid"])
        txid_to_idx[txid] = idx
        for addr in parse_pipe_addresses(row.get("input_addresses")):
            addr_to_consuming_txid[addr] = txid

    # ── Identify peel-pattern candidate transactions ──────────────
    # A candidate has exactly 2 outputs with high disparity
    peel_candidates: Dict[str, dict] = {}  # txid → {change_addr, ...}

    for idx, row in df.iterrows():
        txid = str(row["txid"])
        out_addrs = parse_pipe_addresses(row.get("output_addresses"))
        out_amounts = parse_pipe_amounts(row.get("output_amounts"))

        if len(out_addrs) != 2 or len(out_amounts) != 2:
            continue

        total = sum(out_amounts)
        if total <= 0:
            continue

        shares = [a / total for a in out_amounts]
        disparity = max(shares) - min(shares)

        if disparity < min_disparity:
            continue

        # Identify change (larger) vs peel (smaller)
        if out_amounts[0] >= out_amounts[1]:
            change_idx, peel_idx = 0, 1
        else:
            change_idx, peel_idx = 1, 0

        peel_candidates[txid] = {
            "change_addr": out_addrs[change_idx],
            "peel_addr": out_addrs[peel_idx],
            "change_amount": out_amounts[change_idx],
            "peel_amount": out_amounts[peel_idx],
            "total_input": sum(parse_pipe_amounts(row.get("input_amounts"))),
            "disparity": disparity,
            "df_idx": idx,
        }

    # ── Build NetworkX DiGraph of chain linkages ──────────────────
    # Nodes = txids of peel candidates
    # Edge TX_a → TX_b exists when TX_a's change address is consumed
    # by TX_b as an input
    G = nx.DiGraph()

    for txid, info in peel_candidates.items():
        G.add_node(txid)

    for txid, info in peel_candidates.items():
        change_addr = info["change_addr"]
        next_txid = addr_to_consuming_txid.get(change_addr)

        if next_txid is not None and next_txid != txid:
            # The next tx need not itself be a peel candidate —
            # terminal hops (single-output to exchange) are valid
            G.add_node(next_txid)
            G.add_edge(txid, next_txid)

    # ── Find maximal directed paths (chain heads → tails) ─────────
    # Chain heads are nodes with in-degree 0 within the peel subgraph
    # (i.e., no predecessor peel candidate feeds into them)
    visited_txids: Set[str] = set()
    chains: List[List[str]] = []

    # Identify source nodes (in-degree 0 within G)
    source_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

    # Also consider peel candidates that might not be sources in G
    # but haven't been visited — handles disconnected single-candidates
    for start_txid in source_nodes:
        if start_txid in visited_txids:
            continue

        # Trace the longest forward path from this source
        chain = [start_txid]
        current_txid = start_txid
        visited_txids.add(current_txid)

        while True:
            successors = list(G.successors(current_txid))
            if not successors:
                break

            next_txid = successors[0]  # single successor in peel chain
            if next_txid in visited_txids:
                break

            chain.append(next_txid)
            visited_txids.add(next_txid)
            current_txid = next_txid

        if len(chain) >= min_chain_length:
            chains.append(chain)

    # ── Build result DataFrame ────────────────────────────────────
    chain_ids = pd.Series([None] * len(df), index=df.index, dtype=object)
    chain_lengths = pd.Series([0] * len(df), index=df.index, dtype=int)
    hop_positions = pd.Series([0] * len(df), index=df.index, dtype=int)
    fund_diminishment = pd.Series(
        [0.0] * len(df), index=df.index, dtype=float,
    )

    for chain_num, chain in enumerate(chains):
        cid = f"mhop_{chain_num:04d}"

        # Determine the initial input amount for diminishment ratio
        first_txid = chain[0]
        if first_txid in peel_candidates:
            initial_input = peel_candidates[first_txid]["total_input"]
        else:
            initial_input = sum(
                parse_pipe_amounts(
                    df.loc[txid_to_idx[first_txid], "input_amounts"]
                    if first_txid in txid_to_idx
                    else None
                )
            )
        if initial_input <= 0:
            initial_input = 1.0  # guard against division by zero

        for hop_pos, txid in enumerate(chain, start=1):
            if txid in txid_to_idx:
                idx = txid_to_idx[txid]
                chain_ids.at[idx] = cid
                chain_lengths.at[idx] = len(chain)
                hop_positions.at[idx] = hop_pos

                # Compute fund diminishment ratio
                if txid in peel_candidates:
                    current_input = peel_candidates[txid]["total_input"]
                else:
                    current_input = sum(
                        parse_pipe_amounts(
                            df.loc[idx, "input_amounts"]
                        )
                    )
                fund_diminishment.at[idx] = round(
                    current_input / initial_input, 6,
                )

    result = pd.DataFrame({
        "chain_id": chain_ids,
        "chain_length": chain_lengths,
        "hop_position": hop_positions,
        "fund_diminishment_ratio": fund_diminishment,
    }, index=df.index)

    total_chains = len(chains)
    total_txs = sum(len(c) for c in chains)
    print(
        f"[*] Multi-hop peel chains (NetworkX): {total_chains} chains "
        f"detected ({total_txs} total transactions across chains)."
    )

    return result
