"""
flow_tracer.py -- Hop-by-Hop Fund Flow Tracer
==============================================

Forensic blockchain fund flow tracking engine for MITHYA:
- Forward traversal: trace where funds flowed via breadth-first search on spending inputs.
- Backward traversal: trace source of funds back to originating transactions.
- Classifies hop roles (Mixer, Exchange endpoint, Dispersal, Consolidation, Layering, Source).
- Computes time deltas, retained percentages, and peeled amounts.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def _parse_addresses(val: Any) -> List[str]:
    """Parse pipe-separated or list/tuple addresses into a list of cleaned strings."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(a).strip() for a in val if str(a).strip()]
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    return [a.strip() for a in s.split("|") if a.strip()]


def _parse_amounts(val: Any) -> List[float]:
    """Parse pipe-separated or scalar amounts into a list of floats."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            try:
                if pd.notna(x):
                    out.append(float(x))
            except (ValueError, TypeError):
                pass
        return out
    if isinstance(val, (int, float)):
        return [float(val)]
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    out = []
    for p in s.split("|"):
        p = p.strip()
        if p:
            try:
                out.append(float(p))
            except (ValueError, TypeError):
                pass
    return out


def build_spend_index(df: pd.DataFrame) -> Dict[str, List[Tuple[str, Any]]]:
    """
    Build a mapping from address to a list of (txid, timestamp) tuples
    where that address appears in input_addresses.
    """
    spend_index: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
    if "input_addresses" not in df.columns or "txid" not in df.columns:
        return dict(spend_index)

    for _, row in df.iterrows():
        txid = str(row["txid"])
        ts = row.get("timestamp")
        addrs = _parse_addresses(row.get("input_addresses"))
        for addr in addrs:
            spend_index[addr].append((txid, ts))
    return dict(spend_index)


def build_creation_index(df: pd.DataFrame) -> Dict[str, List[Tuple[str, Any]]]:
    """
    Build a mapping from address to a list of (txid, timestamp) tuples
    where that address appears in output_addresses.
    """
    creation_index: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
    if "output_addresses" not in df.columns or "txid" not in df.columns:
        return dict(creation_index)

    for _, row in df.iterrows():
        txid = str(row["txid"])
        ts = row.get("timestamp")
        addrs = _parse_addresses(row.get("output_addresses"))
        for addr in addrs:
            creation_index[addr].append((txid, ts))
    return dict(creation_index)


def _classify_hop_role(
    row: pd.Series | Dict[str, Any],
    hop: int,
) -> str:
    """
    Determine the forensic role of a transaction in the fund flow.
    Role rules:
      - "Mixer" if detected_type is CoinJoin_Mixer
      - "Exchange endpoint" if whitelisted_side is not null
      - "Dispersal" if fan_out >= 5
      - "Consolidation" if fan_in >= 5
      - "Layering (peel)" if detected_type is Peel_Chain
      - "Source" for hop 0
      - otherwise "Intermediary"
    """
    if hop == 0:
        return "Source"

    det_type = str(row.get("detected_type") or row.get("attack_type") or "")
    if det_type == "CoinJoin_Mixer":
        return "Mixer"

    wl_side = row.get("whitelisted_side")
    if wl_side is not None and not (isinstance(wl_side, float) and pd.isna(wl_side)):
        s_wl = str(wl_side).strip()
        if s_wl and s_wl.lower() not in ("none", "nan", "false", "0"):
            return "Exchange endpoint"

    fan_o = row.get("fan_out")
    if fan_o is None or (isinstance(fan_o, float) and pd.isna(fan_o)):
        fan_o = len(_parse_addresses(row.get("output_addresses")))
    else:
        fan_o = float(fan_o)

    fan_i = row.get("fan_in")
    if fan_i is None or (isinstance(fan_i, float) and pd.isna(fan_i)):
        fan_i = len(_parse_addresses(row.get("input_addresses")))
    else:
        fan_i = float(fan_i)

    if fan_o >= 5:
        return "Dispersal"
    if fan_i >= 5:
        return "Consolidation"
    if det_type == "Peel_Chain":
        return "Layering (peel)"

    return "Intermediary"


def trace_forward(
    df: pd.DataFrame,
    txid: str,
    max_hops: int = 6,
    spend_index: Optional[Dict[str, List[Tuple[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Trace forward hop-by-hop fund flow using breadth-first search.
    From a tx, for each output address, finds later txs that spend it (timestamp > current).

    Stops at exchange endpoints and mixers, at max_hops, or after 200 nodes.

    Returns
    -------
    List[Dict[str, Any]]
        List of hop dicts with:
        - hop: int (0, 1, 2, ...)
        - txid: str
        - timestamp: str/timestamp
        - delta_t_minutes: float (since previous hop)
        - btc_in: float
        - largest_output_btc: float
        - retained_pct: float (largest output / input * 100)
        - peeled_btc: float (input - largest output - fee)
        - role: str
    """
    if "txid" not in df.columns:
        return []

    # Map txid to row
    tx_rows: Dict[str, pd.Series] = {}
    for _, r in df.iterrows():
        tx_rows[str(r["txid"])] = r

    if str(txid) not in tx_rows:
        return []

    if spend_index is None:
        spend_index = build_spend_index(df)

    start_row = tx_rows[str(txid)]
    start_in_amounts = _parse_amounts(start_row.get("input_amounts"))
    initial_input = (
        sum(start_in_amounts)
        if start_in_amounts
        else float(start_row.get("amount_btc", 0.0))
    )

    # Queue items: (txid, hop, parent_txid, parent_timestamp)
    queue: deque[Tuple[str, int, Optional[str], Any]] = deque([(str(txid), 0, None, None)])
    visited = {str(txid)}
    hops_result: List[Dict[str, Any]] = []

    while queue and len(hops_result) < 200:
        curr_txid, curr_hop, parent_txid, parent_ts = queue.popleft()
        row = tx_rows[curr_txid]

        curr_ts = row.get("timestamp")
        delta_t_minutes = 0.0
        if parent_ts is not None and curr_ts is not None:
            try:
                dt_curr = pd.to_datetime(curr_ts)
                dt_parent = pd.to_datetime(parent_ts)
                delta_t_minutes = round(
                    max(0.0, (dt_curr - dt_parent).total_seconds() / 60.0), 1
                )
            except Exception:
                delta_t_minutes = 0.0

        in_amounts = _parse_amounts(row.get("input_amounts"))
        btc_in = (
            round(sum(in_amounts), 8)
            if in_amounts
            else round(float(row.get("amount_btc", 0.0)), 8)
        )

        out_amounts = _parse_amounts(row.get("output_amounts"))
        largest_output_btc = round(max(out_amounts), 8) if out_amounts else 0.0

        fee = (
            float(row.get("fee", 0.0))
            if pd.notna(row.get("fee"))
            else 0.0
        )
        peeled_btc = max(0.0, round(btc_in - largest_output_btc - fee, 8))

        # Retained percentage: fraction of initial input funds retained in main trunk
        denom = initial_input if initial_input > 0 else (btc_in if btc_in > 0 else 1.0)
        retained_pct = round(min(100.0, max(0.0, (largest_output_btc / denom) * 100.0)), 4)

        role = _classify_hop_role(row, curr_hop)

        whitelisted_entity = row.get("whitelisted_entity") or row.get("peel_destination_type")

        hop_dict = {
            "hop": curr_hop,
            "txid": curr_txid,
            "timestamp": curr_ts,
            "delta_t_minutes": delta_t_minutes,
            "btc_in": btc_in,
            "largest_output_btc": largest_output_btc,
            "retained_pct": retained_pct,
            "peeled_btc": peeled_btc,
            "role": role,
            "parent_txid": parent_txid,
            "whitelisted_entity": whitelisted_entity,
            "fee": fee,
        }
        hops_result.append(hop_dict)

        if len(hops_result) >= 200:
            break

        # Stop condition: stop at exchange endpoints and mixers, or at max_hops
        if curr_hop >= max_hops:
            continue
        if curr_hop > 0 and role in ("Exchange endpoint", "Mixer"):
            continue

        # Expand next spending transactions
        out_addrs = _parse_addresses(row.get("output_addresses"))
        curr_dt = None
        if curr_ts is not None:
            try:
                curr_dt = pd.to_datetime(curr_ts)
            except Exception:
                curr_dt = None

        candidate_txs: List[Tuple[str, Any, float]] = []
        for addr in out_addrs:
            for next_txid, next_ts in spend_index.get(addr, []):
                if next_txid not in visited and next_txid in tx_rows:
                    is_later = True
                    if curr_dt is not None and next_ts is not None:
                        try:
                            next_dt = pd.to_datetime(next_ts)
                            is_later = next_dt > curr_dt
                        except Exception:
                            is_later = True
                    if is_later:
                        # Prioritize higher btc_in / main flow
                        next_row = tx_rows[next_txid]
                        next_in = sum(_parse_amounts(next_row.get("input_amounts")))
                        candidate_txs.append((next_txid, next_ts, next_in))

        # Sort candidates descending by input amount so primary trunk is traversed first
        candidate_txs.sort(key=lambda x: x[2], reverse=True)
        for c_txid, c_ts, _ in candidate_txs:
            if c_txid not in visited:
                visited.add(c_txid)
                queue.append((c_txid, curr_hop + 1, curr_txid, curr_ts))

    return hops_result


def trace_backward(
    df: pd.DataFrame,
    txid: str,
    max_hops: int = 6,
    creation_index: Optional[Dict[str, List[Tuple[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Trace backward hop-by-hop fund flow.
    Follow input addresses back to the transactions that created them (timestamp < current).

    Stops at exchange endpoints and mixers, at max_hops, or after 200 nodes.
    """
    if "txid" not in df.columns:
        return []

    tx_rows: Dict[str, pd.Series] = {}
    for _, r in df.iterrows():
        tx_rows[str(r["txid"])] = r

    if str(txid) not in tx_rows:
        return []

    if creation_index is None:
        creation_index = build_creation_index(df)

    start_row = tx_rows[str(txid)]
    start_in_amounts = _parse_amounts(start_row.get("input_amounts"))
    initial_input = (
        sum(start_in_amounts)
        if start_in_amounts
        else float(start_row.get("amount_btc", 0.0))
    )

    queue: deque[Tuple[str, int, Optional[str], Any]] = deque([(str(txid), 0, None, None)])
    visited = {str(txid)}
    hops_result: List[Dict[str, Any]] = []

    while queue and len(hops_result) < 200:
        curr_txid, curr_hop, parent_txid, parent_ts = queue.popleft()
        row = tx_rows[curr_txid]

        curr_ts = row.get("timestamp")
        delta_t_minutes = 0.0
        if parent_ts is not None and curr_ts is not None:
            try:
                dt_curr = pd.to_datetime(curr_ts)
                dt_parent = pd.to_datetime(parent_ts)
                delta_t_minutes = round(
                    max(0.0, abs((dt_parent - dt_curr).total_seconds()) / 60.0), 1
                )
            except Exception:
                delta_t_minutes = 0.0

        in_amounts = _parse_amounts(row.get("input_amounts"))
        btc_in = (
            round(sum(in_amounts), 8)
            if in_amounts
            else round(float(row.get("amount_btc", 0.0)), 8)
        )

        out_amounts = _parse_amounts(row.get("output_amounts"))
        largest_output_btc = round(max(out_amounts), 8) if out_amounts else 0.0

        fee = (
            float(row.get("fee", 0.0))
            if pd.notna(row.get("fee"))
            else 0.0
        )
        peeled_btc = max(0.0, round(btc_in - largest_output_btc - fee, 8))

        denom = initial_input if initial_input > 0 else (btc_in if btc_in > 0 else 1.0)
        retained_pct = round(min(100.0, max(0.0, (largest_output_btc / denom) * 100.0)), 4)

        role = _classify_hop_role(row, curr_hop)

        whitelisted_entity = row.get("whitelisted_entity") or row.get("peel_destination_type")

        hop_dict = {
            "hop": curr_hop,
            "txid": curr_txid,
            "timestamp": curr_ts,
            "delta_t_minutes": delta_t_minutes,
            "btc_in": btc_in,
            "largest_output_btc": largest_output_btc,
            "retained_pct": retained_pct,
            "peeled_btc": peeled_btc,
            "role": role,
            "parent_txid": parent_txid,
            "whitelisted_entity": whitelisted_entity,
            "fee": fee,
        }
        hops_result.append(hop_dict)

        if len(hops_result) >= 200:
            break

        if curr_hop >= max_hops:
            continue
        if curr_hop > 0 and role in ("Exchange endpoint", "Mixer"):
            continue

        # Look up txs that created this tx's inputs
        in_addrs = _parse_addresses(row.get("input_addresses"))
        curr_dt = None
        if curr_ts is not None:
            try:
                curr_dt = pd.to_datetime(curr_ts)
            except Exception:
                curr_dt = None

        candidate_txs: List[Tuple[str, Any, float]] = []
        for addr in in_addrs:
            for prev_txid, prev_ts in creation_index.get(addr, []):
                if prev_txid not in visited and prev_txid in tx_rows:
                    is_earlier = True
                    if curr_dt is not None and prev_ts is not None:
                        try:
                            prev_dt = pd.to_datetime(prev_ts)
                            is_earlier = prev_dt < curr_dt
                        except Exception:
                            is_earlier = True
                    if is_earlier:
                        prev_row = tx_rows[prev_txid]
                        prev_in = sum(_parse_amounts(prev_row.get("input_amounts")))
                        candidate_txs.append((prev_txid, prev_ts, prev_in))

        candidate_txs.sort(key=lambda x: x[2], reverse=True)
        for c_txid, c_ts, _ in candidate_txs:
            if c_txid not in visited:
                visited.add(c_txid)
                queue.append((c_txid, curr_hop + 1, curr_txid, curr_ts))

    return hops_result
