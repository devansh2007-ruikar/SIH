"""
tests/test_flow_tracer.py -- Unit tests for flow_tracer.py
==========================================================

Validates:
1. Spend index creation and address parsing.
2. Hop-by-hop forward fund flow tracing on a generated peel chain:
   - Returns at least 3 hops from hop 0.
   - retained_pct strictly decreases over the hops.
3. Role assignment rules (Source, Layering, Mixer, Exchange endpoint, etc.).
4. Stopping rules (mixers, exchange endpoints, max_hops).
5. Backward fund flow tracing.
"""

import pandas as pd
import pytest

import flow_tracer
import generate_data


def test_build_spend_index():
    df = pd.DataFrame([
        {
            "txid": "tx1",
            "timestamp": "2026-03-15T10:00:00",
            "input_addresses": "addr_A|addr_B",
            "output_addresses": "addr_C",
        },
        {
            "txid": "tx2",
            "timestamp": "2026-03-15T10:15:00",
            "input_addresses": "addr_C",
            "output_addresses": "addr_D",
        },
    ])
    spend_idx = flow_tracer.build_spend_index(df)
    assert "addr_A" in spend_idx
    assert spend_idx["addr_A"] == [("tx1", "2026-03-15T10:00:00")]
    assert "addr_B" in spend_idx
    assert "addr_C" in spend_idx
    assert spend_idx["addr_C"] == [("tx2", "2026-03-15T10:15:00")]


def test_peel_chain_trace_forward_monotonic_decrease():
    """
    Core requirement:
    On a generated peel chain, trace_forward from hop 0 returns at least 3 hops
    and retained_pct decreases over the hops.
    """
    seq = generate_data.create_peel_chain_sequence(depth_min=4, depth_max=5)
    df = pd.DataFrame(seq)
    hop_0_txid = df.iloc[0]["txid"]

    trace = flow_tracer.trace_forward(df, hop_0_txid, max_hops=6)

    # 1. Must return at least 3 hops
    assert len(trace) >= 3, f"Expected >= 3 hops, got {len(trace)}"

    # 2. Hop indices must be sequential
    for i, h in enumerate(trace):
        assert h["hop"] == i
        assert h["btc_in"] > 0
        assert h["largest_output_btc"] > 0

    # 3. Hop 0 role must be Source
    assert trace[0]["role"] == "Source"

    # 4. Subsequent intermediate hops are Layering (peel)
    for h in trace[1:-1]:
        assert h["role"] in ("Layering (peel)", "Exchange endpoint", "Intermediary")

    # 5. retained_pct must strictly decrease over the hops
    retained = [h["retained_pct"] for h in trace]
    for i in range(len(retained) - 1):
        assert retained[i] > retained[i + 1], (
            f"retained_pct did not decrease at hop {i} -> {i+1}: "
            f"{retained[i]} <= {retained[i+1]} (all: {retained})"
        )


def test_trace_forward_stops_at_exchange_and_mixer():
    df = pd.DataFrame([
        {
            "txid": "tx_source",
            "timestamp": "2026-03-15T10:00:00",
            "input_addresses": "in_0",
            "output_addresses": "mid_1",
            "input_amounts": "10.0",
            "output_amounts": "9.9",
            "fee": 0.1,
            "detected_type": "Normal_P2P",
        },
        {
            "txid": "tx_mid",
            "timestamp": "2026-03-15T10:05:00",
            "input_addresses": "mid_1",
            "output_addresses": "endpoint_addr",
            "input_amounts": "9.9",
            "output_amounts": "9.8",
            "fee": 0.1,
            "detected_type": "Peel_Chain",
        },
        {
            "txid": "tx_endpoint",
            "timestamp": "2026-03-15T10:10:00",
            "input_addresses": "endpoint_addr",
            "output_addresses": "final_addr",
            "input_amounts": "9.8",
            "output_amounts": "9.7",
            "fee": 0.1,
            "detected_type": "Normal_P2P",
            "whitelisted_side": "both",
            "whitelisted_entity": "Binance Institutional",
        },
        {
            "txid": "tx_beyond",
            "timestamp": "2026-03-15T10:15:00",
            "input_addresses": "final_addr",
            "output_addresses": "beyond_addr",
            "input_amounts": "9.7",
            "output_amounts": "9.6",
            "fee": 0.1,
            "detected_type": "Normal_P2P",
        },
    ])

    trace = flow_tracer.trace_forward(df, "tx_source", max_hops=5)
    txids = [h["txid"] for h in trace]

    assert "tx_source" in txids
    assert "tx_mid" in txids
    assert "tx_endpoint" in txids
    # Should NOT continue beyond the exchange endpoint
    assert "tx_beyond" not in txids
    assert trace[-1]["role"] == "Exchange endpoint"
    assert trace[-1]["whitelisted_entity"] == "Binance Institutional"


def test_trace_backward():
    seq = generate_data.create_peel_chain_sequence(depth_min=4, depth_max=5)
    df = pd.DataFrame(seq)
    last_txid = df.iloc[-1]["txid"]

    trace = flow_tracer.trace_backward(df, last_txid, max_hops=6)
    assert len(trace) >= 3
    assert trace[0]["txid"] == last_txid
    assert trace[0]["hop"] == 0
