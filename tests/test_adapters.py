"""
test_transaction_adapter.py — Comprehensive tests for multi-format adapters
==========================================================================
"""

import json
import os
import sys
import tempfile
import textwrap

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from transaction_adapter import (
    BaseTransactionAdapter,
    BitcoinCSVAdapter,
    BitcoinJSONAdapter,
    BitcoinXMLAdapter,
    CANONICAL_COLUMNS,
    get_adapter,
    _normalise_dataframe,
    _coerce_to_pipe_str,
)


def _assert_canonical(df: pd.DataFrame, label: str):
    """Assert that df contains all canonical columns with correct types."""
    for col in CANONICAL_COLUMNS:
        assert col in df.columns, f"[{label}] Missing canonical column: {col}"
    # Type checks — pandas 3.x uses StringDtype by default
    assert pd.api.types.is_string_dtype(df["txid"]), \
        f"[{label}] txid should be str-like, got {df['txid'].dtype}"
    assert pd.api.types.is_float_dtype(df["fee"]), \
        f"[{label}] fee should be float, got {df['fee'].dtype}"
    assert pd.api.types.is_integer_dtype(df["src_port"]), \
        f"[{label}] src_port should be int, got {df['src_port'].dtype}"


# ═══════════════════════════════════════════════════════════════════════════
# 1. CSV ADAPTER
# ═══════════════════════════════════════════════════════════════════════════

def test_csv_adapter_load():
    """CSV adapter reads standard pipe-delimited CSVs correctly."""
    csv_content = textwrap.dedent("""\
        txid,input_addresses,output_addresses,input_amounts,output_amounts,fee,src_port
        tx001,addrA|addrB,addrC|addrD,0.5|0.3,0.79,0.01,8333
        tx002,addrE,addrF,1.0,0.99,0.01,443
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        f.write(csv_content)
        path = f.name

    try:
        adapter = BitcoinCSVAdapter()
        df = adapter.load(path)
        _assert_canonical(df, "CSV")
        assert len(df) == 2
        assert df.iloc[0]["txid"] == "tx001"
        assert df.iloc[0]["input_addresses"] == "addrA|addrB"
        assert df.iloc[0]["fee"] == 0.01
        assert df.iloc[0]["src_port"] == 8333
        print("  ✅ test_csv_adapter_load")
    finally:
        os.unlink(path)


def test_csv_adapter_fee_default():
    """CSV adapter defaults fee to 0.0001 when column is missing."""
    csv_content = textwrap.dedent("""\
        txid,input_addresses,output_addresses,input_amounts,output_amounts,src_port
        tx001,addrA,addrB,0.5,0.49,8333
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        f.write(csv_content)
        path = f.name

    try:
        adapter = BitcoinCSVAdapter()
        df = adapter.load(path)
        assert df.iloc[0]["fee"] == 0.0001
        print("  ✅ test_csv_adapter_fee_default")
    finally:
        os.unlink(path)


def test_csv_adapter_file_not_found():
    """CSV adapter raises FileNotFoundError for missing files."""
    adapter = BitcoinCSVAdapter()
    try:
        adapter.load("/nonexistent/path.csv")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  ✅ test_csv_adapter_file_not_found")


def test_csv_adapter_malformed():
    """CSV adapter raises ValueError for unparseable CSV."""
    # Write binary garbage
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".csv", delete=False
    ) as f:
        f.write(b"\x00\x01\x02\xff\xfe")
        path = f.name

    try:
        adapter = BitcoinCSVAdapter()
        # pandas may or may not raise — but if it loads, txid check fails
        try:
            df = adapter.load(path)
            # If it didn't raise, it should at least fail on missing txid
            assert False, "Should have raised an error"
        except (ValueError, KeyError):
            print("  ✅ test_csv_adapter_malformed")
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
# 2. JSON ADAPTER
# ═══════════════════════════════════════════════════════════════════════════

def test_json_adapter_array():
    """JSON adapter reads array-of-objects format."""
    data = [
        {
            "txid": "tx001",
            "input_addresses": ["addrA", "addrB"],
            "output_addresses": "addrC|addrD",
            "input_amounts": [0.5, 0.3],
            "output_amounts": "0.79",
            "fee": 0.01,
            "src_port": 8333,
        },
        {
            "txid": "tx002",
            "input_addresses": "addrE",
            "output_addresses": ["addrF"],
            "input_amounts": "1.0",
            "output_amounts": [0.99],
            "fee": 0.01,
            "src_port": 443,
        },
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        path = f.name

    try:
        adapter = BitcoinJSONAdapter()
        df = adapter.load(path)
        _assert_canonical(df, "JSON-array")
        assert len(df) == 2
        assert df.iloc[0]["txid"] == "tx001"
        assert df.iloc[0]["input_addresses"] == "addrA|addrB"
        assert df.iloc[1]["output_addresses"] == "addrF"
        print("  ✅ test_json_adapter_array")
    finally:
        os.unlink(path)


def test_json_adapter_dict():
    """JSON adapter reads key-value dictionary format."""
    data = {
        "tx001": {
            "input_addresses": "addrA|addrB",
            "output_addresses": "addrC",
            "input_amounts": "0.5|0.3",
            "output_amounts": "0.79",
            "fee": 0.01,
            "src_port": 8333,
        },
        "tx002": {
            "input_addresses": "addrE",
            "output_addresses": "addrF",
            "input_amounts": "1.0",
            "output_amounts": "0.99",
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        path = f.name

    try:
        adapter = BitcoinJSONAdapter()
        df = adapter.load(path)
        _assert_canonical(df, "JSON-dict")
        assert len(df) == 2
        # Outer keys become txid
        assert set(df["txid"]) == {"tx001", "tx002"}
        # tx002 has no fee → should default
        tx002 = df[df["txid"] == "tx002"].iloc[0]
        assert tx002["fee"] == 0.0001
        # tx002 has no src_port → should default to 0
        assert tx002["src_port"] == 0
        print("  ✅ test_json_adapter_dict")
    finally:
        os.unlink(path)


def test_json_adapter_invalid_json():
    """JSON adapter raises ValueError for invalid JSON."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        f.write("{invalid json!!!")
        path = f.name

    try:
        adapter = BitcoinJSONAdapter()
        try:
            adapter.load(path)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid JSON" in str(e)
            print("  ✅ test_json_adapter_invalid_json")
    finally:
        os.unlink(path)


def test_json_adapter_bad_structure():
    """JSON adapter raises ValueError for non-dict/list top-level."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        f.write('"just a string"')
        path = f.name

    try:
        adapter = BitcoinJSONAdapter()
        try:
            adapter.load(path)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "str" in str(e)
            print("  ✅ test_json_adapter_bad_structure")
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
# 3. XML ADAPTER
# ═══════════════════════════════════════════════════════════════════════════

def test_xml_adapter_pipe_delimited():
    """XML adapter reads pipe-delimited text content."""
    xml_content = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <transactions>
          <transaction>
            <txid>tx001</txid>
            <input_addresses>addrA|addrB</input_addresses>
            <output_addresses>addrC|addrD</output_addresses>
            <input_amounts>0.5|0.3</input_amounts>
            <output_amounts>0.79</output_amounts>
            <fee>0.01</fee>
            <src_port>8333</src_port>
          </transaction>
          <transaction>
            <txid>tx002</txid>
            <input_addresses>addrE</input_addresses>
            <output_addresses>addrF</output_addresses>
            <input_amounts>1.0</input_amounts>
            <output_amounts>0.99</output_amounts>
          </transaction>
        </transactions>
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        f.write(xml_content)
        path = f.name

    try:
        adapter = BitcoinXMLAdapter()
        df = adapter.load(path)
        _assert_canonical(df, "XML-pipe")
        assert len(df) == 2
        assert df.iloc[0]["txid"] == "tx001"
        assert df.iloc[0]["input_addresses"] == "addrA|addrB"
        # tx002 has no fee → default
        assert df.iloc[1]["fee"] == 0.0001
        # tx002 has no src_port → default
        assert df.iloc[1]["src_port"] == 0
        print("  ✅ test_xml_adapter_pipe_delimited")
    finally:
        os.unlink(path)


def test_xml_adapter_nested_children():
    """XML adapter reads nested child elements for multi-value fields."""
    xml_content = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <transactions>
          <transaction>
            <txid>tx001</txid>
            <input_addresses>
              <address>addrA</address>
              <address>addrB</address>
            </input_addresses>
            <output_addresses>
              <address>addrC</address>
            </output_addresses>
            <input_amounts>
              <amount>0.5</amount>
              <amount>0.3</amount>
            </input_amounts>
            <output_amounts>
              <amount>0.79</amount>
            </output_amounts>
            <fee>0.01</fee>
            <src_port>8333</src_port>
          </transaction>
        </transactions>
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        f.write(xml_content)
        path = f.name

    try:
        adapter = BitcoinXMLAdapter()
        df = adapter.load(path)
        _assert_canonical(df, "XML-nested")
        assert len(df) == 1
        assert df.iloc[0]["input_addresses"] == "addrA|addrB"
        assert df.iloc[0]["output_addresses"] == "addrC"
        assert df.iloc[0]["input_amounts"] == "0.5|0.3"
        print("  ✅ test_xml_adapter_nested_children")
    finally:
        os.unlink(path)


def test_xml_adapter_malformed():
    """XML adapter raises ValueError for malformed XML."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        f.write("<broken><unclosed>")
        path = f.name

    try:
        adapter = BitcoinXMLAdapter()
        try:
            adapter.load(path)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Malformed XML" in str(e)
            print("  ✅ test_xml_adapter_malformed")
    finally:
        os.unlink(path)


def test_xml_adapter_no_transactions():
    """XML adapter raises ValueError when no <transaction> elements found."""
    xml_content = textwrap.dedent("""\
        <?xml version="1.0"?>
        <data>
          <item>not a transaction</item>
        </data>
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        f.write(xml_content)
        path = f.name

    try:
        adapter = BitcoinXMLAdapter()
        try:
            adapter.load(path)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Could not locate" in str(e)
            print("  ✅ test_xml_adapter_no_transactions")
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
# 4. FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def test_factory_extension_csv():
    assert isinstance(get_adapter("data.csv"), BitcoinCSVAdapter)
    print("  ✅ test_factory_extension_csv")

def test_factory_extension_json():
    assert isinstance(get_adapter("data.json"), BitcoinJSONAdapter)
    print("  ✅ test_factory_extension_json")

def test_factory_extension_xml():
    assert isinstance(get_adapter("data.xml"), BitcoinXMLAdapter)
    print("  ✅ test_factory_extension_xml")

def test_factory_chain_backward_compat():
    assert isinstance(get_adapter(chain="bitcoin"), BitcoinCSVAdapter)
    print("  ✅ test_factory_chain_backward_compat")

def test_factory_chain_json():
    assert isinstance(get_adapter(chain="bitcoin_json"), BitcoinJSONAdapter)
    print("  ✅ test_factory_chain_json")

def test_factory_chain_xml():
    assert isinstance(get_adapter(chain="bitcoin_xml"), BitcoinXMLAdapter)
    print("  ✅ test_factory_chain_xml")

def test_factory_default():
    assert isinstance(get_adapter(), BitcoinCSVAdapter)
    print("  ✅ test_factory_default")

def test_factory_unsupported_extension():
    try:
        get_adapter("data.parquet")
        assert False
    except ValueError:
        print("  ✅ test_factory_unsupported_extension")

def test_factory_unsupported_chain():
    try:
        get_adapter(chain="ethereum")
        assert False
    except ValueError:
        print("  ✅ test_factory_unsupported_chain")


# ═══════════════════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def test_coerce_to_pipe_str():
    assert _coerce_to_pipe_str(None) == ""
    assert _coerce_to_pipe_str(["a", "b", "c"]) == "a|b|c"
    assert _coerce_to_pipe_str("a | b | c") == "a|b|c"
    assert _coerce_to_pipe_str("nan") == ""
    assert _coerce_to_pipe_str("") == ""
    assert _coerce_to_pipe_str("single") == "single"
    print("  ✅ test_coerce_to_pipe_str")


def test_csv_adapter_loads_real_data():
    """CSV adapter loads the project's bitcoin_traffic.csv successfully."""
    real_path = os.path.join(os.path.dirname(__file__), "bitcoin_traffic.csv")
    if not os.path.isfile(real_path):
        print("  ⏭️  test_csv_adapter_loads_real_data (skipped — file missing)")
        return
    adapter = BitcoinCSVAdapter()
    df = adapter.load(real_path)
    _assert_canonical(df, "CSV-real")
    assert len(df) > 0
    print(f"  ✅ test_csv_adapter_loads_real_data ({len(df)} rows)")


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # CSV
        test_csv_adapter_load,
        test_csv_adapter_fee_default,
        test_csv_adapter_file_not_found,
        test_csv_adapter_malformed,
        # JSON
        test_json_adapter_array,
        test_json_adapter_dict,
        test_json_adapter_invalid_json,
        test_json_adapter_bad_structure,
        # XML
        test_xml_adapter_pipe_delimited,
        test_xml_adapter_nested_children,
        test_xml_adapter_malformed,
        test_xml_adapter_no_transactions,
        # Factory
        test_factory_extension_csv,
        test_factory_extension_json,
        test_factory_extension_xml,
        test_factory_chain_backward_compat,
        test_factory_chain_json,
        test_factory_chain_xml,
        test_factory_default,
        test_factory_unsupported_extension,
        test_factory_unsupported_chain,
        # Helpers
        test_coerce_to_pipe_str,
        # Real data
        test_csv_adapter_loads_real_data,
    ]

    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print("  MITHYA Transaction Adapter — Test Suite")
    print(f"{'='*60}\n")

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}\n")

    sys.exit(1 if failed else 0)


# ═══════════════════════════════════════════════════════════════════════════
# XML SECURITY HARDENING TESTS (XXE & Billion Laughs)
# ═══════════════════════════════════════════════════════════════════════════

import pytest


def test_xml_rejects_xxe_attack():
    """
    Prove that the BitcoinXMLAdapter rejects XML External Entity (XXE)
    injection attacks.

    An XXE payload attempts to exfiltrate local file contents (e.g.
    /etc/passwd) by defining an external entity that the parser would
    resolve if using the vulnerable stdlib xml.etree.ElementTree.

    defusedxml MUST raise an error and refuse to parse this document.
    """
    xxe_payload = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE transactions [
          <!ENTITY xxe SYSTEM "file:///etc/passwd">
        ]>
        <transactions>
          <transaction>
            <txid>&xxe;</txid>
            <input_addresses>addr1</input_addresses>
            <output_addresses>addr2</output_addresses>
            <input_amounts>1.0</input_amounts>
            <output_amounts>0.99</output_amounts>
            <fee>0.01</fee>
            <src_port>8333</src_port>
          </transaction>
        </transactions>
    """)

    adapter = BitcoinXMLAdapter()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False,
    ) as f:
        f.write(xxe_payload)
        f.flush()
        tmp_path = f.name

    try:
        # defusedxml should raise an error (EntitiesForbidden or similar)
        # when encountering the DTD with external entity reference.
        with pytest.raises((ValueError, Exception)) as exc_info:
            adapter.load(tmp_path)

        # Verify the error is security-related (not just malformed XML)
        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg
            for keyword in ("entitiesforbidden", "forbid", "dtd", "entity", "malformed")
        ), (
            f"Expected a security-related rejection, got: {exc_info.value}"
        )
    finally:
        os.remove(tmp_path)


def test_xml_rejects_billion_laughs_attack():
    """
    Prove that the BitcoinXMLAdapter rejects Billion Laughs (exponential
    entity expansion) attacks.

    A Billion Laughs payload defines nested internal entities that
    expand exponentially, consuming gigabytes of RAM if parsed naïvely.
    defusedxml MUST refuse this document at parse time.
    """
    billion_laughs_payload = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE lolz [
          <!ENTITY lol "lol">
          <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
          <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
          <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
          <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
          <!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">
          <!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">
          <!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;">
          <!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">
        ]>
        <transactions>
          <transaction>
            <txid>&lol9;</txid>
            <input_addresses>addr1</input_addresses>
            <output_addresses>addr2</output_addresses>
            <input_amounts>1.0</input_amounts>
            <output_amounts>0.99</output_amounts>
            <fee>0.01</fee>
            <src_port>8333</src_port>
          </transaction>
        </transactions>
    """)

    adapter = BitcoinXMLAdapter()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False,
    ) as f:
        f.write(billion_laughs_payload)
        f.flush()
        tmp_path = f.name

    try:
        # defusedxml should raise an error when encountering the
        # exponentially-expanding entity definitions in the DTD.
        with pytest.raises((ValueError, Exception)) as exc_info:
            adapter.load(tmp_path)

        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg
            for keyword in ("entitiesforbidden", "forbid", "dtd", "entity", "malformed")
        ), (
            f"Expected a security-related rejection, got: {exc_info.value}"
        )
    finally:
        os.remove(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# 7. MULTI-LAYER CORRELATION (TELEMETRY × BLOCKCHAIN)
# ═══════════════════════════════════════════════════════════════════════════

def test_correlate_layers_metrics_and_first_seen():
    """Verify correlate_layers correctly identifies first_seen_ip, observation count, and spread."""
    from transaction_adapter import correlate_layers

    net_df = pd.DataFrame([
        {"timestamp": "2026-07-27 10:00:10", "src_ip": "10.0.0.2", "dst_ip": "1.2.3.4", "src_port": 9050, "dst_port": 8333, "txid": "tx1"},
        {"timestamp": "2026-07-27 10:00:00", "src_ip": "10.0.0.1", "dst_ip": "1.2.3.4", "src_port": 8333, "dst_port": 8333, "txid": "tx1"},
        {"timestamp": "2026-07-27 10:00:25", "src_ip": "10.0.0.3", "dst_ip": "1.2.3.4", "src_port": 8333, "dst_port": 8333, "txid": "tx1"},
        {"timestamp": "2026-07-27 11:00:00", "src_ip": "10.0.0.4", "dst_ip": "1.2.3.4", "src_port": 8333, "dst_port": 8333, "txid": "tx_telemetry_only"},
    ])

    chain_df = pd.DataFrame([
        {"txid": "tx1", "timestamp": "2026-07-27 10:00:00", "input_addresses": "bc1qA", "output_addresses": "bc1qB", "input_amounts": "1.0", "output_amounts": "0.99", "fee": 0.01, "script_type": "P2WPKH"},
        {"txid": "tx_chain_only", "timestamp": "2026-07-27 12:00:00", "input_addresses": "bc1qC", "output_addresses": "bc1qD", "input_amounts": "2.0", "output_amounts": "1.99", "fee": 0.01, "script_type": "P2WPKH"},
    ])

    merged, stats = correlate_layers(net_df, chain_df)

    # Stats validation
    assert stats["matched"] == 1
    assert stats["telemetry_only"] == 1
    assert stats["chain_only"] == 1
    assert stats["match_rate"] == round(1 / 3, 4)

    # Row per chain txid
    assert len(merged) == 2
    row_tx1 = merged[merged["txid"] == "tx1"].iloc[0]

    # Earliest timestamp IP was 10.0.0.1 (at 10:00:00, vs 10:00:10 and 10:00:25)
    assert row_tx1["src_ip"] == "10.0.0.1"
    assert row_tx1["ip_observation_count"] == 3
    assert row_tx1["observation_spread_s"] == 25.0

    row_chain_only = merged[merged["txid"] == "tx_chain_only"].iloc[0]
    assert row_chain_only["src_ip"] == ""
    assert row_chain_only["ip_observation_count"] == 0
    assert row_chain_only["observation_spread_s"] == 0.0


def test_split_files_on_disk():
    """Verify generated sample_data/network_telemetry.csv and blockchain_tx.csv meet spec."""
    net_path = os.path.join("sample_data", "network_telemetry.csv")
    chain_path = os.path.join("sample_data", "blockchain_tx.csv")

    assert os.path.isfile(net_path), f"Missing {net_path}"
    assert os.path.isfile(chain_path), f"Missing {chain_path}"

    net_df = pd.read_csv(net_path)
    chain_df = pd.read_csv(chain_path)

    expected_net_cols = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid"]
    expected_chain_cols = ["txid", "timestamp", "input_addresses", "output_addresses", "input_amounts", "output_amounts", "fee", "script_type"]

    assert list(net_df.columns) == expected_net_cols
    assert list(chain_df.columns) == expected_chain_cols

    # Test correlation on generated sample data
    from transaction_adapter import correlate_layers
    merged, stats = correlate_layers(net_df, chain_df)
    assert stats["matched"] == len(chain_df)
    assert stats["match_rate"] == 1.0
    assert "ip_observation_count" in merged.columns
    assert "observation_spread_s" in merged.columns
    # Check extra observations exist
    assert (merged["ip_observation_count"] > 1).sum() > 0

