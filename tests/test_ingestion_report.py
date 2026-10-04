"""
tests/test_ingestion_report.py — Ingestion Validation Report Tests
================================================================
Verifies that BaseTransactionAdapter tracks self.last_report correctly,
rejects invalid rows (missing txid, unparseable timestamp, negative/non-numeric
amounts, empty addresses, duplicate txids) without crashing the pipeline,
and keeps the first copy of duplicate txids while counting the rest.
"""

import json
import os
import sys
import tempfile
import textwrap
import pytest
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transaction_adapter import BitcoinCSVAdapter, BitcoinJSONAdapter, BitcoinXMLAdapter


def test_ingestion_report_csv_rejections():
    """
    Test requirement:
    A CSV with 10 good rows plus:
    - 1 missing txid
    - 1 bad timestamp
    - 1 duplicate txid
    → rows_valid == 10, rows_rejected == 3, and reasons counted correctly.
    """
    csv_lines = [
        "timestamp,src_ip,dst_ip,src_port,dst_port,txid,input_addresses,output_addresses,input_amounts,output_amounts,fee,script_type,geo_country"
    ]

    # 10 good rows
    for i in range(1, 11):
        txid = f"tx{i:03d}"
        csv_lines.append(
            f"2026-07-27 12:{i:02d}:00,10.0.0.1,10.0.0.2,8333,8333,{txid},1AddrIn{i},1AddrOut{i},1.0,0.99,0.01,P2PKH,US"
        )

    # 1 missing txid (empty txid)
    csv_lines.append(
        "2026-07-27 12:11:00,10.0.0.1,10.0.0.2,8333,8333,,1AddrIn11,1AddrOut11,1.0,0.99,0.01,P2PKH,US"
    )

    # 1 bad timestamp
    csv_lines.append(
        "INVALID_DATETIME_STRING,10.0.0.1,10.0.0.2,8333,8333,tx012,1AddrIn12,1AddrOut12,1.0,0.99,0.01,P2PKH,US"
    )

    # 1 duplicate txid (repeating tx001)
    csv_lines.append(
        "2026-07-27 12:13:00,10.0.0.1,10.0.0.2,8333,8333,tx001,1AddrIn01_dup,1AddrOut01_dup,2.0,1.99,0.01,P2PKH,US"
    )

    csv_content = "\n".join(csv_lines) + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        adapter = BitcoinCSVAdapter()
        df = adapter.load(tmp_path)

        report = adapter.last_report
        assert report is not None
        assert report["format"] == "CSV"
        assert report["rows_read"] == 13
        assert report["rows_valid"] == 10
        assert report["rows_rejected"] == 3
        assert len(df) == 10

        # Verify duplicate txids count
        assert report["duplicate_txids"] == 1

        # Check rejection reasons
        reasons = report["reject_reasons"]
        assert reasons["missing_txid"] == 1
        assert reasons["unparseable_timestamp"] == 1
        assert reasons["duplicate_txid"] == 1

        # Verify first copy of duplicate txid is kept
        assert df["txid"].iloc[0] == "tx001"
        assert df["input_addresses"].iloc[0] == "1AddrIn1"

        # Check sha256 and timestamp range
        assert len(report["sha256"]) == 64
        assert report["timestamp_range"][0] is not None
        assert report["timestamp_range"][1] is not None
        assert report["timestamp_range"][0] <= report["timestamp_range"][1]

        # Verify JSON serializability
        report_json = json.dumps(report)
        assert isinstance(report_json, str)

        print("  ✅ test_ingestion_report_csv_rejections passed")
    finally:
        os.remove(tmp_path)


def test_ingestion_report_negative_amounts_and_empty_addresses():
    """
    Verify rejection of rows with negative amounts, non-numeric amounts,
    and empty addresses without crashing.
    """
    csv_content = textwrap.dedent("""\
        timestamp,src_ip,dst_ip,src_port,dst_port,txid,input_addresses,output_addresses,input_amounts,output_amounts,fee,script_type,geo_country
        2026-07-27 12:00:00,10.0.0.1,10.0.0.2,8333,8333,tx_valid,addr1,addr2,1.0,0.99,0.01,P2PKH,US
        2026-07-27 12:01:00,10.0.0.1,10.0.0.2,8333,8333,tx_neg_in,addr1,addr2,-0.5,0.49,0.01,P2PKH,US
        2026-07-27 12:02:00,10.0.0.1,10.0.0.2,8333,8333,tx_non_num,addr1,addr2,not_a_number,0.49,0.01,P2PKH,US
        2026-07-27 12:03:00,10.0.0.1,10.0.0.2,8333,8333,tx_empty_in,,addr2,1.0,0.99,0.01,P2PKH,US
        2026-07-27 12:04:00,10.0.0.1,10.0.0.2,8333,8333,tx_empty_out,addr1,,1.0,0.99,0.01,P2PKH,US
        2026-07-27 12:05:00,10.0.0.1,10.0.0.2,8333,8333,tx_neg_fee,addr1,addr2,1.0,0.99,-0.05,P2PKH,US
    """)

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        adapter = BitcoinCSVAdapter()
        df = adapter.load(tmp_path)

        report = adapter.last_report
        assert report["rows_read"] == 6
        assert report["rows_valid"] == 1
        assert report["rows_rejected"] == 5
        assert len(df) == 1
        assert df["txid"].iloc[0] == "tx_valid"

        reasons = report["reject_reasons"]
        assert reasons["empty_addresses"] == 2
        assert reasons["negative_or_non_numeric_amount"] == 3
        print("  ✅ test_ingestion_report_negative_amounts_and_empty_addresses passed")
    finally:
        os.remove(tmp_path)


def test_ingestion_report_pipeline_execution():
    """
    Ensure run_pipeline works seamlessly when invalid rows are present in the source file,
    enriching and scoring only the valid rows.
    """
    csv_content = textwrap.dedent("""\
        timestamp,src_ip,dst_ip,src_port,dst_port,txid,input_addresses,output_addresses,input_amounts,output_amounts,fee,script_type,geo_country
        2026-07-27 12:00:00,10.0.0.1,10.0.0.2,8333,8333,tx001,1AddrA,1AddrB,1.0,0.99,0.01,P2PKH,US
        2026-07-27 12:01:00,10.0.0.1,10.0.0.2,8333,8333,tx002,1AddrC,1AddrD,2.0,1.99,0.01,P2PKH,US
        2026-07-27 12:02:00,10.0.0.1,10.0.0.2,8333,8333,tx003,1AddrE,1AddrF,3.0,2.99,0.01,P2PKH,US
        2026-07-27 12:03:00,10.0.0.1,10.0.0.2,8333,8333,tx004,1AddrG,1AddrH,4.0,3.99,0.01,P2PKH,US
        2026-07-27 12:04:00,10.0.0.1,10.0.0.2,8333,8333,tx005,1AddrI,1AddrJ,5.0,4.99,0.01,P2PKH,US
        bad_time,10.0.0.1,10.0.0.2,8333,8333,tx006,1AddrK,1AddrL,6.0,5.99,0.01,P2PKH,US
        2026-07-27 12:06:00,10.0.0.1,10.0.0.2,8333,8333,,1AddrM,1AddrN,7.0,6.99,0.01,P2PKH,US
    """)

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        adapter = BitcoinCSVAdapter()
        enriched_df, model, features = adapter.run_pipeline(tmp_path)

        assert adapter.last_report["rows_read"] == 7
        assert adapter.last_report["rows_valid"] == 5
        assert adapter.last_report["rows_rejected"] == 2
        assert len(enriched_df) == 5
        assert "risk_score" in enriched_df.columns
        assert "risk_tier" in enriched_df.columns
        print("  ✅ test_ingestion_report_pipeline_execution passed")
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    test_ingestion_report_csv_rejections()
    test_ingestion_report_negative_amounts_and_empty_addresses()
    test_ingestion_report_pipeline_execution()
    print("\n🎉 All ingestion report tests passed successfully!")
