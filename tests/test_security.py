"""Tests for upload size / row-count safety limits in transaction_adapter.

Verifies that:
- An oversized file is rejected BEFORE parsing.
- A file with too many rows is rejected AFTER parsing.
- A normal-size file loads without error.
- Both CSV and JSON adapters respect the limits.
"""

import os
import sys

import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import transaction_adapter as ta
from transaction_adapter import (
    BitcoinCSVAdapter,
    BitcoinJSONAdapter,
    _validate_file_size,
    _validate_row_count,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _write_csv(path, n_rows=5):
    """Write a minimal valid CSV with *n_rows* transactions."""
    rows = []
    for i in range(n_rows):
        rows.append({
            "txid": f"tx_{i:04d}",
            "input_addresses": f"addr_in_{i}",
            "output_addresses": f"addr_out_{i}",
            "input_amounts": "0.5",
            "output_amounts": "0.49",
            "fee": "0.01",
            "timestamp": "2026-01-01T00:00:00",
            "src_ip": "192.168.1.1",
            "dst_ip": "10.0.0.1",
            "src_port": "8333",
            "dst_port": "443",
            "script_type": "P2PKH",
        })
    pd.DataFrame(rows).to_csv(str(path), index=False)


def _write_json(path, n_rows=5):
    """Write a minimal valid JSON array with *n_rows* transactions."""
    import json
    records = []
    for i in range(n_rows):
        records.append({
            "txid": f"tx_{i:04d}",
            "input_addresses": f"addr_in_{i}",
            "output_addresses": f"addr_out_{i}",
            "input_amounts": "0.5",
            "output_amounts": "0.49",
            "fee": 0.01,
            "timestamp": "2026-01-01T00:00:00",
            "src_ip": "192.168.1.1",
            "dst_ip": "10.0.0.1",
            "src_port": 8333,
            "dst_port": 443,
            "script_type": "P2PKH",
        })
    with open(str(path), "w") as f:
        json.dump(records, f)


# ── Unit tests for the validation helpers ───────────────────────────────────

class TestValidateFileSize:
    """Tests for _validate_file_size()."""

    def test_rejects_oversized_file(self, tmp_path):
        big_file = tmp_path / "big.csv"
        big_file.write_bytes(b"x" * 2048)  # 2 KB

        with pytest.raises(ValueError, match="File is too large"):
            _validate_file_size(big_file, max_bytes=1024)

    def test_accepts_normal_file(self, tmp_path):
        normal_file = tmp_path / "small.csv"
        normal_file.write_bytes(b"x" * 512)

        # Should not raise
        _validate_file_size(normal_file, max_bytes=1024)

    def test_respects_monkeypatched_global(self, tmp_path, monkeypatch):
        """The `if x is None: x = GLOBAL` pattern lets tests override."""
        test_file = tmp_path / "test.csv"
        test_file.write_bytes(b"x" * 100)

        # Monkeypatch the module global to 50 bytes
        monkeypatch.setattr(ta, "MAX_UPLOAD_FILE_SIZE_BYTES", 50)

        # Calling without explicit max_bytes should now use 50
        with pytest.raises(ValueError, match="File is too large"):
            _validate_file_size(test_file)  # uses global

    def test_nonexistent_path_skips(self):
        """Non-seekable / nonexistent path should silently skip (OSError catch)."""
        _validate_file_size("/nonexistent/path/foo.csv", max_bytes=1)
        # Should not raise


class TestValidateRowCount:
    """Tests for _validate_row_count()."""

    def test_rejects_too_many_rows(self):
        df = pd.DataFrame({"a": range(100)})
        with pytest.raises(ValueError, match="too many rows"):
            _validate_row_count(df, max_rows=50)

    def test_accepts_normal_row_count(self):
        df = pd.DataFrame({"a": range(10)})
        _validate_row_count(df, max_rows=50)  # should not raise

    def test_respects_monkeypatched_global(self, monkeypatch):
        monkeypatch.setattr(ta, "MAX_UPLOAD_ROWS", 5)
        df = pd.DataFrame({"a": range(10)})
        with pytest.raises(ValueError, match="too many rows"):
            _validate_row_count(df)  # uses global


# ── Integration tests with actual adapters ──────────────────────────────────

class TestCSVAdapterLimits:
    """CSV adapter respects file-size and row-count limits."""

    def test_csv_rejects_oversized_file(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, n_rows=10)

        # Set limit smaller than the file
        monkeypatch.setattr(ta, "MAX_UPLOAD_FILE_SIZE_BYTES", 10)

        adapter = BitcoinCSVAdapter()
        with pytest.raises(ValueError, match="File is too large"):
            adapter.load(str(csv_path))

    def test_csv_rejects_too_many_rows(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, n_rows=20)

        monkeypatch.setattr(ta, "MAX_UPLOAD_ROWS", 5)

        adapter = BitcoinCSVAdapter()
        with pytest.raises(ValueError, match="too many rows"):
            adapter.load(str(csv_path))

    def test_csv_loads_normal_file(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, n_rows=5)

        adapter = BitcoinCSVAdapter()
        df = adapter.load(str(csv_path))
        assert len(df) == 5


class TestJSONAdapterLimits:
    """JSON adapter respects file-size and row-count limits."""

    def test_json_rejects_oversized_file(self, tmp_path, monkeypatch):
        json_path = tmp_path / "data.json"
        _write_json(json_path, n_rows=10)

        monkeypatch.setattr(ta, "MAX_UPLOAD_FILE_SIZE_BYTES", 10)

        adapter = BitcoinJSONAdapter()
        with pytest.raises(ValueError, match="File is too large"):
            adapter.load(str(json_path))

    def test_json_rejects_too_many_rows(self, tmp_path, monkeypatch):
        json_path = tmp_path / "data.json"
        _write_json(json_path, n_rows=20)

        monkeypatch.setattr(ta, "MAX_UPLOAD_ROWS", 5)

        adapter = BitcoinJSONAdapter()
        with pytest.raises(ValueError, match="too many rows"):
            adapter.load(str(json_path))

    def test_json_loads_normal_file(self, tmp_path):
        json_path = tmp_path / "data.json"
        _write_json(json_path, n_rows=5)

        adapter = BitcoinJSONAdapter()
        df = adapter.load(str(json_path))
        assert len(df) == 5
