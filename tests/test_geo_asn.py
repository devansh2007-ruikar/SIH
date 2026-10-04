import os
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_asn import resolve_ip, enrich_dataframe, db_status


def test_resolve_ip_public():
    """Verify that a known public IP resolves to its correct country code."""
    res = resolve_ip("8.8.8.8")
    assert res["geo_country"] == "US"
    assert "Google" in res.get("asn_org", "")


def test_resolve_ip_private():
    """Verify that private and reserved IPs return reserved/private designations."""
    res_private = resolve_ip("192.168.1.1")
    assert res_private["geo_country"] in ("PRIVATE", "RESERVED", "reserved", "Unknown")
    assert res_private["asn"] == "AS0"

    res_loopback = resolve_ip("127.0.0.1")
    assert res_loopback["geo_country"] in ("LOCAL", "PRIVATE", "RESERVED", "Unknown")

    res_10 = resolve_ip("10.0.0.1")
    assert res_10["geo_country"] in ("PRIVATE", "RESERVED", "reserved", "Unknown")


def test_enrich_dataframe_fills_xx():
    """Verify that enrich_dataframe overwrites 'XX' sentinel values with resolved countries."""
    df = pd.DataFrame({
        "txid": ["tx_001", "tx_002"],
        "src_ip": ["8.8.8.8", "1.1.1.1"],
        "geo_country": ["XX", "XX"],
    })
    enriched = enrich_dataframe(df, ip_col="src_ip")
    assert enriched.loc[0, "geo_country"] == "US"
    assert enriched.loc[1, "geo_country"] == "AU"
    assert enriched.loc[0, "asn"] != "AS0"


def test_db_status_reports_database():
    """Verify that db_status accurately reflects the active database."""
    status = db_status()
    assert "country_db" in status
    assert "asn_db" in status
    assert "database_in_use" in status
    assert status["maxminddb_installed"] is True
