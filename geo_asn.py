"""
geo_asn.py — Offline Geo-ASN IP Resolver (MaxMind MMDB Implementation)
=======================================================================

Provides fully offline IP-to-Country and IP-to-ASN resolution using
local MaxMind GeoLite2 ``.mmdb`` database files, satisfying the SIH
2026 Problem Statement 26146 requirement for air-gapped operation.

Architecture
------------
1.  On import, the module probes for ``GeoLite2-Country.mmdb`` and
    ``GeoLite2-ASN.mmdb`` in the following search paths (first match wins):

    - ``<project_root>/data/``
    - ``<project_root>/``
    - ``/usr/share/GeoIP/``         (Arch Linux ``geoip2-database`` pkg)
    - ``/var/lib/GeoIP/``           (Debian/Ubuntu ``geoip2-database`` pkg)
    - ``/opt/geoip/``               (custom deployment path)

2.  If a database file is found, lookups use :mod:`maxminddb` (pure-Python
    MMDB reader — no C extension or network calls required).

3.  If NO database file is found, **every lookup returns the strict
    unknown sentinel**: ``geo_country='XX'``, ``asn='AS0'``,
    ``asn_org='Unknown'``.  The module **never fabricates** plausible
    geolocations for unmatched IPs.

4.  RFC-1918, loopback, link-local, and documentation-range IPs are
    handled at the Python level (before hitting the MMDB) and return
    descriptive labels (PRIVATE / LOCAL / DOC).

Usage
-----
::

    from geo_asn import resolve_ip, enrich_dataframe

    result = resolve_ip("1.1.1.1")
    # With DB:    {"geo_country": "AU", "asn": "AS13335", "asn_org": "Cloudflare, Inc."}
    # Without DB: {"geo_country": "XX", "asn": "AS0", "asn_org": "Unknown"}

    df = enrich_dataframe(df, ip_col="src_ip")

Forensic Disclaimer
-------------------
IP geolocation is a PROBABILISTIC technique subject to:
- VPN / proxy misattribution
- NAT / CGNAT masking
- Tor / I2P exit node aliasing
- GeoLite2 accuracy: ~75% at country level (MaxMind docs)

Results are INVESTIGATIVE PRIORITY SIGNALS — not absolute identity
attribution.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from typing import Dict, List, Optional, Union

import pandas as pd

# ============================================================================
# MODULE LOGGER
# ============================================================================

logger = logging.getLogger("mithya.geo_asn")

# ============================================================================
# TYPE ALIAS & STRICT UNKNOWN SENTINEL
# ============================================================================

GeoResult = Dict[str, str]  # {"geo_country": str, "asn": str, "asn_org": str}

_UNKNOWN: GeoResult = {"geo_country": "XX", "asn": "AS0", "asn_org": "Unknown"}


# ============================================================================
# MAXMIND MMDB DATABASE DISCOVERY & LOADING
# ============================================================================

_SEARCH_PATHS: List[str] = []

# Build search paths relative to this file's location
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SEARCH_PATHS.extend([
    os.path.join(_THIS_DIR, "data", "geo"),
    os.path.join(_THIS_DIR, "data"),
    _THIS_DIR,
    "/usr/share/GeoIP",
    "/var/lib/GeoIP",
    "/opt/geoip",
])


def _find_mmdb(filenames: Union[str, List[str]]) -> Optional[str]:
    """Search for a .mmdb file across known paths in priority order. Returns absolute path or None."""
    if isinstance(filenames, str):
        filenames = [filenames]
    for fn in filenames:
        for search_dir in _SEARCH_PATHS:
            candidate = os.path.join(search_dir, fn)
            if os.path.isfile(candidate):
                return candidate
    return None


# Attempt to import maxminddb (pure-Python reader, no network calls)
_country_reader = None
_asn_reader = None

try:
    import maxminddb

    _country_path = _find_mmdb(["GeoLite2-Country.mmdb", "dbip-country-lite.mmdb"])
    _asn_path = _find_mmdb(["GeoLite2-ASN.mmdb", "dbip-asn-lite.mmdb"])

    if _country_path:
        _country_reader = maxminddb.open_database(_country_path)
        logger.info("[geo_asn] Loaded Country DB: %s", _country_path)
    else:
        logger.warning(
            "[geo_asn] Neither GeoLite2-Country.mmdb nor dbip-country-lite.mmdb found in search paths. "
            "Country lookups will return 'XX'."
        )

    if _asn_path:
        _asn_reader = maxminddb.open_database(_asn_path)
        logger.info("[geo_asn] Loaded ASN DB: %s", _asn_path)
    else:
        logger.warning(
            "[geo_asn] Neither GeoLite2-ASN.mmdb nor dbip-asn-lite.mmdb found in search paths. "
            "ASN lookups will return 'AS0'."
        )

except ImportError:
    logger.warning(
        "[geo_asn] maxminddb is not installed. All lookups will return "
        "unknown. Install with: pip install maxminddb"
    )


# ============================================================================
# RFC-1918 / RESERVED RANGE HANDLING (pure Python, no MMDB needed)
# ============================================================================

_RESERVED_RANGES = [
    # (network, geo_country, asn, asn_org)
    (ipaddress.ip_network("10.0.0.0/8"),       "PRIVATE", "AS0", "RFC-1918 Private Network"),
    (ipaddress.ip_network("172.16.0.0/12"),     "PRIVATE", "AS0", "RFC-1918 Private Network"),
    (ipaddress.ip_network("192.168.0.0/16"),    "PRIVATE", "AS0", "RFC-1918 Private Network"),
    (ipaddress.ip_network("127.0.0.0/8"),       "LOCAL",   "AS0", "Loopback"),
    (ipaddress.ip_network("169.254.0.0/16"),    "LOCAL",   "AS0", "Link-Local / APIPA"),
    (ipaddress.ip_network("192.0.2.0/24"),      "DOC",     "AS0", "RFC-5737 TEST-NET-1"),
    (ipaddress.ip_network("198.51.100.0/24"),   "DOC",     "AS0", "RFC-5737 TEST-NET-2"),
    (ipaddress.ip_network("203.0.113.0/24"),    "DOC",     "AS0", "RFC-5737 TEST-NET-3"),
    (ipaddress.ip_network("100.64.0.0/10"),     "CGNAT",   "AS0", "RFC-6598 CGNAT"),
    (ipaddress.ip_network("240.0.0.0/4"),       "RESERVED","AS0", "RFC-1112 Reserved"),
]


def _check_reserved(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> Optional[GeoResult]:
    """Return a GeoResult if the address falls in a reserved range, else None."""
    for net, country, asn, org in _RESERVED_RANGES:
        if addr in net:
            return {"geo_country": country, "asn": asn, "asn_org": org}
    return None


# ============================================================================
# MMDB LOOKUP (country + ASN)
# ============================================================================

def _lookup_mmdb(ip_str: str) -> GeoResult:
    """
    Query the MaxMind MMDB databases for country and ASN.

    If no database is loaded or the IP is not found, returns _UNKNOWN.
    NEVER fabricates plausible data.
    """
    country = "XX"
    asn = "AS0"
    asn_org = "Unknown"

    # Country lookup
    if _country_reader is not None:
        try:
            record = _country_reader.get(ip_str)
            if record and "country" in record:
                country = record["country"].get("iso_code", "XX") or "XX"
            elif record and "registered_country" in record:
                country = record["registered_country"].get("iso_code", "XX") or "XX"
        except Exception:
            pass  # IP not in database → keep XX

    # ASN lookup
    if _asn_reader is not None:
        try:
            record = _asn_reader.get(ip_str)
            if record:
                asn_num = record.get("autonomous_system_number")
                if asn_num is not None:
                    asn = f"AS{asn_num}"
                asn_org = record.get("autonomous_system_organization", "Unknown") or "Unknown"
        except Exception:
            pass  # IP not in database → keep AS0/Unknown

    return {"geo_country": country, "asn": asn, "asn_org": asn_org}


# ============================================================================
# PUBLIC API
# ============================================================================

def resolve_ip(ip: str) -> GeoResult:
    """
    Resolve a single IP address to its country and ASN entirely offline.

    Parameters
    ----------
    ip : str
        IPv4 or IPv6 address string.

    Returns
    -------
    dict
        ``{"geo_country": str, "asn": str, "asn_org": str}``

        - If the IP matches a reserved range: returns PRIVATE/LOCAL/DOC labels.
        - If the MMDB database is loaded and the IP is found: returns real data.
        - Otherwise: returns ``{"geo_country": "XX", "asn": "AS0", "asn_org": "Unknown"}``.

    Notes
    -----
    This function performs ZERO network I/O. It reads only from local
    ``.mmdb`` files (or returns unknown if no database is present).

    FORENSIC DISCLAIMER: IP geolocation is subject to VPN/NAT/Tor limits.
    The result is an INVESTIGATIVE SIGNAL — not absolute identity attribution.
    """
    ip_str = str(ip).strip()

    # Fast path: invalid / empty
    if not ip_str or ip_str in ("nan", "None", "N/A", ""):
        return dict(_UNKNOWN)

    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return dict(_UNKNOWN)

    # Check reserved ranges first (no MMDB needed)
    reserved = _check_reserved(addr)
    if reserved is not None:
        return reserved

    # MMDB lookup (returns _UNKNOWN if no DB or IP not found)
    return _lookup_mmdb(ip_str)


def resolve_ips_batch(
    ips: Union[pd.Series, List[str]],
) -> pd.Series:
    """
    Vectorised resolver for a pandas Series or list of IP strings.

    Returns a pandas Series of dicts (one per IP).

    Parameters
    ----------
    ips : pd.Series or list of str

    Returns
    -------
    pd.Series
        Series of GeoResult dicts, same length as input.
    """
    if isinstance(ips, pd.Series):
        return ips.apply(resolve_ip)
    return pd.Series([resolve_ip(ip) for ip in ips])


def enrich_dataframe(
    df: pd.DataFrame,
    ip_col: str = "src_ip",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add ``geo_country``, ``asn``, and ``asn_org`` columns using offline resolution.

    If ``geo_country`` already exists, it is overwritten with the
    offline-resolved value only when the existing value is null/unknown/XX.

    Parameters
    ----------
    df : pd.DataFrame
    ip_col : str
        Column containing source IP addresses.
    inplace : bool
        If True, mutates df in place. If False (default), returns a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``geo_country``, ``asn``, and ``asn_org`` columns.
    """
    if not inplace:
        df = df.copy()

    if ip_col not in df.columns:
        df["geo_country"] = "XX"
        df["asn"] = "AS0"
        df["asn_org"] = "Unknown"
        return df

    # Speed optimization: resolve each unique IP address only once
    unique_ips = df[ip_col].dropna().drop_duplicates()
    resolved_dict = {ip: resolve_ip(ip) for ip in unique_ips}

    geo_map = {ip: res["geo_country"] for ip, res in resolved_dict.items()}
    asn_map = {ip: res["asn"] for ip, res in resolved_dict.items()}
    org_map = {ip: res["asn_org"] for ip, res in resolved_dict.items()}

    geo_col = df[ip_col].map(geo_map).fillna("XX")
    asn_col = df[ip_col].map(asn_map).fillna("AS0")
    org_col = df[ip_col].map(org_map).fillna("Unknown")

    # Overwrite only when existing value is null / unknown / missing / XX
    if "geo_country" in df.columns:
        mask = df["geo_country"].isnull() | df["geo_country"].isin(["", "nan", "XX"])
        df.loc[mask, "geo_country"] = geo_col[mask]
    else:
        df["geo_country"] = geo_col

    df["asn"] = asn_col
    df["asn_org"] = org_col

    return df


def db_status() -> Dict[str, Optional[str]]:
    """Return the paths and status of loaded databases, reporting which DB is in use."""
    country_in_use = os.path.basename(_country_path) if (_country_reader and _country_path) else None
    asn_in_use = os.path.basename(_asn_path) if (_asn_reader and _asn_path) else None

    # Determine user-friendly database name / type
    if country_in_use and "dbip" in country_in_use.lower():
        db_label = "DB-IP Lite (2026-10)"
    elif country_in_use and "geolite2" in country_in_use.lower():
        db_label = "MaxMind GeoLite2"
    elif country_in_use:
        db_label = country_in_use
    else:
        db_label = None

    return {
        "country_db": str(_country_path) if (_country_reader and _country_path) else None,
        "country_db_name": country_in_use,
        "asn_db": str(_asn_path) if (_asn_reader and _asn_path) else None,
        "asn_db_name": asn_in_use,
        "database_in_use": db_label or "none",
        "db_name": db_label,
        "maxminddb_installed": _country_reader is not None or _asn_reader is not None,
    }


# ============================================================================
# SELF-TEST
# ============================================================================

def _self_test() -> None:
    """Quick smoke-test for the resolver (no network I/O)."""
    print("\n[*] geo_asn.py self-test (offline mode)")
    print("-" * 56)

    status = db_status()
    print(f"  Country DB : {status['country_db'] or 'NOT LOADED'}")
    print(f"  ASN DB     : {status['asn_db'] or 'NOT LOADED'}")
    print()

    tests = [
        ("10.0.0.1",      "PRIVATE", "AS0"),
        ("192.168.1.1",   "PRIVATE", "AS0"),
        ("127.0.0.1",     "LOCAL",   "AS0"),
        ("169.254.1.1",   "LOCAL",   "AS0"),
        ("192.0.2.1",     "DOC",     "AS0"),
    ]

    all_pass = True
    for ip, exp_country, exp_asn in tests:
        r = resolve_ip(ip)
        ok = (r["geo_country"] == exp_country and r["asn"] == exp_asn)
        status_str = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(
            f"  [{status_str}] {ip:<20}  "
            f"country={r['geo_country']:<8} asn={r['asn']:<12} "
            f"org={r['asn_org']}"
        )

    # Test public IPs (result depends on DB availability)
    public_ips = ["1.1.1.1", "8.8.8.8", "invalid_ip", "nan"]
    for ip in public_ips:
        r = resolve_ip(ip)
        print(
            f"  [INFO] {ip:<20}  "
            f"country={r['geo_country']:<8} asn={r['asn']:<12} "
            f"org={r['asn_org']}"
        )

    # Batch test
    batch = resolve_ips_batch(public_ips)
    print(f"\n  Batch({len(public_ips)} IPs): {len(batch)} results OK")
    print("-" * 56)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}\n")
    print("  FORENSIC DISCLAIMER:")
    print("  IP geolocation is subject to VPN/NAT/Tor limits.")
    print("  Results are INVESTIGATIVE SIGNALS — not absolute identity attribution.\n")


if __name__ == "__main__":
    _self_test()
