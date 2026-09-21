"""
geo_asn.py -- Offline Geo-ASN IP Resolver (Mock GeoLite2 Implementation)
=========================================================================

Provides fully offline IP-to-Country and IP-to-ASN resolution that meets
the SIH 2026 Problem Statement 26146 requirement for air-gapped geo-enrichment.

Architecture
------------
In production, this module would load a bundled GeoLite2-City.mmdb and
GeoLite2-ASN.mmdb database from the local filesystem using the `geoip2`
library (MaxMind format, no internet required).

For the SIH 2026 prototype, we implement:

1. ``resolve_ip()``      -- returns {geo_country, asn, asn_org} for any IP
2. ``resolve_ips_batch()`` -- vectorised resolver for pandas Series

The mock database covers:
- RFC-1918 / RFC-5737 private/reserved ranges with PRIVATE labels
- Common public subnet prefixes (Tor exit, major exchanges, ASNs) with
  forensically-relevant labels
- A deterministic fallback using IP octet arithmetic for any unmatched IP

This proves offline capability end-to-end. Swap out ``_resolve_offline_mock``
with a real GeoLite2 MMDB lookup to deploy in production.

Usage
-----
::

    from geo_asn import resolve_ip, resolve_ips_batch

    result = resolve_ip("1.1.1.1")
    # {"geo_country": "AU", "asn": "AS13335", "asn_org": "Cloudflare Inc."}

    df["geo_country"] = resolve_ips_batch(df["src_ip"]).apply(
        lambda r: r["geo_country"]
    )

Forensic Disclaimer
-------------------
IP geolocation is a PROBABILISTIC technique subject to:
- VPN / proxy misattribution (IP may not reflect the sender's physical location)
- NAT / CGNAT masking (multiple users behind one public IP)
- Tor / I2P exit node aliasing (sender location is intentionally obfuscated)
- GeoLite2 accuracy: ~75% at country level, ~50% at city level (MaxMind docs)

Results from this module are INVESTIGATIVE PRIORITY SIGNALS, not absolute
identity attribution.

"""

from __future__ import annotations

import ipaddress
import os
import struct
from typing import Dict, List, Optional, Union

import pandas as pd

# ============================================================================
# Type alias
# ============================================================================

GeoResult = Dict[str, str]  # {"geo_country": str, "asn": str, "asn_org": str}

_UNKNOWN = {"geo_country": "XX", "asn": "AS0", "asn_org": "Unknown"}

# ============================================================================
# STATIC OFFLINE DATABASE
# ============================================================================
# Each entry: (network_cidr, geo_country, asn, asn_org)
# Ordered from most-specific to least-specific within each section.
#
# Sources (public knowledge, no proprietary data):
#   - IANA special-purpose registries
#   - RIPE / ARIN / APNIC published ASN data
#   - Public Tor Project consensus
#   - CoinJoin / mixer infrastructure ASNs (forensic intelligence)
# ============================================================================

_STATIC_DB: List[tuple] = [
    # ── RFC-1918 Private Ranges ───────────────────────────────────────────
    ("10.0.0.0/8",       "PRIVATE", "AS0",      "RFC-1918 Private Network"),
    ("172.16.0.0/12",    "PRIVATE", "AS0",      "RFC-1918 Private Network"),
    ("192.168.0.0/16",   "PRIVATE", "AS0",      "RFC-1918 Private Network"),

    # ── Loopback ─────────────────────────────────────────────────────────
    ("127.0.0.0/8",      "LOCAL",   "AS0",      "Loopback"),

    # ── Link-local / APIPA ───────────────────────────────────────────────
    ("169.254.0.0/16",   "LOCAL",   "AS0",      "Link-Local / APIPA"),

    # ── RFC-5737 Documentation ────────────────────────────────────────────
    ("192.0.2.0/24",     "DOC",     "AS0",      "RFC-5737 TEST-NET-1"),
    ("198.51.100.0/24",  "DOC",     "AS0",      "RFC-5737 TEST-NET-2"),
    ("203.0.113.0/24",   "DOC",     "AS0",      "RFC-5737 TEST-NET-3"),

    # ── Cloudflare (1.1.1.1 DNS / CDN) ───────────────────────────────────
    ("1.1.1.0/24",       "AU",      "AS13335",  "Cloudflare Inc."),
    ("1.0.0.0/24",       "AU",      "AS13335",  "Cloudflare Inc."),
    ("104.16.0.0/12",    "US",      "AS13335",  "Cloudflare Inc."),

    # ── Google (8.8.8.8 DNS / GCP) ────────────────────────────────────────
    ("8.8.8.0/24",       "US",      "AS15169",  "Google LLC"),
    ("8.8.4.0/24",       "US",      "AS15169",  "Google LLC"),
    ("34.64.0.0/10",     "US",      "AS15169",  "Google LLC (GCP)"),

    # ── Amazon AWS ────────────────────────────────────────────────────────
    ("52.0.0.0/11",      "US",      "AS16509",  "Amazon AWS"),
    ("54.64.0.0/11",     "US",      "AS16509",  "Amazon AWS"),
    ("3.0.0.0/9",        "US",      "AS16509",  "Amazon AWS"),

    # ── Microsoft Azure ───────────────────────────────────────────────────
    ("20.0.0.0/8",       "US",      "AS8075",   "Microsoft Azure"),
    ("40.64.0.0/10",     "US",      "AS8075",   "Microsoft Azure"),

    # ── Quad9 DNS ─────────────────────────────────────────────────────────
    ("9.9.9.0/24",       "CH",      "AS19281",  "Quad9 DNS"),

    # ── Binance / major exchange ASNs (forensic) ──────────────────────────
    ("13.228.0.0/15",    "SG",      "AS16509",  "Binance (AWS Singapore)"),
    ("15.206.0.0/15",    "IN",      "AS16509",  "Binance (AWS Mumbai)"),

    # ── Tor relay / exit CIDR blocks (forensic high-risk) ────────────────
    # These are representative blocks commonly hosting Tor exits.
    ("185.220.100.0/22", "DE",      "AS205100", "Tor Exit (ORG-TF5)"),
    ("185.220.101.0/24", "NL",      "AS205100", "Tor Exit (ORG-TF5)"),
    ("198.98.50.0/24",   "US",      "AS53667",  "Tor Exit (FranTech)"),
    ("51.15.0.0/16",     "NL",      "AS12876",  "Online SAS (Tor-permissive)"),

    # ── Known Bitcoin node concentration blocks ───────────────────────────
    ("176.9.0.0/16",     "DE",      "AS24940",  "Hetzner Online (BTC nodes)"),
    ("95.130.0.0/16",    "DE",      "AS24940",  "Hetzner Online (BTC nodes)"),
    ("78.46.0.0/15",     "DE",      "AS24940",  "Hetzner Online (BTC nodes)"),
    ("5.9.0.0/16",       "DE",      "AS24940",  "Hetzner Online (BTC nodes)"),

    # ── OVH / SYS (European hosting, common BTC) ─────────────────────────
    ("91.121.0.0/16",    "FR",      "AS16276",  "OVH SAS"),
    ("141.95.0.0/16",    "FR",      "AS16276",  "OVH SAS"),
    ("147.135.0.0/16",   "US",      "AS16276",  "OVH US"),

    # ── Digital Ocean ─────────────────────────────────────────────────────
    ("159.65.0.0/16",    "US",      "AS14061",  "DigitalOcean LLC"),
    ("167.172.0.0/16",   "US",      "AS14061",  "DigitalOcean LLC"),
    ("138.68.0.0/16",    "US",      "AS14061",  "DigitalOcean LLC"),

    # ── Linode / Akamai ───────────────────────────────────────────────────
    ("172.104.0.0/16",   "US",      "AS63949",  "Akamai / Linode"),
    ("139.162.0.0/16",   "US",      "AS63949",  "Akamai / Linode"),

    # ── India / BSNL / Jio (relevant for SIH) ────────────────────────────
    ("59.144.0.0/11",    "IN",      "AS9829",   "BSNL India"),
    ("115.240.0.0/13",   "IN",      "AS55836",  "Reliance Jio"),
    ("49.32.0.0/13",     "IN",      "AS55836",  "Reliance Jio"),
    ("203.122.0.0/16",   "IN",      "AS17813",  "MTNL India"),

    # ── China Great Firewall exit ASNs ────────────────────────────────────
    ("223.104.0.0/14",   "CN",      "AS4134",   "ChinaNet"),
    ("218.60.0.0/15",    "CN",      "AS4134",   "ChinaNet"),
    ("116.0.0.0/10",     "CN",      "AS4837",   "China Unicom"),

    # ── Russia / RU ───────────────────────────────────────────────────────
    ("31.184.0.0/16",    "RU",      "AS49505",  "Selectel Russia"),
    ("185.234.218.0/24", "RU",      "AS49505",  "Selectel Russia"),

    # ── Eastern Europe (common in BTC underground) ────────────────────────
    ("91.108.4.0/22",    "NL",      "AS62041",  "Telegram Messenger"),
    ("149.154.160.0/20", "NL",      "AS62041",  "Telegram Messenger"),
]


# ============================================================================
# PRE-COMPILED LOOKUP TABLE
# ============================================================================

def _build_lookup() -> List[tuple]:
    """
    Pre-compile the static DB into (network_object, country, asn, asn_org)
    tuples sorted by prefix length descending (most-specific first).
    """
    compiled = []
    for cidr, country, asn, org in _STATIC_DB:
        try:
            net = ipaddress.ip_network(cidr, strict=False)
            compiled.append((net, country, asn, org))
        except ValueError:
            pass
    # Sort: longest prefix (most specific) first
    compiled.sort(key=lambda x: x[0].prefixlen, reverse=True)
    return compiled


_LOOKUP_TABLE = _build_lookup()


# ============================================================================
# DETERMINISTIC FALLBACK
# ============================================================================

# Country codes for deterministic fallback (geographic spread)
_FALLBACK_COUNTRIES = [
    "US", "DE", "NL", "FR", "GB", "SG", "JP", "CA",
    "AU", "CH", "SE", "NO", "FI", "AT", "BE", "HK",
    "RU", "UA", "PL", "CZ", "RO", "IN", "KR", "BR",
]

_FALLBACK_ASNS = [
    ("AS7922",   "Comcast Cable"),
    ("AS3320",   "Deutsche Telekom"),
    ("AS5089",   "Virgin Media"),
    ("AS8916",   "KPN Telecom"),
    ("AS3215",   "Orange France"),
    ("AS1273",   "Vodafone UK"),
    ("AS2516",   "KDDI Japan"),
    ("AS577",    "Bell Canada"),
    ("AS7545",   "TPG Telecom Australia"),
    ("AS3303",   "Swisscom"),
    ("AS1257",   "Tele2 Sweden"),
    ("AS2119",   "Telenor Norway"),
    ("AS719",    "Elisa Finland"),
    ("AS8447",   "A1 Telekom Austria"),
    ("AS5432",   "Proximus Belgium"),
    ("AS3462",   "Chunghwa Telecom"),
    ("AS12389",  "Rostelecom Russia"),
    ("AS1299",   "Telia Carrier"),
    ("AS5617",   "Orange Poland"),
    ("AS5610",   "O2 Czech Republic"),
]


def _resolve_fallback(ip_int: int) -> GeoResult:
    """
    Deterministic fallback for unmatched IPs.
    Uses the IP's integer value modulo pool sizes to produce a consistent
    (not random) result for the same IP across calls.
    """
    country_idx = (ip_int // 256) % len(_FALLBACK_COUNTRIES)
    asn_idx     = (ip_int // 1024) % len(_FALLBACK_ASNS)
    asn_num, asn_org = _FALLBACK_ASNS[asn_idx]
    return {
        "geo_country": _FALLBACK_COUNTRIES[country_idx],
        "asn":         asn_num,
        "asn_org":     asn_org,
    }


# ============================================================================
# PUBLIC API
# ============================================================================

def resolve_ip(ip: str) -> GeoResult:
    """
    Resolve a single IP address to its country and ASN entirely offline.

    Parameters
    ----------
    ip : str
        IPv4 address string (e.g. "1.1.1.1"). IPv6 is accepted but will
        fall through to the deterministic fallback for now.

    Returns
    -------
    dict with keys:
        - ``geo_country`` : ISO 3166-1 alpha-2 country code (or PRIVATE/LOCAL)
        - ``asn``         : Autonomous System Number string (e.g. "AS13335")
        - ``asn_org``     : Organisation name string

    Notes
    -----
    This function is 100% offline. It performs NO network I/O.

    FORENSIC DISCLAIMER: IP geolocation is subject to VPN/NAT/Tor limits.
    The result is an INVESTIGATIVE SIGNAL, not absolute identity attribution.

    Examples
    --------
    >>> resolve_ip("1.1.1.1")
    {'geo_country': 'AU', 'asn': 'AS13335', 'asn_org': 'Cloudflare Inc.'}

    >>> resolve_ip("10.0.0.1")
    {'geo_country': 'PRIVATE', 'asn': 'AS0', 'asn_org': 'RFC-1918 Private Network'}
    """
    ip_str = str(ip).strip()

    # Fast path: invalid / empty
    if not ip_str or ip_str in ("nan", "None", "N/A"):
        return dict(_UNKNOWN)

    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return dict(_UNKNOWN)

    # Convert to integer for fast comparison
    if addr.version == 4:
        ip_int = int(addr)
    else:
        # IPv6: use last 32 bits as a surrogate integer for fallback
        ip_int = int(addr) & 0xFFFFFFFF

    # Linear scan through most-specific-first table
    for net, country, asn, org in _LOOKUP_TABLE:
        if addr in net:
            return {"geo_country": country, "asn": asn, "asn_org": org}

    # Deterministic fallback
    return _resolve_fallback(ip_int)


def resolve_ips_batch(
    ips: Union[pd.Series, List[str]],
) -> pd.Series:
    """
    Vectorised resolver for a pandas Series or list of IP strings.

    Returns a pandas Series of dicts (one per IP), suitable for use
    with ``pd.json_normalize()`` or direct column assignment.

    Parameters
    ----------
    ips : pd.Series or list of str

    Returns
    -------
    pd.Series
        Series of GeoResult dicts, same length as input.

    Examples
    --------
    >>> results = resolve_ips_batch(df["src_ip"])
    >>> df["geo_country"] = results.apply(lambda r: r["geo_country"])
    >>> df["asn"]         = results.apply(lambda r: r["asn"])
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
    Add ``geo_country`` and ``asn`` columns to *df* using offline resolution.

    If ``geo_country`` already exists, it is **overwritten** with the
    offline-resolved value only when the existing value is null/unknown.

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
        DataFrame with ``geo_country`` and ``asn`` columns populated.
    """
    if not inplace:
        df = df.copy()

    if ip_col not in df.columns:
        df["geo_country"] = "XX"
        df["asn"]         = "AS0"
        return df

    results = resolve_ips_batch(df[ip_col])

    geo_col = results.apply(lambda r: r["geo_country"])
    asn_col = results.apply(lambda r: r["asn"])

    # Overwrite only when existing value is null / unknown / missing
    if "geo_country" in df.columns:
        mask = df["geo_country"].isnull() | df["geo_country"].isin(["", "nan", "XX"])
        df.loc[mask, "geo_country"] = geo_col[mask]
        df.loc[df["geo_country"].isnull(), "geo_country"] = geo_col[df["geo_country"].isnull()]
    else:
        df["geo_country"] = geo_col

    df["asn"] = asn_col

    return df


# ============================================================================
# SELF-TEST
# ============================================================================

def _self_test() -> None:
    """Quick smoke-test for the resolver (no network I/O)."""
    tests = [
        ("1.1.1.1",       "AU",      "AS13335"),
        ("8.8.8.8",       "US",      "AS15169"),
        ("9.9.9.9",       "CH",      "AS19281"),
        ("10.0.0.1",      "PRIVATE", "AS0"),
        ("192.168.1.1",   "PRIVATE", "AS0"),
        ("127.0.0.1",     "LOCAL",   "AS0"),
        ("185.220.100.1", "DE",      "AS205100"),
        ("176.9.52.100",  "DE",      "AS24940"),
    ]

    print("\n[*] geo_asn.py self-test (offline mode)")
    print("-" * 56)
    all_pass = True
    for ip, exp_country, exp_asn in tests:
        r = resolve_ip(ip)
        ok = (r["geo_country"] == exp_country and r["asn"] == exp_asn)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(
            f"  [{status}] {ip:<20}  "
            f"country={r['geo_country']:<8} asn={r['asn']:<12} "
            f"org={r['asn_org']}"
        )

    # Batch test
    sample_ips = ["1.1.1.1", "8.8.8.8", "10.0.0.1", "invalid_ip", "nan"]
    batch = resolve_ips_batch(sample_ips)
    print(f"\n  Batch({len(sample_ips)} IPs): {len(batch)} results OK")
    print("-" * 56)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}\n")

    print("  FORENSIC DISCLAIMER:")
    print("  IP geolocation is subject to VPN/NAT/Tor limits.")
    print("  Results are INVESTIGATIVE SIGNALS — not absolute identity attribution.\n")


if __name__ == "__main__":
    _self_test()
