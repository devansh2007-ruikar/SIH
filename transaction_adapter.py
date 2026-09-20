"""
transaction_adapter.py — Cross-Chain Transaction Adapter Layer
==============================================================

Provides an abstract base class for transaction ingestion and three
concrete Bitcoin adapters supporting CSV, JSON, and XML formats.
Future chains (Ethereum, Monero, …) can be supported by adding new
adapter subclasses — no changes to the core ML pipeline are required.

Architecture
------------
::

    ┌──────────────────────────────────────┐
    │   BaseTransactionAdapter (ABC)       │
    │  ────────────────────────────────── │
    │  + load(filepath) -> pd.DataFrame   │
    │  + load_data()                      │
    │  + normalize_addresses()            │
    │  + extract_features()               │
    │  + run_pipeline()  ← template       │
    └──────────────┬──────────────────────┘
                   │  inherits
    ┌──────────────▼──────────────────────┐
    │     BitcoinCSVAdapter               │
    │     BitcoinJSONAdapter              │
    │     BitcoinXMLAdapter               │
    │  ────────────────────────────────── │
    │  Delegates to ml_engine.*()         │
    └─────────────────────────────────────┘

    ┌─────────────────────────────────────┐
    │  get_adapter(file_path_or_buffer)   │
    │  ─────────────────────────────────  │
    │  Factory: inspects extension /      │
    │  mime-type → returns adapter        │
    └─────────────────────────────────────┘

Supported Formats
-----------------
- **CSV** (.csv):  Standard comma-delimited with pipe-delimited multi-value
  columns for addresses and amounts.
- **JSON** (.json):  Array of transaction objects *or* a top-level dictionary
  whose values are transaction objects (keyed by txid).
- **XML** (.xml):  Hierarchical ``<transactions><transaction>…</transaction>
  …</transactions>`` structure parsed via ``xml.etree.ElementTree``.

Canonical DataFrame Schema
--------------------------
All adapters normalise into the following columns:

    ========== ========================================= ==================
    Column     Type                                      Notes
    ========== ========================================= ==================
    txid       str                                       Transaction hash
    input_addresses   str (pipe-delimited)                Sender addresses
    output_addresses  str (pipe-delimited)                Receiver addresses
    input_amounts     str (pipe-delimited floats)         Input values
    output_amounts    str (pipe-delimited floats)         Output values
    fee        float                                     Default 0.0001
    src_port   int                                       Source port
    ========== ========================================= ==================

Usage
-----
>>> from transaction_adapter import get_adapter
>>> adapter = get_adapter("bitcoin_traffic.csv")
>>> df = adapter.load("bitcoin_traffic.csv")

>>> # Full ML pipeline (backward-compatible)
>>> adapter = BitcoinCSVAdapter()
>>> enriched_df, model, features = adapter.run_pipeline("bitcoin_traffic.csv")

The Streamlit frontend instantiates the adapter and calls
``adapter.run_pipeline_from_df(df, contamination)`` — swapping in a
future ``EthereumAdapter`` requires only changing the instantiation.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# pyrefly: ignore [missing-import]
from sklearn.ensemble import IsolationForest

# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL LOGGING
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("mithya.transaction_adapter")

# Default fee when the source data does not provide one (in BTC)
_DEFAULT_FEE: float = 0.0001

# Canonical column order for normalised DataFrames
CANONICAL_COLUMNS: List[str] = [
    "txid",
    "input_addresses",
    "output_addresses",
    "input_amounts",
    "output_amounts",
    "fee",
    "src_port",
]


# ═══════════════════════════════════════════════════════════════════════════
# HELPER — Schema normalisation (shared by all adapters)
# ═══════════════════════════════════════════════════════════════════════════

def _coerce_to_pipe_str(value: Any) -> str:
    """
    Convert a value to a pipe-delimited string.

    Handles:
    - ``None`` / ``NaN`` → ``""``
    - Python list → ``"a|b|c"``
    - Already pipe-delimited str → stripped and returned
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(v).strip() for v in value if str(v).strip())
    s = str(value).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    # Normalise each element in a pipe-delimited string
    return "|".join(a.strip() for a in s.split("|") if a.strip())


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Parse a value to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_dataframe(df: pd.DataFrame, adapter_name: str) -> pd.DataFrame:
    """
    Enforce the canonical schema on *df* regardless of source format.

    Guarantees:
    - All :data:`CANONICAL_COLUMNS` exist.
    - ``fee`` defaults to :data:`_DEFAULT_FEE` when missing or NaN.
    - ``src_port`` defaults to ``0`` when missing.
    - Pipe-delimited string columns are stripped and de-duplicated.
    - Informative ``WARNING`` logs for any column that required defaults.

    Returns a **new** DataFrame (does not mutate the original).
    """
    df = df.copy()

    # ── txid ─────────────────────────────────────────────────────
    if "txid" not in df.columns:
        raise ValueError(
            f"[{adapter_name}] Loaded data has no 'txid' column — "
            f"cannot identify transactions.  Available columns: "
            f"{list(df.columns)}"
        )
    df["txid"] = df["txid"].astype(str)

    # ── pipe-delimited columns ───────────────────────────────────
    for col in ("input_addresses", "output_addresses",
                "input_amounts", "output_amounts"):
        if col not in df.columns:
            logger.warning(
                "[%s] Column '%s' missing — filling with empty strings.",
                adapter_name, col,
            )
            df[col] = ""
        df[col] = df[col].apply(_coerce_to_pipe_str)

    # ── fee ──────────────────────────────────────────────────────
    if "fee" not in df.columns:
        logger.warning(
            "[%s] Column 'fee' missing — defaulting all rows to %.4f BTC.",
            adapter_name, _DEFAULT_FEE,
        )
        df["fee"] = _DEFAULT_FEE
    else:
        df["fee"] = df["fee"].apply(
            lambda v: _safe_float(v, default=_DEFAULT_FEE)
        )
        na_mask = df["fee"].isna()
        if na_mask.any():
            logger.warning(
                "[%s] %d rows had NaN/null 'fee' — defaulting to %.4f.",
                adapter_name, int(na_mask.sum()), _DEFAULT_FEE,
            )
            df.loc[na_mask, "fee"] = _DEFAULT_FEE

    # ── src_port ─────────────────────────────────────────────────
    if "src_port" not in df.columns:
        logger.warning(
            "[%s] Column 'src_port' missing — defaulting all rows to 0.",
            adapter_name,
        )
        df["src_port"] = 0
    else:
        df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").fillna(0).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════

class BaseTransactionAdapter(ABC):
    """
    Abstract contract for chain-specific transaction ingestion.

    Every concrete adapter **must** implement:

    * :meth:`load`  — read a file and return a canonical DataFrame
    * :meth:`load_data`  — read raw data from a source (file or DataFrame)
    * :meth:`normalize_addresses`  — standardise address columns
    * :meth:`extract_features`  — produce the ML feature matrix

    The :meth:`run_pipeline` and :meth:`run_pipeline_from_df` template
    methods orchestrate the full detection pipeline and should **not**
    be overridden unless the chain requires a fundamentally different
    scoring strategy.
    """

    # ── Required chain identifier (override in subclass) ─────────
    CHAIN_NAME: str = "abstract"

    # ── Columns that the normalised DataFrame MUST contain ───────
    REQUIRED_COLUMNS: List[str] = [
        "timestamp", "txid",
        "input_addresses", "output_addresses",
        "input_amounts", "output_amounts",
    ]

    # ──────────────────────────────────────────────────────────────
    # Abstract interface
    # ──────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self, filepath: str) -> pd.DataFrame:
        """
        Read a transaction file and return a canonical DataFrame.

        Parameters
        ----------
        filepath : str
            Path to a data file (CSV, JSON, or XML).

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with at least the columns defined in
            :data:`CANONICAL_COLUMNS`.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If the file content is malformed or missing required fields.
        """
        ...

    @abstractmethod
    def load_data(self, source: str | pd.DataFrame) -> pd.DataFrame:
        """
        Ingest raw transaction data from *source*.

        Parameters
        ----------
        source : str | pd.DataFrame
            A file path (CSV, JSON, Parquet …) **or** an already-loaded
            DataFrame.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame with chain-native column names.
        """
        ...

    @abstractmethod
    def normalize_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure ``input_addresses`` and ``output_addresses`` exist as
        pipe-delimited strings, ready for the ML pipeline.

        This is the chain-specific mapping layer: for Bitcoin the
        columns already use ``|``; for Ethereum you would extract
        ``from`` / ``to``; for Monero you might decode stealth
        addresses, etc.

        Returns the DataFrame **in-place** (mutated) for efficiency.
        """
        ...

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Produce a numeric feature matrix ``X`` for IsolationForest,
        plus the (possibly enriched) transaction DataFrame.

        Returns
        -------
        features : pd.DataFrame
            Numeric-only feature matrix (rows = transactions).
        df : pd.DataFrame
            The input DataFrame, potentially with extra computed
            columns (``entity_id``, ``total_amount_btc``, …).
        """
        ...

    # ──────────────────────────────────────────────────────────────
    # Template methods (concrete — do NOT override normally)
    # ──────────────────────────────────────────────────────────────

    def run_pipeline(
        self,
        source: str | pd.DataFrame,
        contamination: float = 0.05,
        output_csv: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, IsolationForest, pd.DataFrame]:
        """
        Full detection pipeline: load → normalise → features →
        train → score → explain → whitelist → (optional) export.
        """
        import ml_engine  # deferred to avoid circular imports

        ml_engine._log(f"[*] Adapter: {self.CHAIN_NAME}")

        # 1. Load
        df = self.load_data(source)

        # 2. Normalise addresses
        df = self.normalize_addresses(df)

        # 3. Validate required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{self.CHAIN_NAME}] Missing required columns after "
                f"normalisation: {missing}"
            )

        # 4. Parse timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 5. Extract features (chain-specific)
        features, df = self.extract_features(df)

        # 6–8. Train → Score → Explain  (chain-agnostic core)
        model, predictions, raw_scores = ml_engine.train_model(
            features, contamination=contamination,
        )
        risk_scores = ml_engine.compute_risk_scores(raw_scores)
        enriched_df = ml_engine.generate_all_explanations(
            df, features, predictions, risk_scores,
        )

        # 9. Institutional whitelist
        enriched_df = ml_engine.apply_institutional_whitelist(enriched_df)

        # 10. Optional export
        if output_csv:
            ml_engine.export_results(enriched_df, output_csv)

        return enriched_df, model, features

    def run_pipeline_from_df(
        self,
        df: pd.DataFrame,
        contamination: float = 0.05,
    ) -> Tuple[pd.DataFrame, IsolationForest, pd.DataFrame]:
        """Convenience wrapper when the source is already a DataFrame."""
        return self.run_pipeline(df, contamination=contamination)


# ═══════════════════════════════════════════════════════════════════════════
# BITCOIN CSV ADAPTER (concrete)
# ═══════════════════════════════════════════════════════════════════════════

class BitcoinCSVAdapter(BaseTransactionAdapter):
    """
    Concrete adapter for the MITHYA Bitcoin CSV schema.

    Expected CSV columns::

        timestamp, src_ip, dst_ip, src_port, dst_port,
        txid, input_addresses, output_addresses,
        input_amounts, output_amounts, fee, script_type, geo_country

    Pipe-delimited multi-value columns (``input_addresses``,
    ``output_addresses``, ``input_amounts``, ``output_amounts``) are
    normalised on load — whitespace is stripped and empty segments
    are removed.
    """

    CHAIN_NAME = "Bitcoin"

    # Bitcoin-specific required columns (superset of base)
    REQUIRED_COLUMNS = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "txid", "input_addresses", "output_addresses",
        "input_amounts", "output_amounts", "fee", "script_type",
        "geo_country",
    ]

    def load(self, filepath: str) -> pd.DataFrame:
        """
        Read a Bitcoin CSV file and return a canonical DataFrame.

        Parameters
        ----------
        filepath : str
            Path to a ``.csv`` file.

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with canonical schema.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not point to an existing file.
        ValueError
            If the CSV is malformed or missing the ``txid`` column.
        """
        filepath = str(filepath)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"[{self.CHAIN_NAME}/CSV] File not found: {filepath}"
            )

        logger.info("[%s/CSV] Loading file: %s", self.CHAIN_NAME, filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as exc:
            raise ValueError(
                f"[{self.CHAIN_NAME}/CSV] Failed to parse CSV file "
                f"'{filepath}': {exc}"
            ) from exc

        if df.empty:
            logger.warning(
                "[%s/CSV] File '%s' parsed successfully but contains "
                "0 rows.", self.CHAIN_NAME, filepath,
            )

        logger.info(
            "[%s/CSV] Loaded %d transactions with columns: %s",
            self.CHAIN_NAME, len(df), list(df.columns),
        )

        return _normalise_dataframe(df, f"{self.CHAIN_NAME}/CSV")

    def load_data(self, source: str | pd.DataFrame) -> pd.DataFrame:
        """
        Load Bitcoin transaction data from a CSV path or DataFrame.
        """
        import ml_engine

        if isinstance(source, pd.DataFrame):
            df = source.copy()
            ml_engine._log(
                f"[*] [{self.CHAIN_NAME}] Received DataFrame with "
                f"{len(df)} transactions"
            )
        elif isinstance(source, str) and os.path.isfile(source):
            df = pd.read_csv(source, parse_dates=["timestamp"])
            ml_engine._log(
                f"[*] [{self.CHAIN_NAME}] Loaded {len(df)} transactions "
                f"from {source}"
            )
        else:
            raise FileNotFoundError(
                f"[{self.CHAIN_NAME}] Source not found: {source}"
            )

        return df

    def normalize_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise Bitcoin pipe-delimited address columns.

        * Strips leading/trailing whitespace from each address
        * Replaces NaN / None with empty strings to prevent
          downstream ``str.split()`` failures
        """
        for col in ("input_addresses", "output_addresses"):
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    # Normalise each address within the pipe list
                    .apply(
                        lambda s: "|".join(
                            a.strip() for a in s.split("|") if a.strip()
                        )
                    )
                )
        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Delegate to the existing ``ml_engine.engineer_features()`` —
        reuses all graph-aware, behavioural, and port-risk features.
        """
        import ml_engine
        return ml_engine.engineer_features(df)


# ═══════════════════════════════════════════════════════════════════════════
# BITCOIN JSON ADAPTER (concrete)
# ═══════════════════════════════════════════════════════════════════════════

class BitcoinJSONAdapter(BaseTransactionAdapter):
    """
    Concrete adapter for Bitcoin transaction data in JSON format.

    Accepts two JSON structures:

    1. **Array of objects** (most common)::

        [
          {
            "txid": "abc123…",
            "input_addresses": ["addr1", "addr2"],
            "output_addresses": ["addr3"],
            "input_amounts": [0.5, 0.3],
            "output_amounts": [0.79],
            "fee": 0.01,
            "src_port": 8333
          },
          …
        ]

    2. **Key-value dictionary** (keyed by txid)::

        {
          "abc123…": {
            "input_addresses": "addr1|addr2",
            "output_addresses": "addr3",
            "input_amounts": "0.5|0.3",
            "output_amounts": "0.79",
            "fee": 0.01,
            "src_port": 8333
          },
          …
        }

    In dictionary mode, the outer key is used as ``txid`` if the inner
    object does not already contain one.

    Multi-value fields accept both JSON arrays and pipe-delimited
    strings — the adapter normalises everything to pipe-delimited
    strings in the canonical DataFrame.
    """

    CHAIN_NAME = "Bitcoin"

    REQUIRED_COLUMNS = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "txid", "input_addresses", "output_addresses",
        "input_amounts", "output_amounts", "fee", "script_type",
        "geo_country",
    ]

    def load(self, filepath: str) -> pd.DataFrame:
        """
        Read a Bitcoin JSON file and return a canonical DataFrame.

        Parameters
        ----------
        filepath : str
            Path to a ``.json`` file.

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with canonical schema.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not point to an existing file.
        ValueError
            If the JSON is malformed, not a dict/list, or missing
            required fields.
        """
        filepath = str(filepath)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"[{self.CHAIN_NAME}/JSON] File not found: {filepath}"
            )

        logger.info("[%s/JSON] Loading file: %s", self.CHAIN_NAME, filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[{self.CHAIN_NAME}/JSON] Invalid JSON in '{filepath}': "
                f"{exc}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"[{self.CHAIN_NAME}/JSON] Could not read '{filepath}': "
                f"{exc}"
            ) from exc

        records = self._json_to_records(raw, filepath)

        if not records:
            logger.warning(
                "[%s/JSON] File '%s' parsed successfully but produced "
                "0 transaction records.", self.CHAIN_NAME, filepath,
            )

        df = pd.DataFrame(records)

        logger.info(
            "[%s/JSON] Loaded %d transactions with columns: %s",
            self.CHAIN_NAME, len(df), list(df.columns),
        )

        return _normalise_dataframe(df, f"{self.CHAIN_NAME}/JSON")

    # ── Internal: JSON structure detection ────────────────────────

    @staticmethod
    def _json_to_records(
        raw: Any, filepath: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert the parsed JSON object into a flat list of dicts.

        Handles:
        - ``list``  →  assumed to be an array of transaction dicts.
        - ``dict``  →  check if it looks like a single transaction
          (has ``txid``) or a keyed collection of transactions.

        Raises
        ------
        ValueError
            If the top-level type is neither list nor dict, or if dict
            values are not themselves dicts.
        """
        if isinstance(raw, list):
            # Validate that every element is a dict
            for idx, item in enumerate(raw):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"[Bitcoin/JSON] Expected array of objects but "
                        f"element at index {idx} is {type(item).__name__} "
                        f"in '{filepath}'."
                    )
            return raw

        if isinstance(raw, dict):
            # Case A: single transaction object (has 'txid' key)
            if "txid" in raw and any(
                k in raw for k in ("input_addresses", "output_addresses")
            ):
                logger.info(
                    "[Bitcoin/JSON] Detected single transaction object."
                )
                return [raw]

            # Case B: keyed collection  { txid_str: { … }, … }
            records: List[Dict[str, Any]] = []
            for key, value in raw.items():
                if not isinstance(value, dict):
                    raise ValueError(
                        f"[Bitcoin/JSON] Expected dict values in keyed "
                        f"collection, but key '{key}' maps to "
                        f"{type(value).__name__} in '{filepath}'."
                    )
                record = dict(value)
                # Inject the outer key as txid if not already present
                record.setdefault("txid", key)
                records.append(record)

            return records

        raise ValueError(
            f"[Bitcoin/JSON] Top-level JSON type must be list or dict, "
            f"got {type(raw).__name__} in '{filepath}'."
        )

    # ── Pipeline-oriented methods (reuse CSV adapter logic) ──────

    def load_data(self, source: str | pd.DataFrame) -> pd.DataFrame:
        """
        Load Bitcoin transaction data from a JSON path or DataFrame.
        """
        import ml_engine

        if isinstance(source, pd.DataFrame):
            df = source.copy()
            ml_engine._log(
                f"[*] [{self.CHAIN_NAME}/JSON] Received DataFrame with "
                f"{len(df)} transactions"
            )
            return df

        if isinstance(source, str) and os.path.isfile(source):
            # Delegate to the canonical load() method and return the
            # full DataFrame (which may contain more columns than the
            # canonical set — e.g. timestamp, src_ip, etc.).
            try:
                with open(source, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(
                    f"[{self.CHAIN_NAME}/JSON] Failed to read '{source}': "
                    f"{exc}"
                ) from exc

            records = self._json_to_records(raw, source)
            df = pd.DataFrame(records)

            # Normalise pipe-delimited columns
            for col in ("input_addresses", "output_addresses",
                        "input_amounts", "output_amounts"):
                if col in df.columns:
                    df[col] = df[col].apply(_coerce_to_pipe_str)

            # Parse timestamp if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], errors="coerce",
                )

            ml_engine._log(
                f"[*] [{self.CHAIN_NAME}/JSON] Loaded {len(df)} "
                f"transactions from {source}"
            )
            return df

        raise FileNotFoundError(
            f"[{self.CHAIN_NAME}/JSON] Source not found: {source}"
        )

    def normalize_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise address columns — identical logic to the CSV adapter
        since both formats use pipe-delimited strings post-load.
        """
        for col in ("input_addresses", "output_addresses"):
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .apply(
                        lambda s: "|".join(
                            a.strip() for a in s.split("|") if a.strip()
                        )
                    )
                )
        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Delegate to ``ml_engine.engineer_features()``.
        """
        import ml_engine
        return ml_engine.engineer_features(df)


# ═══════════════════════════════════════════════════════════════════════════
# BITCOIN XML ADAPTER (concrete)
# ═══════════════════════════════════════════════════════════════════════════

class BitcoinXMLAdapter(BaseTransactionAdapter):
    """
    Concrete adapter for Bitcoin transaction data in XML format.

    Uses Python's native :mod:`xml.etree.ElementTree` to parse
    hierarchical transaction nodes.

    Expected XML structure::

        <?xml version="1.0" encoding="UTF-8"?>
        <transactions>
          <transaction>
            <txid>abc123…</txid>
            <input_addresses>addr1|addr2</input_addresses>
            <output_addresses>addr3</output_addresses>
            <input_amounts>0.5|0.3</input_amounts>
            <output_amounts>0.79</output_amounts>
            <fee>0.01</fee>
            <src_port>8333</src_port>
            <!-- Additional optional elements preserved as-is -->
            <timestamp>2026-07-27 12:18:39</timestamp>
            <src_ip>1.2.3.4</src_ip>
            <dst_ip>5.6.7.8</dst_ip>
            <dst_port>18332</dst_port>
            <script_type>P2WPKH</script_type>
            <geo_country>US</geo_country>
          </transaction>
          …
        </transactions>

    Multi-value fields can use:

    - Pipe-delimited text: ``<input_addresses>a|b|c</input_addresses>``
    - Nested ``<address>`` / ``<amount>`` child elements::

        <input_addresses>
          <address>addr1</address>
          <address>addr2</address>
        </input_addresses>
    """

    CHAIN_NAME = "Bitcoin"

    REQUIRED_COLUMNS = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "txid", "input_addresses", "output_addresses",
        "input_amounts", "output_amounts", "fee", "script_type",
        "geo_country",
    ]

    def load(self, filepath: str) -> pd.DataFrame:
        """
        Read a Bitcoin XML file and return a canonical DataFrame.

        Parameters
        ----------
        filepath : str
            Path to a ``.xml`` file.

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with canonical schema.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not point to an existing file.
        ValueError
            If the XML is malformed or has an unexpected structure.
        """
        filepath = str(filepath)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"[{self.CHAIN_NAME}/XML] File not found: {filepath}"
            )

        logger.info("[%s/XML] Loading file: %s", self.CHAIN_NAME, filepath)

        try:
            tree = ET.parse(filepath)
        except ET.ParseError as exc:
            raise ValueError(
                f"[{self.CHAIN_NAME}/XML] Malformed XML in '{filepath}': "
                f"{exc}"
            ) from exc

        root = tree.getroot()
        records = self._parse_transactions(root, filepath)

        if not records:
            logger.warning(
                "[%s/XML] File '%s' parsed successfully but produced "
                "0 transaction records.", self.CHAIN_NAME, filepath,
            )

        df = pd.DataFrame(records)

        logger.info(
            "[%s/XML] Loaded %d transactions with columns: %s",
            self.CHAIN_NAME, len(df), list(df.columns),
        )

        return _normalise_dataframe(df, f"{self.CHAIN_NAME}/XML")

    # ── Internal: XML parsing ─────────────────────────────────────

    # Tags that may contain nested child elements for multi-value data
    _MULTI_VALUE_TAGS: Dict[str, str] = {
        "input_addresses": "address",
        "output_addresses": "address",
        "input_amounts": "amount",
        "output_amounts": "amount",
    }

    @classmethod
    def _parse_transactions(
        cls, root: ET.Element, filepath: str,
    ) -> List[Dict[str, Any]]:
        """
        Walk the XML tree and extract transaction records.

        Supports two root conventions:
        - ``<transactions><transaction>…`` (wrapped)
        - ``<transaction>…`` (root IS a single transaction)
        """
        tag = root.tag.lower()

        # Determine the iterable of <transaction> elements
        if tag == "transactions":
            tx_elements = root.findall("transaction")
        elif tag == "transaction":
            # The root itself is a single transaction
            tx_elements = [root]
        else:
            # Try to find <transaction> anywhere as a fallback
            tx_elements = root.findall(".//transaction")
            if not tx_elements:
                raise ValueError(
                    f"[Bitcoin/XML] Could not locate <transaction> "
                    f"elements in '{filepath}'.  Root tag: <{root.tag}>."
                )

        records: List[Dict[str, Any]] = []
        for idx, tx_el in enumerate(tx_elements):
            try:
                record = cls._parse_single_transaction(tx_el)
                records.append(record)
            except Exception as exc:
                logger.error(
                    "[Bitcoin/XML] Error parsing <transaction> #%d in "
                    "'%s': %s — skipping.", idx, filepath, exc,
                )
        return records

    @classmethod
    def _parse_single_transaction(
        cls, tx_el: ET.Element,
    ) -> Dict[str, Any]:
        """
        Convert a single ``<transaction>`` element into a flat dict.

        For multi-value tags (addresses, amounts), checks for nested
        child elements first, then falls back to pipe-delimited text.
        """
        record: Dict[str, Any] = {}

        for child in tx_el:
            tag = child.tag.strip()

            if tag in cls._MULTI_VALUE_TAGS:
                # Check for nested child elements
                child_tag = cls._MULTI_VALUE_TAGS[tag]
                nested = child.findall(child_tag)
                if nested:
                    # Build pipe-delimited string from nested elements
                    values = [
                        (el.text or "").strip()
                        for el in nested
                        if (el.text or "").strip()
                    ]
                    record[tag] = "|".join(values)
                else:
                    # Fallback to text content (may be pipe-delimited)
                    record[tag] = (child.text or "").strip()
            else:
                # Simple scalar element
                record[tag] = (child.text or "").strip()

        return record

    # ── Pipeline-oriented methods ────────────────────────────────

    def load_data(self, source: str | pd.DataFrame) -> pd.DataFrame:
        """
        Load Bitcoin transaction data from an XML path or DataFrame.
        """
        import ml_engine

        if isinstance(source, pd.DataFrame):
            df = source.copy()
            ml_engine._log(
                f"[*] [{self.CHAIN_NAME}/XML] Received DataFrame with "
                f"{len(df)} transactions"
            )
            return df

        if isinstance(source, str) and os.path.isfile(source):
            try:
                tree = ET.parse(source)
            except ET.ParseError as exc:
                raise ValueError(
                    f"[{self.CHAIN_NAME}/XML] Malformed XML in "
                    f"'{source}': {exc}"
                ) from exc

            root = tree.getroot()
            records = self._parse_transactions(root, source)
            df = pd.DataFrame(records)

            # Normalise pipe-delimited columns
            for col in ("input_addresses", "output_addresses",
                        "input_amounts", "output_amounts"):
                if col in df.columns:
                    df[col] = df[col].apply(_coerce_to_pipe_str)

            # Parse timestamp if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], errors="coerce",
                )

            # Ensure numeric types for port / fee columns
            for col in ("src_port", "dst_port"):
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col], errors="coerce",
                    ).fillna(0).astype(int)
            if "fee" in df.columns:
                df["fee"] = pd.to_numeric(
                    df["fee"], errors="coerce",
                ).fillna(_DEFAULT_FEE)

            ml_engine._log(
                f"[*] [{self.CHAIN_NAME}/XML] Loaded {len(df)} "
                f"transactions from {source}"
            )
            return df

        raise FileNotFoundError(
            f"[{self.CHAIN_NAME}/XML] Source not found: {source}"
        )

    def normalize_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise address columns — identical to CSV adapter logic.
        """
        for col in ("input_addresses", "output_addresses"):
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .apply(
                        lambda s: "|".join(
                            a.strip() for a in s.split("|") if a.strip()
                        )
                    )
                )
        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Delegate to ``ml_engine.engineer_features()``.
        """
        import ml_engine
        return ml_engine.engineer_features(df)


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER FACTORY
# ═══════════════════════════════════════════════════════════════════════════

# ── Extension → Adapter mapping ──────────────────────────────────
_EXTENSION_REGISTRY: Dict[str, type[BaseTransactionAdapter]] = {
    ".csv":  BitcoinCSVAdapter,
    ".json": BitcoinJSONAdapter,
    ".xml":  BitcoinXMLAdapter,
}

# ── MIME-type → Adapter mapping (fallback) ───────────────────────
_MIME_REGISTRY: Dict[str, type[BaseTransactionAdapter]] = {
    "text/csv":                  BitcoinCSVAdapter,
    "application/json":          BitcoinJSONAdapter,
    "application/xml":           BitcoinXMLAdapter,
    "text/xml":                  BitcoinXMLAdapter,
}

# ── Chain-name registry (backward compatibility) ─────────────────
_ADAPTER_REGISTRY: dict[str, type[BaseTransactionAdapter]] = {
    "bitcoin":      BitcoinCSVAdapter,
    "bitcoin_csv":  BitcoinCSVAdapter,
    "bitcoin_json": BitcoinJSONAdapter,
    "bitcoin_xml":  BitcoinXMLAdapter,
}


def get_adapter(
    file_path_or_buffer: Union[str, None] = None,
    *,
    chain: Optional[str] = None,
) -> BaseTransactionAdapter:
    """
    Factory function: return the appropriate adapter instance.

    Resolution order:

    1. If *file_path_or_buffer* is provided, inspect its **file
       extension** first (fastest), then fall back to **MIME-type**
       guessing via :func:`mimetypes.guess_type`.
    2. If *chain* is provided (e.g. ``"bitcoin_json"``), look it up
       in the chain-name registry.
    3. If neither is provided, default to :class:`BitcoinCSVAdapter`.

    Parameters
    ----------
    file_path_or_buffer : str | None
        A filesystem path or URL-like string whose extension or
        MIME-type determines the adapter.
    chain : str | None
        Explicit chain/format identifier (case-insensitive).
        Overrides extension-based detection when both are given.

    Returns
    -------
    BaseTransactionAdapter
        A ready-to-use adapter instance.

    Raises
    ------
    ValueError
        If the format cannot be determined or is unsupported.

    Examples
    --------
    >>> adapter = get_adapter("data/transactions.json")
    >>> isinstance(adapter, BitcoinJSONAdapter)
    True

    >>> adapter = get_adapter(chain="bitcoin_xml")
    >>> isinstance(adapter, BitcoinXMLAdapter)
    True

    >>> adapter = get_adapter("data.csv")
    >>> df = adapter.load("data.csv")
    """
    # ── Priority 1: explicit chain override ──────────────────────
    if chain is not None:
        key = chain.strip().lower()
        if key not in _ADAPTER_REGISTRY:
            available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
            raise ValueError(
                f"No adapter registered for chain '{chain}'. "
                f"Available: {available}"
            )
        logger.info(
            "Factory resolved chain='%s' → %s",
            chain, _ADAPTER_REGISTRY[key].__name__,
        )
        return _ADAPTER_REGISTRY[key]()

    # ── Priority 2: file extension ───────────────────────────────
    if file_path_or_buffer is not None:
        path_str = str(file_path_or_buffer)
        ext = Path(path_str).suffix.lower()

        if ext in _EXTENSION_REGISTRY:
            adapter_cls = _EXTENSION_REGISTRY[ext]
            logger.info(
                "Factory resolved extension '%s' → %s",
                ext, adapter_cls.__name__,
            )
            return adapter_cls()

        # ── Priority 3: MIME-type fallback ───────────────────────
        mime_type, _ = mimetypes.guess_type(path_str)
        if mime_type and mime_type in _MIME_REGISTRY:
            adapter_cls = _MIME_REGISTRY[mime_type]
            logger.info(
                "Factory resolved MIME '%s' → %s",
                mime_type, adapter_cls.__name__,
            )
            return adapter_cls()

        raise ValueError(
            f"Cannot determine adapter for '{path_str}'. "
            f"Supported extensions: {sorted(_EXTENSION_REGISTRY.keys())}. "
            f"Supported MIME types: {sorted(_MIME_REGISTRY.keys())}."
        )

    # ── Default: CSV adapter ─────────────────────────────────────
    logger.info(
        "Factory: no file path or chain specified — defaulting to "
        "BitcoinCSVAdapter."
    )
    return BitcoinCSVAdapter()
