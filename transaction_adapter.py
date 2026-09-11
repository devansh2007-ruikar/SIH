"""
transaction_adapter.py — Cross-Chain Transaction Adapter Layer
==============================================================

Provides an abstract base class for transaction ingestion and a
concrete Bitcoin CSV adapter.  Future chains (Ethereum, Monero, …)
can be supported by adding new adapter subclasses — no changes to
the core ML pipeline are required.

Architecture
------------
::

    ┌────────────────────────────────┐
    │   BaseTransactionAdapter (ABC) │
    │  ─────────────────────────────│
    │  + load_data()                │
    │  + normalize_addresses()      │
    │  + extract_features()         │
    │  + run_pipeline()  ← template │
    └───────────┬───────────────────┘
                │  inherits
    ┌───────────▼───────────────────┐
    │     BitcoinCSVAdapter         │
    │  ─────────────────────────────│
    │  Delegates to ml_engine.*()   │
    └───────────────────────────────┘

Usage
-----
>>> from transaction_adapter import BitcoinCSVAdapter
>>> adapter = BitcoinCSVAdapter()
>>> enriched_df, model = adapter.run_pipeline("bitcoin_traffic.csv")

The Streamlit frontend instantiates the adapter and calls
``adapter.run_pipeline_from_df(df, contamination)`` — swapping in a
future ``EthereumAdapter`` requires only changing the instantiation.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

import pandas as pd

# pyrefly: ignore [missing-import]
from sklearn.ensemble import IsolationForest

# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════

class BaseTransactionAdapter(ABC):
    """
    Abstract contract for chain-specific transaction ingestion.

    Every concrete adapter **must** implement:

    * :meth:`load_data`  — read raw data from a source
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
    """

    CHAIN_NAME = "Bitcoin"

    # Bitcoin-specific required columns (superset of base)
    REQUIRED_COLUMNS = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "txid", "input_addresses", "output_addresses",
        "input_amounts", "output_amounts", "fee", "script_type",
        "geo_country",
    ]

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
# ADAPTER FACTORY
# ═══════════════════════════════════════════════════════════════════════════

# Registry of available adapters (extend when adding new chains)
_ADAPTER_REGISTRY: dict[str, type[BaseTransactionAdapter]] = {
    "bitcoin": BitcoinCSVAdapter,
}


def get_adapter(chain: str = "bitcoin") -> BaseTransactionAdapter:
    """
    Factory function: return an adapter instance for the given chain.

    Parameters
    ----------
    chain : str
        Chain name (case-insensitive).  Currently supported: ``bitcoin``.

    Returns
    -------
    BaseTransactionAdapter
        A ready-to-use adapter instance.

    Raises
    ------
    ValueError
        If the chain is not registered.

    Examples
    --------
    >>> adapter = get_adapter("bitcoin")
    >>> enriched_df, model = adapter.run_pipeline("data.csv")
    """
    key = chain.strip().lower()
    if key not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise ValueError(
            f"No adapter registered for chain '{chain}'. "
            f"Available: {available}"
        )
    return _ADAPTER_REGISTRY[key]()
