"""Data loading and schema validation interfaces for the Olist dataset.

Supports automatic path resolution across Kaggle, Google Colab, and local environments.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from insightx.config import resolve_data_dir
from insightx.utils.logger import get_logger

logger = get_logger(__name__)

# Standard filenames for the Olist Brazilian E-Commerce dataset
OLIST_TABLE_NAMES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]


class OlistDataLoader:
    """Handles loading, inspecting, and caching Olist relational tables."""

    def __init__(self, raw_data_dir: Optional[Path] = None):
        """Initializes loader with automatic cloud or local path resolution.

        Args:
            raw_data_dir: Optional custom path. If None, uses resolve_data_dir().
        """
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else resolve_data_dir()
        logger.info(f"OlistDataLoader initialized with data directory: {self.raw_data_dir}")

    def load_table(self, filename: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """Loads a single CSV table from the resolved data directory.

        Args:
            filename: Name of the CSV file.
            nrows: Optional limit on rows to read (useful for rapid cloud inspection).

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}.\n"
                f"Current environment search path: {self.raw_data_dir}"
            )

        logger.info(f"Loading {filename} (nrows={nrows})...")
        return pd.read_csv(filepath, nrows=nrows)

    def load_all_tables(self, nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Loads all present Olist tables into a dictionary.

        Args:
            nrows: Optional row limit per table for quick memory-efficient checks.

        Returns:
            Dict[str, pd.DataFrame]: Mapping from table short name to DataFrame.
        """
        tables = {}
        for name in OLIST_TABLE_NAMES:
            table_key = (
                name.replace(".csv", "")
                .replace("olist_", "")
                .replace("_dataset", "")
            )
            try:
                tables[table_key] = self.load_table(name, nrows=nrows)
            except FileNotFoundError:
                logger.warning(f"File {name} not found in {self.raw_data_dir}. Skipping.")
        return tables

    def inspect_dataset_summary(self) -> pd.DataFrame:
        """Inspects all available tables and generates a structural summary without heavy memory use.

        Returns:
            pd.DataFrame: Summary containing table names, file sizes, row counts, and column counts.
        """
        summary_records: List[Dict[str, any]] = []

        for name in OLIST_TABLE_NAMES:
            filepath = self.raw_data_dir / name
            if filepath.exists():
                file_size_mb = round(filepath.stat().st_size / (1024 * 1024), 2)
                # Sample the first row to get column count
                sample_df = pd.read_csv(filepath, nrows=1)
                # Count total rows efficiently
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    row_count = sum(1 for _ in f) - 1

                summary_records.append(
                    {
                        "table_name": name,
                        "size_mb": file_size_mb,
                        "rows": max(0, row_count),
                        "columns": len(sample_df.columns),
                        "status": "Available",
                    }
                )
            else:
                summary_records.append(
                    {
                        "table_name": name,
                        "size_mb": 0.0,
                        "rows": 0,
                        "columns": 0,
                        "status": "Not Found",
                    }
                )

        return pd.DataFrame(summary_records)
