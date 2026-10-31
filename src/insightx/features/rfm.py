"""RFM (Recency, Frequency, Monetary) feature calculation module."""

from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from insightx.utils.logger import get_logger

logger = get_logger(__name__)


class RFMAnalyzer:
    """Computes customer-level RFM metrics and segment labels."""

    def __init__(self, reference_date: Optional[datetime] = None):
        """Args:

        reference_date: Snapshot date for recency calculation (if None, max
        purchase date + 1 day is used).
        """
        self.reference_date = reference_date

    def calculate_rfm(
        self,
        orders_df: pd.DataFrame,
        payments_df: pd.DataFrame,
        customers_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculates Recency, Frequency, and Monetary values per unique customer.

        Note on Olist:
            `customer_id` is unique per order, whereas `customer_unique_id`
            identifies the actual individual customer across multiple purchases.

        Returns:
            pd.DataFrame: Customer-level RFM DataFrame indexed or keyed by `customer_unique_id`.
        """
        logger.info("Computing customer RFM metrics...")
        # Placeholder signature - full implementation to be developed with user approval
        return pd.DataFrame(columns=["customer_unique_id", "recency", "frequency", "monetary"])

    def assign_rfm_scores(self, rfm_df: pd.DataFrame, quantiles: int = 4) -> pd.DataFrame:
        """Assigns R, F, and M quartile/quantile scores (1 to quantiles) and segment labels.

        Args:
            rfm_df: Output from calculate_rfm.
            quantiles: Number of bins (e.g., 4 or 5).

        Returns:
            pd.DataFrame: RFM DataFrame with score columns and customer segment tags.
        """
        logger.info(f"Assigning RFM scores using {quantiles} quantiles...")
        return rfm_df
