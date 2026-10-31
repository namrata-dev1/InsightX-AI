"""Shared pytest fixtures for InsightX test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_orders_data() -> pd.DataFrame:
    """Provides a synthetic sample of orders data for quick unit tests."""
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_id": ["c1", "c2", "c1", "c3"],
            "order_status": ["delivered", "delivered", "delivered", "delivered"],
            "order_purchase_timestamp": [
                "2018-01-01 10:00:00",
                "2018-02-01 12:00:00",
                "2018-03-01 14:00:00",
                "2018-04-01 16:00:00",
            ],
        }
    )


@pytest.fixture
def sample_payments_data() -> pd.DataFrame:
    """Provides a synthetic sample of payment data for quick unit tests."""
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "payment_value": [100.0, 50.0, 150.0, 200.0],
            "payment_type": ["credit_card", "boleto", "credit_card", "voucher"],
        }
    )
