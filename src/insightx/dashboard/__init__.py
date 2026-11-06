"""Interactive BI and Machine Learning Dashboard."""

from insightx.dashboard.data_loader import (
    load_churn_metrics,
    load_clv_metrics,
    load_monthly_revenue_timeseries,
    load_raw_views,
    load_recsys_metrics,
    load_rfm_distribution,
    load_summary_kpis,
)

__all__ = [
    "load_churn_metrics",
    "load_clv_metrics",
    "load_monthly_revenue_timeseries",
    "load_raw_views",
    "load_recsys_metrics",
    "load_rfm_distribution",
    "load_summary_kpis",
]
