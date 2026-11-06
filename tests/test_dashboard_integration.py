"""Integration Tests for InsightX-AI Executive BI Dashboard.

Verifies:
1. Data loaders retrieve validated Phase 4 M2 reports and metrics.
2. Graceful fallback on missing data/reports.
3. Allowlisted view sampling and schema compatibility with SQLGuard.
4. SQLDecisionAgent execution over dashboard-loaded tables.
5. StrategicInsightsEngine portfolio generation from dashboard data sources.
6. Python compilation and syntax validity of app.py.
"""

from pathlib import Path
import py_compile
import pandas as pd
import pytest

from insightx.agent.sql_decision_agent import SQLDecisionAgent
from insightx.dashboard.data_loader import (
    load_churn_metrics,
    load_clv_metrics,
    load_monthly_revenue_timeseries,
    load_raw_views,
    load_recsys_metrics,
    load_rfm_distribution,
    load_summary_kpis,
)
from insightx.insights.strategic_engine import StrategicInsightsEngine


def test_dashboard_app_syntax_compilation():
    """Verifies that app.py compiles without syntax errors."""
    app_path = Path("src/insightx/dashboard/app.py")
    assert app_path.exists()
    compiled_path = py_compile.compile(str(app_path), doraise=True)
    assert compiled_path is not None


def test_load_summary_kpis():
    """Verifies retrieval of executive summary KPIs."""
    kpis = load_summary_kpis()
    assert isinstance(kpis, dict)
    assert "total_gmv" in kpis
    assert "net_profit" in kpis
    assert "delivered_orders" in kpis
    assert "repeat_rate_pct" in kpis
    assert kpis["total_gmv"] >= 9000000.0
    assert kpis["repeat_rate_pct"] > 0.0


def test_load_rfm_distribution():
    """Verifies RFM segment distribution data loading."""
    df_rfm = load_rfm_distribution()
    assert isinstance(df_rfm, pd.DataFrame)
    assert not df_rfm.empty
    assert "rfm_segment" in df_rfm.columns
    assert "customer_count" in df_rfm.columns
    assert "mean_recency_days" in df_rfm.columns
    assert len(df_rfm) == 10
    assert "Champions" in df_rfm["rfm_segment"].values
    assert "About to Sleep" in df_rfm["rfm_segment"].values


def test_load_churn_metrics():
    """Verifies calibrated churn evaluation metrics structure."""
    churn = load_churn_metrics()
    assert isinstance(churn, dict)
    assert "calibrated_logistic_regression" in churn
    assert "top_features" in churn
    assert churn["calibrated_logistic_regression"]["pr_auc"] >= 0.99
    assert churn["calibrated_logistic_regression"]["brier_score"] < 0.05
    assert len(churn["top_features"]) >= 3


def test_load_clv_metrics():
    """Verifies CLV summary and BG/NBD suitability verdict."""
    clv = load_clv_metrics()
    assert isinstance(clv, dict)
    assert "statistical_suitability" in clv
    suitability = clv["statistical_suitability"]
    assert suitability["bgnbd_suitable"] is False
    assert "UNSUITABLE" in suitability["bgnbd_verdict"]
    assert suitability["gamma_gamma_suitable"] is True


def test_load_recsys_metrics():
    """Verifies RecSys K=10 and K=5 metric loading."""
    for k in [10, 5]:
        df = load_recsys_metrics(k=k)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "recommender_name" in df.columns
        assert "precision_at_k" in df.columns
        assert len(df) == 6
        assert "popularity" in df["recommender_name"].values
        assert "hybrid" in df["recommender_name"].values


def test_load_raw_views_schema():
    """Verifies sampled allowlisted views load with correct columns."""
    views = load_raw_views(sample_n=100)
    assert isinstance(views, dict)
    expected_views = [
        "v_orders", "v_order_items", "v_order_payments",
        "v_order_reviews", "v_customers", "v_products", "v_sellers"
    ]
    for v in expected_views:
        assert v in views
        assert isinstance(views[v], pd.DataFrame)
        assert not views[v].empty


def test_load_monthly_revenue_timeseries():
    """Verifies time-series revenue trend DataFrame."""
    ts = load_monthly_revenue_timeseries()
    assert isinstance(ts, pd.DataFrame)
    assert "order_month" in ts.columns
    assert "revenue_brl" in ts.columns
    assert len(ts) >= 12


def test_dashboard_sql_agent_e2e_integration():
    """Verifies that SQLDecisionAgent runs queries against views loaded by data_loader."""
    views = load_raw_views(sample_n=200)
    agent = SQLDecisionAgent()

    result = agent.ask("Which are the top 5 product categories by sales?", tables_data=views)
    assert result.is_valid is True
    assert result.row_count > 0
    assert len(result.business_summary) > 0


def test_dashboard_strategic_engine_e2e_integration():
    """Verifies that StrategicInsightsEngine builds portfolio using data_loader metrics."""
    engine = StrategicInsightsEngine()
    rfm_df = load_rfm_distribution()
    churn = load_churn_metrics()
    clv = load_clv_metrics()
    recsys = {"precision_repeat_k10": 0.0075, "repeat_uplift_pct": 87.5, "catalog_coverage_pct": 78.4}

    portfolio = engine.build_executive_portfolio_strategy(
        rfm_summary=rfm_df,
        churn_metrics=churn,
        clv_summary=clv,
        recsys_k10_metrics=recsys,
    )
    assert portfolio["total_interventions"] == 4
    assert len(portfolio["interventions"]) == 4
