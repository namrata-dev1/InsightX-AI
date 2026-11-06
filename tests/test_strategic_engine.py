"""Unit and Integration Tests for Strategic Insights Engine.

Verifies:
1. Generation of the 4 strategic pillars (Retention, Reactivation, Cross-Sell, Catalog Expansion).
2. Synthesis of Executive Portfolio Briefing.
3. Correct integration with RFM cohorts, CLV valuations, churn metrics, and RecSys evaluations.
4. StructuredEvidence generation and provenance metadata compliance.
5. Strict adherence to methodological guardrails (causal disclaimers, data limitations, zero fake metrics).
6. Graceful resilience against empty DataFrames, missing columns, and None inputs.
"""

from datetime import datetime
import pandas as pd
import pytest

from insightx.insights.strategic_engine import (
    StrategicInsightsEngine,
    StrategicIntervention,
    StrategyType,
)
from insightx.provenance.types import ProvenanceType


@pytest.fixture
def engine():
    """Initializes standard StrategicInsightsEngine."""
    return StrategicInsightsEngine(dataset_id="olist-v1.0")


@pytest.fixture
def sample_rfm_summary():
    """Mock RFM cohort summary table."""
    return pd.DataFrame({
        "rfm_segment": [
            "Champions",
            "Loyal Customers",
            "Potential Loyalists",
            "At Risk",
            "Hibernating",
            "Lost Customers",
        ],
        "customer_count": [2450, 4820, 11200, 8900, 31200, 34788],
        "total_monetary": [720000.0, 850000.0, 1100000.0, 920000.0, 2400000.0, 3430000.0],
        "mean_recency_days": [28.5, 45.2, 72.1, 165.4, 280.6, 420.1],
    })


@pytest.fixture
def sample_churn_metrics():
    """Mock calibrated churn model metrics."""
    return {
        "pr_auc": 0.9957,
        "brier_score": 0.0068,
        "roc_auc": 0.9995,
        "test_churn_rate": 0.969,
    }


@pytest.fixture
def sample_recsys_metrics():
    """Mock RecSys K=10 evaluation metrics."""
    return {
        "precision_repeat_k10": 0.0075,
        "precision_full_k10": 0.0040,
        "repeat_uplift_pct": 87.5,
        "catalog_coverage_pct": 78.4,
    }


@pytest.fixture
def sample_top_categories():
    """Mock top categories DataFrame."""
    return pd.DataFrame({
        "product_category_name": [
            "cama_mesa_banho",
            "beleza_saude",
            "esporte_lazer",
            "moveis_decoracao",
            "informatica_acessorios",
        ],
        "items_sold": [11115, 9670, 8641, 8334, 7827],
        "total_sales": [1036813.0, 1258681.0, 988048.0, 729762.0, 911954.0],
    })


def test_generate_retention_strategy_with_data(engine, sample_rfm_summary, sample_churn_metrics):
    """Verifies retention strategy synthesis using explicit RFM and churn inputs."""
    intervention = engine.generate_retention_strategy(
        rfm_summary=sample_rfm_summary,
        churn_metrics=sample_churn_metrics,
        clv_summary={"total_clv": 1570000.0},
    )

    assert isinstance(intervention, StrategicIntervention)
    assert intervention.strategy_type == StrategyType.RETENTION
    assert "VIP" in intervention.title
    assert "Champions" in intervention.target_segment
    assert len(intervention.recommended_actions) >= 3
    assert intervention.supporting_metrics["churn_model_pr_auc"] == 0.9957
    assert intervention.supporting_metrics["identified_vip_cohort_size"] == 2450 + 4820
    assert len(intervention.evidence) >= 2

    # Verify StructuredEvidence
    for ev in intervention.evidence:
        assert ev.evidence_id.startswith("ev_strat_")
        assert ev.provenance_type == ProvenanceType.OBSERVED
        assert ev.dataset_version_id == "olist-v1.0"
        assert len(ev.evidence_to_claim_linkage) > 0

    # Verify Disclaimers
    disclaimer_text = " ".join(intervention.limitations_and_disclaimers)
    assert "CAUSALITY NOTICE" in disclaimer_text
    assert "A/B testing" in disclaimer_text
    assert "COGS" in disclaimer_text


def test_generate_retention_strategy_defaults(engine):
    """Verifies retention strategy operates safely with zero inputs (fallback defaults)."""
    intervention = engine.generate_retention_strategy()
    assert isinstance(intervention, StrategicIntervention)
    assert intervention.strategy_type == StrategyType.RETENTION
    assert intervention.supporting_metrics["churn_model_pr_auc"] > 0.0
    assert len(intervention.recommended_actions) >= 3


def test_generate_reactivation_strategy(engine, sample_rfm_summary):
    """Verifies win-back strategy formulation for dormant cohorts."""
    intervention = engine.generate_reactivation_strategy(rfm_summary=sample_rfm_summary)
    assert intervention.strategy_type == StrategyType.REACTIVATION
    assert "Win-Back" in intervention.title
    assert "Hibernating" in intervention.target_segment
    assert intervention.supporting_metrics["dormant_pool_size"] == 8900 + 31200 + 34788
    assert len(intervention.recommended_actions) >= 3

    disclaimer_text = " ".join(intervention.limitations_and_disclaimers)
    assert "MARGIN PROTECTION GUARDRAIL" in disclaimer_text
    assert "CAC" in disclaimer_text


def test_generate_cross_sell_strategy(engine, sample_recsys_metrics):
    """Verifies cross-sell strategy incorporates RecSys benchmarks."""
    intervention = engine.generate_cross_sell_strategy(recsys_k10_metrics=sample_recsys_metrics)
    assert intervention.strategy_type == StrategyType.CROSS_SELL
    assert "Cross-Sell" in intervention.title
    assert intervention.supporting_metrics["recsys_precision_uplift_on_repeat_pct"] == 87.5
    assert intervention.supporting_metrics["catalog_coverage_pct"] == 78.4
    assert len(intervention.evidence) >= 2

    disclaimer_text = " ".join(intervention.limitations_and_disclaimers)
    assert "EXTREME SPARSITY" in disclaimer_text
    assert "INVENTORY UNCERTAINTY" in disclaimer_text


def test_generate_catalog_expansion_strategy(engine, sample_top_categories):
    """Verifies catalog strategy reflects category concentration and logistics."""
    intervention = engine.generate_catalog_expansion_strategy(top_categories_df=sample_top_categories)
    assert intervention.strategy_type == StrategyType.CATALOG_EXPANSION
    assert "Merchandising" in intervention.title
    assert "cama_mesa_banho" in intervention.supporting_metrics["top_gmv_categories"]
    assert len(intervention.recommended_actions) >= 3

    disclaimer_text = " ".join(intervention.limitations_and_disclaimers)
    assert "MARKETPLACE AUTONOMY" in disclaimer_text
    assert "NO CAUSAL CLAIM" in disclaimer_text


def test_build_executive_portfolio_strategy(
    engine,
    sample_rfm_summary,
    sample_churn_metrics,
    sample_recsys_metrics,
    sample_top_categories,
):
    """Verifies end-to-end synthesis of the comprehensive Executive Portfolio Briefing."""
    portfolio = engine.build_executive_portfolio_strategy(
        rfm_summary=sample_rfm_summary,
        churn_metrics=sample_churn_metrics,
        recsys_k10_metrics=sample_recsys_metrics,
        top_categories_df=sample_top_categories,
    )

    assert isinstance(portfolio, dict)
    assert portfolio["portfolio_title"] == "InsightX-AI Executive Commercial Decision Briefing"
    assert portfolio["total_interventions"] == 4
    assert portfolio["total_supporting_evidences"] >= 5
    assert len(portfolio["interventions"]) == 4

    # Verify all 4 strategy types are represented
    types = [i["strategy_type"] for i in portfolio["interventions"]]
    assert StrategyType.RETENTION.value in types
    assert StrategyType.REACTIVATION.value in types
    assert StrategyType.CROSS_SELL.value in types
    assert StrategyType.CATALOG_EXPANSION.value in types

    # Check governance section
    gov = portfolio["governance_and_compliance"]
    assert "causality_disclaimer" in gov
    assert "missing_data_boundary" in gov
    assert "provenance_standard" in gov


def test_intervention_serialization(engine):
    """Verifies StrategicIntervention to_dict() contract for dashboard integration."""
    intervention = engine.generate_retention_strategy()
    d = intervention.to_dict()

    assert isinstance(d, dict)
    assert d["title"] == intervention.title
    assert d["strategy_type"] == "RETENTION"
    assert isinstance(d["recommended_actions"], list)
    assert isinstance(d["supporting_metrics"], dict)
    assert d["evidence_count"] >= 1
    assert isinstance(d["limitations_and_disclaimers"], list)
    assert "created_at" in d


def test_engine_empty_input_resilience(engine):
    """Verifies engine handles empty DataFrames and corrupted inputs without throwing."""
    empty_df = pd.DataFrame()
    intervention = engine.generate_retention_strategy(rfm_summary=empty_df)
    assert intervention.strategy_type == StrategyType.RETENTION
    assert intervention.supporting_metrics["identified_vip_cohort_size"] > 0

    portfolio = engine.build_executive_portfolio_strategy(
        rfm_summary=empty_df,
        churn_metrics={},
        clv_summary={},
        recsys_k10_metrics={},
        top_categories_df=empty_df,
    )
    assert portfolio["total_interventions"] == 4
