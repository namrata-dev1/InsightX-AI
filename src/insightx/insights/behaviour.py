"""Customer Behavioural Insights and Strategic Retention Generator.

Translates RFM segments, forward CLV estimates, and churn risk predictions into
auditable, evidence-backed business insights with strict causality disclaimers.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from insightx.provenance.evidence import StructuredEvidence
from insightx.provenance.types import ProvenanceType
from insightx.utils.logger import get_logger

logger = get_logger(__name__)


class CustomerBehaviourInsights:
    """Generates evidence-backed business insights and executive briefings from customer intelligence."""

    def __init__(self, dataset_id: str = "olist-v1.0"):
        self.dataset_id = dataset_id

    def generate_segment_clv_insights(
        self,
        customer_portfolio_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Cross-tabulates RFM segments with forward CLV projections and churn probabilities.

        Args:
            customer_portfolio_df: DataFrame containing rfm_segment,
                                   forward_projected_gross_clv, and optionally churn probability.

        Returns:
            Dict containing segment breakdown table, evidence objects, and strategic findings.
        """
        df = customer_portfolio_df.copy()

        segment_col = "rfm_segment" if "rfm_segment" in df.columns else "segment"
        clv_col = "forward_projected_gross_clv" if "forward_projected_gross_clv" in df.columns else "monetary"
        prob_col = "predicted_churn_probability" if "predicted_churn_probability" in df.columns else None

        agg_dict = {
            "customer_count": (segment_col, "count"),
            "total_projected_clv": (clv_col, "sum"),
            "avg_projected_clv": (clv_col, "mean"),
        }
        if prob_col:
            agg_dict["avg_churn_risk"] = (prob_col, "mean")

        segment_summary = df.groupby(segment_col).agg(**agg_dict).reset_index()
        total_customers = len(df)
        total_clv = float(df[clv_col].sum())

        segment_summary["customer_share_pct"] = np.round(
            (segment_summary["customer_count"] / max(total_customers, 1)) * 100.0, 1
        )
        segment_summary["clv_share_pct"] = np.round(
            (segment_summary["total_projected_clv"] / max(total_clv, 1e-4)) * 100.0, 1
        )

        # Identify top valuable segment and top at-risk segment
        top_segment = segment_summary.sort_values(by="total_projected_clv", ascending=False).iloc[0]
        top_segment_name = str(top_segment[segment_col])
        top_segment_clv = float(top_segment["total_projected_clv"])

        evidence_top = StructuredEvidence(
            evidence_id="ev_segment_top_clv",
            claim_id="claim_top_clv_segment",
            dataset_version_id=self.dataset_id,
            query_hash="agg_segment_clv_v1",
            execution_id="exec_insights_01",
            source_type="RECSYS_CORE",
            table_or_view_name="v_customers",
            column_name="forward_projected_gross_clv",
            result_row_or_key=(top_segment_name,),
            raw_value=round(top_segment_clv, 2),
            unit="BRL",
            calculation_provenance="SUM(forward_projected_gross_clv) GROUP BY rfm_segment",
            provenance_type=ProvenanceType.SIMULATION_ASSUMPTION,
            evidence_to_claim_linkage=f"Segment '{top_segment_name}' represents the largest forward CLV share (R${top_segment_clv:,.2f}).",
            created_at=datetime.utcnow(),
        )

        return {
            "segment_summary": segment_summary.to_dict(orient="records"),
            "total_portfolio_customers": total_customers,
            "total_projected_clv": round(total_clv, 2),
            "top_valuable_segment": top_segment_name,
            "evidence": [evidence_top],
            "correlation_disclaimer": (
                "STATISTICAL NOTE: Segment associations reflect historical purchase clusters. "
                "Higher CLV projections indicate observed purchasing patterns, not guaranteed future revenue."
            ),
        }

    def generate_churn_drivers_insight(
        self,
        global_importance_df: pd.DataFrame,
        top_n: int = 3,
    ) -> Dict[str, Any]:
        """Formulates key statistical churn predictors with non-causal language.

        Args:
            global_importance_df: Output from ModelExplainer.compute_global_importance().
            top_n: Number of top features to report.

        Returns:
            Dict containing top drivers and strict distinction between prediction and causation.
        """
        top_drivers = global_importance_df.head(top_n).to_dict(orient="records")
        driver_names = [d["feature"] for d in top_drivers]

        evidence = StructuredEvidence(
            evidence_id="ev_churn_top_drivers",
            claim_id="claim_top_churn_drivers",
            dataset_version_id=self.dataset_id,
            query_hash="churn_feature_attribution_v1",
            execution_id="exec_insights_02",
            source_type="RECSYS_CORE",
            table_or_view_name="v_customers",
            column_name="churn_risk",
            result_row_or_key=("top_churn_drivers",),
            raw_value=", ".join(driver_names),
            unit="feature_ranking",
            calculation_provenance="ModelExplainer.compute_global_importance()",
            provenance_type=ProvenanceType.SIMULATION_ASSUMPTION,
            evidence_to_claim_linkage=f"Top {top_n} statistical predictive features for customer churn are: {', '.join(driver_names)}.",
            created_at=datetime.utcnow(),
        )

        return {
            "top_churn_predictive_features": top_drivers,
            "evidence": evidence,
            "causality_statement": (
                "CORRELATION vs CAUSATION: Features identified as top risk predictors (e.g., recency, low review ratio) "
                "correlate strongly with churn probability. Intervening on these features (e.g. sending promotional emails) "
                "should be tested via randomized A/B trials to confirm causal treatment impact before enterprise budget allocation."
            ),
        }

    def generate_retention_playbook(
        self,
        rfm_segment: str,
        predicted_churn_prob: float,
        clv_tier: str = "Bronze",
    ) -> Dict[str, str]:
        """Generates evidence-backed retention action playbooks for specific customer tiers."""
        is_high_value = clv_tier in ["Gold", "Platinum"] or rfm_segment in ["Champions", "Loyal Customers", "Recent High Spenders"]
        is_high_churn = predicted_churn_prob >= 0.70

        if is_high_value and is_high_churn:
            priority = "URGENT_INTERVENTION"
            action = (
                "High-Value VIP at high risk of churn. Trigger proactive concierge outreach: "
                "priority customer support follow-up and personalized high-margin category voucher."
            )
        elif is_high_value and not is_high_churn:
            priority = "LOYALTY_NURTURING"
            action = (
                "High-value engaged customer. Do not discount unnecessarily; enroll in exclusive VIP tier "
                "and recommend complementary cross-category items based on past purchases."
            )
        elif not is_high_value and is_high_churn:
            priority = "AUTOMATED_RE_ENGAGEMENT"
            action = (
                "Low-value inactive customer. Send low-cost automated win-back notification or email sequence; "
                "avoid manual intervention to protect operating margins."
            )
        else:
            priority = "NURTURE_TO_REPEAT"
            action = (
                "Active one-time customer with moderate churn risk. Offer targeted next-order shipping incentive "
                "to encourage repeat purchase conversion within 60 days."
            )

        return {
            "rfm_segment": rfm_segment,
            "clv_tier": clv_tier,
            "predicted_churn_probability": round(predicted_churn_prob, 3),
            "retention_priority": priority,
            "recommended_action": action,
        }
