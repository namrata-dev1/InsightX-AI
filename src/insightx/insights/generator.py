"""Automated Business Insight Generator.

Translates RFM segments, predictive scores, and SQL KPI summaries
into plain-English strategic recommendations for stakeholders.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from insightx.insights.behaviour import CustomerBehaviourInsights
from insightx.utils.logger import get_logger

logger = get_logger(__name__)


class InsightGenerator:
    """Generates rule-based or LLM-assisted executive summaries and recommendations."""

    def __init__(self, use_llm: bool = False, dataset_id: str = "olist-v1.0"):
        self.use_llm = use_llm
        self.dataset_id = dataset_id
        self.behaviour_engine = CustomerBehaviourInsights(dataset_id=dataset_id)

    def generate_segment_recommendations(
        self, cluster_profiles: Union[Dict[str, Dict[str, float]], pd.DataFrame]
    ) -> List[Dict[str, str]]:
        """Produces marketing and retention action points tailored to each customer segment.

        Args:
            cluster_profiles: Mean/median RFM values per cluster or DataFrame.

        Returns:
            List of insight dictionaries containing segment name, characteristic, and strategic action.
        """
        logger.info("Generating segment-specific business recommendations...")
        recommendations = []

        if isinstance(cluster_profiles, pd.DataFrame):
            df = cluster_profiles
            for idx, row in df.iterrows():
                name = str(row.get("persona_name", row.get("rfm_segment", f"Segment_{idx}")))
                clv = float(row.get("total_monetary", row.get("monetary", 100.0)))
                rec = self.behaviour_engine.generate_retention_playbook(
                    rfm_segment=name,
                    predicted_churn_prob=0.5,
                    clv_tier="Gold" if clv > 300 else "Bronze",
                )
                recommendations.append(rec)
        elif isinstance(cluster_profiles, dict):
            for seg_name, metrics in cluster_profiles.items():
                clv = float(metrics.get("monetary", 100.0))
                rec = self.behaviour_engine.generate_retention_playbook(
                    rfm_segment=seg_name,
                    predicted_churn_prob=0.5,
                    clv_tier="Gold" if clv > 300 else "Bronze",
                )
                recommendations.append(rec)

        return recommendations

    def generate_executive_brief(self, kpi_summary: Dict[str, Any]) -> str:
        """Synthesizes high-level revenue, retention, and growth metrics into an executive summary."""
        logger.info("Synthesizing executive brief...")
        total_rev = kpi_summary.get("total_revenue", 0.0)
        total_orders = kpi_summary.get("total_orders", 0)
        repeat_rate = kpi_summary.get("repeat_rate", 0.031)

        brief = (
            f"InsightX-AI Portfolio Executive Brief:\n"
            f"- Total Realized Revenue: R${total_rev:,.2f}\n"
            f"- Total Delivered Orders: {total_orders:,}\n"
            f"- Empirical Repeat Buyer Rate: {repeat_rate*100:.2f}%\n"
            f"- Core Strategy: Focus retention investment on high-value repeat buyers; "
            f"maintain low-cost automated re-engagement for one-time buyer base to safeguard unit economics."
        )
        return brief
