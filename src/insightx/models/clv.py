"""Customer Lifetime Value (CLV) Engine for InsightX-AI 3.0.

Provides:
1. Realized historical customer revenue (GMV) baseline.
2. Parameterized net profit contribution (distinguishing revenue from profit via auditable business assumptions).
3. BG/NBD and Gamma-Gamma statistical suitability auditing for sparse e-commerce data.
4. Robust empirical / heuristic forward CLV projection handling one-time buyers (96.9% of Olist) and repeat buyers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import yaml

from insightx.utils.logger import get_logger

logger = get_logger(__name__)

# Default fallback assumptions (mirrors configs/business_assumptions.yaml)
DEFAULT_CLV_ASSUMPTIONS = {
    "marketplace_take_rate": 0.15,
    "operating_margin": 0.20,
    "annual_discount_rate": 0.10,
    "assumed_annual_retention_rate": 0.05,
    "empirical_repeat_rate": 0.031,
}


class CLVEngine:
    """Computes historical and forward-projected Customer Lifetime Value with auditable assumptions."""

    def __init__(
        self,
        assumptions_path: Optional[Union[str, Path]] = None,
        custom_assumptions: Optional[Dict[str, float]] = None,
    ):
        """Args:
            assumptions_path: Path to business_assumptions.yaml.
            custom_assumptions: Dictionary of overrides for financial parameters.
        """
        self.assumptions = DEFAULT_CLV_ASSUMPTIONS.copy()

        if assumptions_path and Path(assumptions_path).exists():
            try:
                with open(assumptions_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                    clv_cfg = cfg.get("clv_assumptions", {})
                    self.assumptions.update(clv_cfg)
            except Exception as e:
                logger.warning(f"Could not load assumptions from {assumptions_path}: {e}")

        if custom_assumptions:
            self.assumptions.update(custom_assumptions)

    def calculate_historical_value(
        self,
        customer_features_df: pd.DataFrame,
        monetary_col: str = "total_monetary",
    ) -> pd.DataFrame:
        """Calculates realized gross revenue and parameterized net profit per customer to date.

        Mathematical Definitions:
            Realized Gross Revenue = Observed Total Spend (GMV)
            Realized Net Profit = GMV * Take Rate * Operating Margin

        Args:
            customer_features_df: Customer-level DataFrame containing monetary_col.
            monetary_col: Column recording historical spend.

        Returns:
            pd.DataFrame: Augmented DataFrame with realized_gross_revenue,
                          realized_marketplace_take, and realized_net_profit.
        """
        df = customer_features_df.copy()
        take_rate = self.assumptions["marketplace_take_rate"]
        margin = self.assumptions["operating_margin"]

        spend = np.maximum(df[monetary_col].fillna(0.0).values, 0.0)

        df["realized_gross_revenue"] = np.round(spend, 2)
        df["realized_marketplace_take"] = np.round(spend * take_rate, 2)
        df["realized_net_profit"] = np.round(spend * take_rate * margin, 2)

        logger.info(
            f"Computed historical value for {len(df)} customers. "
            f"Total Gross GMV = R${df['realized_gross_revenue'].sum():,.2f}, "
            f"Total Estimated Net Profit = R${df['realized_net_profit'].sum():,.2f}."
        )
        return df

    def evaluate_probabilistic_model_suitability(
        self,
        customer_features_df: pd.DataFrame,
        frequency_col: str = "frequency",
        monetary_col: str = "total_monetary",
    ) -> Dict[str, Any]:
        """Audits whether BG/NBD and Gamma-Gamma model assumptions hold on the customer cohort.

        Checks:
        1. BG/NBD repeat purchase density: Freq >= 2 ratio. If < 10%, BG/NBD degenerates.
        2. Gamma-Gamma spend-frequency independence: Pearson correlation |r| < 0.15.
        3. Sufficient sample size of repeat purchasers.

        Returns:
            Dict containing suitability metrics, statistical test results, and recommendations.
        """
        df = customer_features_df.copy()
        freq = df[frequency_col].fillna(0).values
        mon = df[monetary_col].fillna(0).values

        total_customers = len(df)
        repeat_mask = freq >= 2
        repeat_count = int(repeat_mask.sum())
        repeat_pct = round((repeat_count / max(total_customers, 1)) * 100.0, 2)

        # 1. BG/NBD Suitability
        # BG/NBD requires sufficient repeat transactions to estimate latent transaction rates
        bgnbd_suitable = bool(repeat_pct >= 10.0 and repeat_count >= 500)
        bgnbd_verdict = (
            "SUITABLE"
            if bgnbd_suitable
            else f"UNSUITABLE for full population: Only {repeat_pct}% repeat customers ({repeat_count}/{total_customers}). "
            "BG/NBD parameter MLE estimation degenerates on 90%+ one-time buyer datasets."
        )

        # 2. Gamma-Gamma Suitability (evaluated strictly on repeat customers where freq >= 2)
        if repeat_count >= 10:
            avg_spend_repeat = mon[repeat_mask] / freq[repeat_mask]
            corr, p_value = pearsonr(freq[repeat_mask], avg_spend_repeat)
            corr = float(corr) if not np.isnan(corr) else 0.0
            p_val = float(p_value) if not np.isnan(p_value) else 1.0

            # Gamma-Gamma assumes independence between transaction frequency and monetary value
            independence_satisfied = bool(abs(corr) <= 0.15)
            gamma_suitable = bool(independence_satisfied and repeat_count >= 100)
            gamma_verdict = (
                f"SUITABLE for repeat cohort: Pearson r = {corr:.3f} (p={p_val:.3e}). Independence assumption holds."
                if independence_satisfied
                else f"VIOLATED: Pearson r = {corr:.3f} indicates spend depends significantly on frequency."
            )
        else:
            corr = 0.0
            p_val = 1.0
            gamma_suitable = False
            gamma_verdict = f"INSUFFICIENT DATA: Only {repeat_count} repeat customers found. Minimum 100 required."

        return {
            "total_customers": total_customers,
            "repeat_customers_count": repeat_count,
            "repeat_customers_pct": repeat_pct,
            "bgnbd_suitable": bgnbd_suitable,
            "bgnbd_verdict": bgnbd_verdict,
            "gamma_gamma_suitable": gamma_suitable,
            "gamma_gamma_verdict": gamma_verdict,
            "frequency_monetary_correlation": round(corr, 4),
            "correlation_p_value": round(p_val, 4),
            "recommended_strategy": (
                "Dual-Track Approach: Use Heuristic/Cohort-Based CLV for one-time buyers (96.9% of users) "
                "and Probabilistic BG/NBD + Gamma-Gamma strictly for the repeat-buyer subset."
            ),
        }

    def estimate_forward_clv(
        self,
        customer_features_df: pd.DataFrame,
        horizon_days: int = 365,
        recency_col: str = "recency_days",
        frequency_col: str = "frequency",
        monetary_col: str = "total_monetary",
    ) -> pd.DataFrame:
        """Estimates expected forward Gross and Net CLV over horizon_days.

        Handles sparse one-time buyers and repeat buyers mathematically:
        - Baseline Annualized AOV projection scaled by empirical retention probability and recency decay.
        - Recency Decay: Exp(-Recency / 365).
        - One-time buyers: P(repeat) anchored to empirical baseline (~3.1%).
        - Repeat buyers: P(repeat) scaled by historical frequency velocity.

        Args:
            customer_features_df: Customer feature matrix.
            horizon_days: Forward projection window (default: 365 days / 1 year).
            recency_col: Recency column name.
            frequency_col: Frequency column name.
            monetary_col: Monetary spend column name.

        Returns:
            pd.DataFrame: Augmented with expected_forward_orders,
                          forward_projected_gross_clv, and forward_projected_net_clv.
        """
        df = customer_features_df.copy()

        take_rate = self.assumptions["marketplace_take_rate"]
        margin = self.assumptions["operating_margin"]
        discount_rate = self.assumptions["annual_discount_rate"]
        emp_repeat_rate = self.assumptions.get("empirical_repeat_rate", 0.031)

        rec = df.get(recency_col, pd.Series(100.0, index=df.index)).fillna(100.0).values
        freq = df.get(frequency_col, pd.Series(1, index=df.index)).fillna(1).values
        spend = df.get(monetary_col, pd.Series(100.0, index=df.index)).fillna(100.0).values

        aov = spend / np.maximum(freq, 1)

        # 1. Expected probability of purchasing in forward window
        # Repeat buyers have higher retention; recency exponentially discounts probability
        recency_decay = np.exp(-np.maximum(rec, 0.0) / 365.0)

        # P(active) is higher for multi-order customers, conservative for one-timers
        is_repeat = freq >= 2
        base_p_active = np.where(is_repeat, 0.25 + 0.15 * np.minimum(freq - 1, 4), emp_repeat_rate)
        expected_p_active = np.clip(base_p_active * recency_decay, 0.001, 0.85)

        # 2. Expected forward orders in horizon
        horizon_years = horizon_days / 365.0
        expected_orders_per_year = np.where(is_repeat, np.maximum(freq * 0.8, 1.0), 1.0)
        expected_forward_orders = expected_p_active * expected_orders_per_year * horizon_years

        # 3. Discounted expected forward revenue & profit
        discount_factor = 1.0 / (1.0 + discount_rate * horizon_years)

        forward_gross_clv = expected_forward_orders * aov * discount_factor
        forward_net_clv = forward_gross_clv * take_rate * margin

        df["expected_forward_orders"] = np.round(expected_forward_orders, 3)
        df["forward_projected_gross_clv"] = np.round(forward_gross_clv, 2)
        df["forward_projected_net_clv"] = np.round(forward_net_clv, 2)

        # CLV tier classification (safe across any sample size)
        if len(df) >= 4:
            pct = df["forward_projected_gross_clv"].rank(pct=True, method="first")
            tier_idx = np.ceil(pct * 4).astype(int).clip(1, 4)
            tier_map = {1: "Bronze", 2: "Silver", 3: "Gold", 4: "Platinum"}
            df["clv_tier"] = tier_idx.map(tier_map)
        else:
            df["clv_tier"] = "Bronze"

        logger.info(
            f"Estimated forward {horizon_days}-day CLV across {len(df)} customers. "
            f"Total Projected Gross CLV = R${df['forward_projected_gross_clv'].sum():,.2f}, "
            f"Total Projected Net CLV = R${df['forward_projected_net_clv'].sum():,.2f}."
        )
        return df
