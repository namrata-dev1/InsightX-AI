"""Natural Language to SQL Decision Agent for InsightX-AI.

Translates business questions into hardened SQL queries over verified Olist views,
enforcing SQLGuard AST validation and DuckDB process isolation. Synthesizes
tabular results into plain-English executive briefings with StructuredEvidence.
"""

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from insightx.agent.sql_guard import (
    SQLGuard,
    SQLSecurityError,
    DisallowedOperationError,
    UnapprovedEntityError,
    QueryTimeoutError,
)
from insightx.provenance.evidence import StructuredEvidence
from insightx.provenance.types import ProvenanceType
from insightx.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SQLDecisionResult:
    """Structured result of a natural language SQL decision query."""

    natural_query: str
    generated_sql: str
    is_valid: bool
    error_message: Optional[str] = None
    result_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    row_count: int = 0
    business_summary: str = ""
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    evidence: Optional[StructuredEvidence] = None
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to dictionary for API/Dashboard consumption."""
        return {
            "natural_query": self.natural_query,
            "generated_sql": self.generated_sql,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "row_count": self.row_count,
            "business_summary": self.business_summary,
            "key_metrics": self.key_metrics,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "evidence_id": self.evidence.evidence_id if self.evidence else None,
        }


class SQLDecisionAgent:
    """Enterprise decision agent converting natural business queries to secure SQL.
    
    Guarantees:
    - 100% read-only execution.
    - Deep AST allowlist enforcement via SQLGuard.
    - Process-isolated DuckDB sandbox with memory ceilings and execution watchdogs.
    - Plain-English executive synthesis with provenance tracking.
    """

    def __init__(
        self,
        sql_guard: Optional[SQLGuard] = None,
        dataset_id: str = "olist-v1.0",
    ):
        self.sql_guard = sql_guard or SQLGuard()
        self.dataset_id = dataset_id
        logger.info(f"SQLDecisionAgent initialized with dataset_id='{self.dataset_id}'")

    def get_supported_intents(self) -> List[Dict[str, str]]:
        """Returns list of pre-configured business analytical query templates."""
        return [
            {
                "intent": "monthly_revenue",
                "description": "Monthly revenue trend, order count, and average order value",
                "sample_query": "What is our monthly revenue trend?",
            },
            {
                "intent": "top_categories",
                "description": "Top product categories by total sales volume or items sold",
                "sample_query": "Which are the top 10 categories by sales?",
            },
            {
                "intent": "delivery_performance",
                "description": "Average delivery duration and schedule variance by state",
                "sample_query": "Show delivery performance across customer states",
            },
            {
                "intent": "payment_breakdown",
                "description": "Payment method distribution, total value, and installment patterns",
                "sample_query": "How do customers pay for orders?",
            },
            {
                "intent": "customer_geography",
                "description": "Customer geographic concentration by state and city",
                "sample_query": "Which states have the most customers?",
            },
            {
                "intent": "review_scores",
                "description": "Customer satisfaction score distribution and proportions",
                "sample_query": "What is the breakdown of customer review ratings?",
            },
            {
                "intent": "order_status",
                "description": "Distribution of order statuses across platform history",
                "sample_query": "What is the fulfillment status distribution of all orders?",
            },
            {
                "intent": "seller_performance",
                "description": "Top seller volume, items sold, and geographic origin",
                "sample_query": "Which sellers have the highest sales volume?",
            },
        ]

    def translate_to_sql(self, natural_query: str) -> Tuple[str, str]:
        """Translates natural language prompt into compliant SQL over allowlisted views.
        
        Args:
            natural_query: Business question in natural language.
            
        Returns:
            Tuple of (detected_intent, generated_sql_query).
        """
        q = natural_query.lower().strip()

        # Check if user passed raw SQL directly (SELECT, DDL, DML, or queries with semicolons)
        is_raw_sql = bool(re.match(r"^(select|insert|update|delete|drop|create|alter|truncate|with|pragma|describe|exec)\b", q))
        if not is_raw_sql and (q.startswith("show tables") or q.startswith("show columns")):
            is_raw_sql = True
        if not is_raw_sql and ";" in q:
            is_raw_sql = True

        if is_raw_sql:
            return "raw_sql", natural_query

        # Extract potential limit
        limit_match = re.search(r"\btop\s+(\d+)\b|\blimit\s+(\d+)\b", q)
        limit_val = 10
        if limit_match:
            limit_val = int(limit_match.group(1) or limit_match.group(2))
            limit_val = min(max(1, limit_val), 500)

        # 1. Monthly Revenue / Sales Over Time
        if any(w in q for w in ["monthly revenue", "sales over time", "revenue trend", "monthly sales", "sales trend"]):
            # Check for year filter
            year_match = re.search(r"\b(2016|2017|2018)\b", q)
            where_clause = "WHERE o.order_status = 'delivered'"
            if year_match:
                year = year_match.group(1)
                where_clause += f" AND strftime('%Y', CAST(o.order_purchase_timestamp AS TIMESTAMP)) = '{year}'"

            sql = f"""
                SELECT 
                    strftime('%Y-%m', CAST(o.order_purchase_timestamp AS TIMESTAMP)) AS order_month,
                    COUNT(DISTINCT o.order_id) AS total_orders,
                    ROUND(SUM(p.payment_value), 2) AS total_revenue,
                    ROUND(AVG(p.payment_value), 2) AS avg_order_value
                FROM v_orders o
                JOIN v_order_payments p ON o.order_id = p.order_id
                {where_clause}
                GROUP BY 1
                ORDER BY 1
            """
            return "monthly_revenue", sql.strip()

        # 2. Top Product Categories
        if any(w in q for w in ["top categor", "best selling categor", "product categor", "category sales"]):
            sql = f"""
                SELECT 
                    COALESCE(p.product_category_name, 'Other') AS category,
                    COUNT(oi.order_item_id) AS items_sold,
                    ROUND(SUM(oi.price), 2) AS total_sales,
                    ROUND(AVG(oi.price), 2) AS avg_item_price
                FROM v_order_items oi
                JOIN v_products p ON oi.product_id = p.product_id
                GROUP BY 1
                ORDER BY total_sales DESC
                LIMIT {limit_val}
            """
            return "top_categories", sql.strip()

        # 3. Delivery Performance
        if any(w in q for w in ["delivery", "shipping duration", "delay", "on time", "transit time"]):
            state_match = re.search(r"\bin\s+([a-z]{2})\b|\bstate\s+([a-z]{2})\b", q)
            where_extra = ""
            if state_match:
                st_code = (state_match.group(1) or state_match.group(2)).upper()
                where_extra = f"AND c.customer_state = '{st_code}'"

            sql = f"""
                SELECT 
                    c.customer_state,
                    COUNT(o.order_id) AS delivered_orders,
                    ROUND(AVG(date_diff('day', CAST(o.order_purchase_timestamp AS TIMESTAMP), CAST(o.order_delivered_customer_date AS TIMESTAMP))), 1) AS avg_delivery_days,
                    ROUND(AVG(date_diff('day', CAST(o.order_delivered_customer_date AS TIMESTAMP), CAST(o.order_estimated_delivery_date AS TIMESTAMP))), 1) AS avg_days_ahead_of_estimate
                FROM v_orders o
                JOIN v_customers c ON o.customer_id = c.customer_id
                WHERE o.order_status = 'delivered'
                  AND o.order_delivered_customer_date IS NOT NULL
                  {where_extra}
                GROUP BY c.customer_state
                ORDER BY delivered_orders DESC
                LIMIT {limit_val}
            """
            return "delivery_performance", sql.strip()

        # 4. Payment Methods Breakdown
        if any(w in q for w in ["payment", "pay", "credit card", "boleto", "voucher"]):
            sql = """
                SELECT 
                    payment_type,
                    COUNT(order_id) AS transaction_count,
                    ROUND(SUM(payment_value), 2) AS total_payment_value,
                    ROUND(AVG(payment_value), 2) AS avg_payment_value,
                    ROUND(AVG(payment_installments), 1) AS avg_installments
                FROM v_order_payments
                GROUP BY payment_type
                ORDER BY total_payment_value DESC
            """
            return "payment_breakdown", sql.strip()

        # 5. Customer Geography
        if any(w in q for w in ["customer geography", "customers by state", "customer state", "top states", "customer location"]):
            sql = f"""
                SELECT 
                    customer_state,
                    COUNT(customer_id) AS total_customers,
                    COUNT(DISTINCT customer_unique_id) AS unique_customers
                FROM v_customers
                GROUP BY customer_state
                ORDER BY unique_customers DESC
                LIMIT {limit_val}
            """
            return "customer_geography", sql.strip()

        # 6. Customer Review Ratings
        if any(w in q for w in ["review", "rating", "satisfaction", "score", "feedback"]):
            sql = """
                SELECT 
                    review_score,
                    COUNT(review_id) AS review_count,
                    ROUND(COUNT(review_id) * 100.0 / SUM(COUNT(review_id)) OVER (), 2) AS percentage
                FROM v_order_reviews
                GROUP BY review_score
                ORDER BY review_score DESC
            """
            return "review_scores", sql.strip()

        # 7. Order Status Breakdown
        if any(w in q for w in ["order status", "status distribution", "cancellation", "fulfillment status"]):
            sql = """
                SELECT 
                    order_status,
                    COUNT(order_id) AS order_count,
                    ROUND(COUNT(order_id) * 100.0 / SUM(COUNT(order_id)) OVER (), 2) AS percentage
                FROM v_orders
                GROUP BY order_status
                ORDER BY order_count DESC
            """
            return "order_status", sql.strip()

        # 8. Seller Performance
        if any(w in q for w in ["seller", "merchant", "vendor"]):
            sql = f"""
                SELECT 
                    s.seller_id,
                    s.seller_state,
                    COUNT(oi.order_item_id) AS items_sold,
                    ROUND(SUM(oi.price), 2) AS total_sales,
                    ROUND(AVG(oi.price), 2) AS avg_item_price
                FROM v_order_items oi
                JOIN v_sellers s ON oi.seller_id = s.seller_id
                GROUP BY s.seller_id, s.seller_state
                ORDER BY total_sales DESC
                LIMIT {limit_val}
            """
            return "seller_performance", sql.strip()

        # Default fallback: General summary of orders & revenue
        sql = """
            SELECT 
                COUNT(DISTINCT o.order_id) AS total_orders,
                ROUND(SUM(p.payment_value), 2) AS total_revenue,
                ROUND(AVG(p.payment_value), 2) AS avg_order_value
            FROM v_orders o
            JOIN v_order_payments p ON o.order_id = p.order_id
            WHERE o.order_status = 'delivered'
        """
        return "general_summary", sql.strip()

    def explain_results(
        self,
        intent: str,
        df: pd.DataFrame,
        natural_query: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Synthesizes query DataFrame into an executive plain-English summary."""
        if df.empty:
            return "No matching records found for this query within the dataset.", {}

        key_metrics: Dict[str, Any] = {}

        if intent == "monthly_revenue":
            total_rev = float(df["total_revenue"].sum()) if "total_revenue" in df.columns else 0.0
            total_ord = int(df["total_orders"].sum()) if "total_orders" in df.columns else 0
            avg_aov = float(df["avg_order_value"].mean()) if "avg_order_value" in df.columns else 0.0
            peak_month_row = df.loc[df["total_revenue"].idxmax()] if "total_revenue" in df.columns else None
            peak_month = str(peak_month_row["order_month"]) if peak_month_row is not None else "N/A"
            peak_val = float(peak_month_row["total_revenue"]) if peak_month_row is not None else 0.0

            key_metrics = {
                "total_revenue": round(total_rev, 2),
                "total_orders": total_ord,
                "average_aov": round(avg_aov, 2),
                "peak_month": peak_month,
                "peak_month_revenue": round(peak_val, 2),
                "months_covered": len(df),
            }
            summary = (
                f"Across {len(df)} observed months, total realized revenue reached R${total_rev:,.2f} "
                f"from {total_ord:,} delivered orders (Average Order Value: R${avg_aov:,.2f}). "
                f"The highest sales volume occurred in {peak_month} generating R${peak_val:,.2f}."
            )
            return summary, key_metrics

        if intent == "top_categories":
            top_cat = str(df.iloc[0]["category"])
            top_sales = float(df.iloc[0]["total_sales"])
            total_top_sales = float(df["total_sales"].sum())
            key_metrics = {
                "top_category": top_cat,
                "top_category_sales": round(top_sales, 2),
                "combined_sales_top_categories": round(total_top_sales, 2),
                "categories_analyzed": len(df),
            }
            summary = (
                f"Top product category by gross sales is '{top_cat}' generating R${top_sales:,.2f}. "
                f"The top {len(df)} categories combined account for R${total_top_sales:,.2f} in merchandise volume."
            )
            return summary, key_metrics

        if intent == "delivery_performance":
            avg_duration = float(df["avg_delivery_days"].mean()) if "avg_delivery_days" in df.columns else 0.0
            avg_ahead = float(df["avg_days_ahead_of_estimate"].mean()) if "avg_days_ahead_of_estimate" in df.columns else 0.0
            top_state = str(df.iloc[0]["customer_state"]) if "customer_state" in df.columns else "N/A"
            key_metrics = {
                "overall_avg_delivery_days": round(avg_duration, 1),
                "overall_avg_ahead_of_estimate": round(avg_ahead, 1),
                "highest_volume_state": top_state,
                "states_reported": len(df),
            }
            summary = (
                f"Average delivery duration across reported states is {avg_duration:.1f} days, "
                f"arriving an average of {avg_ahead:.1f} days ahead of the initial delivery estimate. "
                f"State '{top_state}' accounts for the highest volume of delivered orders."
            )
            return summary, key_metrics

        if intent == "payment_breakdown":
            primary_method = str(df.iloc[0]["payment_type"])
            primary_val = float(df.iloc[0]["total_payment_value"])
            total_paid = float(df["total_payment_value"].sum())
            share = (primary_val / max(total_paid, 1e-4)) * 100.0
            key_metrics = {
                "primary_payment_method": primary_method,
                "primary_method_value": round(primary_val, 2),
                "primary_method_share_pct": round(share, 1),
                "total_payment_volume": round(total_paid, 2),
            }
            summary = (
                f"Primary customer payment method is '{primary_method}' accounting for "
                f"{share:.1f}% (R${primary_val:,.2f}) of total transaction value across {len(df)} payment types."
            )
            return summary, key_metrics

        if intent == "customer_geography":
            top_state = str(df.iloc[0]["customer_state"])
            top_users = int(df.iloc[0]["unique_customers"])
            total_users = int(df["unique_customers"].sum())
            share = (top_users / max(total_users, 1)) * 100.0
            key_metrics = {
                "top_state": top_state,
                "top_state_unique_customers": top_users,
                "top_state_share_pct": round(share, 1),
                "total_unique_customers_listed": total_users,
            }
            summary = (
                f"Customer geographic concentration is led by '{top_state}', hosting {top_users:,} "
                f"unique customers ({share:.1f}% of the top cohort)."
            )
            return summary, key_metrics

        if intent == "review_scores":
            scores_5 = df[df["review_score"] == 5]["percentage"].values
            pct_5 = float(scores_5[0]) if len(scores_5) > 0 else 0.0
            scores_1 = df[df["review_score"] == 1]["percentage"].values
            pct_1 = float(scores_1[0]) if len(scores_1) > 0 else 0.0
            key_metrics = {
                "five_star_percentage": round(pct_5, 1),
                "one_star_percentage": round(pct_1, 1),
            }
            summary = (
                f"Customer sentiment demonstrates positive concentration: {pct_5:.1f}% of reviews award 5 stars, "
                f"while negative feedback (1 star) represents {pct_1:.1f}% of total ratings."
            )
            return summary, key_metrics

        if intent == "order_status":
            delivered_share = 0.0
            delivered_row = df[df["order_status"] == "delivered"]
            if not delivered_row.empty and "percentage" in delivered_row.columns:
                delivered_share = float(delivered_row.iloc[0]["percentage"])
            key_metrics = {"delivered_order_percentage": round(delivered_share, 1)}
            summary = (
                f"Fulfillment breakdown reveals {delivered_share:.1f}% of all logged transactions successfully reached 'delivered' status."
            )
            return summary, key_metrics

        # Generic summary
        num_cols = df.select_dtypes(include=[np.number]).columns
        key_metrics = {"rows_returned": len(df), "columns_returned": len(df.columns)}
        if len(num_cols) > 0:
            first_num = num_cols[0]
            val_sum = float(df[first_num].sum())
            key_metrics[f"{first_num}_sum"] = round(val_sum, 2)
            summary = f"Query returned {len(df)} rows across {len(df.columns)} dimensions. Total {first_num}: {val_sum:,.2f}."
        else:
            summary = f"Query executed successfully, returning {len(df)} records across {len(df.columns)} columns."

        return summary, key_metrics

    def create_evidence(
        self,
        df: pd.DataFrame,
        sql_query: str,
        claim: str,
        table_name: str = "v_orders",
    ) -> StructuredEvidence:
        """Constructs an immutable StructuredEvidence object tracking query provenance."""
        query_hash = "sha256:" + hashlib.sha256(sql_query.encode("utf-8")).hexdigest()[:16]
        exec_id = f"exec_sql_agent_{int(time.time())}"

        raw_val: Any = len(df)
        unit = "rows"
        if not df.empty:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                raw_val = float(df[num_cols[0]].iloc[0])
                unit = "BRL" if "revenue" in num_cols[0] or "price" in num_cols[0] or "value" in num_cols[0] else "count"

        return StructuredEvidence(
            evidence_id=f"ev_{int(time.time() * 1000) % 10000000}",
            claim_id=f"claim_{int(time.time()) % 100000}",
            dataset_version_id=self.dataset_id,
            query_hash=query_hash,
            execution_id=exec_id,
            source_type="SQL_GUARD_SANDBOX",
            table_or_view_name=table_name,
            column_name="analytical_aggregate",
            result_row_or_key=("sql_query_result",),
            raw_value=raw_val,
            unit=unit,
            calculation_provenance=f"DuckDB Sandbox Execution: {sql_query.strip()[:120]}...",
            provenance_type=ProvenanceType.OBSERVED,
            evidence_to_claim_linkage=claim,
            created_at=datetime.utcnow(),
        )

    def ask(
        self,
        natural_query: str,
        tables_data: Dict[str, pd.DataFrame],
        timeout_seconds: Optional[float] = None,
    ) -> SQLDecisionResult:
        """Translates natural language to SQL, executes under SQLGuard, and returns business insights.
        
        Args:
            natural_query: Plain-English question from decision-maker or raw SQL.
            tables_data: Serializable dictionary of view DataFrames.
            timeout_seconds: Optional execution timeout override.
            
        Returns:
            SQLDecisionResult with validation status, DataFrame, plain-English summary, and evidence.
        """
        start_time = time.time()
        logger.info(f"SQLDecisionAgent processing query: '{natural_query}'")

        try:
            intent, generated_sql = self.translate_to_sql(natural_query)
        except Exception as exc:
            elapsed = time.time() - start_time
            return SQLDecisionResult(
                natural_query=natural_query,
                generated_sql="",
                is_valid=False,
                error_message=f"Query translation error: {exc}",
                execution_time_seconds=elapsed,
            )

        try:
            # Enforce SQLGuard deep AST validation and process isolation
            result_df = self.sql_guard.execute_safely(
                generated_sql,
                tables_data=tables_data,
                timeout_seconds=timeout_seconds,
            )
            elapsed = time.time() - start_time

            # Generate plain-English executive explanation
            business_summary, key_metrics = self.explain_results(
                intent=intent,
                df=result_df,
                natural_query=natural_query,
            )

            # Generate StructuredEvidence provenance link
            evidence = self.create_evidence(
                df=result_df,
                sql_query=generated_sql,
                claim=business_summary,
            )

            return SQLDecisionResult(
                natural_query=natural_query,
                generated_sql=generated_sql,
                is_valid=True,
                error_message=None,
                result_df=result_df,
                row_count=len(result_df),
                business_summary=business_summary,
                key_metrics=key_metrics,
                evidence=evidence,
                execution_time_seconds=elapsed,
            )

        except SQLSecurityError as sec_err:
            elapsed = time.time() - start_time
            logger.warning(f"SQLGuard rejected query: {sec_err}")
            return SQLDecisionResult(
                natural_query=natural_query,
                generated_sql=generated_sql,
                is_valid=False,
                error_message=f"Security Violation: {sec_err}",
                execution_time_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(f"Execution error: {exc}")
            return SQLDecisionResult(
                natural_query=natural_query,
                generated_sql=generated_sql,
                is_valid=False,
                error_message=f"Execution error: {exc}",
                execution_time_seconds=elapsed,
            )
