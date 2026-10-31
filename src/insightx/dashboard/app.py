"""InsightX AI - Streamlit Analytics & ML Dashboard.

Interactive multi-page dashboard displaying:
1. Executive KPI Summary & SQL Analytics
2. RFM & K-Means Customer Segmentation
3. Repeat Purchase Predictions & Feature Impact (SHAP)
4. Revenue Forecasting & Trend Projections
5. AI-Generated Strategic Recommendations
"""

import streamlit as st


def main():
    st.set_page_config(
        page_title="InsightX AI | E-commerce Intelligence",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 InsightX AI - E-commerce Customer & Business Intelligence")
    st.caption("Powered by Olist Brazilian E-Commerce Ecosystem")

    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Select Module",
        [
            "Overview & SQL KPIs",
            "Customer Segmentation (RFM & K-Means)",
            "Repeat Purchase Prediction (ML & SHAP)",
            "Revenue & Sales Forecasting",
            "AI Strategic Insights",
        ],
    )

    if section == "Overview & SQL KPIs":
        st.subheader("📈 Executive KPIs & Relational Metrics")
        st.info("Module initialized. Awaiting dataset download and ingestion.")

    elif section == "Customer Segmentation (RFM & K-Means)":
        st.subheader("👥 Customer Segments & Behavioral Clusters")
        st.info("Segmentation pipeline initialized. Awaiting clustering execution.")

    elif section == "Repeat Purchase Prediction (ML & SHAP)":
        st.subheader("🔮 Predictive Repeat Purchase & SHAP Explainability")
        st.info("ML pipeline initialized. Awaiting model training.")

    elif section == "Revenue & Sales Forecasting":
        st.subheader("📅 Sales & Revenue Projections")
        st.info("Forecasting pipeline initialized. Awaiting time-series fitting.")

    elif section == "AI Strategic Insights":
        st.subheader("💡 Automated Executive Insights & Action Plan")
        st.info("Insight generator initialized. Awaiting analytical inputs.")


if __name__ == "__main__":
    main()
