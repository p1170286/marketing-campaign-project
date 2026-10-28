import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Marketing Campaign Predictor",
    page_icon="📊"
)

st.title("📊 Marketing Campaign Performance Predictor")
st.subheader("Nykaa, Purplle, Tira campaign results prediction")

model_path = Path("marketing_models.joblib")

if model_path.exists():
    models = joblib.load(model_path)
    st.success("Saved models loaded successfully!")
    st.subheader("Predict Campaign Performance")

    with st.form("campaign_form"):
        brand = st.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])

        campaign_type = st.selectbox(
            "Campaign Type",
            ["Email", "Influencer", "Paid Ads", "SEO", "Social Media"]
        )

        target_audience = st.selectbox(
            "Target Audience",
            ["College Students", "Working Women", "Youth", "Premium Shoppers"]
        )

        duration = st.number_input(
            "Duration (days)", min_value=1.0, value=30.0
        )

        channel = st.selectbox(
            "Marketing Channel",
            ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]
        )

        impressions = st.number_input(
            "Impressions", min_value=0.0, value=10000.0
        )
        clicks = st.number_input("Clicks", min_value=0.0, value=1000.0)
        leads = st.number_input("Leads", min_value=0.0, value=100.0)
        conversions = st.number_input(
            "Conversions", min_value=0.0, value=50.0
        )
        acquisition_cost = st.number_input(
            "Acquisition Cost", min_value=0.0, value=500.0
        )

        language = st.selectbox("Language", ["English", "Hindi", "Tamil"])

        engagement_score = st.number_input(
            "Engagement Score", min_value=0.0, value=5.0
        )

        customer_segment = st.selectbox(
            "Customer Segment",
            ["College Students", "Working Women", "Youth", "Premium Shoppers"]
        )

        predict_button = st.form_submit_button("Predict")

    if predict_button:
        base_features = {
            "Campaign_Type": campaign_type,
            "Target_Audience": target_audience,
            "Duration": duration,
            "Impressions": impressions,
            "Clicks": clicks,
            "Leads": leads,
            "Conversions": conversions,
            "Acquisition_Cost": acquisition_cost,
            "Language": language,
            "Engagement_Score": engagement_score,
            "Customer_Segment": customer_segment,
            "Brand": brand,
        }

        revenue_columns = list(
            models["revenue_model"].feature_names_in_
        )

        for column in revenue_columns:
            if column.startswith("Channel_"):
                base_features[column] = int(
                    column == f"Channel_{channel}"
                )

        revenue_input = pd.DataFrame(
            [[base_features[column] for column in revenue_columns]],
            columns=revenue_columns
        )

        predicted_revenue = models["revenue_model"].predict(
            revenue_input
        )[0]

        profit_features = base_features.copy()
        profit_features["Revenue"] = predicted_revenue

        profit_columns = list(
            models["profit_model"].feature_names_in_
        )

        profit_input = pd.DataFrame(
            [[profit_features[column] for column in profit_columns]],
            columns=profit_columns
        )

        profit_prediction = models["profit_model"].predict(
            profit_input
        )[0]

        st.subheader("Prediction Results")
        st.metric("Predicted Revenue", f"₹{predicted_revenue:,.2f}")

        if profit_prediction == 1:
            st.success("The campaign is predicted to be profitable.")
        else:
            st.warning("The campaign is predicted to make a loss.")

else:
    st.error(
        "marketing_models.joblib file not found. "
        "Please check that it is in the project folder."
    )