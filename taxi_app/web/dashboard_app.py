import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px

API_URL = "http://localhost:5000"

st.set_page_config(
    page_title="NYC Taxi Prediction Dashboard",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 NYC Taxi Analytics & Prediction Dashboard")
st.markdown("This dashboard visualizes trip insights and uses 3 ML models stored as `.pt` files.")

# -----------------------------------------
# Sidebar
# -----------------------------------------
st.sidebar.header("Upload Trip Dataset")
trip_file = st.sidebar.file_uploader("Upload CSV (Optional for Visualization)", type=["csv"])

if trip_file:
    df = pd.read_csv(trip_file)
else:
    df = None

# -----------------------------------------
# TAB LAYOUT
# -----------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "🔵 High vs Low Fare",
    "🟡 Tip Prediction",
    "🟢 Payment Type Prediction"
])

# =========================================================
# 1. DATA OVERVIEW
# =========================================================
with tab1:
    st.header("📊 Data Overview & Visualizations")

    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=["number"]).columns

        # ---- Histogram ----
        st.subheader("Distribution Plot")
        selected_col = st.selectbox("Select numeric column", numeric_cols)
        fig = px.histogram(df, x=selected_col, nbins=50, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

        # ---- Scatter ----
        st.subheader("Distance vs Fare / Tip")
        scatter_y = st.selectbox("Select Y-axis", ["fare_amount", "tip_amount"])
        fig2 = px.scatter(df, x="trip_distance", y=scatter_y, color="trip_distance",
                          title=f"Trip Distance vs {scatter_y}")
        st.plotly_chart(fig2, use_container_width=True)

        # ---- Payment Bar Plot ----
        if "payment_type" in df.columns:
            st.subheader("Payment Type Distribution")
            fig3 = px.bar(df["payment_type"].value_counts(),
                          title="Payment Type Counts")
            st.plotly_chart(fig3, use_container_width=True)

        # ---- Heatmap ----
        st.subheader("Correlation Heatmap")
        fig4 = px.imshow(df[numeric_cols].corr(), text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig4, use_container_width=True)

    else:
        st.info("Upload a CSV file from the sidebar to unlock visualizations.")

# =========================================================
# 2. HIGH vs LOW FARE PREDICTION
# =========================================================
with tab2:
    st.header("🔵 High vs Low Fare Prediction")

    colA, colB, colC = st.columns(3)
    with colA: distance = st.number_input("Trip Distance (miles)", min_value=0.0)
    with colB: duration = st.number_input("Trip Duration (minutes)", min_value=0.0)
    with colC: fare = st.number_input("Fare Amount", min_value=0.0)

    if st.button("Predict Fare Type"):
        payload = {"distance": distance, "duration": duration, "fare_amount": fare}
        res = requests.post(f"{API_URL}/predict/highlow", json=payload).json()

        st.success(f"Prediction: **{res['prediction']}**")

        # Visualization
        fig = px.pie(names=["Low Fare", "High Fare"],
                     values=[1, 1] if res["prediction"] == "High Fare" else [1, 1],
                     title="Prediction Visualization",
                     color_discrete_sequence=["skyblue", "gold"])
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TIP PREDICTION
# =========================================================
with tab3:
    st.header("🟡 Tip Amount Prediction")

    col1, col2, col3 = st.columns(3)
    with col1: tdist = st.number_input("Distance", min_value=0.0)
    with col2: tdur = st.number_input("Duration", min_value=0.0)
    with col3: tfare = st.number_input("Fare", min_value=0.0)

    if st.button("Predict Tip"):
        payload = {"distance": tdist, "duration": tdur, "fare_amount": tfare}
        res = requests.post(f"{API_URL}/predict/tip", json=payload).json()

        st.success(f"Predicted Tip: **${res['predicted_tip_amount']}**")

        # Visualization
        fig_tip = px.bar(x=["Predicted Tip"], y=[res["predicted_tip_amount"]],
                         title="Predicted Tip Visualization")
        st.plotly_chart(fig_tip, use_container_width=True)

# =========================================================
# PAYMENT TYPE PREDICTION
# =========================================================
with tab4:
    st.header("🟢 Payment Type Prediction")

    colx, coly, colz = st.columns(3)
    with colx: pdist = st.number_input("Distance", min_value=0.0)
    with coly: pdur = st.number_input("Duration", min_value=0.0)
    with colz: pfare = st.number_input("Fare", min_value=0.0)

    if st.button("Predict Payment Type"):
        payload = {"distance": pdist, "duration": pdur, "fare_amount": pfare}
        res = requests.post(f"{API_URL}/predict/payment", json=payload).json()

        st.success(f"Payment Method: **{res['predicted_payment_type']}**")

        # Visualization
        fig_pay = px.pie(
            names=["Cash", "Credit Card", "No Charge", "Dispute"],
            values=[1 if res["predicted_payment_type"] == p else 0 for p in ["Cash", "Credit Card", "No Charge", "Dispute"]],
            title="Payment Type Prediction Visualization"
        )
        st.plotly_chart(fig_pay, use_container_width=True)
