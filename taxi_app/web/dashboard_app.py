import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:5001"
 
st.set_page_config(
    page_title="NYC Taxi ML Dashboard",
    
    layout="wide",
)

 
st.markdown("""
<style>
 
body, .stApp {
    background: #e0e5ec;
}

 
section.main > div {
    padding-top: 1.5rem;
}

 
h1 {
    font-weight: 800 !important;
    letter-spacing: 0.03em;
}

 
 
.pred-box {
    background: #e0e5ec;
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow:
        inset 5px 5px 10px #b8bec7,
        inset -5px -5px 10px #ffffff;
    font-size: 1.05rem;
    font-weight: 600;
}

 
.section-title {
    font-weight: 700;
    font-size: 1.2rem;
    margin-bottom: 0.3rem;
}

 
.stButton > button {
    background: #e0e5ec;
    color: #333333;
    border-radius: 999px;
    padding: 0.5rem 1.5rem;
    border: none;
    box-shadow:
        4px 4px 8px #a3b1c6,
        -4px -4px 8px #ffffff;
    font-weight: 600;
    transition: all 0.15s ease-in-out;
}

.stButton > button:hover {
    box-shadow:
        2px 2px 4px #a3b1c6,
        -2px -2px 4px #ffffff;
    transform: translateY(1px);
}

 
button[data-baseweb="tab"] > div {
    font-weight: 600 !important;
}

 
.stNumberInput > div, .stSelectbox > div, .stTextInput > div {
    border-radius: 12px !important;
    box-shadow:
        inset 3px 3px 6px #bec4cf,
        inset -3px -3px 6px #ffffff;
}

 
.css-1d391kg, .css-1lcbmhc {
    background: #e0e5ec !important;
}
</style>
""", unsafe_allow_html=True)

 
st.title("NYC Taxi App")


 
st.sidebar.header("Upload Dataset")
u_file = st.sidebar.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])

df = None
if u_file:
    if u_file.name.endswith(".csv"):
        df = pd.read_csv(u_file)
    else:
        df = pd.read_parquet(u_file, engine="pyarrow")

# ---------- TABS ----------
tab_overview, tab_highlow, tab_tip, tab_payment = st.tabs([
    "Data Overview",
    "High vs Low Fare",
    "Tip Prediction",
    "Payment Type Prediction"
])

 
with tab_overview:
    st.markdown("<div class='neu-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    if df is None:
        st.info("Upload a CSV or Parquet file from the left sidebar to explore the data.")
    else:
        st.dataframe(df.head(), use_container_width=True)

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='section-title'>Distribution Plot</div>", unsafe_allow_html=True)
                sel_col = st.selectbox("Choose numeric column", numeric_cols)
                fig = px.histogram(df, x=sel_col, nbins=50)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "trip_distance" in df.columns and "fare_amount" in df.columns:
                    st.markdown("<div class='section-title'>Trip Distance vs Fare</div>", unsafe_allow_html=True)
                    fig2 = px.scatter(df, x="trip_distance", y="fare_amount",
                                      opacity=0.7)
                    st.plotly_chart(fig2, use_container_width=True)

            st.markdown("<div class='section-title'>Correlation Heatmap</div>", unsafe_allow_html=True)
            fig3 = px.imshow(df[numeric_cols].corr(), text_auto=True)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No numeric columns found in the dataset.")

    st.markdown("</div>", unsafe_allow_html=True)


 
with tab_highlow:
    st.markdown("<div class='neu-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔵 High vs Low Fare – Trip Inputs</div>", unsafe_allow_html=True)

    
    r1c1, r1c2, r1c3 = st.columns(3)
    VendorID = r1c1.number_input("VendorID", 1, 10, 1)
    RatecodeID = r1c2.number_input("RatecodeID", 1, 10, 1)
    passenger_count = r1c3.number_input("Passenger Count", 0, 10, 1)

    
    r2c1, r2c2, r2c3 = st.columns(3)
    PULocationID = r2c1.number_input("PULocationID", 1, 500, 132)
    DOLocationID = r2c2.number_input("DOLocationID", 1, 500, 256)
    pickup_hour = r2c3.number_input("Pickup Hour (0–23)", 0, 23, 14)

    
    r3c1, r3c2, r3c3 = st.columns(3)
    pickup_day = r3c1.number_input("Pickup Day (0–6)", 0, 6, 2)
    trip_distance = r3c2.number_input("Trip Distance (miles)", 0.0, None, 2.4)
    trip_duration_min = r3c3.number_input("Trip Duration (min)", 0.0, None, 12.0)

    
    r4c1, r4c2, r4c3 = st.columns(3)
    fare_amount = r4c1.number_input("Fare Amount", 0.0, None, 15.5)
    tolls_amount = r4c2.number_input("Tolls Amount", 0.0, None, 0.0)
    improvement_surcharge = r4c3.number_input("Improvement Surcharge", 0.0, None, 0.3)

    
    r5c1, r5c2, r5c3 = st.columns(3)
    congestion_surcharge = r5c1.number_input("Congestion Surcharge", 0.0, None, 2.5)
    Airport_fee = r5c2.number_input("Airport Fee", 0.0, None, 1.25)
    cbd_congestion_fee = r5c3.number_input("CBD Congestion Fee", 0.0, None, 0.0)

    highlow_payload = {
        "VendorID": VendorID,
        "RatecodeID": RatecodeID,
        "PULocationID": PULocationID,
        "DOLocationID": DOLocationID,
        "pickup_hour": pickup_hour,
        "pickup_day_of_week": pickup_day,
        "trip_distance": trip_distance,
        "trip_duration_min": trip_duration_min,
        "passenger_count": passenger_count,
        "fare_amount": fare_amount,
        "tolls_amount": tolls_amount,
        "improvement_surcharge": improvement_surcharge,
        "congestion_surcharge": congestion_surcharge,
        "Airport_fee": Airport_fee,
        "cbd_congestion_fee": cbd_congestion_fee,
         
        "extra": 0.0,
        "mta_tax": 0.0
    }

    if st.button("Predict High / Low Fare"):
        res = requests.post(f"{API_URL}/predict/highlow", json=highlow_payload)
        out = res.json()

        if "error" in out:
            st.error(out["error"])
        else:
            pred = out["prediction"]

             
            st.markdown(
                f"<div class='pred-box'>Prediction: {pred}</div>",
                unsafe_allow_html=True
            )

             
            fig = px.pie(
                names=["High Fare", "Low Fare"],
                values=[1, 0] if pred == "High Fare" else [0, 1],
                title="High vs Low Fare Prediction",
                color_discrete_sequence=["#4a6cf7", "#c3c9e9"]
            )
            st.plotly_chart(fig, use_container_width=True)


    st.markdown("</div>", unsafe_allow_html=True)


 
with tab_tip:
    st.markdown("<div class='neu-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🟡 Tip Prediction – Trip Inputs</div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)
    fare_amount_tip = t1.number_input("Fare Amount", 0.0, None, 15.5, key="tip_fare")
    trip_distance_tip = t2.number_input("Trip Distance", 0.0, None, 2.4, key="tip_dist")
    trip_duration_tip = t3.number_input("Trip Duration (min)", 0.0, None, 12.0, key="tip_dur")
    pickup_hour_tip = t4.number_input("Pickup Hour", 0, 23, 14, key="tip_hour")

    tip_payload = {
        "fare_amount": float(fare_amount_tip),
        "trip_distance": float(trip_distance_tip),
        "trip_duration_min": float(trip_duration_tip),
        "pickup_hour": int(pickup_hour_tip),

         
        "extra": 0.0,
        "mta_tax": 0.0,
        "tolls_amount": 0.0,
        "Airport_fee": 0.0,
        "VendorID": 1,
        "RatecodeID": 1,
        "PULocationID": 100,
        "DOLocationID": 100,
        "pickup_day_of_week": 2,
        "passenger_count": 1,
        "improvement_surcharge": 0.3,
        "congestion_surcharge": 2.5,
        "cbd_congestion_fee": 0.0
    }

    if st.button("Predict Tip Amount"):
        res = requests.post(f"{API_URL}/predict/tip", json=tip_payload)
        out = res.json()

        if "error" in out:
            st.error(out["error"])
        else:
            tip_val = out["predicted_tip_amount"]

             
            st.markdown(
                f"<div class='pred-box'>Predicted Tip Amount: ${tip_val}</div>",
                unsafe_allow_html=True
            )

             
            fig_tip = px.bar(
                x=["Predicted Tip"],
                y=[tip_val],
                title="Predicted Tip Amount",
                text=[f"${tip_val}"],
                color=["Predicted Tip"],
                color_discrete_sequence=["#f7b731"]
            )
            fig_tip.update_traces(textposition="outside")
            st.plotly_chart(fig_tip, use_container_width=True)


    st.markdown("</div>", unsafe_allow_html=True)


 
with tab_payment:
    st.markdown("<div class='neu-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🟢 Payment Type Prediction – Trip Inputs</div>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    VendorID_pay = p1.number_input("VendorID", 1, 10, 1, key="pay_vendor")
    RatecodeID_pay = p2.number_input("RatecodeID", 1, 10, 1, key="pay_rate")
    PULocationID_pay = p3.number_input("PULocationID", 1, 500, 132, key="pay_pu")

    p4, p5, p6 = st.columns(3)
    DOLocationID_pay = p4.number_input("DOLocationID", 1, 500, 256, key="pay_do")
    pickup_hour_pay = p5.number_input("Pickup Hour", 0, 23, 14, key="pay_hour")
    pickup_day_pay = p6.number_input("Pickup Day (0–6)", 0, 6, 2, key="pay_day")

    p7, p8, p9 = st.columns(3)
    trip_distance_pay = p7.number_input("Trip Distance", 0.0, None, 2.4, key="pay_dist")
    fare_amount_pay = p8.number_input("Fare Amount", 0.0, None, 15.5, key="pay_fare")
    extra_pay = p9.number_input("Extra", 0.0, None, 0.5, key="pay_extra")

    p10, p11, p12 = st.columns(3)
    mta_tax_pay = p10.number_input("MTA Tax", 0.0, None, 0.5, key="pay_mta")
    tolls_amount_pay = p11.number_input("Tolls Amount", 0.0, None, 0.0, key="pay_tolls")
    Airport_fee_pay = p12.number_input("Airport Fee", 0.0, None, 1.25, key="pay_air")

    trip_duration_pay = st.number_input("Trip Duration (min)", 0.0, None, 12.0, key="pay_dur")

    payment_payload = {
        "VendorID": VendorID_pay,
        "RatecodeID": RatecodeID_pay,
        "PULocationID": PULocationID_pay,
        "DOLocationID": DOLocationID_pay,
        "pickup_hour": pickup_hour_pay,
        "pickup_day_of_week": pickup_day_pay,
        "trip_distance": trip_distance_pay,
        "fare_amount": fare_amount_pay,
        "extra": extra_pay,
        "mta_tax": mta_tax_pay,
        "tolls_amount": tolls_amount_pay,
        "Airport_fee": Airport_fee_pay,
        "trip_duration_min": trip_duration_pay,
        # required but unused by payment model:
        "passenger_count": 1,
        "improvement_surcharge": 0.3,
        "congestion_surcharge": 2.5,
        "cbd_congestion_fee": 0.0
    }

    if st.button("Predict Payment Type"):
        res = requests.post(f"{API_URL}/predict/payment", json=payment_payload)
        out = res.json()

        if "error" in out:
            st.error(out["error"])
        else:
            pay_type = out["predicted_payment_type"]

             
            st.markdown(
                f"<div class='pred-box'>Predicted Payment Type: {pay_type}</div>",
                unsafe_allow_html=True
            )

            labels = ["Credit Card", "Cash", "No Charge", "Dispute", "Unknown/Other"]
            values = [1 if label == pay_type else 0 for label in labels]

             
            fig_pay = px.pie(
                names=labels,
                values=values,
                title="Payment Type Prediction",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_pay, use_container_width=True)


    st.markdown("</div>", unsafe_allow_html=True)
