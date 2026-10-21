from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# Load Model and Scaler
@st.cache_resource
def load_models():
    model = joblib.load(BASE_DIR / "final_lasso_model.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    return model, scaler

model, scaler = load_models()

FEATURES = [
    "Week of the month (first week, second, third, fourth or fifth week",
    "Day of the week (Monday to Friday)",
    "Non-urgent order",
    "Urgent order",
    "Fiscal sector orders",
    "Orders from the traffic controller sector",
    "Banking orders (1)",
    "Banking orders (2)",
    "Banking orders (3)"
]

st.set_page_config(page_title="Daily Order Forecasting", layout="centered")

st.title("📦 Daily Order Forecasting App")
st.write("Enter the order details below to predict total daily orders.")

col1, col2 = st.columns(2)

with col1:
    week_of_month = st.number_input("Week of the month (1-5)", min_value=1.0, max_value=5.0, value=1.0)
    day_of_week = st.number_input("Day of the week (1-5)", min_value=1.0, max_value=5.0, value=1.0)
    non_urgent_order = st.number_input("Non-urgent orders", min_value=0.0, value=100.0)
    urgent_order = st.number_input("Urgent orders", min_value=0.0, value=50.0)
    fiscal_sector_orders = st.number_input("Fiscal sector orders", min_value=0.0, value=20.0)

with col2:
    traffic_controller_orders = st.number_input("Traffic controller orders", min_value=0.0, value=10.0)
    banking_orders_1 = st.number_input("Banking orders (1)", min_value=0.0, value=5.0)
    banking_orders_2 = st.number_input("Banking orders (2)", min_value=0.0, value=5.0)
    banking_orders_3 = st.number_input("Banking orders (3)", min_value=0.0, value=5.0)

if st.button("🔮 Predict Total Orders", type="primary"):
    try:
        values = [[
            week_of_month,
            day_of_week,
            non_urgent_order,
            urgent_order,
            fiscal_sector_orders,
            traffic_controller_orders,
            banking_orders_1,
            banking_orders_2,
            banking_orders_3
        ]]

        input_df = pd.DataFrame(values, columns=FEATURES)
        input_scaled = scaler.transform(input_df)
        prediction = float(model.predict(input_scaled)[0])

        st.success(f"### Predicted Total Orders: **{round(prediction, 3)}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
