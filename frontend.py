import streamlit as st
import requests

st.set_page_config(
    page_title="Daily Demand Forecasting App",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Daily Demand Forecasting App")
st.write(
    "This application uses a machine learning model to predict the "
    "**Target (Total daily orders)** based on various sector metrics."
)

st.sidebar.header("⚙️ Input Features")

# Input fields according to the 9 features used in model training
week_of_month = st.sidebar.slider("Week of the month", 1, 5, 2)
day_of_week = st.sidebar.slider("Day of the week (2=Mon, 6=Fri)", 2, 6, 3)
non_urgent_order = st.sidebar.number_input("Non-urgent orders", min_value=0.0, value=150.0)
urgent_order = st.sidebar.number_input("Urgent orders", min_value=0.0, value=115.0)
fiscal_sector_orders = st.sidebar.number_input("Fiscal sector orders", min_value=0.0, value=10.0)
traffic_controller_orders = st.sidebar.number_input("Traffic controller orders", min_value=0.0, value=45000.0)
banking_orders_1 = st.sidebar.number_input("Banking orders (1)", min_value=0.0, value=40000.0)
banking_orders_2 = st.sidebar.number_input("Banking orders (2)", min_value=0.0, value=75000.0)
banking_orders_3 = st.sidebar.number_input("Banking orders (3)", min_value=0.0, value=20000.0)

if st.button("🔮 Predict Total Orders", type="primary"):
    payload = {
        "week_of_month": float(week_of_month),
        "day_of_week": float(day_of_week),
        "non_urgent_order": float(non_urgent_order),
        "urgent_order": float(urgent_order),
        "fiscal_sector_orders": float(fiscal_sector_orders),
        "traffic_controller_orders": float(traffic_controller_orders),
        "banking_orders_1": float(banking_orders_1),
        "banking_orders_2": float(banking_orders_2),
        "banking_orders_3": float(banking_orders_3)
    }

    try:
        # Connect to local FastAPI microservice
        response = requests.post("http://127.0.0.1:8001/predict", json=payload, timeout=5)
        
        if response.status_code == 200:
            prediction = response.json()["predicted_total_orders"]
            st.success(f"### 🎯 Predicted Total Daily Orders: **{prediction}**")
        else:
            st.error(f"API Error (Status {response.status_code}): {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend forecasting API. Please ensure the FastAPI server is running on port 8000.")