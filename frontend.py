
import streamlit as st
import requests

API_URL = "https://forecasting-2.onrender.com/predict"

st.set_page_config(
    page_title="Daily Demand Forecasting",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Daily Demand Forecasting App")
st.write(
    "Predict total daily orders using sector metrics."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        week = st.number_input(
            "Week of the month (1-5)",
            min_value=1, max_value=5, value=1
        )
        day = st.number_input(
            "Day of the week (1-5)",
            min_value=1, max_value=5, value=1
        )
        non_urgent = st.number_input(
            "Non-urgent orders", min_value=0, value=100
        )
        urgent = st.number_input(
            "Urgent orders", min_value=0, value=50
        )
        fiscal = st.number_input(
            "Fiscal sector orders", min_value=0, value=20
        )

    with col2:
        traffic = st.number_input(
            "Traffic controller orders", min_value=0, value=10
        )
        bank1 = st.number_input(
            "Banking orders (1)", min_value=0, value=5
        )
        bank2 = st.number_input(
            "Banking orders (2)", min_value=0, value=5
        )
        bank3 = st.number_input(
            "Banking orders (3)", min_value=0, value=5
        )

    submitted = st.form_submit_button("🔮 Predict Total Orders")

if submitted:
    payload = {
        "week_of_month": week,
        "day_of_week": day,
        "non_urgent_order": non_urgent,
        "urgent_order": urgent,
        "fiscal_sector_orders": fiscal,
        "traffic_controller_orders": traffic,
        "banking_orders_1": bank1,
        "banking_orders_2": bank2,
        "banking_orders_3": bank3
    }

    try:
        with st.spinner("Getting prediction from backend..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=90
            )

        if response.status_code == 200:
            result = response.json()
            prediction = result["predicted_total_orders"]
            st.success(
                f"Predicted Total Orders: {prediction}"
            )
        else:
            st.error(
                f"Backend error ({response.status_code}): "
                f"{response.text}"
            )

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the Render API: {e}")
