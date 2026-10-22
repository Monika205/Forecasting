
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "final_lasso_model.pkl")
scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")

app = FastAPI(title="Daily Order Forecasting API")

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

class PredictionInput(BaseModel):
    week_of_month: float
    day_of_week: float
    non_urgent_order: float
    urgent_order: float
    fiscal_sector_orders: float
    traffic_controller_orders: float
    banking_orders_1: float
    banking_orders_2: float
    banking_orders_3: float

@app.get("/")
def home():
    return {"message": "Daily Order Forecasting API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        values = [[
            data.week_of_month,
            data.day_of_week,
            data.non_urgent_order,
            data.urgent_order,
            data.fiscal_sector_orders,
            data.traffic_controller_orders,
            data.banking_orders_1,
            data.banking_orders_2,
            data.banking_orders_3
        ]]

        input_df = pd.DataFrame(values, columns=FEATURES)
        input_scaled = scaler.transform(input_df)
        prediction = float(model.predict(input_scaled)[0])

        return {"predicted_total_orders": round(prediction, 3)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
