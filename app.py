from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "final_lasso_model.pkl")
scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")

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

app = FastAPI(
    title="Daily Order Forecasting API",
    description="Predicts total orders using the trained regression model.",
    version="1.0.0"
)

class OrderInput(BaseModel):
    week_of_month: float = Field(ge=1, le=5)
    day_of_week: float = Field(ge=1, le=5)
    non_urgent_order: float = Field(ge=0)
    urgent_order: float = Field(ge=0)
    fiscal_sector_orders: float = Field(ge=0)
    traffic_controller_orders: float = Field(ge=0)
    banking_orders_1: float = Field(ge=0)
    banking_orders_2: float = Field(ge=0)
    banking_orders_3: float = Field(ge=0)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "Tuned Lasso Regression"}

@app.post("/predict")
def predict(data: OrderInput):
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

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )