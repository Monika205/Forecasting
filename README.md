# Daily Order Forecasting Project

## Project Overview
This project predicts total orders using a trained regression model.
The model uses nine input features.

## Project Files
- app.py: FastAPI prediction backend
- frontend.py: Streamlit user interface
- final_lasso_model.pkl: Saved Tuned Lasso Regression model
- feature_scaler.pkl: Fitted feature scaler
- requirements.txt: Required Python packages
- Dockerfile: Docker image configuration

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Start the API from this folder:

    uvicorn app:app --host 127.0.0.1 --port 8000

Start the frontend in a separate terminal:

    streamlit run frontend.py

## API Endpoints
- GET /health: Checks API availability
- POST /predict: Returns a predicted total-orders value

Interactive API documentation is available at:

    http://127.0.0.1:8000/docs

## Model Evaluation
The selected Lasso model achieved approximately:
- MAE: 20.166
- RMSE: 47.692
- R-squared: 0.373

Tuned Ridge achieved a lower MAE of approximately 18.870.
The final model choice should consider the evaluation metric and project objective.

## Deployment Notes
The Dockerfile has been created, but a Docker image has not yet been built.
A Docker-enabled environment is required to build and run the container.
