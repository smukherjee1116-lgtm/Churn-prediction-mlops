from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import Optional

# Load model artifacts
with open('./models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('./models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability",
    version="1.0.0"
)

# Input schema
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    gender: int
    Partner: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int
    NumServices: int
    IsMonthToMonth: int
    IsHighValue: int

# Health check endpoint
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!",
            "status": "healthy"}

# Prediction endpoint
@app.post("/predict")
def predict(customer: CustomerData):
    
    # Convert input to dataframe
    input_dict = customer.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Add engineered features
    input_df['AvgMonthlySpend'] = (input_df['TotalCharges'] / 
                                    (input_df['tenure'] + 1))
    input_df['ChargePerService'] = (input_df['MonthlyCharges'] / 
                                     (input_df['NumServices'] + 1))
    input_df['TenureGroup'] = pd.cut(input_df['tenure'],
                                      bins=[0, 12, 36, 72],
                                      labels=[0, 1, 2],
                                      include_lowest=True).astype(float).fillna(0).astype(int)
    
    # Align with training features
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    
    # Scale numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'AvgMonthlySpend', 'ChargePerService']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.7 else 
                      "Medium" if probability > 0.4 else "Low",
        "message": "Customer likely to churn!" if prediction == 1 
                   else "Customer likely to stay"
    }

# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    results = []
    for customer in customers:
        result = predict(customer)
        results.append(result)
    return {"predictions": results, "total": len(results)}