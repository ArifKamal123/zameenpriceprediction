from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow.pyfunc
import pandas as pd
import numpy as np
from app.custom_transform import AreaUnitConverter
import os
model_path = os.path.join(os.path.dirname(__file__), "mlruns/471310202707317169/a2f3a59be7654ae799101585623431b2/artifacts/xgb_price_pipeline")
model = mlflow.pyfunc.load_model(f"file:{model_path}")


#model = joblib.load('../model/xgb_price_pipeline.pkl')
mlflow.set_tracking_uri("file:mlruns")



app = FastAPI(title='Zameen Price Prediction API')

class PropertyInput(BaseModel):
    area: float
    bedrooms: int
    baths: int
    city: str
    location: str
    purpose: str
    property_type: str

@app.post("/predict")
def predict_price(data: PropertyInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict log price
    log_price = float(model.predict(input_df)[0])

    # Convert back to original price scale
    price = float(np.expm1(log_price))


    return {
        "predicted_price": round(price)
                }
