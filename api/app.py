from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from config_utils import load_config
from prepare_features import prepare_features
from schema_harmonizer import align_features
from reliability import assess_reliability
import joblib
from pathlib import Path
import numpy as np

config = load_config()
model_path = Path(config["model_path"])
model = joblib.load(model_path)

app = FastAPI(title = "House Price Prediction API")

class HouseData(BaseModel):
    data: dict

@app.get("/")
def root():
    return {"message": "House Price Prediction API kører"}

@app.post("/predict")
def predict_house_price(request: HouseData):
    df = pd.DataFrame([request.data])

    # Ensure the expected target column exists for prepare_features()
    if "SalePrice" not in df.columns:
        df["SalePrice"] = 0  # dummy value


    X_new, _ = prepare_features(df, target_col="SalePrice")

    # inputtet harmoniseres
    X_aligned, info = align_features(X_new, model)

    # Vurdering af robusthed
    reliability = assess_reliability(info["completeness"])

    preds = model.predict(X_aligned)

    price = np.expm1(preds[0])
    return {
        "PredictedPrice": round(float(price), 2),
        "SchemaInfo": info,
        "Reliability": reliability
        }
