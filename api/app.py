from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from config.config_utils import load_config
from core.prepare_features import prepare_features
from core.schema_harmonizer import align_features
from core.reliability import assess_reliability
from data.data_logger import log_prediction
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

    # Sikrer at Target kolonne eksisterer
    if "SalePrice" not in df.columns:
        df["SalePrice"] = 0  # dummy value

    # Forbereder features
    X_new, _ = prepare_features(df, target_col="SalePrice")

    # inputtet harmoniseres
    X_aligned, info = align_features(X_new, model)

    # Vurdering af robusthed
    reliability_dict = assess_reliability(info["completeness"])
    reliability_score = reliability_dict["score"]
    reliability_fraction = reliability_dict["completeness"]

    preds = model.predict(X_aligned)
    price = np.expm1(preds[0])

    log_prediction(
        input_data=request.data,
        prediction=float(np.expm1(preds[0])),
        reliability=float(reliability_fraction),
        model_version=model_path.name
    )

    return {
        "PredictedPrice": round(float(price), 2),
        "SchemaInfo": info,
        "Reliability": reliability_dict
        }
