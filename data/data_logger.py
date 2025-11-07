# Data logger har til formål at gemme de prediction som modellen via /predict endpointet

import os
import pandas as pd
from datetime import datetime
from pathlib import Path

def log_prediction(input_data: dict, prediction: float, reliability: float, model_version: str):
    """ 
    Logger prediction og metadata fra request til online learning.

    Parameters er:

    input_data : dict
        Feature dictionary sendt til modellen
    prediction : float
        Modellens predicted value
    Reliability : float
        Reliability/completeness score
    model_version : str
        Current model version (f.eks 'random_forest_v3.pkl')
    """

    # Definerer hvilken folder det skal gemmes i, og bekræfter path dertil
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "predictions_log.csv"

    # Opretter et log input/entry
    entry = input_data.copy()
    entry.update({
        "timestamp": datetime.utcnow().isoformat(),
        "PredictedPrice": prediction,
        "Reliability": reliability,
        "model_version": model_version
    })

    # Konverterer til DataFrame -> en række
    df_entry = pd.DataFrame([entry])

    header = not log_path.exists()
    df_entry = pd.DataFrame([entry])
    df_entry.to_csv(log_path, mode="a", header=header, index=False, encoding="utf-8")

    