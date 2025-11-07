import joblib
import pandas as pd
from config_utils import load_config
from pathlib import Path
from prepare_features import prepare_features

def load_model(model_path: str):
    """Loader den serialized model pipeline fra disk"""
    model = joblib.load(model_path)
    print(f"Modellen er loaded fra {model_path}")

    return model

def prepare_input(raw_df: pd.DataFrame):
    """Behandler det rå datainput så det matcher med de features modellen forventer"""

    try:
        X_prepared, _ = prepare_features(raw_df, target_col="SalePrice")
    except ValueError:
        X_new, _ = prepare_features(df.assign(SalePrice=0), target_col="SalePrice")
    return X_prepared


def predict (model, data: pd.DataFrame):
    """Laver prediction ud fra den trænede model"""
    preds = model.predict(data)
    preds_exp = pd.Series(preds, name="PredictedPrice")
    return preds_exp

def main():
    config = load_config()
    model_path = Path("model_path")
    model = load_model(model_path)

    # Loader den nye data
    new_data = pd.read_csv("HOUSE/housetest_sample.csv")

    # Sikrer at den nye data matcher med hvad modellen forventer af features
    X_new = prepare_input(new_data)

    # Laver prediction
    preds = predict(model, X_new)

    print("\nPredictions:")
    print(preds.head())


if __name__ == "__main__":
    main()

