import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from core.prepare_features import prepare_features
from core.preprocessing import build_preprocessor
from core.modeling import train_random_forest
from core.evaluation import evaluate_model
from config.config_utils import load_config

def retrain_model(
    house_data_path="HOUSE/housetrain.csv",
    feedback_path="data/logs/feedback_log.csv",
    model_dir="models",
    registry_path="models/registry.json"):
    
    """ gentræner modellen på baggrund af de nye datainput fra /predict endpoint og det gamle dataset"""

    base_data = pd.read_csv(house_data_path)
    feedback_file = Path(feedback_path)

    # sikrer at der er en feedback file og loader det ind i en feedback_data variabel
    if feedback_file.exists():
        feedback_data = pd.read_csv(feedback_file)
        print(f"[RETRAIN] Loaded feedback data with {len(feedback_data)} rows.")
    else:
        feedback_data = pd.DataFrame()
        print("[RETRAIN] No feedback data found, training on base data only.")
    
    # Kombinerer det nye og det gamle datasæt i en variablen combined
    # Ensure feedback data matches base schema
    if not feedback_data.empty:
        if "ActualPrice" in feedback_data.columns:
            feedback_data = feedback_data.rename(columns={"ActualPrice": "SalePrice"})
            print("[RETRAIN] Renamed ActualPrice → SalePrice in feedback data.")
        else:
            print("[RETRAIN] Warning: feedback data has no ActualPrice column.")

        # Drop extra columns that don't exist in base data
        extra_cols = ["timestamp", "PredictedPrice", "Reliability", "model_version"]
        cols_to_drop = [col for col in extra_cols if col in feedback_data.columns]
        if cols_to_drop:
            feedback_data = feedback_data.drop(columns=cols_to_drop)
            print(f"[RETRAIN] Dropped columns from feedback data: {cols_to_drop}")

        # Only keep columns that exist in base data (align schemas)
        common_cols = [col for col in base_data.columns if col in feedback_data.columns]
        feedback_data = feedback_data[common_cols]
        print(f"[RETRAIN] Aligned feedback data to {len(common_cols)} common columns.")

    # Combine base + feedback datasets
    combined = pd.concat([base_data, feedback_data], ignore_index=True)
    print(f"[RETRAIN] Total combined samples: {len(combined)}")

    # Drop rows with missing target
    combined = combined.dropna(subset=["SalePrice"])

    if combined["SalePrice"].isna().any():
        print(f"[RETRAIN] Warning: {combined['SalePrice'].isna().sum()} missing target values dropped.")

    # Forbereder features via den preprocessoren for den originale pipeline
    X, y = prepare_features(combined, target_col="SalePrice")
    preprocessor = build_preprocessor(X)
    config = load_config()

    # Træner modellen på det nye combined datasæt
    model, train_metadata = train_random_forest(X, y, preprocessor, config)



    models_dir = Path(model_dir)
    models_dir.mkdir(exist_ok=True)
    new_version = f"random_forest_v{len(list(models_dir.glob('random_forest_v*.pkl'))) + 1}.pkl"
    model_path = models_dir / new_version

    joblib.dump(model, model_path)
    print(f"[RETRAIN] Model saved → {model_path}")

    # Gemmer den nye model i /models folderen
    registry = {"active_model": new_version, "last_update": datetime.utcnow().isoformat()}
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[RETRAIN] Registry updated → {registry_path}")

    # Evaluerer modellen efter træning
    results = evaluate_model(model, X, y)
    print(f"[RETRAIN] Evaluation: {results}")

    return model, results


if __name__ == "__main__":
    model, results = retrain_model()





