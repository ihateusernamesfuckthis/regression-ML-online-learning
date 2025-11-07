import pandas as pd

def align_features(df: pd.DataFrame, model) -> tuple[pd.DataFrame, dict]:
    """Få inference input til at matche modellens forventede features.
        - Tilføjer manglende kolonner
        - Sletter ekstra kolonner
        - Sorterer rækkefølge på kolonne baseret på forventede features
        
        returns:
            (aligned_df, info)
            info = {
            "missing_columns": [...],
            "extra_columns": [...],
            "completeness": float
            }
    """

    preprocessor = model.named_steps["preprocess"]
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]

    expected_cols = list(model.feature_names_in_)
    input_cols = list(df.columns)

    missing = [c for c in expected_cols if c not in input_cols]
    extra = [c for c in input_cols if c not in expected_cols]

    for col in missing:
        if col in num_cols:
            df[col] = 0.0
        elif col in cat_cols:
            df[col] = "None"
        else:
            df[col] = 0  # fallback

    df = df[[c for c in expected_cols if c in df.columns]]

    completeness = 1 - len(missing) / len(expected_cols)

    info = {
        "missing_columns": missing,
        "extra_columns": extra,
        "completeness": round(completeness, 3)
    }

    return df, info
