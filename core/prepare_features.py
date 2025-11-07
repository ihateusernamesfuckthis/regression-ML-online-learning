import pandas as pd
import numpy as np

def prepare_features (df: pd.DataFrame, target_col: str = "SalePrice"):
    
    # Sikkerhedsnet - modificerer en kopi, og ikke den 'rigtige' dataframe
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target Column '{target_col}' not found in DataFrame")

    # Kategoriske features behandles - hvis kolonnen mangler ved inferens inputtet, så erstattes det med default
    for col, default in {
        "Fence": "NoFence",
        "PoolQC": "NoPool",
        "FireplaceQu": "NoFireplace",
        "MasVnrType": "None",
        "Alley": "NoAlleyAccess"
    }.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    
    
    # Garage Domæne - hvis kolonne mangler ved inferens input, oprettes kolonne og fyldes med default
    garage_features = ["GarageType", "GarageQual", "GarageFinish", "GarageCond"]
    for col in garage_features:
        if col not in df.columns:
            df[col] = "NoGarage"
        else:
            df[col] = df[col].fillna("NoGarage")

    if "GarageYrBlt" not in df.columns:
        df["GarageYrBlt"] = 0
    else:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0).astype(int)

    df["HasGarage"] = (df["GarageYrBlt"] > 0).astype(int)
    
    # Numeriske Features
    if "LotFrontage" not in df.columns:
        df["LotFrontage"] = 0.0
    else:
        df["LotFrontage"] = df["LotFrontage"].fillna(0.0)

    if "MasVnrArea" not in df.columns:
        df["MasVnrArea"] = 0.0
    else:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0.0)


    X = df.drop(columns=[target_col])
    y_log = np.log1p(df[target_col])

    print("Features er forberedt")

    return X, y_log