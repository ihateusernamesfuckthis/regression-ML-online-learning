import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(X_train: pd.DataFrame):

    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(exclude=np.number).columns

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    return preprocessor


# Skriv hver gang jeg laver en antagelse

