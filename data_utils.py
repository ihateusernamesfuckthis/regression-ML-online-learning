import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# load_data funktionen er gjort mere dynamisk, og der er tilføjet fejlhåndtering
def load_data (path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Datasættet er ikke fundet: {path.resolve()}")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame):
    drop_cols = ["Id", "MiscFeature", ]
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def split_data(X, y_log, test_size=0.2, random_state=42):

    X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=test_size, random_state=random_state)

    print("Data er splittet")
    return X_train, X_test, y_log_train, y_log_test

