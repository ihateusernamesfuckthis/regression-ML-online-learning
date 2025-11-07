from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import joblib
import json

def train_random_forest (X_train, y_train, preprocessor, config):


    model = RandomForestRegressor(random_state=42)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    param_dist = {
    'model__n_estimators': [100, 200, 400, 600],
    'model__max_depth': [None, 10, 20, 30, 40],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
    'model__bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator = pipeline,
        param_distributions=param_dist,
        n_iter = 30,
        cv = 3,
        scoring = "neg_root_mean_squared_error",
        verbose = 2,
        random_state = 42,
        n_jobs = -1
    )

    random_search.fit(X_train, y_train)

    print("Best parameters:", random_search.best_params_)
    print("Best RMSE:", abs(random_search.best_score_))

    best_model = random_search.best_estimator_

    model_path = Path(config["model_path"])
    metadata_path = Path(config["metadata_path"])
    metadata_path.parent.mkdir(exist_ok = True, parents = True)

    joblib.dump(best_model, model_path)

    metadata = {
        "timestamp": str(pd.Timestamp.now()),
        "best_params": random_search.best_params_,
        "best_score": abs(random_search.best_score_),
        "model_path": str(model_path)
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata og model gemt til {metadata_path}")

    return best_model, metadata





