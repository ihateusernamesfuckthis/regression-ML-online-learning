import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model (model, X_test, y_log_test):

    y_log_pred = model.predict(X_test)
    y_log_true = y_log_test

    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n Evaluation Results:")
    print(f"MAE  (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R²   (Coefficient of Determination): {r2:.4f}")

    results = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return results

# 1 Online learning - konverter projektet til et online learn format
# 2 API - konverter projektet til et API




