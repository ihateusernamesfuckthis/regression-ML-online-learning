import pandas as pd
import numpy as np
from pathlib import Path

def simulate_feedback(log_path="data/logs/predictions_log.csv",
                      feedback_path="data/logs/feedback_log.csv",
                      noise_std=0.1):
    
    """
    Simulerer feedback for nye predictions der ikke har reelle priser.
    Feedback bliver appended til en eksisterende log i stedet for at overskrive.
    
    Parameters
    ----------
    log_path : str
        Path to the predictions log.
    feedback_path : str
        Path to the feedback log file.
    noise_std : float
        Std deviation for simulated noise (e.g. 0.1 = ±10% variation).
    """
    
    log_file = Path(log_path)
    if not log_file.exists():
        raise FileNotFoundError(f"Prediction log ikke fundet ved {log_path}")
    
    # Loader de seneste predictions
    df_pred = pd.read_csv(log_file)
    if "PredictedPrice" not in df_pred.columns:
        raise ValueError("Logfile skal indholde 'PredictedPrice'")
    
    # Loader seneste feedback, hvis de eksisterer
    feedback_file = Path(feedback_path)
    if feedback_file.exists():
        df_feedback = pd.read_csv(feedback_file)

        # Sammenligner og tjekker efter timestamps der ikke eksisterer i loggen endnu
        known_timestamps = set(df_feedback["timestamp"].astype(str))
        df_new = df_pred[~df_pred["timestamp"].astype(str).isin(known_timestamps)]

    else:
        df_feedback = pd.DataFrame()
        df_new = df_pred.copy()
    
    # Hvis ikke der er nogle nye inputs der skal behandles, afsluttes processen
    if df_new.empty:
        print("[FEEDBACK] No new predictions to process.")
        return df_feedback.tail()
    
    # Simulerer en random ActualPrice (true label) for øvelsen
    np.random.seed(42)
    df_new["ActualPrice"] = df_new["PredictedPrice"] * (1 + np.random.normal(0, noise_std, len(df_new)))
    df_new["ActualPrice"] = df_new["ActualPrice"].round(0)

    df_combined = pd.concat([df_feedback, df_new], ignore_index=True)
    df_combined.to_csv(feedback_path, index=False, encoding="utf-8")

    print(f"[FEEDBACK] Added {len(df_new)} new feedback entries → {feedback_path}")
    return df_new.tail()


if __name__ == "__main__":
    result = simulate_feedback()
    if result is not None and not result.empty:
        print("\nLatest feedback entries:")
        print(result)
