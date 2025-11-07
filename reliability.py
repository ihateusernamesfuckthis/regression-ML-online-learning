def assess_reliability(completeness: float) -> dict:

    """Konverterer skema 'completeness' - alstå hvor mange features der er udfyldt - til et label og en score.

    Funktioen skal kaldes inde i api/app.py

    args:
        completeness (float): Fraktionen af modellens features til stede i inputtet.

    returns:
        dict: {
            "score": int (0-100)
            "label": str ("low", "medium", "high"),
            "message": str (læsbar forklaring) 
        }
    """

    completeness = max (0.0, min(1.0, completeness))

    score = round(completeness * 100)

    if completeness > 0.9:
        label = "high"
        message = "Inputtet indeholder størstedelen af model features. Prediction er robust"
    elif completeness > 0.6:
        label = "Medium"
        message = "Inputtet indholder delvist modellens features. Prediction er moderat robust"
    else:
        label = "Low"
        message = "Inputtet er mangelfuldt ift modellens features. Prediction er upålidelig"
    
    return {
        "score": score,
        "label": label,
        "message": message
    }