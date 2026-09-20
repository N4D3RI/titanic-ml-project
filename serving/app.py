"""FastAPI app that serves the trained Titanic survival model."""

import json
import os
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict passenger survival on the Titanic using machine learning",
    version="1.0.0",
)

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODEL_PATH = f"{MODELS_DIR}/titanic_model.joblib"
ENCODERS_PATH = f"{MODELS_DIR}/label_encoders.joblib"
METADATA_PATH = f"{MODELS_DIR}/model_metadata.json"

try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print("Model and preprocessing artifacts loaded successfully!")
except Exception as e:  # noqa: BLE001 - the API must still start to report the error
    print(f"Error loading model artifacts: {e}")
    model = None
    label_encoders = None
    metadata = None


class PassengerInput(BaseModel):
    Pclass: int      # 1, 2, or 3
    Sex: str         # 'male' or 'female'
    Age: float       # Age in years
    SibSp: int       # Number of siblings/spouses aboard
    Parch: int       # Number of parents/children aboard
    Fare: float      # Fare paid
    Embarked: str    # 'S', 'C', or 'Q'

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "Age": 25.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 8.05,
                "Embarked": "S",
            }
        }
    }


class PredictionOutput(BaseModel):
    survived: int                     # 0 or 1
    survival_probability: float       # Probability of survival
    prediction_confidence: str        # High, Medium, or Low
    passenger_profile: Dict[str, Any]  # Passenger characteristics


def preprocess_passenger(passenger_data: dict) -> pd.DataFrame:
    """Rebuild the exact feature set the model was trained on."""
    df = pd.DataFrame([passenger_data])

    # Family features (same as training)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Title: simplified for prediction, since no Name is supplied
    if df["Sex"].iloc[0] == "male":
        df["Title"] = "Mr"
    else:
        df["Title"] = "Mrs" if df["Age"].iloc[0] >= 18 else "Miss"

    # Age groups: must match the training bins
    age = df["Age"].iloc[0]
    if age <= 12:
        age_group = "Child"
    elif age <= 18:
        age_group = "Teen"
    elif age <= 35:
        age_group = "Adult"
    elif age <= 60:
        age_group = "Middle"
    else:
        age_group = "Senior"
    df["AgeGroup"] = age_group

    # Fare groups: quartile cut points learned from the training data
    fare = df["Fare"].iloc[0]
    if fare <= 7.91:
        fare_group = "Low"
    elif fare <= 14.45:
        fare_group = "Medium"
    elif fare <= 31.0:
        fare_group = "High"
    else:
        fare_group = "VeryHigh"
    df["FareGroup"] = fare_group

    # Encode categoricals with the encoders saved at training time
    categorical_cols = ["Sex", "Embarked", "Title", "AgeGroup", "FareGroup"]
    for col in categorical_cols:
        try:
            le = label_encoders[col]
            value = str(df[col].iloc[0])
            if value in le.classes_:
                df[col] = le.transform([value])
            else:
                # Unseen category: fall back to the first known class
                df[col] = 0
        except Exception as e:  # noqa: BLE001
            print(f"Error encoding {col}: {e}")
            df[col] = 0

    # Order the columns exactly as the model expects
    feature_names = metadata["feature_names"]
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    return df[feature_names]


@app.get("/")
async def root():
    """API health check."""
    return {
        "message": "Titanic Survival Prediction API",
        "status": "healthy" if model is not None else "error",
        "model_loaded": model is not None,
        "accuracy": metadata["accuracy"] if metadata else None,
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_survival(passenger: PassengerInput):
    """Predict a single passenger's survival probability."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = preprocess_passenger(passenger.model_dump())

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        if probability > 0.8 or probability < 0.2:
            confidence = "High"
        elif probability > 0.6 or probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        family_size = passenger.SibSp + passenger.Parch + 1
        passenger_profile = {
            "class": f"Class {passenger.Pclass}",
            "gender": passenger.Sex,
            "age_group": "Child" if passenger.Age < 18 else "Adult",
            "family_size": family_size,
            "traveling_alone": family_size == 1,
            "fare_level": (
                "High" if passenger.Fare > 31.0
                else "Medium" if passenger.Fare > 14.45
                else "Low"
            ),
            "embarkation_port": {
                "S": "Southampton",
                "C": "Cherbourg",
                "Q": "Queenstown",
            }.get(passenger.Embarked, "Unknown"),
        }

        return PredictionOutput(
            survived=prediction,
            survival_probability=round(probability, 4),
            prediction_confidence=confidence,
            passenger_profile=passenger_profile,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Return metadata about the trained model."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")

    return {
        "model_type": metadata["model_type"],
        "training_date": metadata["training_date"],
        "accuracy": metadata["accuracy"],
        "features": metadata["feature_names"],
        "preprocessing": metadata["preprocessing_info"],
    }


@app.post("/predict-batch")
async def predict_batch(passengers: List[PassengerInput]):
    """Predict survival for multiple passengers in one call."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = []
        for passenger in passengers:
            prediction = await predict_survival(passenger)
            results.append(
                {
                    "passenger": passenger.model_dump(),
                    "prediction": prediction.model_dump(),
                }
            )
        return {"predictions": results, "count": len(results)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
