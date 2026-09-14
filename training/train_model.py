"""Titanic survival prediction: full training pipeline.

Loads the raw Titanic CSV, engineers features, trains a RandomForest,
and writes every artifact the serving API needs into models/.
"""

import json
import os
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: no display in a container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.environ.get("TITANIC_CSV", "data/titanic.csv")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")


def load_and_explore_data():
    """Load and explore the Titanic dataset."""
    print("Loading Titanic dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSurvival rate by gender:")
    print(df.groupby("Sex")["Survived"].mean())
    print("\nSurvival rate by class:")
    print(df.groupby("Pclass")["Survived"].mean())

    return df


def preprocess_data(df):
    """Clean the data and engineer features."""
    print("\nPreprocessing data...")
    data = df.copy()

    # Handle missing values
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    # Family features
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    # Title extracted from the passenger name
    data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    data["Title"] = data["Title"].replace(
        [
            "Lady", "Countess", "Capt", "Col", "Don", "Dr",
            "Major", "Rev", "Sir", "Jonkheer", "Dona",
        ],
        "Other",
    )
    data["Title"] = data["Title"].replace("Mlle", "Miss")
    data["Title"] = data["Title"].replace("Ms", "Miss")
    data["Title"] = data["Title"].replace("Mme", "Mrs")
    data["Title"] = data["Title"].fillna("Other")

    # Binned age and fare
    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Adult", "Middle", "Senior"],
    )
    data["FareGroup"] = pd.qcut(
        data["Fare"], q=4, labels=["Low", "Medium", "High", "VeryHigh"]
    )

    print(f"Preprocessed data shape: {data.shape}")
    print("\nNew features created:")
    print("- FamilySize: SibSp + Parch + 1")
    print("- IsAlone: Whether passenger traveled alone")
    print("- Title: Title extracted from name")
    print("- AgeGroup: Age categories")
    print("- FareGroup: Fare categories")

    return data


def encode_features(data):
    """Label-encode the categorical features and keep the encoders."""
    print("\nEncoding categorical features...")

    features = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        "FamilySize", "IsAlone", "Title", "AgeGroup", "FareGroup",
    ]
    model_data = data[features + ["Survived"]].copy()

    label_encoders = {}
    categorical_cols = ["Sex", "Embarked", "Title", "AgeGroup", "FareGroup"]

    for col in categorical_cols:
        le = LabelEncoder()
        model_data[col] = le.fit_transform(model_data[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {list(le.classes_)}")

    return model_data, label_encoders, features


def train_model(model_data, features):
    """Train and evaluate the Random Forest classifier."""
    print("\nTraining Random Forest model...")

    X = model_data.drop("Survived", axis=1)
    y = model_data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    os.makedirs(MODELS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))
    top10 = feature_importance.head(10)
    plt.barh(top10["feature"], top10["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    return model, accuracy, list(X.columns)


def save_model_artifacts(model, label_encoders, feature_names, accuracy):
    """Write the model, encoders, metadata, and a sample input to models/."""
    print("\nSaving model artifacts...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_filename = f"{MODELS_DIR}/titanic_model.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved to: {model_filename}")

    encoders_filename = f"{MODELS_DIR}/label_encoders.joblib"
    joblib.dump(label_encoders, encoders_filename)
    print(f"Label encoders saved to: {encoders_filename}")

    metadata = {
        "model_type": "RandomForestClassifier",
        "training_date": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "feature_names": feature_names,
        "model_parameters": model.get_params(),
        "preprocessing_info": {
            "age_fillna": "median",
            "embarked_fillna": "mode",
            "fare_fillna": "median",
            "new_features": [
                "FamilySize", "IsAlone", "Title", "AgeGroup", "FareGroup",
            ],
        },
    }
    metadata_filename = f"{MODELS_DIR}/model_metadata.json"
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Model metadata saved to: {metadata_filename}")

    sample_input = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 8.05,
        "Embarked": "S",
    }
    sample_filename = f"{MODELS_DIR}/sample_input.json"
    with open(sample_filename, "w") as f:
        json.dump(sample_input, f, indent=2)
    print(f"Sample input saved to: {sample_filename}")

    print("\nAll model artifacts saved successfully!")
    return metadata


def main():
    """Main training pipeline."""
    print("Starting Titanic Survival Prediction Model Training")
    print("=" * 60)

    df = load_and_explore_data()
    data = preprocess_data(df)
    model_data, label_encoders, features = encode_features(data)
    model, accuracy, feature_names = train_model(model_data, features)
    save_model_artifacts(model, label_encoders, feature_names, accuracy)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Final model accuracy: {accuracy:.4f}")
    print(f"Model artifacts saved to '{MODELS_DIR}/' directory")
    print("Ready for deployment!")


if __name__ == "__main__":
    main()
