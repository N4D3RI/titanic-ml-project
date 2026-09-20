# Titanic Survival Prediction API

A machine learning API that predicts passenger survival on the Titanic. A Random Forest model is trained from raw CSV, packaged with its preprocessing artifacts into a FastAPI service, containerized, and published to Docker Hub by GitHub Actions.

**Model accuracy: 81.6%** on a held-out 20% test split (RandomForestClassifier, 12 features).

## Run it with one command

```bash
docker run -p 8000:8000 naderi11/titanic-survival-api:latest
```

Then open http://localhost:8000/docs for the interactive API documentation.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Health check: reports whether the model loaded, and its accuracy |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/model-info` | Model type, training date, accuracy, feature list, preprocessing notes |
| POST | `/predict` | Predict survival for one passenger |
| POST | `/predict-batch` | Predict survival for a list of passengers |

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female",
    "Age": 30.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100.0,
    "Embarked": "C"
  }'
```

### Example response

```json
{
  "survived": 1,
  "survival_probability": 0.9647,
  "prediction_confidence": "High",
  "passenger_profile": {
    "class": "Class 1",
    "gender": "female",
    "age_group": "Adult",
    "family_size": 1,
    "traveling_alone": true,
    "fare_level": "High",
    "embarkation_port": "Cherbourg"
  }
}
```

### Input fields

| Field | Type | Values |
| --- | --- | --- |
| `Pclass` | int | 1, 2, or 3 |
| `Sex` | str | `male` or `female` |
| `Age` | float | Age in years |
| `SibSp` | int | Siblings and spouses aboard |
| `Parch` | int | Parents and children aboard |
| `Fare` | float | Ticket fare |
| `Embarked` | str | `S` (Southampton), `C` (Cherbourg), or `Q` (Queenstown) |

## Project structure

```
titanic-ml-project/
├── .github/workflows/
│   └── docker-publish.yml     # builds the image and pushes it to Docker Hub
├── training/
│   ├── Dockerfile             # Jupyter environment for experimentation
│   ├── requirements.txt
│   ├── train_model.py         # full training pipeline
│   ├── data/titanic.csv       # raw dataset (891 passengers)
│   └── models/                # artifacts produced by training
├── serving/
│   ├── Dockerfile             # the published image
│   ├── requirements.txt
│   ├── app.py                 # FastAPI application
│   └── models/                # artifacts copied from training
├── docker-compose.yml
├── NOTES.md                   # walkthrough of every step and why it works
└── README.md
```

## The model

A `RandomForestClassifier` (100 trees, max depth 10) trained on 12 features:

**Raw:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
**Engineered:** FamilySize, IsAlone, Title (extracted from the passenger's name), AgeGroup, FareGroup

Missing values are imputed before feature engineering: Age and Fare with the median, Embarked with the mode. Categorical features are label-encoded, and the fitted encoders are saved alongside the model so that serving applies the exact same encoding.

Most important features, by Gini importance: Sex (0.28), Fare (0.17), Title (0.13), Age (0.12), Pclass (0.09).

## Reproducing the training

```bash
cd training
docker build -t titanic-training .
docker run --name titanic-trainer -p 8888:8888 -v "$(pwd):/home/jovyan/work" titanic-training
docker exec titanic-trainer python train_model.py
```

Artifacts land in `training/models/`. Copy them into the serving image with:

```bash
cp training/models/* serving/models/
```

## Building the serving image

The image is built and published automatically by GitHub Actions on every push to `main` that touches `serving/`. The workflow builds the image, runs a smoke test against a live container (health check plus a real prediction), and only then pushes to Docker Hub.

To build it locally instead:

```bash
cd serving
docker build -t titanic-survival-api .
docker run -p 8000:8000 titanic-survival-api
```

Or with Compose, from the project root:

```bash
docker compose up --build
```

## Continuous delivery

Two repository secrets drive the publish step:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (a Docker Hub personal access token, not the account password)

Published tags: `latest`, `v1.0`, and a short commit SHA for traceability.

## Dataset

Titanic passenger data, 891 rows. Source: [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/data).
