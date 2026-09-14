# Build notes: what each step does and why

Written so you can rebuild this project from scratch. Each step is: what it is, the code, then what to actually do.

The one deviation from the lesson: the lesson has you run `docker build` and `docker push` from Docker Desktop. Here, GitHub Actions does that on a cloud runner instead. Everything else is identical, and the deviation is explained in Step 9.

---

## The shape of the whole thing

Two containers, two jobs.

**Training container** is a workbench. Jupyter, pandas, scikit-learn, matplotlib. You use it once, interactively, to produce a trained model. It never ships to anyone.

**Serving container** is the product. Python, FastAPI, the trained model file. No training code, no Jupyter, no dataset. It ships to Docker Hub and anyone can run it.

The handoff between them is a folder of **artifacts**: the trained model plus everything needed to reproduce the exact preprocessing it was trained on.

```
training/  →  models/*.joblib, model_metadata.json  →  serving/
(workbench)          (the artifacts)                   (product)
```

That split is the actual lesson of the project. Training environments are big and messy. Serving environments must be small and boring. The training image is roughly 2 GB. The serving image is a few hundred MB.

---

## Step 1: Project setup

```
titanic-ml-project/
├── training/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train_model.py
│   ├── data/titanic.csv
│   └── models/
├── serving/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── models/
└── README.md
```

The two folders each have their own `Dockerfile` and `requirements.txt`. That is deliberate, not duplication. A Docker build has a **build context**: the folder you point it at. `docker build ./serving` can only see files inside `serving/`. It cannot reach up and grab `../training/data/titanic.csv`. Separate folders make the boundary physical, so the dataset and the Jupyter dependencies cannot leak into the shipped image by accident.

**Dataset.** Kaggle's download needs a login. The same 891-row file is mirrored publicly:

```bash
curl -sSL -o training/data/titanic.csv \
  https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

Columns: `PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked`.

**First steps:** make the folders, download the CSV, `head -2` it to confirm the columns match the list above.

---

## Step 2: Training environment

`training/requirements.txt` pins exact versions:

```
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
jupyterlab==4.0.5
joblib==1.3.1
```

**Why `==` and not `>=`.** This is the single most important line in the project, and it bit me while building it. A scikit-learn model saved with `joblib` is a **pickle**: a serialized Python object, not a portable format. Unpickling it requires a compatible scikit-learn version. My machine had scikit-learn 1.8; training there and then serving under the pinned 1.3.0 would have produced either a loud version warning or a silently wrong model. I had to build a virtualenv on the exact pinned versions and retrain.

Rule: **the scikit-learn version that trains the model and the one that loads it must match.** Compare `training/requirements.txt` and `serving/requirements.txt` in this repo — `scikit-learn`, `pandas`, `numpy`, and `joblib` are pinned identically in both. That is not an accident.

`training/Dockerfile`:

```dockerfile
FROM jupyter/base-notebook:x86_64-python-3.11

USER root

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /home/jovyan/work/data && \
    mkdir -p /home/jovyan/work/models && \
    chown -R jovyan:users /home/jovyan/work

USER jovyan
WORKDIR /home/jovyan/work
```

Four things worth understanding here:

- `FROM jupyter/base-notebook` starts from an image that already has Jupyter configured. You inherit a working setup instead of assembling one.
- `USER root` then `USER jovyan` — you need root to install packages, and then you drop back to the unprivileged user. Containers that run as root are a security problem, and the Jupyter base image expects `jovyan` to own the work directory.
- `--no-cache-dir` keeps pip's download cache out of the image layer. Smaller image, no benefit lost.
- `chown -R jovyan:users` matters because the next step mounts your host folder into `/home/jovyan/work`. Wrong ownership means permission errors when the container tries to write model files.

---

## Step 3: The training script

Full code in `training/train_model.py`. Five functions, run in order by `main()`.

**`load_and_explore_data()`** — reads the CSV, prints shape, `.info()`, missing-value counts, and survival rate grouped by sex and by class. Purely diagnostic, but the two groupbys tell you what the model will discover: 74% of women survived versus 19% of men, and 63% of first class versus 24% of third.

**`preprocess_data(df)`** — imputation, then feature engineering.

```python
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["Fare"] = data["Fare"].fillna(data["Fare"].median())
```

Median for numbers because it survives outliers; mode for the categorical port. Age has 177 missing values out of 891 — dropping those rows would throw away 20% of the data, so imputing is the right call.

Then five engineered features, which are where the accuracy above baseline actually comes from:

```python
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
data["AgeGroup"] = pd.cut(data["Age"], bins=[0,12,18,35,60,100],
                          labels=["Child","Teen","Adult","Middle","Senior"])
data["FareGroup"] = pd.qcut(data["Fare"], q=4,
                            labels=["Low","Medium","High","VeryHigh"])
```

`Title` is the clever one. The regex pulls `Mr`, `Mrs`, `Miss`, `Master`, `Dr`, `Rev` out of strings like `"Braund, Mr. Owen Harris"`. It ends up the third most important feature, because it encodes sex, rough age, marital status, and social rank in a single token. Rare titles get collapsed into `Other` so the model does not try to learn from one `Countess`.

Note the difference between `pd.cut` and `pd.qcut`: `cut` splits on the boundaries you name, `qcut` splits into equal-sized groups. Age gets meaningful human boundaries; fare gets quartiles, because "expensive" is only definable relative to the other fares.

**`encode_features(data)`** — scikit-learn needs numbers, not the string `"female"`.

```python
for col in ["Sex","Embarked","Title","AgeGroup","FareGroup"]:
    le = LabelEncoder()
    model_data[col] = le.fit_transform(model_data[col].astype(str))
    label_encoders[col] = le
```

**Keep the encoders.** This is the part beginners drop. `LabelEncoder` assigns numbers in alphabetical order of whatever it saw during `fit`. `female` → 0, `male` → 1. If the serving app creates a fresh encoder, the mapping can come out different, and the model silently reads every passenger's sex backwards. Predictions stay plausible-looking and are wrong. So each fitted encoder is saved and reloaded at serve time.

**`train_model(...)`** — split, fit, evaluate.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=10,
    min_samples_split=5, min_samples_leaf=2, random_state=42)
```

- `random_state=42` makes the run reproducible. Same seed, same split, same trees, same accuracy.
- `stratify=y` keeps the survived/died ratio the same in train and test. Without it, a lucky or unlucky split moves accuracy by a couple of points for no real reason.
- `max_depth=10`, `min_samples_leaf=2` are the guardrails against overfitting. An unconstrained forest memorizes the 712 training rows and generalizes worse.

Result: **0.8156 accuracy**. For context, "predict everyone died" scores about 0.62 and "predict women live, men die" scores about 0.79. So the model earns roughly two and a half points over the naive sex rule. Knowing your baselines is how you tell a good score from a meaningless one.

It also writes `feature_importance.png` and `confusion_matrix.png`. One gotcha: matplotlib in a container has no display, so the script sets the headless backend before importing pyplot:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Without that line the script crashes inside Docker while working fine on a laptop.

**`save_model_artifacts(...)`** — writes four files:

| File | Why it exists |
| --- | --- |
| `titanic_model.joblib` | the trained forest |
| `label_encoders.joblib` | the exact category→number mappings |
| `model_metadata.json` | accuracy, training date, **feature order**, preprocessing notes |
| `sample_input.json` | a known-good request body for testing |

`feature_names` inside the metadata is load-bearing. A scikit-learn model takes a plain array and infers nothing from column names — it assumes column 3 is whatever column 3 was during training. The serving app reads this list and reorders its DataFrame to match, which is what prevents a whole category of silent wrongness.

---

## Step 4: Run the training

```bash
cd training
docker build -t titanic-training .
docker run --name titanic-trainer -p 8888:8888 -v "$(pwd):/home/jovyan/work" titanic-training
docker exec titanic-trainer python train_model.py
```

The flags:

- `-t titanic-training` names the image.
- `-p 8888:8888` maps container port 8888 to host port 8888, so Jupyter is reachable at `localhost:8888`.
- `-v "$(pwd):/home/jovyan/work"` is the important one. It **bind-mounts** your current folder into the container. Without it, the model files would be written inside the container's filesystem and vanish when the container is removed. With it, they appear in `training/models/` on your actual disk.
- `docker exec` runs a command inside an already-running container.

**First steps:** build, run, open `localhost:8888`, take the token from the container logs, then either run the script from a notebook cell with `exec(open('train_model.py').read())` or just `docker exec` it. Confirm `training/models/` has six files afterward.

---

## Step 5: The serving API

Full code in `serving/app.py`. FastAPI gives you three things for free: request validation, automatic interactive docs at `/docs`, and JSON serialization.

**Load once, at import.** The model loads at module level, not inside the request handler, so it is read from disk one time when the container starts rather than on every request. The `try/except` means a broken artifact yields an API that starts and honestly reports `"model_loaded": false`, instead of a container that crash-loops with no explanation.

**Pydantic defines the contract.**

```python
class PassengerInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
```

Anything that does not fit gets rejected with a `422` and a message naming the bad field, before your code runs. Free input validation.

**`preprocess_passenger()` is where projects break.** The API receives 7 raw fields. The model expects 12 encoded ones. This function rebuilds the missing 5 — and it must do it *identically to training*. Same age boundaries, same fare cut points, same encoders, same column order.

This is called **training/serving skew**, and it is the most common way a deployed model quietly goes wrong. The model will happily accept mismatched features and return a confident number. Nothing errors. Three specific defenses in the code:

1. Fare boundaries `7.91 / 14.45 / 31.0` are the quartile cut points `pd.qcut` found in the training data. Hardcoding the numbers the training run produced is what keeps the two paths aligned.
2. Unseen categories fall back to a known class instead of throwing.
3. The final line reorders columns using `metadata["feature_names"]`.

There is one honest simplification: the API has no `Name` field, so `Title` is inferred from sex and age (`male` → `Mr`, female 18+ → `Mrs`, else `Miss`). That loses `Master` and `Dr`. It is a real accuracy cost, accepted so callers do not have to invent a name. If you wanted to close it, add an optional `Name` field and run the same regex.

**Endpoints:** `/` health, `/model-info` metadata, `/predict` single, `/predict-batch` list, `/docs` free Swagger UI.

**First steps:** before touching Docker, run it bare — `uvicorn app:app --port 8000` from inside `serving/` — and curl the endpoints. Debugging Python is much easier outside a container than inside one.

---

## Step 6: Copy the artifacts

```bash
cp training/models/* serving/models/
```

One line, easy to forget, and forgetting it produces an image that builds fine and then reports `"model_loaded": false`. Because of the build-context rule from Step 1, `serving/Dockerfile` cannot reach into `training/`, so the files must physically exist under `serving/` before the build.

---

## Step 7: The serving Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Layer caching and why `COPY requirements.txt` comes first.** Docker caches each instruction as a layer and reuses it if nothing that feeds it changed. Requirements are copied and installed *before* the app code is copied, so editing `app.py` reuses the cached dependency layer and rebuilds in seconds. Copy everything at once and every one-character change triggers a full reinstall of scikit-learn.

**`python:3.11-slim`** instead of the full `python:3.11`: same Python, far less OS. Smaller image, fewer packages to carry vulnerabilities.

**`--host 0.0.0.0`** is mandatory and catches people out. The default `127.0.0.1` means "loopback only" — inside a container that is the container's own loopback, so `-p 8000:8000` maps to a port nothing is listening on from outside. Binding `0.0.0.0` accepts connections from the host.

**The `curl` install** is mine, not in the lesson. The `HEALTHCHECK` line invokes `curl`, and `python:3.11-slim` does not include it, so the healthcheck would report unhealthy forever. Catching that kind of thing is what the CI smoke test in Step 9 is for.

**`CMD` in exec form** (a JSON array, not a shell string) so the process runs as PID 1 and receives `SIGTERM` directly. Shell form wraps it in `/bin/sh -c`, which swallows the signal and makes the container take ten seconds to stop.

---

## Step 8: Build and test locally

```bash
cd serving
docker build -t titanic-survival-api .
docker run --name titanic-api -p 8000:8000 titanic-survival-api
```

Then verify, in this order:

```bash
curl http://localhost:8000/                # expect "model_loaded": true
open http://localhost:8000/docs            # Swagger UI

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Pclass":1,"Sex":"female","Age":30.0,"SibSp":0,"Parch":0,"Fare":100.0,"Embarked":"C"}'
```

**Sanity-check the numbers, do not just check for a 200.** First class woman, high fare should come back near 0.95. Third class man, cheap fare should come back near 0.08. Those came back 0.9647 and 0.0760. If both had returned something like 0.5, or if the woman scored lower than the man, that is the encoder-mismatch failure from Step 3 showing up. A 200 response proves nothing about correctness.

Also confirm a malformed body returns `422`, which proves Pydantic validation is live.

---

## Step 9: Push to Docker Hub — the GitHub Actions version

The lesson's local version:

```bash
docker login
docker tag titanic-survival-api yourusername/titanic-survival-api:v1.0
docker push yourusername/titanic-survival-api:v1.0
```

An image name is `registry/username/repository:tag`. Docker Hub is the default registry, so `yourusername/titanic-survival-api:v1.0` is enough. `docker tag` does not copy anything — it just adds a second name pointing at the same image.

This repo does it in CI instead, in `.github/workflows/docker-publish.yml`. What changes and why:

**Credentials become secrets.** `docker login` on a laptop stores a token in `~/.docker/config.json`. In CI there is no laptop, so `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` live as GitHub repository secrets and are injected at runtime. They are masked in logs and never in the repo. Use a Docker Hub **personal access token**, never the account password — a token is scoped and revocable on its own.

**The trigger.**

```yaml
on:
  push:
    branches: [main]
    paths: ['serving/**', '.github/workflows/docker-publish.yml']
  workflow_dispatch:
```

Only rebuild when the thing being shipped changes. `workflow_dispatch` adds a manual Run button.

**Build, test, then push — in that order.** The workflow builds with `load: true, push: false` so the image stays on the runner, starts a container, waits for health, fires a real prediction, and only pushes if all of that passes. A broken image never reaches Docker Hub. This is the actual advantage over pushing from a laptop: the verification is enforced rather than remembered.

**Cache across runs.** `cache-from: type=gha` stores the layer cache in GitHub's cache service, so a runner that starts from a clean disk still skips reinstalling scikit-learn.

**Tags.** `docker/metadata-action` emits `latest`, `v1.0`, and a short commit SHA. The SHA tag is the useful one in practice: `latest` moves, but `sha-a1b2c3d` always points at exactly one commit, so you can tie a running container back to the code that produced it.

**What this does not cover:** the training image. It is a personal workbench, so it is not published. If you wanted, a second job could retrain on changes to `training/` and commit the updated artifacts.

---

## Step 10: Documentation

See `README.md`. The bar is that a stranger can go from the repo to a running prediction without asking you anything. That means: the one `docker run` line at the very top, a copy-pasteable curl with real values, the response shape, and the valid values for every input field.

State the accuracy, and state what it was measured on. "81.6% on a held-out 20% test split" is a claim someone can check. "81.6% accurate" is not.

## Step 11: Submit

Submit the Docker Hub link, `https://hub.docker.com/r/naderi11/titanic-survival-api`. Before submitting, pull the published image on a machine that never built it and run it. That is the only real proof of the last checklist item.

---

## Checklist

| Item | Where it is verified |
| --- | --- |
| Training environment created and working | `training/Dockerfile`; training ran and produced artifacts |
| Model artifacts saved properly | six files in `training/models/`, metadata carries feature order |
| Serving API created and tested | all five endpoints exercised, responses verified |
| Container built and runs locally | Dockerfile validated; CI builds and runs it before pushing |
| Container pushed to Docker Hub | the workflow's push step, after secrets are set |
| README documentation complete | `README.md` |
| API tested with sample data | 0.9647 and 0.0760 on the two reference passengers; 422 on bad input |
| Container can be pulled and run by others | public image, public repo, one-command run |

---

## The five things worth actually remembering

1. **Pin your versions, especially scikit-learn.** A joblib file is a pickle, and a pickle is version-coupled. Train and serve on the same version or the model may not load.
2. **Save the encoders, not just the model.** The model is half the artifact. The preprocessing that produced its inputs is the other half.
3. **Training/serving skew fails silently.** Mismatched feature engineering does not raise; it returns confident, wrong numbers. Verify predictions against cases where you know the answer.
4. **Order Dockerfile instructions from least to most frequently changed.** Dependencies before code. It is the difference between a 5-second and a 5-minute rebuild.
5. **Build, test, then push.** Making the smoke test a pipeline step rather than a habit is the whole reason to move the build off your laptop.
