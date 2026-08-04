"""Allena un classificatore sul dataset Breast Cancer Wisconsin (sklearn)
e salva la pipeline (scaler + modello) in model/model.pkl.

La pipeline include lo StandardScaler insieme al modello proprio per evitare
training-serving skew: chi usa il modello in app.py non deve ricordarsi di
riapplicare lo stesso scaling a mano.
"""

import json
import os

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
RANDOM_STATE = 42


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("Metriche sul test set (20% hold-out, stratificato):")
    for name, value in metrics.items():
        print(f"  {name:10s}: {value:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": list(data.feature_names),
            "target_names": list(data.target_names),
        },
        MODEL_PATH,
    )
    print(f"\nModello salvato in {MODEL_PATH}")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
