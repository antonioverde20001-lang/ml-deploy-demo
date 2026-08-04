"""API FastAPI che serve il modello allenato da train.py."""

import time
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_PATH = "model/model.pkl"

PREDICTION_COUNT = Counter(
    "prediction_requests_total", "Numero totale di richieste a /predict", ["outcome"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Latenza delle richieste a /predict"
)

_artifact = None


def load_model():
    global _artifact
    _artifact = joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Breast Cancer Classifier API", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description=(
            "30 feature numeriche, nello stesso ordine di "
            "sklearn.datasets.load_breast_cancer().feature_names"
        ),
    )


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _artifact is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    expected = len(_artifact["feature_names"])
    if len(request.features) != expected:
        PREDICTION_COUNT.labels(outcome="error").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Attese {expected} feature, ricevute {len(request.features)}",
        )

    start = time.perf_counter()
    pipeline = _artifact["pipeline"]
    proba = pipeline.predict_proba([request.features])[0]
    prediction = int(proba[1] >= 0.5)
    PREDICTION_LATENCY.observe(time.perf_counter() - start)
    PREDICTION_COUNT.labels(outcome="success").inc()

    return PredictResponse(
        prediction=prediction,
        label=_artifact["target_names"][prediction],
        probability=float(proba[prediction]),
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
