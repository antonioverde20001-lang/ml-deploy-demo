from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app import app, load_model

load_model()
client = TestClient(app)


def real_sample():
    data = load_breast_cancer()
    return data.data[0].tolist()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_sample():
    response = client.post("/predict", json={"features": real_sample()})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("malignant", "benign")
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_wrong_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"prediction_requests_total" in response.content
