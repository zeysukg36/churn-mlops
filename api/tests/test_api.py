from fastapi.testclient import TestClient
from main import app

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model/info")
        assert response.status_code == 200
        assert "version" in response.json()

def test_predict_valid_payload(valid_payload):
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        body = response.json()
        assert body["churn_prediction"] in ["Yes", "No"]
        assert 0.0 <= body["churn_probability"] <= 1.0

def test_predict_invalid_category_returns_422(valid_payload):
    bad_payload = {**valid_payload, "PaymentMethod": "Cash"}  # tanımlı olmayan kategori
    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422
