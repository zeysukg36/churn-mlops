from pydantic import BaseModel, Field
from typing import Literal

class ChurnPredictionRequest(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(..., description="0 = Hayır, 1 = Evet (65 yaş üstü)")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Müşterinin şirkette kaldığı ay sayısı")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0, description="Aylık ücret (USD)")
    TotalCharges: float = Field(..., ge=0, description="Bugüne kadar tahsil edilen toplam ücret (USD)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35, "TotalCharges": 844.20
            }
        }
    }   


class ChurnPredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    threshold_used: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    records: list[ChurnPredictionRequest]


class BatchPredictionResponse(BaseModel):
    results: list[ChurnPredictionResponse]


class ModelInfoResponse(BaseModel):
    version: str
    trained_at: str
    metrics: dict
    features_expected: int

class ProbabilityBucket(BaseModel):
    range: str
    count: int

class StatsResponse(BaseModel):
    total_logged: int
    window_size: int
    avg_churn_probability: float
    churn_rate: float
    probability_distribution: list[ProbabilityBucket]