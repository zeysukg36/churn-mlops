from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import joblib
import json
import pandas as pd
from datetime import datetime, timezone
from fastapi.concurrency import run_in_threadpool

from config import settings
from models import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    StatsResponse, 
)
from database import log_collection, get_stats_sync

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["pipeline"] = joblib.load(settings.MODEL_PATH)
    with open(settings.MODEL_METADATA_PATH) as f:
        ml_models["metadata"] = json.load(f)
    yield
    ml_models.clear()

app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

def _log_prediction_sync(input_data: dict, output_data: dict):
    log_collection.insert_one({
        "input": input_data,
        "output": output_data,
        "logged_at": datetime.now(timezone.utc),
    })

async def _log_safely(input_data: dict, output_data: dict):
    """Loglama başarısız olsa bile asıl tahmin isteğini asla düşürmez."""
    try:
        await run_in_threadpool(_log_prediction_sync, input_data, output_data)
    except Exception as e:
        print(f"[WARN] Loglama başarısız: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": "pipeline" in ml_models,
        "environment": settings.APP_ENV,
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if "metadata" not in ml_models:
        raise HTTPException(status_code=503, detail="Model metadata henüz yüklenmedi")

    metadata = ml_models["metadata"]
    return ModelInfoResponse(
        version=metadata.get("version", "unknown"),
        trained_at=metadata.get("trained_at", "unknown"),
        metrics=metadata.get("metrics", {}),
        features_expected=len(metadata.get("categorical_columns", [])) + len(metadata.get("numeric_columns", [])),
    )

@app.post("/predict", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    if "pipeline" not in ml_models:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    pipeline = ml_models["pipeline"]
    metadata = ml_models["metadata"]

    input_df = pd.DataFrame([request.model_dump()])
    probability = pipeline.predict_proba(input_df)[0, 1]
    threshold = metadata["best_threshold"]

    response = ChurnPredictionResponse(
        churn_prediction="Yes" if probability >= threshold else "No",
        churn_probability=round(float(probability), 4),
        threshold_used=threshold,
        model_version=metadata.get("version", "1.0.0"),
    )

    await _log_safely(request.model_dump(), response.model_dump())

    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(batch_request: BatchPredictionRequest):
    if "pipeline" not in ml_models:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    pipeline = ml_models["pipeline"]
    metadata = ml_models["metadata"]
    threshold = metadata["best_threshold"]

    records = [r.model_dump() for r in batch_request.records]
    input_df = pd.DataFrame(records)
    probabilities = pipeline.predict_proba(input_df)[:, 1]  # tek çağrıda tüm batch

    results = []
    logs = []
    now = datetime.now(timezone.utc)

    for record, probability in zip(records, probabilities):
        result = ChurnPredictionResponse(
            churn_prediction="Yes" if probability >= threshold else "No",
            churn_probability=round(float(probability), 4),
            threshold_used=threshold,
            model_version=metadata.get("version", "1.0.0"),
        )
        results.append(result)
        logs.append({"input": record, "output": result.model_dump(), "logged_at": now})

    try:
        await run_in_threadpool(log_collection.insert_many, logs)
    except Exception as e:
        print(f"[WARN] Toplu loglama başarısız: {e}")

    return BatchPredictionResponse(results=results)

@app.get("/stats", response_model=StatsResponse)
async def get_stats(limit: int = Query(default=100, ge=1, le=1000)):
    try:
        stats = await run_in_threadpool(get_stats_sync, limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"İstatistikler alınamadı: {e}")
    return StatsResponse(**stats)
