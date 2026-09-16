import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../ml/rf_model.pkl")
    MODEL_METADATA_PATH: str = os.getenv("MODEL_METADATA_PATH", "../ml/model_metadata.json")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27018")
    DB_NAME: str = os.getenv("DB_NAME", "churn_predictions")
    LOG_COLLECTION_NAME: str = os.getenv("LOG_COLLECTION_NAME", "prediction_logs")

settings = Settings()
