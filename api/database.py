from pymongo import MongoClient
from config import settings

client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[settings.DB_NAME]
log_collection = db[settings.LOG_COLLECTION_NAME]

def get_stats_sync(limit: int = 100) -> dict:
    total = log_collection.count_documents({})
    recent = list(log_collection.find().sort("logged_at", -1).limit(limit))
    window_size = len(recent)
    
    if window_size == 0:
        return {
            "total_logged": total,
            "window_size": 0,
            "avg_churn_probability": 0.0,
            "churn_rate": 0.0,
            "probability_distribution": [],
        }
        
    probs = [r["output"]["churn_probability"] for r in recent]
    yes_count = sum(1 for r in recent if r["output"]["churn_prediction"] == "Yes")
    
    bucket_edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    distribution = [
        {
            "range": f"{low:.1f}-{min(high, 1.0):.1f}",
            "count": sum(1 for p in probs if low <= p < high),
        }
        for low, high in bucket_edges
    ]
    
    return {
        "total_logged": total,
        "window_size": window_size,
        "avg_churn_probability": round(sum(probs) / window_size, 4),
        "churn_rate": round(yes_count / window_size, 4),
        "probability_distribution": distribution,
    }