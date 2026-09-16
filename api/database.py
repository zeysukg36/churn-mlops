from pymongo import MongoClient
from config import settings

client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[settings.DB_NAME]
log_collection = db[settings.LOG_COLLECTION_NAME]
