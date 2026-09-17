# 🔮 Churn Prediction API

An end-to-end MLOps service for Telco customer churn prediction — a RandomForest-based model served with FastAPI, containerized with Docker, and observable via MongoDB.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square)

**Live Demo:** https://churn-prediction-api-jq48.onrender.com/docs
*(Hosted on a free tier — spins down after 15 minutes of inactivity, so the first request may take up to ~50 seconds.)*

---

## ✨ Features

- 🎯 Single and batch (`/predict/batch`) churn prediction
- 📦 Encoding and model are serialized together in a single `scikit-learn Pipeline` (no column-mismatch risk in production)
- 🗄️ Every prediction request is logged to MongoDB asynchronously and fault-tolerantly
- 📊 `/stats` for a probability-distribution and churn-rate summary of logged predictions (lightweight drift observability)
- 🔍 `/model/info` for the active model version and validated test metrics
- 🐳 Docker + Docker Compose for a one-command local stack (API + MongoDB)
- ✅ Schema and endpoint tests with pytest

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | scikit-learn (RandomForestClassifier), Pipeline + ColumnTransformer |
| API | Python 3.12, FastAPI, Pydantic v2 |
| Database | MongoDB (local: Docker container, production: MongoDB Atlas M0) |
| Containerization | Docker & Docker Compose |
| Deployment | Render (Web Service, Docker runtime) |
| Testing | pytest, FastAPI TestClient |

## 📊 Model Performance

| Metric | Value |
|---|---|
| Version | `1.1.1` |
| F1-Score (test set, leakage-free) | `0.5813` |
| ROC-AUC (test set) | `0.8051` |
| Decision Threshold | `0.50` (selected from the validation set) |
| Training Data | Telco Customer Churn (IBM/Kaggle), 7,043 records |

## 📂 Project Structure
churn-mlops/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── README.md
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── ml/
│ ├── baseline_pipeline.py # Training script (Pipeline + leakage-free threshold tuning)
│ ├── rf_model.pkl # Serialized pipeline (encoding included)
│ └── model_metadata.json # Version, threshold, metrics
└── api/
├── Dockerfile
├── requirements.txt
├── config.py # Environment variable management
├── database.py # MongoDB connection + stats queries
├── models.py # Pydantic schemas
├── main.py # FastAPI application and endpoints
└── tests/ # pytest: schema + endpoint tests


## 🚀 Local Setup

### Requirements
- [Docker](https://www.docker.com/) & Docker Compose

### Steps

```bash
git clone https://github.com/zeysukg36/churn-mlops.git
cd churn-mlops
cp .env.example .env
cp api/.env.example api/.env
docker compose up --build
```

### Access Points

| Service | URL |
|---|---|
| Swagger Docs | http://localhost:8001/docs |
| Health Check | http://localhost:8001/health |

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service and model health check |
| `GET` | `/model/info` | Active model version and metrics |
| `POST` | `/predict` | Single churn prediction |
| `POST` | `/predict/batch` | Batch churn prediction |
| `GET` | `/stats?limit=100` | Summary statistics of logged predictions |

For interactive testing of all endpoints, use the Swagger UI at `/docs` — every endpoint comes with a ready-to-use example payload.

## 🔧 Environment Variables

| Variable | Description |
|---|---|
| `MODEL_PATH` | Path to the serialized pipeline file |
| `MODEL_METADATA_PATH` | Path to the model metadata JSON file |
| `MONGO_URI` | MongoDB connection string (local or Atlas) |
| `APP_ENV` | `development` / `production` |

## 🔄 Retraining Strategy

This project does **not** implement an automated retraining pipeline — but here's how it's designed to work in a system headed for real production use:

**Triggers (when should retraining happen?)**
- Time-based: e.g. once a month, since data distributions drift slowly over time
- Signal-based: the `probability_distribution` in `/stats` diverging noticeably from the training-data distribution (a real drift-detection system is out of scope for this project — noted as a v2.0 item)
- Performance-based: once new labeled data arrives, the model's real-world F1/AUC dropping meaningfully below the values recorded in `model_metadata.json`

**Process**
1. `ml/baseline_pipeline.py` is re-run on the new data (the existing leakage-free train/val/test discipline is preserved)
2. The new model's test metrics are compared against the **current production model's** metrics in `model_metadata.json`
3. If the new model is better: the `version` field in `model_metadata.json` is bumped (e.g. `1.1.1` → `1.2.0`) and committed along with the new `rf_model.pkl`
4. `git push` → Render automatically builds and deploys the new image
5. After deployment, `/model/info` is checked to confirm the new version is live

**What's missing here (for v2.0)**
- Wiring the steps above into a scheduled (cron) GitHub Actions workflow
- **Automatic rejection** of a new model if its performance is worse (this decision is currently manual)
- A real statistical drift test (e.g. Kolmogorov-Smirnov) — the current `/stats` is purely an observational summary

## 🗺️ Roadmap

- [ ] Run `pytest` automatically on every push via GitHub Actions
- [ ] Wire `/stats` up to a real statistical drift test

## 📄 License

This project was built for personal/portfolio purposes.

---

## 👩‍💻 Author

**Zeynep Karagöz**
Management Information Systems (MIS) Student

- LinkedIn: [linkedin.com/in/zeynepkaragozz](https://www.linkedin.com/in/zeynepkaragozz)
- Email: [zeynepkaragoz3637@gmail.com](mailto:zeynepkaragoz3637@gmail.com)
- GitHub: [github.com/zeysukg36](https://github.com/zeysukg36)