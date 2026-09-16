# api/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Önce sadece requirements — kod değişse de bu katman cache'te kalır
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model artefaktları — reproducibility için image'a GÖMÜLÜYOR
COPY ml/rf_model.pkl ./ml/rf_model.pkl
COPY ml/model_metadata.json ./ml/model_metadata.json

# API uygulama kodu
COPY api/main.py api/config.py api/models.py api/database.py ./

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}
