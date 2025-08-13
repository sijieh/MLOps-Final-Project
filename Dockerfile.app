# FastAPI dashboard that proxies to MLflow serving
FROM python:3.11-slim
WORKDIR /workspace

RUN pip install --no-cache-dir fastapi uvicorn[standard] pandas requests python-dotenv

COPY app.py ./app.py
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
