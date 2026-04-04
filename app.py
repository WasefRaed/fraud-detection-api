import time
import mlflow
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from predict import pipeline

# ── MLflow setup ──────────────────────────────────────────────
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("fraud-detection")

app = FastAPI(title="Fraud Detection API")
app.mount("/static", StaticFiles(directory="frontend"), name="static")


class Transaction(BaseModel):
    Time: float;   Amount: float
    V1:  float;    V2:  float;   V3:  float;   V4:  float;   V5:  float
    V6:  float;    V7:  float;   V8:  float;   V9:  float;   V10: float
    V11: float;    V12: float;   V13: float;   V14: float;   V15: float
    V16: float;    V17: float;   V18: float;   V19: float;   V20: float
    V21: float;    V22: float;   V23: float;   V24: float;   V25: float
    V26: float;    V27: float;   V28: float


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/predict")
def predict(tx: Transaction):
    start = time.time()
    result = pipeline.predict(tx.dict())
    duration = round(time.time() - start, 4)

    # Log every prediction to MLflow
    with mlflow.start_run():
        mlflow.log_param("amount",      tx.Amount)
        mlflow.log_param("time",        tx.Time)
        mlflow.log_metric("fraud_probability", result["fraud_probability"])
        mlflow.log_metric("latency_seconds",   duration)
        mlflow.log_param("is_fraud",    result["is_fraud"])
        mlflow.log_param("risk_level",  result["risk_level"])

    result["latency_seconds"] = duration
    return result