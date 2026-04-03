from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from predict import pipeline

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
    return pipeline.predict(tx.dict())