from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

from src.utils.io import load_model


class PredictRequest(BaseModel):
    records: List[dict]


class PredictResponse(BaseModel):
    probabilities: List[float]
    predictions: List[int]


MODEL_ENV_PATH = Path("artifacts/churn_model.joblib")


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


@app.on_event("startup")
def _load() -> None:
    global MODEL
    model_path = MODEL_ENV_PATH
    if not model_path.exists():
        MODEL = None
        return
    MODEL = load_model(model_path)


@app.get("/healthz")
async def health() -> JSONResponse:
    status = "ok" if "MODEL" in globals() and MODEL is not None else "model_not_loaded"
    return JSONResponse({"status": status})


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    if "MODEL" not in globals() or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and place at artifacts/churn_model.joblib")

    if not payload.records:
        raise HTTPException(status_code=400, detail="No records provided")

    df = pd.DataFrame(payload.records)

    proba = MODEL.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return PredictResponse(probabilities=proba.tolist(), predictions=preds.tolist())
