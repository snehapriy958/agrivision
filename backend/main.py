"""
backend/main.py — FastAPI Inference Server
Run : uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import os
import tempfile

from pydantic import BaseModel
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile

from inference.predictor import predict


app = FastAPI(title="Plant Classifier API", version="1.0.0")

class Prediction(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {"message": "API is running"}


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse, include_in_schema=False)
def predict_endpoint(file: UploadFile = File(...)):

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. An image file is required.",
        )

    # Validate file size (max 5MB)
    if file.size and file.size > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 5MB)"
        )

    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.file.seek(0)
            tmp.write(file.file.read())
            tmp_path = tmp.name

        predictions = predict(tmp_path)
        return {"predictions": predictions}

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)