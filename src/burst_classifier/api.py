from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LABELS = ["b", "mb", "h"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Burst Classifier API")

_session: ort.InferenceSession | None = None


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = _load_session()
    return _session


def _load_session() -> ort.InferenceSession:
    gcs_bucket = os.getenv("GCS_BUCKET")
    model_path = os.getenv("MODEL_PATH", "models/latest/model.onnx")
    local_path = "/tmp/model.onnx"

    if gcs_bucket:
        from google.cloud import storage
        client = storage.Client()
        client.bucket(gcs_bucket).blob(model_path).download_to_filename(local_path)
        log.info(json.dumps({"event": "model_loaded", "source": f"gs://{gcs_bucket}/{model_path}"}))
    else:
        local_path = model_path

    return ort.InferenceSession(local_path, providers=["CPUExecutionProvider"])


class SpectrogramRequest(BaseModel):
    spectrogram: list[list[list[float]]] = Field(
        description="Spectrogram shape [1, n_mels, time]"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _session is not None}


@app.post("/predict")
def predict(payload: SpectrogramRequest) -> dict:
    started = time.perf_counter()
    try:
        tensor = np.array(payload.spectrogram, dtype=np.float32)[None, ...]
        if tensor.ndim != 4:
            raise ValueError("Expected shape [1, 1, n_mels, time]")

        session = get_session()
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: tensor})[0][0]
        probabilities = softmax(logits)
        idx = int(np.argmax(probabilities))
        confidence = float(probabilities[idx])
        latency_ms = (time.perf_counter() - started) * 1000

        # Structured log → Cloud Logging → log-based metrics (no Prometheus scraper needed)
        log.info(json.dumps({
            "event": "prediction",
            "label": LABELS[idx],
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "probabilities": {LABELS[i]: round(float(p), 4) for i, p in enumerate(probabilities)},
            "git_sha": os.getenv("GIT_SHA", "unknown"),
        }))

        return {
            "label": LABELS[idx],
            "class_index": idx,
            "confidence": confidence,
            "probabilities": {LABELS[i]: float(p) for i, p in enumerate(probabilities)},
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        log.error(json.dumps({"event": "prediction_error", "error": str(exc), "latency_ms": round(latency_ms, 2)}))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()
