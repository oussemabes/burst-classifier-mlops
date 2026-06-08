# Architecture & Design Guide

## Summary

Audio event classifier trained on labelled bowel sound recordings. Three classes: single burst (`b`), multiple burst (`mb`), harmonic (`h`). Emphasis is on the MLOps pipeline — not model architecture.

---

## Data

Label files are tab-separated: `start_time  end_time  label`. Aliases `sb` and `sbs` map to class `b`. Labels `n` (noise) and `v` (voice artifact) are excluded. Splits are stratified 70/15/15. In production, splits should be grouped by patient or recording session to prevent leakage.

---

## Model

Small 2D CNN (~25k parameters): three Conv2D blocks with BatchNorm and MaxPool, adaptive average pooling, dropout, linear classifier. Input is a log-mel spectrogram. Loss is class-weighted cross-entropy. Intentionally compact — fits CPU inference and edge hardware.

Exports per training run: PyTorch checkpoint (`.pt`), TorchScript (`.ts`), ONNX (`.onnx`).

---

## Q1 — Automated Evaluation Before Deployment

Every training run computes and stores:

- **Macro-F1** — all three classes weighted equally. Used as primary gate metric because all classes matter clinically.
- **Per-class recall** — catches class collapse hidden by accuracy (e.g. model always predicts `b`).
- **Confusion matrix** — identifies `b` vs `mb` confusion which is the hardest separation.
- **Manifest** — SHA-256 hashes of data files and model artifacts for reproducibility.

Gate thresholds live in `configs/default.yaml`:
```yaml
evaluation:
  min_macro_f1: 0.20
  min_per_class_recall: 0.20
```

Cloud Build reads thresholds from the config uploaded with the run — not hardcoded. Non-zero exit blocks the pipeline automatically. No manual approval needed.

For production clinical use: add patient-grouped cross-validation, a locked external validation set, confidence calibration, and regression tests comparing candidate vs. currently deployed model.

---

## Q2 — Cloud Deployment and Change Traceability

**Pipeline (Cloud Build):**
1. Unit tests — fail fast, never deploy broken code
2. Upload training data to GCS
3. Build training image → Artifact Registry
4. Submit Vertex AI Custom Training Job → poll until complete
5. Gate check — read `metrics.json` + `config.yaml` from GCS
6. Build API image → Artifact Registry
7. Deploy to Cloud Run

**Infrastructure (Terraform):** Artifact Registry, Cloud Storage, Cloud Run provisioned as code in `infra/terraform/gcp/`.

**Serving:** FastAPI on Cloud Run with ONNX Runtime. Model downloaded from GCS on first request. Scales to zero — no idle cost.

**Traceability — every deployed version records:**
- Git commit SHA (Docker image tag + artifact manifest)
- SHA-256 of all training data files
- SHA-256 of model artifacts
- Full training config snapshot
- Evaluation metrics and gate decision
- Cloud Build run ID

**MLflow tracking** written to `gs://kwore-web-dev-burst-classifier/mlflow/` after each run. View with:
```bash
mlflow ui --backend-store-uri gs://kwore-web-dev-burst-classifier/mlflow/mlruns
```

---

## Q3 — Edge and Embedded Deployment

Yes, this model could run on embedded hardware. It is small (~25k parameters) and already exported to ONNX which is supported by most edge runtimes.

The main steps would be:

1. **Quantize the model** — reduce weights from 32-bit float to 8-bit integer to shrink size and speed up inference on devices without a GPU
2. **Use a lightweight runtime** — ONNX Runtime and TensorFlow Lite both run on ARM devices (Raspberry Pi, mobile). For very constrained hardware, TFLite Micro is an option
3. **Fix the input size** — most edge runtimes require fixed input dimensions, so the spectrogram window would need to be set to a fixed length
4. **Test on the actual device** — latency and memory usage must be measured on the target hardware, not estimated
5. **Add an update mechanism** — models on edge devices need to be updatable. I would version model binaries in Cloud Storage and push updates OTA with rollback capability, using the same CI/CD pipeline that already handles cloud deployment

---

## Q4 — Semi-Automated Labelling

The labelling workflow is designed as a pipeline, not a manual process. The goal is to minimise annotator time while continuously feeding new verified data back into the training pipeline.

`prelabel.py` runs inference on new raw audio and produces a Label Studio-compatible JSON file. Annotators open Label Studio, see pre-filled predictions, and only correct what is wrong — they do not start from scratch.

```bash
python -m burst_classifier.prelabel \
    --run-dir artifacts/latest \
    --audio new_recording.wav \
    --threshold 0.75
```

**Pipeline:**
1. New audio arrives → `prelabel.py` generates predictions and exports to `reports/`
2. Annotator imports JSON into Label Studio, corrects predictions
3. Corrected export is versioned as a new dataset snapshot in Cloud Storage
4. New snapshot triggers the Cloud Build training pipeline automatically
5. Correction rate is logged as a signal — high rate means the model needs retraining, low rate means it is holding

The key engineering decisions here are: versioning every dataset snapshot with SHA-256 hashes for traceability, keeping the annotation tool (Label Studio) decoupled from the training infrastructure, and wiring the correction export directly into the existing CI/CD trigger.

---

## Current Results

The pipeline ran end-to-end on 2 recordings. The gate passed, the model was deployed to Cloud Run, and the inference endpoint is live.

The metrics reflect the size of the dataset, not the pipeline. With 2 recordings the model has limited generalisation — that is expected and not a pipeline concern. The pipeline is ready to retrain and redeploy automatically as soon as more labelled data is added.

What the results do confirm:
- The evaluation gate correctly blocks or allows deployment based on configurable thresholds
- Every run produces a versioned artifact bundle with full traceability (data hashes, config snapshot, model hashes, git SHA)
- The deployment path from code push to live endpoint is fully automated
