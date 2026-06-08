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

The model has ~25k parameters by design. Path to embedded deployment:

1. Fix input window to a static size — edge runtimes require fixed shapes
2. Export ONNX and benchmark CPU latency on target hardware
3. Apply INT8 static quantization (calibration data required)
4. Convert to the appropriate runtime:
   - NVIDIA Jetson → TensorRT
   - ARM Cortex-A (Raspberry Pi, mobile) → ONNX Runtime Mobile or TFLite
   - MCU (STM32, nRF) → TFLite Micro or CMSIS-NN
5. Profile memory, power draw, and latency on the actual device
6. Add abstention: if confidence < threshold, output "uncertain" rather than force a class
7. OTA update loop: cloud retraining pipeline pushes versioned model updates to devices with rollback

The mel spectrogram extraction is often more expensive than the model itself on constrained hardware — benchmark the full preprocessing + inference chain, not just the CNN.

---

## Q4 — Semi-Automated Labelling

`prelabel.py` runs sliding-window inference over raw audio and exports Label Studio predictions with confidence scores.

```bash
python -m burst_classifier.prelabel \
    --run-dir artifacts/latest \
    --audio new_recording.wav \
    --threshold 0.75
```

**Workflow:**
1. Model predicts on new audio → exports predictions above confidence threshold
2. Annotator imports into Label Studio → corrects only uncertain or wrong predictions
3. Corrected annotations exported and schema-validated
4. Added to next dataset snapshot as a versioned GCS object
5. Model disagreement rate (% predictions corrected) tracked as active-learning signal

High correction rate → model is degrading → trigger retraining. Low correction rate → model is holding → defer retraining.

---

## Current Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.563 |
| Macro-F1 | 0.532 |
| `b` recall | 0.485 |
| `mb` recall | 0.629 |
| `h` recall | 0.750 |

Trained on 2 recordings only. `b` vs `mb` confusion is the main error — both are burst events, small dataset cannot fully separate them. With more labelled data the pipeline is ready to retrain and promote automatically.
