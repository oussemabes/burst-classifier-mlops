# Burst Classifier — MLOps Pipeline

Audio event classifier (single burst / multiple burst / harmonic) built as a production-grade ML training and deployment pipeline. Emphasis on engineering workflow: reproducible training, automated evaluation gates, artifact traceability, GCP deployment, and a human-in-the-loop labelling loop.

---

## Repository Layout

```
.
├── src/burst_classifier/         # Python package
│   ├── audio.py                  # audio loading + log-mel feature extraction
│   ├── config.py                 # config loader + artifact path helpers
│   ├── dataset.py                # PyTorch Dataset
│   ├── labels.py                 # label file parser + alias normalisation
│   ├── model.py                  # BurstCNN — compact 2D CNN (~25k params)
│   ├── train.py                  # training entry point + GCS upload + MLflow
│   ├── export_triton.py          # ONNX → Triton model repository (local dev)
│   ├── prelabel.py               # sliding-window pre-labelling for Label Studio
│   └── api.py                    # FastAPI inference service + structured logging
├── configs/
│   └── default.yaml              # all hyperparameters and gate thresholds
├── tests/
│   ├── test_labels.py
│   └── test_model.py
├── deploy/
│   ├── docker-compose.triton.yml # local dev stack
│   └── monitoring/               # Prometheus config + alert rules
├── infra/terraform/gcp/          # GCP infrastructure as code
├── labelling/
│   └── labelstudio_config.xml
├── docs/
│   └── presentation_guide.md
├── cloudbuild.yaml               # Cloud Build CI/CD pipeline
├── vertex_job_template.yaml      # Vertex AI custom job spec template
├── Dockerfile.train              # CPU training container
├── Dockerfile.api                # Cloud Run inference container
├── AS_1.wav / AS_1.txt
└── 23M74M.wav / 23M74M.txt
```

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest tests/ -q
python -m burst_classifier.train --config configs/default.yaml
```

---

## GCP Stack

| Component | Service |
|-----------|---------|
| CI/CD | Cloud Build (`cloudbuild.yaml`) |
| Training | Vertex AI Custom Training Job |
| Model + MLflow storage | Cloud Storage (`kwore-web-dev-burst-classifier`) |
| Container registry | Artifact Registry (`europe-west1`) |
| Serving | Cloud Run (FastAPI + ONNX Runtime) |
| Monitoring | Cloud Monitoring + Cloud Logging |

**Provision infrastructure:**
```bash
cd infra/terraform/gcp && terraform init && terraform apply
```

---

## Pipeline

```
Push to master
      │
      ▼
Cloud Build
  1. pytest tests/
  2. Upload data to GCS
  3. Build + push training image (Artifact Registry)
  4. Submit Vertex AI Custom Job → wait for completion
  5. Gate check: read metrics.json + config.yaml from GCS
     macro_f1 >= threshold AND min_per_class_recall >= threshold
     → FAIL: blocks deployment
     → PASS: continues
  6. Build + push API image (Artifact Registry)
  7. Deploy to Cloud Run
```

**Training job (Vertex AI):**
- Downloads data from `gs://bucket/data/`
- Trains CNN with stratified 70/15/15 split
- Uploads artifacts to `gs://bucket/artifacts/<run_id>/` and `gs://bucket/models/latest/`
- Uploads MLflow tracking to `gs://bucket/mlflow/`

---

## Automated Evaluation Gate

Thresholds in `configs/default.yaml`:

```yaml
evaluation:
  min_macro_f1: 0.20
  min_per_class_recall: 0.20
```

Gate reads thresholds directly from the config uploaded with each run — single source of truth. Non-zero exit from gate step blocks the pipeline.

Artifacts written per run:

| File | Content |
|------|---------|
| `metrics.json` | accuracy, macro-F1, weighted-F1, per-class P/R/F1 |
| `confusion_matrix.csv` | class-level error analysis |
| `manifest.json` | git SHA, data file hashes, artifact hashes |
| `model_card.md` | summary for review |
| `config.yaml` | exact config used for this run |

---

## Change Traceability

Every deployed model records:
- Git commit SHA (Docker image tag + manifest)
- SHA-256 of all training data files
- SHA-256 of model artifacts (`.pt`, `.onnx`, `.ts`)
- Training config snapshot
- Evaluation metrics + gate decision
- Cloud Build run ID

---

## MLflow Experiment Tracking

Each training run writes to `gs://kwore-web-dev-burst-classifier/mlflow/`. View runs locally:

```bash
mlflow ui --backend-store-uri gs://kwore-web-dev-burst-classifier/mlflow/mlruns
```

---

## Semi-Automated Labelling

```bash
python -m burst_classifier.prelabel \
    --run-dir artifacts/latest \
    --audio 23M74M.wav \
    --threshold 0.75 \
    --output reports/prelabels_23M74M.json
```

Import into Label Studio via `labelling/labelstudio_config.xml`. Annotators correct uncertain predictions only. Corrections become the next versioned dataset snapshot.

---

## Live API

```
https://burst-classifier-api-359903658076.europe-west1.run.app

GET  /health    → {"status":"ok","model_loaded":true}
POST /predict   → {"label":"b","confidence":0.60,"probabilities":{...}}
```

Request format:
```json
{
  "spectrogram": [[[0.1, 0.1, ...], ...]]
}
```
Shape: `[1, n_mels, time_frames]` (1 channel, 64 mel bins, variable time).

---

## Local Dev

```bash
python -m burst_classifier.train --config configs/default.yaml
docker compose -f deploy/docker-compose.triton.yml up --build
```
