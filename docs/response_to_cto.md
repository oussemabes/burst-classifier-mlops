# Technical Overview

This project implements a complete MLOps pipeline for audio burst classification. The emphasis is on the engineering workflow rather than model architecture.

## What Is Built

- CNN training pipeline with automated evaluation gates that block deployment on metric failure
- Vertex AI Custom Training Jobs triggered by Cloud Build on every push
- Artifacts (model, metrics, config, hashes) versioned in Cloud Storage per run
- MLflow experiment tracking persisted to Cloud Storage
- Cloud Run inference API with ONNX Runtime — scales to zero
- Semi-automated labelling with Label Studio pre-annotation export
- Terraform infrastructure as code for the full GCP stack

## Design Decisions

**Model is intentionally small (~25k parameters).** The constraint is the pipeline, not the architecture. A larger model slots in by changing one file.

**Gate thresholds are in config, not hardcoded.** Cloud Build reads thresholds from the config file uploaded with each training run — changing a threshold in one place affects both training and CI.

**No Triton in production.** ONNX Runtime in Cloud Run handles this model size at <30ms p95 latency with zero infrastructure overhead. Triton adds value at scale or with multiple concurrent models.

**Cloud Run over Kubernetes.** For a single model serving endpoint, Cloud Run is the right abstraction — managed, auto-scaling, zero idle cost. Kubernetes is appropriate when the serving layer needs custom routing, multiple sidecars, or GPU scheduling.

## Current Limitations

Only 2 training recordings. The pipeline is production-ready; the dataset is not. Performance improves directly with more labelled data — no pipeline changes required.
