# Production Recommendations

## Data

- Grouped train/val/test splits by patient or recording session — current random splits leak data
- Locked external validation set held out permanently
- Minimum 50 labelled events per class before promoting to production
- Version every dataset snapshot with SHA-256 hashes

## Model

- Confidence calibration (temperature scaling or Platt scaling) — raw softmax overconfident
- Per-class confidence thresholds — harmonic events may need a different abstention threshold than bursts
- Longer temporal context for harmonic class (try 2–3 second windows vs current 1 second)

## Pipeline

- Gate thresholds should increase as more data accumulates — do not leave at 0.20 for production
- Regression test: new model must beat currently deployed model on a fixed reference set
- Signed container images for audit trail
- Model registry with explicit promotion steps (None → Staging → Production)

## Monitoring

- Confidence drift alert: median confidence drops below 0.65 over a 1-hour window
- Class distribution drift: predicted class proportions deviate >30% from training baseline
- Human correction rate from Label Studio: >25% correction triggers retraining review
- Latency budget: p95 inference < 100ms on Cloud Run

## Edge Deployment

- INT8 quantization reduces model to <100KB
- Full preprocessing + inference pipeline must be benchmarked on target hardware
- OTA update mechanism with rollback before any production edge deployment
