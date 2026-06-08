# Current Results

Training run: `20260607-232903` — 2 recordings, 334 test samples.

| Metric | Value | Gate threshold |
|--------|-------|---------------|
| Accuracy | 0.563 | — |
| Macro-F1 | 0.532 | 0.20 ✅ |
| `b` recall | 0.485 | 0.20 ✅ |
| `mb` recall | 0.629 | 0.20 ✅ |
| `h` recall | 0.750 | 0.20 ✅ |

Gate passes at 0.20 threshold. Current deployment is live at:
`https://burst-classifier-api-359903658076.europe-west1.run.app`

## Why Metrics Are Low

Only 2 recordings. `b` vs `mb` heavily confused — 71 single bursts predicted as multiple burst, 46 multiple bursts predicted as single burst. Both classes are burst events and the model cannot reliably separate them with this data volume.

Harmonic class has only 16 test samples — recall is high (0.75) but not statistically significant.

## Next Steps for Better Performance

1. Add more recordings — ideally 10+ per class
2. Use patient/session grouped splits to measure true generalisation
3. Apply targeted sampling for minority classes
4. Tune spectrogram window length — harmonic patterns may need longer context
5. Calibrate confidence thresholds per class
