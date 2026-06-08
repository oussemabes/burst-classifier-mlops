from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from burst_classifier.audio import LogMelExtractor, extract_centered_segment, load_audio
from burst_classifier.config import latest_run_dir
from burst_classifier.model import BurstCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pre-labels for annotator correction")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--step-seconds", type=float, default=0.25)
    parser.add_argument("--label-studio-audio-prefix", default="/data/local-files/?d=")
    return parser.parse_args()


def load_model(run_dir: Path) -> tuple[BurstCNN, dict, list[str]]:
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu")
    labels = [
        label
        for label, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda item: item[1])
    ]
    model = BurstCNN(num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["config"], labels


def suppress_overlaps(predictions: list[dict], min_gap_seconds: float = 0.1) -> list[dict]:
    predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)
    kept: list[dict] = []
    for candidate in predictions:
        overlaps = False
        for existing in kept:
            latest_start = max(candidate["start"], existing["start"])
            earliest_end = min(candidate["end"], existing["end"])
            if earliest_end - latest_start > min_gap_seconds:
                overlaps = True
                break
        if not overlaps:
            kept.append(candidate)
    return sorted(kept, key=lambda item: item["start"])


def label_studio_payload(audio_path: Path, predictions: list[dict], prefix: str) -> list[dict]:
    results = []
    for idx, prediction in enumerate(predictions):
        results.append(
            {
                "id": f"prelabel-{idx}",
                "from_name": "label",
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": prediction["start"],
                    "end": prediction["end"],
                    "labels": [prediction["label"]],
                },
                "score": prediction["score"],
            }
        )
    return [
        {
            "data": {"audio": f"{prefix}{audio_path.name}"},
            "predictions": [
                {
                    "model_version": "burst-classifier",
                    "score": max([item["score"] for item in predictions], default=0.0),
                    "result": results,
                }
            ],
        }
    ]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_run_dir()
    audio_path = Path(args.audio).resolve()
    model, config, labels = load_model(run_dir)
    extractor = LogMelExtractor(config)
    window_seconds = args.window_seconds or config["features"]["segment_seconds"]
    sample_rate = int(config["data"]["target_sample_rate"])
    waveform, _ = load_audio(str(audio_path), sample_rate)
    duration = waveform.shape[1] / sample_rate

    predictions: list[dict] = []
    center = window_seconds / 2.0
    while center < duration:
        segment = extract_centered_segment(audio_path, center, window_seconds, sample_rate)
        features = extractor(segment).unsqueeze(0)
        with torch.no_grad():
            probabilities = F.softmax(model(features), dim=1)[0]
        score, idx = torch.max(probabilities, dim=0)
        if float(score) >= args.threshold:
            predictions.append(
                {
                    "start": max(0.0, center - window_seconds / 2.0),
                    "end": min(duration, center + window_seconds / 2.0),
                    "label": labels[int(idx)],
                    "score": round(float(score), 6),
                }
            )
        center += args.step_seconds

    predictions = suppress_overlaps(predictions)
    payload = label_studio_payload(audio_path, predictions, args.label_studio_audio_prefix)
    output = (
        Path(args.output)
        if args.output
        else Path("reports") / f"prelabels_{audio_path.stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(predictions)} pre-labels to {output}")


if __name__ == "__main__":
    main()
