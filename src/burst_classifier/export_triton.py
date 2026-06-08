from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from burst_classifier.config import latest_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package ONNX model for NVIDIA Triton")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--model-repository", default="model_repository")
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def triton_config(
    model_name: str,
    labels: list[str],
    n_mels: int,
    max_batch_size: int = 32,
) -> str:
    label_count = len(labels)
    return f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 1, {n_mels}, -1 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ {label_count} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 1000
}}

instance_group [
  {{
    count: 1
    kind: KIND_AUTO
  }}
]
"""


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_run_dir()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    packaged_manifest = json.loads(json.dumps(manifest))

    labels = [
        label
        for label, _ in sorted(manifest["class_to_idx"].items(), key=lambda item: item[1])
    ]
    model_name = args.model_name or "burst_classifier"
    n_mels = 64
    if (run_dir / "config.yaml").exists():
        try:
            import yaml

            run_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            model_name = args.model_name or run_config["export"]["model_name"]
            n_mels = int(run_config["features"]["n_mels"])
        except Exception:
            pass

    model_dir = Path(args.model_repository) / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "model.onnx", version_dir / "model.onnx")
    (model_dir / "config.pbtxt").write_text(
        triton_config(model_name, labels, n_mels),
        encoding="utf-8",
    )
    (model_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (model_dir / "manifest.json").write_text(
        json.dumps(packaged_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Triton model repository written to {model_dir}")


if __name__ == "__main__":
    main()
