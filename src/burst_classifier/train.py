from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from burst_classifier.config import load_config, resolve_data_root
from burst_classifier.dataset import BurstDataset, class_weights
from burst_classifier.labels import LabelEvent, class_counts, discover_events
from burst_classifier.model import BurstCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train burst classification CNN")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
        return {"sha": sha, "dirty": bool(status)}
    except Exception:
        return {"sha": None, "dirty": None}


def split_events(events: list[LabelEvent], config: dict) -> tuple[list[LabelEvent], ...]:
    split_cfg = config["split"]
    labels = [event.label for event in events]
    train_val, test = train_test_split(
        events,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=labels,
    )
    train_val_labels = [event.label for event in train_val]
    val_fraction = split_cfg["validation_size"] / (1.0 - split_cfg["test_size"])
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=split_cfg["random_state"],
        stratify=train_val_labels,
    )
    return list(train), list(val), list(test)


def collate_spectrograms(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = zip(*batch)
    max_frames = max(item.shape[-1] for item in features)
    padded = [F.pad(item, (0, max_frames - item.shape[-1])) for item in features]
    return torch.stack(padded), torch.stack(labels)


def make_loader(
    events: list[LabelEvent],
    config: dict,
    class_to_idx: dict[str, int],
    shuffle: bool,
) -> DataLoader:
    dataset = BurstDataset(events, config, class_to_idx)
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_spectrograms,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for features, labels in tqdm(loader, leave=False):
        features = features.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return float(np.mean(losses)), y_true, y_pred


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    labels: list[str],
) -> dict[str, Any]:
    loss, y_true, y_pred = run_epoch(model, loader, criterion, device)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": loss,
        "accuracy": report["accuracy"],
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
        ).tolist(),
    }


def write_model_card(
    run_dir: Path,
    metrics: dict[str, Any],
    counts: dict[str, dict[str, int]],
) -> None:
    gate = metrics["deployment_gate"]
    lines = [
        "# Burst Classifier Model Card",
        "",
        f"Deployment gate: {'PASS' if gate['passed'] else 'FAIL'}",
        f"Macro-F1: {metrics['macro_f1']:.4f}",
        f"Weighted-F1: {metrics['weighted_f1']:.4f}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        "",
        "## Dataset Counts",
        "",
        f"- Train: {counts['train']}",
        f"- Validation: {counts['validation']}",
        f"- Test: {counts['test']}",
    ]
    (run_dir / "model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_models(
    model: nn.Module,
    sample_batch: torch.Tensor,
    run_dir: Path,
    config: dict,
) -> dict[str, str]:
    model.eval()
    onnx_path = run_dir / "model.onnx"
    torchscript_path = run_dir / "model.ts"
    checkpoint_path = run_dir / "model.pt"

    scripted = torch.jit.trace(model.cpu(), sample_batch.cpu())
    scripted.save(str(torchscript_path))
    torch.onnx.export(
        model.cpu(),
        sample_batch.cpu(),
        str(onnx_path),
        input_names=["input__0"],
        output_names=["logits"],
        dynamic_axes={"input__0": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        opset_version=config["export"]["opset_version"],
    )
    return {
        "checkpoint": sha256_file(checkpoint_path),
        "torchscript": sha256_file(torchscript_path),
        "onnx": sha256_file(onnx_path),
    }


def upload_to_gcs(run_dir: Path, run_id: str, config: dict) -> None:
    bucket_name = (
        os.environ.get("GCS_BUCKET")
        or config.get("gcp", {}).get("bucket")
    )
    if not bucket_name:
        return
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        upload_files = [
            "model.pt", "model.onnx", "model.ts",
            "metrics.json", "manifest.json", "model_card.md",
            "confusion_matrix.csv", "config.yaml", "mlflow_warning.txt",
        ]
        for filename in upload_files:
            src = run_dir / filename
            if not src.exists():
                continue
            bucket.blob(f"artifacts/{run_id}/{filename}").upload_from_filename(str(src))
            bucket.blob(f"models/latest/{filename}").upload_from_filename(str(src))
        print(f"Uploaded artifacts to gs://{bucket_name}/artifacts/{run_id}/")

        mlruns_dir = Path("/tmp/mlruns")
        if mlruns_dir.exists():
            for mlflow_file in mlruns_dir.rglob("*"):
                if mlflow_file.is_file():
                    rel = mlflow_file.relative_to(Path("/tmp"))
                    bucket.blob(f"mlflow/{rel}").upload_from_filename(str(mlflow_file))
            print(f"Uploaded MLflow tracking to gs://{bucket_name}/mlflow/")
    except Exception as exc:
        print(f"GCS upload warning: {exc}")


def update_latest_pointer(run_dir: Path, artifacts_root: Path) -> None:
    latest_file = artifacts_root / "latest_run.txt"
    latest_file.write_text(str(run_dir.resolve()), encoding="utf-8")
    latest_dir = artifacts_root / "latest"
    if latest_dir.exists() and latest_dir.is_symlink():
        latest_dir.unlink()
    if not latest_dir.exists():
        try:
            latest_dir.symlink_to(run_dir.resolve(), target_is_directory=True)
        except OSError:
            pass


def log_mlflow(run_dir: Path, run_id: str, metrics: dict[str, Any], config: dict) -> None:
    try:
        import mlflow

        mlflow_cfg = config.get("mlflow", {})
        local_mlruns = Path("/tmp/mlruns") if os.environ.get("GCS_BUCKET") else run_dir.parent / "mlruns"
        mlflow.set_tracking_uri(str(local_mlruns))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "burst_classifier"))
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({
                "n_mels": config["features"]["n_mels"],
                "segment_seconds": config["features"]["segment_seconds"],
                "epochs": config["training"]["epochs"],
                "batch_size": config["training"]["batch_size"],
                "learning_rate": config["training"]["learning_rate"],
                "run_id": run_id,
            })
            mlflow.log_metrics({
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
                "deployment_gate_passed": float(metrics["deployment_gate"]["passed"]),
                "b_recall": metrics["deployment_gate"]["per_class_recall"].get("b", 0),
                "mb_recall": metrics["deployment_gate"]["per_class_recall"].get("mb", 0),
                "h_recall": metrics["deployment_gate"]["per_class_recall"].get("h", 0),
            })
            mlflow.log_artifacts(str(run_dir))
    except Exception as exc:
        (run_dir / "mlflow_warning.txt").write_text(str(exc), encoding="utf-8")


def download_data_from_gcs(gcs_data_dir: str, local_dir: Path) -> None:
    from google.cloud import storage as gcs_storage
    local_dir.mkdir(parents=True, exist_ok=True)
    client = gcs_storage.Client()
    bucket_name, prefix = gcs_data_dir.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    for blob in blobs:
        filename = Path(blob.name).name
        dest = local_dir / filename
        if not dest.exists():
            blob.download_to_filename(str(dest))
    print(f"Downloaded {len(blobs)} files from {gcs_data_dir} to {local_dir}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.fast_dev_run:
        config["training"]["epochs"] = 1

    gcs_data_dir = os.environ.get("GCS_DATA_DIR")
    if gcs_data_dir:
        local_data = Path("/tmp/data")
        download_data_from_gcs(gcs_data_dir, local_data)
        config["data"]["root"] = str(local_data)

    set_seed(int(config["seed"]))
    data_root = resolve_data_root(config, args.config)
    artifacts_root = Path(args.artifacts_dir)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    events = discover_events(data_root, config)
    labels = list(config["data"]["target_labels"])
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    train_events, val_events, test_events = split_events(events, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(train_events, config, class_to_idx, shuffle=True)
    val_loader = make_loader(val_events, config, class_to_idx, shuffle=False)
    test_loader = make_loader(test_events, config, class_to_idx, shuffle=False)

    model = BurstCNN(num_classes=len(labels)).to(device)
    weights = class_weights(train_events, class_to_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_macro_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = evaluate_model(model, val_loader, criterion, device, labels)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation": val_metrics})
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(config["training"]["early_stopping_patience"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, criterion, device, labels)
    per_class_recall = {
        label: test_metrics["classification_report"][label]["recall"] for label in labels
    }
    gate = {
        "passed": bool(
            test_metrics["macro_f1"] >= config["evaluation"]["min_macro_f1"]
            and min(per_class_recall.values()) >= config["evaluation"]["min_per_class_recall"]
        ),
        "min_macro_f1": config["evaluation"]["min_macro_f1"],
        "min_per_class_recall": config["evaluation"]["min_per_class_recall"],
        "per_class_recall": per_class_recall,
    }
    test_metrics["deployment_gate"] = gate

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "class_to_idx": class_to_idx,
        "config": config,
        "labels": labels,
    }
    torch.save(checkpoint, run_dir / "model.pt")

    sample_batch, _ = next(iter(test_loader))
    artifact_hashes = export_models(model, sample_batch[:1], run_dir, config)

    (run_dir / "config.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    np.savetxt(
        run_dir / "confusion_matrix.csv",
        np.array(test_metrics["confusion_matrix"], dtype=int),
        fmt="%d",
        delimiter=",",
        header=",".join(labels),
        comments="",
    )

    data_files = sorted(
        {event.label_path for event in events} | {event.audio_path for event in events}
    )
    manifest = {
        "run_id": run_id,
        "git": git_metadata(),
        "data_root": str(data_root),
        "source_files": {
            str(path.relative_to(data_root)): sha256_file(path)
            for path in data_files
        },
        "class_to_idx": class_to_idx,
        "class_counts": {
            "all": class_counts(events),
            "train": class_counts(train_events),
            "validation": class_counts(val_events),
            "test": class_counts(test_events),
        },
        "artifact_hashes": artifact_hashes,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_model_card(run_dir, test_metrics, manifest["class_counts"])
    update_latest_pointer(run_dir, artifacts_root)
    shutil.copy2("requirements.txt", run_dir / "requirements.txt")
    log_mlflow(run_dir, run_id, test_metrics, config)
    upload_to_gcs(run_dir, run_id, config)

    if not gate["passed"] and not args.skip_gate:
        raise SystemExit(f"Deployment gate failed. See {run_dir / 'metrics.json'}")
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
