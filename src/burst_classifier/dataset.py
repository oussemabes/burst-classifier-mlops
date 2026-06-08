from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import Dataset

from burst_classifier.audio import LogMelExtractor, extract_centered_segment
from burst_classifier.labels import LabelEvent


class BurstDataset(Dataset):
    def __init__(self, events: list[LabelEvent], config: dict, class_to_idx: dict[str, int]):
        self.events = events
        self.config = config
        self.class_to_idx = class_to_idx
        self.extractor = LogMelExtractor(config)
        feature_cfg = config["features"]
        self.segment_seconds = float(feature_cfg["segment_seconds"])
        self.context_seconds = float(feature_cfg.get("context_seconds", 0.0))
        self.sample_rate = int(config["data"]["target_sample_rate"])

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        event = self.events[index]
        segment_seconds = max(self.segment_seconds, event.duration + 2 * self.context_seconds)
        waveform = extract_centered_segment(
            event.audio_path,
            event.center,
            segment_seconds,
            self.sample_rate,
        )
        features = self.extractor(waveform)
        return features.float(), torch.tensor(self.class_to_idx[event.label], dtype=torch.long)


def class_weights(events: list[LabelEvent], class_to_idx: dict[str, int]) -> torch.Tensor:
    counts = Counter(event.label for event in events)
    total = sum(counts.values())
    weights = torch.ones(len(class_to_idx), dtype=torch.float32)
    for label, idx in class_to_idx.items():
        weights[idx] = total / max(1, len(class_to_idx) * counts[label])
    return weights

