from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


@lru_cache(maxsize=8)
def load_audio(path: str, target_sample_rate: int) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def extract_centered_segment(
    audio_path: str | Path,
    center_seconds: float,
    segment_seconds: float,
    target_sample_rate: int,
) -> torch.Tensor:
    waveform, sample_rate = load_audio(str(Path(audio_path).resolve()), target_sample_rate)
    segment_frames = int(segment_seconds * sample_rate)
    center_frame = int(center_seconds * sample_rate)
    start = center_frame - segment_frames // 2
    end = start + segment_frames

    pad_left = max(0, -start)
    pad_right = max(0, end - waveform.shape[1])
    start = max(0, start)
    end = min(waveform.shape[1], end)
    segment = waveform[:, start:end]

    if pad_left or pad_right:
        segment = F.pad(segment, (pad_left, pad_right))
    if segment.shape[1] < segment_frames:
        segment = F.pad(segment, (0, segment_frames - segment.shape[1]))
    return segment[:, :segment_frames]


class LogMelExtractor(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        features = config["features"]
        sample_rate = config["data"]["target_sample_rate"]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=features["n_fft"],
            win_length=features["win_length"],
            hop_length=features["hop_length"],
            f_min=features["f_min"],
            f_max=features["f_max"],
            n_mels=features["n_mels"],
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.to_db(self.mel(waveform))
        spec = (spec - spec.mean()) / (spec.std().clamp_min(1e-6))
        return spec

