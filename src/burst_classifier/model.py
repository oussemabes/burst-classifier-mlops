from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class BurstCNN(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_classes": self.num_classes,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BurstCNN":
        checkpoint = torch.load(path, map_location=device)
        num_classes = checkpoint.get("num_classes", 3)
        model = cls(num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
