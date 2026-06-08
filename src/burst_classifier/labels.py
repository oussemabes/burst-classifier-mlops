from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelEvent:
    audio_path: Path
    label_path: Path
    start: float
    end: float
    label: str
    source_label: str
    event_id: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


def normalize_label(label: str, aliases: dict[str, str]) -> str | None:
    normalized = label.strip().lower()
    return aliases.get(normalized)


def read_label_file(
    label_path: str | Path,
    audio_path: str | Path,
    aliases: dict[str, str],
    target_labels: set[str],
) -> list[LabelEvent]:
    events: list[LabelEvent] = []
    label_path = Path(label_path)
    audio_path = Path(audio_path)

    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid label line {label_path}:{line_number}: {raw_line!r}")
            start, end = float(parts[0]), float(parts[1])
            source_label = parts[2]
            label = normalize_label(source_label, aliases)
            if label is None or label not in target_labels:
                continue
            if end <= start:
                continue
            event_id = f"{audio_path.stem}:{line_number}:{start:.6f}:{end:.6f}:{label}"
            events.append(
                LabelEvent(
                    audio_path=audio_path,
                    label_path=label_path,
                    start=start,
                    end=end,
                    label=label,
                    source_label=source_label.strip().lower(),
                    event_id=event_id,
                )
            )
    return events


def discover_events(data_root: str | Path, config: dict) -> list[LabelEvent]:
    data_root = Path(data_root)
    aliases = dict(config["data"]["label_aliases"])
    target_labels = set(config["data"]["target_labels"])
    events: list[LabelEvent] = []

    for label_path in sorted(data_root.glob(config["data"]["label_glob"])):
        audio_path = label_path.with_suffix(".wav")
        if not audio_path.exists():
            continue
        events.extend(read_label_file(label_path, audio_path, aliases, target_labels))

    if not events:
        raise ValueError(f"No target events found in {data_root}")
    return events


def class_counts(events: list[LabelEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.label] = counts.get(event.label, 0) + 1
    return dict(sorted(counts.items()))
