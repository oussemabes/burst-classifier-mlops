from pathlib import Path

from burst_classifier.labels import read_label_file


def test_read_label_file_maps_aliases_and_filters_non_targets(tmp_path: Path) -> None:
    label_path = tmp_path / "sample.txt"
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"placeholder")
    label_path.write_text(
        "\n".join(
            [
                "0.0\t0.2\tsb",
                "0.3\t0.5\tmb",
                "0.6\t0.8\th",
                "0.9\t1.0\tn",
                "1.1\t1.2\tv",
            ]
        ),
        encoding="utf-8",
    )

    events = read_label_file(
        label_path,
        audio_path,
        aliases={"b": "b", "sb": "b", "sbs": "b", "mb": "mb", "h": "h"},
        target_labels={"b", "mb", "h"},
    )

    assert [event.label for event in events] == ["b", "mb", "h"]
    assert [event.source_label for event in events] == ["sb", "mb", "h"]


def test_read_label_file_skips_invalid_duration(tmp_path: Path) -> None:
    label_path = tmp_path / "sample.txt"
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"placeholder")
    label_path.write_text("1.0\t0.9\tb\n1.1\t1.3\tb\n", encoding="utf-8")

    events = read_label_file(
        label_path,
        audio_path,
        aliases={"b": "b"},
        target_labels={"b"},
    )

    assert len(events) == 1
    assert events[0].start == 1.1

