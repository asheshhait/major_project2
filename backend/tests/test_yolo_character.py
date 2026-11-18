from __future__ import annotations

from pathlib import Path

from backend.app.ml.yolo_character import (
    CharacterDetection,
    build_yolo_character_dataset,
    group_character_detections,
)


def _write_character_sample(path: Path, offset: int) -> None:
    path.write_text(
        "\n".join(
            [
                f"{offset},0,1",
                f"{offset + 5},10,1",
                f"{offset + 10},20,1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_yolo_dataset_builder_renders_characters_with_valid_boxes(tmp_path: Path):
    source = tmp_path / "characters"
    for label, offset in (("অ", 0), ("আ", 2)):
        class_dir = source / label
        class_dir.mkdir(parents=True)
        for sample_index in range(5):
            _write_character_sample(class_dir / f"sample-{sample_index}.txt", offset)

    result = build_yolo_character_dataset(
        source,
        tmp_path / "generated",
        train_images=8,
        validation_images=4,
        seed=17,
    )

    assert result["classes"] == ["অ", "আ"]
    assert len(list((tmp_path / "generated" / "images" / "train").glob("*.jpg"))) == 8
    assert len(list((tmp_path / "generated" / "images" / "val").glob("*.jpg"))) == 4
    for label_path in (tmp_path / "generated" / "labels").rglob("*.txt"):
        annotations = label_path.read_text(encoding="utf-8").splitlines()
        assert annotations
        for annotation in annotations:
            class_id, center_x, center_y, width, height = map(float, annotation.split())
            assert class_id in {0, 1}
            assert 0 < center_x < 1
            assert 0 < center_y < 1
            assert 0 < width < 1
            assert 0 < height < 1


def test_character_boxes_are_grouped_as_non_overlapping_words():
    detections = [
        CharacterDetection(0, 10, 10, 30, 0.9, 0),
        CharacterDetection(13, 10, 23, 30, 0.8, 1),
        CharacterDetection(70, 10, 80, 30, 0.9, 0),
    ]
    strokes = [
        {"id": 0, "x_min": 1, "x_max": 8, "y_min": 15, "y_max": 25},
        {"id": 1, "x_min": 14, "x_max": 20, "y_min": 15, "y_max": 25},
        {"id": 2, "x_min": 72, "x_max": 78, "y_min": 15, "y_max": 25},
    ]

    words = group_character_detections(detections, strokes)

    assert len(words) == 2
    assert [word.stroke_ids for word in words] == [[0, 1], [2]]
    left_edge = words[0].bbox["x"] + words[0].bbox["width"]
    assert left_edge < words[1].bbox["x"]


def test_strokes_are_assigned_to_only_one_word_across_multiple_lines():
    detections = [
        CharacterDetection(0, 10, 10, 30, 0.9, 0),
        CharacterDetection(70, 10, 80, 30, 0.9, 0),
        CharacterDetection(0, 80, 10, 100, 0.9, 0),
        CharacterDetection(70, 80, 80, 100, 0.9, 0),
    ]
    strokes = [
        {"id": 0, "x_min": 1, "x_max": 8, "y_min": 15, "y_max": 25},
        {"id": 1, "x_min": 72, "x_max": 78, "y_min": 15, "y_max": 25},
        {"id": 2, "x_min": 1, "x_max": 8, "y_min": 85, "y_max": 95},
        {"id": 3, "x_min": 72, "x_max": 78, "y_min": 85, "y_max": 95},
    ]

    words = group_character_detections(detections, strokes)

    assert len(words) == 4
    assert [word.stroke_ids for word in words] == [[0], [1], [2], [3]]


def test_learned_word_gap_threshold_changes_character_grouping(monkeypatch):
    import backend.app.core.segmentation as segmentation

    detections = [
        CharacterDetection(0, 10, 10, 30, 0.9, 0),
        CharacterDetection(13, 10, 23, 30, 0.8, 1),
        CharacterDetection(70, 10, 80, 30, 0.9, 0),
    ]
    strokes = [
        {"id": 0, "x_min": 1, "x_max": 8, "y_min": 15, "y_max": 25},
        {"id": 1, "x_min": 14, "x_max": 20, "y_min": 15, "y_max": 25},
        {"id": 2, "x_min": 72, "x_max": 78, "y_min": 15, "y_max": 25},
    ]
    monkeypatch.setattr(segmentation, "LEARNED_GAP_THRESHOLD_FACTOR", 0.1)

    words = group_character_detections(detections, strokes)

    assert len(words) == 3
