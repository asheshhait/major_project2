from __future__ import annotations

from pathlib import Path

from backend.app.core.segmentation import segment_strokes
from backend.app.core.parser import parse_csv_like
from backend.app.core.strokes import split_into_strokes, stroke_metadata
from scripts.make_synthetic_data import make_pen_file


def _stroke(stroke_id: int, x_min: float, x_max: float, y_min: float = 0, y_max: float = 10):
    return {
        "id": stroke_id,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def test_segmentation_splits_close_words_and_assigns_every_stroke_once():
    strokes = [
        _stroke(0, 0, 8),
        _stroke(1, 12, 20),
        _stroke(2, 24, 32),
        _stroke(3, 46, 54),
        _stroke(4, 58, 66),
    ]

    words = segment_strokes(strokes)

    assert len(words) == 2
    assert words[0].stroke_ids == [0, 1, 2]
    assert words[1].stroke_ids == [3, 4]
    all_stroke_ids = [stroke_id for word in words for stroke_id in word.stroke_ids]
    assert sorted(all_stroke_ids) == [stroke["id"] for stroke in strokes]
    assert len(all_stroke_ids) == len(set(all_stroke_ids))


def test_segmentation_merges_overlapping_parts_of_one_word():
    strokes = [
        _stroke(0, 10, 30),
        _stroke(1, 15, 22, y_min=-4, y_max=0),
        _stroke(2, 34, 50),
    ]

    words = segment_strokes(strokes)

    assert len(words) == 1
    assert words[0].stroke_ids == [0, 1, 2]


def test_adjacent_word_boxes_do_not_overlap_after_padding():
    strokes = [
        _stroke(0, 0, 100),
        _stroke(1, 112, 212),
    ]

    words = segment_strokes(strokes)

    assert len(words) == 2
    left_edge = words[0].bbox["x"] + words[0].bbox["width"]
    assert left_edge == words[1].bbox["x"]


def test_segmentation_splits_words_from_gap_clusters_in_a_long_line():
    gap_sizes = [4, 6, 8, 12, 5, 6, 14, 7]
    strokes = []
    x = 0
    for stroke_id, gap in enumerate(gap_sizes + [0]):
        strokes.append(_stroke(stroke_id, x, x + 10))
        x += 10 + gap

    words = segment_strokes(strokes)

    assert len(words) == 3
    assert [word.stroke_ids for word in words] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8],
    ]
    all_stroke_ids = [stroke_id for word in words for stroke_id in word.stroke_ids]
    assert sorted(all_stroke_ids) == list(range(len(strokes)))
    assert len(all_stroke_ids) == len(set(all_stroke_ids))


def test_segmentation_detects_one_clear_word_gap_in_a_short_line():
    gap_sizes = [8, 10, 24, 0]
    strokes = []
    x = 0
    for stroke_id, gap in enumerate(gap_sizes):
        strokes.append(_stroke(stroke_id, x, x + 20, y_max=60))
        x += 20 + gap

    words = segment_strokes(strokes)

    assert [word.stroke_ids for word in words] == [[0, 1, 2], [3]]


def test_segmentation_splits_two_words_with_one_clear_gap():
    strokes = [
        _stroke(0, 0, 30, y_max=60),
        _stroke(1, 54, 84, y_max=60),
    ]

    words = segment_strokes(strokes)

    assert [word.stroke_ids for word in words] == [[0], [1]]


def test_generated_file_marks_every_requested_word_across_all_lines(tmp_path: Path):
    source = tmp_path / "synthetic.txt"
    requested_word_count = 17
    make_pen_file(source, line_count=3, total_words=requested_word_count)

    points, summary = parse_csv_like(source.read_text(encoding="utf-8"))
    stroke_infos = stroke_metadata(
        split_into_strokes([(point.x, point.y, point.pen) for point in points])
    )
    boxes = segment_strokes(
        [
            {
                "id": stroke.stroke_id,
                "x_min": stroke.x_min,
                "x_max": stroke.x_max,
                "y_min": stroke.y_min,
                "y_max": stroke.y_max,
                "width": stroke.width,
                "height": stroke.height,
            }
            for stroke in stroke_infos
        ]
    )

    assert summary.valid_rows > 0
    assert len(stroke_infos) == requested_word_count
    assert len(boxes) == requested_word_count
    assert sorted(stroke_id for box in boxes for stroke_id in box.stroke_ids) == list(
        range(requested_word_count)
    )
