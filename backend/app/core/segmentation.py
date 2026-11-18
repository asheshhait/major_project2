from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


DEFAULT_GAP_SENSITIVITY = 1.0
LINE_HEIGHT_FACTOR = 1.5
STROKE_MERGE_FACTOR = 0.05
WORD_GAP_HEIGHT_FACTOR = 0.9
MIN_WORD_GAP_HEIGHT_FACTOR = 0.35
GROUP_WIDTH_GAP_FACTOR = 0.75
GAP_CLUSTER_RATIO = 1.45
GAP_CLUSTER_MIN_HEIGHT_FACTOR = 0.25
MIN_GAPS_PER_CLUSTER = 2
GAP_CLUSTER_ITERATIONS = 20
BOX_PADDING_FACTOR = 0.08
LEARNED_GAP_THRESHOLD_FACTOR: float | None = None


@dataclass
class WordBoxData:
    id: int
    line_index: int
    word_index: int
    stroke_ids: list[int]
    bbox: dict[str, float]


def segment_strokes(
    strokes: list[dict[str, Any]],
    gap_sensitivity: float = DEFAULT_GAP_SENSITIVITY,
) -> list[WordBoxData]:
    """Group stroke boxes into lines and words using vertical and horizontal gaps."""
    if not strokes:
        return []
    if gap_sensitivity <= 0:
        raise ValueError("gap_sensitivity must be greater than zero.")

    stroke_heights = [max(float(stroke.get("height", 0.0)), 1.0) for stroke in strokes]
    typical_height = median(stroke_heights)
    line_threshold = typical_height * LINE_HEIGHT_FACTOR
    sorted_strokes = sorted(strokes, key=lambda stroke: _center_y(stroke))

    lines: list[list[dict[str, Any]]] = []
    line_centers: list[float] = []
    for stroke in sorted_strokes:
        center_y = _center_y(stroke)
        if not lines or center_y - line_centers[-1] > line_threshold:
            lines.append([stroke])
            line_centers.append(center_y)
        else:
            lines[-1].append(stroke)
            line_centers[-1] = sum(_center_y(item) for item in lines[-1]) / len(lines[-1])

    word_boxes: list[WordBoxData] = []
    next_id = 0
    for line_index, line in enumerate(lines):
        line_height = median(max(float(stroke.get("height", 0.0)), 1.0) for stroke in line)
        groups = _merge_overlapping_strokes(line, line_height * STROKE_MERGE_FACTOR)
        gaps = [
            max(0.0, right["x_min"] - left["x_max"])
            for left, right in zip(groups, groups[1:])
        ]
        group_widths = [group["x_max"] - group["x_min"] for group in groups]
        gap_threshold = _word_gap_threshold(
            gaps,
            line_height,
            gap_sensitivity,
            group_widths,
            LEARNED_GAP_THRESHOLD_FACTOR,
        )

        words: list[list[dict[str, Any]]] = []
        for group in groups:
            if not words or group["x_min"] - words[-1][-1]["x_max"] > gap_threshold:
                words.append([group])
            else:
                words[-1].append(group)

        line_word_boxes: list[WordBoxData] = []
        for word_index, word_groups in enumerate(words):
            members = [member for group in word_groups for member in group["members"]]
            x_min = min(float(member["x_min"]) for member in members)
            x_max = max(float(member["x_max"]) for member in members)
            y_min = min(float(member["y_min"]) for member in members)
            y_max = max(float(member["y_max"]) for member in members)
            padding = max(x_max - x_min, y_max - y_min, 1.0) * BOX_PADDING_FACTOR
            line_word_boxes.append(
                WordBoxData(
                    id=next_id,
                    line_index=line_index,
                    word_index=word_index,
                    stroke_ids=sorted(int(member["id"]) for member in members),
                    bbox={
                        "x": x_min - padding,
                        "y": y_min - padding,
                        "width": x_max - x_min + padding * 2,
                        "height": y_max - y_min + padding * 2,
                    },
                )
            )
            next_id += 1
        for left_box, right_box in zip(line_word_boxes, line_word_boxes[1:]):
            left_edge = left_box.bbox["x"] + left_box.bbox["width"]
            right_edge = right_box.bbox["x"]
            if left_edge > right_edge:
                boundary = (left_edge + right_edge) / 2
                left_box.bbox["width"] = boundary - left_box.bbox["x"]
                right_box.bbox["width"] += right_box.bbox["x"] - boundary
                right_box.bbox["x"] = boundary
        word_boxes.extend(line_word_boxes)
    return word_boxes


def _word_gap_threshold(
    gaps: list[float],
    line_height: float,
    sensitivity: float,
    group_widths: list[float],
    learned_threshold_factor: float | None = None,
) -> float:
    """Separate character gaps from inter-word outliers within a line."""
    if learned_threshold_factor is not None:
        return line_height * learned_threshold_factor / sensitivity

    height_threshold = line_height * WORD_GAP_HEIGHT_FACTOR
    minimum_threshold = line_height * MIN_WORD_GAP_HEIGHT_FACTOR
    width_threshold = median(group_widths) * GROUP_WIDTH_GAP_FACTOR if group_widths else height_threshold
    threshold = min(height_threshold, max(minimum_threshold, width_threshold)) / sensitivity
    positive_gaps = [gap for gap in gaps if gap > 0]
    if len(positive_gaps) < MIN_GAPS_PER_CLUSTER + 1:
        return threshold

    low_center = min(positive_gaps)
    high_center = max(positive_gaps)
    low_gaps: list[float] = []
    high_gaps: list[float] = []

    for _ in range(GAP_CLUSTER_ITERATIONS):
        low_gaps = []
        high_gaps = []
        for gap in positive_gaps:
            if abs(gap - low_center) <= abs(gap - high_center):
                low_gaps.append(gap)
            else:
                high_gaps.append(gap)

        if not low_gaps or not high_gaps:
            return threshold

        next_low_center = sum(low_gaps) / len(low_gaps)
        next_high_center = sum(high_gaps) / len(high_gaps)
        if next_low_center == low_center and next_high_center == high_center:
            break
        low_center, high_center = next_low_center, next_high_center

    if len(low_gaps) < MIN_GAPS_PER_CLUSTER or not high_gaps:
        return threshold

    gap_ratio = high_center / max(low_center, line_height * 0.1)
    gap_delta = high_center - low_center
    if gap_ratio >= GAP_CLUSTER_RATIO and gap_delta >= line_height * GAP_CLUSTER_MIN_HEIGHT_FACTOR:
        threshold = ((low_center + high_center) / 2) / sensitivity
    return threshold


def train_gap_threshold(
    examples: list[tuple[float, bool]],
) -> float | None:
    """Fit a gap/line-height threshold from manually labeled stroke-group boundaries."""
    positive = [feature for feature, is_boundary in examples if is_boundary]
    negative = [feature for feature, is_boundary in examples if not is_boundary]
    if not positive or not negative:
        return None

    values = sorted({feature for feature, _ in examples})
    candidates = [values[0] - 1e-9]
    candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    candidates.append(values[-1])

    def score(threshold: float) -> tuple[float, float]:
        true_positive = sum(feature > threshold for feature in positive)
        false_negative = len(positive) - true_positive
        false_positive = sum(feature > threshold for feature in negative)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return f1, threshold

    return max(candidates, key=score)


def gap_examples_from_corrections(
    strokes: list[dict[str, Any]],
    word_assignments: dict[int, tuple[int, int]],
) -> list[tuple[float, bool]]:
    """Extract labeled adjacent-group gaps from manually corrected word boxes."""
    examples: list[tuple[float, bool]] = []
    line_indices = sorted({line_index for line_index, _ in word_assignments.values()})
    for line_index in line_indices:
        line = [
            stroke for stroke in strokes
            if int(stroke["id"]) in word_assignments
            and word_assignments[int(stroke["id"])][0] == line_index
        ]
        if len(line) < 2:
            continue
        line_height = median(max(float(stroke.get("height", 0.0)), 1.0) for stroke in line)
        groups = _merge_overlapping_strokes(line, line_height * STROKE_MERGE_FACTOR)
        for left, right in zip(groups, groups[1:]):
            left_words = {
                word_assignments[int(member["id"])][1]
                for member in left["members"]
            }
            right_words = {
                word_assignments[int(member["id"])][1]
                for member in right["members"]
            }
            if len(left_words) != 1 or len(right_words) != 1:
                continue
            gap = max(0.0, float(right["x_min"]) - float(left["x_max"]))
            examples.append((gap / line_height, left_words != right_words))
    return examples


def set_learned_gap_threshold(threshold_factor: float | None) -> None:
    global LEARNED_GAP_THRESHOLD_FACTOR
    LEARNED_GAP_THRESHOLD_FACTOR = threshold_factor


def get_learned_gap_threshold() -> float | None:
    return LEARNED_GAP_THRESHOLD_FACTOR


def _center_y(stroke: dict[str, Any]) -> float:
    return (float(stroke.get("y_min", 0.0)) + float(stroke.get("y_max", 0.0))) / 2


def _merge_overlapping_strokes(
    strokes: list[dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    ordered = sorted(strokes, key=lambda stroke: float(stroke["x_min"]))
    groups: list[dict[str, Any]] = []
    for stroke in ordered:
        if not groups or float(stroke["x_min"]) > groups[-1]["x_max"] + tolerance:
            groups.append(
                {
                    "x_min": float(stroke["x_min"]),
                    "x_max": float(stroke["x_max"]),
                    "members": [stroke],
                }
            )
            continue
        groups[-1]["x_max"] = max(groups[-1]["x_max"], float(stroke["x_max"]))
        groups[-1]["members"].append(stroke)
    return groups
