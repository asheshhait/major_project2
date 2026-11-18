from __future__ import annotations

import math
from typing import Iterable


POINTS_PER_TEMPLATE = 48
UNKNOWN_DISTANCE = 0.9


class DTWMatcher:
    """Small template matcher for teaching words from a handful of samples."""

    def __init__(self) -> None:
        self.templates: list[tuple[str, list[tuple[float, float]]]] = []

    def fit(self, samples: Iterable[tuple[str, Iterable[tuple[float, float]]]]) -> int:
        templates = []
        for label, values in samples:
            points = list(values)
            if label.strip() and len(points) >= 2:
                templates.append((label, _normalize(_resample(points, POINTS_PER_TEMPLATE))))
        self.templates = templates
        return len(self.templates)

    def predict(self, sample: Iterable[tuple[float, float]]) -> dict:
        points = list(sample)
        if not self.templates or len(points) < 2:
            return _unknown_result()

        normalized = _normalize(_resample(points, POINTS_PER_TEMPLATE))
        best_by_label: dict[str, float] = {}
        for label, template in self.templates:
            distance = _dtw_distance(normalized, template)
            best_by_label[label] = min(distance, best_by_label.get(label, math.inf))

        ranked = sorted(best_by_label.items(), key=lambda item: item[1])
        best_label, best_distance = ranked[0]
        confidence = math.exp(-3.0 * best_distance)
        if best_distance > UNKNOWN_DISTANCE:
            return _unknown_result(
                confidence=confidence,
                distance=best_distance,
                topk=ranked[:3],
            )
        return {
            "text": best_label,
            "confidence": confidence,
            "engine": "DTW template matcher",
            "distance": best_distance,
            "accepted": True,
            "topk": [
                {"text": label, "confidence": math.exp(-3.0 * distance)}
                for label, distance in ranked[:3]
            ],
        }


def _unknown_result(
    confidence: float = 0.0,
    distance: float = math.inf,
    topk: list[tuple[str, float]] | None = None,
) -> dict:
    return {
        "text": "Unknown",
        "confidence": confidence,
        "engine": "DTW template matcher",
        "distance": distance if math.isfinite(distance) else None,
        "accepted": False,
        "topk": [
            {"text": label, "confidence": math.exp(-3.0 * item_distance)}
            for label, item_distance in (topk or [])[:3]
        ],
    }


def _resample(points: Iterable[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    source = list(points)
    if len(source) < 2:
        return source
    distances = [0.0]
    for (x1, y1), (x2, y2) in zip(source, source[1:]):
        distances.append(distances[-1] + math.hypot(x2 - x1, y2 - y1))
    total = distances[-1]
    if total == 0:
        return [source[0]] * count

    result: list[tuple[float, float]] = []
    segment = 0
    for index in range(count):
        target = total * index / (count - 1)
        while segment < len(distances) - 2 and distances[segment + 1] < target:
            segment += 1
        start_distance, end_distance = distances[segment], distances[segment + 1]
        ratio = (target - start_distance) / max(end_distance - start_distance, 1e-12)
        start, end = source[segment], source[segment + 1]
        result.append(
            (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
        )
    return result


def _normalize(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    height = max(max(y for _, y in points) - min_y, 1e-9)
    return [((x - min_x) / height, (y - min_y) / height) for x, y in points]


def _dtw_distance(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    previous = [math.inf] * (len(right) + 1)
    previous[0] = 0.0
    for x1, y1 in left:
        current = [math.inf] * (len(right) + 1)
        for index, (x2, y2) in enumerate(right, start=1):
            cost = math.hypot(x1 - x2, y1 - y2)
            current[index] = cost + min(current[index - 1], previous[index], previous[index - 1])
        previous = current
    return previous[-1] / (len(left) + len(right))
