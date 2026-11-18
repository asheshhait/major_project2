from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class StrokeInfo:
    """Represents a single pen-down sequence with bounding-box metadata."""

    points: list[tuple[float, float]]
    stroke_id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: float
    height: float
    centroid_x: float
    centroid_y: float
    point_count: int


def split_into_strokes(points: list[tuple[float, float, int]], pen_down_value: int = 1) -> list[list[tuple[float, float, int]]]:
    """Split a point stream into consecutive pen-down strokes."""
    strokes: list[list[tuple[float, float, int]]] = []
    current: list[tuple[float, float, int]] = []

    for x, y, pen in points:
        if pen == pen_down_value:
            current.append((x, y, pen))
        elif current:
            strokes.append(current)
            current = []

    if current:
        strokes.append(current)

    return strokes


def stroke_metadata(strokes: Iterable[list[tuple[float, float, int]]]) -> list[StrokeInfo]:
    """Compute metadata for each stroke and drop very small noise segments."""
    infos: list[StrokeInfo] = []
    for stroke_id, stroke in enumerate(strokes):
        xs = [x for x, _, _ in stroke]
        ys = [y for _, y, _ in stroke]
        if len(stroke) < 3:
            continue
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = x_max - x_min
        height = y_max - y_min
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        infos.append(
            StrokeInfo(
                points=[(x, y) for x, y, _ in stroke],
                stroke_id=stroke_id,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                width=width,
                height=height,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                point_count=len(stroke),
            )
        )
    return infos
