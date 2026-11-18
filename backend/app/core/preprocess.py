from __future__ import annotations

from typing import Iterable


def normalize_word(points: Iterable[tuple[float, float]], target_height: float = 128.0):
    """Normalize a sequence of points to a canonical coordinate system."""
    pts = list(points)
    if not pts:
        return []
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    scale = target_height / height
    translated = [(x - min_x, y - min_y) for x, y in pts]
    return [(x * scale, y * scale) for x, y in translated]
