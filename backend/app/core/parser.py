from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

from ..schemas import Point, ValidationSummary


SEPARATOR_PATTERN = re.compile(r"[\t,\s]+")
HEADER_NAMES = {"x", "y", "pen", "pen_state", "timestamp", "time", "t"}


def _iter_rows(lines: Iterable[str]) -> Iterable[list[str]]:
    for line in lines:
        if not line.strip():
            continue
        yield [part for part in SEPARATOR_PATTERN.split(line.strip()) if part]


def parse_pen_text(text: str) -> tuple[list[Point], ValidationSummary]:
    """Parse x,y,pen rows, accepting an optional header and timestamp column."""
    summary = ValidationSummary()
    points: list[Point] = []
    x_index, y_index, pen_index = 0, 1, 2
    first_data_row = True

    for row_number, row in enumerate(_iter_rows(text.splitlines()), start=1):
        lowered = [value.lower().lstrip("\ufeff") for value in row]
        if first_data_row and any(value in {"x", "y", "pen", "pen_state"} for value in lowered):
            summary.header_detected = True
            try:
                x_index = lowered.index("x")
                y_index = lowered.index("y")
                pen_index = lowered.index("pen") if "pen" in lowered else lowered.index("pen_state")
            except ValueError:
                summary.skipped_rows += 1
                summary.warnings.append(f"Header on row {row_number} must include x, y, and pen columns.")
            first_data_row = False
            continue

        first_data_row = False
        if len(row) <= max(x_index, y_index, pen_index):
            summary.skipped_rows += 1
            summary.warnings.append(f"Skipped malformed row {row_number}: expected x, y, and pen values.")
            continue

        try:
            x = float(row[x_index])
            y = float(row[y_index])
            pen_value = float(row[pen_index])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("x and y must be finite numbers")
            if not (pen_value.is_integer() and int(pen_value) in (0, 1)):
                raise ValueError("pen state must be 0 or 1")
            points.append(Point(x=x, y=y, pen=int(pen_value)))
            summary.valid_rows += 1
        except ValueError as error:
            summary.skipped_rows += 1
            summary.warnings.append(f"Skipped invalid row {row_number}: {error}.")

    return points, summary


def parse_pen_file(path: str | Path, pen_down_value: int = 1) -> tuple[list[list[float]], ValidationSummary]:
    """Read a CSV/TXT pen file and return validated ``[x, y, pen]`` rows."""
    source = Path(path)
    points, summary = parse_pen_text(source.read_text(encoding="utf-8-sig"))
    return [[point.x, point.y, point.pen] for point in points], summary


def parse_csv_like(text: str, pen_down_value: int = 1) -> tuple[list[Point], ValidationSummary]:
    """Parse an in-memory CSV/TXT file using the same rules as ``parse_pen_file``."""
    return parse_pen_text(text)
