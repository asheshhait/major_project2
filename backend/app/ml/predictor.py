from __future__ import annotations

import json

from sqlalchemy import select

from ..db import Sample, SessionLocal
from .dtw_matcher import DTWMatcher


matcher = DTWMatcher()
trained_sample_count = 0
trained_word_count = 0


def train_template_matcher(samples) -> int:
    global trained_sample_count, trained_word_count
    sample_list = list(samples)
    trained_sample_count = matcher.fit(sample_list)
    trained_word_count = len({label for label, _ in sample_list})
    return trained_sample_count


def reload_saved_templates() -> int:
    """Rebuild the in-memory matcher from all valid labeled samples in SQLite."""
    templates: list[tuple[str, list[tuple[float, float]]]] = []
    with SessionLocal() as db:
        samples = db.scalars(
            select(Sample)
            .where(Sample.is_bad.is_(False))
            .order_by(Sample.id)
        ).all()
        for sample in samples:
            if not sample.raw_points:
                continue
            point_rows = json.loads(sample.raw_points)
            templates.append(
                (
                    sample.text,
                    [(float(point["x"]), float(point["y"])) for point in point_rows],
                )
            )
    return train_template_matcher(templates)


def predict_word(points):
    return matcher.predict(points)
