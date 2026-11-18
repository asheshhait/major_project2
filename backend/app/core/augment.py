from __future__ import annotations

import random


def augment_points(points, scale_range=(0.9, 1.1), rotation_deg_range=(-8, 8)):
    """Apply lightweight augmentation suitable for later training stages."""
    if not points:
        return points
    scale = random.uniform(*scale_range)
    rotation = random.uniform(*rotation_deg_range)
    rad = rotation * 3.141592653589793 / 180.0
    cos_a, sin_a = __import__("math").cos(rad), __import__("math").sin(rad)
    augmented = []
    for x, y in points:
        x_new = (x * cos_a - y * sin_a) * scale
        y_new = (x * sin_a + y * cos_a) * scale
        augmented.append((x_new, y_new))
    return augmented
