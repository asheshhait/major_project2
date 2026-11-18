from __future__ import annotations

import argparse
import random
from pathlib import Path


SYNTHETIC_WORD_PITCH = 180


def synthetic_word(word_id: int, line_index: int, seed: int) -> list[tuple[float, float, int]]:
    """Generate a simple fake Bangla-like stroke sequence for a word."""
    rng = random.Random(seed)
    points: list[tuple[float, float, int]] = []
    start_x = 20 + word_id * SYNTHETIC_WORD_PITCH
    baseline = 60 + line_index * 140 + rng.randint(0, 10)
    for i in range(12):
        x = start_x + i * 8 + rng.uniform(-2, 2)
        y = baseline + rng.uniform(-6, 6)
        points.append((x, y, 1))
    for i in range(4):
        x = start_x + 10 + i * 7 + rng.uniform(-3, 3)
        y = baseline - 18 + i * 4 + rng.uniform(-2, 2)
        points.append((x, y, 1))
    points.append((start_x + 80, baseline, 0))
    return points


def make_pen_file(
    output_path: Path,
    words_per_line: int = 5,
    line_count: int = 2,
    total_words: int | None = None,
):
    """Create traces with clear inter-word gaps and, optionally, an exact word count."""
    lines: list[str] = ["x,y,pen"]
    if total_words is None:
        words_by_line = [words_per_line] * line_count
    else:
        base_count, remainder = divmod(total_words, line_count)
        words_by_line = [
            base_count + (1 if line_index < remainder else 0)
            for line_index in range(line_count)
        ]

    for line_index in range(line_count):
        for word_index in range(words_by_line[line_index]):
            for point_x, point_y, pen in synthetic_word(word_index, line_index, line_index * 101 + word_index):
                lines.append(f"{point_x:.2f},{point_y:.2f},{pen}")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic pen-trace data")
    parser.add_argument("--output", type=str, default="data/synthetic_pen_data.txt")
    parser.add_argument("--words", type=int, default=8)
    parser.add_argument("--lines", type=int, default=2)
    args = parser.parse_args()

    total_words = max(1, args.words)
    line_count = max(1, args.lines)
    make_pen_file(Path(args.output), line_count=line_count, total_words=total_words)
    print(f"Synthetic data written to {args.output}")
