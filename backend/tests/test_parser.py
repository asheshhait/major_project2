from __future__ import annotations

from pathlib import Path

from backend.app.core.parser import parse_pen_file


def test_parse_pen_file_tolerates_headers(tmp_path: Path):
    sample = tmp_path / "sample.csv"
    sample.write_text("x,y,pen\n1,2,1\n3,4,1\n0,0,0\n", encoding="utf-8")

    rows, summary = parse_pen_file(sample)

    assert len(rows) == 3
    assert summary.valid_rows == 3
    assert summary.skipped_rows == 0


def test_parse_pen_file_supports_whitespace_and_timestamp_header(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("time x y pen\n0 1.5 2.5 1\n1 2.5 3.5 0\nbad row\n", encoding="utf-8")

    rows, summary = parse_pen_file(sample)

    assert rows == [[1.5, 2.5, 1], [2.5, 3.5, 0]]
    assert summary.header_detected is True
    assert summary.valid_rows == 2
    assert summary.skipped_rows == 1
