from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class ValidationSummary(BaseModel):
    valid_rows: int = 0
    skipped_rows: int = 0
    header_detected: bool = False
    warnings: list[str] = Field(default_factory=list)


class Point(BaseModel):
    x: float
    y: float
    pen: int = 1


class Stroke(BaseModel):
    id: int
    points: list[Point]
    bbox: Optional[dict[str, float]] = None
    width: Optional[float] = None
    height: Optional[float] = None


class WordBox(BaseModel):
    id: int
    line_index: int
    word_index: int
    stroke_ids: list[int]
    bbox: BoundingBox | dict[str, float]


class UploadResponse(BaseModel):
    session_id: str
    strokes: list[Stroke]
    words: list[WordBox]
    stats: dict[str, Any]


class WordCorrection(BaseModel):
    id: int | None = None
    line_index: int = Field(ge=0)
    word_index: int = Field(ge=0)
    stroke_ids: list[int] = Field(min_length=1)
    bbox: BoundingBox


class WordCorrectionsRequest(BaseModel):
    words: list[WordCorrection]


class StrokeBounds(BaseModel):
    id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: float
    height: float
    points: list[Point] = Field(default_factory=list)


class SegmentRequest(BaseModel):
    strokes: list[StrokeBounds] = Field(min_length=1)


class SampleCreate(BaseModel):
    session_id: str
    word_id: int
    text: str = Field(min_length=1, max_length=255)
    chars: list[str] = Field(min_length=1)
    points: list[Point] = Field(min_length=1)


class BadSampleCreate(BaseModel):
    session_id: str
    word_id: int
    points: list[Point] = Field(min_length=1)
    reason: str = Field(default="Marked as a bad sample", max_length=500)


class PredictionRequest(BaseModel):
    points: list[Point] = Field(min_length=2)


class TrainStatus(BaseModel):
    status: str
    epoch: int = 0
    loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = ""
