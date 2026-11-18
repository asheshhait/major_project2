from __future__ import annotations

import hashlib
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile
from sqlalchemy import select

from ..core.parser import parse_csv_like
from ..core.segmentation import segment_strokes
from ..core.strokes import split_into_strokes, stroke_metadata
from ..db import Session as DbSession, SessionLocal, UploadDocument, Word as DbWord
from ..ml.yolo_character import detect_word_boxes
from ..schemas import Point, Stroke, UploadResponse, WordBox

router = APIRouter(prefix="/api", tags=["upload"])
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_POINTS_PER_STROKE = 3


@router.post("/upload")
async def upload_pen_file(file: UploadFile) -> UploadResponse:
    """Parse an uploaded pen-trace file and return strokes with detected word boxes."""
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="Choose a CSV or TXT file to upload.")
    if not filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a .csv or .txt file.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large. Maximum upload size is 10 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=400, detail="File must be text encoded as UTF-8.") from error

    points, summary = parse_csv_like(text)
    if not points:
        message = "No valid x, y, pen rows were found."
        if summary.warnings:
            message += f" {summary.warnings[0]}"
        raise HTTPException(status_code=422, detail=message)

    raw_strokes = split_into_strokes([(point.x, point.y, point.pen) for point in points])
    metadata = [
        stroke
        for stroke in stroke_metadata(raw_strokes)
        if stroke.point_count >= MIN_POINTS_PER_STROKE
    ]
    if not metadata:
        raise HTTPException(
            status_code=422,
            detail="No usable pen-down strokes were found. Each stroke must contain at least three points.",
        )

    fingerprint = hashlib.sha256(content).hexdigest()
    with SessionLocal() as db:
        saved_document = db.scalar(
            select(UploadDocument).where(UploadDocument.fingerprint == fingerprint)
        )
        if saved_document is not None:
            db_session = db.get(DbSession, saved_document.session_id)
            if db_session is None:
                raise HTTPException(status_code=409, detail="Saved upload session is missing.")
            try:
                session_metadata = json.loads(db_session.notes or "{}")
            except json.JSONDecodeError as error:
                raise HTTPException(status_code=409, detail="Saved upload metadata is invalid.") from error
            active_ids = session_metadata.get("active_word_ids", [])
            saved_words = db.scalars(
                select(DbWord)
                .where(DbWord.session_id == db_session.id, DbWord.id.in_(active_ids))
                .order_by(DbWord.line_index, DbWord.word_index)
            ).all() if active_ids else []
            saved_stats = {
                **saved_document.stats_json,
                "stroke_count": len(json.loads(saved_document.strokes_json)),
                "word_count": len(saved_words),
                "restored": True,
            }
            return UploadResponse(
                session_id=db_session.session_id,
                strokes=[Stroke.model_validate(item) for item in json.loads(saved_document.strokes_json)],
                words=[
                    WordBox(
                        id=word.id,
                        line_index=word.line_index,
                        word_index=word.word_index,
                        stroke_ids=[int(value) for value in (word.stroke_ids or "").split(",") if value],
                        bbox=word.bbox_json,
                    )
                    for word in saved_words
                ],
                stats=saved_stats,
            )

    stroke_payloads: list[Stroke] = []
    segmentation_inputs: list[dict[str, float | int]] = []
    for stroke in metadata:
        bbox = {
            "x": stroke.x_min,
            "y": stroke.y_min,
            "width": stroke.width,
            "height": stroke.height,
        }
        stroke_payloads.append(
            Stroke(
                id=stroke.stroke_id,
                points=[Point(x=x, y=y, pen=1) for x, y in stroke.points],
                bbox=bbox,
                width=stroke.width,
                height=stroke.height,
            )
        )
        segmentation_inputs.append(
            {
                "id": stroke.stroke_id,
                "x_min": stroke.x_min,
                "x_max": stroke.x_max,
                "y_min": stroke.y_min,
                "y_max": stroke.y_max,
                "width": stroke.width,
                "height": stroke.height,
            }
        )

    session_id = str(uuid4())
    yolo_strokes = [
        {
            "id": stroke.id,
            "points": [point.model_dump() for point in stroke.points],
            "x_min": stroke.bbox["x"],
            "x_max": stroke.bbox["x"] + stroke.bbox["width"],
            "y_min": stroke.bbox["y"],
            "y_max": stroke.bbox["y"] + stroke.bbox["height"],
            "width": stroke.width,
            "height": stroke.height,
        }
        for stroke in stroke_payloads
    ]
    segmented_words = detect_word_boxes(yolo_strokes) or segment_strokes(segmentation_inputs)
    stats = {
        "valid_rows": summary.valid_rows,
        "skipped_rows": summary.skipped_rows,
        "header_detected": summary.header_detected,
        "stroke_count": len(stroke_payloads),
        "word_count": len(segmented_words),
        "warnings": summary.warnings[:20],
        "restored": False,
    }
    with SessionLocal() as db:
        db_session = DbSession(session_id=session_id)
        db.add(db_session)
        db.flush()
        db_document = UploadDocument(
            fingerprint=fingerprint,
            session_id=db_session.id,
            strokes_json=json.dumps(
                [stroke.model_dump() for stroke in stroke_payloads],
                ensure_ascii=False,
            ),
            stats_json=stats,
        )
        db.add(db_document)
        words = []
        for word in segmented_words:
            db_word = DbWord(
                session_id=db_session.id,
                line_index=word.line_index,
                word_index=word.word_index,
                bbox_json=word.bbox,
                stroke_ids=",".join(str(stroke_id) for stroke_id in word.stroke_ids),
            )
            db.add(db_word)
            db.flush()
            words.append(
                WordBox(
                    id=db_word.id,
                    line_index=word.line_index,
                    word_index=word.word_index,
                    stroke_ids=word.stroke_ids,
                    bbox=word.bbox,
                )
            )
        db_session.notes = json.dumps(
            {
                "active_word_ids": [word.id for word in words],
                "stroke_ids": [stroke.id for stroke in stroke_payloads],
            }
        )
        db.commit()

    logger.info(
        "Uploaded %s: %d rows, %d strokes, %d word boxes",
        filename,
        summary.valid_rows,
        len(stroke_payloads),
        len(words),
    )
    return UploadResponse(
        session_id=session_id,
        strokes=stroke_payloads,
        words=words,
        stats=stats,
    )
