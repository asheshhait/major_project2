from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from ..core.segmentation import segment_strokes
from ..db import Session as DbSession, SessionLocal, UploadDocument, Word
from ..ml.yolo_character import detect_word_boxes
from ..schemas import SegmentRequest, WordBox, WordCorrectionsRequest

router = APIRouter(prefix="/api", tags=["words"])


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    with SessionLocal() as db:
        db_session = db.scalar(select(DbSession).where(DbSession.session_id == session_id))
        if db_session is None:
            raise HTTPException(status_code=404, detail="Upload session not found.")
        active_ids = _active_word_ids(db_session)
        words = db.scalars(
            select(Word)
            .where(Word.session_id == db_session.id, Word.id.in_(active_ids))
            .order_by(Word.line_index, Word.word_index)
        ).all() if active_ids else []
        return {
            "session_id": session_id,
            "status": "ready",
            "words": [
                WordBox(
                    id=word.id,
                    line_index=word.line_index,
                    word_index=word.word_index,
                    stroke_ids=[int(value) for value in (word.stroke_ids or "").split(",") if value],
                    bbox=word.bbox_json,
                )
                for word in words
            ],
        }


@router.put("/sessions/{session_id}/words")
async def replace_session_words(session_id: str, request: WordCorrectionsRequest):
    with SessionLocal() as db:
        db_session = db.scalar(select(DbSession).where(DbSession.session_id == session_id))
        if db_session is None:
            raise HTTPException(status_code=404, detail="Upload session not found.")

        try:
            session_metadata = json.loads(db_session.notes or "{}")
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=409, detail="Session word metadata is invalid.") from error
        existing_words = {
            word.id: word
            for word in db.scalars(
                select(Word).where(Word.session_id == db_session.id)
            ).all()
        }
        valid_stroke_ids = set(
            session_metadata.get(
                "stroke_ids",
                [
                    int(stroke_id)
                    for word in existing_words.values()
                    for stroke_id in (word.stroke_ids or "").split(",")
                    if stroke_id
                ],
            )
        )
        used_stroke_ids: set[int] = set()
        submitted_word_ids: set[int] = set()
        saved_words = []

        for correction in request.words:
            stroke_ids = set(correction.stroke_ids)
            if len(stroke_ids) != len(correction.stroke_ids):
                raise HTTPException(status_code=422, detail="A word box cannot list the same stroke more than once.")
            if not stroke_ids.issubset(valid_stroke_ids):
                raise HTTPException(status_code=422, detail="A box references a stroke outside this upload.")
            if used_stroke_ids.intersection(stroke_ids):
                raise HTTPException(status_code=422, detail="A stroke cannot belong to more than one word box.")
            used_stroke_ids.update(stroke_ids)

            if correction.id is not None and correction.id in submitted_word_ids:
                raise HTTPException(status_code=422, detail="A word box cannot be submitted more than once.")
            word = existing_words.get(correction.id) if correction.id is not None else None
            if correction.id is not None and word is None:
                raise HTTPException(status_code=404, detail="A word box does not belong to this upload.")
            if word is None:
                word = Word(session_id=db_session.id)
                db.add(word)

            word.line_index = correction.line_index
            word.word_index = correction.word_index
            word.stroke_ids = ",".join(str(stroke_id) for stroke_id in correction.stroke_ids)
            word.bbox_json = correction.bbox.model_dump()
            db.flush()
            submitted_word_ids.add(word.id)
            saved_words.append(
                WordBox(
                    id=word.id,
                    line_index=word.line_index,
                    word_index=word.word_index,
                    stroke_ids=correction.stroke_ids,
                    bbox=word.bbox_json,
                )
            )

        session_metadata["active_word_ids"] = [word.id for word in saved_words]
        db_session.notes = json.dumps(session_metadata)
        document = db.scalar(
            select(UploadDocument).where(UploadDocument.session_id == db_session.id)
        )
        if document is not None:
            document.corrected = True
        db.commit()
        return {"session_id": session_id, "words": saved_words}


@router.post("/sessions/{session_id}/segment")
async def segment_session(session_id: str, request: SegmentRequest):
    with SessionLocal() as db:
        db_session = db.scalar(select(DbSession).where(DbSession.session_id == session_id))
        if db_session is None:
            raise HTTPException(status_code=404, detail="Upload session not found.")

        try:
            session_metadata = json.loads(db_session.notes or "{}")
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=409, detail="Session word metadata is invalid.") from error

        stroke_ids = [stroke.id for stroke in request.strokes]
        if len(stroke_ids) != len(set(stroke_ids)):
            raise HTTPException(status_code=422, detail="Each uploaded stroke must have a unique ID.")
        expected_stroke_ids = set(session_metadata.get("stroke_ids", []))
        if set(stroke_ids) != expected_stroke_ids:
            raise HTTPException(
                status_code=422,
                detail="Segmentation must include every stroke from this upload exactly once.",
            )
        for stroke in request.strokes:
            if stroke.x_max < stroke.x_min or stroke.y_max < stroke.y_min:
                raise HTTPException(status_code=422, detail="Stroke bounds must have non-negative dimensions.")

        stroke_rows = [stroke.model_dump() for stroke in request.strokes]
        segmented = detect_word_boxes(stroke_rows) or segment_strokes(stroke_rows)
        active_words = []
        for box in segmented:
            word = Word(
                session_id=db_session.id,
                line_index=box.line_index,
                word_index=box.word_index,
                bbox_json=box.bbox,
                stroke_ids=",".join(str(stroke_id) for stroke_id in box.stroke_ids),
            )
            db.add(word)
            db.flush()
            active_words.append(
                WordBox(
                    id=word.id,
                    line_index=box.line_index,
                    word_index=box.word_index,
                    stroke_ids=box.stroke_ids,
                    bbox=box.bbox,
                )
            )

        session_metadata["active_word_ids"] = [word.id for word in active_words]
        db_session.notes = json.dumps(session_metadata)
        document = db.scalar(
            select(UploadDocument).where(UploadDocument.session_id == db_session.id)
        )
        if document is not None:
            document.corrected = False
        db.commit()
        return {"session_id": session_id, "words": active_words}


def _active_word_ids(db_session: DbSession) -> list[int]:
    try:
        metadata = json.loads(db_session.notes or "{}")
    except json.JSONDecodeError:
        metadata = {}
    active_ids = metadata.get("active_word_ids")
    if active_ids is not None:
        return [int(word_id) for word_id in active_ids]
    return [word.id for word in db_session.words]
